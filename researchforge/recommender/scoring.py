"""Methodology score card — a multi-dimensional read on a recommended method.

Offline + deterministic: dimensions are derived from a rule/metadata rubric
(by family, with per-id overrides) plus the data-specific rigor verdict. No
network. The self-evolution trend engine (later phase) can refine the popularity
and novelty dimensions from live CRAN/PyPI/GitHub/literature signals; until then
these are static editorial priors, surfaced honestly as such.

Dimensions (0-100, higher = more of that attribute):
  popularity     流行   how widely used the method is
  publishability 可发表 how much it supports high-impact publication
  aesthetics     美观   strength of its signature figures/visual output
  difficulty     难度   interpretation / assumption burden (a COST, not a good)
  fit            契合   how well it fits THIS dataset (= the rigor score)
  novelty        新颖   how fresh / trendy the method currently is
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.recommender.affinity import data_signals, get_affinity, match_score
from researchforge.recommender.rigor import RigorVerdict

# base scores by family: (popularity, publishability, aesthetics, difficulty, novelty).
# Editorial priors (subjective, disclosed as such) — keys MUST match catalog family
# strings exactly (test_scoring guards full coverage, so a family never silently falls
# back to _DEFAULT). Populated for all 45 catalog families.
_FAMILY: dict[str, tuple[int, int, int, int, int]] = {
    "statistics": (85, 55, 45, 35, 25),
    "causal": (70, 88, 68, 70, 70),
    "sem": (65, 82, 80, 75, 55),
    "meta": (60, 80, 76, 50, 50),
    "ml": (80, 65, 70, 60, 65),
    "time-series": (70, 62, 66, 62, 42),
    "spatial": (55, 74, 88, 66, 62),
    "ecology": (60, 66, 80, 55, 46),
    "mcda": (60, 46, 60, 40, 46),
    "efficiency": (52, 72, 60, 72, 56),
    "econometrics": (66, 82, 55, 78, 56),
    "configurational": (42, 70, 62, 70, 76),
    "soil": (46, 42, 70, 32, 36),
    # — filled in so the scorecard is meaningful for every family (was _DEFAULT before) —
    "bayesian": (62, 80, 72, 78, 66),
    "survival": (72, 80, 80, 62, 48),
    "regression": (60, 45, 42, 32, 22),
    "conditional_process": (66, 74, 66, 68, 58),
    "irt": (55, 76, 66, 74, 52),
    "psychometrics": (58, 66, 60, 55, 40),
    "latent_class": (52, 74, 70, 76, 62),
    "mixture": (55, 66, 72, 68, 56),
    "nonparametric": (72, 48, 44, 32, 28),
    "categorical": (70, 50, 48, 36, 26),
    "categorical_tests": (72, 48, 45, 32, 24),
    "distribution": (58, 46, 56, 42, 32),
    "distribution_extra": (52, 48, 58, 46, 42),
    "effect_sizes": (66, 62, 50, 36, 34),
    "epidemiology": (64, 78, 68, 55, 50),
    "finance": (60, 60, 66, 62, 52),
    "hydrology": (44, 60, 66, 58, 46),
    "marketing": (58, 50, 62, 45, 50),
    "actuarial": (46, 58, 55, 66, 44),
    "operations_research": (56, 52, 56, 55, 40),
    "game_theory": (48, 62, 58, 70, 58),
    "reliability": (50, 62, 60, 62, 46),
    "spc": (58, 55, 66, 48, 36),
    "survey_methods": (60, 66, 50, 60, 44),
    "nlp": (74, 68, 66, 62, 68),
    "choice": (58, 74, 60, 72, 52),
    "missing_data": (58, 66, 52, 64, 54),
    "resource": (50, 60, 66, 52, 56),
    "techno_economic": (52, 58, 62, 50, 50),
    "experimental_design": (60, 66, 58, 48, 40),
    "experimental_stats": (64, 62, 55, 50, 38),
    "agreement": (58, 60, 55, 42, 34),
}
_DEFAULT = (50, 56, 56, 55, 50)


@lru_cache(maxsize=1)
def _trend_snapshot() -> Optional[dict]:
    """Process-cached momentum snapshot (written by `cli discover --live`). Read once
    per process — a refreshed snapshot is picked up on the next run. Hot-path safe:
    pure file read, never network. Returns None when no fresh snapshot exists."""
    try:
        from researchforge.catalog.trends import load_snapshot

        return load_snapshot()
    except Exception:
        return None


def _live_momentum(entry_id: str, family: str) -> Optional[int]:
    """Real PyPI/GitHub/CRAN momentum for this method from the cached snapshot:
    per-id if known, else the per-family mean. None when no live signal exists."""
    snap = _trend_snapshot()
    if not snap:
        return None
    by_id = snap.get("by_id", {})
    if entry_id in by_id:
        return int(by_id[entry_id])
    fam = snap.get("by_family", {})
    if family in fam:
        return int(fam[family])
    return None

# per-id overrides (only the dimensions worth nudging from the family base)
_ID: dict[str, dict[str, int]] = {
    "synthetic_control": {"novelty": 88, "publishability": 88, "aesthetics": 80},
    "did": {"publishability": 90, "aesthetics": 78, "popularity": 75},
    "gam": {"novelty": 70, "aesthetics": 82, "publishability": 76},
    "glmm": {"publishability": 80, "difficulty": 78},
    "meta_analysis": {"aesthetics": 80, "publishability": 82},
    "dynamic_panel_gmm": {"difficulty": 90, "publishability": 84},
    "fsqca": {"novelty": 80}, "qca_necessity": {"novelty": 80}, "csqca": {"novelty": 76},
    "nca": {"novelty": 82},
    "spatial_regression": {"aesthetics": 88, "publishability": 78},
    "survival_analysis": {"aesthetics": 82, "publishability": 78},
    "sfa": {"difficulty": 82, "publishability": 76},
    "descriptive_stats": {"publishability": 30, "novelty": 12, "aesthetics": 35},
    "correlation": {"publishability": 35, "novelty": 15},
}


# specific (non-generic) precondition flag -> (data signal that satisfies it, weight). A
# method whose SPECIFIC precondition matches this data's structure is tailored to it, so
# it earns a fit bonus over generic methods in the same family (this is what lifts logistic
# on binary data, network methods on an edge list, spatial on geo, … out of the
# ols/random_forest/descriptive soup). requires_edgelist is weighted highest because there
# is no "network" family to grant a structure bonus, so this stands in for it.
_SPECIFIC_PRECOND = {
    "requires_edgelist": ("has_edgelist", 14.0),
    "is_panel": ("is_panel", 16.0),
    "is_timeseries": ("is_timeseries", 16.0),
    "requires_count_outcome": ("has_count", 14.0),
    "requires_geo": ("has_geo", 12.0),
    "requires_binary_outcome": ("has_binary", 12.0),
    "requires_treatment": ("has_treatment", 12.0),
}

# Structure-defining preconditions: when the data has this structure, a method that
# requires it is clearly appropriate REGARDLESS of its (possibly heterogeneous) family —
# e.g. network methods live in family "ml" alongside random forests but don't need ml's
# outcome/predictors, so they must not eat the ml family penalty on an edge list. When
# one of these is met we floor the family base at a structure-match level.
_STRUCTURE_PRECOND = {
    "requires_edgelist": "has_edgelist",
    "requires_geo": "has_geo",
    "is_panel": "is_panel",
    "is_timeseries": "is_timeseries",
}
_STRUCTURE_FLOOR = 72.0


def _precond_bonus(signals: dict, pre) -> float:
    """Per-method tailoring bonus (0–30): reward a method whose specific precondition
    matches this data's special structure."""
    pm = pre.model_dump()
    bonus = sum(w for flag, (sig, w) in _SPECIFIC_PRECOND.items()
                if pm.get(flag) and signals.get(sig))
    # requires_group fires on a GENUINE group (a binary/categorical that isn't the
    # role-detected outcome). With binary-outcome role detection, this no longer mis-fires
    # for binary-OUTCOME regression data (where logistic, not a 2-arm test, is right), so
    # it gets its full weight again.
    if pm.get("requires_group") and signals.get("has_group"):
        bonus += 10.0
    if pm.get("min_count_cols") and signals["has_count"]:
        bonus += 8.0
    if pm.get("min_categorical_cols") and signals["has_categorical"]:
        bonus += 6.0
    return min(bonus, 30.0)


def _affinity_fit(fp: DataFingerprint, entry: AnalysisEntry, rigor: RigorVerdict) -> int:
    """Real data-fit (0–100): how well this method suits THIS dataset = family
    structure/outcome affinity (affinity.match_score) + per-method precondition
    tailoring. Replaces the old fit = rigor.score (which was just bias-count). An
    infeasible (red) method can't be a good fit no matter its affinity, so it stays
    capped at its (low) rigor score; feasible methods are ranked by affinity."""
    signals = data_signals(fp)
    base = match_score(signals, get_affinity(entry.family))
    pm = entry.preconditions.model_dump()
    if any(pm.get(flag) and signals.get(sig) for flag, sig in _STRUCTURE_PRECOND.items()):
        base = max(base, _STRUCTURE_FLOOR)
    raw = min(100.0, base + _precond_bonus(signals, entry.preconditions))
    if rigor.light == "red":
        return max(0, min(int(round(rigor.score)), int(round(raw))))
    return int(round(raw))


class MethodologyScore(BaseModel):
    popularity: int
    publishability: int
    aesthetics: int
    difficulty: int
    fit: int
    novelty: int
    overall: int  # display blend (excludes difficulty, which is a cost)
    note: str = ""

    def as_dict(self) -> dict[str, int]:
        return {
            "popularity": self.popularity,
            "publishability": self.publishability,
            "aesthetics": self.aesthetics,
            "difficulty": self.difficulty,
            "fit": self.fit,
            "novelty": self.novelty,
            "overall": self.overall,
        }


def score_method(
    fp: DataFingerprint, entry: AnalysisEntry, rigor: RigorVerdict
) -> MethodologyScore:
    """Compute the methodology score card for a data/method pairing. `fit` comes
    from the data-specific rigor score; the rest from the offline rubric."""
    pop, pub, aes, diff, nov = _FAMILY.get(entry.family, _DEFAULT)
    for k, v in _ID.get(entry.id, {}).items():
        if k == "popularity":
            pop = v
        elif k == "publishability":
            pub = v
        elif k == "aesthetics":
            aes = v
        elif k == "difficulty":
            diff = v
        elif k == "novelty":
            nov = v

    # Live trend feed (phase 2): blend real PyPI/GitHub/CRAN momentum from the cached
    # snapshot into popularity when available. Hot-path safe (file read, no network).
    mom = _live_momentum(entry.id, entry.family)
    if mom is not None:
        pop = int(round(0.5 * pop + 0.5 * mom))
        trend_note = "流行含实时趋势（PyPI/GitHub/CRAN 动量，快照缓存）"
    else:
        trend_note = "流行·新颖为离线编辑先验，趋势引擎接入后将动态更新"

    fit = _affinity_fit(fp, entry, rigor)
    # overall display blend — fit and publishability weighted most; difficulty is a
    # cost and deliberately excluded (shown separately).
    overall = round(0.35 * fit + 0.25 * pub + 0.15 * pop + 0.15 * nov + 0.10 * aes)
    note = (
        f"契合 {fit}（本数据）/ 流行 {pop} / 可发表 {pub} / 美观 {aes} / 新颖 {nov} / "
        f"难度 {diff}（越高越难）。{trend_note}。"
    )
    return MethodologyScore(
        popularity=pop, publishability=pub, aesthetics=aes, difficulty=diff,
        fit=fit, novelty=nov, overall=int(overall), note=note,
    )
