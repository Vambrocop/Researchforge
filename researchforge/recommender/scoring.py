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

from pydantic import BaseModel

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.recommender.rigor import RigorVerdict

# base scores by family: (popularity, publishability, aesthetics, difficulty, novelty)
_FAMILY: dict[str, tuple[int, int, int, int, int]] = {
    "statistics": (85, 55, 45, 35, 25),
    "causal": (70, 88, 68, 70, 70),
    "sem": (65, 82, 80, 75, 55),
    "meta": (60, 80, 76, 50, 50),
    "ml": (80, 65, 70, 60, 65),
    "timeseries": (70, 62, 66, 62, 42),
    "spatial": (55, 74, 88, 66, 62),
    "ecology": (60, 66, 80, 55, 46),
    "microbiology": (55, 72, 76, 56, 62),
    "mcda": (60, 46, 60, 40, 46),
    "efficiency": (52, 72, 60, 72, 56),
    "panel": (66, 82, 55, 78, 56),
    "qualitative": (42, 70, 62, 70, 76),
    "soil": (46, 42, 70, 32, 36),
}
_DEFAULT = (50, 56, 56, 55, 50)

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

    fit = max(0, min(100, int(rigor.score)))
    # overall display blend — fit and publishability weighted most; difficulty is a
    # cost and deliberately excluded (shown separately).
    overall = round(0.35 * fit + 0.25 * pub + 0.15 * pop + 0.15 * nov + 0.10 * aes)
    note = (
        f"契合 {fit}（本数据）/ 流行 {pop} / 可发表 {pub} / 美观 {aes} / 新颖 {nov} / "
        f"难度 {diff}（越高越难）。流行·新颖为离线编辑先验，趋势引擎接入后将动态更新。"
    )
    return MethodologyScore(
        popularity=pop, publishability=pub, aesthetics=aes, difficulty=diff,
        fit=fit, novelty=nov, overall=int(overall), note=note,
    )
