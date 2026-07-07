"""Semantic role hints — a step toward smarter auto-selection.

The engine's run-time defaults are deliberately simple conventions ("first
continuous column = outcome"). On real data that can pick the wrong column —
e.g. a disease-progression *target* that happens to be integer-valued profiles
as a `count` and gets skipped as an outcome candidate, so regression silently
models `age` instead.

This module produces NON-BINDING *hints* (likely outcome / treatment / time) from
column names + position + kind. They do NOT change any run-time default — they are
surfaced to the user (in `recommend` and as a one-line nudge in a modeling run's
summary) so they can set `config` deliberately. High precision over recall: a hint
is offered only when there's a real signal, and always labelled with its reason.
"""

from __future__ import annotations

import re

# Name signals (word-boundary, case-insensitive). Outcome words are the dependent
# variable a study predicts/explains; treatment a binary intervention; time a period.
#
# Split by CONFIDENCE so the hint can safely BIND execution (see resolve_outcome):
#  - HIGH = unambiguous dependent-variable names (a column called 'target'/'outcome'/'y' is
#    essentially never a predictor) → safe to override the "first continuous column" default.
#  - MEDIUM = domain words that are OFTEN the DV but just as often a feature (a 'price' or
#    'score' column may be either) → surfaced as a hint only, must NOT bind (else we'd model
#    the wrong column on data where the domain word is actually a predictor).
_OUTCOME_HIGH_RE = re.compile(
    r"(?:^|_|\b)(outcome|target|response|dependent|depvar|label|y)(?:$|_|\b)",
    re.I,
)
_OUTCOME_MED_RE = re.compile(
    r"(?:^|_|\b)(yield|score|rating|progression|growth|sales|revenue|profit|"
    r"price|result|grade|gpa|severity|risk)(?:$|_|\b)",
    re.I,
)
_TREATMENT_RE = re.compile(
    r"(?:^|_|\b)(treat|treatment|treated|intervention|arm|group|condition|"
    r"exposed|exposure|policy|program|assigned|dose)(?:$|_|\b)",
    re.I,
)
_TIME_RE = re.compile(
    r"(?:^|_|\b)(year|yr|date|time|month|quarter|period|day|week|wave)(?:$|_|\b)",
    re.I,
)

# Binary outcome / event names — a binary column with one of these is a classification
# TARGET (not a grouping/arm variable). High precision: only clear event words, so a
# demographic binary (gender, region) is NOT mistaken for an outcome. Checked before the
# continuous name match so a true binary target beats a continuous predictor that merely
# name-matches (e.g. a 'score' feature alongside an 'approved' target).
_BIN_OUTCOME_RE = re.compile(
    r"(?:^|_|\b)(approv\w*|default\w*|churn\w*|success\w*|succeed\w*|fail\w*|died|death|dead|"
    r"surviv\w*|deceased|alive|cured|recover\w*|infected|relapse|recur\w*|readmit\w*|"
    r"positive|convert\w*|conversion|fraud\w*|won|win|lost|pass|passed|retain\w*|retention|"
    r"click\w*|purchas\w*|bought|subscrib\w*|respond\w*|response|accept\w*|reject\w*|"
    r"enroll\w*|qualif\w*|eligible|attrition|complete\w*|label|target|outcome|y)(?:$|_|\b)",
    re.I,
)

_NUMERIC_KINDS = {"continuous", "count"}


def is_treatment_named(name: str) -> bool:
    """True when a column NAME carries a treatment/arm signal (treat/arm/exposed/dose…).

    Public so the executor's outcome resolver can skip a treatment-named column when
    falling back to "first candidate" — a treatment indicator is almost never the
    dependent variable. Name-signal ONLY: deliberately not based on detect_roles'
    positional likely_treatment fallback (which tags the first non-outcome binary even
    without any name evidence, and would flip conventions arbitrarily)."""
    return bool(_TREATMENT_RE.search(str(name)))


def _structure_evidence(med_named, numeric, df):
    """MEDIUM→HIGH promotion evidence (Wave H3). A domain outcome word (price/sales/
    score…) is ambiguous by NAME alone — but data STRUCTURE can corroborate it:

      • explained-R² (REQUIRED for promotion) — regressing each numeric column on the others, the candidate is
        SIGNIFICANTLY (F-test, p<0.05 — n-aware, so a spurious R²≈0.10 at n≈25 doesn't
        bind) and NON-TRIVIALLY (R²≥0.10) explained AND in the top tier (within 0.05 of
        max). Tier, not strict max: on standardized data a single-predictor pair
        (sales=3·adspend+ε) has essentially SYMMETRIC R² — the asymmetric NAME/POSITION
        signals break the tie; R² only certifies "structurally outcome-like". Needs ≥3
        numeric columns (with 2, R² is exactly symmetric and discriminates nothing).
      • position (LAST numeric column) — NOT a promotion signal on its own (at small n it
        flips on noise, cold-review S1); only annotates an R²-promoted column or breaks a
        rare tie between two equally-R²-evidenced candidates.

    Only CONTINUOUS candidates promote: a promoted HIGH outcome binds in the continuous
    _regression family, so promoting a count column would make the run's "已自动选取"
    nudge name a column the continuous regression then silently drops (cold-review M2).

    Promotion (conservative — a wrong promotion silently models the wrong column
    engine-wide): a SINGLE candidate promotes on either valid signal; with MULTIPLE
    candidates only a uniquely-R²-evidenced one promotes (position alone can't break a
    name tie). Any failure (missing df / small n / degenerate) → no promotion.
    Returns (column_name, evidence_desc) or None."""
    import numpy as np
    import pandas as pd

    cand = [c for c in med_named if getattr(c, "kind", None) == "continuous"]
    if df is None or len(df) < 20 or len(numeric) < 3 or not cand:
        return None
    try:
        from scipy.stats import f as _f

        cols = [c.name for c in numeric]
        X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) > 1000:  # strided (not head) so a time-sorted CSV isn't one regime
            X = X.iloc[np.linspace(0, len(X) - 1, 1000).astype(int)]
        n = len(X)
        if n < 20:
            return None
        std = X.std(ddof=0)
        keep = [c for c in cols if std[c] > 0]
        if len(keep) < 3:  # <3 non-constant cols → R² symmetric, discriminates nothing
            return None
        Z = (X[keep] - X[keep].mean()) / X[keep].std(ddof=0)
        k = len(keep) - 1  # predictors when regressing one column on the others
        r2: dict[str, float] = {}
        sig: dict[str, bool] = {}
        for c in keep:
            others = [o for o in keep if o != c]
            y = Z[c].to_numpy()
            A = np.column_stack([Z[o].to_numpy() for o in others] + [np.ones(n)])
            resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            sst = float((y**2).sum())
            rr = 1.0 - float((resid**2).sum()) / sst if sst > 0 else 0.0
            r2[c] = rr
            if 0.0 < rr < 1.0 and n - k - 1 > 0:
                fstat = (rr / k) / ((1 - rr) / (n - k - 1))
                sig[c] = float(_f.sf(fstat, k, n - k - 1)) < 0.05
            else:
                sig[c] = rr >= 1.0

        max_r2 = max(r2.values(), default=0.0)

        def _struct_ok(name):  # significantly + non-trivially + top-tier explained
            return r2.get(name, 0.0) >= 0.10 and sig.get(name, False) and r2[name] >= max_r2 - 0.05

        # Promotion REQUIRES R² structure (significant + top-tier). Position alone never
        # promotes — at small n / weak effects it's the only signal and flips on noise
        # (cold-review S1), and a name+position bet with no data confirmation doesn't
        # justify BINDING high confidence. Position only annotates / breaks an R² tie.
        last_name = numeric[-1].name
        r2_evidenced = [c.name for c in cand if _struct_ok(c.name)]
        if len(r2_evidenced) == 1:
            nm = r2_evidenced[0]
            pos = "、末位数值列" if nm == last_name else ""
            return nm, f"最高被解释R²={r2[nm]:.2f}{pos}"
        if len(r2_evidenced) >= 2:  # rare tie among medium-named cols → position breaks it
            last_evid = [nm for nm in r2_evidenced if nm == last_name]
            if len(last_evid) == 1:
                nm = last_evid[0]
                return nm, f"最高被解释R²={r2[nm]:.2f}、末位数值列"
        return None
    except Exception:
        return None


def detect_roles(columns, df=None) -> dict:
    """Return {likely_outcome, likely_treatment, likely_time, reason} from a list
    of ColumnInfo (in dataframe order). Any value may be None. When ``df`` is given,
    a MEDIUM name hint corroborated by data structure is promoted to HIGH (binding —
    see _structure_evidence); without df, behavior is unchanged (name/position only)."""
    out: dict[str, object] = {
        "likely_outcome": None, "likely_outcome_confidence": "",
        "likely_treatment": None, "likely_time": None, "reason": "",
    }
    names = [c.name for c in columns]
    numeric = [c for c in columns if c.kind in _NUMERIC_KINDS]
    binary = [c for c in columns if c.kind == "binary"]

    # --- likely_outcome (with a confidence tier; see the regex split above) --------------
    # 0. a binary column named like an outcome/event = a classification TARGET. Highest
    #    precision, checked first so a true binary target beats a continuous predictor
    #    that merely name-matches (e.g. an 'approved' target alongside a 'score' feature).
    b_named = [c for c in binary if _BIN_OUTCOME_RE.search(str(c.name))]
    high_named = [c for c in numeric if _OUTCOME_HIGH_RE.search(str(c.name))]
    med_named = [c for c in numeric if _OUTCOME_MED_RE.search(str(c.name))]
    if b_named:
        out["likely_outcome"] = b_named[0].name
        out["likely_outcome_confidence"] = "high"
        out["reason"] = f"binary column '{b_named[0].name}' matches an outcome/event pattern (classification target)"
    # 1. unambiguous dependent-variable name among numeric columns → HIGH confidence
    elif high_named:
        out["likely_outcome"] = high_named[0].name
        out["likely_outcome_confidence"] = "high"
        out["reason"] = f"name '{high_named[0].name}' matches an unambiguous outcome pattern"
    # 2. domain outcome-ish name → MEDIUM (often the DV, but could be a predictor).
    #    With data present, structural corroboration promotes it to HIGH (binding).
    elif med_named:
        promoted = _structure_evidence(med_named, numeric, df)
        if promoted is not None:
            name, evidence = promoted
            out["likely_outcome"] = name
            out["likely_outcome_confidence"] = "high"
            out["reason"] = (
                f"name '{name}' matches a domain outcome pattern + structural evidence"
                f"（{evidence}）— promoted to high confidence"
            )
        else:
            out["likely_outcome"] = med_named[0].name
            out["likely_outcome_confidence"] = "medium"
            out["reason"] = (
                f"name '{med_named[0].name}' matches a domain outcome pattern "
                f"(ambiguous — could be a predictor; verify before modeling)"
            )
    # 3. position fallback: the LAST numeric column, when there are >=2 numeric
    #    columns before it (the common ML convention that the target is last) and
    #    it isn't a time/id-looking column → LOW confidence
    elif len(numeric) >= 3:
        last = numeric[-1]
        if last.kind in _NUMERIC_KINDS and not _TIME_RE.fullmatch(str(last.name).strip().lower() or " "):
            out["likely_outcome"] = last.name
            out["likely_outcome_confidence"] = "low"
            out["reason"] = f"'{last.name}' is the last numeric column (common target position)"

    # --- likely_treatment ---------------------------------------------------
    # a binary already taken as the outcome can't also be the treatment/arm
    treat_cands = [c for c in binary if c.name != out["likely_outcome"]]
    t_named = [c for c in treat_cands if _TREATMENT_RE.search(str(c.name))]
    if t_named:
        out["likely_treatment"] = t_named[0].name
    elif treat_cands:
        out["likely_treatment"] = treat_cands[0].name

    # --- likely_time --------------------------------------------------------
    for c in columns:
        if c.kind == "datetime" or _TIME_RE.search(str(c.name)):
            out["likely_time"] = c.name
            break

    return out
