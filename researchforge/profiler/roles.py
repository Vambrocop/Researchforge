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


def detect_roles(columns) -> dict:
    """Return {likely_outcome, likely_treatment, likely_time, reason} from a list
    of ColumnInfo (in dataframe order). Any value may be None."""
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
    # 2. domain outcome-ish name → MEDIUM (often the DV, but could be a predictor; hint only)
    elif med_named:
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
