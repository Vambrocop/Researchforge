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
_OUTCOME_RE = re.compile(
    r"(?:^|_|\b)(outcome|target|response|dependent|depvar|label|yield|"
    r"score|rating|progression|growth|sales|revenue|profit|price|"
    r"result|grade|gpa|severity|risk|y)(?:$|_|\b)",
    re.I,
)
_TREATMENT_RE = re.compile(
    r"(?:^|_|\b)(treat|treatment|treated|intervention|arm|group|condition|"
    r"exposed|exposure|policy|program|assigned|dose|d|t)(?:$|_|\b)",
    re.I,
)
_TIME_RE = re.compile(
    r"(?:^|_|\b)(year|yr|date|time|month|quarter|period|day|week|wave|t)(?:$|_|\b)",
    re.I,
)

_NUMERIC_KINDS = {"continuous", "count"}


def detect_roles(columns) -> dict:
    """Return {likely_outcome, likely_treatment, likely_time, reason} from a list
    of ColumnInfo (in dataframe order). Any value may be None."""
    out: dict[str, object] = {
        "likely_outcome": None, "likely_treatment": None,
        "likely_time": None, "reason": "",
    }
    names = [c.name for c in columns]
    numeric = [c for c in columns if c.kind in _NUMERIC_KINDS]

    # --- likely_outcome -----------------------------------------------------
    # 1. name signal among numeric columns (highest precision)
    named = [c for c in numeric if _OUTCOME_RE.search(str(c.name))]
    if named:
        out["likely_outcome"] = named[0].name
        out["reason"] = f"name '{named[0].name}' matches an outcome pattern"
    # 2. position fallback: the LAST numeric column, when there are >=2 numeric
    #    columns before it (the common ML convention that the target is last) and
    #    it isn't a time/id-looking column.
    elif len(numeric) >= 3:
        last = numeric[-1]
        if last.kind in _NUMERIC_KINDS and not _TIME_RE.fullmatch(str(last.name).strip().lower() or " "):
            out["likely_outcome"] = last.name
            out["reason"] = f"'{last.name}' is the last numeric column (common target position)"

    # --- likely_treatment ---------------------------------------------------
    binary = [c for c in columns if c.kind == "binary"]
    t_named = [c for c in binary if _TREATMENT_RE.search(str(c.name))]
    if t_named:
        out["likely_treatment"] = t_named[0].name
    elif binary:
        out["likely_treatment"] = binary[0].name

    # --- likely_time --------------------------------------------------------
    for c in columns:
        if c.kind == "datetime" or _TIME_RE.search(str(c.name)):
            out["likely_time"] = c.name
            break

    return out
