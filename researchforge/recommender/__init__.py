"""Recommender layer: match data to analyses, review rigor, gauge novelty."""

from researchforge.recommender.diagnostics import (
    Diagnostic,
    DiagnosticPlan,
    build_plan,
    diagnose_data,
)
from researchforge.recommender.goals import GOALS, entry_matches_goal, resolve_goal
from researchforge.recommender.match import check_preconditions
from researchforge.recommender.novelty import NoveltyHint, novelty_hint
from researchforge.recommender.recommend import (
    Recommendation,
    apply_diagnostic_ranking,
    recommend,
    select_top,
)
from researchforge.recommender.rigor import RigorVerdict, assess_rigor
from researchforge.recommender.scoring import MethodologyScore, score_method

__all__ = [
    "build_plan",
    "diagnose_data",
    "Diagnostic",
    "DiagnosticPlan",
    "check_preconditions",
    "assess_rigor",
    "RigorVerdict",
    "recommend",
    "select_top",
    "apply_diagnostic_ranking",
    "Recommendation",
    "novelty_hint",
    "NoveltyHint",
    "MethodologyScore",
    "score_method",
    "GOALS",
    "resolve_goal",
    "entry_matches_goal",
]
