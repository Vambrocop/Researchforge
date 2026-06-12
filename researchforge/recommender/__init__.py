"""Recommender layer: match data to analyses and review methodological rigor."""

from researchforge.recommender.match import check_preconditions
from researchforge.recommender.recommend import Recommendation, recommend
from researchforge.recommender.rigor import RigorVerdict, assess_rigor

__all__ = [
    "check_preconditions",
    "assess_rigor",
    "RigorVerdict",
    "recommend",
    "Recommendation",
]
