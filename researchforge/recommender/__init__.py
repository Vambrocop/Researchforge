"""Recommender layer: match data to analyses, review rigor, gauge novelty."""

from researchforge.recommender.match import check_preconditions
from researchforge.recommender.novelty import NoveltyHint, novelty_hint
from researchforge.recommender.recommend import Recommendation, recommend
from researchforge.recommender.rigor import RigorVerdict, assess_rigor

__all__ = [
    "check_preconditions",
    "assess_rigor",
    "RigorVerdict",
    "recommend",
    "Recommendation",
    "novelty_hint",
    "NoveltyHint",
]
