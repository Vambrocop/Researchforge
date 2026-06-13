"""Produce a ranked recommendation menu from a DataFingerprint."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from researchforge.catalog.registry import Catalog
from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.recommender.rigor import RigorVerdict, assess_rigor
from researchforge.recommender.scoring import MethodologyScore, score_method

_ORDER = {"green": 0, "yellow": 1, "red": 2}


class Recommendation(BaseModel):
    entry: AnalysisEntry
    rigor: RigorVerdict
    feasible: bool  # green/yellow feasible; red needs informed override
    score: MethodologyScore  # multi-dimensional methodology score card


def recommend(
    fp: DataFingerprint,
    catalog: Optional[Catalog] = None,
    include_infeasible: bool = True,
) -> list[Recommendation]:
    catalog = catalog or Catalog.load()
    recs: list[Recommendation] = []
    for entry in catalog.all():
        rigor = assess_rigor(fp, entry)
        feasible = rigor.light in {"green", "yellow"}
        if feasible or include_infeasible:
            recs.append(
                Recommendation(
                    entry=entry, rigor=rigor, feasible=feasible,
                    score=score_method(fp, entry, rigor),
                )
            )
    # rank by feasibility/rigor first (unchanged); the score card is informational
    recs.sort(key=lambda r: (_ORDER[r.rigor.light], -r.rigor.score))
    return recs
