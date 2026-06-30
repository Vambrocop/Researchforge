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
    diagnostic_fit: int = 0   # rank nudge from value-level diagnostics (+ preferred / − argued-against)
    diagnostic_note: str = ""  # which diagnostic(s) drove the nudge (disclosed)


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
    # Rank by rigor tier first (green→yellow→red is the primary key, never crossed),
    # then WITHIN a tier by the real data↔method fit (Stage 4) — so the method that
    # actually suits this data's structure rises, instead of whichever has the fewest
    # listed biases. rigor.score is the final tiebreaker (cleaner method wins ties).
    recs.sort(key=lambda r: (_ORDER[r.rigor.light], -r.score.fit, -r.rigor.score))
    return recs


_PREFER_BONUS = 12   # a fired diagnostic favouring a method lifts it within its rigor tier
_OVER_PENALTY = 12   # …and demotes a method it argues against


def apply_diagnostic_ranking(recs: list[Recommendation], plan) -> list[Recommendation]:
    """Smarter auto-selection (deeper): let the value-level diagnostic plan actually
    MOVE the ranking, not just annotate it. A method that a fired diagnostic prefers
    (e.g. negative_binomial_regression when the count outcome is overdispersed) gets a
    bonus; one it argues against (poisson_regression) gets a penalty. Rigor stays the
    primary key — green→yellow→red is preserved — so a preferred yellow never jumps a
    green; the nudge only reorders WITHIN a tier (via score+fit). Mutates+returns recs."""
    if not plan or not plan.diagnostics:
        return recs
    prefer: dict[str, list[str]] = {}
    over: dict[str, list[str]] = {}
    for dgn in plan.diagnostics:
        for mid in dgn.prefer:
            prefer.setdefault(mid, []).append(dgn.code)
        for mid in dgn.over:
            over.setdefault(mid, []).append(dgn.code)
    for r in recs:
        fit, notes = 0, []
        if r.entry.id in prefer:
            fit += _PREFER_BONUS * len(prefer[r.entry.id])
            notes.append("📋 诊断契合（" + "、".join(dict.fromkeys(prefer[r.entry.id])) + "）↑")
        if r.entry.id in over:
            fit -= _OVER_PENALTY * len(over[r.entry.id])
            notes.append("📋 诊断不利（" + "、".join(dict.fromkeys(over[r.entry.id])) + "）↓")
        r.diagnostic_fit = fit
        r.diagnostic_note = "；".join(notes)
    recs.sort(key=lambda r: (_ORDER[r.rigor.light], -(r.score.fit + r.diagnostic_fit), -r.rigor.score))
    return recs


def select_top(
    fp: DataFingerprint,
    goal: Optional[str] = None,
    top: int = 6,
    catalog: Optional[Catalog] = None,
    plan=None,
    diagnostic_aware: bool = True,
) -> list[Recommendation]:
    """Fast picker: the top feasible analyses, optionally narrowed to a research goal.
    Layered on recommend() so with 75+ methods the user gets a handful, not the menu.

    When ``diagnostic_aware`` (default), the value-level diagnostic plan re-ranks the
    pool so the data's actual structure (overdispersion, non-normality, collinearity…)
    promotes the methods that handle it. ``plan`` can be passed in to avoid re-reading
    the file; otherwise it is built on demand."""
    from researchforge.recommender.goals import entry_matches_goal, resolve_goal

    recs = recommend(fp, catalog=catalog, include_infeasible=True)
    gk = resolve_goal(goal)
    if gk:
        recs = [r for r in recs if entry_matches_goal(r.entry, gk)]
    if diagnostic_aware:
        if plan is None:
            from researchforge.recommender.diagnostics import build_plan

            plan = build_plan(fp, catalog=catalog)
        recs = apply_diagnostic_ranking(recs, plan)
    feasible = [r for r in recs if r.feasible]
    pool = feasible or recs  # nothing feasible -> show the closest (red, needs informed override)
    return pool[:top]
