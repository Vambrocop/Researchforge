"""Methodological rigor review — the 🟢🟡🔴 verdict for a data/method pairing.

green  = hard preconditions met, no notable caveats
yellow = met but with known biases or borderline sample (proceed with care)
red    = data does not meet the method's hard preconditions (informed override only)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.recommender.match import check_preconditions


class RigorVerdict(BaseModel):
    light: str  # green | yellow | red
    score: int
    unmet: list[str] = Field(default_factory=list)
    biases: list[str] = Field(default_factory=list)
    note: str = ""


def assess_rigor(fp: DataFingerprint, entry: AnalysisEntry) -> RigorVerdict:
    met, unmet = check_preconditions(fp, entry.preconditions)
    biases = list(entry.biases)

    if not met:
        score = max(0, 60 - 10 * len(unmet))
        return RigorVerdict(
            light="red", score=score, unmet=unmet, biases=biases,
            note="数据不满足前提：" + "；".join(unmet),
        )

    score = 100 - 8 * len(biases)
    borderline = False
    min_rows = entry.preconditions.min_rows
    if min_rows is not None and fp.n_rows < 1.5 * min_rows:
        borderline = True
        score -= 10
    score = max(0, min(100, score))

    if biases or borderline:
        parts: list[str] = []
        if borderline:
            parts.append("样本量偏小")
        parts.extend(biases)
        return RigorVerdict(
            light="yellow", score=score, biases=biases,
            note="可做，但需注意：" + "、".join(parts),
        )

    return RigorVerdict(light="green", score=score, biases=biases, note="前提满足")
