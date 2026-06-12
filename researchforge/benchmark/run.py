"""Run the benchmark: score the engine on each case and record the result."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from researchforge import __version__
from researchforge.benchmark.cases import BenchmarkCase, default_cases
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend


@dataclass
class BenchmarkReport:
    version: str
    n_cases: int
    profile_accuracy: float
    recommendation_score: float
    recovery_pass_rate: float
    recovery_mae: float
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    cases: Optional[list[BenchmarkCase]] = None, work_dir: Optional[str] = None
) -> BenchmarkReport:
    cases = cases or default_cases()
    catalog = Catalog.load()
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="rf_bench_"))
    work.mkdir(parents=True, exist_ok=True)

    profile_hits = 0
    rec_scores: list[float] = []
    rec_errors: list[float] = []
    rec_passes: list[bool] = []
    details: list[dict] = []

    for case in cases:
        csv = work / f"{case.name}.csv"
        case.build().to_csv(csv, index=False)
        fp = profile_dataset(csv)

        profile_ok = fp.is_panel == case.expect_panel
        profile_hits += int(profile_ok)

        feasible = {r.entry.id for r in recommend(fp) if r.feasible}
        checks = [rid in feasible for rid in case.expect_feasible]
        checks += [rid not in feasible for rid in case.expect_infeasible]
        rec_score = sum(checks) / len(checks) if checks else 1.0
        rec_scores.append(rec_score)

        detail = {"case": case.name, "profile_ok": profile_ok, "rec_score": round(rec_score, 3)}

        if case.recover:
            aid, var, true_v, tol = case.recover
            res = run_analysis(fp, catalog.by_id(aid), output_root=str(work / "outputs"))
            est = res.estimates.get(var)
            if est is not None:
                err = abs(est - true_v)
                rec_errors.append(err)
                rec_passes.append(err <= tol)
                detail.update(
                    recover_var=var, estimate=round(est, 4), true=true_v,
                    error=round(err, 4), passed=bool(err <= tol),
                )
        details.append(detail)

    n = len(cases)
    return BenchmarkReport(
        version=__version__,
        n_cases=n,
        profile_accuracy=round(profile_hits / n, 3),
        recommendation_score=round(sum(rec_scores) / len(rec_scores), 3),
        recovery_pass_rate=round(sum(rec_passes) / len(rec_passes), 3) if rec_passes else 1.0,
        recovery_mae=round(sum(rec_errors) / len(rec_errors), 4) if rec_errors else 0.0,
        details=details,
    )


def save_report(report: BenchmarkReport, out_dir: str = "benchmark/results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report.version}.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
