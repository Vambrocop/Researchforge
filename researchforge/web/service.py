"""Pure, testable service functions — no web framework objects.

These are the only functions the FastAPI app should call; they can also be
tested in isolation without spinning up an HTTP server.
"""

from __future__ import annotations

from pathlib import Path


def analyze_path(path: str | Path) -> dict:
    """Profile a CSV/Excel file and return a recommendation menu.

    Returns a dict with keys:
      fingerprint: {n_rows, n_cols, is_panel, is_timeseries, time_col,
                    unit_col, n_issues, columns:[{name, kind},...]}
      recommendations: [{id, method, family, light, score, feasible,
                         note, biases:[...]},...]
    """
    from researchforge.profiler import profile_dataset
    from researchforge.recommender import recommend

    fp = profile_dataset(Path(path))
    recs = recommend(fp)

    fingerprint = {
        "n_rows": fp.n_rows,
        "n_cols": fp.n_cols,
        "is_panel": fp.is_panel,
        "is_timeseries": fp.is_timeseries,
        "time_col": fp.time_col,
        "unit_col": fp.unit_col,
        "n_issues": len(fp.issues),
        "columns": [{"name": c.name, "kind": c.kind} for c in fp.columns],
        "issues": [
            {"kind": i.kind, "column": i.column, "severity": i.severity, "detail": i.detail}
            for i in fp.issues
        ],
    }

    recommendations = [
        {
            "id": r.entry.id,
            "method": r.entry.method,
            "family": r.entry.family,
            "light": r.rigor.light,
            "score": r.rigor.score,
            "feasible": r.feasible,
            "note": r.rigor.note,
            "biases": list(r.rigor.biases),
            "methodology_score": r.score.as_dict(),
            "score_note": r.score.note,
        }
        for r in recs
    ]

    return {"fingerprint": fingerprint, "recommendations": recommendations}


def clean_path(path: str | Path, cleaned_out: str | Path) -> dict:
    """Run a cleaning plan on a CSV and save the cleaned file.

    Returns a dict with keys:
      plan: [{action, column, reason}, ...]
      log:  [{action, column, reason, applied, detail}, ...]
    """
    from researchforge.profiler import profile_dataset
    from researchforge.profiler.profile import read_table
    from researchforge.cleaning import make_cleaning_plan, apply_cleaning_plan

    fp = profile_dataset(Path(path))
    plan = make_cleaning_plan(fp)
    df = read_table(Path(path))
    cleaned, log = apply_cleaning_plan(df, plan)
    cleaned.to_csv(Path(cleaned_out), index=False, encoding="utf-8")
    return {
        "plan": [{"action": s.action, "column": s.column, "reason": s.reason} for s in plan],
        "log": log,
    }


def run_for_path(
    path: str | Path,
    analysis_id: str,
    output_root: str = "outputs",
    config: dict | None = None,
) -> dict:
    """Run one analysis and return a result dict. `config` carries user overrides
    for the engine's substantive defaults (column roles, anchors, …).

    Returns {"summary", "output_dir", "files", "report", "estimates"}
    or {"error": "unknown analysis"} if the id is not in the catalog.
    """
    from researchforge.catalog import Catalog
    from researchforge.executor import run_analysis
    from researchforge.profiler import profile_dataset

    entry = Catalog.load().by_id(analysis_id)
    if entry is None:
        return {"error": "unknown analysis"}

    fp = profile_dataset(Path(path))
    res = run_analysis(fp, entry, output_root=output_root, config=config)

    report_text = ""
    try:
        report_text = Path(res.report_path).read_text(encoding="utf-8")
    except Exception:
        pass

    return {
        "summary": res.summary,
        "output_dir": res.output_dir,
        "files": res.files,
        "report": report_text,
        "estimates": res.estimates,
    }
