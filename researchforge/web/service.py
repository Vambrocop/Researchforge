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
        }
        for r in recs
    ]

    return {"fingerprint": fingerprint, "recommendations": recommendations}


def run_for_path(
    path: str | Path,
    analysis_id: str,
    output_root: str = "outputs",
) -> dict:
    """Run one analysis and return a result dict.

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
    res = run_analysis(fp, entry, output_root=output_root)

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
