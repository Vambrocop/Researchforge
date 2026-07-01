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
      diagnostic_plan: {outcome, diagnostics:[{code, finding, detail,
                         prefer:[...], over:[...]},...]}  (smarter auto-selection)
    """
    from researchforge.catalog.registry import Catalog
    from researchforge.profiler import profile_dataset
    from researchforge.recommender import apply_diagnostic_ranking, build_plan, recommend
    from researchforge.recommender.goals import GOALS, entry_matches_goal

    fp = profile_dataset(Path(path))
    catalog = Catalog.load()
    plan = build_plan(fp, catalog=catalog)
    # diagnostic-aware ranking: the data's actual structure re-ranks the menu within
    # each rigor tier (preferred methods rise, argued-against fall). Disclosed per rec.
    recs = apply_diagnostic_ranking(recommend(fp, catalog=catalog), plan)

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
        # non-binding semantic role hints (smarter auto-selection slice 1)
        "likely_outcome": fp.likely_outcome,
        "likely_treatment": fp.likely_treatment,
        "role_hint_reason": fp.role_hint_reason,
    }

    # auto-diagnose → recommend-with-a-plan (smarter auto-selection slice 2): value-level
    # findings mapped to model-choice nudges; advisory only, does not change run defaults.
    diagnostic_plan = {
        "outcome": plan.outcome,
        "diagnostics": [d.model_dump() for d in plan.diagnostics],
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
            # diagnostic-aware ranking nudge (smarter auto-selection, deeper)
            "diagnostic_fit": r.diagnostic_fit,
            "diagnostic_note": r.diagnostic_note,
            # goal keys this method matches (server-side, mirrors the CLI selector incl. keywords)
            "goals": [k for k in GOALS if entry_matches_goal(r.entry, k)],
            # machine-readable config params (single source of truth for the run form)
            "params": [p.model_dump() for p in r.entry.params],
        }
        for r in recs
    ]

    # goal taxonomy (key + label, in canonical order) so the frontend never drifts from goals.py
    goals = [{"key": k, "label": GOALS[k]["label"]} for k in GOALS]

    return {
        "fingerprint": fingerprint,
        "recommendations": recommendations,
        "goals": goals,
        "diagnostic_plan": diagnostic_plan,
    }


def analyze_folder_files(items) -> list[dict]:
    """Batch-profile a folder of tables. `items` is an iterable of (file_id, filename,
    path). For each table returns a compact summary — shape, structure flags, and the
    top-3 feasible recommendations — so the UI can show a folder overview and let the
    user open any file in the full single-file flow. Never raises: a per-file failure
    is captured as {ok: False, error: ...} so one bad table doesn't sink the batch."""
    from researchforge.profiler import profile_dataset
    from researchforge.recommender import select_top

    out: list[dict] = []
    for file_id, filename, path in items:
        rec: dict = {"file_id": file_id, "filename": filename}
        try:
            fp = profile_dataset(Path(path))
            tops = select_top(fp, top=3)
            rec.update({
                "ok": True,
                "n_rows": fp.n_rows,
                "n_cols": fp.n_cols,
                "is_panel": fp.is_panel,
                "is_timeseries": fp.is_timeseries,
                "n_issues": len(fp.issues),
                "top": [
                    {"id": r.entry.id, "method": r.entry.method,
                     "family": r.entry.family, "light": r.rigor.light}
                    for r in tops
                ],
            })
        except Exception as e:  # a single unreadable table must not sink the batch
            rec.update({"ok": False, "error": str(e)[:200]})
        out.append(rec)
    return out


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
