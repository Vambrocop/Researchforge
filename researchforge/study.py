"""Study mode: full-pipeline orchestration — profile -> (optional clean) -> pick a
diverse handful of methods -> run each -> assemble one honest merged report.

Pure orchestration over already-reviewed pieces (profiler / recommender / catalog /
executor); this module makes ZERO changes to ``run_analysis`` or any executor
internals — every call is a plain, isolated ``run_analysis(fp, entry, ...)``, exactly
as ``cli.py``'s ``run`` command already does one at a time. See
docs/design-study-mode.md (Wave I, Fable-drafted, authoritative — its §6 STOP points
are hard boundaries, not suggestions we get to redesign around).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from researchforge import __version__
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.executor.run import _run_dir
from researchforge.profiler import profile_dataset
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.recommender import build_plan, select_top
from researchforge.study_report import render_report

# Always-on helpers that answer "what does the data look like", not a substantive
# research method — never one of the K picks. Mirrors cli.py's ``_PICK_SKIP``
# exactly; kept as an independent copy because study.py must not import cli's
# private names (see CLAUDE.md read-code discipline / the build prompt's red line).
_PICK_SKIP = {"descriptive_stats", "correlation", "correlation_matrix", "summary_statistics"}

# Candidate pool size select_top draws from before the diversity filter narrows it
# down to the requested K (docs/design-study-mode.md §2 step 2-3).
_POOL_SIZE = 12


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _auto_clean(path: str, fp: DataFingerprint, study_dir: Path):
    """Same semantics as cli.py's ``_auto_clean_before_run`` (apply only when the
    plan has real, non-advisory steps) but writes the cleaned file + log INSIDE the
    study dir and returns the structured log for report disclosure instead of
    printing it. Returns (run_path, fp, clean_log) — clean_log is None when there
    was nothing substantive to clean (fp/path are returned unchanged)."""
    from researchforge.cleaning import apply_cleaning_plan, make_cleaning_plan, write_cleaning_log
    from researchforge.profiler.profile import read_table

    plan = make_cleaning_plan(fp)
    applicable = [s for s in plan if not s.action.startswith("flag_")]
    if not applicable:
        return path, fp, None
    df = read_table(Path(path))
    cleaned, log = apply_cleaning_plan(df, plan)
    out_path = study_dir / "cleaned_data.csv"
    cleaned.to_csv(out_path, index=False, encoding="utf-8")
    write_cleaning_log(log, study_dir / "cleaned_data.cleaning.json")
    new_fp = profile_dataset(str(out_path))
    return str(out_path), new_fp, log


def _diversity_pick(recs: list, k: int) -> list:
    """Greedy top-ranked pick, <=1 per family, backfilled by rank if the number of
    distinct families is short of k. ``recs`` must already be sorted best-first
    (select_top's own ordering) and already exclude _PICK_SKIP. See
    docs/design-study-mode.md §2 step 3 / §6 STOP point 2 (if this still can't
    fill k, the caller reports the actual count — never loosens the rigor gate)."""
    chosen: list = []
    seen_families: set = set()
    for r in recs:
        if len(chosen) >= k:
            break
        if r.entry.family not in seen_families:
            chosen.append(r)
            seen_families.add(r.entry.family)
    if len(chosen) < k:
        chosen_ids = {r.entry.id for r in chosen}
        for r in recs:
            if len(chosen) >= k:
                break
            if r.entry.id not in chosen_ids:
                chosen.append(r)
                chosen_ids.add(r.entry.id)
    return chosen


def run_study(
    path: str,
    goal: Optional[str] = None,
    top: int = 3,
    clean: bool = False,
    config: Optional[dict] = None,
) -> dict:
    """Orchestrate one merged study: profile -> (optional clean) -> pick up to
    ``top`` diverse substantive methods -> run each -> write one honest report.

    Never raises for a single method's failure — each ``run_analysis`` call is
    independently try/excepted, so one crash can never sink the study (mirrors
    ``run_analysis``'s own "never crash the run" contract, one layer up). Returns a
    dict even when every method fails; the caller can detect that via an empty
    ``methods_run``.

    Returns {study_dir, report_path, report_text, methods_run: list[str], meta: dict}.
    """
    fp = profile_dataset(path)
    study_dir = _run_dir("outputs", "study")

    clean_log = None
    if clean:
        _, fp, clean_log = _auto_clean(path, fp, study_dir)

    catalog = Catalog.load()
    plan = build_plan(fp, catalog=catalog)
    pool = select_top(
        fp, goal=goal, top=_POOL_SIZE, catalog=catalog, plan=plan, diagnostic_aware=True
    )
    substantive = [r for r in pool if r.entry.id not in _PICK_SKIP]
    chosen = _diversity_pick(substantive, top)

    # §0 baseline: descriptive_stats, unconditional, never counted toward K.
    base_entry = catalog.by_id("descriptive_stats")
    base_result = None
    base_error: Optional[str] = None
    if base_entry is not None:
        try:
            base_result = run_analysis(
                fp, base_entry, output_root=str(study_dir), config=config
            )
        except Exception as err:  # noqa: BLE001 — the baseline must not sink the study
            base_error = f"{type(err).__name__}: {str(err)[:200]}"
    else:
        base_error = "catalog 中未找到 descriptive_stats（基线跳过）"

    run_entries: list[dict] = []
    methods_run: list[str] = []
    for rec in chosen:
        entry = rec.entry
        result = None
        error: Optional[str] = None
        try:
            result = run_analysis(fp, entry, output_root=str(study_dir), config=config)
            methods_run.append(entry.id)
        except Exception as err:  # noqa: BLE001 — one method's crash must not sink the study
            error = f"{type(err).__name__}: {str(err)[:200]}"
        run_entries.append({"rec": rec, "result": result, "error": error})

    report_text = render_report(
        fp=fp,
        plan=plan,
        goal=goal,
        base_entry=base_entry,
        base_result=base_result,
        base_error=base_error,
        run_entries=run_entries,
        clean_log=clean_log,
        study_dir=study_dir,
        requested_k=top,
    )
    report_path = study_dir / "study_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    meta = {
        "engine_version": __version__,
        "data_path": str(fp.path),
        "data_sha256": _sha256(fp.path),
        "n_rows": fp.n_rows,
        "n_cols": fp.n_cols,
        "goal": goal,
        "top_requested": top,
        "methods": [
            {
                "id": e["rec"].entry.id,
                "method": e["rec"].entry.method,
                "family": e["rec"].entry.family,
                "run_dir": e["result"].output_dir if e["result"] else None,
                "status": "ok" if e["result"] else "orchestration_failed",
                "error": e["error"],
            }
            for e in run_entries
        ],
        "baseline": {
            "id": "descriptive_stats",
            "run_dir": base_result.output_dir if base_result else None,
            "status": "ok" if base_result else "failed",
            "error": base_error,
        },
        "config": config or {},
        "clean_applied": clean_log is not None,
        "study_dir": str(study_dir),
    }
    (study_dir / "study_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "study_dir": str(study_dir),
        "report_path": str(report_path),
        "report_text": report_text,
        "methods_run": methods_run,
        "meta": meta,
    }
