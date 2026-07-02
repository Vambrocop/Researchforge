"""P2-2: a branch handler that raises (e.g. in setup/precondition code, outside its
own internal try/except) must not crash run_analysis — the dispatch chokepoint
should degrade to a report instead of propagating, skipping teardown, and handing
the web layer an opaque 500 / the CLI a raw traceback."""

from pathlib import Path

import pytest

from researchforge.catalog.schema import AnalysisEntry
from researchforge.executor._branch_api import BRANCH_REGISTRY
from researchforge.executor.run import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.synth import make_panel

_DUMMY_ID = "test_dispatch_guard_boom"


def _boom(ctx):
    raise ValueError("kaboom from dummy handler")


@pytest.fixture
def dummy_handler():
    assert _DUMMY_ID not in BRANCH_REGISTRY, "dummy id collides with a real branch"
    BRANCH_REGISTRY[_DUMMY_ID] = _boom
    try:
        yield _DUMMY_ID
    finally:
        BRANCH_REGISTRY.pop(_DUMMY_ID, None)


def test_raising_handler_still_yields_degraded_report(tmp_path, dummy_handler):
    csv = tmp_path / "panel.csv"
    make_panel(seed=1).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = AnalysisEntry(
        id=dummy_handler,
        method="Dummy Boom Method",
        domain="test",
        family="test",
        goal="describe",
    )

    # (a) no exception propagates
    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))

    # (b) teardown still ran: report.md exists on disk and is listed in res.files
    out = Path(res.output_dir)
    assert (out / "report.md").exists()
    assert (out / "analysis_code.py").exists()
    assert "report.md" in res.files

    # (c) the truncated failure message made it into the summary
    assert f"⚠ {_DUMMY_ID} 执行失败" in res.summary
    assert "ValueError" in res.summary
    assert "kaboom from dummy handler" in res.summary
