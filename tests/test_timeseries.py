"""Tests for the arima end-to-end feature: catalog, profiler, recommender, executor."""

from pathlib import Path

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend
from researchforge.synth import make_timeseries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arima_entry():
    return Catalog.load().by_id("arima")


def _write_timeseries_csv(tmp_path: Path, seed: int = 1) -> Path:
    csv = tmp_path / "timeseries.csv"
    make_timeseries(n_periods=60, seed=seed).to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_loads_arima():
    entry = _arima_entry()
    assert entry is not None, "arima entry not found in catalog"
    assert entry.goal == "predict"
    assert entry.preconditions.is_timeseries is True
    assert entry.preconditions.min_rows == 30


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def test_profiler_detects_timeseries(tmp_path):
    csv = _write_timeseries_csv(tmp_path, seed=1)
    fp = profile_dataset(csv)

    # Critical guard: if these fail, the profiler is NOT detecting the series —
    # do NOT hack around them; investigate profile.py instead.
    assert fp.is_timeseries is True, (
        f"Expected is_timeseries=True but got {fp.is_timeseries}. "
        f"time_col={fp.time_col!r}, is_panel={fp.is_panel}"
    )
    assert fp.is_panel is False, (
        f"Expected is_panel=False but got {fp.is_panel}"
    )


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


def test_recommender_arima_feasible_on_timeseries(tmp_path):
    csv = _write_timeseries_csv(tmp_path)
    fp = profile_dataset(csv)

    recs = recommend(fp)
    rec_map = {r.entry.id: r for r in recs}

    assert "arima" in rec_map, "arima should appear in recommendations"
    assert rec_map["arima"].feasible, (
        f"arima should be feasible for a time-series fingerprint; "
        f"unmet={rec_map['arima'].rigor.unmet}"
    )


def test_recommender_arima_infeasible_on_cross_section(tmp_path):
    import numpy as np
    import pandas as pd

    # Plain cross-section: two continuous columns, no date column at all.
    rng = np.random.default_rng(42)
    csv = tmp_path / "cross.csv"
    pd.DataFrame({"x": rng.normal(0, 1, 80), "y": rng.normal(0, 1, 80)}).to_csv(csv, index=False)
    fp = profile_dataset(csv)

    recs = recommend(fp)
    rec_map = {r.entry.id: r for r in recs}

    assert "arima" in rec_map, "arima should still appear (include_infeasible=True by default)"
    assert not rec_map["arima"].feasible, (
        "arima should NOT be feasible for a plain cross-section"
    )
    assert any("需要时间序列" in u for u in rec_map["arima"].rigor.unmet), (
        f"Expected '需要时间序列' in unmet reasons; got {rec_map['arima'].rigor.unmet}"
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def test_executor_arima(tmp_path):
    csv = _write_timeseries_csv(tmp_path)
    fp = profile_dataset(csv)
    entry = _arima_entry()

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert (out / "forecast.csv").exists(), "forecast.csv should be produced"
    assert (out / "report.md").exists(), "report.md should be produced"
    assert "aic" in res.estimates, (
        f"estimates should contain 'aic'; got keys: {list(res.estimates.keys())}"
    )

    import math

    import pandas as pd

    assert math.isfinite(res.estimates["aic"])
    fc = pd.read_csv(out / "forecast.csv")
    assert list(fc.columns) == ["step", "forecast"]
    assert len(fc) == 10  # forecasts the contracted 10 periods
    # honesty disclosure: order (1,1,1) is hardcoded, not auto-selected -- AIC is informational only.
    assert "阶数固定为 (1,1,1)" in res.summary


def test_arima_degenerate_series_fails_gracefully(tmp_path):
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=40, freq="MS")
    df = pd.DataFrame({"date": dates, "value": [5.0] * 40})  # constant -> cannot fit
    csv = tmp_path / "const_ts.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _arima_entry(), output_root=str(tmp_path / "outputs"))

    assert "aic" not in res.estimates  # did not fit a model
    assert "失败" in res.summary or "不足" in res.summary  # graceful, explained failure
