"""时序族的沉默：哪一条序列被建模了，以及用户能不能改。

Measured on a perfectly ordinary 3-series frame ``[temperature, rainfall, sales]``:
every time-series method forecast ``temperature`` (the first continuous column) and the
report said **nothing** about the other two — while the role detector had flagged ``sales``.

Two distinct defects, both narrow:

1. ``arima`` — alone in its own module — had NO config override for the series, and its
   catalog entry declared no params at all. Its three siblings in ``timeseries.py`` all
   accept ``cfg["value"]``, and every other TS entry declares ``column``/``value``. So a
   user who wanted ``sales`` forecast simply could not ask for it.
2. The smart-selection nudge was gated on the param NAMES ``outcome``/``y``; the TS family
   names its role param ``column``/``value``, so the whole family was invisible to the
   disclosure — and the suggested override key would have been wrong even if it had fired.

Deliberately NOT changed: the TS value column still does not go through ``resolve_outcome``.
Wave H4's reason stands — a univariate series is not an outcome-vs-predictor distinction,
and a name hint like "sales looks like a DV" must never silently redirect a forecast.
Disclosing the choice is the fix; binding by name is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()
_TS_METHODS = ["arima", "exponential_smoothing", "acf_pacf", "theta_method"]


def _multi_series(n=180, seed=3):
    """`temperature` is first (so it is what gets picked), `sales` is what the role detector
    flags — the exact shape where silence costs the user a wrong forecast."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=n, freq="MS").strftime("%Y-%m"),
        "temperature": (15 + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n)).round(2),
        "rainfall": (80 + 30 * np.sin(2 * np.pi * (t + 3) / 12) + rng.normal(0, 8, n)).round(1),
        "sales": (1000 + 6 * t + 120 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 40, n)).round(1),
    })


def _fp(df, tmp_path, name="ts.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _nudge(res) -> str:
    return next((ln for ln in res.summary.split("\n") if ln.startswith("💡")), "")


def _ran(res) -> bool:
    return bool(res.estimates) and not any(
        k in res.summary for k in ("失败", "跳过", "暂未接入", "未检测到"))


@pytest.mark.parametrize("cid", _TS_METHODS)
def test_a_multi_series_frame_discloses_that_a_choice_was_made(cid, tmp_path):
    """The family was invisible to the nudge because it names its param `column`/`value`."""
    fp = _fp(_multi_series(), tmp_path)
    assert fp.likely_outcome == "sales"
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:70]}")
    nud = _nudge(res)
    assert nud, f"{cid} said nothing about which of 3 series it modeled"
    assert "3 个候选列" in nud, nud
    # ...and the suggested key must be one this method actually accepts
    declared = {p.name for p in entry.params}
    assert any(f"config {k} 指定" in nud for k in declared & {"outcome", "column", "value"}), nud


def test_a_single_series_frame_does_not_claim_there_is_a_choice(tmp_path):
    """Silence is only a defect when something was actually chosen over something else."""
    df = _multi_series()[["date", "sales"]]
    fp = _fp(df, tmp_path, name="one.csv")
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "one"))
    if not _ran(res):
        pytest.skip(f"arima degraded: {res.summary[:70]}")
    assert "候选列" not in _nudge(res)


@pytest.mark.parametrize("key", ["value", "column"])
def test_arima_can_finally_be_pointed_at_another_series(key, tmp_path):
    """arima was the ONE branch in timeseries.py without the override, and its entry
    declared no params — the user could not ask for `sales` at all."""
    fp = _fp(_multi_series(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / f"cfg_{key}"),
                       config={key: "sales"})
    if not _ran(res):
        pytest.skip(f"arima degraded: {res.summary[:70]}")
    assert "对 sales" in res.summary, res.summary.split("\n")[-1][:120]
    assert not _nudge(res), "an explicit choice needs no nudge"


def test_arima_declares_the_keys_it_reads(tmp_path):
    entry = _CAT.by_id("arima")
    assert {"column", "value"} <= {p.name for p in entry.params}


def test_the_series_is_still_not_bound_by_name(tmp_path):
    """The guard on this wave: disclosure must NOT turn into name-based redirection. The
    role hint says `sales`, but the forecast stays on the first series unless asked."""
    fp = _fp(_multi_series(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("arima"), output_root=str(tmp_path / "guard"))
    if not _ran(res):
        pytest.skip("arima degraded")
    assert "对 temperature" in res.summary
    assert res.outcome is None, "a TS series is not an `outcome`; it must not be bound as one"
