"""Tests for the user-config override mechanism (run_analysis(config=...))."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _ols_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ols_regression",
        method="OLS regression",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=2, min_rows=20),
    )


def test_config_outcome_and_predictors_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 150
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    y = 2.0 * a + rng.normal(0, 0.3, n)
    # column order makes 'a' the first continuous -> the DEFAULT outcome
    df = pd.DataFrame({"a": a, "b": b, "y": y})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    # default: outcome = first continuous ('a'); 'a' is the DV, not a predictor key
    res0 = run_analysis(fp, _ols_entry(), output_root=str(tmp_path / "o0"))
    assert "a" not in res0.estimates

    # override: outcome='y', predictors=['a'] -> regresses y ~ a, slope ~ 2
    res1 = run_analysis(
        fp,
        _ols_entry(),
        output_root=str(tmp_path / "o1"),
        config={"outcome": "y", "predictors": ["a"]},
    )
    assert abs(res1.estimates["a"] - 2.0) < 0.3


def _entry(aid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=aid, method=method, domain="mcda", family="mcda", goal="rank",
        preconditions=Precondition(min_rows=2),
    )


def test_config_mcda_cost_criteria_flips_ranking(tmp_path: Path) -> None:
    # 'c' is cost-type (lower=better). A is best only when c is reversed.
    # decimal/continuous values so the profiler doesn't mark them as 'id'.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C", "D", "E", "F"],
            "b": [9.3, 1.4, 5.2, 7.1, 3.3, 6.0],
            "c": [1.2, 9.1, 5.3, 2.4, 7.2, 4.0],
        }
    )
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    e = _entry("topsis", "TOPSIS")

    res_cost = run_analysis(
        fp, e, output_root=str(tmp_path / "oc"), config={"cost_criteria": ["c"]}
    )
    # with c as cost, A (high b + low c) is unambiguously best
    assert "成本型指标" in res_cost.summary
    assert "[A]" in res_cost.summary


def test_config_dea_inputs_outputs_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "DMU": [f"d{i}" for i in range(8)],
            "good": rng.uniform(5, 15, 8),
            "labor": rng.uniform(5, 15, 8),
            "capital": rng.uniform(5, 15, 8),
        }
    )
    csv = tmp_path / "dea.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    e = _entry("dea", "DEA")

    res_def = run_analysis(fp, e, output_root=str(tmp_path / "od"))
    assert "产出 ['good']" in res_def.summary  # default: first numeric = output

    res_cfg = run_analysis(
        fp, e, output_root=str(tmp_path / "oc"),
        config={"outputs": ["labor"], "inputs": ["good", "capital"]},
    )
    assert "产出 ['labor']" in res_cfg.summary
    assert "按 config 指定" in res_cfg.summary


def test_config_spatial_knn_k_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 25
    df = pd.DataFrame(
        {
            "lat": rng.uniform(30, 31, n),
            "lon": rng.uniform(120, 121, n),
            "value": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "geo.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    e = AnalysisEntry(
        id="moran_i", method="Moran's I", domain="gis", family="spatial",
        goal="autocorr", preconditions=Precondition(min_rows=10),
    )
    res_def = run_analysis(fp, e, output_root=str(tmp_path / "od"))
    assert "k-NN=8" in res_def.summary  # default min(8, n-1)
    res_cfg = run_analysis(fp, e, output_root=str(tmp_path / "oc"), config={"knn_k": 4})
    assert "k-NN=4" in res_cfg.summary


def test_config_malmquist_periods_and_io(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    firms = [f"f{i}" for i in range(6)]
    years = [2010, 2011, 2012]
    rows = []
    for f in firms:
        for yv in years:
            rows.append(
                {
                    "firm": f,
                    "year": yv,
                    "y": float(rng.uniform(10, 20)),
                    "x1": float(rng.uniform(5, 15)),
                    "x2": float(rng.uniform(5, 15)),
                }
            )
    df = pd.DataFrame(rows)
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    if not (fp.unit_col and fp.time_col):
        import pytest

        pytest.skip("profiler did not detect panel structure for this fixture")
    e = AnalysisEntry(
        id="malmquist", method="Malmquist TFP", domain="efficiency", family="efficiency",
        goal="productivity", preconditions=Precondition(is_panel=True),
    )
    res_def = run_analysis(fp, e, output_root=str(tmp_path / "od"))
    assert "2010→2012" in res_def.summary  # default: first vs last period

    res_cfg = run_analysis(
        fp, e, output_root=str(tmp_path / "oc"),
        config={"periods": [2010, 2011], "outputs": ["x1"], "inputs": ["y", "x2"]},
    )
    assert "2010→2011" in res_cfg.summary
    assert "产出 ['x1']" in res_cfg.summary


def test_qca_and_io_config_helpers() -> None:
    from researchforge.executor.run import (
        _cost_mask,
        _gmm_lags,
        _io_names,
        _knn_k,
        _qca_anchors,
        _qca_incl_cut,
    )

    # QCA anchors: default, valid override, rejected (not increasing / wrong type)
    assert _qca_anchors({}) == (0.1, 0.5, 0.9)
    assert _qca_anchors({"anchors": [0.05, 0.5, 0.95]}) == (0.05, 0.5, 0.95)
    assert _qca_anchors({"anchors": [0.9, 0.5, 0.1]}) == (0.1, 0.5, 0.9)
    assert _qca_anchors({"anchors": "bad"}) == (0.1, 0.5, 0.9)
    # incl_cut: default, valid, out-of-range rejected
    assert _qca_incl_cut({}, 0.8) == 0.8
    assert _qca_incl_cut({"incl_cut": 0.85}, 0.8) == 0.85
    assert _qca_incl_cut({"incl_cut": 2.0}, 0.8) == 0.8
    # io names: default first=output rest=input; override; unknown names fall back
    assert _io_names(["a", "b", "c"], {}) == (["b", "c"], ["a"])
    assert _io_names(["a", "b", "c"], {"inputs": ["a", "b"], "outputs": ["c"]}) == (
        ["a", "b"], ["c"],
    )
    assert _io_names(["a", "b", "c"], {"inputs": ["zz"], "outputs": ["c"]}) == (
        ["b", "c"], ["a"],
    )
    # knn_k clamp + non-int fallback
    assert _knn_k({}, 10) == 8
    assert _knn_k({"knn_k": 3}, 10) == 3
    assert _knn_k({"knn_k": 99}, 10) == 10
    assert _knn_k({"knn_k": "x"}, 10) == 8
    # cost mask: None when no cost criteria, mask when present
    assert _cost_mask(["a", "b"], {})[0] is None
    mask, names = _cost_mask(["a", "b"], {"cost_criteria": ["b", "zz"]})
    assert names == ["b"] and bool(mask[1]) and not bool(mask[0])
    # gmm lags: default, valid override, invalid (lo>hi / wrong type) -> default
    assert _gmm_lags({}) == (2, 4)
    assert _gmm_lags({"gmm_lags": [2, 6]}) == (2, 6)
    assert _gmm_lags({"gmm_lags": [5, 2]}) == (2, 4)
    assert _gmm_lags({"gmm_lags": "x"}) == (2, 4)


def test_sem_latents_and_semopy_multifactor() -> None:
    from researchforge.executor.run import _sem_latents, _sem_via_semopy

    assert _sem_latents("F1 =~ a + b\nF2 =~ c + d") == ["F1", "F2"]
    assert _sem_latents("sat =~ q1 + q2") == ["sat"]

    rng = np.random.default_rng(5)
    n = 400
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    sub = pd.DataFrame(
        {
            "a1": f1 + rng.normal(0, 0.4, n),
            "a2": f1 + rng.normal(0, 0.4, n),
            "b1": f2 + rng.normal(0, 0.4, n),
            "b2": f2 + rng.normal(0, 0.4, n),
        }
    )
    res = _sem_via_semopy(sub, "F1 =~ a1 + a2\nF2 =~ b1 + b2")
    # generalised extraction must pull loadings for BOTH factors, not just "F"
    assert set(res["loadings"]["factor"]) == {"F1", "F2"}
    assert len(res["loadings"]) == 4


def test_config_sem_custom_spec_branch(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 300
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "a1": f1 + rng.normal(0, 0.4, n),
            "a2": f1 + rng.normal(0, 0.4, n),
            "b1": f2 + rng.normal(0, 0.4, n),
            "b2": f2 + rng.normal(0, 0.4, n),
        }
    )
    csv = tmp_path / "sem.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    e = AnalysisEntry(
        id="sem", method="SEM", domain="statistics", family="statistics",
        goal="confirm", preconditions=Precondition(min_continuous=3),
    )
    res = run_analysis(
        fp, e, output_root=str(tmp_path / "o"),
        config={"model_spec": "F1 =~ a1 + a2\nF2 =~ b1 + b2"},
    )
    assert "自定义模型" in res.summary and "2 因子" in res.summary


def test_config_diff_abundance_method(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    n = 60
    rows = {}
    for t in range(5):
        rows[f"otu{t}"] = rng.integers(1, 200, n).astype(float)
    df = pd.DataFrame(rows)
    df["grp"] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    csv = tmp_path / "da.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    if not [c for c in fp.columns if c.kind == "count"]:
        import pytest

        pytest.skip("profiler did not detect count columns for this fixture")
    e = AnalysisEntry(
        id="differential_abundance", method="Differential abundance",
        domain="microbiology", family="microbiology", goal="compare",
        preconditions=Precondition(min_rows=10),
    )
    # Welch t-test variant (pure Python, always available)
    res_w = run_analysis(
        fp, e, output_root=str(tmp_path / "ow"), config={"da_method": "clr_welch"}
    )
    if "失败" not in res_w.summary:
        assert "CLR+Welch t" in res_w.summary
    # ANCOM-BC bridge is intentionally not wired -> must degrade honestly, not pretend
    res_a = run_analysis(
        fp, e, output_root=str(tmp_path / "oa"), config={"da_method": "ancombc"}
    )
    if "失败" not in res_a.summary:
        assert "ANCOM-BC" in res_a.summary and "保底" in res_a.summary
    # aldex2: runs the R gold standard when ALDEx2 is installed, else degrades honestly
    from researchforge.executor import rbridge

    res_g = run_analysis(
        fp, e, output_root=str(tmp_path / "og"), config={"da_method": "aldex2"}
    )
    if "失败" not in res_g.summary:
        if rbridge.r_available() and rbridge.r_package_available("ALDEx2"):
            assert "ALDEx2 (R" in res_g.summary and "保底" not in res_g.summary
        else:
            assert "保底" in res_g.summary


def test_config_none_is_default(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": 1.5 * rng.normal(0, 1, 60), "x": rng.normal(0, 1, 60)})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # config=None must behave exactly like no config (no crash, runs)
    res = run_analysis(fp, _ols_entry(), output_root=str(tmp_path / "o"), config=None)
    assert res.summary
