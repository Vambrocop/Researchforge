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


def test_config_none_is_default(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": 1.5 * rng.normal(0, 1, 60), "x": rng.normal(0, 1, 60)})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # config=None must behave exactly like no config (no crash, runs)
    res = run_analysis(fp, _ols_entry(), output_root=str(tmp_path / "o"), config=None)
    assert res.summary
