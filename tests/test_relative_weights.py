"""Tests for relative_weights (Johnson 2000 epsilon relative weights).

Same known-ordering data as dominance (x1 >> x2 >> x3 under mild correlation).
Asserts (a) the recovered weight ranking matches, (b) the relative weights SUM to
the full-model R² (the orthogonalization preserves the variance partition), and
(c) honest degrade with <2 predictors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="relative_weights",
        method="Relative weights (Johnson epsilon)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_numeric_cols=3, min_rows=20),
    )


def _make_df(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    x1 = z + rng.normal(0, 1, n)
    x2 = 0.5 * z + rng.normal(0, 1, n)
    x3 = 0.3 * z + rng.normal(0, 1, n)
    y = 3.0 * x1 + 1.0 * x2 + 0.2 * x3 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def test_relative_weights_recovers_ordering_and_sums_to_r2(tmp_path: Path) -> None:
    df = _make_df()
    csv = tmp_path / "rw.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "relative_weights.csv").exists()
    assert (out / "relative_weights.png").exists()

    tab = pd.read_csv(out / "relative_weights.csv")
    w = dict(zip(tab["predictor"], tab["relative_weight"]))
    assert w["x1"] > w["x2"] > w["x3"]
    rank = dict(zip(tab["predictor"], tab["rank"]))
    assert rank["x1"] == 1 and rank["x2"] == 2 and rank["x3"] == 3

    est = res.estimates
    assert est["n_predictors"] == 3.0
    # the relative weights sum to the model R² (strong correctness check)
    assert abs(est["weights_sum_check"] - est["model_r2"]) < 1e-6
    assert abs(float(tab["relative_weight"].sum()) - est["model_r2"]) < 1e-6
    assert abs(float(tab["pct_of_R2"].sum()) - 100.0) < 0.5


def test_relative_weights_agrees_with_dominance_ranking(tmp_path: Path) -> None:
    # cross-method sanity: relative weights and OLS R² should give the same top predictor
    df = _make_df(seed=2)
    csv = tmp_path / "rw2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    tab = pd.read_csv(Path(res.output_dir) / "relative_weights.csv")
    assert tab.iloc[0]["predictor"] == "x1"


def test_relative_weights_degrades_with_one_predictor(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"y": 2 * x + rng.normal(0, 1, n), "x": x})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "相对权重跳过" in res.summary
    assert not (Path(res.output_dir) / "relative_weights.csv").exists()
