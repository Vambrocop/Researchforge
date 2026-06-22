"""Tests for commonality_analysis (unique vs common variance partition).

Plants two STRONGLY correlated predictors (x1, x2 share a common factor) plus an
independent one (x3) so a non-trivial COMMON component appears between x1 and x2.
Asserts (a) unique component = R²(full) − R²(full without j) sanity, (b) the
2^p − 1 coefficients SUM to the full-model R², (c) a sizeable common component
exists between the correlated pair, and (d) honest degrade above the 6-predictor cap.
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
        id="commonality_analysis",
        method="Commonality analysis (unique vs common variance)",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_numeric_cols=3, min_rows=20),
    )


def _make_df(seed: int = 0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, n)
    # x1, x2 strongly correlated (share `shared`) -> large COMMON component
    x1 = shared + rng.normal(0, 0.4, n)
    x2 = shared + rng.normal(0, 0.4, n)
    x3 = rng.normal(0, 1, n)  # independent -> mostly UNIQUE
    y = 1.0 * x1 + 1.0 * x2 + 1.5 * x3 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def test_commonality_partition_sums_and_common_component(tmp_path: Path) -> None:
    df = _make_df()
    csv = tmp_path / "comm.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "commonality.csv").exists()
    assert (out / "commonality.png").exists()

    tab = pd.read_csv(out / "commonality.csv")
    # 2^p - 1 = 7 commonality coefficients for 3 predictors
    assert len(tab) == 7

    est = res.estimates
    assert est["n_predictors"] == 3.0
    # decomposition exact: total_unique + total_common == model R²
    assert abs((est["total_unique"] + est["total_common"]) - est["model_r2"]) < 1e-6
    # the CSV column is rounded for display, so its sum carries accumulated rounding
    assert abs(float(tab["coefficient"].sum()) - est["model_r2"]) < 1e-4
    assert est["total_common"] > 0  # correlated predictors -> real shared variance

    # the x1,x2 pairwise common component is the largest common piece (they share a factor)
    common_rows = tab[tab["component"].str.startswith("Common:")].copy()
    top_common = common_rows.sort_values("coefficient", ascending=False).iloc[0]
    assert "x1" in top_common["component"] and "x2" in top_common["component"]
    assert top_common["coefficient"] > 0.05

    # unique component identity: x1's unique = R²(full) - R²(full without x1).
    # x3 is independent and has the largest coefficient -> should carry the largest unique part.
    uniq = tab[tab["component"].str.startswith("Unique:")]
    uq = dict(zip(uniq["component"].str.replace("Unique:", "", regex=False), uniq["coefficient"]))
    assert uq["x3"] > uq["x1"] and uq["x3"] > uq["x2"]


def test_commonality_degrades_above_cap(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 120
    # 7 predictors > cap of 6 -> honest skip
    data = {f"x{i}": rng.normal(0, 1, n) for i in range(1, 8)}
    y = sum((i + 1) * data[f"x{i}"] for i in range(1, 8)) + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, **data})
    csv = tmp_path / "many.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "共同性分析跳过" in res.summary
    assert not (Path(res.output_dir) / "commonality.csv").exists()
