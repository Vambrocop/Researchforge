"""PROCESS-style mediation/moderation extensions: serial, parallel, moderated moderation.

Simulated data with planted path coefficients; assert the engine recovers the right
indirect / interaction structure (sign + bootstrap-CI / analytic-p significance) and
that the serial indirect matches an independent OLS recompute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _run(csv, aid, tmp_path, config=None):
    fp = profile_dataset(csv)
    return run_analysis(fp, _CAT.by_id(aid), output_root=str(tmp_path / "o"), config=config)


# --------------------------------------------------------------------------- #
# serial_mediation (X -> M1 -> M2 -> Y)
# --------------------------------------------------------------------------- #
def _serial_df(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    M1 = 0.6 * X + rng.normal(0, 0.6, n)
    M2 = 0.5 * M1 + 0.3 * X + rng.normal(0, 0.6, n)
    Y = 0.4 * M2 + 0.2 * M1 + 0.1 * X + rng.normal(0, 0.6, n)
    return pd.DataFrame({"Y": Y, "X": X, "M1": M1, "M2": M2})


def test_serial_recovers_serial_path(tmp_path: Path) -> None:
    csv = tmp_path / "serial.csv"
    _serial_df().to_csv(csv, index=False)
    res = _run(csv, "serial_mediation", tmp_path,
               config={"x": "X", "m1": "M1", "m2": "M2", "y": "Y"})
    e = res.estimates
    # planted serial path a1·d21·b2 ≈ 0.6·0.5·0.4 = 0.12 > 0, CI excludes 0
    assert e["indirect_serial"] > 0
    assert e["indirect_serial_lo"] > 0
    assert e["total_indirect"] > e["indirect_serial"]   # plus the via-M1/M2 paths
    assert (Path(res.output_dir) / "serial_mediation_effects.csv").exists()


def test_serial_matches_independent_ols(tmp_path: Path) -> None:
    df = _serial_df(seed=4)
    csv = tmp_path / "serial2.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "serial_mediation", tmp_path,
               config={"x": "X", "m1": "M1", "m2": "M2", "y": "Y"})
    import statsmodels.api as sm

    a1 = sm.OLS(df["M1"], sm.add_constant(df["X"])).fit().params.iloc[1]
    r2 = sm.OLS(df["M2"], sm.add_constant(df[["X", "M1"]])).fit().params
    d21 = r2.iloc[2]
    r3 = sm.OLS(df["Y"], sm.add_constant(df[["X", "M1", "M2"]])).fit().params
    b2 = r3.iloc[3]
    expect = float(a1 * d21 * b2)
    assert abs(res.estimates["indirect_serial"] - expect) < 1e-4


def test_serial_degrades_few_columns(tmp_path: Path) -> None:
    csv = tmp_path / "few.csv"
    pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) * 1.1}).to_csv(csv, index=False)
    res = _run(csv, "serial_mediation", tmp_path)
    assert "跳过" in res.summary
    assert "indirect_serial" not in res.estimates


# --------------------------------------------------------------------------- #
# parallel_mediation
# --------------------------------------------------------------------------- #
def _parallel_df(n=300, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    M1 = 0.5 * X + rng.normal(0, 0.6, n)
    M2 = 0.4 * X + rng.normal(0, 0.6, n)
    Y = 0.3 * M1 + 0.5 * M2 + 0.1 * X + rng.normal(0, 0.6, n)
    return pd.DataFrame({"Y": Y, "X": X, "M1": M1, "M2": M2})


def test_parallel_recovers_two_mediators(tmp_path: Path) -> None:
    csv = tmp_path / "par.csv"
    _parallel_df().to_csv(csv, index=False)
    res = _run(csv, "parallel_mediation", tmp_path,
               config={"x": "X", "y": "Y", "mediators": ["M1", "M2"]})
    e = res.estimates
    assert e["n_mediators"] == 2.0
    assert e["indirect_M1"] > 0 and e["indirect_M1_lo"] > 0   # 0.5·0.3 ≈ 0.15
    assert e["indirect_M2"] > 0 and e["indirect_M2_lo"] > 0   # 0.4·0.5 ≈ 0.20
    assert e["total_indirect"] > 0 and e["total_indirect_lo"] > 0
    assert (Path(res.output_dir) / "parallel_mediation_contrasts.csv").exists()


def test_parallel_total_equals_sum_of_specifics(tmp_path: Path) -> None:
    csv = tmp_path / "par2.csv"
    _parallel_df(seed=9).to_csv(csv, index=False)
    res = _run(csv, "parallel_mediation", tmp_path,
               config={"x": "X", "y": "Y", "mediators": ["M1", "M2"]})
    e = res.estimates
    assert abs(e["total_indirect"] - (e["indirect_M1"] + e["indirect_M2"])) < 1e-4


# --------------------------------------------------------------------------- #
# moderated_moderation (three-way X×W×Z)
# --------------------------------------------------------------------------- #
def _modmod_df(n=400, three_way=0.5, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    W = rng.normal(0, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.2 * X + 0.1 * W + three_way * X * W * Z + rng.normal(0, 0.6, n)
    return pd.DataFrame({"Y": Y, "X": X, "W": W, "Z": Z})


def test_modmod_detects_three_way(tmp_path: Path) -> None:
    csv = tmp_path / "mm.csv"
    _modmod_df(three_way=0.5).to_csv(csv, index=False)
    res = _run(csv, "moderated_moderation", tmp_path,
               config={"x": "X", "w": "W", "z": "Z", "y": "Y"})
    e = res.estimates
    assert e["p_three_way_XWZ"] < 0.05
    assert e["b_three_way_XWZ"] > 0
    grid = pd.read_csv(Path(res.output_dir) / "conditional_x_effects.csv")
    assert len(grid) == 9        # 3 (W) × 3 (Z)


def test_modmod_null_three_way_not_significant(tmp_path: Path) -> None:
    csv = tmp_path / "mm0.csv"
    _modmod_df(three_way=0.0, seed=5).to_csv(csv, index=False)
    res = _run(csv, "moderated_moderation", tmp_path,
               config={"x": "X", "w": "W", "z": "Z", "y": "Y"})
    assert res.estimates["p_three_way_XWZ"] > 0.05


def test_modmod_degrades_few_columns(tmp_path: Path) -> None:
    csv = tmp_path / "few2.csv"
    pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) * 1.2,
                  "c": np.arange(30.0) * 0.7}).to_csv(csv, index=False)
    res = _run(csv, "moderated_moderation", tmp_path)
    assert "跳过" in res.summary
    assert "b_three_way_XWZ" not in res.estimates
