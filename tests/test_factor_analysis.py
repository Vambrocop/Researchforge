"""Tests for factor_analysis — exploratory factor analysis (pure Python, sklearn).

Known structure: indicators generated from 2 latent factors -> the selection rule
should recover ~2 factors and the loadings should reflect the block structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry():
    return Catalog.load().by_id("factor_analysis")


def _two_factor_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """6 continuous indicators driven by 2 orthogonal latent factors (3 each)."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    cols = {}
    for i in range(3):  # factor-1 block
        cols[f"a{i}"] = 0.85 * f1 + rng.normal(0, 0.5, n)
    for i in range(3):  # factor-2 block
        cols[f"b{i}"] = 0.85 * f2 + rng.normal(0, 0.5, n)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------

def test_catalog_loads_factor_analysis():
    entry = _entry()
    assert entry is not None
    assert entry.executor_ref == "py::factor_analysis"
    assert isinstance(entry.biases, list) and entry.biases
    assert isinstance(entry.produces, list) and entry.produces


# ---------------------------------------------------------------------------
# 2. Executor — recovers ~2 factors from a 2-factor design
# ---------------------------------------------------------------------------

def test_executor_recovers_two_factors(tmp_path):
    csv = tmp_path / "ef.csv"
    _two_factor_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "factor_loadings.csv").exists()
    assert (out / "factor_variance.csv").exists()
    assert "n_factors" in res.estimates
    # the rule should land on ~2 factors (parallel analysis / Kaiser on a 2-factor design)
    assert res.estimates["n_factors"] in (2.0, 3.0), f"got {res.estimates['n_factors']}"
    assert 0.0 <= res.estimates["total_var_explained"] <= 1.0
    assert res.estimates["max_communality"] >= res.estimates["min_communality"]

    load = pd.read_csv(out / "factor_loadings.csv", index_col=0)
    assert "communality" in load.columns
    assert len(load) == 6


# ---------------------------------------------------------------------------
# 3. Config override — force n_factors
# ---------------------------------------------------------------------------

def test_config_force_n_factors(tmp_path):
    csv = tmp_path / "ef.csv"
    _two_factor_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"), config={"n_factors": 1})
    assert res.estimates["n_factors"] == 1.0


# ---------------------------------------------------------------------------
# 4. Honest degrade — too few indicators / rows
# ---------------------------------------------------------------------------

def test_degrade_too_few_indicators(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 50), "y": rng.normal(0, 1, 50)})
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_factors" not in res.estimates


def test_degrade_too_few_rows(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({c: rng.normal(0, 1, 8) for c in ("a", "b", "c", "d")})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
