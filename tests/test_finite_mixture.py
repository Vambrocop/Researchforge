"""Tests for finite_mixture — 1-D finite Gaussian mixture (EM), k chosen by BIC.

Known structure:
  * a well-separated bimodal mixture -> k=2 chosen, large delta-BIC vs k=1, two
    components with means near the planted means;
  * a clean unimodal normal (negative control) -> k=1 chosen;
  * estimates contract is satisfied;
  * config column / max_k overrides honoured;
  * too-few-rows honest skip (no crash, no fabricated estimates).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_SK = importlib.util.find_spec("sklearn") is not None

_ENTRY = AnalysisEntry(
    id="finite_mixture",
    method="Finite Gaussian mixture (EM) with BIC model selection",
    domain="statistics",
    family="distribution_extra",
    goal="describe",
    preconditions={"min_numeric_cols": 1, "min_rows": 8},
)


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"), config=config)


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_bimodal_picks_k2(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    # two well-separated Gaussians: means 0 and 12, sd 1 -> clearly multimodal
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(12.0, 1.0, 400)
    df = pd.DataFrame({"x": np.concatenate([a, b])})
    res = _run(df, tmp_path)

    for k in ("best_k", "best_bic", "delta_bic_vs_k1", "largest_weight", "n"):
        assert k in res.estimates
    assert res.estimates["n"] == 800.0
    assert res.estimates["best_k"] == 2.0  # bimodal recovered
    assert res.estimates["delta_bic_vs_k1"] > 10.0  # strong multimodality evidence

    out = Path(res.output_dir)
    comp = pd.read_csv(out / "finite_mixture_components.csv")
    assert len(comp) == 2
    # weights sum to ~1
    assert abs(comp["weight"].sum() - 1.0) < 1e-6
    # the two component means recover the planted means (sorted by mean)
    means = comp.sort_values("mean")["mean"].to_numpy()
    assert abs(means[0] - 0.0) < 1.0
    assert abs(means[1] - 12.0) < 1.0
    bic = pd.read_csv(out / "finite_mixture_bic.csv")
    assert set(bic["k"]) >= {1, 2}
    assert "多峰" in res.summary


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_unimodal_normal_picks_k1(tmp_path: Path) -> None:
    # negative control: a clean single normal -> k=1, no multimodality
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(50.0, 5.0, 600)})
    res = _run(df, tmp_path)
    assert res.estimates["best_k"] == 1.0
    # delta-BIC vs k=1 is 0 by construction when k=1 is chosen
    assert res.estimates["delta_bic_vs_k1"] <= 1e-6
    assert "单峰" in res.summary


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_config_max_k_caps_search(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 300)
    b = rng.normal(10.0, 1.0, 300)
    df = pd.DataFrame({"x": np.concatenate([a, b])})
    res = _run(df, tmp_path, config={"max_k": 1})
    # capped at 1 -> cannot detect the second mode
    assert res.estimates["best_k"] == 1.0
    out = Path(res.output_dir)
    bic = pd.read_csv(out / "finite_mixture_bic.csv")
    assert bic["k"].max() == 1


@pytest.mark.skipif(not _HAS_SK, reason="scikit-learn not available")
def test_config_column_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "junk": rng.normal(0, 1, 400),
        "target": np.concatenate([rng.normal(0, 1, 200), rng.normal(15, 1, 200)]),
    })
    res = _run(df, tmp_path, config={"column": "target"})
    assert "target" in res.summary
    assert res.estimates["best_k"] == 2.0


def test_too_few_rows_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})  # n=5 < 8
    res = _run(df, tmp_path)
    assert "best_k" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_constant_column_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [7.0] * 50})
    res = _run(df, tmp_path)
    assert "best_k" not in res.estimates
    assert "跳过" in res.summary
