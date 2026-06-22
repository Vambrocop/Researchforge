"""Tests for tail_index — heavy-tail estimation via the Hill estimator.

Known structure:
  * a Pareto sample with shape alpha=2 -> Hill recovers alpha ~ 2 (heavy tail flag set);
    Hill's consistency means the stable-region estimate lands near the true shape;
  * a Pareto sample with a large shape (light-ish tail, alpha=6) -> alpha recovered
    high enough that the heavy-tail flag is NOT set;
  * estimates contract is satisfied;
  * a small sample (n<50) honest skip;
  * non-positive data with too few positive values -> honest skip (no silent shift);
  * config tail=lower estimates the lower tail of a negated Pareto sample;
  * an independent Hill recompute matches the engine's stable-region alpha.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="tail_index",
    method="Heavy-tail / extreme-value tail estimation via the Hill estimator",
    domain="statistics",
    family="distribution_extra",
    goal="describe",
    preconditions={"min_numeric_cols": 1, "min_rows": 50},
)


def _run(df: pd.DataFrame, tmp_path: Path, config=None):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"), config=config)


def _pareto(alpha: float, n: int, seed: int) -> np.ndarray:
    """Pareto(shape=alpha) on [1, inf): X = U**(-1/alpha), U~Uniform(0,1).
    The Hill estimator of the upper tail recovers alpha."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n)
    return u ** (-1.0 / alpha)


def test_pareto_recovers_alpha2(tmp_path: Path) -> None:
    x = _pareto(alpha=2.0, n=5000, seed=0)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path)

    for k in ("alpha_hat", "xi_tail_index", "k_used", "heavy_tail", "n"):
        assert k in res.estimates
    assert res.estimates["n"] == 5000.0
    # Hill recovers the Pareto shape within tolerance (stable-region heuristic)
    assert abs(res.estimates["alpha_hat"] - 2.0) < 0.5
    # xi = 1/alpha
    assert abs(res.estimates["xi_tail_index"] - 1.0 / res.estimates["alpha_hat"]) < 1e-3
    assert res.estimates["heavy_tail"] == 1.0  # alpha=2 < 4 -> heavy

    out = Path(res.output_dir)
    hill = pd.read_csv(out / "tail_index_hill.csv")
    assert set(hill.columns) == {"k", "alpha_hat", "xi_tail_index"}
    assert hill["k"].min() == 1
    assert "重尾" in res.summary


def test_pareto_light_alpha6_not_heavy(tmp_path: Path) -> None:
    # a large shape -> a much lighter tail; the heavy-tail flag (alpha<4) is not set
    x = _pareto(alpha=6.0, n=5000, seed=1)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path)
    assert res.estimates["alpha_hat"] > 4.0
    assert res.estimates["heavy_tail"] == 0.0


def test_independent_hill_recompute(tmp_path: Path) -> None:
    # recompute the stable-region alpha independently and compare to the engine
    x = _pareto(alpha=3.0, n=4000, seed=2)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path)

    y = np.sort(x[x > 0])[::-1]
    n = y.size
    k_max = max(5, min(int(0.10 * n), n - 2))
    logx = np.log(y)
    cum = np.cumsum(logx)
    ks = np.arange(1, k_max + 1)
    gamma = cum[:k_max] / ks - logx[ks]
    alpha = np.where(gamma > 0, 1.0 / gamma, np.nan)
    lo_k = max(1, int(0.2 * k_max))
    hi_k = max(lo_k + 1, int(0.6 * k_max))
    band = alpha[lo_k - 1:hi_k]
    band = band[np.isfinite(band)]
    alpha_ref = float(np.median(band))
    assert abs(res.estimates["alpha_hat"] - alpha_ref) < 1e-6


def test_lower_tail_via_config(tmp_path: Path) -> None:
    # negate a Pareto sample so the heavy tail is on the LOWER side; tail=lower
    # should sign-flip and recover the same alpha
    x = -_pareto(alpha=2.5, n=4000, seed=3)
    df = pd.DataFrame({"x": x})
    res = _run(df, tmp_path, config={"tail": "lower"})
    assert "下尾" in res.summary
    assert abs(res.estimates["alpha_hat"] - 2.5) < 0.6


def test_small_sample_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.uniform(1, 5, 40)})  # n=40 < 50
    res = _run(df, tmp_path)
    assert "alpha_hat" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_nonpositive_upper_tail_skips(tmp_path: Path) -> None:
    # an all-negative column has no positive upper tail -> honest skip (no silent shift)
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": -rng.uniform(1, 5, 200)})
    res = _run(df, tmp_path)
    assert "alpha_hat" not in res.estimates
    assert "跳过" in res.summary
