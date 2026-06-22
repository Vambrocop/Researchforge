"""Tests for brant_test: the proportional-odds (parallel-lines) assumption check.

Two DGPs:
  * proportional   — slopes equal across cut points -> Brant does NOT reject.
  * non-proportional — one predictor's effect differs by cut -> Brant DOES reject.
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
        id="brant_test",
        method="Brant test of the proportional-odds assumption",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(requires_ordinal=True, min_rows=30),
    )


def _proportional_data(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Single common slope acts on every cut -> proportional odds holds."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 1.0 * x1 - 0.7 * x2 + rng.logistic(0, 1, n)
    sat = np.digitize(latent, [-1.2, 0.0, 1.2]) + 1  # 4 ordered levels
    return pd.DataFrame({"sat": sat.astype(int), "x1": x1, "x2": x2})


def _nonproportional_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Build levels directly from cut-specific binary logits where x1's effect
    differs across the three cut points -> proportional odds VIOLATED for x1.

    For each cut k, P(y > level_k) = sigmoid(a_k + b_k*x1 + c*x2). b_k varies
    strongly with k; the cumulative cut probabilities are forced monotone (a_k
    decreasing) so the levels remain ordered.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    # intercepts decreasing so cumulative P(y>k) decreases with k (ordered)
    a = [1.6, 0.0, -1.6]
    # x1 slope CHANGES across cuts (this is the non-proportionality)
    b = [2.0, 0.2, -1.6]
    c = -0.5  # x2 proportional (same across cuts)

    # cumulative probabilities P(y > level_k), k = 0,1,2
    p_gt = [sig(a[k] + b[k] * x1 + c * x2) for k in range(3)]
    # enforce monotone non-increasing across cuts per row (ordered cumulative)
    p_gt = np.vstack(p_gt)
    p_gt = np.minimum.accumulate(p_gt, axis=0)

    u = rng.uniform(0, 1, n)
    # level = 1 + number of cuts exceeded
    level = 1 + (u < p_gt[0]).astype(int) + (u < p_gt[1]).astype(int) + (u < p_gt[2]).astype(int)
    return pd.DataFrame({"sat": level.astype(int), "x1": x1, "x2": x2})


def test_brant_does_not_reject_on_proportional(tmp_path: Path) -> None:
    df = _proportional_data()
    csv = tmp_path / "prop.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "brant_slopes.csv").exists()
    for k in ("global_chi2", "global_p", "global_df", "n_violations", "n_predictors"):
        assert k in res.estimates
    # proportional DGP: should NOT reject (p >= 0.05)
    assert res.estimates["global_p"] >= 0.05
    assert res.estimates["n_violations"] == 0.0


def test_brant_rejects_on_nonproportional(tmp_path: Path) -> None:
    df = _nonproportional_data()
    csv = tmp_path / "nonprop.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    # x1's effect differs by cut -> reject proportional odds (small global p)
    assert res.estimates["global_p"] < 0.05
    assert res.estimates["n_violations"] >= 1.0
    assert "拒绝" in res.summary  # RunResult.summary is the joined string


def test_brant_degrades_when_no_ordinal(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary  # RunResult.summary is the joined string
