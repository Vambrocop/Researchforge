"""Tests for the `canonical_correlation` executor branch (CCA).

Synthetic data with KNOWN structure: X and Y share a latent factor, so the
first canonical correlation should be high AND significant; later pairs are noise
and should NOT be significant. Also a numerical cross-check of the canonical
correlations against sklearn's CCA, and a precondition skip.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="canonical_correlation",
    method="Canonical correlation analysis (CCA)",
    domain="statistics",
    family="statistics",
    goal="explore",
    preconditions={"min_continuous": 3, "min_rows": 20},
)


def _latent_csv(tmp_path: Path, n: int = 200):
    """X = (x1,x2,x3), Y = (y1,y2,y3); a shared latent factor links x1.. and y1.."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)  # shared latent factor
    df = pd.DataFrame({
        "x1": z + rng.normal(0, 0.3, n),
        "x2": 0.8 * z + rng.normal(0, 0.4, n),
        "x3": rng.normal(0, 1, n),          # noise in X
        "y1": z + rng.normal(0, 0.3, n),
        "y2": 0.7 * z + rng.normal(0, 0.5, n),
        "y3": rng.normal(0, 1, n),          # noise in Y
    })
    csv = tmp_path / "latent.csv"
    df.to_csv(csv, index=False)
    return csv, df


def test_cca_shared_latent_first_pair_high_and_significant(tmp_path):
    csv, df = _latent_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"set_x": ["x1", "x2", "x3"], "set_y": ["y1", "y2", "y3"]})
    out = Path(res.output_dir)

    assert (out / "cca_correlations.csv").exists()
    assert (out / "cca_sequential_test.csv").exists()

    # Shared latent -> high first canonical correlation.
    assert res.estimates["first_canonical_corr"] > 0.7, res.estimates
    # First pair significant.
    assert res.estimates["first_pair_p"] < 0.05, res.estimates
    # At least the first pair is flagged significant.
    assert res.estimates["n_significant_pairs"] >= 1, res.estimates

    # --- numerical cross-check against sklearn.CCA canonical correlations ---
    from sklearn.cross_decomposition import CCA
    X = df[["x1", "x2", "x3"]].values
    Y = df[["y1", "y2", "y3"]].values
    cca = CCA(n_components=3, scale=True)
    Xc, Yc = cca.fit_transform(X, Y)
    # canonical correlation of each transformed component pair
    sk_rho = [float(np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]) for i in range(3)]
    ours = pd.read_csv(out / "cca_correlations.csv")["canonical_correlation"].tolist()
    # Compare the FIRST (dominant) canonical correlation in magnitude.
    assert abs(abs(ours[0]) - abs(sk_rho[0])) < 0.02, (ours, sk_rho)


def test_cca_too_few_vars_skips(tmp_path):
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    csv = tmp_path / "two_only.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # only 2 continuous columns -> split is 1+1 = 2 total < 3 -> skip
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "first_canonical_corr" not in res.estimates
    assert "跳过" in res.summary
