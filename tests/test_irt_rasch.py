"""Tests for irt_rasch — Rasch / 1PL Item Response Theory (girth MML).

Like the 2PL test, the load-bearing risk is girth's items x persons orientation
(the TRANSPOSE of our respondents x items frame). Happy-path tests SIMULATE from a
known Rasch truth (theta ~ N(0,1), EQUAL discrimination a, per-item difficulty b,
P=1/(1+exp(-a(theta-b))), Bernoulli draws) and assert recovered difficulties
correlate strongly with the truth and land in the right rank order — which only
holds if the matrix is transposed correctly. Also checks the parallel-ICC product,
person-separation reliability, the Rasch-vs-2PL log-likelihood comparison, config
override and honest-skip degrade paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="irt_rasch",
    method="Rasch / 1PL IRT (equal-discrimination)",
    domain="psychometrics",
    family="irt",
    goal="describe",
    preconditions={"min_categorical_cols": 3, "min_rows": 20},
)


def _simulate_rasch(
    n_people: int, b_true: np.ndarray, a_shared: float, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bernoulli responses from a known Rasch truth (equal discrimination)."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_people)
    k = len(b_true)
    P = 1.0 / (1.0 + np.exp(-a_shared * (theta[:, None] - b_true[None, :])))
    data = (rng.uniform(size=P.shape) < P).astype(int)
    df = pd.DataFrame(data, columns=[f"q{j + 1}" for j in range(k)])
    return df, theta


def test_rasch_recovers_known_difficulties(tmp_path: Path) -> None:
    b_true = np.array([-2.0, -1.4, -0.8, -0.3, 0.2, 0.7, 1.2, 1.8, -0.5, 0.9])
    df, theta = _simulate_rasch(600, b_true, a_shared=1.0, seed=42)
    csv = tmp_path / "rasch.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "binary") >= 3

    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == float(len(b_true))
    assert res.estimates["n_respondents"] == 600.0

    params = pd.read_csv(Path(res.output_dir) / "irt_rasch_item_params.csv")
    assert list(params["item"]) == list(df.columns)
    b_hat = params["difficulty_b"].to_numpy(float)

    # ---- orientation + estimation proof ----
    r_b = np.corrcoef(b_hat, b_true)[0, 1]
    assert r_b > 0.8, f"difficulty recovery weak (r={r_b:.3f}) — check girth orientation"
    order_true = np.argsort(b_true)
    order_hat = np.argsort(b_hat)
    rank_corr = np.corrcoef(order_true.argsort(), order_hat.argsort())[0, 1]
    assert rank_corr > 0.8, f"difficulty rank order off (rho={rank_corr:.3f})"

    # all items share one discrimination (the Rasch signature)
    a_col = params["discrimination_a_shared"].to_numpy(float)
    assert np.allclose(a_col, a_col[0]), "Rasch must use a single shared discrimination"
    assert "shared_discrimination" in res.estimates

    # abilities recover the latent trait and track raw scores
    ab = pd.read_csv(Path(res.output_dir) / "irt_rasch_abilities.csv")
    assert len(ab) == 600
    assert "theta_se" in ab.columns
    r_theta = np.corrcoef(ab["theta_eap"].to_numpy(float), theta)[0, 1]
    assert r_theta > 0.7, f"ability recovery weak (r={r_theta:.3f})"

    # products + estimates
    assert (Path(res.output_dir) / "irt_rasch_iccs.png").exists()
    assert "loglik_rasch" in res.estimates
    assert "loglik_2pl" in res.estimates  # same-data 2PL comparison was run
    assert "person_separation_reliability" in res.estimates


def test_rasch_2pl_loglik_comparison_present(tmp_path: Path) -> None:
    # 2PL nests Rasch -> its log-likelihood should be >= Rasch's on the same data.
    b_true = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, -0.8, 0.8])
    df, _ = _simulate_rasch(400, b_true, a_shared=1.0, seed=11)
    csv = tmp_path / "cmp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    ll_rasch = res.estimates["loglik_rasch"]
    ll_2pl = res.estimates["loglik_2pl"]
    # The same-data 2PL comparison actually ran (not the -1.0 failure sentinel)
    # and both log-likelihoods are finite, negative numbers (Bernoulli log-prob).
    # NOTE: these are plug-in conditional log-likelihoods (EAP-theta plugged in),
    # not true marginal likelihoods, so we do NOT assert the strict nesting
    # inequality ll_2pl >= ll_rasch — only that both are reported and sensible.
    assert ll_2pl != -1.0
    assert np.isfinite(ll_rasch) and np.isfinite(ll_2pl)
    assert ll_rasch < 0 and ll_2pl < 0
    # data was generated with EQUAL discrimination, so Rasch fits about as well as
    # 2PL — the gap should not be wildly large relative to the number of responses.
    n_resp = res.estimates["n_respondents"]
    assert abs(ll_2pl - ll_rasch) < n_resp, "Rasch-truth data: 2PL should not dominate Rasch"


def test_rasch_config_items_override(tmp_path: Path) -> None:
    b_true = np.array([-1.0, -0.2, 0.4, 1.0])
    df, _ = _simulate_rasch(300, b_true, a_shared=1.0, seed=7)
    rng = np.random.default_rng(50)
    df["noise"] = rng.integers(0, 2, len(df))
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4"]},
    )
    assert res.estimates["n_items"] == 4.0
    params = pd.read_csv(Path(res.output_dir) / "irt_rasch_item_params.csv")
    assert "noise" not in list(params["item"])


def test_rasch_too_few_items_skips(tmp_path: Path) -> None:
    b_true = np.array([-0.5, 0.5])
    df, _ = _simulate_rasch(150, b_true, a_shared=1.0, seed=1)
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
    assert (Path(res.output_dir) / "report.md").exists()


def test_rasch_non_binary_items_skip(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({f"q{j}": rng.integers(0, 4, 80) for j in range(1, 5)})
    csv = tmp_path / "poly.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4"]},
    )
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
