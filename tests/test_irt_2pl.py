"""Tests for irt_2pl — 2-parameter logistic Item Response Theory (girth MML).

The load-bearing risk is girth's data orientation: it wants an items x persons
binary matrix, the TRANSPOSE of our respondents x items frame. To prove the branch
transposes correctly AND recovers the right parameters, every "happy path" test
SIMULATES responses from a KNOWN 2PL truth (theta ~ N(0,1), known item a/b,
P=1/(1+exp(-a(theta-b))), Bernoulli draws) and asserts the recovered a/b correlate
strongly with the truth and difficulties land in the right rank order. If the
transpose were backwards the recovered parameters would be garbage and these
correlations would collapse. Plus config override + honest-skip degrade paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="irt_2pl",
    method="2PL IRT (item discrimination + difficulty)",
    domain="psychometrics",
    family="irt",
    goal="describe",
    preconditions={"min_categorical_cols": 3, "min_rows": 20},
)


def _simulate_2pl(
    n_people: int, a_true: np.ndarray, b_true: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bernoulli responses from a known 2PL truth -> respondents x items frame."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_people)
    k = len(a_true)
    # P is persons x items here (our engine orientation)
    P = 1.0 / (1.0 + np.exp(-a_true[None, :] * (theta[:, None] - b_true[None, :])))
    data = (rng.uniform(size=P.shape) < P).astype(int)
    df = pd.DataFrame(data, columns=[f"q{j + 1}" for j in range(k)])
    return df, theta


def test_2pl_recovers_known_parameters(tmp_path: Path) -> None:
    # Known 2PL truth with spread-out discriminations and difficulties.
    a_true = np.array([0.8, 1.0, 1.3, 1.6, 1.9, 2.2, 0.9, 1.4, 1.7, 2.0])
    b_true = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -0.2])
    # n=1000: 2PL DISCRIMINATION recovery is genuinely noisy (r_a≈0.71 at n=600, ≈0.93 at
    # n=1000); difficulty recovers fine at either. Larger n proves orientation cleanly.
    df, theta = _simulate_2pl(1000, a_true, b_true, seed=42)
    csv = tmp_path / "irt.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # the simulated 0/1 columns must profile as binary (so the branch picks them up)
    assert sum(1 for c in fp.columns if c.kind == "binary") >= 3

    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == float(len(a_true))
    assert res.estimates["n_respondents"] == 1000.0

    params = pd.read_csv(Path(res.output_dir) / "irt_2pl_item_params.csv")
    assert list(params["item"]) == list(df.columns)
    a_hat = params["discrimination_a"].to_numpy(float)
    b_hat = params["difficulty_b"].to_numpy(float)

    # ---- the orientation + estimation proof ----
    r_a = np.corrcoef(a_hat, a_true)[0, 1]
    r_b = np.corrcoef(b_hat, b_true)[0, 1]
    assert r_a > 0.8, f"discrimination recovery weak (r={r_a:.3f}) — check girth orientation"
    assert r_b > 0.8, f"difficulty recovery weak (r={r_b:.3f}) — check girth orientation"
    # difficulties recovered in (near) the right order
    order_true = np.argsort(b_true)
    order_hat = np.argsort(b_hat)
    rank_corr = np.corrcoef(order_true.argsort(), order_hat.argsort())[0, 1]
    assert rank_corr > 0.8, f"difficulty rank order off (rho={rank_corr:.3f})"

    # ability EAP scores correlate with the true thetas
    ab = pd.read_csv(Path(res.output_dir) / "irt_2pl_abilities.csv")
    assert len(ab) == 1000
    r_theta = np.corrcoef(ab["theta_eap"].to_numpy(float), theta)[0, 1]
    assert r_theta > 0.7, f"ability recovery weak (r={r_theta:.3f})"
    # higher raw score should mean higher ability
    assert np.corrcoef(ab["raw_score"], ab["theta_eap"])[0, 1] > 0.9

    # products + estimates exist
    assert (Path(res.output_dir) / "irt_2pl_iccs.png").exists()
    assert "loglik_2pl" in res.estimates
    assert "mean_discrimination" in res.estimates


def test_2pl_config_items_override(tmp_path: Path) -> None:
    # Extra noise binary column present; config restricts to the 4 real items.
    a_true = np.array([1.2, 1.5, 1.0, 1.8])
    b_true = np.array([-1.0, -0.3, 0.4, 1.1])
    df, _ = _simulate_2pl(300, a_true, b_true, seed=7)
    rng = np.random.default_rng(99)
    df["noise"] = rng.integers(0, 2, len(df))  # unrelated binary column
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4"]},
    )
    assert res.estimates["n_items"] == 4.0
    params = pd.read_csv(Path(res.output_dir) / "irt_2pl_item_params.csv")
    assert "noise" not in list(params["item"])


def test_2pl_model_config_rejects_non_2pl(tmp_path: Path) -> None:
    a_true = np.array([1.0, 1.2, 1.4, 1.6])
    b_true = np.array([-0.8, -0.2, 0.3, 0.9])
    df, _ = _simulate_2pl(120, a_true, b_true, seed=3)
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"), config={"model": "3pl"}
    )
    assert "跳过" in res.summary
    assert "irt_2pl_item_params.csv" not in res.files


def test_2pl_too_few_items_skips(tmp_path: Path) -> None:
    a_true = np.array([1.0, 1.5])
    b_true = np.array([-0.5, 0.5])
    df, _ = _simulate_2pl(150, a_true, b_true, seed=1)
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
    assert (Path(res.output_dir) / "report.md").exists()


def test_2pl_non_binary_items_skip(tmp_path: Path) -> None:
    # Polytomous (0/1/2) data is NOT dichotomous -> honest skip, no crash.
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {f"q{j}": rng.integers(0, 3, 80) for j in range(1, 5)}  # values 0,1,2
    )
    csv = tmp_path / "poly.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # force the columns as items (they profile as count, not binary)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4"]},
    )
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates


def test_2pl_few_respondents_skip(tmp_path: Path) -> None:
    # <20 respondents -> MML not trustworthy -> honest skip.
    a_true = np.array([1.0, 1.3, 1.5, 1.7])
    b_true = np.array([-0.6, -0.2, 0.2, 0.6])
    df, _ = _simulate_2pl(15, a_true, b_true, seed=2)
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
