"""Tests for irt_grm — Graded Response Model (Samejima), girth.grm_mml MML.

The load-bearing risks are (1) girth's polytomous orientation — grm_mml wants an
items x persons INTEGER matrix coded 0..K-1 (the TRANSPOSE of our respondents x
items frame; a float matrix raises girth's "arrays used as indices" IndexError),
and (2) recovering ORDERED category thresholds. Each happy-path test SIMULATES
ordinal responses from a KNOWN GRM truth (theta ~ N(0,1), known per-item
discrimination a, known ORDERED thresholds; category drawn from the cumulative-
logit boundaries) and asserts the recovered discrimination correlates with the
truth and the recovered thresholds come out ordered. If the transpose were
backwards the parameters would be garbage and these checks would collapse. Plus a
binary-only honest-skip and a config override path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="irt_grm",
    method="Graded Response Model (Samejima)",
    domain="psychometrics",
    family="irt",
    goal="describe",
    preconditions={"min_categorical_cols": 3, "min_rows": 20},
)


def _simulate_grm(
    n_people: int, a_true: np.ndarray, thresholds: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ordinal responses from a known GRM truth -> respondents x items frame.

    thresholds: (n_items, n_cats-1) ORDERED boundaries. Category prob from the
    cumulative-logit P(X>=k)=1/(1+exp(-a(theta-thr_k))); a multinomial draw picks
    the category 0..K-1 per (person,item).
    """
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_people)
    k = len(a_true)
    n_thr = thresholds.shape[1]
    n_cats = n_thr + 1
    data = np.zeros((n_people, k), dtype=int)
    for j in range(k):
        a = a_true[j]
        b = thresholds[j]  # ordered thresholds
        # cumulative P(X>=k) for k=1..n_cats-1 ; person x thr
        cum = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b[None, :])))
        upper = np.hstack([np.ones((n_people, 1)), cum])      # P(X>=0..n_cats-1)
        lower = np.hstack([cum, np.zeros((n_people, 1))])     # P(X>=1..n_cats)
        cat_p = np.clip(upper - lower, 1e-9, 1.0)             # person x n_cats
        cat_p /= cat_p.sum(axis=1, keepdims=True)
        u = rng.uniform(size=n_people)
        cdf = np.cumsum(cat_p, axis=1)
        data[:, j] = (u[:, None] > cdf).sum(axis=1)
    df = pd.DataFrame(data, columns=[f"q{j + 1}" for j in range(k)])
    return df, theta


def test_grm_recovers_known_parameters(tmp_path: Path) -> None:
    # Known GRM truth: 6 items, 4 ordered categories (3 thresholds each).
    a_true = np.array([1.0, 1.3, 1.6, 0.9, 1.8, 1.2])
    thresholds = np.array(
        [
            [-1.5, -0.3, 1.0],
            [-1.8, -0.5, 0.8],
            [-1.2, 0.0, 1.2],
            [-2.0, -0.7, 0.6],
            [-1.0, 0.2, 1.5],
            [-1.6, -0.2, 0.9],
        ]
    )
    df, theta = _simulate_grm(1200, a_true, thresholds, seed=42)
    csv = tmp_path / "grm.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # 0..3 ordinal columns should profile as count (non-neg ints, non-unique)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 3

    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == float(len(a_true))
    assert res.estimates["n_respondents"] == 1200.0
    assert res.estimates["n_categories"] == 4.0

    params = pd.read_csv(Path(res.output_dir) / "irt_grm_item_params.csv")
    assert list(params["item"]) == list(df.columns)
    a_hat = params["discrimination_a"].to_numpy(float)

    # ---- orientation + estimation proof: discrimination recovery ----
    r_a = np.corrcoef(a_hat, a_true)[0, 1]
    assert r_a > 0.7, f"discrimination recovery weak (r={r_a:.3f}) — check girth orientation"

    # recovered thresholds must be ORDERED within every item
    thr_cols = [c for c in params.columns if c.startswith("threshold_")]
    assert len(thr_cols) == 3
    thr_hat = params[thr_cols].to_numpy(float)
    for j in range(len(a_true)):
        assert np.all(np.diff(thr_hat[j]) >= -1e-6), f"item {j} thresholds not ordered: {thr_hat[j]}"
    assert res.estimates["n_unordered_threshold_items"] == 0.0

    # thresholds recovered roughly in the right place (mean threshold correlates)
    mean_thr_hat = thr_hat.mean(axis=1)
    mean_thr_true = thresholds.mean(axis=1)
    r_thr = np.corrcoef(mean_thr_hat, mean_thr_true)[0, 1]
    assert r_thr > 0.5, f"threshold location recovery weak (r={r_thr:.3f})"

    # ability EAP scores correlate with the true thetas
    ab = pd.read_csv(Path(res.output_dir) / "irt_grm_abilities.csv")
    assert len(ab) == 1200
    r_theta = np.corrcoef(ab["theta_eap"].to_numpy(float), theta)[0, 1]
    assert r_theta > 0.7, f"ability recovery weak (r={r_theta:.3f})"
    assert np.corrcoef(ab["raw_score"], ab["theta_eap"])[0, 1] > 0.9

    assert (Path(res.output_dir) / "irt_grm_thresholds.png").exists()
    assert "mean_discrimination" in res.estimates


def test_grm_config_items_override(tmp_path: Path) -> None:
    a_true = np.array([1.2, 1.5, 1.0, 1.6])
    thresholds = np.array(
        [[-1.0, 0.0, 1.0], [-1.2, -0.2, 0.9], [-0.8, 0.3, 1.1], [-1.4, -0.3, 0.7]]
    )
    df, _ = _simulate_grm(400, a_true, thresholds, seed=7)
    rng = np.random.default_rng(99)
    df["noise"] = rng.integers(0, 4, len(df))  # unrelated ordinal column
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4"]},
    )
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == 4.0
    params = pd.read_csv(Path(res.output_dir) / "irt_grm_item_params.csv")
    assert "noise" not in list(params["item"])


def test_grm_binary_only_skips(tmp_path: Path) -> None:
    # Binary 0/1 items have only 2 categories -> GRM needs >=3 -> honest skip.
    rng = np.random.default_rng(5)
    df = pd.DataFrame({f"q{j}": rng.integers(0, 2, 120) for j in range(1, 6)})
    csv = tmp_path / "bin.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4", "q5"]},
    )
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
    assert (Path(res.output_dir) / "report.md").exists()


def test_grm_few_respondents_skip(tmp_path: Path) -> None:
    a_true = np.array([1.0, 1.3, 1.5])
    thresholds = np.array([[-1.0, 0.0, 1.0], [-1.2, -0.2, 0.9], [-0.8, 0.3, 1.1]])
    df, _ = _simulate_grm(15, a_true, thresholds, seed=2)
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3"]},
    )
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
