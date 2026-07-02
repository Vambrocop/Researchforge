"""Tests for irt_pcm — Partial Credit Model (Masters), girth.pcm_mml MML.

PCM is the Rasch-family polytomous model: ONE shared discrimination, per-item
step difficulties. The load-bearing risks mirror GRM: girth.pcm_mml wants an
items x persons INTEGER matrix coded 0..K-1 (the TRANSPOSE of our frame; a float
matrix raises girth's "arrays used as indices" IndexError), and we must recover
the step difficulties. The happy-path test SIMULATES partial-credit responses
from a KNOWN PCM truth (theta ~ N(0,1), equal a, per-item step difficulties; the
PCM category probability is the softmax of cumulative step sums) and asserts the
recovered step difficulties recover the known structure. Plus binary-only skip
and config override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="irt_pcm",
    method="Partial Credit Model (Masters)",
    domain="psychometrics",
    family="irt",
    goal="describe",
    preconditions={"min_categorical_cols": 3, "min_rows": 20},
)


def _simulate_pcm(
    n_people: int, steps: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Partial-credit responses from a known PCM truth -> respondents x items.

    steps: (n_items, n_cats-1) step difficulties (the "deltas"). PCM category
    probability P(X=k) proportional to exp(sum_{m=1..k}(theta - step_m)),
    with P(X=0) proportional to 1. Equal discrimination is implicit (a=1).
    """
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_people)
    k = steps.shape[0]
    n_thr = steps.shape[1]
    n_cats = n_thr + 1
    data = np.zeros((n_people, k), dtype=int)
    for j in range(k):
        # numerator for category c = exp(sum_{m=1..c}(theta - step_m)); cat 0 -> 0 (exp=1)
        cum_terms = np.zeros((n_people, n_cats))
        for c in range(1, n_cats):
            cum_terms[:, c] = cum_terms[:, c - 1] + (theta - steps[j, c - 1])
        num = np.exp(cum_terms - cum_terms.max(axis=1, keepdims=True))
        cat_p = num / num.sum(axis=1, keepdims=True)
        u = rng.uniform(size=n_people)
        cdf = np.cumsum(cat_p, axis=1)
        data[:, j] = (u[:, None] > cdf).sum(axis=1)
    df = pd.DataFrame(data, columns=[f"q{j + 1}" for j in range(k)])
    return df, theta


def test_pcm_recovers_step_difficulties(tmp_path: Path) -> None:
    # Known PCM truth: 6 items, 4 categories (3 step difficulties each).
    steps = np.array(
        [
            [-1.5, -0.2, 1.2],
            [-1.0, 0.0, 1.0],
            [-2.0, -0.5, 0.5],
            [-1.2, 0.3, 1.4],
            [-0.8, 0.1, 1.1],
            [-1.6, -0.3, 0.8],
        ]
    )
    df, theta = _simulate_pcm(1200, steps, seed=42)
    csv = tmp_path / "pcm.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert sum(1 for c in fp.columns if c.kind == "count") >= 3

    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == float(steps.shape[0])
    assert res.estimates["n_respondents"] == 1200.0
    assert res.estimates["n_categories"] == 4.0
    # girth pcm_mml is GPCM (free per-item discrimination) — reports mean + SD, not "shared"
    assert "discrimination_mean" in res.estimates
    assert "discrimination_sd" in res.estimates

    params = pd.read_csv(Path(res.output_dir) / "irt_pcm_item_params.csv")
    assert "discrimination_a" in params.columns
    assert list(params["item"]) == list(df.columns)
    step_cols = [c for c in params.columns if c.startswith("step_")]
    assert len(step_cols) == 3
    step_hat = params[step_cols].to_numpy(float)

    # ---- recovery proof: per-item mean step difficulty correlates with truth ----
    # (the mean step is the item's overall location, robust to PCM step ordering).
    mean_step_hat = step_hat.mean(axis=1)
    mean_step_true = steps.mean(axis=1)
    r_step = np.corrcoef(mean_step_hat, mean_step_true)[0, 1]
    assert r_step > 0.6, f"step-difficulty location recovery weak (r={r_step:.3f}) — check girth orientation"

    # flattened step vector should also broadly track the truth (orientation guard)
    r_flat = np.corrcoef(step_hat.ravel(), steps.ravel())[0, 1]
    assert r_flat > 0.5, f"step recovery weak (r={r_flat:.3f})"

    # ability EAP scores correlate with the true thetas
    ab = pd.read_csv(Path(res.output_dir) / "irt_pcm_abilities.csv")
    assert len(ab) == 1200
    r_theta = np.corrcoef(ab["theta_eap"].to_numpy(float), theta)[0, 1]
    assert r_theta > 0.7, f"ability recovery weak (r={r_theta:.3f})"
    assert np.corrcoef(ab["raw_score"], ab["theta_eap"])[0, 1] > 0.9

    assert (Path(res.output_dir) / "irt_pcm_steps.png").exists()


def test_gpcm_scorer_matches_model_and_differs_from_grm(tmp_path: Path) -> None:
    # PCM/GPCM θ must be scored with the ADJACENT-CATEGORY (divide-by-total)
    # likelihood, NOT the GRM cumulative one. On ≥3-category data the two scorers
    # give materially different θ; the GPCM scorer (matching _simulate_pcm's
    # data-generating model) recovers the true θ at least as well as the mismatched
    # GRM scorer. This is the regression guard for the model-family fix.
    import importlib.util

    import pytest

    if importlib.util.find_spec("girth") is None:
        pytest.skip("girth not installed")
    import girth

    from researchforge.executor.branches.irt import (
        _gpcm_abilities,
        _poly_abilities,
        _poly_thresholds,
    )

    steps = np.array(
        [
            [-1.5, -0.2, 1.2],
            [-1.0, 0.0, 1.0],
            [-2.0, -0.5, 0.5],
            [-1.2, 0.3, 1.4],
            [-0.8, 0.1, 1.1],
            [-1.6, -0.3, 0.8],
        ]
    )
    df, theta = _simulate_pcm(1500, steps, seed=11)
    X_ip = df.to_numpy(int).T  # items x persons
    disc, thr = _poly_thresholds(girth.pcm_mml(X_ip))
    n_cats = 4

    theta_gpcm = _gpcm_abilities(X_ip, disc, thr, n_cats)
    theta_grm = _poly_abilities(X_ip, disc, thr, n_cats)  # the old (wrong) scorer

    r_gpcm = np.corrcoef(theta_gpcm, theta)[0, 1]
    r_grm = np.corrcoef(theta_grm, theta)[0, 1]
    # matched scorer recovers the true abilities strongly
    assert r_gpcm > 0.85, f"GPCM θ recovery weak (r={r_gpcm:.3f})"
    # the two are genuinely different models — θ estimates diverge on 4-category data
    assert not np.allclose(theta_gpcm, theta_grm, atol=1e-3)
    mean_abs_diff = float(np.abs(theta_gpcm - theta_grm).mean())
    assert mean_abs_diff > 0.01, f"GPCM/GRM scorers indistinguishable (mean|Δ|={mean_abs_diff:.4f})"
    # the correct model should fit the data at least as well as the mismatched one
    assert r_gpcm >= r_grm - 0.02, f"GPCM ({r_gpcm:.3f}) worse than GRM ({r_grm:.3f})"

    # --- where the model-family difference BITES: disordered (non-monotone) steps ---
    # PCM/GPCM steps may be non-monotone; the GRM cumulative scorer assumes ordered
    # boundaries, so it is most wrong here. The GPCM scorer must recover θ with
    # strictly lower error — a strong, falsifiable proof the fix matters (a silent
    # revert to the GRM scorer would fail this).
    steps_dis = np.array(
        [
            [1.2, -0.5, 0.3],
            [0.8, -1.0, 0.5],
            [1.5, -0.8, 0.2],
            [0.9, -1.2, 0.6],
            [1.1, -0.6, 0.4],
            [1.3, -0.9, 0.1],
        ]
    )
    df2, theta2 = _simulate_pcm(1500, steps_dis, seed=23)
    X_ip2 = df2.to_numpy(int).T
    disc2, thr2 = _poly_thresholds(girth.pcm_mml(X_ip2))
    rmse_gpcm = float(np.sqrt(np.mean((_gpcm_abilities(X_ip2, disc2, thr2, 4) - theta2) ** 2)))
    rmse_grm = float(np.sqrt(np.mean((_poly_abilities(X_ip2, disc2, thr2, 4) - theta2) ** 2)))
    assert rmse_gpcm < rmse_grm, f"on disordered steps GPCM RMSE {rmse_gpcm:.3f} not < GRM {rmse_grm:.3f}"


def test_gpcm_abilities_survives_nan_step_without_poisoning_all_thetas() -> None:
    # girth returns a NaN top-step for an item that never reaches its top category.
    # The divide-by-total normalizer must NOT let that one NaN poison every person's
    # θ (the regression the inference reviewer caught: 400/400 NaN). Masked → finite.
    import numpy as np

    from researchforge.executor.branches.irt import _gpcm_abilities

    disc = np.array([1.0, 1.0])
    thr = np.array([[-1.0, 0.0, 1.0], [-1.1, -2.5, np.nan]])  # item 1 tops out early
    # persons only ever use categories the items actually reach (0..2 for item 1)
    X_ip = np.array([[0, 1, 2, 3, 2, 1], [0, 1, 2, 2, 1, 0]])
    theta = _gpcm_abilities(X_ip, disc, thr, n_cats=4)
    assert np.isfinite(theta).all(), f"NaN step poisoned θ: {theta}"


def test_pcm_config_items_override(tmp_path: Path) -> None:
    steps = np.array(
        [[-1.0, 0.0, 1.0], [-1.2, -0.2, 0.9], [-0.8, 0.3, 1.1], [-1.4, -0.3, 0.7]]
    )
    df, _ = _simulate_pcm(400, steps, seed=7)
    rng = np.random.default_rng(99)
    df["noise"] = rng.integers(0, 4, len(df))
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4"]},
    )
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == 4.0
    params = pd.read_csv(Path(res.output_dir) / "irt_pcm_item_params.csv")
    assert "noise" not in list(params["item"])


def test_pcm_binary_only_skips(tmp_path: Path) -> None:
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


def test_pcm_few_respondents_skip(tmp_path: Path) -> None:
    steps = np.array([[-1.0, 0.0, 1.0], [-1.2, -0.2, 0.9], [-0.8, 0.3, 1.1]])
    df, _ = _simulate_pcm(15, steps, seed=2)
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3"]},
    )
    assert "跳过" in res.summary
    assert "n_items" not in res.estimates
