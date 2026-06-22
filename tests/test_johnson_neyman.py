"""Tests for johnson_neyman — probing a two-way interaction X*W on Y.

Data-generating process plants a TRUE interaction:
  Y = b1*X + b2*W + b3*(X*W) + noise   (b3 != 0)
so the conditional slope of X on Y depends on W and the J-N boundaries exist.
Cross-checks the boundary against a brute-force scan of |theta/SE| = t_crit, plus
honest-degrade and config role override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="johnson_neyman",
    method="Johnson-Neyman probing of a two-way interaction",
    domain="statistics",
    family="conditional_process",
    goal="explain",
    preconditions={"min_continuous": 3, "min_rows": 10},
)


def _make_interaction(n: int = 400, b3: float = 0.7, seed: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    W = rng.normal(0, 1, n)
    b1, b2 = 0.4, 0.2
    Y = b1 * X + b2 * W + b3 * (X * W) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"outcome_y": Y, "pred_x": X, "mod_w": W})


def test_interaction_and_jn_recovered(tmp_path: Path) -> None:
    df = _make_interaction()
    csv = tmp_path / "jn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"x": "pred_x", "w": "mod_w", "y": "outcome_y"},
    )
    est = res.estimates
    for k in ("jn_lower", "jn_upper", "b3_interaction", "p_interaction",
              "slope_lo", "slope_mean", "slope_hi"):
        assert k in est, f"missing estimate {k}"
    # planted b3 = 0.7 recovered and significant
    assert est["b3_interaction"] > 0.4
    assert est["p_interaction"] < 0.01
    # conditional slope of X grows with W (positive interaction)
    assert est["slope_hi"] > est["slope_mean"] > est["slope_lo"]
    # at least one finite J-N boundary exists
    assert np.isfinite(est["jn_lower"]) or np.isfinite(est["jn_upper"])


def test_jn_boundary_matches_bruteforce(tmp_path: Path) -> None:
    # Recompute J-N boundaries by a brute-force scan of |theta(w)/SE(w)| = t_crit
    # and confirm the reported boundaries match.
    df = _make_interaction(seed=21)
    csv = tmp_path / "jn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"x": "pred_x", "w": "mod_w", "y": "outcome_y"},
    )
    import statsmodels.api as sm
    from scipy import stats

    sub = df.dropna()
    Y = sub["outcome_y"].to_numpy(float)
    X = sub["pred_x"].to_numpy(float)
    W = sub["mod_w"].to_numpy(float)
    n = len(sub)
    wm = W.mean()
    xc = X - X.mean()
    wc = W - wm
    D = np.column_stack([np.ones(n), xc, wc, xc * wc])
    fit = sm.OLS(Y, D).fit()
    b1, b3 = fit.params[1], fit.params[3]
    cov = fit.cov_params()
    vb1, vb3, c13 = cov[1, 1], cov[3, 3], cov[1, 3]
    tcrit = stats.t.ppf(0.975, df=n - 4)

    # scan a fine grid in centered-w space; find sign changes of |t|-tcrit
    ws = np.linspace(wc.min() - 2, wc.max() + 2, 20001)
    theta = b1 + b3 * ws
    se = np.sqrt(np.maximum(vb1 + ws ** 2 * vb3 + 2 * ws * c13, 1e-30))
    f = np.abs(theta / se) - tcrit
    crossings = []
    for i in range(len(ws) - 1):
        if f[i] == 0 or f[i] * f[i + 1] < 0:
            # linear interpolation of the zero, then back to original W scale
            w0 = ws[i] - f[i] * (ws[i + 1] - ws[i]) / (f[i + 1] - f[i])
            crossings.append(w0 + wm)
    crossings = sorted(crossings)
    assert len(crossings) >= 1
    bf_lower = crossings[0]
    bf_upper = crossings[1] if len(crossings) >= 2 else float("nan")

    if np.isfinite(res.estimates["jn_lower"]):
        assert abs(res.estimates["jn_lower"] - bf_lower) < 0.05
    if np.isfinite(res.estimates["jn_upper"]) and np.isfinite(bf_upper):
        assert abs(res.estimates["jn_upper"] - bf_upper) < 0.05


def test_products_and_disclosure(tmp_path: Path) -> None:
    df = _make_interaction(seed=9)
    csv = tmp_path / "jn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"x": "pred_x", "w": "mod_w", "y": "outcome_y"},
    )
    out = Path(res.output_dir)
    assert (out / "johnson_neyman_slopes.csv").exists()
    assert (out / "johnson_neyman_plot.png").exists()
    assert "Johnson-Neyman" in res.summary
    assert "连续调节变量" in res.summary


def test_default_role_assignment_discloses(tmp_path: Path) -> None:
    df = _make_interaction(seed=2)
    csv = tmp_path / "jn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "b3_interaction" in res.estimates
    assert "角色按列序自动指派" in res.summary


def test_too_few_continuous_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "label": ["p", "q"] * 25,  # not continuous
    })
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "b3_interaction" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
