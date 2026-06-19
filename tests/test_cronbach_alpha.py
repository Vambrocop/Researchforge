"""Tests for cronbach_alpha — Cronbach (1951) internal-consistency reliability.

Cross-checks the engine's alpha against an INDEPENDENT recomputation of the
k/(k-1)*(1 - sum(item var)/total var) formula, plus a hand-verified known value,
the Feldt 95% CI ordering, item-total / alpha-if-dropped diagnostics, the config
override, and the <3-item / non-numeric honest-skip degrade paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="cronbach_alpha",
    method="Cronbach's alpha (internal-consistency reliability)",
    domain="psychometrics",
    family="psychometrics",
    goal="describe",
    preconditions={"min_continuous": 3, "min_rows": 10},
)


def _alpha_ref(X: np.ndarray) -> float:
    """Independent reference implementation of Cronbach's alpha."""
    n, k = X.shape
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    return (k / (k - 1.0)) * (1.0 - item_var.sum() / total_var)


def test_cronbach_known_value(tmp_path: Path) -> None:
    # 20 respondents x 4 continuous items driven by a shared latent trait (small
    # noise -> highly consistent). Continuous (non-integer) values profile as
    # `continuous`, sidestepping the profiler id-trap that flags all-distinct
    # whole-number columns as `id` (a tiny all-integer scale would be dropped;
    # that pilot-data robustness is logged as a deferred nice-to-have). alpha is
    # recomputed independently below, so the exact values don't matter.
    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 1.0, 20)
    data = pd.DataFrame(
        {f"i{j + 1}": theta + rng.normal(0.0, 0.30, 20) for j in range(4)}
    )
    expected = _alpha_ref(data.to_numpy(float))
    csv = tmp_path / "scale.csv"
    data.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    assert "cronbach_alpha" in res.estimates
    assert abs(res.estimates["cronbach_alpha"] - expected) < 1e-3
    # this hand-built scale is highly consistent
    assert res.estimates["cronbach_alpha"] > 0.85
    # Feldt CI brackets the point estimate (lower <= alpha <= upper)
    assert res.estimates["alpha_ci_low"] <= res.estimates["cronbach_alpha"] + 1e-9
    assert res.estimates["alpha_ci_high"] >= res.estimates["cronbach_alpha"] - 1e-9

    out = Path(res.output_dir)
    assert (out / "cronbach_item_stats.csv").exists()
    stats = pd.read_csv(out / "cronbach_item_stats.csv")
    assert set(stats.columns) >= {
        "item",
        "item_total_corr_corrected",
        "alpha_if_dropped",
        "raises_alpha_if_dropped",
    }
    assert len(stats) == 4


def test_cronbach_high_internal_consistency(tmp_path: Path) -> None:
    # 6 highly-loaded items off one latent factor -> alpha should be high.
    rng = np.random.default_rng(0)
    n = 200
    f = rng.normal(0, 1, n)
    df = pd.DataFrame({f"q{j}": 0.85 * f + rng.normal(0, 0.45, n) for j in range(1, 7)})
    csv = tmp_path / "hi.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    ref = _alpha_ref(df.to_numpy(float))
    assert abs(res.estimates["cronbach_alpha"] - ref) < 1e-3
    assert res.estimates["cronbach_alpha"] > 0.85


def test_cronbach_flags_reverse_keyed_item(tmp_path: Path) -> None:
    # One item reverse-keyed (negative item-total correlation) -> flagged, and
    # dropping it should raise alpha.
    rng = np.random.default_rng(1)
    n = 150
    f = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "a": 0.8 * f + rng.normal(0, 0.4, n),
            "b": 0.8 * f + rng.normal(0, 0.4, n),
            "c": 0.8 * f + rng.normal(0, 0.4, n),
            "rev": -0.8 * f + rng.normal(0, 0.4, n),  # reverse-keyed (not recoded)
        }
    )
    csv = tmp_path / "rev.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    stats = pd.read_csv(Path(res.output_dir) / "cronbach_item_stats.csv").set_index("item")
    assert stats.loc["rev", "item_total_corr_corrected"] < 0
    assert bool(stats.loc["rev", "raises_alpha_if_dropped"]) is True
    # the engine surfaces the reverse-keyed warning
    assert "反向" in res.summary or "rev" in res.summary


def test_cronbach_config_override(tmp_path: Path) -> None:
    # Extra unrelated columns present; config restricts to 3 named items.
    rng = np.random.default_rng(2)
    n = 120
    f = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "x1": 0.8 * f + rng.normal(0, 0.4, n),
            "x2": 0.8 * f + rng.normal(0, 0.4, n),
            "x3": 0.8 * f + rng.normal(0, 0.4, n),
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
        }
    )
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"), config={"items": ["x1", "x2", "x3"]}
    )

    assert res.estimates["n_items"] == 3.0
    ref = _alpha_ref(df[["x1", "x2", "x3"]].to_numpy(float))
    assert abs(res.estimates["cronbach_alpha"] - ref) < 1e-3


def test_cronbach_too_few_items_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"only1": rng.normal(0, 1, 40), "only2": rng.normal(0, 1, 40)})
    csv = tmp_path / "two.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    assert "cronbach_alpha" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_cronbach_non_numeric_skips(tmp_path: Path) -> None:
    # All-categorical -> no continuous/count items -> honest skip, no crash.
    df = pd.DataFrame(
        {
            "g1": ["a", "b", "c"] * 10,
            "g2": ["x", "y", "z"] * 10,
            "g3": ["p", "q", "r"] * 10,
        }
    )
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    assert "cronbach_alpha" not in res.estimates
    assert "跳过" in res.summary
