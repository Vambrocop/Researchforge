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
from researchforge.catalog.schema import Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def test_likert_count_items_pass_min_numeric_precondition(tmp_path: Path) -> None:
    # Regression for the inference-reviewer SHOULD-FIX: integer Likert items profile as
    # `count` (not `continuous`), so gating on min_continuous hid these analyses from their
    # canonical data. min_numeric_cols counts continuous OR count, so a Likert scale passes.
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1, 40)
    df = pd.DataFrame(
        {f"q{j}": np.clip(np.round(2.5 + theta + rng.normal(0, 0.5, 40)), 1, 5).astype(int)
         for j in range(4)}
    )
    csv = tmp_path / "likert.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # the Likert columns are NOT continuous (so min_continuous would have failed)...
    assert sum(1 for c in fp.columns if c.kind == "continuous") < 3
    # ...but min_numeric_cols (continuous OR count) is satisfied -> recommendable
    ok, unmet = check_preconditions(fp, Precondition(min_numeric_cols=3, min_rows=10))
    assert ok, f"Likert scale should meet min_numeric_cols: {unmet}"

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


def test_cronbach_default_prefers_ordinal_like_over_age(tmp_path: Path) -> None:
    # Wave K E2 (dogfood P1): a Likert questionnaire (8 bounded 1-5 satisfaction
    # items -> profiler ordinal_like=True) sits alongside a demographic `age`
    # column that also profiles as numeric (`count`, non-negative integer) but
    # is NOT ordinal_like (wide range, >7 distinct levels). Age has nothing to
    # do with the scale's internal consistency; if the old "all continuous+count"
    # default swept it in, its huge/unrelated variance collapses alpha toward 0.
    # The new default should auto-restrict to the 8 ordinal_like items and
    # recover a plausible internal-consistency reading.
    rng = np.random.default_rng(7)
    n = 150
    theta = rng.normal(0, 1, n)
    items = {
        f"q{j}": np.clip(np.round(3 + theta + rng.normal(0, 0.9, n)), 1, 5).astype(int)
        for j in range(1, 9)
    }
    df = pd.DataFrame(items)
    df["age"] = rng.integers(18, 70, n)  # unrelated demographic, wide range -> not ordinal_like
    csv = tmp_path / "p1_likert.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    age_col = next(c for c in fp.columns if c.name == "age")
    assert age_col.ordinal_like is False, "test fixture assumption: age must not be ordinal_like"
    q_cols = [c for c in fp.columns if c.name.startswith("q")]
    assert all(c.ordinal_like for c in q_cols), "test fixture assumption: q1..q8 must be ordinal_like"

    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    assert res.estimates["n_items"] == 8.0
    stats = pd.read_csv(Path(res.output_dir) / "cronbach_item_stats.csv")
    assert set(stats["item"]) == {f"q{j}" for j in range(1, 9)}
    assert "age" not in set(stats["item"])
    # alpha recovers to a plausible internal-consistency band instead of
    # collapsing toward 0 under age's unrelated variance.
    assert 0.5 <= res.estimates["cronbach_alpha"] <= 0.9


def test_cronbach_falls_back_to_full_default_without_ordinal_like_columns(tmp_path: Path) -> None:
    # A hand-built continuous scale (no profiler ordinal_like columns at all,
    # e.g. z-scored / non-integer items) must still resolve to the old
    # unrestricted continuous+count default -- alpha is also legitimately used
    # for continuous scales, so prefer_ordinal must never turn into a hard
    # requirement / honest-skip that didn't exist before.
    rng = np.random.default_rng(8)
    theta = rng.normal(0.0, 1.0, 40)
    df = pd.DataFrame(
        {f"i{j + 1}": theta + rng.normal(0.0, 0.30, 40) for j in range(4)}
    )
    csv = tmp_path / "continuous_scale.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert not any(getattr(c, "ordinal_like", False) for c in fp.columns)

    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    assert "cronbach_alpha" in res.estimates
    assert res.estimates["n_items"] == 4.0


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
