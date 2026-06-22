"""Tests for weighted_estimation — Horvitz-Thompson design-weighted mean & total.

Cross-checks:
  * hand-computed weighted mean / total vs the engine (weighted != unweighted);
  * the Kish design effect deff = n·Σw²/(Σw)² > 1 under unequal weights, and
    n_eff = (Σw)²/Σw² < n;
  * the 95% CI brackets the weighted mean and the products are written;
  * a per-group weighted-means table when config group is supplied;
  * honest skip when no weight column / no config weight is available.
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
        id="weighted_estimation",
        method="Design-weighted estimation (Horvitz-Thompson)",
        domain="statistics",
        family="survey_methods",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def test_weighted_mean_matches_hand_calc_and_deff_gt_one(tmp_path: Path) -> None:
    # value + a known (unequal) weight column where weighted != unweighted.
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    # weights as non-integer floats (so profiler -> continuous) named "weight".
    w = np.array([0.5, 1.5, 2.5, 3.5, 0.7, 1.2, 4.1, 2.3])
    df = pd.DataFrame({"income": y, "weight": w})
    csv = tmp_path / "w.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "income", "weight": "weight"})

    wmean_ref = float((w * y).sum() / w.sum())
    wtotal_ref = float((w * y).sum())
    n = len(y)
    deff_ref = float(n * (w * w).sum() / w.sum() ** 2)
    neff_ref = float(w.sum() ** 2 / (w * w).sum())

    assert abs(res.estimates["weighted_mean"] - wmean_ref) < 1e-6
    assert abs(res.estimates["weighted_total"] - wtotal_ref) < 1e-4
    assert abs(res.estimates["design_effect"] - deff_ref) < 1e-6
    assert abs(res.estimates["n_eff"] - neff_ref) < 1e-3
    assert res.estimates["n"] == n
    # unequal weights -> deff > 1 and n_eff < n
    assert res.estimates["design_effect"] > 1.0
    assert res.estimates["n_eff"] < n
    # weighted mean differs from unweighted (sanity that weighting bites)
    assert abs(res.estimates["weighted_mean"] - float(y.mean())) > 1.0
    # CI brackets the point estimate
    assert res.estimates["ci_low"] < res.estimates["weighted_mean"] < res.estimates["ci_high"]

    out = Path(res.output_dir)
    assert (out / "weighted_estimates.csv").exists()
    assert "Horvitz" in res.summary or "加权均值" in res.summary


def test_design_se_form_known_value(tmp_path: Path) -> None:
    # Independently recompute the Kish/ratio design SE and check the CI half-width.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    w = np.array([0.8, 1.3, 2.2, 0.9, 3.1, 1.7])
    df = pd.DataFrame({"val": y, "pweight": w})
    csv = tmp_path / "se.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "val", "weight": "pweight"})

    wmean = float((w * y).sum() / w.sum())
    z = w * (y - wmean)
    n = len(y)
    var = (n / (n - 1)) * float((z * z).sum()) / w.sum() ** 2
    se_ref = float(np.sqrt(var))
    assert abs(res.estimates["se_mean"] - se_ref) < 1e-6
    half = res.estimates["ci_high"] - res.estimates["weighted_mean"]
    assert abs(half - 1.959963984540054 * se_ref) < 1e-4


def test_auto_detects_weight_by_name_and_group(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 60
    grp = (["A"] * 30) + (["B"] * 30)
    y = rng.normal(50, 5, n).round(3)
    w = rng.uniform(0.5, 3.0, n).round(3)  # float -> continuous, name hints weight
    df = pd.DataFrame({"score": y, "sampwt": w, "region": grp})
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # no config weight -> auto-detect "sampwt" by name; group via config
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"value": "score", "group": "region"})
    assert "weighted_mean" in res.estimates
    out = Path(res.output_dir)
    assert (out / "weighted_by_group.csv").exists()
    gdf = pd.read_csv(out / "weighted_by_group.csv")
    assert set(gdf["group"]) == {"A", "B"}


def test_skips_when_no_weight_available(tmp_path: Path) -> None:
    # two continuous columns but neither hints "weight" and no config weight -> skip.
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20).round(3),
                       "y": rng.normal(0, 1, 20).round(3)})
    csv = tmp_path / "nw.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "weighted_mean" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()
