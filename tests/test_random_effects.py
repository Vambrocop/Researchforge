"""Tests for random_effects: panel gate + RE/FE + Hausman (skips without linearmodels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="random_effects",
        method="Random-effects panel model (+ Hausman test)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=12),
    )


def _hausman_entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="hausman_test",
        method="Hausman specification test (FE vs RE)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=12),
    )


def test_random_effects_recovers_slope_and_hausman(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(0)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)  # unit effect, INDEPENDENT of x -> RE consistent
        for t in range(5):
            x = rng.normal(0, 1)
            y = 1.5 * x + alpha + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2015 + t, "y": round(y, 4), "x": round(x, 4)})
    csv = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "fe_re_coefficients.csv").exists()
    assert abs(res.estimates["x"] - 1.5) < 0.4  # slope recovered
    assert "hausman_p" in res.estimates
    assert res.estimates["hausman_p"] > 0.05  # effect independent of x -> RE not rejected
    # Fix 3: strict-exogeneity / no-time-effects disclosure always present
    assert "严格外生" in res.summary


def test_random_effects_drops_time_invariant_predictor(tmp_path: Path) -> None:
    """Fix 1: a time-invariant covariate mixed in with time-varying ones must NOT
    kill the method (PanelOLS(entity_effects=True) would otherwise raise
    AbsorbingEffectError on it) -- it should be dropped, disclosed, and FE/RE +
    Hausman still computed on the remaining time-varying predictor(s)."""
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(2)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)  # unit effect, independent of x -> RE consistent
        z = rng.normal(0, 1)  # time-invariant covariate (constant within unit)
        for t in range(5):
            x = rng.normal(0, 1)
            y = 1.5 * x + alpha + rng.normal(0, 0.5)
            rows.append(
                {
                    "firm": f"u{u}",
                    "year": 2015 + t,
                    "y": round(y, 4),
                    "x": round(x, 4),
                    "z": round(z, 4),
                }
            )
    csv = tmp_path / "panel_ti.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    # must not raise / must not silently fail -- FE/RE + Hausman still produced
    assert (out / "fe_re_coefficients.csv").exists()
    assert "hausman_p" in res.estimates
    assert "z" not in res.estimates  # time-invariant predictor dropped, not fit
    assert abs(res.estimates["x"] - 1.5) < 0.4
    # dropped-column disclosure present (mirrors panel_extra siblings' wording)
    assert "z" in res.summary and "剔除" in res.summary


def test_random_effects_all_time_invariant_predictors_skips(tmp_path: Path) -> None:
    """Fix 1: if EVERY predictor is time-invariant, degrade honestly with a clear
    skip message instead of raising AbsorbingEffectError."""
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(3)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)
        x = rng.normal(0, 1)  # time-invariant (constant within unit)
        for t in range(5):
            y = 1.5 * x + alpha + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2015 + t, "y": round(y, 4), "x": round(x, 4)})
    csv = tmp_path / "panel_all_ti.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    assert "hausman_p" not in res.estimates  # degraded, no fit attempted
    assert "跳过" in res.summary


def test_random_effects_hausman_fallback_matches_hausman_test(tmp_path: Path) -> None:
    """Fix 2: when V_FE-V_RE is non-PSD (common in small/short panels), the classic
    pinv-clipped-to-zero Hausman would spuriously clip H to 0 (p=1 -> "use RE").
    random_effects must instead fall back to the regression-based (Mundlak) Hausman,
    exactly like panel_extra.hausman_test -- and, on the SAME data, agree with it."""
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(2)
    rows = []
    # small N, T=2, independent alpha/x: empirically produces a non-PSD V_FE-V_RE
    # for this seed (verified: eigmin(V_FE-V_RE) < 0), exercising the fallback path
    # in BOTH random_effects and hausman_test.
    for u in range(10):
        alpha = rng.normal(0, 1)
        for t in range(2):
            x = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            y = 1.0 * x + 0.5 * x2 + alpha + rng.normal(0, 0.5)
            rows.append(
                {
                    "firm": f"u{u}",
                    "year": 2020 + t,
                    "y": round(y, 6),
                    "x": round(x, 6),
                    "x2": round(x2, 6),
                }
            )
    csv = tmp_path / "panel_nonpsd.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel

    res_re = run_analysis(fp, _entry(), output_root=str(tmp_path / "o_re"))
    res_h = run_analysis(
        fp,
        _hausman_entry(),
        output_root=str(tmp_path / "o_h"),
        config={"unit": "firm", "time": "year", "outcome": "y", "predictors": ["x", "x2"]},
    )

    # both must complete without raising and both must have taken the fallback path
    assert "hausman_p" in res_re.estimates
    assert "hausman_p" in res_h.estimates
    assert "回归式" in res_re.summary  # fallback disclosure fired (non-classic path)
    assert "回归式" in res_h.summary

    # the guarded regression-based Hausman is deterministic given the same data/
    # predictors -> random_effects and hausman_test must agree numerically.
    assert res_re.estimates["hausman_stat"] == pytest.approx(res_h.estimates["hausman_chi2"], abs=1e-6)
    assert res_re.estimates["hausman_p"] == pytest.approx(res_h.estimates["hausman_p"], abs=1e-6)


def test_random_effects_precondition_unmet(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # not panel
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)

    assert not ok
    assert any("面板" in u for u in unmet)
