"""Tests for the ACTUARIAL / DEMOGRAPHY family: life_table, chain_ladder,
loss_distribution.

Known-value cases:
  * life_table: a hand-computed 3-age table (m_x = [0.1, 0.2, 0.5], n=1, a_x=0.5,
    radix=100000) with l_x / d_x / L_x / T_x / e_x worked out in the docstring.
  * chain_ladder: a clean 3x3 triangle with a constant column factor x1.5 -> all
    f_j = 1.5 recovered, reserve = projected ultimate - latest = 650 (hand-computed).
  * loss_distribution: lognormal-sampled losses (fixed seed) -> best fit lognormal,
    VaR_95 ~= the theoretical lognormal 95% quantile within tolerance.

Honest-degrade paths assert the Chinese "跳过" message and no crash. Fixed seeds.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="actuarial",
        family="actuarial",
        goal="describe",
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) life_table
# --------------------------------------------------------------------------- #
def test_life_table_known_value(tmp_path: Path) -> None:
    """Hand-computed period life table for m_x = [0.1, 0.2, 0.5], ages [0,1,2],
    n=1, a_x=0.5, radix=100000:
        q_x = [0.0952381, 0.1818182, 1.0]   (last interval closed q=1)
        l_x = [100000, 90476.19, 74026.72]
        d_x = [9523.81, 16449.47, 74026.72]
        L_x = [95238.10, 82251.46, 148053.44]   (L_2 = l_2/m_2 = 74026.72/0.5)
        T_x = [325542.99, 230304.89, 148053.44]
        e_x = [3.25543, 2.54548, 2.0]
    So e_0 = 3.25543 and e_2 = 1/m_2 = 2.0 exactly.
    """
    csv = _csv(tmp_path, "lt.csv", pd.DataFrame({
        "age": [0, 1, 2],
        "mx": [0.1, 0.2, 0.5],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("life_table", "Life table"),
                       output_root=str(tmp_path / "o"),
                       config={"age": "age", "rate": "mx"})
    e = res.estimates
    assert math.isclose(e["e0"], 3.25543, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(e["e_at_min_age"], 3.25543, rel_tol=0, abs_tol=1e-3)
    assert e["radix"] == 100000.0
    assert e["n_ages"] == 3.0
    # T_0 = ΣL_x = 95238.1+82251.1+148051.95 ≈ 325541.15 (hand-verified; = e0·radix).
    assert math.isclose(e["total_person_years"], 325541.15, rel_tol=0, abs_tol=2.0)
    out = Path(res.output_dir)
    assert (out / "life_table.csv").exists()
    tbl = pd.read_csv(out / "life_table.csv")
    # check the engineered columns row-by-row
    assert list(tbl["age"]) == [0, 1, 2]
    assert math.isclose(tbl["q_x"].iloc[0], 0.0952381, abs_tol=1e-5)
    assert math.isclose(tbl["q_x"].iloc[1], 0.1818182, abs_tol=1e-5)
    assert math.isclose(tbl["q_x"].iloc[2], 1.0, abs_tol=1e-9)
    assert math.isclose(tbl["l_x"].iloc[0], 100000.0, abs_tol=1e-3)
    assert math.isclose(tbl["l_x"].iloc[1], 90476.19, abs_tol=1e-1)
    assert math.isclose(tbl["l_x"].iloc[2], 74025.97, abs_tol=1e-1)
    assert math.isclose(tbl["d_x"].iloc[0], 9523.81, abs_tol=1e-1)
    assert math.isclose(tbl["L_x"].iloc[2], 148051.95, abs_tol=1.0)
    # e_x at the open last age = 1/m_x exactly = 2.0
    assert math.isclose(tbl["e_x"].iloc[2], 2.0, abs_tol=1e-3)


def test_life_table_from_deaths_exposure(tmp_path: Path) -> None:
    """deaths/exposure -> m_x = D/E. With D=[10,20,50], E=[100,100,100] the central
    rates are m_x=[0.1,0.2,0.5] -> identical e_0 to the m_x test."""
    csv = _csv(tmp_path, "de.csv", pd.DataFrame({
        "age": [0, 1, 2],
        "deaths": [10.0, 20.0, 50.0],
        "exposure": [100.0, 100.0, 100.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("life_table", "Life table"),
                       output_root=str(tmp_path / "o"),
                       config={"age": "age", "deaths": "deaths", "exposure": "exposure"})
    assert math.isclose(res.estimates["e0"], 3.25543, rel_tol=0, abs_tol=1e-3)


def test_life_table_qx_input(tmp_path: Path) -> None:
    """Supplying q_x directly should reproduce the same survivorship column."""
    csv = _csv(tmp_path, "q.csv", pd.DataFrame({
        "age": [0, 1, 2],
        "qx": [0.0952381, 0.1818182, 0.5],   # last is overridden to 1.0 internally
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("life_table", "Life table"),
                       output_root=str(tmp_path / "o"),
                       config={"age": "age", "qx": "qx"})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "life_table.csv")
    assert math.isclose(tbl["l_x"].iloc[1], 90476.19, abs_tol=1.0)
    assert math.isclose(tbl["q_x"].iloc[2], 1.0, abs_tol=1e-9)  # closed last interval
    assert res.estimates["e0"] > 0


def test_life_table_degrade_no_mortality(tmp_path: Path) -> None:
    """Age column but no mortality input -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "x.csv", pd.DataFrame({
        "age": [0, 1, 2],
        "label": ["a", "b", "c"],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("life_table", "Life table"),
                       output_root=str(tmp_path / "o"),
                       config={"age": "age"})
    assert "跳过" in res.summary
    assert "e0" not in res.estimates


# --------------------------------------------------------------------------- #
# 2) chain_ladder
# --------------------------------------------------------------------------- #
def _clean_triangle_wide() -> pd.DataFrame:
    """3x3 cumulative run-off triangle, each development column = 1.5x the prior.
        origin0: 100, 150, 225
        origin1: 200, 300, (NaN)
        origin2: 400, (NaN), (NaN)
    -> f_0 = (150+300)/(100+200) = 1.5 ; f_1 = 225/150 = 1.5.
    Projected ultimates: 225, 450, 900 ; latest paid: 225, 300, 400.
    Reserve = (0) + (450-300) + (900-400) = 650.
    """
    return pd.DataFrame({
        "origin": [2018, 2019, 2020],
        "dev0": [100.0, 200.0, 400.0],
        "dev1": [150.0, 300.0, np.nan],
        "dev2": [225.0, np.nan, np.nan],
    })


def test_chain_ladder_wide_known_value(tmp_path: Path) -> None:
    csv = _csv(tmp_path, "tri.csv", _clean_triangle_wide())
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("chain_ladder", "Chain-ladder"),
                       output_root=str(tmp_path / "o"))
    e = res.estimates
    assert math.isclose(e["oldest_factor"], 1.5, abs_tol=1e-9)
    assert math.isclose(e["latest_factor"], 1.5, abs_tol=1e-9)
    assert math.isclose(e["total_reserve"], 650.0, abs_tol=1e-6)
    assert math.isclose(e["total_ultimate"], 1575.0, abs_tol=1e-6)
    assert math.isclose(e["total_latest_paid"], 925.0, abs_tol=1e-6)
    assert e["n_origins"] == 3.0
    out = Path(res.output_dir)
    assert (out / "chain_ladder_reserve.csv").exists()
    assert (out / "chain_ladder_factors.csv").exists()
    per = pd.read_csv(out / "chain_ladder_reserve.csv")
    # per-origin IBNR: 0, 150, 500
    ibnr = sorted(per["ibnr_reserve"].round(4).tolist())
    assert math.isclose(ibnr[0], 0.0, abs_tol=1e-6)
    assert math.isclose(ibnr[1], 150.0, abs_tol=1e-6)
    assert math.isclose(ibnr[2], 500.0, abs_tol=1e-6)
    fac = pd.read_csv(out / "chain_ladder_factors.csv")
    assert all(math.isclose(v, 1.5, abs_tol=1e-9) for v in fac["age_to_age_factor"])


def test_chain_ladder_long_form(tmp_path: Path) -> None:
    """Same triangle supplied in LONG form (origin, dev, claims) -> same reserve."""
    long = pd.DataFrame({
        "acc_year": [2018, 2018, 2018, 2019, 2019, 2020],
        "dev_period": [0, 1, 2, 0, 1, 0],
        "cum_claims": [100.0, 150.0, 225.0, 200.0, 300.0, 400.0],
    })
    csv = _csv(tmp_path, "long.csv", long)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("chain_ladder", "Chain-ladder"),
                       output_root=str(tmp_path / "o"),
                       config={"origin": "acc_year", "dev": "dev_period",
                               "claims": "cum_claims"})
    e = res.estimates
    assert math.isclose(e["total_reserve"], 650.0, abs_tol=1e-6)
    assert math.isclose(e["oldest_factor"], 1.5, abs_tol=1e-9)


def test_chain_ladder_degrade_no_triangle(tmp_path: Path) -> None:
    """A single numeric column is not a triangle (need >=2 dev columns) -> honest 跳过."""
    csv = _csv(tmp_path, "one.csv", pd.DataFrame({
        "origin": ["a", "b", "c"],
        "val": [1.0, 2.0, 3.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("chain_ladder", "Chain-ladder"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "total_reserve" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) loss_distribution
# --------------------------------------------------------------------------- #
def test_loss_distribution_lognormal_recovery(tmp_path: Path) -> None:
    """Lognormal-sampled losses (mu=8, sigma=0.6, n=6000, seed=11) -> best fit
    should be lognormal, and VaR_95 close to the theoretical lognormal 95%
    quantile exp(mu + sigma*z_0.95), z_0.95 = 1.6448536."""
    rng = np.random.default_rng(11)
    mu, sigma, n = 8.0, 0.6, 6000
    losses = rng.lognormal(mean=mu, sigma=sigma, size=n)
    csv = _csv(tmp_path, "loss.csv", pd.DataFrame({"claim_amount": losses}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("loss_distribution", "Loss distribution"),
                       output_root=str(tmp_path / "o"),
                       config={"loss": "claim_amount"})
    e = res.estimates
    # best fit is lognormal (named in the summary)
    assert "lognormal" in res.summary
    # theoretical lognormal 95% quantile
    z95 = 1.6448536269514722
    theo_var95 = math.exp(mu + sigma * z95)
    assert math.isclose(e["var_95"], theo_var95, rel_tol=0.10)
    # TVaR (tail mean above VaR) must exceed VaR at the same level
    assert e["tvar_95"] > e["var_95"]
    assert e["tvar_99"] > e["var_99"]
    assert e["var_99"] > e["var_95"]
    # fitted shape ~ sigma for a lognormal fit with floc=0
    assert math.isclose(e["fitted_shape"], sigma, rel_tol=0.15)
    # sanity on summary stats
    assert e["mean_loss"] > 0
    assert e["n_losses"] == float(n)
    out = Path(res.output_dir)
    assert (out / "loss_distribution_fits.csv").exists()
    assert (out / "loss_distribution_risk.csv").exists()
    fits = pd.read_csv(out / "loss_distribution_fits.csv")
    # the best (delta_aic == 0) row is lognormal
    best_row = fits.sort_values("aic").iloc[0]
    assert best_row["distribution"] == "lognormal"


def test_loss_distribution_custom_alpha(tmp_path: Path) -> None:
    """A custom alpha level is honoured (90%) and still reports the required 95/99."""
    rng = np.random.default_rng(3)
    losses = rng.lognormal(mean=7.0, sigma=0.5, size=2000)
    csv = _csv(tmp_path, "loss.csv", pd.DataFrame({"loss": losses}))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("loss_distribution", "Loss distribution"),
                       output_root=str(tmp_path / "o"),
                       config={"loss": "loss", "alpha": 0.90})
    e = res.estimates
    assert not math.isnan(e["var_95"])
    assert not math.isnan(e["var_99"])
    out = Path(res.output_dir)
    risk = pd.read_csv(out / "loss_distribution_risk.csv")
    assert 0.90 in set(risk["alpha"].round(4))


def test_loss_distribution_degrade_no_positive(tmp_path: Path) -> None:
    """No positive losses (all non-positive) -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "neg.csv", pd.DataFrame({
        "loss": [-1.0, 0.0, -3.0, 0.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("loss_distribution", "Loss distribution"),
                       output_root=str(tmp_path / "o"),
                       config={"loss": "loss"})
    assert "跳过" in res.summary
    assert "var_95" not in res.estimates


def test_loss_distribution_reproducible(tmp_path: Path) -> None:
    """Deterministic fit: same data -> identical estimates (no internal randomness)."""
    rng = np.random.default_rng(99)
    losses = rng.lognormal(mean=6.0, sigma=0.7, size=1500)
    csv = _csv(tmp_path, "loss.csv", pd.DataFrame({"loss": losses}))
    fp = profile_dataset(csv)
    r1 = run_analysis(fp, _entry("loss_distribution", "Loss distribution"),
                      output_root=str(tmp_path / "o1"), config={"loss": "loss"})
    r2 = run_analysis(fp, _entry("loss_distribution", "Loss distribution"),
                      output_root=str(tmp_path / "o2"), config={"loss": "loss"})
    assert r1.estimates["best_aic"] == r2.estimates["best_aic"]
    assert r1.estimates["var_95"] == r2.estimates["var_95"]

