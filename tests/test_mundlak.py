"""Tests for mundlak: Mundlak CRE — panel gate + within beta recovery + robust FE-vs-RE
test (rejects RE when the entity effect correlates with x). Skips without linearmodels."""

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
        id="mundlak",
        method="Mundlak correlated random effects (CRE)",
        domain="economics",
        family="econometrics",
        goal="explain",
        preconditions=Precondition(is_panel=True, min_continuous=2, min_rows=12),
    )


def _correlated_panel(seed: int = 0, beta: float = 1.5):
    """Entity effect alpha is CORRELATED with x (x = 0.8*alpha + noise) so the
    Mundlak test should reject RE and recover the within beta."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(30):
        alpha = rng.normal(0, 1)  # unit effect, correlated with x below
        for t in range(5):
            x = 0.8 * alpha + rng.normal(0, 1)
            y = beta * x + alpha + rng.normal(0, 0.5)
            rows.append({"firm": f"u{u}", "year": 2015 + t, "y": round(y, 4), "x": round(x, 4)})
    return pd.DataFrame(rows)


def test_mundlak_recovers_within_beta_and_rejects_re(tmp_path: Path) -> None:
    pytest.importorskip("linearmodels")
    csv = tmp_path / "panel.csv"
    _correlated_panel().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    assert fp.is_panel
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "predictors": ["x"]},
    )
    out = Path(res.output_dir)

    assert (out / "mundlak_coefficients.csv").exists()
    # within (beta) estimate recovered
    assert abs(res.estimates["x"] - 1.5) < 0.4
    # Mundlak test rejects RE (entity effect correlates with x)
    assert "mundlak_p" in res.estimates
    assert "mundlak_wald_chi2" in res.estimates
    assert res.estimates["mundlak_p"] < 0.05
    assert res.estimates["n_entities"] == 30.0
    assert res.estimates["n_predictors"] == 1.0


def test_mundlak_not_panel_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 30), "x": rng.normal(0, 1, 30)})  # not panel
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("面板" in u for u in unmet)


def test_mundlak_degrades_without_linearmodels(monkeypatch, tmp_path: Path) -> None:
    """If linearmodels import fails, the branch degrades honestly (Chinese skip msg,
    no crash)."""
    import builtins

    pytest.importorskip("linearmodels")
    csv = tmp_path / "panel.csv"
    _correlated_panel().to_csv(csv, index=False)
    fp = profile_dataset(csv)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("linearmodels"):
            raise ImportError("simulated missing linearmodels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "predictors": ["x"]},
    )
    assert "mundlak_wald_chi2" not in res.estimates
    assert "Mundlak" in res.summary and "跳过" in res.summary
