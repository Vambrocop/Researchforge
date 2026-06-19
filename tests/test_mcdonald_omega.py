"""Tests for mcdonald_omega — McDonald (1999) omega_total from a 1-factor solution.

Cross-checks: on a (near) tau-equivalent 1-factor synthetic with high loadings,
omega and alpha are both high and CLOSE to each other (omega relaxes alpha's
equal-loading assumption, so they coincide when loadings are roughly equal); an
independent recomputation of omega from the engine's saved loadings/communalities;
the alpha contrast; the config override; and <3-item / non-numeric honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="mcdonald_omega",
    method="McDonald's omega (congeneric reliability)",
    domain="psychometrics",
    family="psychometrics",
    goal="describe",
    preconditions={"min_continuous": 3, "min_rows": 10},
)


def _one_factor_csv(tmp_path: Path, loadings: list[float], n: int = 400, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    cols = {}
    for j, lam in enumerate(loadings, start=1):
        resid_sd = float(np.sqrt(max(1e-6, 1.0 - lam**2)))  # unit-variance items
        cols[f"q{j}"] = lam * f + rng.normal(0, resid_sd, n)
    csv = tmp_path / "omega.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    return csv


def test_omega_tau_equivalent_close_to_alpha(tmp_path: Path) -> None:
    # Equal high loadings -> tau-equivalent -> omega ~ alpha, both high.
    csv = _one_factor_csv(tmp_path, [0.8, 0.8, 0.8, 0.8, 0.8], n=500, seed=0)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    omega = res.estimates["mcdonald_omega"]
    alpha = res.estimates["cronbach_alpha_standardized"]
    assert omega > 0.8
    assert alpha > 0.8
    # tau-equivalent => omega and alpha coincide closely
    assert abs(omega - alpha) < 0.05


def test_omega_recompute_from_saved_loadings(tmp_path: Path) -> None:
    # Independently recompute omega = (sum L)^2 / ((sum L)^2 + sum(1-h2)) from
    # the engine's own omega_loadings.csv -> must match the reported estimate.
    csv = _one_factor_csv(tmp_path, [0.85, 0.75, 0.7, 0.8, 0.65], n=500, seed=1)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    load = pd.read_csv(Path(res.output_dir) / "omega_loadings.csv")
    s = load["loading"].sum()
    recomputed = s**2 / (s**2 + load["residual_var"].sum())
    assert abs(recomputed - res.estimates["mcdonald_omega"]) < 1e-2
    # residual var should equal 1 - communality (standardised items)
    assert np.allclose(load["residual_var"], 1.0 - load["communality_h2"], atol=1e-3)


def test_omega_congeneric_unequal_loadings(tmp_path: Path) -> None:
    # Unequal loadings (congeneric): omega is still high and >= alpha (alpha
    # under-estimates when tau-equivalence fails).
    csv = _one_factor_csv(tmp_path, [0.9, 0.8, 0.6, 0.5, 0.4], n=600, seed=2)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    omega = res.estimates["mcdonald_omega"]
    alpha = res.estimates["cronbach_alpha_standardized"]
    assert omega > 0.6
    # omega >= alpha (allow a small numeric slack)
    assert omega >= alpha - 0.02


def test_omega_config_override(tmp_path: Path) -> None:
    csv = _one_factor_csv(tmp_path, [0.8, 0.8, 0.8], n=300, seed=3)
    df = pd.read_csv(csv)
    df["junk"] = np.random.default_rng(9).normal(0, 1, len(df))
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"), config={"items": ["q1", "q2", "q3"]}
    )
    assert res.estimates["n_items"] == 3.0
    assert res.estimates["mcdonald_omega"] > 0.7


def test_omega_too_few_items_skips(tmp_path: Path) -> None:
    csv = _one_factor_csv(tmp_path, [0.8, 0.8], n=100, seed=4)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    assert "mcdonald_omega" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_omega_non_numeric_skips(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"a": list("xyz") * 20, "b": list("pqr") * 20, "c": list("uvw") * 20}
    )
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "mcdonald_omega" not in res.estimates
    assert "跳过" in res.summary
