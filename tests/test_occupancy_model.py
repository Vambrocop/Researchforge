"""Tests for occupancy_model: MacKenzie single-season occupancy psi(.)p(.).

Simulate sites with KNOWN occupancy psi and detection p over K repeat visits, build
the 0/1 detection matrix, and assert the MLE recovers psi and p within tolerance,
that the detection-corrected psi-hat exceeds the naive occupancy (the whole point of
the model), and that n_sites is right. Plus an honest-skip test (too few sites).
Pure Python (numpy/scipy) — no skip-on-missing-library guard needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="occupancy_model",
        method="MacKenzie single-season occupancy model",
        domain="ecology",
        family="ecology",
        goal="estimate",
        preconditions=Precondition(min_rows=15),
    )


def _simulate_occupancy(
    seed: int = 0, n_sites: int = 300, n_visits: int = 5,
    psi: float = 0.6, p: float = 0.4,
) -> pd.DataFrame:
    """Each site occupied ~ Bernoulli(psi); if occupied, each of K visits detects ~
    Bernoulli(p); if unoccupied, all visits are 0 (no false positives)."""
    rng = np.random.default_rng(seed)
    occupied = rng.random(n_sites) < psi
    rows = []
    for i in range(n_sites):
        row = {}
        for v in range(n_visits):
            if occupied[i] and rng.random() < p:
                row[f"v{v + 1}"] = 1
            else:
                row[f"v{v + 1}"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def test_occupancy_recovers_psi_p_and_corrects_naive(tmp_path: Path) -> None:
    true_psi, true_p = 0.6, 0.4
    csv = tmp_path / "occ.csv"
    _simulate_occupancy(psi=true_psi, p=true_p).to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"visits": ["v1", "v2", "v3", "v4", "v5"]},
    )
    out = Path(res.output_dir)

    assert (out / "occupancy_estimates.csv").exists()

    e = res.estimates
    # psi-hat and p-hat recovered within tolerance
    assert abs(e["occupancy_psi"] - true_psi) < 0.15
    assert abs(e["detection_p"] - true_p) < 0.15
    # detection correction: psi-hat > naive occupancy (the headline insight)
    assert e["occupancy_psi"] > e["naive_occupancy"]
    # naive is an under-estimate of true psi (imperfect detection)
    assert e["naive_occupancy"] < true_psi
    assert e["n_sites"] == 300.0
    # SE / CI keys present and the CI brackets the point estimate
    assert "occupancy_psi_se" in e
    assert e["occupancy_psi_ci_low"] <= e["occupancy_psi"] <= e["occupancy_psi_ci_high"]
    assert e["detection_p_ci_low"] <= e["detection_p"] <= e["detection_p_ci_high"]
    assert e["converged"] == 1.0
    # summary carries the headline + disclosure
    assert any("占据" in s for s in [res.summary]) or "占据" in res.summary
    assert "⚠" in res.summary


def test_occupancy_auto_detects_binary_visit_columns(tmp_path: Path) -> None:
    """Without config["visits"], the branch should auto-pick the 0/1 columns."""
    df = _simulate_occupancy(seed=3, psi=0.6, p=0.4)
    # add a non-binary nuisance column that must be ignored
    df["elevation"] = np.linspace(100.0, 900.0, len(df))
    csv = tmp_path / "occ_auto.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    e = res.estimates
    assert "occupancy_psi" in e
    assert e["n_sites"] == float(len(df))
    assert e["occupancy_psi"] > e["naive_occupancy"]


def test_occupancy_too_few_sites_skips(tmp_path: Path) -> None:
    df = _simulate_occupancy(seed=1, n_sites=10, n_visits=4)  # < 15 sites
    csv = tmp_path / "small.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # precondition gate
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("行" in u for u in unmet)

    # and if forced to run, the branch skips honestly without crashing
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "occupancy_psi" not in res.estimates
    assert "占据" in res.summary and "跳过" in res.summary
