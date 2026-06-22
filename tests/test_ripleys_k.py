"""Tests for ripleys_k: CSR-envelope point-pattern clustering detection.

A clustered pattern (points in tight blobs) should push L(r)-r above the CSR
envelope; a uniform-random pattern should stay (mostly) inside it.
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
        id="ripleys_k",
        method="Ripley's K / L function (point-pattern clustering)",
        domain="gis",
        family="spatial",
        goal="explain",
        preconditions=Precondition(requires_geo=True, min_rows=20),
    )


def _clustered_points(rng, n_clusters=8, per=14, spread=0.6):
    """Tight Gaussian blobs scattered in a 0-100 box -> strong clustering."""
    centers = rng.uniform(5, 95, size=(n_clusters, 2))
    pts = []
    for cx, cy in centers:
        pts.append(rng.normal([cx, cy], spread, size=(per, 2)))
    return np.vstack(pts)


def test_ripleys_k_detects_clustering(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pts = _clustered_points(rng)
    df = pd.DataFrame({"longitude": pts[:, 0], "latitude": pts[:, 1]})
    csv = tmp_path / "clustered.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    # config keeps n_sim modest for test speed; coords resolved from geo cols
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"n_sim": 39}
    )
    out = Path(res.output_dir)
    tab = pd.read_csv(out / "ripleys_k.csv")

    assert {"r", "K", "L", "L_minus_r", "env_low", "env_high"} <= set(tab.columns)
    assert res.estimates["n_points"] == float(len(pts))
    # clustering -> observed L-r exceeds the upper envelope at some scale
    assert res.estimates["frac_radii_clustered"] > 0.0
    assert res.estimates["max_L_minus_r"] > 0.0
    assert (out / "ripleys_l.png").exists()


def test_ripleys_k_random_pattern_near_csr(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "longitude": rng.uniform(0, 100, n),
            "latitude": rng.uniform(0, 100, n),
        }
    )
    csv = tmp_path / "random.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"n_sim": 39}
    )
    # a CSR pattern should be flagged clustered at few/no scales
    assert res.estimates["frac_radii_clustered"] < 0.5


def test_ripleys_k_too_few_points_degrades(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "longitude": rng.uniform(0, 10, 12),
            "latitude": rng.uniform(0, 10, 12),
        }
    )
    csv = tmp_path / "few.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    # honest degrade: no crash, a skip message, no K/L table
    assert "跳过" in res.summary
    assert "max_L_minus_r" not in res.estimates


def test_ripleys_k_config_xy_continuous(tmp_path: Path) -> None:
    # plain planar point pattern (no geo-named cols): resolve via config x/y
    rng = np.random.default_rng(7)
    pts = _clustered_points(rng)
    df = pd.DataFrame({"px": pts[:, 0], "py": pts[:, 1]})
    csv = tmp_path / "planar.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _entry(),
        output_root=str(tmp_path / "o"),
        config={"x": "px", "y": "py", "n_sim": 39},
    )
    assert res.estimates["n_points"] == float(len(pts))
    assert res.estimates["frac_radii_clustered"] > 0.0
