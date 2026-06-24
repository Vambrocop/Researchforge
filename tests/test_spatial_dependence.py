"""Spatial dependence & regionalization: bivariate Moran, local Geary, SKATER.

Planted spatial structure (a gradient field / two spatial blocks); assert the
statistics recover it (positive cross-correlation, significant local clusters,
contiguous regions that match the blocks). Coordinates passed via config to keep
roles deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _run(csv, aid, tmp_path, config=None):
    fp = profile_dataset(csv)
    return run_analysis(fp, _CAT.by_id(aid), output_root=str(tmp_path / "o"), config=config)


def _grid_df(side=8, seed=0) -> pd.DataFrame:
    """side×side grid; a smooth west→east gradient field + a correlated second field."""
    rng = np.random.default_rng(seed)
    gx, gy, grad = [], [], []
    for i in range(side):
        for j in range(side):
            gx.append(float(i) + 0.5)
            gy.append(float(j) + 0.5)
            grad.append(float(i))
    grad = np.array(grad)
    v1 = grad + rng.normal(0, 0.3, grad.size)
    v2 = grad + rng.normal(0, 0.3, grad.size)
    return pd.DataFrame({"gx": gx, "gy": gy, "v1": v1, "v2": v2})


# --------------------------------------------------------------------------- #
# bivariate_moran
# --------------------------------------------------------------------------- #
def test_bivariate_moran_positive_when_cofields_align(tmp_path: Path) -> None:
    csv = tmp_path / "bm.csv"
    _grid_df().to_csv(csv, index=False)
    res = _run(csv, "bivariate_moran", tmp_path,
               config={"x": "gx", "y": "gy", "var1": "v1", "var2": "v2", "n_perm": 199})
    e = res.estimates
    assert e["bivariate_moran_I"] > 0.2          # both fields share the spatial gradient
    assert e["p_value"] < 0.05
    assert (Path(res.output_dir) / "bivariate_moran_points.csv").exists()


def test_bivariate_moran_degrades_no_coords(tmp_path: Path) -> None:
    csv = tmp_path / "one.csv"
    pd.DataFrame({"v1": np.arange(20.0)}).to_csv(csv, index=False)
    res = _run(csv, "bivariate_moran", tmp_path, config={"var1": "v1"})
    assert "跳过" in res.summary
    assert "bivariate_moran_I" not in res.estimates


# --------------------------------------------------------------------------- #
# local_geary
# --------------------------------------------------------------------------- #
def test_local_geary_finds_similar_clusters(tmp_path: Path) -> None:
    csv = tmp_path / "lg.csv"
    _grid_df().to_csv(csv, index=False)
    res = _run(csv, "local_geary", tmp_path,
               config={"x": "gx", "y": "gy", "value": "v1", "n_perm": 199})
    e = res.estimates
    # a smooth gradient -> strong local similarity -> significant low-C locations
    assert e["n_significant"] > 0
    assert e["n_similar_clusters"] > 0
    tab = pd.read_csv(Path(res.output_dir) / "local_geary.csv")
    assert "local_geary_c" in tab.columns and len(tab) == e["n"]


# --------------------------------------------------------------------------- #
# skater
# --------------------------------------------------------------------------- #
def _two_block_df(side=8, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(side):
        for j in range(side):
            block = 0 if i < side // 2 else 1     # west vs east halves
            rows.append({"gx": float(i) + 0.5, "gy": float(j) + 0.5,
                         "attr": block * 10.0 + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def test_skater_recovers_two_contiguous_blocks(tmp_path: Path) -> None:
    df = _two_block_df()
    csv = tmp_path / "sk.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "skater", tmp_path,
               config={"x": "gx", "y": "gy", "features": ["attr"], "n_clusters": 2, "knn_k": 4})
    e = res.estimates
    assert e["n_regions"] == 2.0
    regions = pd.read_csv(Path(res.output_dir) / "skater_regions.csv")
    # the two regions should align with the west/east split: each half ~one region
    west = regions[regions["gx"] < 4.0]["region"]
    east = regions[regions["gx"] >= 4.0]["region"]
    assert west.nunique() == 1 and east.nunique() == 1
    assert west.iloc[0] != east.iloc[0]


def test_skater_within_ssd_beats_random_split(tmp_path: Path) -> None:
    # SKATER's homogeneous regions should have lower within-SSD than a random 2-way split
    df = _two_block_df(seed=5)
    csv = tmp_path / "sk2.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "skater", tmp_path,
               config={"x": "gx", "y": "gy", "features": ["attr"], "n_clusters": 2, "knn_k": 4})
    ssd = res.estimates["within_region_ssd"]
    a = df["attr"].to_numpy()
    total_ssd = float(((a - a.mean()) ** 2).sum())
    assert ssd < 0.5 * total_ssd                 # regions capture the between-block variance


def test_skater_degrades_too_few_points(tmp_path: Path) -> None:
    df = _two_block_df(side=2)                    # only 4 points
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "skater", tmp_path,
               config={"x": "gx", "y": "gy", "features": ["attr"], "n_clusters": 3})
    assert "跳过" in res.summary
    assert "n_regions" not in res.estimates
