"""Tests for the `correspondence_analysis` executor branch (CA / MCA, prince).

Known structure: two categorical variables with a STRONG association (the second
variable's level is almost determined by the first) -> a significant chi-square test
and a first dimension that captures most of the inertia. Plus an MCA path (>=3
categorical variables), a degrade check (only one categorical column -> honest skip),
and a config-override check.

The catalog yaml exists (ordination.yaml) but the AnalysisEntry is built inline so
the test is self-contained.

prince must be installed for the happy-path tests; they skip cleanly if not.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_HAS_PRINCE = importlib.util.find_spec("prince") is not None

_ENTRY = AnalysisEntry(
    id="correspondence_analysis",
    method="Correspondence analysis (CA / MCA)",
    domain="statistics",
    family="ml",
    goal="explore",
    preconditions={"min_categorical_cols": 2, "min_rows": 4},
)


def _associated_csv(tmp_path: Path, n: int = 300) -> Path:
    """Two 3-level categoricals with a strong (near block-diagonal) association."""
    rng = np.random.default_rng(0)
    a_levels = ["red", "green", "blue"]
    # b is mostly determined by a (90% on-diagonal), with 10% noise.
    b_map = {"red": "north", "green": "south", "blue": "east"}
    all_b = ["north", "south", "east"]
    a, b = [], []
    for _ in range(n):
        av = rng.choice(a_levels)
        a.append(av)
        if rng.random() < 0.9:
            b.append(b_map[av])
        else:
            b.append(rng.choice(all_b))
    df = pd.DataFrame({"colour": a, "region": b})
    csv = tmp_path / "ca_assoc.csv"
    df.to_csv(csv, index=False)
    return csv


def _mca_csv(tmp_path: Path, n: int = 300) -> Path:
    """Three correlated 3-level categoricals (an MCA-shaped dataset)."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 3, n)
    def jitter(b):
        out = b.copy()
        flip = rng.random(n) < 0.15
        out[flip] = rng.integers(0, 3, flip.sum())
        return out
    name = {0: "low", 1: "mid", 2: "high"}
    df = pd.DataFrame({
        "v1": [name[x] for x in base],
        "v2": [name[x] for x in jitter(base)],
        "v3": [name[x] for x in jitter(base)],
    })
    csv = tmp_path / "mca.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.skipif(not _HAS_PRINCE, reason="prince not installed")
def test_ca_strong_association(tmp_path):
    csv = _associated_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"columns": ["colour", "region"]})
    out = Path(res.output_dir)

    assert (out / "ca_inertia.csv").exists(), "ca_inertia.csv missing"
    assert (out / "ca_row_coordinates.csv").exists()
    assert (out / "ca_chi_square.csv").exists(), "ca_chi_square.csv missing (CA path)"

    # chi-square significant for a strong association
    assert "chi_square_p" in res.estimates
    assert res.estimates["chi_square_p"] < 0.01, res.estimates
    assert res.estimates["chi_square"] > 0

    # first dimension captures most of the inertia
    assert "dim1_inertia_pct" in res.estimates
    assert res.estimates["dim1_inertia_pct"] > 40.0, res.estimates

    # inertia == chi-square / n (CA identity), cross-checked against the CSV
    chi = pd.read_csv(out / "ca_chi_square.csv")
    n_total = int(chi["n"].iloc[0])
    total_inertia = res.estimates["total_inertia"]
    assert abs(total_inertia - res.estimates["chi_square"] / n_total) < 1e-3, (
        "total inertia must equal chi-square / n in CA"
    )


@pytest.mark.skipif(not _HAS_PRINCE, reason="prince not installed")
def test_mca_three_variables(tmp_path):
    csv = _mca_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"columns": ["v1", "v2", "v3"]})
    out = Path(res.output_dir)

    assert (out / "ca_inertia.csv").exists()
    assert (out / "ca_row_coordinates.csv").exists()
    # MCA does NOT run the 2-variable chi-square test
    assert not (out / "ca_chi_square.csv").exists()
    assert "chi_square" not in res.estimates
    assert "dim1_inertia_pct" in res.estimates
    assert "MCA" in res.summary


@pytest.mark.skipif(not _HAS_PRINCE, reason="prince not installed")
def test_ca_autodetects_categoricals(tmp_path):
    """Without config the branch should pick the two categorical columns itself."""
    csv = _associated_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    out = Path(res.output_dir)
    assert (out / "ca_inertia.csv").exists()
    assert "chi_square_p" in res.estimates


def test_ca_one_categorical_skips(tmp_path):
    rng = np.random.default_rng(3)
    n = 50
    df = pd.DataFrame({
        "only_cat": rng.choice(["a", "b", "c"], n),
        "value": rng.normal(0, 1, n),
    })
    csv = tmp_path / "one_cat.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "dim1_inertia_pct" not in res.estimates
    assert "跳过" in res.summary
