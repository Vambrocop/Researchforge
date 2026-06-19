"""Tests for the `manova` executor branch (multivariate ANOVA).

Synthetic data with KNOWN structure:
- groups with clearly shifted DV means  -> Wilks' Lambda p < 0.05 (reject equality).
- groups drawn from the SAME distribution -> p well above 0.05 (do not reject).
Plus precondition/degrade checks (only 1 DV, only 1 group -> honest skip).

Catalog yaml exists (multivariate.yaml) but the entry is constructed inline so the
test is self-contained.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="manova",
    method="Multivariate analysis of variance (MANOVA)",
    domain="statistics",
    family="statistics",
    goal="explain",
    preconditions={"requires_group": True, "min_continuous": 2, "min_rows": 12},
)


def _shifted_csv(tmp_path: Path, n_per: int = 40, shift: float = 3.0) -> Path:
    """Three DVs, two groups whose mean vectors are clearly separated."""
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("A", 0.0), ("B", shift)]:
        for _ in range(n_per):
            rows.append({
                "y1": rng.normal(mu, 1.0),
                "y2": rng.normal(mu, 1.0),
                "y3": rng.normal(0.0, 1.0),  # null DV
                "grp": g,
            })
    csv = tmp_path / "manova_shift.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _null_csv(tmp_path: Path, n_per: int = 40) -> Path:
    """Two groups drawn from the SAME distribution — no real effect."""
    rng = np.random.default_rng(123)
    rows = []
    for g in ["A", "B"]:
        for _ in range(n_per):
            rows.append({
                "y1": rng.normal(0, 1),
                "y2": rng.normal(0, 1),
                "y3": rng.normal(0, 1),
                "grp": g,
            })
    csv = tmp_path / "manova_null.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_manova_shifted_means_significant(tmp_path):
    csv = _shifted_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["y1", "y2", "y3"], "group": "grp"})
    out = Path(res.output_dir)

    assert (out / "manova_tests.csv").exists()
    assert (out / "manova_group_means.csv").exists()

    # All four multivariate stats present.
    for key in ["wilks_p", "pillai_p", "hotelling_lawley_p", "roy_p"]:
        assert key in res.estimates, f"{key} missing"

    # Clearly shifted means -> Wilks (and Pillai) reject.
    assert res.estimates["wilks_p"] < 0.05, res.estimates
    assert res.estimates["pillai_p"] < 0.05, res.estimates

    tbl = pd.read_csv(out / "manova_tests.csv")
    assert set(tbl["statistic"]) >= {
        "Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root",
    }


def test_manova_null_not_significant(tmp_path):
    csv = _null_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["y1", "y2", "y3"], "group": "grp"})
    # Same distribution -> do not reject.
    assert res.estimates["pillai_p"] > 0.05, res.estimates


def test_manova_one_dv_skips(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "y1": rng.normal(0, 1, n),
        "grp": (["A", "B"] * (n // 2)),
    })
    csv = tmp_path / "one_dv.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "wilks_p" not in res.estimates
    assert "跳过" in res.summary


def test_manova_one_group_skips(tmp_path):
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame({
        "y1": rng.normal(0, 1, n),
        "y2": rng.normal(0, 1, n),
        "grp": ["A"] * n,  # single level
    })
    csv = tmp_path / "one_grp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"outcomes": ["y1", "y2"], "group": "grp"})
    assert "wilks_p" not in res.estimates
    assert "跳过" in res.summary
