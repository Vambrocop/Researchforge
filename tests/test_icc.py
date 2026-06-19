"""Tests for icc — Shrout & Fleiss (1979) intraclass correlations.

Cross-checks the engine against the canonical Shrout & Fleiss (1979) example
(6 targets x 4 judges) whose ICC values are published in the paper:
  ICC(1,1)=0.17  ICC(2,1)=0.29  ICC(3,1)=0.71
  ICC(1,k)=0.44  ICC(2,k)=0.62  ICC(3,k)=0.91   (k=4)
Also verifies an INDEPENDENT recomputation of the two-way ANOVA mean squares,
the F-test, config override, and the <2-rater / non-numeric honest skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="icc",
    method="Intraclass correlation (rater / measurement reliability)",
    domain="psychometrics",
    family="psychometrics",
    goal="describe",
    preconditions={"min_continuous": 2, "min_rows": 5},
)

# Shrout & Fleiss (1979), Table 1 — 6 targets (rows) x 4 judges (cols).
_SF_DATA = pd.DataFrame(
    {
        "j1": [9, 6, 8, 7, 10, 6],
        "j2": [2, 1, 4, 1, 5, 2],
        "j3": [5, 3, 6, 2, 6, 4],
        "j4": [8, 2, 8, 6, 9, 7],
    }
).astype(float)


def _icc_ref(Y: np.ndarray) -> dict[str, float]:
    """Independent ANOVA-decomposition reference for the six ICC forms."""
    n, k = Y.shape
    g = Y.mean()
    SSR = k * ((Y.mean(1) - g) ** 2).sum()
    SSC = n * ((Y.mean(0) - g) ** 2).sum()
    SST = ((Y - g) ** 2).sum()
    SSE = SST - SSR - SSC
    SSW = SST - SSR
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    MSW = SSW / (n * (k - 1))
    return {
        "icc_1_1": (MSR - MSW) / (MSR + (k - 1) * MSW),
        "icc_1_k": (MSR - MSW) / MSR,
        "icc_2_1": (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)),
        "icc_2_k": (MSR - MSE) / (MSR + (MSC - MSE) / n),
        "icc_3_1": (MSR - MSE) / (MSR + (k - 1) * MSE),
        "icc_3_k": (MSR - MSE) / MSR,
    }


def test_icc_shrout_fleiss_known_values(tmp_path: Path) -> None:
    csv = tmp_path / "sf.csv"
    _SF_DATA.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    # Published Shrout & Fleiss (1979) values (rounded to 2 dp in the paper).
    assert abs(res.estimates["icc_1_1"] - 0.17) < 0.02
    assert abs(res.estimates["icc_2_1"] - 0.29) < 0.02
    assert abs(res.estimates["icc_3_1"] - 0.71) < 0.02
    assert abs(res.estimates["icc_1_k"] - 0.44) < 0.02
    assert abs(res.estimates["icc_2_k"] - 0.62) < 0.02
    assert abs(res.estimates["icc_3_k"] - 0.91) < 0.02

    assert res.estimates["n_subjects"] == 6.0
    assert res.estimates["n_raters"] == 4.0


def test_icc_matches_independent_recompute(tmp_path: Path) -> None:
    csv = tmp_path / "sf.csv"
    _SF_DATA.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))

    ref = _icc_ref(_SF_DATA.to_numpy(float))
    for key, val in ref.items():
        assert abs(res.estimates[key] - val) < 1e-3, key


def test_icc_anova_table_and_ftest(tmp_path: Path) -> None:
    csv = tmp_path / "sf.csv"
    _SF_DATA.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "icc_anova.csv").exists()
    assert (out / "icc_estimates.csv").exists()
    anova = pd.read_csv(out / "icc_anova.csv")
    assert set(anova.columns) >= {"source", "SS", "df", "MS"}

    # F = MSR/MSE for the consistency form; with high ICC(3,1) it is significant.
    assert res.estimates["f_consistency"] > 1.0
    assert res.estimates["p_consistency"] < 0.05


def test_icc_perfect_agreement(tmp_path: Path) -> None:
    # Identical columns -> perfect reliability -> ICC near 1.
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 12)
    df = pd.DataFrame({"r1": base, "r2": base, "r3": base})
    csv = tmp_path / "perfect.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert res.estimates["icc_3_1"] > 0.99
    assert res.estimates["icc_2_1"] > 0.99


def test_icc_config_override(tmp_path: Path) -> None:
    df = _SF_DATA.copy()
    df["unrelated"] = np.arange(len(df), dtype=float) * 0.1
    csv = tmp_path / "extra.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp,
        _ENTRY,
        output_root=str(tmp_path / "o"),
        config={"items": ["j1", "j2", "j3", "j4"]},
    )
    assert res.estimates["n_raters"] == 4.0
    assert abs(res.estimates["icc_3_1"] - 0.71) < 0.02


def test_icc_too_few_raters_skips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"only_one": rng.normal(0, 1, 20), "label": ["a", "b"] * 10})
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "icc_3_1" not in res.estimates
    assert "跳过" in res.summary
    assert (Path(res.output_dir) / "report.md").exists()


def test_icc_non_numeric_skips(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": list("xyz") * 5, "b": list("pqr") * 5})
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "o"))
    assert "icc_3_1" not in res.estimates
    assert "跳过" in res.summary
