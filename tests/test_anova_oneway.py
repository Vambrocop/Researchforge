"""Tests for the one-way ANOVA executor branch (experimental_stats family)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="anova_oneway", method="One-way ANOVA", domain="experimental design",
        family="experimental_stats", goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=6),
    )


def _run(tmp_path: Path, rows: list[dict], config: dict | None = None):
    csv = tmp_path / "a.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                        config=config if config is not None else {"outcome": "y", "group": "grp"})


def test_oneway_detects_group_difference(tmp_path: Path) -> None:
    # three groups with clearly separated means: A~10, B~10.3 (close to A), C~20 (far)
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("A", 10.0), ("B", 10.3), ("C", 20.0)]:
        rows += [{"y": mu + rng.normal(0, 1.0), "grp": g} for _ in range(25)]
    res = _run(tmp_path, rows)
    assert "完成" in res.summary
    e = res.estimates
    # omnibus highly significant
    assert e["p_value"] < 0.001
    assert e["f_stat"] > 0
    # effect-size keys present and sane
    assert 0.0 <= e["eta_squared"] <= 1.0
    assert e["omega_squared"] <= e["eta_squared"] + 1e-9  # ω² <= η²
    # diagnostics present
    assert "levene_p" in e and "welch_p" in e
    assert e["n_groups"] == 3.0
    # Tukey should flag the C vs (A,B) separation; C↔A and C↔B differ, A↔B does not
    assert ("C↔A" in res.summary or "A↔C" in res.summary)
    assert "tukey_hsd.csv" in res.files
    assert "group_stats.csv" in res.files


def test_oneway_null_negative_control(tmp_path: Path) -> None:
    # all groups share the same mean -> ANOVA should NOT reject
    rng = np.random.default_rng(7)
    rows = []
    for g in ["A", "B", "C"]:
        rows += [{"y": 5.0 + rng.normal(0, 1.0), "grp": g} for _ in range(30)]
    res = _run(tmp_path, rows)
    assert res.estimates["p_value"] > 0.05
    assert res.estimates["eta_squared"] < 0.1


def test_oneway_too_few_groups_degrades(tmp_path: Path) -> None:
    rows = [{"y": v, "grp": "only"} for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    res = _run(tmp_path, rows, config={"outcome": "y", "group": "grp"})
    assert "失败" in res.summary
