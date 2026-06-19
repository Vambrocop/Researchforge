"""Tests for the `discriminant_analysis` executor branch (LDA & QDA).

Synthetic data with KNOWN structure:
- well-separated classes -> high CV accuracy (>> chance).
- confusable (overlapping) classes -> CV accuracy near chance.
Plus precondition/degrade checks (1 predictor, 1 class -> honest skip).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="discriminant_analysis",
    method="Discriminant analysis (LDA & QDA)",
    domain="statistics",
    family="statistics",
    goal="explain",
    preconditions={"requires_group": True, "min_continuous": 2, "min_rows": 20},
)


def _separated_csv(tmp_path: Path, n_per: int = 60, sep: float = 4.0) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for g, mu in [("A", [0, 0]), ("B", [sep, sep]), ("C", [sep, -sep])]:
        for _ in range(n_per):
            rows.append({
                "x1": rng.normal(mu[0], 1.0),
                "x2": rng.normal(mu[1], 1.0),
                "cls": g,
            })
    csv = tmp_path / "sep.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _confusable_csv(tmp_path: Path, n_per: int = 60) -> Path:
    """Two classes with identical distribution -> features carry no signal."""
    rng = np.random.default_rng(7)
    rows = []
    for g in ["A", "B"]:
        for _ in range(n_per):
            rows.append({
                "x1": rng.normal(0, 1),
                "x2": rng.normal(0, 1),
                "x3": rng.normal(0, 1),
                "cls": g,
            })
    csv = tmp_path / "conf.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_discriminant_well_separated_high_accuracy(tmp_path):
    csv = _separated_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"predictors": ["x1", "x2"], "group": "cls"})
    out = Path(res.output_dir)

    assert (out / "lda_confusion_matrix.csv").exists()
    assert "lda_cv_accuracy" in res.estimates
    assert "qda_cv_accuracy" in res.estimates
    # 3 well-separated classes -> CV accuracy should be very high (> 0.9).
    assert res.estimates["lda_cv_accuracy"] > 0.9, res.estimates
    # explained-discriminant ratio present (multi-class LDA has >= 1 axis).
    assert (out / "lda_explained_variance.csv").exists()
    assert "ld1_explained_ratio" in res.estimates


def test_discriminant_confusable_near_chance(tmp_path):
    csv = _confusable_csv(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"predictors": ["x1", "x2", "x3"], "group": "cls"})
    chance = res.estimates["chance_accuracy"]
    # No signal -> CV accuracy should be near the chance baseline (within ~0.15).
    assert res.estimates["lda_cv_accuracy"] < chance + 0.15, res.estimates


def test_discriminant_one_predictor_skips(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "cls": ["A", "B"] * (n // 2),
    })
    csv = tmp_path / "one_pred.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"))
    assert "lda_cv_accuracy" not in res.estimates
    assert "跳过" in res.summary


def test_discriminant_one_class_skips(tmp_path):
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "cls": ["A"] * n,
    })
    csv = tmp_path / "one_cls.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _ENTRY, output_root=str(tmp_path / "out"),
                       config={"predictors": ["x1", "x2"], "group": "cls"})
    assert "lda_cv_accuracy" not in res.estimates
    assert "跳过" in res.summary
