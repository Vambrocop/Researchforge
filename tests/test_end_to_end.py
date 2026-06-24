"""End-to-end validation on REAL public datasets (v1.0 hardening).

Drives the full pipeline — profile -> recommend (select_top) -> run each top
recommendation -> report — on real datasets of different shapes (regression,
classification, time-series, panel), and asserts the engine NEVER crashes and
always produces a report + a non-empty summary. Methods may honestly "跳过" when
the data doesn't fit; that is a pass, not a failure. A raised exception is a fail.

Datasets are bundled with sklearn / statsmodels (no network, no repo bloat).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender import select_top


def _diabetes() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True).frame.rename(columns={"target": "progression"})


def _breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    # 569 x 30 features + a binary diagnosis target
    return load_breast_cancer(as_frame=True).frame


def _co2() -> pd.DataFrame:
    import statsmodels.api as sm

    df = sm.datasets.co2.load_pandas().data.reset_index()
    df.columns = ["date", "co2"]
    return df.dropna().reset_index(drop=True)


def _grunfeld() -> pd.DataFrame:
    import statsmodels.api as sm

    # classic investment panel: firm × year
    return sm.datasets.grunfeld.load_pandas().data


_DATASETS = {
    "diabetes_regression": (_diabetes, None, {"is_panel": False}),
    "breast_cancer_clf": (_breast_cancer, None, {}),
    "co2_timeseries": (_co2, None, {"is_timeseries": True}),
    "grunfeld_panel": (_grunfeld, "causal", {"is_panel": True}),
}


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_pipeline_runs_end_to_end_on_real_data(name, tmp_path: Path) -> None:
    builder, goal, expect = _DATASETS[name]
    df = builder()
    csv = tmp_path / f"{name}.csv"
    df.to_csv(csv, index=False)

    # 1. profile
    fp = profile_dataset(csv)
    assert fp.n_rows == len(df) and fp.n_cols == df.shape[1]
    for k, v in expect.items():
        assert getattr(fp, k) == v, f"{name}: expected fp.{k}=={v}, got {getattr(fp, k)}"

    # 2. recommend
    recs = select_top(fp, top=6, goal=goal)
    assert recs, f"{name}: no recommendations produced"

    # 3. run every top recommendation end-to-end — must never raise
    crashes = []
    for r in recs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_analysis(fp, r.entry, output_root=str(tmp_path / "out"))
            assert Path(res.report_path).exists(), f"{r.entry.id}: no report written"
            assert isinstance(res.summary, str) and res.summary.strip(), f"{r.entry.id}: empty summary"
        except Exception as e:  # noqa: BLE001 — a crash is exactly what we're guarding against
            crashes.append(f"{r.entry.id}: {type(e).__name__}: {e}")
    assert not crashes, f"{name}: analyses crashed on real data:\n" + "\n".join(crashes)


def test_recommend_explains_every_pick(tmp_path: Path) -> None:
    """Each recommendation carries a rigor light + a methodology score (the engine
    always explains WHY), on real data."""
    df = _diabetes()
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    for r in select_top(fp, top=8):
        assert r.rigor.light in {"green", "yellow", "red"}
        assert isinstance(r.rigor.note, str) and r.rigor.note
        assert 0 <= r.score.overall <= 100
