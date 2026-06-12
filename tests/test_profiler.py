import pandas as pd
import pytest

from researchforge.profiler import infer_kind, profile_dataset
from researchforge.synth import make_panel


@pytest.mark.parametrize(
    "series, expected",
    [
        (pd.Series([0, 1, 1, 0], name="t"), "binary"),
        (pd.Series([1.5, 2.3, 3.1, 4.2], name="x"), "continuous"),
        (pd.Series([0, 1, 2, 3, 2, 1], name="c"), "count"),
        (pd.Series(["A1", "A2", "A3", "A4"], name="code"), "id"),
        (pd.Series(["x", "y", "x", "y", "z"], name="g"), "categorical"),
        (pd.Series(["2020-01-01", "2020-02-01", "2020-03-01"], name="d"), "datetime"),
    ],
)
def test_infer_kind(series, expected):
    assert infer_kind(series) == expected


def test_profile_detects_panel(tmp_path):
    df = make_panel(n_units=5, n_periods=6, treated=True, seed=3)
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    assert fp.n_rows == 30
    assert fp.is_panel is True
    assert fp.unit_col == "unit"
    assert fp.time_col == "year"
    assert "treated" in fp.treatment_candidates
    assert fp.column("y").kind == "continuous"
