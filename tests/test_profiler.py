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


def test_bare_bounded_integer_without_temporal_name_is_not_time_col(tmp_path):
    """P3-1 regression guard: a rating/price-like integer column with
    duplicate values that merely happen to fall in the 1900-2100 range
    (e.g. an Elo rating) must NOT be misread as a time axis just because
    it lacks a temporal name anchor."""
    df = pd.DataFrame(
        {
            "player": [f"p{i}" for i in range(10)],
            "rating": [1950, 2000, 2050, 2000, 1975, 2020, 1950, 2010, 1990, 2005],
        }
    )
    csv = tmp_path / "ratings.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    assert fp.time_col is None
    assert fp.is_timeseries is False
    assert fp.is_panel is False


def test_name_anchored_bare_integer_year_still_detected(tmp_path):
    """A bare integer column with year-like values IS still picked up by the
    fallback when its name carries a temporal anchor (e.g. 'obs_year') --
    the P3-1 tightening only removes the no-name-check path."""
    df = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(5)] * 2,
            "obs_year": [2019] * 5 + [2020] * 5,
            "value": range(10),
        }
    )
    csv = tmp_path / "obs_year.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    assert fp.time_col == "obs_year"


def test_year_named_column_still_detected_as_time_col(tmp_path):
    """Regression: an exactly-named 'year' column with year-like values
    must still be detected as time_col after the P3-1 tightening."""
    df = pd.DataFrame({"year": [2018, 2019, 2020, 2018, 2019], "value": range(5)})
    csv = tmp_path / "year.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)

    assert fp.time_col == "year"
