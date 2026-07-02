import pandas as pd
import pytest

from researchforge.profiler import infer_kind


@pytest.mark.parametrize(
    "series, expected",
    [
        # 2-distinct-value integer columns (with repeats) -> binary, regardless of pair
        (pd.Series([1, 2, 1, 2, 1], name="grp"), "binary"),
        (pd.Series([-1, 1, -1, 1, -1], name="sign"), "binary"),
        (pd.Series([0, 100, 0, 100, 0], name="flag100"), "binary"),
        (pd.Series([0, 1, 1, 0], name="t"), "binary"),  # regression: {0,1} still binary
        # unique-integer column (no repeats) -> id, not binary
        (pd.Series([1, 2, 3, 4], name="code"), "id"),
        # 3-distinct-value non-negative int column -> count, unchanged
        (pd.Series([0, 1, 2, 3, 2, 1], name="c"), "count"),
        # 2-distinct-value FLOAT column -> continuous, not binary
        (pd.Series([1.5, 2.5, 1.5, 2.5], name="f"), "continuous"),
    ],
)
def test_infer_kind_binary_encoding(series, expected):
    assert infer_kind(series) == expected
