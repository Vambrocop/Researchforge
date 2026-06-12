from researchforge.synth import make_panel


def test_make_panel_shape_and_columns():
    df = make_panel(n_units=4, n_periods=5, treated=True, seed=1)
    assert len(df) == 20
    assert set(df.columns) == {"unit", "year", "y", "treated"}
    assert df["treated"].isin([0, 1]).all()
    assert df["year"].nunique() == 5
    assert df["unit"].nunique() == 4


def test_make_panel_reproducible():
    a = make_panel(seed=42)
    b = make_panel(seed=42)
    assert a.equals(b)
