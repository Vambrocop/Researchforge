import pandas as pd

from researchforge.profiler.quality import diagnose


def test_diagnose_missing_constant_duplicate():
    df = pd.DataFrame({"a": [1, 1, None, 4], "b": ["x", "x", "x", "x"], "c": [1, 2, 3, 4]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # introduce a duplicate row

    kinds = {i.kind for i in diagnose(df)}

    assert "missing" in kinds
    assert "constant" in kinds  # column b
    assert "duplicate_rows" in kinds


def test_diagnose_clean_data_has_no_issues():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["p", "q", "r", "s"]})
    assert diagnose(df) == []
