import json

import pandas as pd

from researchforge.cleaning import (
    apply_cleaning_plan,
    make_cleaning_plan,
    write_cleaning_log,
)
from researchforge.profiler import profile_dataset


def _messy_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "const": [7, 7, 7, 7, 7],
            "g": ["x", "y", "x", "y", "z"],
        }
    )
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one duplicate row


def test_plan_from_diagnostics(tmp_path):
    csv = tmp_path / "messy.csv"
    _messy_df().to_csv(csv, index=False)

    plan = make_cleaning_plan(profile_dataset(csv))
    actions = {s.action for s in plan}

    assert "drop_duplicates" in actions
    assert "drop_column" in actions  # constant column
    assert "impute_median" in actions  # numeric missing


def test_apply_cleaning(tmp_path):
    df = _messy_df()
    csv = tmp_path / "messy.csv"
    df.to_csv(csv, index=False)
    plan = make_cleaning_plan(profile_dataset(csv))

    cleaned, log = apply_cleaning_plan(df, plan)

    assert "const" not in cleaned.columns
    assert cleaned["a"].isna().sum() == 0
    assert len(cleaned) < len(df)  # duplicate removed
    assert any(e["applied"] for e in log)

    log_path = write_cleaning_log(log, tmp_path / "out" / "cleaning_log.json")
    assert json.loads(log_path.read_text(encoding="utf-8"))
