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


def _rare_tail_df(seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 200
    common = rng.choice(["A", "B", "C"], 150)
    rare = [f"r{i}" for i in range(25)]  # ~25 singleton/near-singleton levels
    grp = list(common) + list(rng.choice(rare, 50))
    rng.shuffle(grp)
    region = rng.choice(["N", "S", "E", "W"], n)  # balanced 4-level: must NOT collapse
    uid = [f"u{i}" for i in range(n)]  # near-unique text: high_cardinality
    return pd.DataFrame({"grp": grp, "region": region, "uid": uid,
                         "val": rng.normal(0, 1, n).round(3)})


def test_rare_categories_detected_and_collapsed(tmp_path):
    df = _rare_tail_df()
    csv = tmp_path / "rare.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)

    kinds = {i.kind for i in fp.issues}
    assert "rare_categories" in kinds  # the long-tailed 'grp'
    assert "high_cardinality" in kinds  # the near-unique 'uid'

    plan = make_cleaning_plan(fp)
    actions = {(s.action, s.column) for s in plan}
    assert ("collapse_rare", "grp") in actions
    assert ("flag_high_cardinality", "uid") in actions

    cleaned, log = apply_cleaning_plan(df, plan)
    # rare tail collapsed into 'Other' -> fewer levels, and 'Other' present
    assert cleaned["grp"].nunique() < df["grp"].nunique()
    assert "Other" in set(cleaned["grp"])
    # GUARDS: balanced few-level factor untouched; near-unique id left alone (advisory only)
    assert set(cleaned["region"]) == set(df["region"])
    assert cleaned["uid"].nunique() == df["uid"].nunique()
    hc = next(e for e in log if e["action"] == "flag_high_cardinality")
    assert hc["applied"] is False


def test_apply_rare_threshold_knob_collapses_more_aggressively(tmp_path):
    # detection (profiler/quality.py) stays fixed at 1%; this only tunes how
    # aggressively `apply_cleaning_plan` collapses once rare_categories is flagged.
    import re

    df = _rare_tail_df()  # seed=0
    csv = tmp_path / "rare_knob.csv"
    df.to_csv(csv, index=False)
    plan = make_cleaning_plan(profile_dataset(csv))

    _, log_default = apply_cleaning_plan(df, plan)  # default rare_threshold=0.01
    _, log_aggressive = apply_cleaning_plan(df, plan, rare_threshold=0.05)

    default_entry = next(e for e in log_default if e["action"] == "collapse_rare")
    aggressive_entry = next(e for e in log_aggressive if e["action"] == "collapse_rare")
    assert default_entry["applied"] and aggressive_entry["applied"]

    def _n_collapsed(detail: str) -> int:
        m = re.search(r"collapsed (\d+) rare levels", detail)
        return int(m.group(1)) if m else 0

    assert _n_collapsed(aggressive_entry["detail"]) > _n_collapsed(default_entry["detail"])


def test_balanced_categorical_not_collapsed(tmp_path):
    # a plain low-cardinality categorical with NO rare tail must not trip collapse_rare
    import numpy as np

    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"sector": rng.choice(["a", "b", "c"], n),
                       "y": rng.normal(0, 1, n).round(3)})
    csv = tmp_path / "bal.csv"
    df.to_csv(csv, index=False)
    plan = make_cleaning_plan(profile_dataset(csv))
    assert not any(s.action == "collapse_rare" for s in plan)
