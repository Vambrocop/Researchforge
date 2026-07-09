"""Tests for study mode (researchforge/study.py + researchforge/study_report.py).

Per docs/design-study-mode.md §4: 3 e2e synthetic datasets (regression-shaped /
binary-outcome / count-outcome) asserting the study runs end to end and the report
has the right shape; 1 failure-injection asserting honest partial-failure
reporting; 1 --clean path asserting the cleaning disclosure reaches §0. A few
extra fast unit tests pin down the diversity filter and the pure cross-method
convergence rule directly (deterministic, independent of what the live catalog
happens to rank top on any given dataset).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# synthetic data (n<=200, seeded — keep the e2e tests fast per §4)
# --------------------------------------------------------------------------- #
def _regression_df(seed: int = 0, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y = 3.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y.round(3), "x1": x1.round(3), "x2": x2.round(3)})


def _binary_df(seed: int = 1, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.3 + 1.1 * x1 - 0.6 * x2)))
    y = rng.binomial(1, p)
    return pd.DataFrame({"approved": y, "income": x1.round(3), "score": x2.round(3)})


def _count_df(seed: int = 2, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    lam = np.exp(0.5 + 0.4 * x1)
    y = rng.poisson(lam)
    return pd.DataFrame(
        {"visits": y, "age": rng.normal(40, 8, n).round(2), "x1": x1.round(3)}
    )


def _assert_report_shape(result: dict, top: int) -> None:
    assert Path(result["report_path"]).exists()
    text = result["report_text"]
    assert text == Path(result["report_path"]).read_text(encoding="utf-8")
    for heading in (
        "## §0 数据与质量", "### 描述性统计基线", "## §选法依据",
        "## §跨方法收敛信号", "## §方法学附录", "## §披露汇总",
    ):
        assert heading in text, f"missing section: {heading}"
    for i in range(1, len(result["meta"]["methods"]) + 1):
        assert f"## §{i} " in text

    meta = result["meta"]
    for key in (
        "engine_version", "data_path", "data_sha256", "n_rows", "n_cols", "goal",
        "top_requested", "methods", "baseline", "config", "clean_applied", "study_dir",
    ):
        assert key in meta, f"missing meta key: {key}"
    assert meta["top_requested"] == top
    assert len(meta["methods"]) <= top
    assert (Path(result["study_dir"]) / "study_meta.json").exists()
    assert (Path(result["study_dir"]) / "study_report.md").exists()


@pytest.mark.parametrize("builder", [_regression_df, _binary_df, _count_df])
def test_study_e2e(builder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "data.csv"
    builder().to_csv(csv, index=False)

    from researchforge.study import run_study

    result = run_study(str(csv), goal=None, top=3, clean=False, config=None)
    assert result["methods_run"], "study produced zero successful methods"
    _assert_report_shape(result, top=3)
    # every method that "ran" must correspond to a real, existing run directory
    for mid in result["methods_run"]:
        entry = next(m for m in result["meta"]["methods"] if m["id"] == mid)
        assert entry["status"] == "ok"
        assert entry["run_dir"] and Path(entry["run_dir"]).exists()


def test_study_diversity_pick_prefers_distinct_families():
    from researchforge.study import _diversity_pick

    def rec(rid: str, family: str):
        return SimpleNamespace(entry=SimpleNamespace(id=rid, family=family))

    recs = [rec("a", "f1"), rec("b", "f1"), rec("c", "f2"), rec("d", "f3")]
    chosen = _diversity_pick(recs, 3)
    assert [r.entry.id for r in chosen] == ["a", "c", "d"]  # 1-per-family, ranked order


def test_study_diversity_pick_backfills_when_families_short():
    from researchforge.study import _diversity_pick

    def rec(rid: str, family: str):
        return SimpleNamespace(entry=SimpleNamespace(id=rid, family=family))

    # only one family available -> can't fill 3 distinct families; backfill by rank
    recs = [rec("a", "f1"), rec("b", "f1"), rec("c", "f1")]
    chosen = _diversity_pick(recs, 3)
    assert [r.entry.id for r in chosen] == ["a", "b", "c"]
    # still honours the cap even with plenty of candidates
    chosen2 = _diversity_pick(recs, 2)
    assert [r.entry.id for r in chosen2] == ["a", "b"]


def test_convergence_section_reports_shared_key_pure_rule():
    from researchforge.study_report import _convergence_section

    class _Res:
        def __init__(self, estimates):
            self.estimates = estimates

    run_entries = [
        {"rec": SimpleNamespace(entry=SimpleNamespace(id="m1")), "result": _Res({"x1": 1.0, "n": 100.0})},
        {"rec": SimpleNamespace(entry=SimpleNamespace(id="m2")), "result": _Res({"x1": 1.2, "n": 100.0})},
        # an orchestration-failed method (no RunResult) must be silently skipped
        {"rec": SimpleNamespace(entry=SimpleNamespace(id="m3")), "result": None},
    ]
    text = "\n".join(_convergence_section(run_entries))
    assert "`x1`" in text and "符号一致" in text and "量级一致" in text


def test_convergence_section_no_shared_keys_is_honest():
    from researchforge.study_report import _convergence_section

    class _Res:
        def __init__(self, estimates):
            self.estimates = estimates

    run_entries = [
        {"rec": SimpleNamespace(entry=SimpleNamespace(id="m1")), "result": _Res({"a": 1.0})},
        {"rec": SimpleNamespace(entry=SimpleNamespace(id="m2")), "result": _Res({"b": 2.0})},
    ]
    text = "\n".join(_convergence_section(run_entries))
    assert "不做数值横比" in text
    assert "`a`" not in text and "`b`" not in text  # no fabricated comparison


def test_study_failure_injection(tmp_path, monkeypatch):
    """One method blows up at the ORCHESTRATION level (study.py's own try/except,
    not run_analysis's internal handler catch) -> its section says failure, the
    rest of the study completes, and methods_run/meta reflect it honestly."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "data.csv"
    _regression_df().to_csv(csv, index=False)

    import researchforge.study as study_mod

    real_run_analysis = study_mod.run_analysis
    boom = {"id": None}

    def flaky(fp, entry, output_root=None, config=None):
        if boom["id"] is None and entry.id != "descriptive_stats":
            boom["id"] = entry.id
            raise RuntimeError("boom: injected failure")
        return real_run_analysis(fp, entry, output_root=output_root, config=config)

    monkeypatch.setattr(study_mod, "run_analysis", flaky)

    result = study_mod.run_study(str(csv), top=3)
    assert boom["id"] is not None, "fixture never reached a substantive method"

    text = result["report_text"]
    assert "执行失败" in text
    assert "boom: injected failure" in text
    assert f"`{boom['id']}`" in text
    # honest bookkeeping: the exploded method is NOT in methods_run...
    assert boom["id"] not in result["methods_run"]
    # ...but IS recorded in meta with an orchestration_failed status + the error
    meta_methods = {m["id"]: m for m in result["meta"]["methods"]}
    assert boom["id"] in meta_methods
    assert meta_methods[boom["id"]]["status"] == "orchestration_failed"
    assert meta_methods[boom["id"]]["run_dir"] is None
    assert "boom: injected failure" in meta_methods[boom["id"]]["error"]
    # the rest of the study is NOT sunk by the one failure
    assert len(result["methods_run"]) == len(meta_methods) - 1
    assert "§跨方法收敛信号" in text and "§披露汇总" in text
    assert Path(result["report_path"]).exists()


def test_study_clean_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(9)
    n = 150
    x1 = rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y.round(3), "x1": x1.round(3)})
    df["const"] = 1  # constant column -> drop_column
    df.loc[:5, "x1"] = np.nan  # missing numeric -> impute_median
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)  # duplicate rows -> drop_duplicates
    csv = tmp_path / "dirty.csv"
    df.to_csv(csv, index=False)

    from researchforge.study import run_study

    result = run_study(str(csv), top=2, clean=True, config=None)
    assert result["meta"]["clean_applied"] is True
    assert "`--clean` 已应用" in result["report_text"]
    assert "✓ drop_duplicates" in result["report_text"] or "drop_duplicates" in result["report_text"]
    study_dir = Path(result["study_dir"])
    assert (study_dir / "cleaned_data.csv").exists()
    assert (study_dir / "cleaned_data.cleaning.json").exists()
    # the const column must be gone from the data actually analyzed
    cleaned = pd.read_csv(study_dir / "cleaned_data.csv")
    assert "const" not in cleaned.columns
