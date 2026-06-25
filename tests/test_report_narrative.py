"""Tests for the report-intelligence narrative (presentation only).

``build_narrative`` re-organises the already-computed ``summary`` (findings vs ⚠
caveats), maps goal codes to Chinese, and emits honest next-steps — it never
recomputes a number. We assert the headline, the 解读 heading, that the ⚠ caveat
is surfaced (not buried), and that a next-step suggestion appears. We also check
``_report`` stays additive: ALL original sections remain AND the new 解读 section
is present, and that an empty-summary case doesn't crash.
"""

from __future__ import annotations

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor.run import _report
from researchforge.executor._helpers.report_narrative import build_narrative
from researchforge.profiler.fingerprint import ColumnInfo, DataFingerprint


def _entry(**kw) -> AnalysisEntry:
    base = dict(
        id="ols_regression",
        method="OLS 回归",
        domain="statistics",
        family="statistics",
        goal="explain",
        preconditions=Precondition(min_continuous=1, min_rows=10),
        biases=["遗漏变量偏差", "测量误差"],
    )
    base.update(kw)
    return AnalysisEntry(**base)


def _fp(path: str = "/data/sales.csv") -> DataFingerprint:
    return DataFingerprint(
        path=path,
        n_rows=120,
        n_cols=4,
        columns=[
            ColumnInfo(name="y", kind="continuous", dtype="float64", n_missing=0, n_unique=120),
            ColumnInfo(name="x1", kind="continuous", dtype="float64", n_missing=0, n_unique=119),
            ColumnInfo(name="g", kind="categorical", dtype="object", n_missing=0, n_unique=3),
        ],
        likely_outcome="y",
    )


def test_build_narrative_core_pieces() -> None:
    summary = [
        "R² = 0.62，x1 系数 1.30 (p<0.001)。",
        "g 组间差异不显著。",
        "⚠ 残差存在异方差迹象，标准误可能被低估。",
    ]
    lines = build_narrative(_entry(), _fp(), summary, override=False)
    md = "\n".join(lines)

    # heading + headline (data file name, N x M, method, goal in Chinese)
    assert "## 解读（自动生成）" in md
    assert "sales.csv" in md
    assert "120 行 × 4 列" in md
    assert "OLS 回归" in md
    assert "解释" in md  # goal=explain -> 解释
    # lead finding surfaced as the key finding
    assert "关键发现" in md
    assert "R² = 0.62" in md
    # the ⚠ caveat is surfaced in a consolidated 注意/局限 list, not buried
    assert "注意 / 局限" in md
    assert "残差存在异方差迹象" in md
    # next-steps section present with a concrete suggestion
    assert "建议下一步" in md
    assert "稳健" in md or "vif" in md or "heteroskedasticity" in md


def test_build_narrative_override_caution() -> None:
    summary = ["效应估计 = 0.4。"]
    lines = build_narrative(_entry(), _fp(), summary, override=True)
    md = "\n".join(lines)
    # override -> indicative-not-confirmatory caution + a precondition-fix next step
    assert "提示性" in md
    assert "前提" in md


def test_build_narrative_empty_summary_does_not_crash() -> None:
    lines = build_narrative(_entry(), _fp(), [], override=False)
    md = "\n".join(lines)
    assert "## 解读（自动生成）" in md
    assert "未产出可解读的结果行" in md
    # still emits next steps (honest default)
    assert "建议下一步" in md


def test_build_narrative_unknown_goal_graceful() -> None:
    lines = build_narrative(_entry(goal="zzz_unknown"), _fp(), ["something"], override=False)
    md = "\n".join(lines)
    # unknown goal falls back to the raw code, doesn't crash, still has a section
    assert "## 解读（自动生成）" in md
    assert "建议下一步" in md


def test_report_is_additive_all_sections_present() -> None:
    summary = [
        "R² = 0.62。",
        "⚠ 样本量偏小，结论需谨慎。",
    ]
    files = ["ols_regression.csv", "ols_regression.png"]
    md = _report(_entry(), _fp(), summary, files, override=False)

    # ALL original sections, in order, unchanged
    assert "# ResearchForge 分析报告：OLS 回归" in md
    assert "- 数据：`/data/sales.csv`（120 行 × 4 列）" in md
    assert "- 分析：OLS 回归（statistics / explain）" in md
    assert "## 结果摘要" in md
    assert "## 偏差提醒（需读者判断）" in md
    assert "- 遗漏变量偏差" in md
    assert "## 产物文件" in md
    assert "- `ols_regression.csv`" in md

    # the NEW 解读 section is inserted right after 结果摘要 and before 偏差提醒
    assert "## 解读（自动生成）" in md
    i_summary = md.index("## 结果摘要")
    i_narr = md.index("## 解读（自动生成）")
    i_bias = md.index("## 偏差提醒（需读者判断）")
    i_files = md.index("## 产物文件")
    assert i_summary < i_narr < i_bias < i_files

    # caveat from summary is surfaced inside the narrative
    assert "样本量偏小" in md


def test_report_override_banner_and_narrative() -> None:
    md = _report(_entry(), _fp(), ["估计 = 1.0。"], ["o.csv"], override=True)
    # original override banner still present
    assert "知情覆盖" in md
    # AND the narrative override caution
    assert "提示性" in md


def test_report_empty_summary_does_not_crash() -> None:
    md = _report(_entry(biases=[]), _fp(), [], [], override=False)
    assert "## 结果摘要" in md
    assert "## 解读（自动生成）" in md
    assert "## 产物文件" in md
    # no biases -> no 偏差提醒 section (unchanged behaviour)
    assert "## 偏差提醒" not in md
