"""Report intelligence — an auto-generated, analyst-style narrative section.

PRESENTATION only. This module produces extra markdown lines for the analysis
report's "## 解读（自动生成）" section. It NEVER computes or alters any statistic:
it only re-organises and contextualises what's already in ``summary`` + the
catalog ``entry`` + the data ``fingerprint``. The narrative reads like a junior
analyst's read-out: a one-line headline, how to interpret it, what to look at
(findings vs caveats), and a couple of honest next steps.

``build_narrative`` must NEVER raise — a narrative failure must not break report
generation. The single public caller (``core._report``) wraps it defensively too,
but we belt-and-braces it here as well so callers can use it directly.
"""

from __future__ import annotations

import os

# goal code -> 中文意思 (graceful fallback for unknown codes).
_GOAL_ZH: dict[str, str] = {
    "describe": "描述",
    "explain": "解释",
    "predict": "预测",
    "compare": "比较",
    "evaluate": "评估",
    "explore": "探索",
    "relate": "关联",
    "reduce": "降维",
    "synthesize": "综合",
    "forecast": "预报",
    "classify": "分类",
    "cluster": "聚类",
    "test": "检验",
    "monitor": "监测",
    "optimize": "优化",
}

# What the family/goal broadly answers — a short interpretive lead.
_GOAL_READ: dict[str, str] = {
    "describe": "回答「数据呈现什么样貌」——刻画分布、结构与基本特征，不做因果断言。",
    "explain": "回答「哪些因素与结果相关、方向与强度如何」——是关联性解释，非随机实验下需谨慎谈因果。",
    "predict": "回答「能否据已知变量预测结果」——重点在样本外泛化，而非系数的因果含义。",
    "compare": "回答「不同组别/条件间是否存在差异」——给出差异方向与显著性。",
    "evaluate": "回答「各方案/单元的表现或效率如何」——产出可比的评分或排序。",
    "explore": "回答「数据里潜藏哪些结构或模式」——探索性，结论需后续确证。",
    "relate": "回答「变量之间如何共变」——刻画相关结构，相关不等于因果。",
    "reduce": "回答「能否用更少维度概括信息」——压缩冗余、提取主要成分/因子。",
    "synthesize": "回答「多项研究证据合起来说明什么」——汇总效应并量化异质性。",
    "forecast": "回答「未来取值的走势如何」——给出预测与不确定区间。",
    "classify": "回答「样本应归入哪一类」——产出类别判定与判别表现。",
    "cluster": "回答「样本可自然分成几群」——无监督分组，群数与解释需斟酌。",
    "test": "回答「某个假设是否成立」——给出检验统计量与显著性。",
    "monitor": "回答「过程是否处于受控状态」——识别异常与超限信号。",
    "optimize": "回答「在约束下如何取最优」——产出最优方案或权重。",
}

# family -> 建议下一步 (suggested next steps). Honest + generic; prefer well-known
# ids or phrase as a capability. Falls back to ``_DEFAULT_STEPS``.
_FAMILY_STEPS: dict[str, list[str]] = {
    "statistics": [
        "做稳健性检查：残差异方差 (heteroskedasticity_test) 与多重共线性 (vif_multicollinearity)。",
        "若残差或离群点可疑，换稳健替代 (robust_regression / quantile_regression) 复核结论。",
        "用 bootstrap_ci 给关键估计配上更稳健的置信区间。",
    ],
    "econometrics": [
        "检查识别假设是否站得住（内生性/平行趋势/外生工具），必要时换更贴切的设定。",
        "做稳健性检查并报告聚类/稳健标准误。",
        "对关键估计用 bootstrap_ci 复核区间。",
    ],
    "causal": [
        "做敏感性分析量化未观测混杂的影响（如 E-value / 安慰剂检验）。",
        "尝试更贴近设计的因果方法（如 双重稳健 / 工具变量）作交叉验证。",
        "明确并复核可识别性假设，再下因果结论。",
    ],
    "panel": [
        "对比固定效应 vs 随机效应（Hausman 思路），并报告聚类标准误。",
        "做稳健性检查：序列相关与异方差。",
        "检查识别假设（如平行趋势）是否成立。",
    ],
    "ml": [
        "做交叉验证评估样本外表现，避免过拟合 (cross-validation)。",
        "用 conformal_prediction 给预测配上可信区间。",
        "结合可解释性方法（特征重要度 / SHAP）核对模型逻辑。",
    ],
    "ml_supervised": [
        "做交叉验证评估样本外表现 (cross-validation)。",
        "用 conformal_prediction 量化预测不确定性。",
        "用可解释性方法核对关键特征的贡献方向。",
    ],
    "interpretability": [
        "对照另一种解释方法交叉验证特征重要度的稳定性。",
        "结合领域知识审视高贡献特征是否合理。",
        "在不同子样本上复核解释结论的稳健性。",
    ],
    "forecasting": [
        "用留出/滚动回测评估预测精度（如 MAE/MAPE）。",
        "检查残差是否仍有自相关 (acf_pacf)。",
        "对比基线模型确认增量价值。",
    ],
    "timeseries": [
        "检查平稳性与残差自相关 (acf_pacf)，必要时差分或换设定。",
        "用滚动回测评估样本外预测精度。",
        "对比基线模型确认增量价值。",
    ],
    "time_series": [
        "检查平稳性与残差自相关 (acf_pacf)。",
        "用滚动回测评估样本外预测精度。",
        "对比基线模型确认增量价值。",
    ],
    "survival": [
        "检验比例风险假设是否成立 (cox_ph_diagnostics)。",
        "审视删失是否随机/非信息性，否则结论会有偏。",
        "做敏感性分析考察对时长/事件列设定的稳健性。",
    ],
    "meta": [
        "用调节因素 (meta_regression) 解释异质性来源。",
        "复核发表偏倚（漏斗图 / Egger 检验）。",
        "做留一法敏感性分析，看单个研究的影响。",
    ],
    "sem": [
        "用理论驱动的测量/结构模型替换自动模板，再复核拟合。",
        "报告并审视拟合指数 (CFI/TLI/RMSEA) 是否达标。",
        "在独立样本上做交叉验证以防过拟合。",
    ],
    "nonparametric": [
        "若样本量允许，用对应的参数方法对照结论。",
        "用 bootstrap_ci 给关键统计量配置信区间。",
        "检查离群点对结果的影响。",
    ],
    "spatial": [
        "检查空间自相关是否被充分建模 (空间残差诊断)。",
        "做稳健性检查：对邻接/权重设定的敏感性。",
        "审视边界效应对结论的影响。",
    ],
    "mcda": [
        "做权重敏感性分析，看排序对权重设定的稳健性。",
        "对比另一种聚合方法 (如 TOPSIS/AHP) 交叉验证排序。",
        "明确并复核准则方向与归一化方式。",
    ],
    "efficiency": [
        "做 bootstrap 校正效率得分的偏倚与置信区间。",
        "对投入产出变量做敏感性分析。",
        "对比另一种前沿方法 (DEA / SFA) 交叉验证。",
    ],
    "bayesian": [
        "检查后验收敛 (R-hat / 有效样本量) 与先验敏感性。",
        "做后验预测检查 (posterior predictive check)。",
        "在先验设定上做敏感性分析。",
    ],
}

_DEFAULT_STEPS: list[str] = [
    "做稳健性检查：在不同子样本/设定下复核结论是否稳定。",
    "结合领域知识审视结果是否合理，并核对前提假设。",
    "如需更强结论，换一个更贴切的方法交叉验证。",
]


def _g(obj, name: str, default=None):
    """getattr that also tolerates None objects and missing fields."""
    try:
        v = getattr(obj, name, default)
    except Exception:  # pragma: no cover - extremely defensive
        return default
    return v if v is not None else default


def _data_name(fp) -> str:
    """Human-friendly data file name from the fingerprint path."""
    path = _g(fp, "path", "")
    if not path:
        return "数据"
    try:
        return os.path.basename(str(path)) or str(path)
    except Exception:  # pragma: no cover - defensive
        return str(path)


def _goal_zh(goal: str) -> str:
    return _GOAL_ZH.get((goal or "").strip().lower(), goal or "分析")


def _next_steps(entry, override: bool) -> list[str]:
    family = (_g(entry, "family", "") or "").strip().lower()
    goal = (_g(entry, "goal", "") or "").strip().lower()
    steps = list(_FAMILY_STEPS.get(family, _DEFAULT_STEPS))
    # predict-goal methods always benefit from out-of-sample validation.
    if goal in ("predict", "forecast", "classify"):
        extra = "做交叉验证 / conformal_prediction 评估样本外表现与预测不确定性。"
        if not any("交叉验证" in s or "conformal" in s for s in steps):
            steps = [extra] + steps
    # When preconditions weren't met, lead with the fix.
    if override:
        fix = "先补齐未满足的前提（清洗/换列/换更匹配的方法），再把当前结果当作确证性结论。"
        steps = [fix] + steps
    return steps[:3]


def _fmt_estimates(estimates, cap: int = 6) -> list[str]:
    """Pick the leading numeric estimates (dict insertion order ≈ salience: branches
    assign the headline quantity first) and format them compactly. Skips NaN/inf.
    Never recomputes — only surfaces what the analysis already produced."""
    import math

    try:
        items = list((estimates or {}).items())
    except Exception:
        return []
    out = []
    for k, v in items:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        out.append(f"`{k}` = {fv:.4g}")
        if len(out) >= cap:
            break
    return out


def build_narrative(entry, fp, summary, override, estimates=None) -> list[str]:
    """Return markdown lines for the "## 解读（自动生成）" section.

    Additive + presentation-only: references ``summary`` content + the leading
    ``estimates`` (never recomputes numbers) and the catalog/fingerprint metadata.
    Returns ``[]`` on any failure so the caller can simply omit the section.
    """
    try:
        summary = list(summary or [])
        method = _g(entry, "method", "该方法") or "该方法"
        goal = (_g(entry, "goal", "") or "").strip().lower()
        goal_zh = _goal_zh(goal)
        n_rows = _g(fp, "n_rows", "?")
        n_cols = _g(fp, "n_cols", "?")
        data = _data_name(fp)

        # Split summary into substantive findings vs ⚠ caveats (don't recompute).
        caveats = [s for s in summary if "⚠" in str(s)]
        findings = [s for s in summary if "⚠" not in str(s)]

        lines: list[str] = ["## 解读（自动生成）", ""]

        # 1) headline — one-line conclusion; lead with the first finding if present.
        lines.append(
            f"在 `{data}`（{n_rows} 行 × {n_cols} 列）上用 **{method}**"
            f"（目的：{goal_zh}）做了分析。"
        )
        if findings:
            lines.append(f"**关键发现**：{str(findings[0]).lstrip('- ').strip()}")
        elif not summary:
            lines.append("本次未产出可解读的结果行（见下方产物文件与日志）。")
        est_lines = _fmt_estimates(estimates)
        if est_lines:
            lines.append(f"**关键数值**：{'；'.join(est_lines)}。")
        lines.append("")

        # 2) how to read — what this goal/family answers + precondition stance.
        lines.append("### 如何解读")
        lead = _GOAL_READ.get(goal, "回答该方法所针对的研究问题；结论应结合数据背景与假设来读。")
        lines.append(lead)
        if override:
            lines.append(
                "> ⚠️ 本次为**知情覆盖**运行：部分前提未完全满足，"
                "故结果为**提示性**而非**确证性**，请谨慎解读、勿据此下定论。"
            )
        else:
            lines.append("本次运行满足该方法声明的前提条件，结果可按常规解读。")
        lines.append("")

        # 3) what to look at — substantive findings + a consolidated caveat list.
        lines.append("### 关键看点")
        if len(findings) > 1:
            lines.append("实质结论（详见上方结果摘要）：")
            for s in findings[1:4]:  # cap to keep it concise
                lines.append(f"- {str(s).lstrip('- ').strip()}")
        elif not findings:
            lines.append("（本次无实质结论行可供提炼。）")
        if caveats:
            lines.append("")
            lines.append("注意 / 局限（请重点留意）：")
            for c in caveats:
                lines.append(f"- {str(c).lstrip('- ').strip()}")
        lines.append("")

        # 4) suggested next steps.
        lines.append("### 建议下一步")
        for step in _next_steps(entry, bool(override)):
            lines.append(f"- {step}")
        lines.append("")

        return lines
    except Exception:  # never let a narrative failure break the report
        return []
