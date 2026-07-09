"""Study-mode report generation — pure markdown assembly from already-computed
results (profiler / recommender / executor outputs). No new inference, no LLM, no
network call happens here: every word/number comes from an existing
``RunResult.summary`` / ``estimates`` / catalog metadata / ``DataFingerprint``.
See docs/design-study-mode.md §3 (report structure, definitive) and §6 STOP point 1
(the cross-method convergence rule must stay a dumb, honest, pure rule).

Split out of ``study.py`` per docs/design-study-mode.md §5 ("逼近 800 行就把 report
生成拆 study_report.py") — kept as a separate concern from the orchestration in
``study.py`` from the start.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from researchforge import __version__
from researchforge.executor._helpers.report_narrative import _fmt_estimates

_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

_ISSUE_LABELS = {
    "missing": "缺失值",
    "duplicate_rows": "重复行",
    "constant": "常量列",
    "outliers": "离群值（IQR 规则）",
    "high_cardinality": "高基数文本列",
    "rare_categories": "稀有类别尾部",
    "coerced_numeric": "文本已强制转数值",
}

# Pure rule for the §跨方法收敛信号 section (docs/design-study-mode.md §3 + §6 STOP#1):
# two methods "agree in magnitude" on a shared estimate key when the ratio of the
# larger to the smaller absolute value is within this factor. Deliberately simple
# and disclosed inline in the report — no fuzzy key matching, no learned threshold.
_MAGNITUDE_RATIO_OK = 3.0


def _mark(light: str) -> str:
    return _EMOJI.get(light, "⚪")


def _sign(v: float) -> int:
    return 0 if v == 0 else (1 if v > 0 else -1)


def _rel(study_dir: Path, abs_path: str) -> str:
    """Path relative to the study dir (for markdown links/images), forward-slashed.
    Falls back to the raw path if it isn't actually inside study_dir."""
    try:
        return str(Path(abs_path).relative_to(study_dir)).replace("\\", "/")
    except ValueError:
        return str(abs_path).replace("\\", "/")


def _fp_quality_lines(fp, clean_log: Optional[list[dict]]) -> list[str]:
    lines: list[str] = []
    struct: list[str] = []
    if fp.is_panel:
        struct.append(f"面板（单位列 `{fp.unit_col}`）")
    if fp.is_timeseries:
        struct.append("时序")
    if fp.has_geo:
        struct.append("含地理信息")
    if fp.time_col and not fp.is_panel:
        struct.append(f"时间列 `{fp.time_col}`")
    lines.append(f"- 规模：{fp.n_rows} 行 × {fp.n_cols} 列")
    lines.append(f"- 结构：{'、'.join(struct) if struct else '截面数据（无面板/时序/地理结构）'}")
    if fp.likely_outcome:
        conf = fp.likely_outcome_confidence or "低"
        lines.append(
            f"- 角色提示：可能的结果变量 `{fp.likely_outcome}`（置信：{conf}）"
            + (f" — {fp.role_hint_reason}" if fp.role_hint_reason else "")
        )
    if fp.likely_treatment:
        lines.append(f"- 角色提示：可能的处理变量 `{fp.likely_treatment}`")
    if fp.issues:
        lines.append(f"- 质量发现（{len(fp.issues)} 项）：")
        for iss in fp.issues:
            label = _ISSUE_LABELS.get(iss.kind, iss.kind)
            col = f"`{iss.column}` " if iss.column else ""
            lines.append(f"  - ⚠ [{iss.severity}] {col}{label}：{iss.detail}")
    else:
        lines.append("- 质量发现：无（未检出重复/缺失/常量/离群等问题）")
    if clean_log is not None:
        n_applied = sum(1 for e in clean_log if e.get("applied"))
        lines.append(f"- `--clean` 已应用（{n_applied}/{len(clean_log)} 步生效，逐条披露）：")
        for e in clean_log:
            m = "✓" if e.get("applied") else "⚠"
            col = f"`{e['column']}` " if e.get("column") else ""
            lines.append(f"  - {m} {e.get('action')} {col}— {e.get('detail')}")
    return lines


def _render_run_block(
    entry_id: str, method_label: str, result, error: Optional[str], study_dir: Path
) -> list[str]:
    """One method's content block: summary verbatim / key estimates / figures /
    artifact list. Reused for the §0 descriptive baseline AND each §1..K method
    section — same rendering, caller picks the heading level. ``result`` is a
    RunResult on success; ``error`` is the exception text when the ORCHESTRATION
    call itself failed (study.py's own try/except — distinct from a handler-level
    failure, which run_analysis already absorbs into a normal RunResult whose
    summary contains a '⚠ ... 执行失败' line)."""
    lines: list[str] = []
    if result is None:
        lines.append(f"⚠ **{method_label}（{entry_id}）执行失败**：{error or '未知错误'}")
        lines.append(
            "> 提示：可能是数据不满足该方法的前提，或所需依赖包/后端不可用；"
            "可参考本节其他候选方法，或用 `py -3 -m researchforge.cli params "
            f"{entry_id}` 查看该方法的可配置项后针对性重试。"
        )
        return lines

    for raw_line in (result.summary or "").split("\n"):
        raw_line = raw_line.strip()
        if raw_line:
            lines.append(f"- {raw_line}")
    if not lines:
        lines.append("- （本次未产出可读的结果行）")
    if "执行失败" in (result.summary or ""):
        lines.append(
            "> 提示：该方法执行时内部失败，可能是前提未满足或依赖包缺失；"
            "可参考本节其他候选方法，或查看该方法的可配置项后针对性重试。"
        )

    est_lines = _fmt_estimates(result.estimates)
    if est_lines:
        lines.append("")
        lines.append(f"**关键数值**：{'；'.join(est_lines)}。")

    figs = [f for f in result.files if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))]
    if figs:
        lines.append("")
        for f in figs:
            rel = _rel(study_dir, str(Path(result.output_dir) / f))
            lines.append(f"![{f}]({rel})")

    lines.append("")
    lines.append(f"产物（{len(result.files)} 个，目录 `{_rel(study_dir, result.output_dir)}/`）：")
    for f in result.files:
        rel = _rel(study_dir, str(Path(result.output_dir) / f))
        lines.append(f"  - [{f}]({rel})")
    return lines


def _why_chosen(rec, plan) -> str:
    if rec.diagnostic_note:
        hits = [d.finding for d in (plan.diagnostics or []) if rec.entry.id in d.prefer]
        if hits:
            return "；".join(dict.fromkeys(hits))
        return rec.diagnostic_note
    return f"{rec.entry.family} 家族方法，与本数据结构/角色亲和度较高（未命中特定值级诊断）"


def _selection_table(chosen: list, plan, requested_k: int) -> list[str]:
    lines = ["## §选法依据", ""]
    if len(chosen) < requested_k:
        lines.append(
            f"⚠ 本次仅选出 {len(chosen)}/{requested_k} 个方法"
            "（候选池按 family 多样性过滤+回填后仍不足；严谨度灯未放宽，如实按实际数出报告）。"
        )
        lines.append("")
    if not chosen:
        lines.append("（无可行的实质方法可选——数据结构/goal 过窄，或 catalog 前提普遍不满足。）")
        lines.append("")
        return lines
    lines.append("| 方法 | id | family | 严谨度 | 契合 | 为何选它 |")
    lines.append("|---|---|---|---|---|---|")
    for rec in chosen:
        lines.append(
            f"| {rec.entry.method} | `{rec.entry.id}` | {rec.entry.family} | "
            f"{_mark(rec.rigor.light)} {rec.rigor.score} | {rec.score.fit} | {_why_chosen(rec, plan)} |"
        )
    lines.append("")
    return lines


def _convergence_section(run_entries: list[dict]) -> list[str]:
    """Pure-rule cross-method comparison — zero LLM, zero network (§3, §6 STOP#1).
    Only estimate keys LITERALLY shared (exact string match) by >=2 methods that
    actually produced a RunResult are compared; anything else is honestly left
    alone rather than fuzzy-matched (fuzzy matching is exactly the kind of thing
    that would misfire and trip STOP point 1)."""
    lines = ["## §跨方法收敛信号", ""]
    by_key: dict[str, list[tuple[str, float]]] = {}
    for e in run_entries:
        res = e["result"]
        if res is None:
            continue
        for k, v in (res.estimates or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            by_key.setdefault(k, []).append((e["rec"].entry.id, fv))

    shared = {k: vs for k, vs in by_key.items() if len({mid for mid, _ in vs}) >= 2}
    if not shared:
        lines.append("各方法产出的 estimate 键互不相同（回答的是不同的问题），不做数值横比。")
        lines.append("")
        return lines

    lines.append(
        "以下键被 ≥2 个方法同时产出（精确同名匹配，不做模糊对齐）。"
        f"量级一致定义为该键上 max(|值|)/min(|值|) ≤ {_MAGNITUDE_RATIO_OK:.0f}；"
        "含 0 值时量级不可比。"
    )
    lines.append("")
    for k, vs in shared.items():
        signs = {_sign(v) for _, v in vs}
        sign_txt = "符号一致" if len(signs) == 1 else "符号不一致"
        abs_vals = [abs(v) for _, v in vs]
        if 0 in abs_vals:
            mag_txt = "量级不可比（含 0 值）"
        else:
            ratio = max(abs_vals) / min(abs_vals)
            mag_txt = f"量级{'一致' if ratio <= _MAGNITUDE_RATIO_OK else '不一致'}（比值 {ratio:.2g}）"
        vals_txt = "、".join(f"{mid}={v:.4g}" for mid, v in vs)
        lines.append(f"- `{k}`：{vals_txt} — {sign_txt}，{mag_txt}")
    lines.append("")
    return lines


def _appendix_section(chosen: list) -> list[str]:
    lines = ["## §方法学附录", ""]
    for rec in chosen:
        s = rec.score
        lines.append(f"### {rec.entry.method}（`{rec.entry.id}`）")
        lines.append(
            f"- 评分卡：总{s.overall} / 契合{s.fit} / 流行{s.popularity} / 可发表{s.publishability} "
            f"/ 美观{s.aesthetics} / 新颖{s.novelty} / 难度{s.difficulty}（越高越难，是成本项）"
        )
        lines.append(f"- 严谨度：{_mark(rec.rigor.light)} {rec.rigor.note}")
        if rec.entry.biases:
            lines.append("- 已知偏差/局限：" + "；".join(rec.entry.biases))
        lines.append("")
    return lines


def _disclosure_roundup(all_lines: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for line in all_lines:
        if "⚠" in line:
            stripped = line.strip()
            # de-dash: most captured lines are already "- ⚠ ..." bullets (from summary
            # bullets / quality findings / clean log); strip that ONE leading bullet
            # marker so re-bulleting below doesn't double up into "- - ⚠ ...".
            if stripped.startswith("- "):
                stripped = stripped[2:]
            seen.setdefault(stripped, None)
    lines = ["## §披露汇总", ""]
    if not seen:
        lines.append("本次研究未产生任何 ⚠ 披露。")
        return lines
    lines.append(f"全文共 {len(seen)} 条 ⚠ 披露（去重后，按出现顺序）：")
    for line in seen:
        lines.append(f"- {line}")
    return lines


def render_report(
    fp,
    plan,
    goal: Optional[str],
    base_entry,
    base_result,
    base_error: Optional[str],
    run_entries: list[dict],
    clean_log: Optional[list[dict]],
    study_dir: Path,
    requested_k: int,
) -> str:
    """Assemble the full study_report.md text. ``run_entries`` is a list of
    ``{"rec": Recommendation, "result": RunResult|None, "error": str|None}`` in
    the order the K methods were run (already the diversity-filtered picks)."""
    body: list[str] = ["# ResearchForge 研究报告（Study Mode）", ""]
    body.append(f"- 数据：`{fp.path}`")
    body.append(f"- 目标：{goal or '（未指定，跨全部目标筛选）'}")
    body.append(f"- 引擎版本：{__version__}")
    body.append("")

    body += ["## §0 数据与质量", ""]
    body += _fp_quality_lines(fp, clean_log)
    body.append("")
    body.append("### 描述性统计基线（不计入所选方法数）")
    body.append("")
    if base_entry is None:
        body.append("（catalog 未找到 descriptive_stats，跳过基线。）")
    else:
        body += _render_run_block(
            "descriptive_stats", base_entry.method, base_result, base_error, study_dir
        )
    body.append("")

    chosen = [e["rec"] for e in run_entries]
    body += _selection_table(chosen, plan, requested_k)

    for i, e in enumerate(run_entries, start=1):
        rec = e["rec"]
        body.append(f"## §{i} {rec.entry.method}（`{rec.entry.id}`）")
        body.append("")
        body += _render_run_block(rec.entry.id, rec.entry.method, e["result"], e["error"], study_dir)
        body.append("")

    body += _convergence_section(run_entries)
    body += _appendix_section(chosen)
    body += _disclosure_roundup(body)

    return "\n".join(body)
