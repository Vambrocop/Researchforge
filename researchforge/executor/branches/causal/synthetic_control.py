"""Causal family branch handler: synthetic_control (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import _synthetic_control


@register("synthetic_control")
def _branch_synthetic_control(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    unit, time = fp.unit_col, fp.time_col
    _excl = {unit, time, *fp.treatment_candidates}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    # treated unit + treatment time: config, else derive from a treatment 0/1 column
    treated = cfg.get("treated_unit")
    treat_time = cfg.get("treatment_time")
    ever_treated: set = set()  # ALL units ever treated → excluded from donor pool
    if unit and time and fp.treatment_candidates:
        tcol = next((c for c in fp.treatment_candidates if c in df.columns), None)
        if tcol is not None:
            trows = df[df[tcol] == 1]
            if len(trows):
                ever_treated = set(trows[unit].dropna().unique())
                if treated is None:
                    treated = trows[unit].dropna().unique()[0]
                if treat_time is None:
                    treat_time = df[(df[unit] == treated) & (df[tcol] == 1)][time].min()
    # coerce a JSON-supplied treatment_time to the time column's dtype
    if treat_time is not None and time and time in df.columns:
        try:
            treat_time = type(df[time].dropna().iloc[0])(treat_time)
        except Exception:
            pass
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in {outcome, unit, time}]
    preds = forced or [c for c in cont if c != outcome][:5]
    if not (unit and time):
        summary.append("合成控制失败：需要面板数据（单位列 + 时间列）。")
    elif outcome is None:
        summary.append("合成控制失败：需要一个连续结果变量。")
    elif treated is None or treat_time is None:
        summary.append(
            "合成控制失败：需指定干预单位与干预时点。"
            "用 config={\"treated_unit\": \"<单位>\", \"treatment_time\": <时点>}，"
            "或在数据中提供处理指示列（0/1）。"
        )
    else:
        try:
            import pysyncon  # noqa: F401
        except ImportError:
            summary.append("合成控制需要 pysyncon 包（未检测到）。安装：pip install pysyncon；或用 did。")
        else:
            try:

                # exclude OTHER ever-treated units from donors (contamination bias).
                other_treated = {u for u in ever_treated if u != treated}
                weights, att, pre_rmspe, n_don, post = _synthetic_control(
                    df, unit, time, outcome, treated, treat_time, preds, d / "gaps.png",
                    exclude=other_treated,
                )
                if len(ever_treated) > 1:
                    contam_note = (
                        f"；⚠ 检测到 {len(ever_treated)} 个被处理单位（疑交错采纳），"
                        f"仅对 {treated} 建模、其余已剔出供体池；交错处理建议改用 did"
                    )
                elif not ever_treated:
                    contam_note = "；⚠ 由 config 指定处理单位、未据处理列校验供体，假定供体未受干预"
                else:
                    contam_note = ""
                wdf = weights.reset_index()
                wdf.columns = ["donor", "weight"]
                wdf = wdf[wdf["weight"] > 1e-4].reset_index(drop=True)
                wdf.to_csv(d / "donor_weights.csv", index=False, encoding="utf-8")
                files.append("donor_weights.csv")
                if (d / "gaps.png").exists():
                    files.append("gaps.png")
                att_val = float(att.get("att", float("nan")))
                att_se = float(att.get("se", float("nan")))
                estimates["att"] = round(att_val, 4)
                estimates["att_se"] = round(att_se, 4)
                estimates["pre_rmspe"] = round(pre_rmspe, 4)
                estimates["n_donors_used"] = float(len(wdf))
                top = wdf.head(5)
                top_txt = ", ".join(f"{r.donor}={r.weight:.2f}" for r in top.itertuples())
                (d / "synth_summary.txt").write_text(
                    f"合成控制法（Abadie）：干预单位 {treated}，干预时点 {treat_time}，"
                    f"结果变量 {outcome}\n"
                    f"ATT（干预后平均处理效应）= {att_val:.4f}（SE≈{att_se:.4f}）\n"
                    f"干预前拟合 RMSPE = {pre_rmspe:.4f}（越小=合成体越贴合真实前期路径）\n"
                    f"合成体权重（前 5）：{top_txt}\n"
                    f"对照供体池 {n_don} 个，实际赋权 {len(wdf)} 个。\n"
                    "注：ATT 可信度取决于干预前拟合好坏（RMSPE 小）；推断需做安慰剂检验。\n\n"
                    + wdf.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("synth_summary.txt")
                summary.append(
                    f"{entry.method} 完成（pysyncon）：干预单位 {treated} @ {treat_time}，"
                    f"结果 {outcome}；ATT={att_val:.4f}（SE≈{att_se:.4f}）；"
                    f"干预前 RMSPE={pre_rmspe:.4f}；合成体由 {len(wdf)} 个供体加权（{top_txt}）。"
                    "⚠ ATT 可信度依赖干预前拟合；正式推断需安慰剂检验（in-space/in-time placebo）。"
                    + contam_note
                )
                code += [
                    "from pysyncon import Dataprep, Synth  # 合成控制法",
                    f"# 干预单位={treated}, 干预时点={treat_time}; 干预前拟合权重 -> 后期 gap=ATT",
                ]
            except Exception as err:
                summary.append(f"合成控制拟合失败：{err}")
