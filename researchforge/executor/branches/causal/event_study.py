"""Causal family branch handler: event_study (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("event_study")
def _branch_event_study(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import numpy as np  # noqa: F401
    import pandas as pd

    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if not unit or not time:
        summary.append('事件研究失败：需要面板数据（单位列 + 时间列）。config={"unit":..,"time":..}。')
        return
    _excl = {unit, time}
    bins_ = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins_[0] if bins_ else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if treatment is None or outcome is None:
        summary.append('事件研究失败：需要 二值处理(随时间开启) + 连续结果。config={"treatment":..,"outcome":..}。')
        return

    sub = df[[unit, time, treatment, outcome]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            sub[treatment] = (sub[treatment] == sorted(tvals)[1]).astype(int)
        else:
            summary.append("事件研究失败：处理变量必须二值。")
            return
    sub[treatment] = sub[treatment].astype(int)
    sub[time] = pd.to_numeric(sub[time], errors="coerce")
    sub = sub.dropna(subset=[time])

    onset = sub[sub[treatment] == 1].groupby(unit)[time].min()
    if onset.empty:
        summary.append("事件研究失败：没有任何单位被处理（处理从未开启）。")
        return
    sub["_onset"] = sub[unit].map(onset)
    sub["_evt"] = sub[time] - sub["_onset"]  # NaN for never-treated (kept as comparison group)

    try:
        L = max(2, min(12, int(cfg.get("window", 5))))
    except (TypeError, ValueError):
        L = 5

    def _bin(e):
        return "never" if pd.isna(e) else str(int(max(-L, min(L, e))))

    sub["_evtb"] = sub["_evt"].apply(_bin)
    has_never = bool(sub["_evt"].isna().any())
    treated_bins = sorted({b for b in sub["_evtb"] if b != "never"}, key=int)
    if len(treated_bins) < 2:
        summary.append("事件研究失败：处理单位的事件时间变化不足（需要前后多期）。")
        return
    # fully-staggered with NO never-treated comparison -> TWFE event study is under-identified
    # (needs a 2nd normalization beyond k=-1) and biased under heterogeneity; statsmodels pinv
    # would silently spread the estimate -> fail honestly instead (inference-reviewer must-fix).
    if not has_never and onset.nunique() > 1:
        summary.append("事件研究失败：所有单位最终都被处理且为交错采纳——纯 TWFE 事件研究此情形需第二个"
                       "归一化(再固定一个远端 lead)、且异质效应下有偏；请改用 Callaway-Sant'Anna / de Chaisemartin。")
        return
    # reference MUST be a pre-treatment lead — a lag reference would invert the event-study reading
    leads_avail = [b for b in treated_bins if int(b) < 0]
    if "-1" in treated_bins:
        ref = "-1"
    elif leads_avail:
        ref = max(leads_avail, key=int)  # closest available pre-period
    else:
        summary.append("事件研究失败：没有任何处理前(lead)期可作参照——无法识别动态效应基线"
                       "（单位是否在 onset 当期才进入面板？）。")
        return

    try:
        import statsmodels.formula.api as smf

        formula = (f'Q("{outcome}") ~ C(_evtb, Treatment("{ref}")) '
                   f'+ C(Q("{unit}")) + C(Q("{time}"))')
        model = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub[unit]})

        ci = model.conf_int()
        rx = re.compile(r"C\(_evtb.*?\)\[T\.(-?\d+)\]")
        rows = [{"event_time": int(ref), "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": float("nan")}]
        for name in model.params.index:
            m = rx.match(name)
            if not m:
                continue
            rows.append({"event_time": int(m.group(1)), "coef": float(model.params[name]),
                         "ci_low": float(ci.loc[name, 0]), "ci_high": float(ci.loc[name, 1]),
                         "p": float(model.pvalues[name])})
        es = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
        es.to_csv(d / "event_study.csv", index=False, encoding="utf-8")
        files.append("event_study.csv")

        leads = es[es["event_time"] < -1]
        pretrend_bad = bool((leads["p"] < 0.05).any()) if len(leads) else False
        post = es[es["event_time"] >= 0]
        att_post = float(post["coef"].mean()) if len(post) else float("nan")
        staggered = bool(onset.nunique() > 1)  # >1 distinct onset TIME = staggered
        estimates.update({"att_post_mean": att_post, "n_event_coefs": float(len(es) - 1),
                          "pretrend_violation": 1.0 if pretrend_bad else 0.0,
                          "n_treated_units": float(len(onset))})  # count of treated units

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(-0.5, color="red", ls="--", lw=0.8)
            ax.errorbar(es["event_time"], es["coef"],
                        yerr=[es["coef"] - es["ci_low"], es["ci_high"] - es["coef"]],
                        fmt="o", capsize=3, color="#4C72B0")
            ax.set_xlabel(f"event time (relative to onset; ref={ref})")
            ax.set_ylabel(f"effect on {outcome}")
            ax.set_title("Event study (dynamic treatment effects)")
            fig.tight_layout()
            fig.savefig(d / "event_study.png", dpi=150)
            plt.close(fig)
            files.append("event_study.png")
        except Exception:
            pass

        pt = "⚠ 检出预趋势(平行趋势存疑)" if pretrend_bad else "前置期系数未见显著(支持平行趋势)"
        stag_note = (" ⚠ 处理时点**交错**——双向固定效应事件研究在异质处理效应下可能有偏(负权重)，"
                     "稳健做法见 Callaway-Sant'Anna / de Chaisemartin。" if staggered else "")
        summary.append(
            f"{entry.method} 完成：{outcome} 围绕处理开启的动态效应（参照期 k={ref}，窗口 ±{L}）；"
            f"处理后(k≥0)平均效应 {att_post:.4f}；{len(es)-1} 个事件期系数（图 event_study.png）。{pt}。"
            f" ⚠ DiD 识别依赖**平行趋势**（前置期系数应≈0，已检）；单位+时间双向固定效应、按单位聚类 SE；"
            f"处理后平均为描述性汇总(非加权 ATT)；端点(±{L})为合并累计期、非单期效应。{stag_note}"
        )
        code += [
            "import statsmodels.formula.api as smf  # 事件研究(动态 DiD)",
            f'# evt=time-onset; ols(Q("{outcome}") ~ C(evt,Treatment("{ref}"))+C(unit)+C(time)).fit(cluster=unit)',
        ]
    except Exception as err:
        summary.append(f"事件研究失败：{err}")
