"""Branch handlers for the survival family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _jm_via_r,
)


@register("joint_longitudinal_survival")
def _branch_joint_longitudinal_survival(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import pandas as pd

    from researchforge.executor import rbridge

    idc = cfg.get("id") or fp.unit_col
    timec = cfg.get("time") or fp.time_col
    marker = surv_time = event = None
    covariates: list[str] = []
    if idc and timec and idc in df.columns and timec in df.columns:
        g = df.groupby(idc)
        others = [c for c in df.columns if c not in {idc, timec}]
        # per-subject (constant within id) vs within-subject varying columns
        const_cols, vary_cols = [], []
        for c in others:
            try:
                (const_cols if g[c].nunique(dropna=True).max() <= 1 else vary_cols).append(c)
            except TypeError:
                const_cols.append(c)
        kind = {c.name: c.kind for c in fp.columns}
        # event: a constant binary 0/1 column; surv_time: a constant continuous
        # column (≠event); marker: a within-subject varying continuous column.
        event = cfg.get("event") or next(
            (c for c in const_cols if kind.get(c) == "binary"
             and set(pd.to_numeric(df[c], errors="coerce").dropna().unique()) <= {0, 1}),
            None,
        )
        surv_time = cfg.get("event_time") or next(
            (c for c in const_cols if c != event and kind.get(c) in {"continuous", "count", "id"}
             and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2),
            None,
        )
        marker = cfg.get("marker") or next(
            (c for c in vary_cols if kind.get(c) == "continuous" and c not in {surv_time, event}),
            None,
        )
        covariates = [c for c in (cfg.get("covariates") or [])
                      if c in const_cols and c not in {event, surv_time}][:4]
        if not cfg.get("covariates"):
            covariates = [c for c in const_cols if c not in {event, surv_time}
                          and kind.get(c) in {"binary", "categorical", "continuous", "count"}][:2]
    names_safe = all(
        x and re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(x))
        for x in [idc, timec, marker, surv_time, event, *covariates]
    )
    if not (rbridge.r_available() and rbridge.r_package_available("JM")):
        summary.append("联合模型需要 R 的 JM 包（未检测到）。安装：install.packages('JM')；或分别用 mixed_effects（纵向）+ survival_analysis（生存）。")
    elif not (idc and timec):
        summary.append("联合模型失败：需要面板结构（受试者 id 列 + 随访时间列）。")
    elif not (marker and surv_time and event):
        summary.append(
            "联合模型失败：需要 纵向标志物(随 id 内时间变化的连续列) + 生存时间(每受试者恒定的连续列) "
            "+ 事件(每受试者恒定的 0/1 列)。用 config={\"marker\":..,\"event_time\":..,\"event\":..} 指定。"
        )
    elif not names_safe:
        summary.append("联合模型失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        cols_all = [idc, timec, marker, surv_time, event, *covariates]
        sub = df[cols_all].dropna()
        # survival cols must be per-subject constant (auto-detect guarantees this,
        # but a config override can bypass it → first-row dedup would silently use
        # the first visit's value; Opus catch).
        _g = sub.groupby(idc)
        nonconst = [c for c in (surv_time, event, *covariates) if _g[c].nunique(dropna=True).max() > 1]
        if sub[idc].nunique() < 20 or sub[event].astype(float).sum() < 10:
            summary.append(
                f"联合模型失败：受试者数 {sub[idc].nunique()}（需 ≥20）或事件数 "
                f"{int(sub[event].astype(float).sum())}（需 ≥10）太少，联合模型不稳。"
            )
        elif nonconst:
            summary.append(
                f"联合模型失败：生存时间/事件/协变量列 {nonconst} 在受试者内不恒定"
                "（应每受试者一个值）；请检查 config 指定的列。"
            )
        else:
            csv = d / "_jm_input.csv"
            sub.to_csv(csv, index=False)
            try:
                ev, lo, fit = _jm_via_r(csv, idc, timec, marker, surv_time, event, covariates)
                ev.to_csv(d / "jm_event_submodel.csv", index=False, encoding="utf-8")
                lo.to_csv(d / "jm_longitudinal_submodel.csv", index=False, encoding="utf-8")
                files += ["jm_event_submodel.csv", "jm_longitudinal_submodel.csv"]
                arow = ev[ev["term"] == "Assoct"]
                assoc = float(arow["value"].iloc[0]) if len(arow) else float("nan")
                assoc_p = float(arow["p_value"].iloc[0]) if len(arow) else float("nan")
                srow = lo[lo["term"] == timec]
                slope = float(srow["value"].iloc[0]) if len(srow) else float("nan")
                estimates["association"] = round(assoc, 4)
                estimates["association_p"] = round(assoc_p, 4)
                estimates["longitudinal_slope"] = round(slope, 4)
                estimates["aic"] = round(fit["aic"], 2)
                estimates["n_subjects"] = float(int(fit["n_subjects"]))
                estimates["n_events"] = float(int(fit["n_events"]))
                sig = "显著" if assoc_p < 0.05 else "不显著"
                direction = "标志物越高、事件风险越低" if assoc < 0 else "标志物越高、事件风险越高"
                (d / "jm_summary.txt").write_text(
                    f"联合纵向-生存模型（R JM，Weibull-PH，关联=当前标志物值）\n"
                    f"标志物 {marker}（随 {timec}）↔ 事件 Surv({surv_time},{event})，"
                    f"{int(fit['n_subjects'])} 受试者 / {int(fit['n_events'])} 事件，协变量 {covariates or '无'}\n"
                    f"关联参数 Assoct = {assoc:.4f}（p={assoc_p:.4g}，{sig}）→ {direction}（log 风险比/单位标志物）\n"
                    f"纵向斜率（{timec}）= {slope:.4f}；AIC={fit['aic']:.1f}\n"
                    "联合模型同时建标志物轨迹与事件时间、用当前真值入风险，校正了"
                    "测量误差与信息性删失（标准两步法/时变 Cox 不能）。\n"
                    "假定：随机效应正态、关联通过当前真值、删失随机（MAR）。\n\n"
                    "事件子模型：\n" + ev.to_string(index=False)
                    + "\n\n纵向子模型：\n" + lo.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("jm_summary.txt")
                summary.append(
                    f"{entry.method} 完成（R/JM）：标志物 {marker} ↔ 事件 {event}；"
                    f"关联参数={assoc:.4f}（p={assoc_p:.3g}，{sig}，{direction}）；"
                    f"纵向斜率={slope:.4f}；{int(fit['n_subjects'])} 受试者/{int(fit['n_events'])} 事件，AIC={fit['aic']:.1f}。"
                    "⚠ 假定随机效应正态、关联经当前真值、删失随机(MAR)；联合建模校正测量误差+信息删失。"
                )
                code += [
                    "library(JM)  # 联合纵向-生存模型",
                    f"# lme({marker}~{timec},random=~{timec}|{idc}) + coxph(Surv({surv_time},{event})~..) -> jointModel",
                ]
            except Exception as err:
                summary.append(f"联合模型拟合失败（可能不收敛）：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass



@register("survival_analysis")
def _branch_survival_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    # NB: do NOT exclude fp.time_col — in survival data the duration IS the
    # time column (profiler may flag a "time"/"days" column as time_col).
    _excl = {fp.unit_col}

    def _named(kws, kinds, extra_excl=()):
        ex = _excl | set(extra_excl)
        return next(
            (
                c.name
                for c in fp.columns
                if any(k in c.name.lower() for k in kws) and c.kind in kinds and c.name not in ex
            ),
            None,
        )

    ev_kws = ["event", "status", "death", "dead", "fail", "censor", "relapse"]
    dur_kws = ["duration", "time", "days", "month", "year", "tenure", "surviv", "followup", "week", "age_at"]
    event_col = _named(ev_kws, {"binary"}) or next(
        (c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl), None
    )
    dur_col = _named(dur_kws, {"continuous", "count"}, extra_excl=(event_col,) if event_col else ())
    if dur_col is None and event_col is not None:
        dur_col = next(
            (
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "count"}
                and c.name not in _excl | {event_col}
                and (df[c.name].dropna() > 0).all()
            ),
            None,
        )
    if event_col is None or dur_col is None:
        summary.append(
            "生存分析失败：需要一个事件列（二值 0/1，如 event/status/death）"
            "+ 一个时长列（正数，如 time/duration/days）。"
        )
    else:
        try:
            from lifelines import CoxPHFitter, KaplanMeierFitter
            from lifelines.statistics import logrank_test, proportional_hazard_test

            group_col = next(
                (
                    c.name
                    for c in fp.columns
                    if c.kind in {"binary", "categorical"}
                    and c.name not in _excl | {event_col}
                    and df[c.name].dropna().nunique() == 2
                ),
                None,
            )
            covars = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in _excl | {dur_col, event_col}
            ][:5]
            keep = [dur_col, event_col] + ([group_col] if group_col else []) + covars
            sub = df[list(dict.fromkeys(keep))].dropna()
            sub = sub[sub[dur_col].astype(float) > 0]
            dur = sub[dur_col].astype(float)
            ev = sub[event_col].astype(int)
            # event must be 0/1 (1=event, 0=censored) for lifelines event_observed.
            # (profiler's "binary" is already exactly {0,1}; defensive guard + the
            # disclosure below also flags the un-detectable reversed-coding risk.)
            if set(ev.dropna().unique()) - {0, 1}:
                raise ValueError(f"事件列 {event_col} 非 0/1 编码（应 1=事件,0=删失）")
            n_ev, n_cens = int(ev.sum()), int((ev == 0).sum())

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            kmf = KaplanMeierFitter()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            logrank_p = None
            if group_col:
                for gval, gdf in sub.groupby(group_col):
                    kmf.fit(gdf[dur_col].astype(float), gdf[event_col].astype(int), label=f"{group_col}={gval}")
                    kmf.plot_survival_function(ax=ax)
                levels = list(sub.groupby(group_col).groups)
                g0 = sub[sub[group_col] == levels[0]]
                g1 = sub[sub[group_col] == levels[1]]
                lr = logrank_test(
                    g0[dur_col].astype(float), g1[dur_col].astype(float),
                    g0[event_col].astype(int), g1[event_col].astype(int),
                )
                logrank_p = float(lr.p_value)
            else:
                kmf.fit(dur, ev)
                kmf.plot_survival_function(ax=ax)
            ax.set_xlabel(f"{dur_col} (time)")
            ax.set_ylabel("survival probability")
            ax.set_title("Kaplan-Meier survival")
            fig.tight_layout()
            fig.savefig(d / "km_curve.png", dpi=150)
            plt.close(fig)
            files.append("km_curve.png")

            kmf_all = KaplanMeierFitter().fit(dur, ev)
            median = float(kmf_all.median_survival_time_)
            estimates["median_survival"] = round(median, 4) if np.isfinite(median) else -1.0
            estimates["n_events"] = float(n_ev)
            estimates["n_censored"] = float(n_cens)

            cox_msg = ""
            if covars and n_ev >= 2 * len(covars) + 2:
                try:
                    cph = CoxPHFitter()
                    cph.fit(sub[[dur_col, event_col, *covars]], dur_col, event_col)
                    cs = cph.summary
                    ctab = cs[["coef", "exp(coef)", "se(coef)", "p"]].copy()
                    ctab.columns = ["coef", "hazard_ratio", "std_err", "p_value"]
                    ctab.round(4).to_csv(d / "cox_hazard_ratios.csv", encoding="utf-8")
                    files.append("cox_hazard_ratios.csv")
                    for cv in covars:
                        if cv in cs.index:
                            estimates[f"HR_{cv}"] = round(float(cs.loc[cv, "exp(coef)"]), 4)
                    try:
                        ph = proportional_hazard_test(cph, sub[[dur_col, event_col, *covars]])
                        ph_p = float(ph.summary["p"].min())
                        cox_msg = (
                            f"；Cox 风险比见 cox_hazard_ratios.csv（{len(covars)} 协变量）；"
                            f"比例风险检验 min-p={ph_p:.3g}"
                            f"（{'满足 PH 假定' if ph_p > 0.05 else '⚠ PH 假定可能不成立'}）"
                        )
                    except Exception:
                        cox_msg = f"；Cox 风险比见 cox_hazard_ratios.csv（{len(covars)} 协变量）"
                except Exception as cerr:
                    cox_msg = f"；Cox 拟合跳过（{str(cerr)[:60]}）"

            med_txt = f"{median:.3g}" if np.isfinite(median) else "未达（>50% 存活到末期）"
            grp_txt = (
                f"；按 {group_col} 分组 KM + log-rank p={logrank_p:.3g}"
                f"（{'组间生存有显著差异' if logrank_p is not None and logrank_p < 0.05 else '组间无显著差异'}）"
                if group_col
                else ""
            )
            summary.append(
                f"{entry.method} 完成：{len(sub)} 例（{n_ev} 事件 / {n_cens} 删失）；"
                f"事件列={event_col}，时长列={dur_col}；中位生存={med_txt}{grp_txt}{cox_msg}。"
                "⚠ 事件/时长列按列名自动识别，请核对；事件列须 0/1（1=事件,0=删失），"
                "若反向编码 HR 会反转；删失假定为随机非信息性。"
            )
            code += [
                "from lifelines import KaplanMeierFitter, CoxPHFitter  # 生存分析",
                f"# KM.fit(df['{dur_col}'], df['{event_col}']); CoxPHFitter().fit(df, '{dur_col}', '{event_col}')",
            ]
        except Exception as err:
            summary.append(f"生存分析失败：{err}")

