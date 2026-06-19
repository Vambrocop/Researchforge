"""Branch handlers for the survival family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _jm_via_r,
)

# Column-name hints shared by the survival branches for auto-detecting roles.
_DUR_KWS = ["duration", "time", "days", "month", "year", "tenure", "surviv",
            "followup", "fu", "week", "age_at"]
_EV_KWS = ["event", "status", "death", "dead", "fail", "censor", "relapse"]
_GRP_KWS = ["group", "arm", "treat", "cohort", "sex", "male"]
_IDENT = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")


def _resolve_duration_event(df, fp, cfg):
    """Resolve (dur_col, event_col) from cfg overrides, else profiler kinds + name hints.

    Mirrors _branch_survival_analysis: the duration IS the time column in survival
    data, so we do NOT exclude fp.time_col. event = a binary 0/1 column (name-hint
    preferred); duration = a positive continuous/count column (name-hint preferred).
    The multi-state event branch (competing risks) accepts a categorical event too.
    """
    _excl = {fp.unit_col}

    def _named(kws, kinds, extra_excl=()):
        ex = _excl | set(extra_excl)
        return next(
            (c.name for c in fp.columns
             if any(k in c.name.lower() for k in kws) and c.kind in kinds and c.name not in ex),
            None,
        )

    event_col = cfg.get("event") or _named(_EV_KWS, {"binary"}) or next(
        (c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl), None
    )
    dur_col = cfg.get("duration") or _named(
        _DUR_KWS, {"continuous", "count", "id"}, extra_excl=(event_col,) if event_col else ()
    )
    if dur_col is None and event_col is not None:
        dur_col = next(
            (c.name for c in fp.columns
             if c.kind in {"continuous", "count", "id"}
             and c.name not in _excl | {event_col}
             and (df[c.name].dropna() > 0).all()),
            None,
        )
    return dur_col, event_col


def _resolve_group(df, fp, cfg, excl, n_levels=None):
    """Resolve a grouping column (cfg override, else binary/categorical name-hint).

    n_levels: if 2, require exactly 2 levels (for two-sample comparisons).
    """
    if cfg.get("group"):
        return cfg["group"] if cfg["group"] in df.columns else None
    cand = [
        c.name for c in fp.columns
        if c.kind in {"binary", "categorical"} and c.name not in excl
        and (n_levels is None or df[c.name].dropna().nunique() == n_levels)
    ]
    # prefer a name-hinted group column, else the first qualifying categorical
    named = [c for c in cand if any(k in c.lower() for k in _GRP_KWS)]
    return (named or cand or [None])[0]


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


@register("competing_risks")
def _branch_competing_risks(ctx: Ctx) -> None:
    """Cumulative Incidence Functions for competing events (Aalen-Johansen).

    The event column is multi-state: 0=censored, `event_of_interest` (default 1)
    = the event we want, any other positive code = a competing event. Aalen-Johansen
    gives the *proper* CIF for the event of interest in the presence of competing
    risks — naively treating competing events as censoring (1-KM) over-estimates it.
    Optional R cmprsk Gray's test compares CIFs across groups.
    """
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    dur_col, event_col = _resolve_duration_event(df, fp, cfg)
    # For competing risks the event is multi-state (≥3 codes incl. 0), so the
    # profiler may call it "categorical"/"count"/"id" rather than "binary". If the
    # binary auto-pick failed, look for a multi-state integer-coded column by name.
    if event_col is None or df[event_col].dropna().nunique() <= 2:
        cand = next(
            (c.name for c in fp.columns
             if any(k in c.name.lower() for k in _EV_KWS)
             and c.name != dur_col and c.name != fp.unit_col
             and pd.api.types.is_numeric_dtype(df[c.name])
             and df[c.name].dropna().nunique() >= 3),
            None,
        )
        if cand is not None:
            event_col = cand
    eoi = int(cfg.get("event_of_interest", 1))

    # The multi-state event was resolved after duration, so a collision is possible
    # (a count-coded event could have been grabbed as duration). Re-pick a positive
    # continuous/count duration that is not the event column.
    if event_col is not None and dur_col == event_col:
        dur_col = next(
            (c.name for c in fp.columns
             if c.kind in {"continuous", "count", "id"}
             and c.name not in {fp.unit_col, event_col}
             and (df[c.name].dropna() > 0).all()),
            None,
        )

    if dur_col is None or event_col is None:
        summary.append(
            "竞争风险失败：需要一个时长列（正数）+ 一个多状态事件列"
            "（0=删失，1=关注事件，≥2=竞争事件）。用 config={\"duration\":..,\"event\":..} 指定。"
        )
        return
    try:
        from lifelines import AalenJohansenFitter, KaplanMeierFitter

        excl = {fp.unit_col, dur_col, event_col}
        group_col = _resolve_group(df, fp, cfg, excl)

        keep = [dur_col, event_col] + ([group_col] if group_col else [])
        sub = df[list(dict.fromkeys(keep))].dropna()
        sub = sub[sub[dur_col].astype(float) > 0]
        dur = sub[dur_col].astype(float)
        ev = sub[event_col].astype(int)
        codes = sorted(set(ev.unique()))
        # Need ≥2 distinct event types beyond censoring (i.e. a genuine competing risk).
        comp_codes = [c for c in codes if c not in (0, eoi)]
        if eoi not in codes:
            summary.append(
                f"竞争风险失败：关注事件编码 {eoi} 不在事件列 {event_col} 中（出现的编码：{codes}）。"
                "约定 0=删失、1=关注事件、≥2=竞争事件；用 config={\"event_of_interest\":k} 指定。"
            )
            return
        if not comp_codes:
            summary.append(
                f"竞争风险失败：事件列 {event_col} 只有 {codes}，没有竞争事件（≥2 编码）。"
                "无竞争事件时 Aalen-Johansen 退化为 1-KM——请改用 survival_analysis（KM/Cox）。"
            )
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Time points at which to report the CIF: 5 evenly spaced quantiles of the
        # observed durations (defensible default; reported in the table).
        qs = np.linspace(0.2, 1.0, 5)
        tpts = np.quantile(dur.values, qs)

        rows = []
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        groups = (list(sub.groupby(group_col)) if group_col else [(None, sub)])
        cif_eoi_at_max = {}
        for gval, gdf in groups:
            gd = gdf[dur_col].astype(float)
            ge = gdf[event_col].astype(int)
            label_g = "" if gval is None else f"{group_col}={gval}: "
            present = set(ge.unique())
            # CIF for the event of interest + each competing type present in THIS group
            # (a group may lack a given code; AJF needs the code present to fit it).
            for et in [eoi] + comp_codes:
                if et not in present:
                    continue
                ajf = AalenJohansenFitter(seed=0)
                ajf.fit(gd, ge, event_of_interest=et, label=f"{label_g}CIF(event {et})")
                ajf.plot(ax=ax)
                cif_at = np.asarray(ajf.predict(list(tpts)))
                if et == eoi:
                    cif_eoi_at_max[gval] = float(cif_at[-1])
                for t, q, v in zip(tpts, qs, cif_at):
                    rows.append({"group": gval if gval is not None else "all",
                                 "event_type": et, "time": round(float(t), 4),
                                 "time_quantile": round(float(q), 2),
                                 "CIF": round(float(v), 4)})

        ax.set_xlabel(f"{dur_col} (time)")
        ax.set_ylabel("cumulative incidence")
        ax.set_title("Cumulative Incidence Functions (Aalen-Johansen)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(d / "cif_curves.png", dpi=150)
        plt.close(fig)
        files.append("cif_curves.png")

        cif_tab = pd.DataFrame(rows)
        cif_tab.to_csv(d / "cif_table.csv", index=False, encoding="utf-8")
        files.append("cif_table.csv")

        # The headline check: at the longest reported time, the proper CIF for the
        # event of interest is *below* the naive 1-KM (which censors competing events).
        kmf = KaplanMeierFitter().fit(dur, (ev == eoi).astype(int))
        naive_1mkm = float(1.0 - kmf.predict(tpts[-1]))
        cif_overall = float(
            AalenJohansenFitter(seed=0)
            .fit(dur, ev, event_of_interest=eoi).predict(tpts[-1])
        )
        estimates["cif_eoi_at_max_time"] = round(cif_overall, 4)
        estimates["naive_1_minus_km_at_max_time"] = round(naive_1mkm, 4)
        estimates["n_obs"] = float(len(sub))
        for code_k in codes:
            estimates[f"n_event_{code_k}"] = float(int((ev == code_k).sum()))
        if group_col is not None:
            for gval, v in cif_eoi_at_max.items():
                estimates[f"cif_eoi_at_max_{gval}"] = round(float(v), 4)

        # Optional: Gray's test (between-group CIF equality) via R cmprsk.
        gray_msg = ""
        if group_col is not None and sub[group_col].nunique() >= 2:
            gray_p = _gray_test_via_r(sub, dur_col, event_col, group_col, eoi, d)
            if gray_p is not None and np.isfinite(gray_p):
                estimates["gray_test_p"] = round(float(gray_p), 4)
                gray_msg = (
                    f"；Gray 检验（组间 CIF 是否相等，R cmprsk）p={gray_p:.3g}"
                    f"（{'组间累积发生率有显著差异' if gray_p < 0.05 else '组间无显著差异'}）"
                )
            else:
                gray_msg = "；Gray 检验需 R cmprsk 包（未检测到/失败），仅给出描述性 CIF"

        diff_txt = (
            f"关注事件 {eoi} 的真实 CIF={cif_overall:.3f} < 朴素 1-KM={naive_1mkm:.3f}"
            f"（朴素法把竞争事件当删失，会高估 {naive_1mkm - cif_overall:.3f}）"
            if cif_overall <= naive_1mkm + 1e-9 else
            f"关注事件 {eoi} 的 CIF={cif_overall:.3f}，1-KM={naive_1mkm:.3f}"
        )
        summary.append(
            f"{entry.method} 完成：{len(sub)} 例，事件编码 {codes}（0=删失，{eoi}=关注，竞争={comp_codes}）；"
            f"时长列={dur_col}，事件列={event_col}"
            + (f"，分组={group_col}" if group_col else "")
            + f"。{diff_txt}{gray_msg}。CIF 表见 cif_table.csv、曲线见 cif_curves.png。"
            "⚠ CIF≠1-KM：把竞争事件朴素当删失会向上偏；竞争事件必须显式编码（≥2）；"
            "Gray 检验比较 CIF（边际发生率），与 cause-specific Cox（比较瞬时风险）是不同问题。"
        )
        code += [
            "from lifelines import AalenJohansenFitter  # 竞争风险 CIF",
            f"# ajf=AalenJohansenFitter(); ajf.fit(df['{dur_col}'], df['{event_col}'], event_of_interest={eoi})",
            "# ajf.cumulative_density_  # 关注事件的累积发生函数（非 1-KM）",
        ]
    except Exception as err:
        summary.append(f"竞争风险失败：{err}")


def _gray_test_via_r(sub, dur_col, event_col, group_col, eoi, d):
    """Gray's test for between-group CIF equality via R cmprsk::cuminc.

    Returns the p-value for the event-of-interest CIF comparison, or None if R /
    cmprsk is unavailable or the call fails (caller degrades to descriptive CIFs).
    Optional + graceful per the engine's R-bridge rules.
    """
    import pandas as pd

    from researchforge.executor import rbridge

    if not (rbridge.r_available() and rbridge.r_package_available("cmprsk")):
        return None
    # identifier guard before any name enters R
    if not all(_IDENT.fullmatch(str(c)) for c in (dur_col, event_col, group_col)):
        return None
    csv = d / "_cmprsk_input.csv"
    try:
        sub[[dur_col, event_col, group_col]].to_csv(csv, index=False)
        # cuminc(ftime, fstatus, group): Gray's test p-values are in $Tests, rows
        # indexed by event code (as character). Pull the row for the event of interest.
        rcode = f"""
        suppressMessages(library(cmprsk))
        d <- read.csv({_r_str(str(csv))}, check.names=FALSE)
        ci <- cuminc(d[["{dur_col}"]], d[["{event_col}"]], group=d[["{group_col}"]])
        tt <- ci$Tests
        key <- as.character({eoi})
        p <- if (key %in% rownames(tt)) tt[key, "pv"] else NA_real_
        cat(p)
        """
        out = rbridge.run_r(rcode)
        txt = (out or "").strip().split("\n")[-1].strip()
        return float(txt) if txt and txt.lower() not in {"na", "nan"} else None
    except Exception:
        return None
    finally:
        try:
            csv.unlink()
        except OSError:
            pass


def _r_str(s: str) -> str:
    """Quote a Python string as an R string literal (backslashes -> forward slashes)."""
    return '"' + s.replace("\\", "/").replace('"', '\\"') + '"'


@register("parametric_survival")
def _branch_parametric_survival(ctx: Ctx) -> None:
    """Parametric AFT survival models (Weibull / LogNormal / LogLogistic, lifelines).

    Fits the 3 AFT families, picks the best by AIC, and reports acceleration factors
    (exp(coef)) with 95% CI + p for the chosen model, the AIC comparison table, and a
    fitted-survival-curve plot. AFT models time directly (not the hazard).
    """
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    dur_col, event_col = _resolve_duration_event(df, fp, cfg)
    if dur_col is None or event_col is None:
        summary.append(
            "参数生存(AFT)失败：需要一个时长列（正数）+ 一个事件列（0/1，1=事件,0=删失）。"
            "用 config={\"duration\":..,\"event\":..} 指定。"
        )
        return
    try:
        from lifelines import (
            LogLogisticAFTFitter,
            LogNormalAFTFitter,
            WeibullAFTFitter,
        )

        excl = {fp.unit_col, dur_col, event_col}
        covariates = cfg.get("covariates")
        if covariates:
            covariates = [c for c in covariates if c in df.columns and c not in excl]
        else:
            covariates = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in excl
            ][:5]

        keep = [dur_col, event_col, *covariates]
        sub = df[list(dict.fromkeys(keep))].dropna()
        sub = sub[sub[dur_col].astype(float) > 0]
        sub[dur_col] = sub[dur_col].astype(float)
        sub[event_col] = sub[event_col].astype(int)
        if set(sub[event_col].unique()) - {0, 1}:
            raise ValueError(f"事件列 {event_col} 非 0/1 编码（应 1=事件,0=删失）")
        n_ev = int(sub[event_col].sum())
        if n_ev < 5:
            summary.append(f"参数生存(AFT)失败：事件数仅 {n_ev}（需 ≥5），参数模型不稳。")
            return

        fitters = {
            "Weibull": WeibullAFTFitter,
            "LogNormal": LogNormalAFTFitter,
            "LogLogistic": LogLogisticAFTFitter,
        }
        fitted: dict[str, object] = {}
        aics: dict[str, float] = {}
        fit_df = sub if covariates else sub[[dur_col, event_col]]
        for name, F in fitters.items():
            try:
                m = F()
                m.fit(fit_df, duration_col=dur_col, event_col=event_col)
                fitted[name] = m
                aics[name] = float(m.AIC_)
            except Exception:
                continue
        if not fitted:
            summary.append("参数生存(AFT)失败：三种分布均未能收敛。")
            return

        best_name = min(aics, key=aics.get)
        best = fitted[best_name]

        # AIC comparison table.
        aic_tab = (
            pd.DataFrame({"distribution": list(aics), "AIC": list(aics.values())})
            .sort_values("AIC")
            .reset_index(drop=True)
        )
        aic_tab["delta_AIC"] = (aic_tab["AIC"] - aic_tab["AIC"].min()).round(3)
        aic_tab["AIC"] = aic_tab["AIC"].round(3)
        aic_tab.to_csv(d / "aft_aic_comparison.csv", index=False, encoding="utf-8")
        files.append("aft_aic_comparison.csv")

        # Acceleration factors (exp(coef)) + 95% CI + p for the best model's covariate
        # block (the location-parameter regression; ancillary shape params excluded).
        bs = best.summary.copy()
        # lifelines indexes AFT summary by (param, covariate); keep the primary
        # location parameter ('lambda_' Weibull, 'mu_' LogNormal, 'alpha_' LogLog),
        # and drop the Intercept so reported AFs are interpretable covariate effects.
        loc_param = {"Weibull": "lambda_", "LogNormal": "mu_", "LogLogistic": "alpha_"}[best_name]
        af_rows = []
        for idx in bs.index:
            param = idx[0] if isinstance(idx, tuple) else None
            cov = idx[1] if isinstance(idx, tuple) else idx
            if param is not None and param != loc_param:
                continue
            if str(cov) == "Intercept" or str(cov) == "_intercept":
                continue
            row = bs.loc[idx]
            af = float(np.exp(row["coef"]))
            lo = float(np.exp(row["coef lower 95%"]))
            hi = float(np.exp(row["coef upper 95%"]))
            af_rows.append({
                "covariate": str(cov),
                "coef": round(float(row["coef"]), 4),
                "accel_factor": round(af, 4),
                "AF_lower95": round(lo, 4),
                "AF_upper95": round(hi, 4),
                "p_value": round(float(row["p"]), 4),
            })
            if str(cov) not in {"Intercept", "_intercept"}:
                estimates[f"AF_{cov}"] = round(af, 4)
        af_tab = pd.DataFrame(af_rows)
        if len(af_tab):
            af_tab.to_csv(d / "aft_acceleration_factors.csv", index=False, encoding="utf-8")
            files.append("aft_acceleration_factors.csv")

        estimates["best_aic"] = round(float(aics[best_name]), 3)
        estimates["n_events"] = float(n_ev)
        estimates["n_obs"] = float(len(sub))
        # record each fitter's AIC for diagnostics / tests
        for name, a in aics.items():
            estimates[f"aic_{name}"] = round(float(a), 3)

        # Fitted survival-curve plot (population-average for the best model).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4.5))
            tmax = float(sub[dur_col].max())
            times = np.linspace(0, tmax, 100)
            # population-average survival across the sample's covariate profiles
            sf = best.predict_survival_function(fit_df, times=times)
            mean_sf = sf.mean(axis=1)
            ax.plot(times, mean_sf.values, label=f"{best_name} AFT (mean)")
            ax.set_xlabel(f"{dur_col} (time)")
            ax.set_ylabel("survival probability")
            ax.set_ylim(0, 1)
            ax.set_title(f"Parametric AFT fitted survival ({best_name}, best AIC)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "aft_survival_curve.png", dpi=150)
            plt.close(fig)
            files.append("aft_survival_curve.png")
        except Exception:
            pass

        sig = [r["covariate"] for r in af_rows if r["p_value"] < 0.05]
        summary.append(
            f"{entry.method} 完成：{len(sub)} 例（{n_ev} 事件）；时长列={dur_col}，事件列={event_col}；"
            f"按 AIC 选出最优分布 = {best_name}（AIC={aics[best_name]:.1f}）；"
            f"AIC 比较见 aft_aic_comparison.csv，加速因子(exp(coef))+95%CI 见 aft_acceleration_factors.csv"
            + (f"；显著协变量：{sig}" if sig else "；无显著协变量")
            + "。⚠ AFT 直接建模事件时间：加速因子>1 表示延长存活/推迟事件（非风险比 HR）；"
            "分布选择影响结论（已报 AIC 比较）；这是参数假定，半参数 Cox 不需指定基线分布。"
        )
        code += [
            "from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter  # 参数 AFT",
            f"# m=WeibullAFTFitter(); m.fit(df, duration_col='{dur_col}', event_col='{event_col}')",
            "# 按 m.AIC_ 选最优分布；exp(m.summary['coef']) = 加速因子(acceleration factor)",
        ]
    except Exception as err:
        summary.append(f"参数生存(AFT)失败：{err}")


def _rmst_with_se(durations, events, tau: float):
    """RMST(tau) = area under the KM curve on [0, tau], with the Klein-Moeschberger /
    survRM2 estimator variance:
        Var = Σ_{t_i <= tau} [∫_{t_i}^{tau} Ŝ(u) du]² · d_i / (Y_i (Y_i − d_i))
    (sum over distinct EVENT times t_i; d_i deaths, Y_i at risk). This is the SAMPLING
    variance of the RMST estimator (it shrinks with n). It is NOT lifelines'
    restricted_mean_survival_time(return_variance=True), which returns the variance of
    the restricted-lifetime DISTRIBUTION (≈ the lifetime variance, ~θ², does not shrink
    with n) — misusing that as the estimator SE inflates the between-group test ~10× and
    masks real differences (inference-reviewer must-fix). Returns (rmst, se)."""
    import numpy as np
    from lifelines.utils import survival_table_from_events

    st = survival_table_from_events(
        np.asarray(durations, dtype=float), np.asarray(events, dtype=int)
    )
    st = st[st.index <= tau]
    times = st.index.to_numpy(dtype=float)
    at_risk = st["at_risk"].to_numpy(dtype=float)
    deaths = st["observed"].to_numpy(dtype=float)
    # KM survival just after each tabulated time; S=1 on [0, times[0])
    surv = np.cumprod(
        1.0 - np.divide(deaths, at_risk, out=np.zeros_like(deaths), where=at_risk > 0)
    )
    edges_aug = np.concatenate([[0.0], times, [tau]])   # interval edges, capped at tau
    svals = np.concatenate([[1.0], surv])               # S is svals[k] on [edges[k], edges[k+1])
    interval_area = np.empty(len(svals))
    for k in range(len(svals)):
        a = min(edges_aug[k], tau)
        b = min(edges_aug[k + 1], tau)
        interval_area[k] = svals[k] * max(b - a, 0.0)
    rmst = float(interval_area.sum())
    starts = edges_aug[:-1]                              # start time of each interval
    var = 0.0
    for i, ti in enumerate(times):
        if deaths[i] <= 0 or at_risk[i] <= deaths[i]:
            continue
        area_from = float(interval_area[starts >= ti].sum())  # ∫_{t_i}^{tau} Ŝ du
        var += (area_from ** 2) * deaths[i] / (at_risk[i] * (at_risk[i] - deaths[i]))
    se = float(np.sqrt(var)) if var >= 0 else float("nan")
    return rmst, se


@register("rmst")
def _branch_rmst(ctx: Ctx) -> None:
    """Restricted Mean Survival Time (RMST) — assumption-light alternative to the HR.

    RMST(tau) = area under the KM curve on [0, tau] = mean event-free time within
    that window (no proportional-hazards needed). With a 2-level group we report the
    between-group RMST DIFFERENCE with a 95% CI and a z-test p-value. tau must be
    common to all groups; default = min over groups of the largest *event* time.
    """
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    dur_col, event_col = _resolve_duration_event(df, fp, cfg)
    if dur_col is None or event_col is None:
        summary.append(
            "RMST 失败：需要一个时长列（正数）+ 一个事件列（0/1，1=事件,0=删失）。"
            "用 config={\"duration\":..,\"event\":..} 指定。"
        )
        return
    try:
        from lifelines import KaplanMeierFitter

        excl = {fp.unit_col, dur_col, event_col}
        group_col = _resolve_group(df, fp, cfg, excl, n_levels=2)

        keep = [dur_col, event_col] + ([group_col] if group_col else [])
        sub = df[list(dict.fromkeys(keep))].dropna()
        sub = sub[sub[dur_col].astype(float) > 0]
        sub[dur_col] = sub[dur_col].astype(float)
        sub[event_col] = sub[event_col].astype(int)
        if set(sub[event_col].unique()) - {0, 1}:
            raise ValueError(f"事件列 {event_col} 非 0/1 编码（应 1=事件,0=删失）")
        if len(sub) < 10:
            summary.append(f"RMST 失败：有效样本仅 {len(sub)}（需 ≥10）。")
            return

        groups = (list(sub.groupby(group_col)) if group_col else [(None, sub)])

        # tau: default = min over groups of each group's largest EVENT time (the
        # largest tau where every group still has informative follow-up). Judgment
        # call: using the largest event time (not the largest *observed* time, which
        # could be a late censoring) keeps RMST estimable in every group; disclosed.
        def _last_event_time(gdf):
            ev_times = gdf.loc[gdf[event_col] == 1, dur_col]
            return float(ev_times.max()) if len(ev_times) else float(gdf[dur_col].max())

        auto_tau = min(_last_event_time(gdf) for _, gdf in groups)
        tau = float(cfg.get("tau", auto_tau))
        tau_was_clamped = False
        if cfg.get("tau") is not None and tau > auto_tau:
            # a user tau beyond a group's follow-up makes RMST extrapolate flat;
            # clamp to the common-support tau and disclose.
            tau = auto_tau
            tau_was_clamped = True
        estimates["tau"] = round(tau, 4)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        results = {}  # gval -> (rmst, se)
        rows = []
        for gval, gdf in groups:
            kmf = KaplanMeierFitter()
            kmf.fit(gdf[dur_col], gdf[event_col],
                    label=("all" if gval is None else f"{group_col}={gval}"))
            # RMST point + Klein-Moeschberger estimator SE (NOT lifelines return_variance,
            # which is the distributional variance ~θ², not the sampling SE — see helper).
            rmst, se = _rmst_with_se(gdf[dur_col], gdf[event_col], tau)
            results[gval] = (float(rmst), se)
            rows.append({"group": "all" if gval is None else gval,
                         "n": int(len(gdf)),
                         "n_events": int(gdf[event_col].sum()),
                         "tau": round(tau, 4),
                         "RMST": round(float(rmst), 4),
                         "SE": round(se, 4)})
            # KM curve + shaded restricted area up to tau
            kmf.plot_survival_function(ax=ax, ci_show=False)
            sf = kmf.survival_function_
            t_idx = sf.index.values
            y = sf.iloc[:, 0].values
            mask = t_idx <= tau
            if mask.any():
                ax.fill_between(t_idx[mask], 0, y[mask], step="post", alpha=0.15)

        ax.axvline(tau, color="grey", linestyle="--", linewidth=1, label=f"tau={tau:.3g}")
        ax.set_xlabel(f"{dur_col} (time)")
        ax.set_ylabel("survival probability")
        ax.set_ylim(0, 1)
        ax.set_title("RMST: KM curves with restricted area up to tau")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(d / "rmst_curves.png", dpi=150)
        plt.close(fig)
        files.append("rmst_curves.png")

        import pandas as pd

        rmst_tab = pd.DataFrame(rows)
        rmst_tab.to_csv(d / "rmst_table.csv", index=False, encoding="utf-8")
        files.append("rmst_table.csv")

        diff_msg = ""
        if group_col is not None and len(results) == 2:
            (g0, (r0, se0)), (g1, (r1, se1)) = list(results.items())
            diff = r1 - r0
            se_diff = float(np.sqrt(se0**2 + se1**2))
            if np.isfinite(se_diff) and se_diff > 0:
                from scipy.stats import norm

                z = diff / se_diff
                p = float(2 * (1 - norm.cdf(abs(z))))
                lo, hi = diff - 1.96 * se_diff, diff + 1.96 * se_diff
                estimates["rmst_diff"] = round(diff, 4)
                estimates["rmst_diff_se"] = round(se_diff, 4)
                estimates["rmst_diff_ci_low"] = round(lo, 4)
                estimates["rmst_diff_ci_high"] = round(hi, 4)
                estimates["rmst_diff_p"] = round(p, 4)
                estimates[f"rmst_{g0}"] = round(r0, 4)
                estimates[f"rmst_{g1}"] = round(r1, 4)
                diff_msg = (
                    f"；RMST 差异（{group_col}={g1} − {g0}）= {diff:.3f}"
                    f"（95%CI [{lo:.3f}, {hi:.3f}]，z 检验 p={p:.3g}，"
                    f"{'组间差异显著' if p < 0.05 else '组间差异不显著'}）"
                )
        elif group_col is None:
            (g, (r, se)), = list(results.items())
            estimates["rmst"] = round(r, 4)
            estimates["rmst_se"] = round(se, 4)

        clamp_note = (
            f"（用户给的 tau 超出某组随访，已夹到共同支撑 tau={tau:.3g} 并披露）"
            if tau_was_clamped else ""
        )
        summary.append(
            f"{entry.method} 完成：{len(sub)} 例；时长列={dur_col}，事件列={event_col}"
            + (f"，分组={group_col}" if group_col else "")
            + f"；tau={tau:.3g}{clamp_note}（取各组最大事件时间的最小值，保证组间共同支撑）"
            f"；RMST 表见 rmst_table.csv，曲线（阴影=[0,tau] 受限面积）见 rmst_curves.png{diff_msg}。"
            "⚠ RMST=「[0,tau] 内的平均无事件时间」，是免比例风险(PH)假定的 HR 替代；"
            "tau 须为各组共同支撑并已报告；不同 tau 会得到不同结论。"
        )
        code += [
            "from lifelines.utils import survival_table_from_events  # RMST = ∫_0^tau S(t)dt",
            "# RMST(tau)=KM 曲线在 [0,tau] 下的面积；估计量方差(Klein-Moeschberger/survRM2)：",
            "# Var=Σ_{t_i<=tau} [∫_{t_i}^{tau} S(u)du]^2 · d_i/(Y_i(Y_i-d_i))  (随 n 收缩)",
            "# 组间差异 SE = sqrt(se1^2+se2^2)，z 检验",
        ]
    except Exception as err:
        summary.append(f"RMST 失败：{err}")

