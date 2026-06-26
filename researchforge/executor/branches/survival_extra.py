"""Branch handlers for the survival-analysis completion family (survival_extra).

Three lifelines-backed Cox extensions that finish the survival family beyond the
basic KM + Cox in survival.py:

  * time_varying_cox    — Cox PH with TIME-VARYING covariates (CoxTimeVaryingFitter)
                          on counting-process long data (id + start/stop + event).
  * stratified_cox      — CoxPHFitter with strata: separate baseline hazards per
                          stratum, common covariate coefficients.
  * cox_ph_diagnostics  — fit CoxPHFitter, then test the proportional-hazards
                          assumption via scaled Schoenfeld residuals
                          (lifelines.statistics.proportional_hazard_test).

Each handler resolves its column roles (cfg overrides, else profiler kinds + name
hints, MIRRORING survival.py's idiom — note the duration IS the time column, so we
do NOT exclude fp.time_col), degrades honestly (no lifelines / wrong data shape /
too few rows-events-columns -> skip with a Chinese ⚠ message; never crash), writes
CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float `estimates`, appends a
Chinese `summary` ending with ⚠ disclosures, and mutates ctx (never rebinds).

config keys (per analysis):
  time_varying_cox : id, start, stop, event, covariates
  stratified_cox   : duration, event, strata, covariates
  cox_ph_diagnostics : duration, event, covariates

lifelines is installed. See executor/_branch_api.py, executor/branches/survival.py,
and CLAUDE.md (引擎约定 / survival caveat).
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register

# Column-name hints (shared with survival.py's idiom).
_DUR_KWS = ["duration", "time", "days", "month", "year", "tenure", "surviv",
            "followup", "fu", "week", "age_at"]
_EV_KWS = ["event", "status", "death", "dead", "fail", "censor", "relapse"]
_START_KWS = ["start", "begin", "entry", "from", "t0", "tstart", "t_start"]
_STOP_KWS = ["stop", "end", "exit", "to", "t1", "tstop", "t_stop", "finish"]
_ID_KWS = ["id", "subject", "patient", "unit", "case", "person"]
_STRATA_KWS = ["strat", "group", "arm", "site", "center", "centre", "region",
               "hospital", "clinic", "sex", "male", "cohort", "treat"]
_IDENT = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")


def _hr_table(cs):
    """Extract a tidy HR table (coef / hazard_ratio / HR_lower95 / HR_upper95 /
    p_value) from a lifelines Cox `.summary`. Prefers the native exp(coef) CI
    columns; falls back to exponentiating the coef CI columns if a lifelines
    version omits them (version-robustness — cannot run-verify in this sandbox)."""
    import numpy as np
    import pandas as pd

    coef = cs["coef"].to_numpy(dtype=float)
    hr = cs["exp(coef)"].to_numpy(dtype=float)
    if "exp(coef) lower 95%" in cs.columns and "exp(coef) upper 95%" in cs.columns:
        lo = cs["exp(coef) lower 95%"].to_numpy(dtype=float)
        hi = cs["exp(coef) upper 95%"].to_numpy(dtype=float)
    else:
        lo = np.exp(cs["coef lower 95%"].to_numpy(dtype=float))
        hi = np.exp(cs["coef upper 95%"].to_numpy(dtype=float))
    p = cs["p"].to_numpy(dtype=float)
    return pd.DataFrame(
        {"coef": coef, "hazard_ratio": hr, "HR_lower95": lo, "HR_upper95": hi, "p_value": p},
        index=cs.index,
    )


def _hr_forest_plot(d, files, hr_tab, title, fname):
    """Forest plot of hazard ratios with 95% CI on a LOG x-axis.

    hr_tab is the tidy table from _hr_table (columns hazard_ratio / HR_lower95 /
    HR_upper95, indexed by covariate name). Best-effort (try/except in the caller):
    ENGLISH labels, Agg backend. Appends fname to files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    covs = list(hr_tab.index.astype(str))
    hr = hr_tab["hazard_ratio"].to_numpy(dtype=float)
    lo = hr_tab["HR_lower95"].to_numpy(dtype=float)
    hi = hr_tab["HR_upper95"].to_numpy(dtype=float)
    y = np.arange(len(covs))[::-1]  # first covariate at top

    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.55 * len(covs) + 1.5)))
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color="#3b6", capsize=3, lw=1.2)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(covs)
    ax.set_xscale("log")
    ax.set_xlabel("hazard ratio exp(coef) (log scale, 95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(d / fname, dpi=150)
    plt.close(fig)
    files.append(fname)


def _resolve_duration_event(df, fp, cfg):
    """Resolve (dur_col, event_col) — same idiom as survival.py (duration IS the
    time column, so fp.time_col is NOT excluded). event = binary 0/1 (name-hint
    preferred); duration = positive continuous/count/id column (name-hint preferred).
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


# ─────────────────────────────────────────────────────────────────────────────
# (A) time_varying_cox — Cox PH with time-varying covariates (counting process).
# ─────────────────────────────────────────────────────────────────────────────
@register("time_varying_cox")
def _branch_time_varying_cox(ctx: Ctx) -> None:
    """Cox PH with TIME-VARYING covariates via CoxTimeVaryingFitter.

    Input must be COUNTING-PROCESS long data: one row per (subject, interval) with
    an id column, a start (entry) + stop (exit) time, an event indicator that is 1
    only on the interval where the event occurs (0 otherwise), and time-varying
    covariates that may change between a subject's intervals. We resolve:
      id        = unit/id column (cfg id, else fp.unit_col or a name-hinted id col),
      start/stop= two time-like columns (cfg start/stop, else name-hinted / detect),
      event     = a binary 0/1 column (cfg event, else name-hint),
      covariates= remaining numeric columns (cfg covariates, else auto).
    Fits CoxTimeVaryingFitter().fit(df, id_col, event_col, start_col, stop_col).
    """
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    EXPECTED = (
        "需要计数过程(counting-process)长表：每行一个 (受试者, 时间区间)——"
        "id 列 + 区间起点 start + 终点 stop + 事件指示(仅在事件发生的那一区间=1) "
        "+ 随时间变化的协变量。用 config={\"id\":..,\"start\":..,\"stop\":..,\"event\":..,\"covariates\":[..]} 指定。"
    )

    def _named(kws, kinds, excl):
        return next(
            (c.name for c in fp.columns
             if any(k in c.name.lower() for k in kws) and c.kind in kinds and c.name not in excl),
            None,
        )

    # id column
    id_col = cfg.get("id") or fp.unit_col or _named(_ID_KWS, {"id", "categorical", "count"}, set())
    excl = {id_col} if id_col else set()

    # event = a binary 0/1 column (name-hint preferred)
    event_col = cfg.get("event") or _named(_EV_KWS, {"binary"}, excl) or next(
        (c.name for c in fp.columns if c.kind == "binary" and c.name not in excl), None
    )
    excl_se = excl | ({event_col} if event_col else set())

    # start / stop = two time-like numeric columns
    time_kinds = {"continuous", "count", "id"}
    start_col = cfg.get("start") or _named(_START_KWS, time_kinds, excl_se)
    stop_col = cfg.get("stop") or _named(_STOP_KWS, time_kinds, excl_se | ({start_col} if start_col else set()))
    # fall back: if exactly one of start/stop named, find another time-like column;
    # if neither named, take the first two name-hinted-by-duration / numeric columns.
    if start_col is None or stop_col is None:
        time_cands = [
            c.name for c in fp.columns
            if c.kind in time_kinds and c.name not in excl_se
        ]
        # prefer duration-keyword columns first
        time_cands.sort(key=lambda nm: 0 if any(k in nm.lower() for k in _DUR_KWS + _START_KWS + _STOP_KWS) else 1)
        for cand in time_cands:
            if cand in {start_col, stop_col}:
                continue
            if start_col is None:
                start_col = cand
            elif stop_col is None and cand != start_col:
                stop_col = cand

    if id_col is None or event_col is None or start_col is None or stop_col is None:
        summary.append(f"时变 Cox 失败：缺 id / start / stop / event 列之一。{EXPECTED}")
        return

    try:
        from lifelines import CoxTimeVaryingFitter
    except Exception:
        summary.append("时变 Cox 跳过：缺 lifelines（pip install lifelines）。")
        return

    try:
        used = {id_col, event_col, start_col, stop_col}
        covariates = cfg.get("covariates")
        if covariates:
            covariates = [c for c in covariates if c in df.columns and c not in used]
        else:
            covariates = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "binary", "count"} and c.name not in used
            ][:6]
        if not covariates:
            summary.append(f"时变 Cox 失败：未找到协变量（除 id/start/stop/event 外的数值列）。{EXPECTED}")
            return

        keep = [id_col, start_col, stop_col, event_col, *covariates]
        sub = df[list(dict.fromkeys(keep))].dropna().copy()
        sub[start_col] = sub[start_col].astype(float)
        sub[stop_col] = sub[stop_col].astype(float)
        sub[event_col] = sub[event_col].astype(int)

        # ── counting-process shape validation (honest degrade if not long format) ──
        if set(sub[event_col].unique()) - {0, 1}:
            summary.append(
                f"时变 Cox 失败：事件列 {event_col} 非 0/1 编码（应在事件区间=1、其余=0）。{EXPECTED}"
            )
            return
        # intervals must satisfy stop > start
        bad_iv = (sub[stop_col] <= sub[start_col]).sum()
        if bad_iv > 0:
            summary.append(
                f"时变 Cox 失败：{int(bad_iv)} 行的 stop({stop_col}) <= start({start_col})，"
                f"不是有效的计数过程区间。{EXPECTED}"
            )
            return
        n_subjects = int(sub[id_col].nunique())
        n_intervals = int(len(sub))
        # genuine long format: more intervals than subjects (≥1 subject has multiple rows)
        if n_intervals <= n_subjects:
            summary.append(
                f"时变 Cox 失败：区间数({n_intervals}) <= 受试者数({n_subjects})，"
                f"数据似乎不是计数过程长表（应有受试者带多个时间区间）。{EXPECTED}"
            )
            return
        n_events = int(sub[event_col].sum())
        if n_events < max(5, 2 * len(covariates)):
            summary.append(
                f"时变 Cox 失败：事件数仅 {n_events}（需 ≥{max(5, 2 * len(covariates))}），"
                f"协变量 {len(covariates)} 个，模型不稳。{EXPECTED}"
            )
            return

        ctv = CoxTimeVaryingFitter()
        ctv.fit(
            sub,
            id_col=id_col,
            event_col=event_col,
            start_col=start_col,
            stop_col=stop_col,
            show_progress=False,
        )
        cs = ctv.summary  # indexed by covariate

        # per-covariate HR table
        ctab = _hr_table(cs)
        ctab.round(4).to_csv(d / "tv_cox_hazard_ratios.csv", encoding="utf-8")
        files.append("tv_cox_hazard_ratios.csv")

        for cv in covariates:
            if cv in cs.index:
                estimates[f"HR_{cv}"] = round(float(cs.loc[cv, "exp(coef)"]), 4)

        loglik = float(getattr(ctv, "log_likelihood_", float("nan")))
        try:
            concord = float(ctv.concordance_index_)
        except Exception:
            concord = float("nan")
        # report max |HR| on the natural scale (max of HR and 1/HR) — "biggest effect"
        max_hr_natural = float(np.nanmax(
            [max(v, 1.0 / v) for v in cs["exp(coef)"].to_numpy(dtype=float) if v > 0]
        )) if len(cs) else float("nan")

        estimates["loglik"] = round(loglik, 4) if np.isfinite(loglik) else float("nan")
        estimates["n_covariates"] = float(len(covariates))
        estimates["n_intervals"] = float(n_intervals)
        estimates["max_abs_hr"] = round(max_hr_natural, 4) if np.isfinite(max_hr_natural) else float("nan")
        estimates["n_subjects"] = float(n_subjects)
        if np.isfinite(concord):
            estimates["concordance"] = round(concord, 4)

        # forest plot of HRs (log scale)
        try:
            _hr_forest_plot(
                d, files, ctab,
                "Time-varying Cox: hazard ratios (95% CI)",
                "tv_cox_forest.png",
            )
        except Exception:
            pass

        sig = [cv for cv in covariates if cv in cs.index and float(cs.loc[cv, "p"]) < 0.05]
        summary.append(
            f"{entry.method} 完成：{n_subjects} 受试者 / {n_intervals} 区间（{n_events} 事件）；"
            f"id={id_col}，区间=[{start_col},{stop_col}]，事件={event_col}，协变量={covariates}；"
            f"偏对数似然={loglik:.2f}"
            + (f"，C-index={concord:.3f}" if np.isfinite(concord) else "")
            + "；风险比见 tv_cox_hazard_ratios.csv、森林图见 tv_cox_forest.png"
            + (f"；显著协变量：{sig}" if sig else "；无显著协变量")
            + "。⚠ 需计数过程(start,stop,event)长表——若数据不是该形状结果无效；"
            "时变协变量须为「外生」(其取值不被事件风险反向影响)，否则系数有偏；"
            "PH 假定仍对时变协变量成立；事件须 0/1(仅事件区间=1)。"
        )
        code += [
            "from lifelines import CoxTimeVaryingFitter  # 时变协变量 Cox（计数过程长表）",
            f"# ctv=CoxTimeVaryingFitter(); ctv.fit(df, id_col='{id_col}', event_col='{event_col}',",
            f"#   start_col='{start_col}', stop_col='{stop_col}')  # exp(coef)=风险比",
        ]
    except Exception as err:
        summary.append(f"时变 Cox 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) stratified_cox — CoxPHFitter with strata (per-stratum baseline hazards).
# ─────────────────────────────────────────────────────────────────────────────
@register("stratified_cox")
def _branch_stratified_cox(ctx: Ctx) -> None:
    """Stratified Cox PH: separate baseline hazards per stratum, common coefficients.

    Resolves duration + event (survival.py idiom), a stratum column (cfg strata, else
    a low-cardinality categorical/binary), covariates = remaining numeric. Fits
    CoxPHFitter().fit(df, duration_col, event_col, strata=[stratum]).
    """
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    dur_col, event_col = _resolve_duration_event(df, fp, cfg)
    if dur_col is None or event_col is None:
        summary.append(
            "分层 Cox 失败：需要一个时长列（正数）+ 一个事件列（0/1，1=事件,0=删失）。"
            "用 config={\"duration\":..,\"event\":..,\"strata\":..} 指定。"
        )
        return

    try:
        from lifelines import CoxPHFitter
    except Exception:
        summary.append("分层 Cox 跳过：缺 lifelines（pip install lifelines）。")
        return

    try:
        _excl = {fp.unit_col, dur_col, event_col}
        # stratum column: cfg override, else a low-cardinality categorical/binary
        strata = cfg.get("strata")
        if strata and strata in df.columns and strata not in {dur_col, event_col}:
            stratum = strata
        else:
            stratum = None
            named = [
                c.name for c in fp.columns
                if c.kind in {"binary", "categorical"} and c.name not in _excl
                and 2 <= df[c.name].dropna().nunique() <= max(2, len(df) // 10)
            ]
            # prefer a name-hinted stratum (site/group/sex/...), else first low-card categorical
            pref = [c for c in named if any(k in c.lower() for k in _STRATA_KWS)]
            cand_list = pref or named
            if cand_list:
                stratum = cand_list[0]
        if stratum is None:
            summary.append(
                "分层 Cox 失败：未找到分层变量（低基数分类/二值列）。"
                "用 config={\"strata\":\"<列>\"} 指定。"
            )
            return

        covariates = cfg.get("covariates")
        if covariates:
            covariates = [c for c in covariates if c in df.columns and c not in _excl | {stratum}]
        else:
            covariates = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in _excl | {stratum}
            ][:6]
        if not covariates:
            summary.append(
                "分层 Cox 失败：除分层变量外未找到协变量（连续/二值数值列）。"
                "用 config={\"covariates\":[..]} 指定。"
            )
            return

        keep = [dur_col, event_col, stratum, *covariates]
        sub = df[list(dict.fromkeys(keep))].dropna().copy()
        sub = sub[sub[dur_col].astype(float) > 0]
        sub[dur_col] = sub[dur_col].astype(float)
        sub[event_col] = sub[event_col].astype(int)
        if set(sub[event_col].unique()) - {0, 1}:
            summary.append(f"分层 Cox 失败：事件列 {event_col} 非 0/1 编码（应 1=事件,0=删失）。")
            return
        n_ev = int(sub[event_col].sum())
        n_strata = int(sub[stratum].nunique())
        if n_strata < 2:
            summary.append(
                f"分层 Cox 失败：分层变量 {stratum} 只有 {n_strata} 个水平（需 ≥2）。"
                "无分层时请改用 survival_analysis（普通 Cox）。"
            )
            return
        if n_ev < max(5, 2 * len(covariates)):
            summary.append(
                f"分层 Cox 失败：事件数仅 {n_ev}（需 ≥{max(5, 2 * len(covariates))}），"
                f"协变量 {len(covariates)} 个，模型不稳。"
            )
            return

        cph = CoxPHFitter()
        cph.fit(sub[[dur_col, event_col, stratum, *covariates]], dur_col, event_col, strata=[stratum])
        cs = cph.summary

        ctab = _hr_table(cs)
        ctab.round(4).to_csv(d / "stratified_cox_hazard_ratios.csv", encoding="utf-8")
        files.append("stratified_cox_hazard_ratios.csv")

        for cv in covariates:
            if cv in cs.index:
                estimates[f"HR_{cv}"] = round(float(cs.loc[cv, "exp(coef)"]), 4)

        loglik = float(cph.log_likelihood_)
        try:
            concord = float(cph.concordance_index_)
        except Exception:
            concord = float("nan")
        max_hr_natural = float(np.nanmax(
            [max(v, 1.0 / v) for v in cs["exp(coef)"].to_numpy(dtype=float) if v > 0]
        )) if len(cs) else float("nan")

        estimates["loglik"] = round(loglik, 4)
        estimates["concordance"] = round(concord, 4) if np.isfinite(concord) else float("nan")
        estimates["n_strata"] = float(n_strata)
        estimates["n_covariates"] = float(len(covariates))
        estimates["max_abs_hr"] = round(max_hr_natural, 4) if np.isfinite(max_hr_natural) else float("nan")
        estimates["n_events"] = float(n_ev)

        try:
            _hr_forest_plot(
                d, files, ctab,
                f"Stratified Cox (by {stratum}): hazard ratios (95% CI)",
                "stratified_cox_forest.png",
            )
        except Exception:
            pass

        sig = [cv for cv in covariates if cv in cs.index and float(cs.loc[cv, "p"]) < 0.05]
        summary.append(
            f"{entry.method} 完成：{len(sub)} 例（{n_ev} 事件）；时长列={dur_col}，事件列={event_col}，"
            f"分层变量={stratum}（{n_strata} 层）；偏对数似然={loglik:.2f}"
            + (f"，C-index={concord:.3f}" if np.isfinite(concord) else "")
            + "；风险比见 stratified_cox_hazard_ratios.csv、森林图见 stratified_cox_forest.png"
            + (f"；显著协变量：{sig}" if sig else "；无显著协变量")
            + f"。⚠ 分层让基线风险在各层(by {stratum})自由不同（非参数地控制该分类混杂），"
            "但假定各层协变量系数相同——不估计 层×协变量 交互；若交互重要，应改建交互模型；"
            "事件须 0/1（1=事件,0=删失），删失假定随机非信息。"
        )
        code += [
            "from lifelines import CoxPHFitter  # 分层 Cox（每层独立基线风险，系数共享）",
            f"# cph=CoxPHFitter(); cph.fit(df, '{dur_col}', '{event_col}', strata=['{stratum}'])",
        ]
    except Exception as err:
        summary.append(f"分层 Cox 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) cox_ph_diagnostics — proportional-hazards test (scaled Schoenfeld residuals).
# ─────────────────────────────────────────────────────────────────────────────
@register("cox_ph_diagnostics")
def _branch_cox_ph_diagnostics(ctx: Ctx) -> None:
    """Fit CoxPHFitter, then TEST the proportional-hazards assumption via scaled
    Schoenfeld residuals (lifelines.statistics.proportional_hazard_test, rank time
    transform). Reports per-covariate PH test stat + p, the global test (nan when
    lifelines only gives per-covariate), and which covariates VIOLATE PH (p<0.05).
    """
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    dur_col, event_col = _resolve_duration_event(df, fp, cfg)
    if dur_col is None or event_col is None:
        summary.append(
            "比例风险诊断失败：需要一个时长列（正数）+ 一个事件列（0/1，1=事件,0=删失）。"
            "用 config={\"duration\":..,\"event\":..,\"covariates\":[..]} 指定。"
        )
        return

    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except Exception:
        summary.append("比例风险诊断跳过：缺 lifelines（pip install lifelines）。")
        return

    try:
        _excl = {fp.unit_col, dur_col, event_col}
        covariates = cfg.get("covariates")
        if covariates:
            covariates = [c for c in covariates if c in df.columns and c not in _excl]
        else:
            covariates = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in _excl
            ][:6]
        if not covariates:
            summary.append(
                "比例风险诊断失败：未找到协变量（连续/二值数值列）。"
                "用 config={\"covariates\":[..]} 指定。"
            )
            return

        keep = [dur_col, event_col, *covariates]
        sub = df[list(dict.fromkeys(keep))].dropna().copy()
        sub = sub[sub[dur_col].astype(float) > 0]
        sub[dur_col] = sub[dur_col].astype(float)
        sub[event_col] = sub[event_col].astype(int)
        if set(sub[event_col].unique()) - {0, 1}:
            summary.append(f"比例风险诊断失败：事件列 {event_col} 非 0/1 编码（应 1=事件,0=删失）。")
            return
        n_ev = int(sub[event_col].sum())
        if n_ev < max(5, 2 * len(covariates)):
            summary.append(
                f"比例风险诊断失败：事件数仅 {n_ev}（需 ≥{max(5, 2 * len(covariates))}），"
                f"协变量 {len(covariates)} 个，Cox 拟合/PH 检验不稳。"
            )
            return

        fit_df = sub[[dur_col, event_col, *covariates]]
        cph = CoxPHFitter()
        cph.fit(fit_df, dur_col, event_col)
        try:
            concord = float(cph.concordance_index_)
        except Exception:
            concord = float("nan")

        # Schoenfeld-residual PH test, rank time transform.
        ph = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_sum = ph.summary.copy()  # indexed per covariate; columns include test_statistic, p

        # lifelines may multi-index by (covariate, transform); flatten to covariate.
        if isinstance(ph_sum.index, pd.MultiIndex):
            ph_sum = ph_sum.reset_index()
            cov_col = ph_sum.columns[0]
            ph_sum = ph_sum.set_index(cov_col)

        pcol = "p" if "p" in ph_sum.columns else ("p_value" if "p_value" in ph_sum.columns else None)
        scol = "test_statistic" if "test_statistic" in ph_sum.columns else None
        if pcol is None:
            # last resort: take the last numeric column as p
            pcol = ph_sum.select_dtypes("number").columns[-1]

        rows = []
        for cov in ph_sum.index.astype(str):
            pval = float(ph_sum.loc[cov, pcol])
            stat = float(ph_sum.loc[cov, scol]) if scol else float("nan")
            rows.append({
                "covariate": cov,
                "ph_test_statistic": round(stat, 4) if np.isfinite(stat) else float("nan"),
                "ph_p_value": round(pval, 4),
                "violates_PH": bool(pval < 0.05),
            })
        ph_tab = pd.DataFrame(rows)
        ph_tab.to_csv(d / "ph_test_table.csv", index=False, encoding="utf-8")
        files.append("ph_test_table.csv")

        pvals = ph_tab["ph_p_value"].to_numpy(dtype=float)
        violations = [r["covariate"] for r in rows if r["violates_PH"]]
        n_violations = int(len(violations))
        min_p = float(np.nanmin(pvals)) if len(pvals) else float("nan")

        # global PH test: lifelines' proportional_hazard_test gives per-covariate only;
        # report nan (honest) unless a global row is present.
        global_p = float("nan")
        idx_strs = [str(x) for x in ph_sum.index]
        for key in ("global", "all", "GLOBAL"):
            if key in idx_strs:
                global_p = float(ph_sum.loc[key, pcol])
                break

        estimates["global_ph_p"] = round(global_p, 4) if np.isfinite(global_p) else float("nan")
        estimates["n_violations"] = float(n_violations)
        estimates["min_ph_p"] = round(min_p, 4) if np.isfinite(min_p) else float("nan")
        estimates["n_covariates"] = float(len(covariates))
        estimates["concordance"] = round(concord, 4) if np.isfinite(concord) else float("nan")

        # bar of -log10(PH p) per covariate with the 0.05 threshold line.
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            covs = ph_tab["covariate"].tolist()
            neglog = -np.log10(np.clip(pvals, 1e-12, 1.0))
            colors = ["#d33" if v else "#3b6" for v in ph_tab["violates_PH"]]
            ypos = list(range(len(covs)))[::-1]
            fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.5 * len(covs) + 1.5)))
            ax.barh(ypos, neglog, color=colors)
            ax.set_yticks(ypos)
            ax.set_yticklabels(covs)
            ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=1,
                       label="p = 0.05 threshold")
            ax.set_xlabel("-log10(PH test p-value)")
            ax.set_title("Proportional-hazards test (scaled Schoenfeld, rank)")
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(d / "ph_test_bar.png", dpi=150)
            plt.close(fig)
            files.append("ph_test_bar.png")
        except Exception:
            pass

        verdict = (
            f"⚠ {n_violations} 个协变量违反 PH（p<0.05）：{violations}"
            if n_violations else "所有协变量均满足 PH 假定（p≥0.05）"
        )
        summary.append(
            f"{entry.method} 完成：{len(sub)} 例（{n_ev} 事件）；时长列={dur_col}，事件列={event_col}，"
            f"协变量={covariates}；比例风险检验（缩放 Schoenfeld 残差，rank 时间变换）"
            f"最小 p={min_p:.3g}；{verdict}；明细见 ph_test_table.csv、"
            "图见 ph_test_bar.png（条=−log10(p)，红=违反，虚线=0.05）。"
            "⚠ Schoenfeld 检验显著=该协变量风险比随时间变化（PH 不成立）→ 可改用 "
            "stratified_cox / time_varying_cox 或加 时间×协变量 交互项；检验依赖时间变换"
            "（此处用 rank）；lifelines 仅给逐协变量检验，无内置全局检验(global_ph_p=nan)。"
        )
        code += [
            "from lifelines import CoxPHFitter  # 比例风险(PH)诊断",
            "from lifelines.statistics import proportional_hazard_test",
            f"# cph=CoxPHFitter().fit(df[['{dur_col}','{event_col}',*covs]], '{dur_col}', '{event_col}')",
            "# proportional_hazard_test(cph, df, time_transform='rank').summary  # p<0.05 = 违反 PH",
        ]
    except Exception as err:
        summary.append(f"比例风险诊断失败：{err}")
