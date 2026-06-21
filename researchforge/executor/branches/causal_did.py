"""Callaway & Sant'Anna (2021) staggered DiD — group-time ATT via R `did`.

A separate causal family file (causal.py is near the size limit). Auto-registered
by branches/__init__'s pkgutil walk. The single public branch
`callaway_santanna` estimates ATT(g,t) with did::att_gt and aggregates with
did::aggte (overall / dynamic event-study / per-cohort). R is OPTIONAL with an
honest degrade pointing to the pure-Python staggered_did (Sun-Abraham) and
event_study branches.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _callaway_santanna_via_r(
    csv_path, yname: str, tname: str, idname: str, gname: str,
    control_group: str = "nevertreated", est_method: str = "dr",
):
    """Callaway & Sant'Anna (2021) via R `did` (the gold standard).

    Fits group-time ATT(g,t) with did::att_gt (bstrap=TRUE for the multiplier
    bootstrap, cband=FALSE so SEs are pointwise), then aggregates with did::aggte
    three ways:
      * type="simple"  -> a single overall ATT + SE   ($overall.att / $overall.se)
      * type="dynamic" -> the EVENT-STUDY: ATT by length-of-exposure e
                          ($egt = e grid, $att.egt = ATT(e), $se.egt = SE(e))
      * type="group"   -> per-cohort ATT
                          ($egt = cohort g, $att.egt = ATT(g), $se.egt = SE(g))

    Slot names verified against `did` v2.x (att_gt/aggte). We read them in R and
    emit labelled `key|...` lines so Python parsing is version-robust. Column
    names are identifier-guarded by the caller; the input CSV is written by the
    caller into the output dir and deleted in its `finally`. No network in R.
    Raises so the caller can degrade honestly. Returns (overall dict, event-study
    DataFrame, per-cohort DataFrame).
    """
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    cg = '"notyettreated"' if control_group == "notyettreated" else '"nevertreated"'
    em = '"' + (est_method if est_method in ("dr", "ipw", "reg") else "dr") + '"'
    rcode = (
        "suppressMessages(library(did))\n"
        f'd <- read.csv("{csv_r}")\n'
        # att_gt wants numeric id/time/g columns
        f'd[["{idname}"]] <- as.numeric(d[["{idname}"]])\n'
        f'd[["{tname}"]] <- as.numeric(d[["{tname}"]])\n'
        f'd[["{gname}"]] <- as.numeric(d[["{gname}"]])\n'
        f'd[["{yname}"]] <- as.numeric(d[["{yname}"]])\n'
        "att <- att_gt(\n"
        f'  yname="{yname}", tname="{tname}", idname="{idname}", gname="{gname}",\n'
        f"  data=d, control_group={cg}, est_method={em},\n"
        "  bstrap=TRUE, cband=FALSE, allow_unbalanced_panel=TRUE\n"
        ")\n"
        # --- overall (simple) ATT --------------------------------------------
        'sm <- aggte(att, type="simple", na.rm=TRUE)\n'
        'cat("##OVERALL\\n")\n'
        'cat(sprintf("att|%.8f\\n", as.numeric(sm$overall.att)))\n'
        'cat(sprintf("se|%.8f\\n", as.numeric(sm$overall.se)))\n'
        # --- dynamic / event-study -------------------------------------------
        'dy <- aggte(att, type="dynamic", na.rm=TRUE)\n'
        'cat("##DYNAMIC\\n")\n'
        # dy$overall.att is the (weighted) overall post-treatment ATT from the
        # event-study aggregation; report it as the headline dynamic-overall too
        'cat(sprintf("dyn_overall|%.8f|%.8f\\n", as.numeric(dy$overall.att), as.numeric(dy$overall.se)))\n'
        'eg <- dy$egt; aa <- dy$att.egt; ss <- dy$se.egt\n'
        'for (i in seq_along(eg)) cat(sprintf("e|%.6f|%.8f|%.8f\\n", as.numeric(eg[i]), as.numeric(aa[i]), as.numeric(ss[i])))\n'
        # --- per-cohort / group ----------------------------------------------
        'gp <- aggte(att, type="group", na.rm=TRUE)\n'
        'cat("##GROUP\\n")\n'
        'gg <- gp$egt; ga <- gp$att.egt; gs <- gp$se.egt\n'
        'for (i in seq_along(gg)) cat(sprintf("g|%.6f|%.8f|%.8f\\n", as.numeric(gg[i]), as.numeric(ga[i]), as.numeric(gs[i])))\n'
    )
    out = rbridge.run_r(rcode, timeout=300)
    section = None
    overall: dict[str, float] = {}
    erows: list[list] = []
    grows: list[list] = []
    for line in out.splitlines():
        s = line.strip()
        if s == "##OVERALL":
            section = "O"
        elif s == "##DYNAMIC":
            section = "D"
        elif s == "##GROUP":
            section = "G"
        elif section == "O" and "|" in s:
            k, v = s.split("|", 1)
            try:
                overall[k] = float(v)
            except ValueError:
                pass
        elif section == "D" and s.startswith("dyn_overall|"):
            parts = s.split("|")
            try:
                overall["dyn_att"] = float(parts[1])
                overall["dyn_se"] = float(parts[2])
            except (IndexError, ValueError):
                pass
        elif section == "D" and s.startswith("e|"):
            erows.append(s.split("|")[1:4])
        elif section == "G" and s.startswith("g|"):
            grows.append(s.split("|")[1:4])
    if not overall or "att" not in overall:
        raise RuntimeError("did::aggte 未返回总体 ATT（输出解析失败）")
    es = pd.DataFrame(erows, columns=["event_time", "att", "se"])
    cohorts = pd.DataFrame(grows, columns=["cohort", "att", "se"])
    for frame in (es, cohorts):
        for c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return overall, es, cohorts


@register("callaway_santanna")
def _branch_callaway_santanna(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import numpy as np
    import pandas as pd

    from researchforge.executor import rbridge

    # --- column roles: config overrides, else auto-detect ---------------------
    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if not unit or not time:
        summary.append(
            'Callaway-Sant\'Anna 失败：需要面板数据（单位列 + 时间列）。config={"unit":..,"time":..}。'
            "无面板结构可用其他横截面方法。"
        )
        return

    _excl = {unit, time}
    bins_ = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (cont[0] if cont else None)
    if outcome is None:
        summary.append('Callaway-Sant\'Anna 失败：需要连续结果变量。config={"outcome":..}。')
        return

    # --- first-treatment-period (gname): explicit column, else derive from a
    #     0/1 treatment indicator -------------------------------------------------
    gname_col = cfg.get("gname") if cfg.get("gname") in df.columns else None
    derived = False
    if gname_col is not None:
        sub = df[[unit, time, outcome, gname_col]].copy()
        sub[time] = pd.to_numeric(sub[time], errors="coerce")
        sub["_g"] = pd.to_numeric(sub[gname_col], errors="coerce")
        sub = sub.dropna(subset=[unit, time, outcome, "_g"])
    else:
        treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
            fp.treatment_candidates[0] if fp.treatment_candidates else (bins_[0] if bins_ else None))
        if treatment is None:
            summary.append(
                'Callaway-Sant\'Anna 失败：需要 首次处理期列(gname) 或 二值处理指示列(treatment) 之一以确定每个单位的处理时点。'
                'config={"gname":..} 或 {"treatment":..}。'
            )
            return
        sub = df[[unit, time, outcome, treatment]].copy()
        sub[time] = pd.to_numeric(sub[time], errors="coerce")
        tvals = set(pd.unique(sub[treatment].dropna()))
        if not (tvals <= {0, 1}):
            if len(tvals) == 2:
                sub[treatment] = (sub[treatment] == sorted(tvals)[1]).astype(float)
            else:
                summary.append("Callaway-Sant'Anna 失败：处理指示变量必须二值(0/1)。")
                return
        sub = sub.dropna(subset=[unit, time, outcome, treatment])
        sub[treatment] = pd.to_numeric(sub[treatment], errors="coerce")
        # gname_i = min time where treatment==1 for unit i; else 0 (never-treated)
        onset = sub[sub[treatment] == 1].groupby(unit)[time].min()
        if onset.empty:
            summary.append("Callaway-Sant'Anna 失败：没有任何单位被处理。")
            return
        sub["_g"] = sub[unit].map(onset).fillna(0.0)
        derived = True

    # need numeric unit ids for did (idname must be numeric); factorize stably
    sub = sub.dropna(subset=[unit, time, outcome, "_g"]).copy()
    sub["_id"] = pd.factorize(sub[unit])[0].astype(float) + 1.0
    n_units = int(sub["_id"].nunique())
    treated_units = int((sub.groupby("_id")["_g"].first() > 0).sum())
    never_units = n_units - treated_units
    n_cohorts = int(sub.loc[sub["_g"] > 0, "_g"].nunique())

    if treated_units == 0:
        summary.append("Callaway-Sant'Anna 失败：没有处理单位（所有 gname=0）。")
        return
    if n_cohorts < 1 or sub[time].nunique() < 2:
        summary.append("Callaway-Sant'Anna 失败：需要至少一个处理队列且 ≥2 个时间期。")
        return

    control_group = "notyettreated" if str(cfg.get("control_group", "")).lower() in (
        "notyettreated", "not_yet_treated", "notyet") else "nevertreated"
    # never-treated control needs never-treated units; fall back honestly
    if control_group == "nevertreated" and never_units == 0:
        control_group = "notyettreated"
        cg_note = "（数据无『从未处理』单位，已自动改用『尚未处理』对照）"
    else:
        cg_note = ""
    est_method = str(cfg.get("est_method", "dr")).lower()
    if est_method not in ("dr", "ipw", "reg"):
        est_method = "dr"

    have_r = rbridge.r_available() and rbridge.r_package_available("did")
    if not have_r:
        summary.append(
            "Callaway-Sant'Anna (group-time ATT) 需要 R 的 did 包（未检测到）。"
            "安装：install.packages('did')。"
            "无 R 时可用纯 Python 的 staggered_did（Sun-Abraham 交互加权，等价的异质稳健交错 DiD）"
            "或 event_study（事件研究）作为替代。"
        )
        return

    # identifier guard before column names enter R (idname/tname/gname are our
    # safe synthetic names; only yname comes from data)
    if not re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(outcome)):
        summary.append("Callaway-Sant'Anna 失败：结果列名需为标识符式（字母/数字/. _）；请重命名后重试。")
        return

    rin = sub[["_id", time, outcome, "_g"]].rename(
        columns={"_id": "csid", time: "cstime", "_g": "csg"})
    csv = d / "_cs_did_input.csv"
    rin.to_csv(csv, index=False)
    try:
        overall, es, cohorts = _callaway_santanna_via_r(
            csv, yname=str(outcome), tname="cstime", idname="csid", gname="csg",
            control_group=control_group, est_method=est_method,
        )
    except Exception as err:
        summary.append(
            f"Callaway-Sant'Anna 失败：{err}。"
            "可改用纯 Python 的 staggered_did（Sun-Abraham）或 event_study。"
        )
        return
    finally:
        try:
            if csv.exists():
                csv.unlink()
        except OSError:
            pass

    att = float(overall["att"])
    se = float(overall.get("se", float("nan")))
    ci_low = att - 1.96 * se if se == se else float("nan")
    ci_high = att + 1.96 * se if se == se else float("nan")

    # --- event-study CSV + pre-trend evidence --------------------------------
    es = es.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)
    es["ci_low"] = es["att"] - 1.96 * es["se"]
    es["ci_high"] = es["att"] + 1.96 * es["se"]
    es.to_csv(d / "callaway_santanna_event_study.csv", index=False, encoding="utf-8")
    files.append("callaway_santanna_event_study.csv")

    cohorts = cohorts.dropna(subset=["cohort"]).sort_values("cohort").reset_index(drop=True)
    if len(cohorts):
        cohorts.to_csv(d / "callaway_santanna_by_cohort.csv", index=False, encoding="utf-8")
        files.append("callaway_santanna_by_cohort.csv")

    # pre-trend test: any pre-period (e<0) event-study coef significant at 5%?
    leads = es[es["event_time"] < 0].copy()
    pretrend_bad = False
    if len(leads):
        with np.errstate(invalid="ignore", divide="ignore"):
            zlead = leads["att"] / leads["se"]
        pretrend_bad = bool((zlead.abs() > 1.96).any())

    estimates.update({
        "att_overall": round(att, 4),
        "att_overall_se": round(se, 4) if se == se else float("nan"),
        "att_overall_ci_low": round(ci_low, 4) if ci_low == ci_low else float("nan"),
        "att_overall_ci_high": round(ci_high, 4) if ci_high == ci_high else float("nan"),
        "att_dynamic_overall": round(float(overall.get("dyn_att", float("nan"))), 4),
        "n_units": float(n_units),
        "n_treated_units": float(treated_units),
        "n_never_treated": float(never_units),
        "n_cohorts": float(n_cohorts),
        "pretrend_violation": 1.0 if pretrend_bad else 0.0,
    })

    # --- event-study plot (the headline parallel-trends visual) --------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(-0.5, color="red", ls="--", lw=0.8)
        pre = es[es["event_time"] < 0]
        post = es[es["event_time"] >= 0]
        for part, col in ((pre, "#C44E52"), (post, "#55A868")):
            if len(part):
                ax.errorbar(
                    part["event_time"], part["att"],
                    yerr=[part["att"] - part["ci_low"], part["ci_high"] - part["att"]],
                    fmt="o", capsize=3, color=col,
                )
        ax.set_xlabel("event time e (length of exposure; pre e<0 should hover near 0)")
        ax.set_ylabel(f"ATT(e) on {outcome}")
        ax.set_title("Callaway & Sant'Anna (2021) — dynamic ATT (event study)")
        fig.tight_layout()
        fig.savefig(d / "callaway_santanna_event_study.png", dpi=150)
        plt.close(fig)
        files.append("callaway_santanna_event_study.png")
    except Exception:
        pass

    cg_label = "尚未处理(not-yet-treated)" if control_group == "notyettreated" else "从未处理(never-treated)"
    em_label = {"dr": "双重稳健(doubly-robust)", "ipw": "逆概率加权(IPW)", "reg": "结果回归(reg)"}[est_method]
    sig = "显著" if (se == se and se > 0 and abs(att / se) > 1.96) else "不显著"
    pt = "⚠ 检出前置期 ATT 显著(平行趋势存疑)" if pretrend_bad else "前置期 ATT 未见显著(支持平行趋势)"
    deriv = "（gname 由二值处理指示列推导：每单位首个 treatment==1 的时点，无则记 0=从未处理）" if derived else "（使用显式 gname 列）"
    ci_txt = f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low == ci_low else "[NA]"
    summary.append(
        f"{entry.method} 完成（R did）：{n_cohorts} 个处理队列 / {treated_units} 个处理单位 / "
        f"{never_units} 个从未处理单位；对照组={cg_label}{cg_note}，估计法={em_label}。"
        f"总体 ATT = {att:.4f}（SE {se:.4f}，95%CI {ci_txt}，{sig}）；"
        f"事件研究(动态 ATT)见 callaway_santanna_event_study.png，按队列 ATT 见 callaway_santanna_by_cohort.csv。{pt}。 "
        f"{deriv} "
        "⚠ CS 估计专为**交错采纳 + 异质处理效应**设计：先估各队列-时点 ATT(g,t)、再按队列加权聚合，"
        "规避了交错下双向固定效应(TWFE)的**负权重 / 禁止比较(已处理单位被当对照)**偏差(这正是 CS 的价值)；"
        "识别假设：(条件)**平行趋势** + **无预期(no anticipation)**——前置期(e<0)事件研究系数应≈0，是平行趋势的"
        "**证据(非保证)**，已在图与 CSV 中报告；对照组选择(从未处理 vs 尚未处理)会影响估计(已报告所用对照)；"
        "采用**双重稳健** est_method；每个队列需要足够多的处理单位，队列过小会使 ATT(g,t) 与聚合噪声偏大。"
    )
    code += [
        "library(did)  # Callaway & Sant'Anna (2021) group-time ATT",
        'att <- att_gt(yname=Y, tname=T, idname=ID, gname=G, data=d,',
        f'              control_group="{control_group}", est_method="{est_method}", bstrap=TRUE, cband=FALSE)',
        'aggte(att, type="simple")   # overall ATT ($overall.att/$overall.se)',
        'aggte(att, type="dynamic")  # event study ($egt/$att.egt/$se.egt)',
        'aggte(att, type="group")    # per-cohort ATT',
    ]
