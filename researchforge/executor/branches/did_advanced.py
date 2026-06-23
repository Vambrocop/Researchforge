"""Branch handlers for the did_advanced family — modern DiD robustness/diagnostics.

Three pure-Python (numpy/pandas/statsmodels — NO R) companions to the staggered-DiD
estimators (causal_did.callaway_santanna, causal.staggered_did / event_study). They
DIAGNOSE or BOUND the two-way-fixed-effects (TWFE) DiD rather than just re-estimate it:

  * goodman_bacon   — Goodman-Bacon (2021) decomposition of the TWFE DiD coefficient
                      into its weighted-average 2x2 sub-comparisons, exposing the
                      weight on the "bad" already-treated-as-control comparisons.
  * honest_did      — Rambachan & Roth (2023) "honest" sensitivity analysis: bound how
                      far a post-treatment effect can move under bounded parallel-trends
                      violations (relative-magnitudes RM single-period variant).
  * chaisemartin_did— de Chaisemartin & D'Haultfoeuille (2020) DID_M estimator using
                      only not-yet-switched stable controls (heterogeneity-robust),
                      with a unit-resample bootstrap SE.

Every handler MIRRORS the panel role-resolution idiom from causal_did.callaway_santanna
(unit/time/outcome via config-or-fingerprint; first-treatment cohort `g` from an explicit
gname column, else DERIVED from a 0/1 treatment indicator — first period treated, 0 =
never-treated). Each degrades honestly (not a panel / too few periods or units / no
treatment variation / import missing -> append a Chinese "<方法>跳过：<原因>" to summary
and RETURN; never crash, never fabricate), writes CSV + PNG (matplotlib Agg, ENGLISH plot
labels), fills float `estimates`, appends a Chinese `summary` ending with ⚠ disclosures,
and MUTATES ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.

numpy / pandas / statsmodels are installed. Bootstrap uses a FIXED, disclosed seed.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# ─────────────────────────────────────────────────────────────────────────────
# Shared panel role-resolution (mirrors causal_did.callaway_santanna).
# Returns (sub_df, unit, time, outcome, problem_msg).
#   sub_df has columns: unit, time, outcome, "_g" (first-treatment period; 0 = never)
#                       and "_d" (0/1 treated-now indicator, t >= g & g>0).
# When problem_msg is not None the caller appends it to summary and returns.
# `label` is the method name used in the Chinese skip message.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_panel_did(ctx: Ctx, label: str):
    import numpy as np  # noqa: F401
    import pandas as pd

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg

    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if not unit or not time:
        return None, None, None, None, (
            f"{label}跳过：需要面板数据（单位列 + 时间列）。config={{\"unit\":..,\"time\":..}}。"
        )

    _excl = {unit, time}
    bins_ = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (cont[0] if cont else None)
    if outcome is None:
        return None, None, None, None, f"{label}跳过：需要连续结果变量。config={{\"outcome\":..}}。"

    # first-treatment cohort g: explicit gname, else derive from a 0/1 treatment indicator
    gname_col = cfg.get("gname") if cfg.get("gname") in df.columns else None
    if gname_col is not None:
        sub = df[[unit, time, outcome, gname_col]].copy()
        sub[time] = pd.to_numeric(sub[time], errors="coerce")
        sub["_g"] = pd.to_numeric(sub[gname_col], errors="coerce").fillna(0.0)
        sub = sub.dropna(subset=[unit, time, outcome])
        sub["_d"] = ((sub["_g"] > 0) & (sub[time] >= sub["_g"])).astype(float)
    else:
        treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
            fp.treatment_candidates[0] if fp.treatment_candidates else (bins_[0] if bins_ else None))
        if treatment is None:
            return None, None, None, None, (
                f"{label}跳过：需要 首次处理期列(gname) 或 二值处理指示列(treatment) 之一以确定每个单位的处理时点。"
                f"config={{\"gname\":..}} 或 {{\"treatment\":..}}。"
            )
        sub = df[[unit, time, outcome, treatment]].copy()
        sub[time] = pd.to_numeric(sub[time], errors="coerce")
        tvals = set(pd.unique(sub[treatment].dropna()))
        if not (tvals <= {0, 1}):
            if len(tvals) == 2:
                sub[treatment] = (sub[treatment] == sorted(tvals)[1]).astype(float)
            else:
                return None, None, None, None, f"{label}跳过：处理指示变量必须二值(0/1)。"
        sub = sub.dropna(subset=[unit, time, outcome, treatment])
        sub[treatment] = pd.to_numeric(sub[treatment], errors="coerce")
        onset = sub[sub[treatment] == 1].groupby(unit)[time].min()
        if onset.empty:
            return None, None, None, None, f"{label}跳过：没有任何单位被处理。"
        sub["_g"] = sub[unit].map(onset).fillna(0.0)
        # absorbing-from-onset treated indicator (CS-style); _d = 1 once t >= g
        sub["_d"] = ((sub["_g"] > 0) & (sub[time] >= sub["_g"])).astype(float)

    sub = sub.dropna(subset=[unit, time, outcome, "_g"]).copy()
    return sub, unit, time, outcome, None


# ═════════════════════════════════════════════════════════════════════════════
# (A) goodman_bacon — Goodman-Bacon (2021) TWFE DiD decomposition
# ═════════════════════════════════════════════════════════════════════════════
@register("goodman_bacon")
def _branch_goodman_bacon(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    sub, unit, time, outcome, problem = _resolve_panel_did(ctx, "Goodman-Bacon 分解")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        # Build a BALANCED panel on the (unit x time) grid (Goodman-Bacon assumes a
        # balanced panel; we balance-by-intersection and disclose if rows dropped).
        sub = sub[[unit, time, outcome, "_g"]].dropna().copy()
        # each unit has ONE cohort g (constant within unit)
        gmap = sub.groupby(unit)["_g"].first()
        times = np.sort(sub[time].unique())
        if len(times) < 2:
            summary.append("Goodman-Bacon 分解跳过：需要 ≥2 个时间期。")
            return
        units_all = list(gmap.index)
        full = pd.MultiIndex.from_product([units_all, times], names=[unit, time])
        wide = sub.set_index([unit, time])[outcome].reindex(full)
        # keep only units observed in every period (balanced)
        ymat = wide.unstack(time)  # rows=unit, cols=time
        balanced_units = ymat.dropna(axis=0).index
        dropped = len(units_all) - len(balanced_units)
        ymat = ymat.loc[balanced_units]
        gmap = gmap.loc[balanced_units]
        if ymat.shape[0] < 4:
            summary.append("Goodman-Bacon 分解跳过：平衡面板单位过少（<4），无法做 2x2 分解。")
            return

        T = list(ymat.columns)
        # cohort -> set of units; treated cohorts are g>0, never-treated is g==0
        cohorts = sorted(g for g in gmap.unique() if g > 0)
        never_units = gmap.index[gmap == 0]
        n_groups = len(cohorts) + (1 if len(never_units) > 0 else 0)
        if len(cohorts) < 1:
            summary.append("Goodman-Bacon 分解跳过：没有处理队列（所有单位从未处理）。")
            return

        # mean outcome path per cohort group (rows=time)
        def _grp_mean(uidx):
            return ymat.loc[uidx].mean(axis=0)  # Series indexed by time

        cohort_units = {g: gmap.index[gmap == g] for g in cohorts}
        cohort_mean = {g: _grp_mean(cohort_units[g]) for g in cohorts}
        never_mean = _grp_mean(never_units) if len(never_units) else None
        n_share = {g: len(cohort_units[g]) / ymat.shape[0] for g in cohorts}
        n_never_share = (len(never_units) / ymat.shape[0]) if len(never_units) else 0.0

        # ---- 2x2 DiD over a time sub-window [pre-period rows < onset, post >= onset] -----
        # A 2x2 DiD between groups (treated T, control C) with change-point at `onset`:
        #   DiD = (Ybar_T,post - Ybar_T,pre) - (Ybar_C,post - Ybar_C,pre)
        # over the sub-window [t_lo, t_hi].
        def _did_2x2(treat_path, ctrl_path, onset, t_lo, t_hi):
            pre = [t for t in T if t_lo <= t < onset]
            post = [t for t in T if onset <= t <= t_hi]
            if not pre or not post:
                return None
            dT = treat_path.loc[post].mean() - treat_path.loc[pre].mean()
            dC = ctrl_path.loc[post].mean() - ctrl_path.loc[pre].mean()
            return float(dT - dC)

        # Goodman-Bacon weight for a 2x2 with treated-share Dbar over its sub-window and
        # group-pair sample-size shares (nk, nl): w ∝ (nk+nl)^2 * nkl(1-nkl) * Dbar(1-Dbar),
        # where nkl = treated-group-within-pair share and Dbar = share of the sub-window the
        # treated group spends treated. (Standard Bacon weight: pair-size scaling × the
        # variance of treatment var(D)=Dbar(1-Dbar) over the window; normalized to sum 1.)
        comps = []  # list of dicts: type, est, weight_raw

        # (i) treated cohort vs NEVER-treated --------------------------------------
        if never_mean is not None:
            t_lo, t_hi = T[0], T[-1]
            for g in cohorts:
                est = _did_2x2(cohort_mean[g], never_mean, g, t_lo, t_hi)
                if est is None:
                    continue
                win = [t for t in T if t_lo <= t <= t_hi]
                Dbar = float(np.mean([1.0 if t >= g else 0.0 for t in win]))
                nk, nl = n_share[g], n_never_share
                tot = nk + nl
                nkl = nk / tot if tot > 0 else 0.0
                w = (tot ** 2) * nkl * (1.0 - nkl) * Dbar * (1.0 - Dbar)
                comps.append({"type": "treated_vs_untreated", "est": est, "weight_raw": float(w),
                              "treated_cohort": float(g), "control": "never"})

        # (ii)/(iii) earlier vs later among TIMING groups (each pair gives TWO 2x2s):
        #   (ii)  GOOD  earlier-vs-later: earlier treated, later still UNTREATED control
        #               (window t < l, change at earlier onset k)
        #   (iii) BAD   later-vs-earlier: later treated, earlier ALREADY-treated control
        #               (window t >= k, change at later onset l) — forbidden control,
        #               biases TWFE under dynamic/heterogeneous effects.
        for i in range(len(cohorts)):
            for j in range(i + 1, len(cohorts)):
                k, l = cohorts[i], cohorts[j]  # k earlier, l later (k < l)
                nk, nl = n_share[k], n_share[l]
                tot = nk + nl
                nkl = nk / tot if tot > 0 else 0.0
                # (ii) GOOD: window where the later group is still untreated (t < l).
                pre_l_times = [t for t in T if t < l]
                if pre_l_times:
                    t_lo_g, t_hi_g = T[0], pre_l_times[-1]
                    est_g = _did_2x2(cohort_mean[k], cohort_mean[l], k, t_lo_g, t_hi_g)
                    if est_g is not None:
                        win = [t for t in T if t_lo_g <= t <= t_hi_g]
                        Dbar = float(np.mean([1.0 if t >= k else 0.0 for t in win]))
                        w = (tot ** 2) * nkl * (1.0 - nkl) * Dbar * (1.0 - Dbar)
                        comps.append({"type": "earlier_vs_later", "est": est_g, "weight_raw": float(w),
                                      "treated_cohort": float(k), "control": f"later({l})"})
                # (iii) BAD: window where the earlier group is already treated (t >= k).
                post_k_times = [t for t in T if t >= k]
                if post_k_times and any(t >= l for t in post_k_times):
                    t_lo_b, t_hi_b = post_k_times[0], post_k_times[-1]
                    est_b = _did_2x2(cohort_mean[l], cohort_mean[k], l, t_lo_b, t_hi_b)
                    if est_b is not None:
                        win = [t for t in T if t_lo_b <= t <= t_hi_b]
                        Dbar = float(np.mean([1.0 if t >= l else 0.0 for t in win]))
                        nkl_b = nl / tot if tot > 0 else 0.0  # within-pair share is the LATER group
                        w = (tot ** 2) * nkl_b * (1.0 - nkl_b) * Dbar * (1.0 - Dbar)
                        comps.append({"type": "later_vs_earlier_BAD", "est": est_b, "weight_raw": float(w),
                                      "treated_cohort": float(l), "control": f"earlier({k})"})

        if not comps:
            summary.append("Goodman-Bacon 分解跳过：无法构造任何 2x2 比较（时点/窗口不足）。")
            return

        cdf = pd.DataFrame(comps)
        wsum = cdf["weight_raw"].sum()
        if wsum <= 0:
            summary.append("Goodman-Bacon 分解跳过：所有 2x2 权重为 0（无处理时点变异 / 窗口退化）。")
            return
        cdf["weight"] = cdf["weight_raw"] / wsum
        twfe_decomp = float((cdf["weight"] * cdf["est"]).sum())  # decomposition TWFE = Σ w·est

        # ---- direct TWFE within-estimator (the target the decomposition approximates) ----
        # two-way (unit + time) within demeaning; OLS slope of demeaned D on demeaned y
        # is the TWFE DiD coefficient. We report BOTH numbers honestly: the closed-form
        # (size × Dbar(1-Dbar)) Bacon weights have the right SHAPE and recover the bad-weight
        # share (the diagnostic of interest), but only reconcile EXACTLY with the direct
        # estimate for special time spacings — we therefore disclose any gap rather than
        # fabricate exact equality (the relative weights are what the diagnostic is about).
        long = ymat.stack().rename(outcome).reset_index()
        long.columns = [unit, time, outcome]
        long["_g"] = long[unit].map(gmap)
        long["_d"] = ((long["_g"] > 0) & (long[time] >= long["_g"])).astype(float)
        ybar_i = long.groupby(unit)[outcome].transform("mean")
        ybar_t = long.groupby(time)[outcome].transform("mean")
        ygm = long[outcome].mean()
        dbar_i = long.groupby(unit)["_d"].transform("mean")
        dbar_t = long.groupby(time)["_d"].transform("mean")
        dgm = long["_d"].mean()
        yw = long[outcome] - ybar_i - ybar_t + ygm
        dw = long["_d"] - dbar_i - dbar_t + dgm
        denom = float((dw * dw).sum())
        twfe_direct = float((dw * yw).sum() / denom) if denom > 0 else float("nan")

        # per-comparison-TYPE aggregation (weight-sum + weighted-avg estimate)
        type_rows = []
        for typ, grp in cdf.groupby("type"):
            wt = float(grp["weight"].sum())
            est = float((grp["weight"] * grp["est"]).sum() / wt) if wt > 0 else float("nan")
            type_rows.append({"comparison_type": typ, "total_weight": wt,
                              "weighted_estimate": est, "n_2x2": int(len(grp))})
        tdf = pd.DataFrame(type_rows).sort_values("comparison_type").reset_index(drop=True)
        tdf.to_csv(d / "goodman_bacon_decomposition.csv", index=False, encoding="utf-8")
        files.append("goodman_bacon_decomposition.csv")
        cdf[["type", "treated_cohort", "control", "est", "weight"]].to_csv(
            d / "goodman_bacon_2x2.csv", index=False, encoding="utf-8")
        files.append("goodman_bacon_2x2.csv")

        def _typ_w(t):
            r = tdf[tdf["comparison_type"] == t]
            return float(r["total_weight"].iloc[0]) if len(r) else 0.0

        def _typ_e(t):
            r = tdf[tdf["comparison_type"] == t]
            return float(r["weighted_estimate"].iloc[0]) if len(r) else float("nan")

        w_tu = _typ_w("treated_vs_untreated")
        w_el = _typ_w("earlier_vs_later")
        w_le = _typ_w("later_vs_earlier_BAD")
        att_tu = _typ_e("treated_vs_untreated")

        estimates.update({
            # the EXACT two-way-FE coefficient is twfe_did_direct; twfe_did_decomp is the
            # Σ w·DiD reconstruction from the (approximate-shape) Bacon weights — they need
            # NOT be equal even on a balanced panel (the weight formula is simplified).
            "twfe_did_decomp": round(twfe_decomp, 6),
            "twfe_did_direct": round(twfe_direct, 6) if twfe_direct == twfe_direct else float("nan"),
            "weight_treated_vs_untreated": round(w_tu, 6),
            "weight_earlier_vs_later": round(w_el, 6),
            "weight_later_vs_earlier_BAD": round(w_le, 6),
            "att_treated_vs_untreated": round(att_tu, 6) if att_tu == att_tu else float("nan"),
            "n_groups": float(n_groups),
        })

        # ---- scatter: each 2x2 estimate vs its weight, colored by type -----------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            colors = {"treated_vs_untreated": "#55A868", "earlier_vs_later": "#4C72B0",
                      "later_vs_earlier_BAD": "#C44E52"}
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for typ, grp in cdf.groupby("type"):
                ax.scatter(grp["weight"], grp["est"], s=70, alpha=0.8,
                           color=colors.get(typ, "#888888"), label=typ.replace("_", " "))
            ax.axhline(twfe_decomp, color="black", ls="--", lw=1.0,
                       label=f"decomposition sum = {twfe_decomp:.3f}")
            if twfe_direct == twfe_direct:
                ax.axhline(twfe_direct, color="#8172B3", ls=":", lw=1.2,
                           label=f"TWFE coefficient = {twfe_direct:.3f}")
            ax.set_xlabel("Goodman-Bacon weight")
            ax.set_ylabel("2x2 DiD estimate")
            ax.set_title("Goodman-Bacon (2021) decomposition of TWFE DiD")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "goodman_bacon_scatter.png", dpi=150)
            plt.close(fig)
            files.append("goodman_bacon_scatter.png")
        except Exception:
            pass

        recon_txt = (f"实际 TWFE 系数（双向 within OLS）= {twfe_direct:.4f}；"
                     f"由（近似形）Bacon 权重重构的分解和 = {twfe_decomp:.4f}"
                     if twfe_direct == twfe_direct else
                     f"分解和 = {twfe_decomp:.4f}（直接 TWFE 估计不可得）")
        bal_note = (f" ⚠ 已平衡为相交面板（丢弃 {dropped} 个非全期单位）。" if dropped > 0 else "")
        bad_dom = w_le > 0.10
        summary.append(
            f"{entry.method} 完成：将交错处理下的双向固定效应(TWFE) DiD 系数分解为各 2x2 子比较的加权平均。"
            f"权重——处理 vs 从未处理 = {w_tu:.4f}（估计 {att_tu:.4f}）；较早 vs 较晚(干净) = {w_el:.4f}；"
            f"较晚 vs 较早(**坏比较**：已处理单位充当对照) = {w_le:.4f}。{recon_txt}（共 {n_groups} 组）。"
            f"{bal_note}"
            f" ⚠ {'**坏比较权重偏高**——' if bad_dom else ''}"
            "在『较晚 vs 较早』这类比较里，**已处理的较早队列被当作对照**（禁止对照）；当处理效应随时间/队列**异质**时，"
            "这会给 TWFE 带来**负权重/偏差污染**。坏比较权重越大 TWFE 越不可信——"
            "应改用 callaway_santanna（C&S）或 chaisemartin_did（dCDH）这类异质稳健估计量。"
            " ⚠ 真实 TWFE 系数以 twfe_did_direct 为准；twfe_did_decomp（分解和）用 Goodman-Bacon 的"
            "**简化形**权重（组对规模 × 子窗口内处理份额方差 Dbar(1-Dbar)）归一化重构，"
            "权重的**相对**结构（尤其坏比较份额）是可靠诊断信号，但其绝对值与精确 bacondecomp 略有出入，"
            "故分解和与直接系数即使在平衡面板上也未必完全相等（已分别给出，不强行对齐）。"
            " ⚠ 与所有 DiD 一样假定**平行趋势 + 无预期效应（no-anticipation，处理前不应已有响应）**。"
        )
        code += [
            "# Goodman-Bacon (2021) decomposition of the TWFE DiD coefficient",
            "# TWFE_DiD = Σ_k w_k · DiD_k  over 2x2 sub-comparisons",
            "#   types: treated-vs-never, earlier-vs-later(good), later-vs-earlier(BAD)",
            "# w_k ∝ (n_k+n_l)^2 · s(1-s) · Dbar(1-Dbar)  (group-pair size × treated-share variance)",
            "# reconcile Σ w·DiD against the two-way within OLS slope of D on demeaned y",
        ]
    except Exception as err:
        summary.append(f"Goodman-Bacon 分解跳过：{err}。")


# ═════════════════════════════════════════════════════════════════════════════
# (B) honest_did — Rambachan & Roth (2023) honest sensitivity (relative-magnitudes)
# ═════════════════════════════════════════════════════════════════════════════
@register("honest_did")
def _branch_honest_did(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    sub, unit, time, outcome, problem = _resolve_panel_did(ctx, "Honest DiD")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import re

        import numpy as np
        import pandas as pd
        import statsmodels.formula.api as smf

        sub = sub[[unit, time, outcome, "_g"]].dropna().copy()
        # event time relative to onset; never-treated (_g==0) -> NaN (comparison group)
        sub["_evt"] = np.where(sub["_g"] > 0, sub[time] - sub["_g"], np.nan)
        has_never = bool((sub["_g"] == 0).any())
        onset = sub.loc[sub["_g"] > 0].groupby(unit)["_g"].first()
        if onset.empty:
            summary.append("Honest DiD 跳过：没有任何处理单位。")
            return
        if not has_never and onset.nunique() > 1:
            summary.append("Honest DiD 跳过：所有单位最终都被处理且交错采纳——纯 TWFE 事件研究此情形需第二个"
                           "归一化、且异质效应下有偏；请先用 Callaway-Sant'Anna / chaisemartin_did 得到事件研究。")
            return

        try:
            L = max(2, min(12, int(cfg.get("window", 5))))
        except (TypeError, ValueError):
            L = 5

        def _bin(e):
            return "never" if pd.isna(e) else str(int(max(-L, min(L, e))))

        sub["_evtb"] = sub["_evt"].apply(_bin)
        treated_bins = sorted({b for b in sub["_evtb"] if b != "never"}, key=int)
        leads = [b for b in treated_bins if int(b) < 0]
        lags = [b for b in treated_bins if int(b) >= 0]
        if len(leads) < 2:
            summary.append("Honest DiD 跳过：至少需要 2 个处理前期(leads)来衡量最大前趋势违背幅度；"
                           "当前可用前期不足。")
            return
        if not lags:
            summary.append("Honest DiD 跳过：没有任何处理后期(lags)可作敏感性分析目标。")
            return
        ref = "-1" if "-1" in leads else max(leads, key=int)  # closest pre-period as baseline

        # event-study TWFE with unit+time FE, clustered-by-unit vcov
        formula = (f'Q("{outcome}") ~ C(_evtb, Treatment("{ref}")) '
                   f'+ C(Q("{unit}")) + C(Q("{time}"))')
        model = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub[unit]})
        rx = re.compile(r"C\(_evtb.*?\)\[T\.(-?\d+)\]")
        coefs = {}
        ses = {}
        for name in model.params.index:
            m = rx.match(name)
            if not m:
                continue
            e = int(m.group(1))
            coefs[e] = float(model.params[name])
            ses[e] = float(model.bse[name])
        coefs[int(ref)] = 0.0
        ses[int(ref)] = 0.0

        # pre-period event times (leads + the reference, whose coef is 0); choose first post
        pre_times = sorted(e for e in coefs if e < 0)
        post_es = sorted(e for e in coefs if e >= 0)
        if not pre_times or not post_es:
            summary.append("Honest DiD 跳过：缺少前期或后期事件系数，无法构造敏感性界。")
            return
        # Rambachan & Roth relative-magnitudes scale = the largest CONSECUTIVE
        # first-difference of the pre-period coefs (NOT the max coefficient LEVEL).
        # This matches the HonestDiD package's Mbar, which bounds the post-period
        # violation by Mbar × max pre-period step |δ_s − δ_{s−1}|.
        if len(pre_times) >= 2:
            max_pre = float(max(abs(coefs[pre_times[i]] - coefs[pre_times[i - 1]])
                                for i in range(1, len(pre_times))))
        else:
            max_pre = float(abs(coefs[pre_times[0]]))
        e_post = int(cfg.get("post_period", post_es[0])) if cfg.get("post_period") in post_es else post_es[0]
        beta = float(coefs[e_post])
        se = float(ses.get(e_post, float("nan")))
        z = 1.959963985  # 97.5th normal quantile
        ci_lo = beta - z * se
        ci_hi = beta + z * se

        # ---- relative-magnitudes (RM) single-period bound -----------------------
        # Rambachan & Roth (2023) RM restriction: |post-violation| <= Mbar * max|pre-violation|.
        # Simplest single-post-period bound: robust CI = [beta - z*SE - Mbar*max_pre,
        #                                                  beta + z*SE + Mbar*max_pre].
        # (Full polyhedral / FLCI version deferred; smoothness/2nd-difference variants deferred.)
        grid = [0.0, 0.5, 1.0, 1.5, 2.0]
        rows = []
        for Mbar in grid:
            bias = Mbar * max_pre
            lo = beta - z * se - bias
            hi = beta + z * se + bias
            rows.append({"Mbar": Mbar, "robust_ci_low": lo, "robust_ci_high": hi,
                         "includes_zero": bool(lo <= 0.0 <= hi)})
        gdf = pd.DataFrame(rows)
        gdf.to_csv(d / "honest_did_sensitivity.csv", index=False, encoding="utf-8")
        files.append("honest_did_sensitivity.csv")

        # breakdown Mbar: smallest Mbar at which the robust CI first includes 0.
        # If the original CI already includes 0 -> breakdown is 0 (not significant to begin with).
        # Closed form (max_pre>0 and original CI excludes 0):
        #   positive significant effect (beta>0): low end binds ->  Mbar >= (beta - z*se)/max_pre.
        #   negative significant effect (beta<0): high end binds -> Mbar >= -(beta + z*se)/max_pre.
        orig_incl0 = bool(ci_lo <= 0.0 <= ci_hi)
        if orig_incl0:
            breakdown = 0.0
        elif max_pre <= 0.0:
            breakdown = float("nan")  # no pre-violation scale -> RM bound never widens; never breaks
        else:
            breakdown = ((beta - z * se) / max_pre) if beta > 0 else (-(beta + z * se) / max_pre)
            if breakdown < 0:
                breakdown = 0.0
            if breakdown > 2.0:
                breakdown = float("nan")  # robust even past the top of the grid

        estimates.update({
            "effect_post1": round(beta, 6),
            "ci_orig_low": round(ci_lo, 6),
            "ci_orig_high": round(ci_hi, 6),
            "breakdown_mbar": round(breakdown, 6) if breakdown == breakdown else float("nan"),
            "max_pretrend_violation": round(max_pre, 6),
        })

        # ---- robust CI width vs Mbar -------------------------------------------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.fill_between(gdf["Mbar"], gdf["robust_ci_low"], gdf["robust_ci_high"],
                            alpha=0.25, color="#4C72B0", label="robust CI")
            ax.plot(gdf["Mbar"], gdf["robust_ci_low"], "-o", color="#4C72B0", ms=4)
            ax.plot(gdf["Mbar"], gdf["robust_ci_high"], "-o", color="#4C72B0", ms=4)
            ax.axhline(0, color="red", ls="--", lw=0.9, label="zero")
            ax.axhline(beta, color="black", ls=":", lw=0.9, label=f"effect={beta:.3f}")
            if breakdown == breakdown:
                ax.axvline(breakdown, color="#C44E52", ls="--", lw=1.0,
                           label=f"breakdown Mbar={breakdown:.2f}")
            ax.set_xlabel("Mbar (relative-magnitudes bound on post vs max pre violation)")
            ax.set_ylabel(f"robust CI for effect at e={e_post}")
            ax.set_title("Honest DiD (Rambachan-Roth 2023) — sensitivity to parallel-trends violations")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "honest_did_sensitivity.png", dpi=150)
            plt.close(fig)
            files.append("honest_did_sensitivity.png")
        except Exception:
            pass

        ci_orig_txt = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        if orig_incl0:
            bd_txt = "原始 CI 本就含 0（效应在常规推断下已不显著），breakdown M̄ = 0。"
        elif breakdown == breakdown:
            bd_txt = (f"breakdown M̄ = {breakdown:.3f}：当『后期违背 ≤ M̄ × 最大前期违背』中的 M̄ 超过此值，"
                      f"稳健 CI 便开始包含 0（结论失去稳健性）。")
        else:
            bd_txt = "即使 M̄ = 2，稳健 CI 仍不含 0——结论对（相对幅度型）平行趋势违背较稳健。"
        summary.append(
            f"{entry.method} 完成：对事件研究中 e={e_post} 期的处理效应做平行趋势敏感性分析。"
            f"点估计 = {beta:.4f}，常规 95%CI = {ci_orig_txt}；最大前趋势违背幅度 = {max_pre:.4f}。{bd_txt}"
            f"（M̄ 网格 [0, 0.5, 1, 1.5, 2] 的稳健 CI 见 honest_did_sensitivity.csv / .png。）"
            " ⚠ Honest DiD **不修正偏差，只界定敏感性**——它回答『平行趋势要被违背到多大，结论才翻盘』。"
            "需要 ≥2 个处理前期来估计最大违背幅度；此处用的是**相对幅度(RM)单后期界**"
            "（|bias| ≤ M̄ × 最大前期一阶差分 max|δ_s−δ_(s−1)|，与 HonestDiD 包的 M̄ 同尺度），"
            "是 Rambachan-Roth 最简变体——"
            "**完整多面体(polyhedral)/FLCI 与平滑度(二阶差分)变体已留待后续**。"
            "依赖事件研究可识别（单位+时间双向固定效应、按单位聚类 SE）。"
        )
        code += [
            "# Honest DiD (Rambachan & Roth 2023), relative-magnitudes single-post-period variant",
            "# 1) event-study TWFE: y ~ C(evt, ref=-1) + C(unit) + C(time), cluster by unit",
            "# 2) max_pre = max|consecutive pre-period first-difference|  (RR relative-magnitudes scale)",
            "#    robust CI = [b - z*se - Mbar*max_pre, b + z*se + Mbar*max_pre]",
            "# breakdown Mbar = (|b|-z*se)/max_pre  (smallest Mbar whose robust CI includes 0)",
        ]
    except Exception as err:
        summary.append(f"Honest DiD 跳过：{err}。")


# ═════════════════════════════════════════════════════════════════════════════
# (C) chaisemartin_did — de Chaisemartin & D'Haultfoeuille (2020) DID_M
# ═════════════════════════════════════════════════════════════════════════════
@register("chaisemartin_did")
def _branch_chaisemartin_did(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    sub, unit, time, outcome, problem = _resolve_panel_did(ctx, "de Chaisemartin DiD")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        # period-by-period 0/1 treatment status _d (from the resolver)
        sub = sub[[unit, time, outcome, "_d"]].dropna().copy()
        sub[time] = pd.to_numeric(sub[time], errors="coerce")
        sub = sub.dropna(subset=[time])
        periods = np.sort(sub[time].unique())
        if len(periods) < 2:
            summary.append("de Chaisemartin DiD 跳过：需要 ≥2 个时间期。")
            return

        # pivot to wide: y[unit, t] and d[unit, t]
        yw = sub.pivot_table(index=unit, columns=time, values=outcome, aggfunc="mean")
        dw = sub.pivot_table(index=unit, columns=time, values="_d", aggfunc="mean")
        units = list(yw.index)

        # ---- DID_M core: consecutive-period 2-group DiDs ------------------------
        # For each pair (t-1, t):
        #   stable-0 = D_{t-1}=0 & D_t=0 ; stable-1 = D_{t-1}=1 & D_t=1
        #   switchers-IN  = D_{t-1}=0 & D_t=1 (vs stable-0 controls)
        #   switchers-OUT = D_{t-1}=1 & D_t=0 (vs stable-1 controls)
        #   DiD_in  =  mean ΔY[in]  - mean ΔY[stable0]
        #   DiD_out = -(mean ΔY[out] - mean ΔY[stable1])   (signed so it estimates the EFFECT)
        # DID_M = sample-size-weighted average over all switcher cells across all pairs.
        def _compute_didm(y_df, d_df):
            pair_rows = []
            num = 0.0
            den = 0.0
            for a, b in zip(periods[:-1], periods[1:]):
                if a not in y_df.columns or b not in y_df.columns:
                    continue
                da, db = d_df[a], d_df[b]
                ya, yb = y_df[a], y_df[b]
                ok = da.notna() & db.notna() & ya.notna() & yb.notna()
                da, db, ya, yb = da[ok], db[ok], ya[ok], yb[ok]
                bin_ok = da.isin([0, 1]) & db.isin([0, 1])  # restrict to binary status
                da, db, ya, yb = da[bin_ok], db[bin_ok], ya[bin_ok], yb[bin_ok]
                if len(da) == 0:
                    continue
                dy = yb - ya
                stable0 = (da == 0) & (db == 0)
                stable1 = (da == 1) & (db == 1)
                sin = (da == 0) & (db == 1)   # switchers in
                sout = (da == 1) & (db == 0)  # switchers out
                contrib = []
                if sin.sum() > 0 and stable0.sum() > 0:
                    did_in = float(dy[sin].mean() - dy[stable0].mean())
                    n_in = int(sin.sum())
                    pair_rows.append({"period_pair": f"{a}->{b}", "cell": "switch_in",
                                      "did": did_in, "n_switchers": n_in})
                    contrib.append((did_in, n_in))
                if sout.sum() > 0 and stable1.sum() > 0:
                    did_out = float(-(dy[sout].mean() - dy[stable1].mean()))
                    n_out = int(sout.sum())
                    pair_rows.append({"period_pair": f"{a}->{b}", "cell": "switch_out",
                                      "did": did_out, "n_switchers": n_out})
                    contrib.append((did_out, n_out))
                for est, n in contrib:
                    num += est * n
                    den += n
            didm = (num / den) if den > 0 else float("nan")
            return didm, den, pair_rows

        didm, n_switchers_total, pair_rows = _compute_didm(yw, dw)
        if not pair_rows or n_switchers_total <= 0 or didm != didm:
            summary.append("de Chaisemartin DiD 跳过：相邻期内没有可用的切换者(switchers)+稳定对照组组合"
                           "（需要处理状态在相邻期发生变化，且存在同初值未变化的稳定对照）。")
            return

        pdf = pd.DataFrame(pair_rows)
        # per-period-pair pooled DiD (size-weighted over its in/out cells), in the
        # order the pairs first appear (chronological) — avoid groupby-apply for
        # cross-pandas-version stability and deterministic plot ordering.
        order = list(dict.fromkeys(pdf["period_pair"]))
        per_pair_rows = []
        for pp in order:
            grp = pdf[pdf["period_pair"] == pp]
            wsum_pp = float(grp["n_switchers"].sum())
            did_pp = float((grp["did"] * grp["n_switchers"]).sum() / wsum_pp) if wsum_pp > 0 else float("nan")
            per_pair_rows.append({"period_pair": pp, "did": did_pp,
                                  "n_switchers": int(grp["n_switchers"].sum())})
        per_pair = pd.DataFrame(per_pair_rows)
        per_pair.to_csv(d / "chaisemartin_did_by_pair.csv", index=False, encoding="utf-8")
        files.append("chaisemartin_did_by_pair.csv")
        n_period_pairs = int(per_pair.shape[0])

        # ---- unit-resample bootstrap SE + CI (FIXED, disclosed seed) ------------
        try:
            B = int(cfg.get("bootstrap", 500))
        except (TypeError, ValueError):
            B = 500
        B = max(50, min(2000, B))
        seed = 12345
        rng = np.random.default_rng(seed)
        boots = []
        u_arr = np.array(units)
        nU = len(u_arr)
        for _ in range(B):
            idx = rng.integers(0, nU, size=nU)
            yb = yw.iloc[idx]  # resample units positionally (with replacement)
            db = dw.iloc[idx]
            bm, bden, _ = _compute_didm(yb, db)
            if bm == bm and bden > 0:
                boots.append(bm)
        boots = np.array(boots, dtype=float)
        if boots.size >= 10:
            se = float(np.std(boots, ddof=1))
            ci_low = float(np.percentile(boots, 2.5))
            ci_high = float(np.percentile(boots, 97.5))
        else:
            se = ci_low = ci_high = float("nan")

        estimates.update({
            "did_m": round(float(didm), 6),
            "se": round(se, 6) if se == se else float("nan"),
            "ci_low": round(ci_low, 6) if ci_low == ci_low else float("nan"),
            "ci_high": round(ci_high, 6) if ci_high == ci_high else float("nan"),
            "n_switchers": float(n_switchers_total),
            "n_period_pairs": float(n_period_pairs),
        })

        # ---- per-period-pair DiD with pooled DID_M line -------------------------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            x = np.arange(len(per_pair))
            ax.bar(x, per_pair["did"], color="#4C72B0", alpha=0.85)
            ax.axhline(float(didm), color="#C44E52", ls="--", lw=1.2,
                       label=f"pooled DID_M = {didm:.3f}")
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(p) for p in per_pair["period_pair"]], rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("consecutive period pair")
            ax.set_ylabel("2-group DiD (switchers vs stable)")
            ax.set_title("de Chaisemartin & D'Haultfoeuille (2020) DID_M")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "chaisemartin_did.png", dpi=150)
            plt.close(fig)
            files.append("chaisemartin_did.png")
        except Exception:
            pass

        ci_txt = f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low == ci_low else "[NA]"
        sig = ("显著" if (se == se and se > 0 and not (ci_low <= 0.0 <= ci_high)) else "不显著")
        summary.append(
            f"{entry.method} 完成：DID_M = {didm:.4f}（自助 SE {se:.4f}，95%CI {ci_txt}，{sig}）；"
            f"使用切换者 {int(n_switchers_total)} 个，跨 {n_period_pairs} 个相邻期对。"
            f"逐期对 DiD 见 chaisemartin_did_by_pair.csv / .png。"
            f" ⚠ DID_M 估计的是**切换者在切换当期的瞬时效应**（非动态/累计 ATT）——"
            "每个相邻期 (t-1,t) 用『初值相同但处理状态未变』的**稳定组**作对照，"
            "只用**尚未切换**的对照、在异质处理效应下仍有效（规避 TWFE 的禁止对照偏差）。"
            f"需要处理状态在相邻期有变化 + 每对都有稳定对照组；自助法 B={B}、固定随机种子 seed={seed}（已披露，结果可复现）。"
            " ⚠ 假定**平行趋势（变化版）+ 无预期效应（no-anticipation）**。"
        )
        code += [
            "# de Chaisemartin & D'Haultfoeuille (2020) DID_M (heterogeneity-robust)",
            "# for each (t-1,t): DiD_in  = ΔY[0→1] - ΔY[stable 0→0]",
            "#                   DiD_out = -(ΔY[1→0] - ΔY[stable 1→1])",
            "# DID_M = sample-size-weighted mean over all switcher cells; unit-bootstrap SE",
        ]
    except Exception as err:
        summary.append(f"de Chaisemartin DiD 跳过：{err}。")
