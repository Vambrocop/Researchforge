"""Branch handlers for the causal SENSITIVITY-ANALYSIS family.

How robust is a causal claim to unmeasured confounding / assumption violations?
Three hand-rolled methods (pure Python: statsmodels/scipy/numpy, no R):

  oster_delta       Oster (2019, JBES) coefficient-stability bound for
                    omitted-variable bias (δ + bias-adjusted β*).
  evalue            VanderWeele & Ding (2017) E-value — minimum confounder
                    strength (RR scale) needed to explain away an effect.
  rosenbaum_bounds  Rosenbaum sensitivity bounds — hidden-bias odds ratio Γ
                    at which a matched/observational effect loses significance.

Convention choices (disclosed in summaries, logged in docs/loop-decisions.md):
  - oster_delta R_max default = min(1.3 * R̃, 1.0) (Oster's heuristic).
  - evalue RR conversion: logistic OR -> RR via sqrt(OR) (VanderWeele 2017
    common-outcome approximation); continuous d -> RR via exp(0.91 * d) (Chinn).

See executor/_branch_api.py and CLAUDE.md「引擎约定」.
"""

from __future__ import annotations

import math

from researchforge.executor._branch_api import Ctx, register


# ---------------------------------------------------------------------------
# shared helpers (local to this family — confined to sensitivity analysis)
# ---------------------------------------------------------------------------
def _evalue_from_rr(rr: float) -> float:
    """E-value for a point/limit risk ratio (VanderWeele & Ding 2017).

    For RR>=1:  E = RR + sqrt(RR * (RR - 1)).
    For RR<1 :  invert to RR' = 1/RR, then apply the same formula (the E-value
                is symmetric on the protective side).  RR exactly at the null (1)
                gives E = 1 (no confounding needed).
    """
    if not (rr == rr) or rr <= 0:  # NaN / non-positive guard
        return float("nan")
    r = rr if rr >= 1.0 else 1.0 / rr
    return r + math.sqrt(r * (r - 1.0))


def _evalue_for_ci(estimate_rr: float, lo_rr: float, hi_rr: float) -> float:
    """E-value for the confidence limit NEAREST the null (VanderWeele & Ding).

    estimate>1 -> use the lower limit; if the CI already crosses the null
    (lo<=1<=hi) the E-value is 1 (the data are compatible with no effect).
    estimate<1 -> use the upper limit (the limit closest to 1 from below).
    """
    if estimate_rr >= 1.0:
        if lo_rr <= 1.0:
            return 1.0
        return _evalue_from_rr(lo_rr)
    # protective estimate
    if hi_rr >= 1.0:
        return 1.0
    return _evalue_from_rr(hi_rr)


def _rosenbaum_signed_rank_pvals(diffs, gamma: float):
    """Rosenbaum (2002) sensitivity bounds for the Wilcoxon SIGNED-RANK statistic.

    For n matched pairs with treated-minus-control differences `diffs`, the
    signed-rank statistic W+ = sum of ranks (of |diff|) over pairs with diff>0.
    Under a hidden bias of magnitude Γ (odds of differential assignment within a
    pair bounded by 1/Γ .. Γ), each positive-rank indicator is Bernoulli with
    success probability bounded between p- = 1/(1+Γ) and p+ = Γ/(1+Γ).  The
    bounding null mean/variance over ranks q_k give a Normal approximation:

        E± = sum_k q_k * p±,   Var± = sum_k q_k^2 * p± * (1 - p±)
        Z±  = (W+ - E±) / sqrt(Var±)

    The UPPER bound on the one-sided p-value (least favourable to significance)
    uses the LARGER expectation E+ (p_hi = Γ/(1+Γ)); the LOWER bound uses E-
    (p_lo = 1/(1+Γ)).  We test for a POSITIVE effect (right tail of W+).

    Returns (p_plus, p_minus, w_plus, n_pairs).
    """
    import numpy as np
    from scipy import stats as _st

    d = np.asarray([x for x in diffs if x == x], dtype=float)
    d = d[d != 0.0]  # zero differences carry no sign information (drop, per Wilcoxon)
    n = d.size
    if n < 1:
        return float("nan"), float("nan"), float("nan"), 0
    ranks = _st.rankdata(np.abs(d))  # average ranks for ties
    w_plus = float(ranks[d > 0].sum())
    p_lo = 1.0 / (1.0 + gamma)   # p-  (lower bound on assignment prob)
    p_hi = gamma / (1.0 + gamma)  # p+  (upper bound)
    sum_q = float(ranks.sum())
    sum_q2 = float((ranks ** 2).sum())
    # UPPER bound on p-value: largest expected W+ (p_hi) => smallest Z => largest p.
    e_hi = sum_q * p_hi
    var_hi = sum_q2 * p_hi * (1.0 - p_hi)
    # LOWER bound on p-value: smallest expected W+ (p_lo) => largest Z => smallest p.
    e_lo = sum_q * p_lo
    var_lo = sum_q2 * p_lo * (1.0 - p_lo)
    z_for_p_upper = (w_plus - e_hi) / math.sqrt(var_hi) if var_hi > 0 else float("nan")
    z_for_p_lower = (w_plus - e_lo) / math.sqrt(var_lo) if var_lo > 0 else float("nan")
    p_plus = float(_st.norm.sf(z_for_p_upper))   # largest plausible p-value
    p_minus = float(_st.norm.sf(z_for_p_lower))  # smallest plausible p-value
    return p_plus, p_minus, w_plus, n


# ===========================================================================
# 1. Oster (2019) coefficient-stability delta
# ===========================================================================
@register("oster_delta")
def _branch_oster_delta(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    numeric = [
        c.name for c in fp.columns
        if c.kind in {"continuous", "binary", "count"} and c.name not in _excl
    ]
    # outcome: config, else first continuous; treatment: config, else a binary, else
    # the first numeric column that isn't the outcome.
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (cont[0] if cont else None)
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    if treatment is None:
        treatment = next((c for c in bins if c != outcome), None)
    if treatment is None:
        treatment = next((c for c in numeric if c != outcome), None)
    if cfg.get("controls"):
        controls = [c for c in cfg["controls"] if c in df.columns and c not in {outcome, treatment}]
    else:
        controls = [c for c in numeric if c not in {outcome, treatment}]

    if outcome is None or treatment is None:
        summary.append(
            'Oster δ 失败：需要 结果(连续) + 处理 两列。'
            'config={"outcome":..,"treatment":..,"controls":[..]}。'
        )
        return
    if not controls:
        summary.append(
            "Oster δ 失败：需要 ≥1 个控制/协变量（系数稳定性需对比加控制前后的 β 与 R²）。"
            '用 config={"controls":[..]} 指定。'
        )
        return

    sub = df[[outcome, treatment, *controls]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < max(10, len(controls) + 3):
        summary.append(f"Oster δ 失败：去缺后样本 {len(sub)} 太少（需 > 控制数+3 且 ≥10）。")
        return

    try:
        import statsmodels.api as sm

        y = sub[outcome].to_numpy(dtype=float)
        # SHORT regression: outcome ~ treatment  ->  beta0, R0
        x_short = sm.add_constant(sub[[treatment]].to_numpy(dtype=float))
        m_short = sm.OLS(y, x_short).fit()
        beta0 = float(m_short.params[1])
        r0 = float(m_short.rsquared)
        # LONG regression: outcome ~ treatment + controls  ->  betaT (β̃), RT (R̃)
        x_long = sm.add_constant(sub[[treatment, *controls]].to_numpy(dtype=float))
        m_long = sm.OLS(y, x_long).fit()
        betaT = float(m_long.params[1])
        rT = float(m_long.rsquared)

        # R_max: config, else Oster's heuristic min(1.3*R̃, 1.0).
        try:
            r_max = float(cfg["r_max"]) if cfg.get("r_max") is not None else min(1.3 * rT, 1.0)
        except (TypeError, ValueError):
            r_max = min(1.3 * rT, 1.0)
        r_max = min(max(r_max, rT), 1.0)  # R_max must lie in [R̃, 1]

        d_rr = rT - r0          # R̃ - R°  (movement in R² from adding controls)
        d_beta = beta0 - betaT  # β° - β̃  (movement in coefficient)

        # δ : selection-on-unobservables relative to observables that would drive
        #     the treatment effect to ZERO.  δ = β̃·(R̃−R°) / [(β°−β̃)·(R_max−R̃)].
        denom_delta = d_beta * (r_max - rT)
        if abs(denom_delta) < 1e-12 or abs(d_rr) < 1e-12:
            delta = float("nan")
        else:
            delta = (betaT * d_rr) / denom_delta

        # β* : bias-adjusted coefficient under PROPORTIONAL selection (δ = 1).
        #      β* ≈ β̃ − (β°−β̃)·(R_max−R̃)/(R̃−R°).
        if abs(d_rr) < 1e-12:
            beta_star = float("nan")
        else:
            beta_star = betaT - d_beta * (r_max - rT) / d_rr

        estimates["beta_short"] = round(beta0, 6)
        estimates["beta_long"] = round(betaT, 6)
        estimates["r2_short"] = round(r0, 6)
        estimates["r2_long"] = round(rT, 6)
        estimates["r_max"] = round(r_max, 6)
        estimates["delta"] = round(delta, 4) if delta == delta else float("nan")
        estimates["beta_star"] = round(beta_star, 6) if beta_star == beta_star else float("nan")
        estimates["n"] = float(len(sub))

        sign_kept = (beta_star == beta_star) and (np.sign(beta_star) == np.sign(betaT)) and abs(betaT) > 0
        # δ>1 => unobservables must be MORE important than observables to overturn => robust.
        robust_delta = (delta == delta) and (delta > 1.0)
        verdict = (
            "稳健" if (robust_delta and sign_kept)
            else ("脆弱（很小的未观测选择即可推翻）" if (delta == delta and delta < 1.0)
                  else "需谨慎判读")
        )

        pd.DataFrame({
            "quantity": ["beta_short(β°)", "beta_long(β̃)", "R2_short(R°)", "R2_long(R̃)",
                         "R_max", "delta(δ)", "beta_star(β*,δ=1)"],
            "value": [round(beta0, 6), round(betaT, 6), round(r0, 6), round(rT, 6),
                      round(r_max, 6),
                      round(delta, 4) if delta == delta else float("nan"),
                      round(beta_star, 6) if beta_star == beta_star else float("nan")],
        }).to_csv(d / "oster_delta.csv", index=False, encoding="utf-8")
        files.append("oster_delta.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3.6))
            vals = [beta0, betaT, beta_star if beta_star == beta_star else betaT]
            labels = ["short b0", "long bT", "bias-adj b* (delta=1)"]
            colors = ["#4C72B0", "#55A868", "#C44E52"]
            ax.bar(labels, vals, color=colors)
            ax.axhline(0, color="grey", ls="--", lw=1)
            ax.set_ylabel(f"coefficient on {treatment}")
            ax.set_title("Oster (2019) coefficient stability")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3g}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "oster_delta.png", dpi=150)
            plt.close(fig)
            files.append("oster_delta.png")
        except Exception:
            pass

        (d / "oster_summary.txt").write_text(
            "Oster (2019, JBES) 系数稳定性 / 遗漏变量偏差界\n"
            f"结果 {outcome}，处理 {treatment}，控制 {len(controls)} 个；样本 n={len(sub)}\n"
            f"短回归（仅处理）：β° = {beta0:.6f}，R° = {r0:.4f}\n"
            f"长回归（+控制）：β̃ = {betaT:.6f}，R̃ = {rT:.4f}\n"
            f"R_max = {r_max:.4f}（{'config 指定' if cfg.get('r_max') is not None else 'Oster 启发式 min(1.3·R̃,1)'}）\n"
            f"δ（使效应归零所需「未观测/可观测」选择比）= {delta:.4f}\n"
            "  δ = β̃·(R̃−R°) / [(β°−β̃)·(R_max−R̃)]\n"
            f"β*（δ=1 比例选择下的偏差校正系数）= {beta_star:.6f}\n"
            "  β* = β̃ − (β°−β̃)·(R_max−R̃)/(R̃−R°)\n"
            f"判语：{verdict}\n"
            "解读：δ>1 表示未观测混杂需比所有可观测控制「更重要」才能把效应归零（→稳健）；"
            "δ<1 表示很小的未观测选择即可推翻。β* 若仍保号、远离 0 则结论稳健。\n"
            "⚠ 假定：比例选择（未观测与可观测的选择结构成比例）；R_max 是关键敏感参数"
            "（默认 1.3·R̃，可用 config['r_max'] 调）；δ 与 β* 是界、非证明，"
            "且依赖线性可加模型设定。\n",
            encoding="utf-8",
        )
        files.append("oster_summary.txt")
        summary.append(
            f"{entry.method} 完成：处理 {treatment} → {outcome}，控制 {len(controls)} 个；"
            f"β°={beta0:.4f}(R°={r0:.3f}) → β̃={betaT:.4f}(R̃={rT:.3f})；"
            f"R_max={r_max:.3f}；δ={delta:.3f}，β*(δ=1)={beta_star:.4f}；判语：{verdict}。"
            " ⚠ Oster 界假定**比例选择**；R_max 是关键敏感参数（默认 1.3·R̃，Oster 启发式，可 config['r_max'] 调）；"
            "δ/β* 是界非证明，依赖线性可加设定，δ>1 ⇒ 未观测需比可观测更重要才能推翻 ⇒ 稳健。"
        )
        code += [
            "import statsmodels.api as sm  # Oster(2019) coefficient stability",
            f"# short: OLS({outcome}~{treatment}) -> b0,R0; long: +controls -> bT,RT",
            "# delta = bT*(RT-R0) / ((b0-bT)*(Rmax-RT)); beta* = bT - (b0-bT)*(Rmax-RT)/(RT-R0)",
        ]
    except Exception as err:
        summary.append(f"Oster δ 失败：{err}")


# ===========================================================================
# 2. VanderWeele & Ding (2017) E-value
# ===========================================================================
@register("evalue")
def _branch_evalue(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np  # noqa: F401  (kept for parity / future use)
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    numeric = [
        c.name for c in fp.columns
        if c.kind in {"continuous", "binary", "count"} and c.name not in _excl
    ]
    # exposure: config, else a binary, else first numeric ≠ outcome.
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else None
    exposure = cfg.get("exposure") or cfg.get("treatment")
    exposure = exposure if exposure in df.columns else None
    # outcome auto: prefer a BINARY column (logistic path), else a continuous one.
    if outcome is None:
        outcome = next((c for c in bins if c != exposure), None) or (cont[0] if cont else None)
    if exposure is None:
        exposure = next((c for c in bins if c != outcome), None) or next(
            (c for c in numeric if c != outcome), None)
    if cfg.get("controls"):
        controls = [c for c in cfg["controls"] if c in df.columns and c not in {outcome, exposure}]
    else:
        controls = []

    if outcome is None or exposure is None:
        summary.append(
            'E-value 失败：需要 结果 + 暴露 两列。config={"outcome":..,"exposure":..}。'
        )
        return

    sub = df[[outcome, exposure, *controls]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10:
        summary.append(f"E-value 失败：去缺后样本 {len(sub)} 太少（需 ≥10）。")
        return
    y_unique = set(pd.unique(sub[outcome].dropna()))
    outcome_binary = y_unique <= {0, 1} and len(y_unique) == 2

    try:
        import statsmodels.api as sm

        rr_kind = ""
        approx_note = ""
        if outcome_binary:
            # logistic outcome ~ exposure (+controls) -> OR -> RR (sqrt approx).
            X = sm.add_constant(sub[[exposure, *controls]].to_numpy(dtype=float))
            yv = sub[outcome].to_numpy(dtype=float)
            mod = sm.Logit(yv, X).fit(disp=0)
            beta = float(mod.params[1])
            se = float(mod.bse[1])
            or_pt = math.exp(beta)
            or_lo = math.exp(beta - 1.959963985 * se)
            or_hi = math.exp(beta + 1.959963985 * se)
            p0 = float(yv.mean())  # marginal outcome prevalence (rarity check)
            # VanderWeele 2017: rare outcome RR≈OR; common outcome use sqrt(OR).
            if p0 <= 0.15:
                rr_pt, rr_lo, rr_hi = or_pt, or_lo, or_hi
                rr_kind = "RR≈OR（罕见结局，p0≤0.15）"
            else:
                rr_pt, rr_lo, rr_hi = math.sqrt(or_pt), math.sqrt(or_lo), math.sqrt(or_hi)
                rr_kind = f"RR≈√OR（常见结局，p0={p0:.2f}）"
            approx_note = "logistic OR → RR 为近似（VanderWeele 2017）"
            effect_txt = f"OR={or_pt:.4f}（95% CI [{or_lo:.4f}, {or_hi:.4f}]）→ {rr_kind}"
            estimates["or_point"] = round(or_pt, 4)
            estimates["outcome_prevalence"] = round(p0, 4)
        else:
            # continuous outcome: standardized mean difference d -> RR via exp(0.91*d).
            # Exposure must be (made) binary to form two groups for the SMD.
            ex_vals = set(pd.unique(sub[exposure].dropna()))
            if not (ex_vals <= {0, 1} and len(ex_vals) == 2):
                if len(ex_vals) == 2:
                    hi_v = sorted(ex_vals)[1]
                    sub[exposure] = (sub[exposure] == hi_v).astype(int)
                else:
                    med = float(sub[exposure].median())
                    sub[exposure] = (sub[exposure] > med).astype(int)
            g1 = sub.loc[sub[exposure] == 1, outcome].to_numpy(dtype=float)
            g0 = sub.loc[sub[exposure] == 0, outcome].to_numpy(dtype=float)
            n1, n0 = g1.size, g0.size
            if n1 < 2 or n0 < 2:
                summary.append("E-value 失败：连续结果需暴露分两组、每组 ≥2 例。")
                return
            sp = math.sqrt(((n1 - 1) * g1.var(ddof=1) + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2))
            dval = (g1.mean() - g0.mean()) / sp if sp > 1e-12 else 0.0
            se_d = math.sqrt((n1 + n0) / (n1 * n0) + dval ** 2 / (2 * (n1 + n0)))
            d_lo, d_hi = dval - 1.959963985 * se_d, dval + 1.959963985 * se_d
            # Chinn (2000) / VanderWeele: RR = exp(0.91 * d).
            rr_pt = math.exp(0.91 * dval)
            rr_lo = math.exp(0.91 * d_lo)
            rr_hi = math.exp(0.91 * d_hi)
            rr_kind = "RR=exp(0.91·d)（连续结果，标准化均差 d）"
            approx_note = "标准化均差 d → RR 为近似（Chinn 2000 / VanderWeele）"
            effect_txt = f"标准化均差 d={dval:.4f}（95% CI [{d_lo:.4f}, {d_hi:.4f}]）→ {rr_kind}"
            estimates["smd_d"] = round(dval, 4)

        # E-values: point + CI-limit-nearest-the-null.
        e_point = _evalue_from_rr(rr_pt)
        e_ci = _evalue_for_ci(rr_pt, rr_lo, rr_hi)
        estimates["rr_used"] = round(rr_pt, 4)
        estimates["rr_ci_lb"] = round(rr_lo, 4)
        estimates["rr_ci_ub"] = round(rr_hi, 4)
        estimates["evalue_point"] = round(e_point, 4)
        estimates["evalue_ci"] = round(e_ci, 4)
        estimates["n"] = float(len(sub))

        ci_crosses = rr_lo <= 1.0 <= rr_hi
        robust = "稳健" if e_ci > 2.0 else ("中等" if e_ci > 1.25 else "脆弱（近零或 CI 跨过无效）")
        if ci_crosses:
            robust = "效应本身不显著（CI 跨 RR=1，E-value(CI)=1）"

        pd.DataFrame({
            "quantity": ["RR_point", "RR_CI_lower", "RR_CI_upper", "E-value(point)", "E-value(CI)"],
            "value": [round(rr_pt, 4), round(rr_lo, 4), round(rr_hi, 4),
                      round(e_point, 4), round(e_ci, 4)],
        }).to_csv(d / "evalue.csv", index=False, encoding="utf-8")
        files.append("evalue.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3.4))
            bars = ["E-value\n(point)", "E-value\n(CI limit)"]
            vals = [e_point, e_ci]
            ax.bar(bars, vals, color=["#4C72B0", "#C44E52"])
            ax.axhline(1.0, color="grey", ls="--", lw=1, label="null (E=1)")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("E-value (RR scale)")
            ax.set_title("VanderWeele-Ding (2017) E-value")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "evalue.png", dpi=150)
            plt.close(fig)
            files.append("evalue.png")
        except Exception:
            pass

        (d / "evalue_summary.txt").write_text(
            "VanderWeele & Ding (2017) E-value（未观测混杂敏感性）\n"
            f"结果 {outcome}，暴露 {exposure}"
            f"{('，控制 ' + str(len(controls)) + ' 个') if controls else ''}；样本 n={len(sub)}\n"
            f"观测效应：{effect_txt}\n"
            f"用于 E-value 的 RR = {rr_pt:.4f}（95% CI [{rr_lo:.4f}, {rr_hi:.4f}]）\n"
            f"E-value（点估计）= {e_point:.4f}\n"
            f"E-value（CI 近无效端）= {e_ci:.4f}\n"
            "  E = RR + √(RR·(RR−1))（RR≥1；RR<1 取 1/RR 后同式）\n"
            f"判语：{robust}\n"
            "解读：E-value 是「一个未观测混杂——同时与暴露、结果都关联——要把观测效应"
            "完全解释掉，所需的最小关联强度（RR 尺度）」。越大越稳健；"
            "CI 的 E-value 衡量把效应推到「统计不显著」所需的强度。\n"
            f"⚠ {approx_note}；E-value 是必要强度的下界、非混杂存在性检验；"
            "RR 转换为近似（已注明用哪种）；高 E-value 不等于无混杂、只是需要更强的混杂才能推翻。\n",
            encoding="utf-8",
        )
        files.append("evalue_summary.txt")
        summary.append(
            f"{entry.method} 完成：暴露 {exposure} → {outcome}；{effect_txt}；"
            f"E-value 点={e_point:.3f}，CI={e_ci:.3f}；判语：{robust}。"
            f" ⚠ {approx_note}；E-value 是「未观测混杂需同时与暴露+结果关联的最小强度（RR 尺度）」，"
            "是必要强度的界、非混杂存在性检验，RR 转换为近似（已注明）；越大越稳健。"
        )
        code += [
            "import math, statsmodels.api as sm  # VanderWeele-Ding(2017) E-value",
            "# binary outcome: Logit -> OR -> RR (rare:RR=OR; common:RR=sqrt(OR))",
            "# continuous: SMD d -> RR=exp(0.91*d); E = RR + sqrt(RR*(RR-1)) [RR>=1]",
        ]
    except Exception as err:
        summary.append(f"E-value 失败：{err}")


# ===========================================================================
# 3. Rosenbaum (2002) sensitivity bounds
# ===========================================================================
@register("rosenbaum_bounds")
def _branch_rosenbaum_bounds(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (cont[0] if cont else None)
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    if treatment is None:
        treatment = next((c for c in bins if c != outcome), None)

    if outcome is None or treatment is None:
        summary.append(
            'Rosenbaum 界失败：需要 结果(连续) + 处理(二值) 两列。'
            'config={"outcome":..,"treatment":..}。'
        )
        return

    sub = df[[outcome, treatment]].apply(pd.to_numeric, errors="coerce").dropna()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1} and len(tvals) == 2):
        if len(tvals) == 2:  # map two arbitrary values -> 0/1 (higher = treated)
            hi = sorted(tvals)[1]
            sub[treatment] = (sub[treatment] == hi).astype(int)
        else:
            summary.append("Rosenbaum 界失败：处理变量必须是二值（0/1 或恰两类）。")
            return
    sub[treatment] = sub[treatment].astype(int)
    treated = sub.loc[sub[treatment] == 1, outcome].to_numpy(dtype=float)
    control = sub.loc[sub[treatment] == 0, outcome].to_numpy(dtype=float)
    n_t, n_c = treated.size, control.size
    if n_t < 3 or n_c < 3:
        summary.append(f"Rosenbaum 界失败：处理组 {n_t}、对照组 {n_c}，样本太少（各需 ≥3）。")
        return

    try:
        # Pair treated with controls (1:1, no replacement) by nearest outcome rank so
        # we can apply the signed-rank sensitivity bound. If unequal sizes, pair as
        # many as min(n_t,n_c) by sorted order (a transparent pairing; user can supply
        # a matched dataset via psm/ipw and re-feed). Differences = treated - control.
        m = min(n_t, n_c)
        t_sorted = np.sort(treated)[:m]
        c_sorted = np.sort(control)[:m]
        diffs = t_sorted - c_sorted
        # ensure the test is oriented toward a positive treatment effect: if the
        # treated group median is lower, flip sign so "breakdown" is well-defined.
        flipped = False
        if np.median(diffs) < 0:
            diffs = -diffs
            flipped = True

        alpha = 0.05
        try:
            g_max = float(cfg.get("gamma_max", 3.0))
        except (TypeError, ValueError):
            g_max = 3.0
        g_max = min(max(g_max, 1.5), 10.0)
        try:
            g_step = float(cfg.get("gamma_step", 0.1))
        except (TypeError, ValueError):
            g_step = 0.1
        g_step = min(max(g_step, 0.05), 0.5)
        gammas = [round(1.0 + g_step * i, 4) for i in range(int((g_max - 1.0) / g_step) + 1)]
        if gammas[-1] < g_max:
            gammas.append(round(g_max, 4))

        rows = []
        breakdown_gamma = float("nan")
        prev_below = None
        for g in gammas:
            p_plus, p_minus, w_plus, npairs = _rosenbaum_signed_rank_pvals(diffs, g)
            rows.append({"gamma": g, "p_upper(p+)": round(p_plus, 5),
                         "p_lower(p-)": round(p_minus, 5)})
            below = (p_plus == p_plus) and (p_plus < alpha)
            if prev_below is True and not below and breakdown_gamma != breakdown_gamma:
                breakdown_gamma = g  # first Γ at which p+ crosses 0.05
            prev_below = below
        # if it was already non-significant at Γ=1, breakdown is essentially 1.0
        if rows and rows[0]["p_upper(p+)"] >= alpha and breakdown_gamma != breakdown_gamma:
            breakdown_gamma = 1.0
        # never crossed within the grid
        all_below = all(r["p_upper(p+)"] < alpha for r in rows if r["p_upper(p+)"] == r["p_upper(p+)"])

        tab = pd.DataFrame(rows)
        tab.to_csv(d / "rosenbaum_bounds.csv", index=False, encoding="utf-8")
        files.append("rosenbaum_bounds.csv")
        p_base = rows[0]["p_upper(p+)"]
        estimates["p_value_gamma1"] = round(p_base, 5)
        estimates["breakdown_gamma"] = (round(breakdown_gamma, 3) if breakdown_gamma == breakdown_gamma
                                        else float("nan"))
        estimates["n_pairs"] = float(len(diffs))
        estimates["gamma_max_tested"] = round(gammas[-1], 3)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(tab["gamma"], tab["p_upper(p+)"], "o-", color="#C44E52", label="upper bound p+")
            ax.plot(tab["gamma"], tab["p_lower(p-)"], "s-", color="#4C72B0", label="lower bound p-")
            ax.axhline(alpha, color="grey", ls="--", lw=1, label="alpha=0.05")
            if breakdown_gamma == breakdown_gamma and breakdown_gamma > 1.0:
                ax.axvline(breakdown_gamma, color="green", ls=":", lw=1.2,
                           label=f"breakdown Gamma={breakdown_gamma:.2f}")
            ax.set_xlabel("Gamma (hidden-bias odds ratio)")
            ax.set_ylabel("bounding p-value")
            ax.set_title("Rosenbaum sensitivity bounds")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "rosenbaum_bounds.png", dpi=150)
            plt.close(fig)
            files.append("rosenbaum_bounds.png")
        except Exception:
            pass

        if all_below:
            bd_txt = f"在测试范围 Γ≤{gammas[-1]:g} 内 p⁺ 始终 <0.05（极稳健，未触发崩溃 Γ）"
            verdict = "极稳健"
        elif breakdown_gamma == breakdown_gamma and breakdown_gamma <= 1.0:
            bd_txt = "Γ=1（无隐藏偏倚）下结果已不显著"
            verdict = "脆弱（基线即不显著）"
        elif breakdown_gamma == breakdown_gamma:
            bd_txt = f"崩溃 Γ ≈ {breakdown_gamma:.2f}"
            verdict = ("稳健" if breakdown_gamma >= 2.0
                       else ("中等" if breakdown_gamma >= 1.3 else "脆弱"))
        else:
            bd_txt = "未确定崩溃 Γ"
            verdict = "需谨慎判读"
        flip_txt = "；⚠ 处理组结果中位低于对照，已翻转符号使「正效应」检验有定义" if flipped else ""
        pair_txt = (f"；按结果秩序 1:1 配对 {len(diffs)} 对（处理 {n_t}/对照 {n_c}）"
                    if n_t != n_c else f"；1:1 配对 {len(diffs)} 对")

        (d / "rosenbaum_summary.txt").write_text(
            "Rosenbaum (2002) 敏感性界（隐藏偏倚 / 未观测混杂，Wilcoxon 符号秩）\n"
            f"结果 {outcome}，处理 {treatment}{pair_txt}{flip_txt}\n"
            f"Γ=1（无隐藏偏倚基线）单侧 p⁺ = {p_base:.5f}\n"
            f"{bd_txt}\n"
            f"判语：{verdict}\n"
            "解读：Γ 是「一对内两单位接受处理几率之比」的隐藏偏倚上界。"
            "Γ=1 即无隐藏偏倚；崩溃 Γ 是「一个未观测混杂需把对内分配几率拉开到多大倍数」"
            "才能让结果失去显著（越大越稳健）。p⁺/p⁻ 是该 Γ 下 p 值的上/下界。\n"
            "⚠ 界定隐藏偏倚的影响、非检验其是否存在；依赖配对/比较结构（此处按结果秩序配对，"
            "理想应喂入 PSM/IPW 的匹配样本）；用符号秩统计量 + 正态近似（小样本偏保守）。\n\n"
            + tab.to_string(index=False),
            encoding="utf-8",
        )
        files.append("rosenbaum_summary.txt")
        summary.append(
            f"{entry.method} 完成：处理 {treatment} → {outcome}；"
            f"Γ=1 时 p⁺={p_base:.4f}；{bd_txt}；判语：{verdict}。"
            " ⚠ Rosenbaum 界**界定**隐藏偏倚影响、非检验其存在；依赖配对/比较结构"
            "（此处按结果秩序 1:1 配对，理想喂入 PSM/IPW 匹配样本）；用 Wilcoxon 符号秩 + 正态近似。"
            + flip_txt
        )
        code += [
            "from scipy import stats  # Rosenbaum(2002) sensitivity bounds",
            "# signed-rank W+ ; per Gamma: E+ = sum_q*Gamma/(1+Gamma); Z=(W+ - E+)/sqrt(Var+)",
            "# breakdown Gamma = smallest Gamma where upper-bound p+ crosses 0.05",
        ]
    except Exception as err:
        summary.append(f"Rosenbaum 界失败：{err}")
