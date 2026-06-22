"""Branch handlers for the causal SENSITIVITY-ANALYSIS family.

How robust is a causal claim to unmeasured confounding / assumption violations?
Hand-rolled methods (pure Python: statsmodels/scipy/numpy, no R):

  oster_delta       Oster (2019, JBES) coefficient-stability bound for
                    omitted-variable bias (δ + bias-adjusted β*).
  evalue            VanderWeele & Ding (2017) E-value — minimum confounder
                    strength (RR scale) needed to explain away an effect.
  rosenbaum_bounds  Rosenbaum (2002) sensitivity bounds for the matched-pair
                    Wilcoxon signed-rank test — how large a hidden bias Γ would
                    have to be to make the effect non-significant.

Convention choices (disclosed in summaries, logged in docs/loop-decisions.md):
  - oster_delta R_max default = min(1.3 * R̃, 1.0) (Oster's heuristic).
  - evalue RR conversion: logistic OR -> RR via sqrt(OR) (VanderWeele 2017
    common-outcome approximation); continuous d -> RR via exp(0.91 * d) (Chinn).
  - rosenbaum_bounds forms pairs by COVARIATE distance (propensity-score nearest
    neighbour, 0.2·SD caliper; standardized-Euclidean fallback) — NEVER by the
    outcome — then applies Rosenbaum's signed-rank bias bounds. This fixes the
    earlier withdrawn version, which paired on the outcome rank (circular).

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
        # |δ|≥1 => unobservables must be AT LEAST as important as observables to overturn => robust.
        # δ<0 has TWO cases: (a) SUPPRESSOR — controls moved β FURTHER from 0 (|β̃|>|β°|, same sign):
        # unobserved selection would have to run OPPOSITE to observed to nullify => robust; (b) the
        # long β̃ crossed through ~0 to the opposite sign of β° (|β̃|<|β°|): the effect is ~null/fragile,
        # NOT a suppressor. So only (a) counts as robust.
        suppressor = (
            delta == delta and delta < 0
            and abs(betaT) > abs(beta0) and np.sign(betaT) == np.sign(beta0)
        )
        if delta != delta:
            verdict = "需谨慎判读（δ 不可估，R̃≈R° 或 β°≈β̃）"
        elif suppressor and sign_kept:
            verdict = "稳健（加控制后效应反而增强，δ<0 抑制子：未观测需与可观测反向选择才能推翻）"
        elif abs(delta) >= 1.0 and sign_kept:
            verdict = "稳健（|δ|≥1：未观测混杂需≥所有可观测控制的重要性才能归零）"
        else:
            verdict = "脆弱（很小的未观测选择即可推翻）"

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
            "解读：|δ|≥1 表示未观测混杂需≥所有可观测控制的重要性才能把效应归零（→稳健）；"
            "|δ|<1 表示很小的未观测选择即可推翻。δ<0（抑制子情形）=加控制后效应反而增强，"
            "未观测需与可观测【反向】选择才能推翻（→更稳健）。β* 若仍保号、远离 0 则结论稳健。\n"
            "⚠ 假定：比例选择（未观测与可观测的选择结构成比例）；R_max 是关键敏感参数"
            "（默认 1.3·R̃，可用 config['r_max'] 调）；β* 用 Oster 简化(线性)界，与 psacalc 精确三次解"
            "在处理-控制相关较强时可有差异；δ 与 β* 是界、非证明，且依赖线性可加模型设定。\n",
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
# 3. Rosenbaum (2002) sensitivity bounds for the matched-pair signed-rank test
# ===========================================================================
def _greedy_nn_match(anchor_idx, pool_idx, score, caliper):
    """Greedy 1:1 nearest-neighbour matching on a 1-D score, no replacement.

    Pairs each anchor (smaller arm) to the closest unused pool unit whose
    |score| distance is within `caliper` (None = no caliper).  Returns a list of
    (anchor_position, pool_position) index pairs.  Deterministic: anchors are
    processed in ascending-score order.
    """
    import numpy as np

    pairs = []
    used = np.zeros(len(pool_idx), dtype=bool)
    pool_scores = score[pool_idx]
    for a in sorted(anchor_idx, key=lambda i: score[i]):
        dist = np.abs(pool_scores - score[a])
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if not np.isfinite(dist[j]):
            break
        if caliper is not None and dist[j] > caliper:
            continue  # no acceptable match within caliper -> anchor dropped
        used[j] = True
        pairs.append((a, pool_idx[j]))
    return pairs


@register("rosenbaum_bounds")
def _branch_rosenbaum_bounds(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy import stats

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    numeric = [
        c.name for c in fp.columns
        if c.kind in {"continuous", "binary", "count"} and c.name not in _excl
    ]
    # treatment: config, else a binary column.  outcome: config, else first continuous.
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    if treatment is None:
        treatment = bins[0] if bins else None
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else None
    if outcome is None:
        outcome = next((c for c in cont if c != treatment), None)
    if cfg.get("covariates"):
        covs = [c for c in cfg["covariates"] if c in df.columns and c not in {outcome, treatment}]
    else:
        covs = [c for c in numeric if c not in {outcome, treatment}]

    if treatment is None or outcome is None:
        summary.append(
            'Rosenbaum 边界失败：需要 二值处理 + 结果(连续) 两列。'
            'config={"treatment":..,"outcome":..,"covariates":[..]}。'
        )
        return
    if not covs:
        summary.append(
            "Rosenbaum 边界失败：需要 ≥1 个协变量按其匹配处理/对照对（绝不按结果配对）。"
            'config={"covariates":[..]}。'
        )
        return

    sub = df[[outcome, treatment, *covs]].apply(pd.to_numeric, errors="coerce").dropna()
    z_vals = set(pd.unique(sub[treatment].dropna()))
    if not (z_vals <= {0, 1} and len(z_vals) == 2):
        # binarize a 2-value non-0/1 treatment; else fail.
        if len(z_vals) == 2:
            hi = sorted(z_vals)[1]
            sub[treatment] = (sub[treatment] == hi).astype(int)
        else:
            summary.append("Rosenbaum 边界失败：处理列须为二值（0/1 或两类）。")
            return
    if len(sub) < 20:
        summary.append(f"Rosenbaum 边界失败：去缺后样本 {len(sub)} 太少（需 ≥20）。")
        return

    sub = sub.reset_index(drop=True)
    z = sub[treatment].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    X = sub[covs].to_numpy(dtype=float)
    idx_t = np.where(z == 1)[0]
    idx_c = np.where(z == 0)[0]
    if len(idx_t) < 3 or len(idx_c) < 3:
        summary.append(
            f"Rosenbaum 边界失败：处理组 {len(idx_t)} / 对照组 {len(idx_c)}，每组需 ≥3。")
        return

    try:
        # --- 1. covariate matching score: propensity (logit), Euclidean fallback ---
        match_kind = ""
        score = None
        try:
            import statsmodels.api as sm

            logit = sm.Logit(z, sm.add_constant(X)).fit(disp=0)
            phat = np.clip(np.asarray(logit.predict()), 1e-6, 1 - 1e-6)
            score = np.log(phat / (1 - phat))  # logit of propensity score
            caliper = 0.2 * float(np.std(score, ddof=1))
            match_kind = "倾向得分最近邻（logit，0.2·SD caliper）"
        except Exception:
            # standardized-covariate distance fallback (1-D summary = z-scored mean)
            mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
            sd[sd < 1e-12] = 1.0
            Xs = (X - mu) / sd
            score = Xs.mean(axis=1)  # crude 1-D proxy; disclosed
            caliper = None
            match_kind = "标准化协变量均值最近邻（无 caliper；倾向模型不可用时的降级）"

        # match the SMALLER arm into the larger arm (maximizes usable pairs).
        if len(idx_t) <= len(idx_c):
            anchor, pool = list(idx_t), idx_c
        else:
            anchor, pool = list(idx_c), idx_t
        raw_pairs = _greedy_nn_match(anchor, pool, score, caliper)
        n_anchor = len(anchor)
        n_dropped = n_anchor - len(raw_pairs)

        if len(raw_pairs) < 5:
            summary.append(
                f"Rosenbaum 边界失败：caliper 内仅匹配到 {len(raw_pairs)} 对（需 ≥5）。"
                "协变量重叠太差或样本太小。")
            return

        # --- 2. pair outcome differences  d = y(treated) - y(control) ---
        diffs = []
        for a, b in raw_pairs:
            t_unit, c_unit = (a, b) if z[a] == 1 else (b, a)
            diffs.append(y[t_unit] - y[c_unit])
        diffs = np.asarray(diffs, dtype=float)
        median_diff = float(np.median(diffs))
        # orient to the OBSERVED effect direction (test is one-sided in that dir).
        increased = median_diff >= 0
        oriented = diffs if increased else -diffs
        nz = oriented[np.abs(oriented) > 1e-12]  # drop zero differences
        S = nz.size
        if S < 5:
            summary.append(f"Rosenbaum 边界失败：非零配对差仅 {S} 个（需 ≥5）。")
            return

        # --- 3. signed-rank statistic + Rosenbaum Γ bounds (normal approx) ---
        ranks = stats.rankdata(np.abs(nz))           # average ranks for ties
        Sq = float(ranks.sum())
        Sq2 = float((ranks ** 2).sum())
        T = float(ranks[nz > 0].sum())               # rank sum on the positive side

        def _bounds(gamma):
            p_plus = gamma / (1.0 + gamma)
            mu_plus = p_plus * Sq                     # worst-case (largest) mean
            mu_minus = (1.0 - p_plus) * Sq            # best-case (smallest) mean
            sd = math.sqrt(p_plus * (1.0 - p_plus) * Sq2)
            if sd < 1e-12:
                return float("nan"), float("nan")
            p_up = float(stats.norm.sf((T - mu_plus) / sd))   # upper bound on p
            p_lo = float(stats.norm.sf((T - mu_minus) / sd))  # lower bound on p
            return p_lo, p_up

        base_lo, base_p = _bounds(1.0)               # Γ=1: p_lo==p_up == base p
        # critical Γ: smallest Γ whose UPPER-bound p exceeds 0.05 (effect breaks).
        gamma_crit = float("nan")
        if base_p <= 0.05:
            for g in np.arange(1.0, 6.0001, 0.05):
                if _bounds(float(g))[1] > 0.05:
                    gamma_crit = round(float(g), 2)
                    break

        estimates["wilcoxon_T"] = round(T, 3)
        estimates["p_value_gamma1"] = round(base_p, 5)
        estimates["gamma_critical"] = gamma_crit
        estimates["n_pairs"] = float(len(raw_pairs))
        estimates["n_nonzero"] = float(S)
        estimates["median_pair_diff"] = round(median_diff, 5)
        estimates["n_dropped_no_match"] = float(n_dropped)

        dir_txt = "提高" if increased else "降低"
        if base_p > 0.05:
            verdict = f"Γ=1 下效应已不显著（p={base_p:.3f}），敏感性分析不适用"
        elif gamma_crit != gamma_crit:
            verdict = "极稳健（Γ≤6 仍显著：需极大的隐性偏差才能推翻）"
        elif gamma_crit >= 2.0:
            verdict = f"稳健（需 Γ≥{gamma_crit:g} 的隐性偏差才能使结论不显著）"
        elif gamma_crit >= 1.3:
            verdict = f"中等（Γ≈{gamma_crit:g} 即可推翻——中度隐性偏差就敏感）"
        else:
            verdict = f"脆弱（Γ≈{gamma_crit:g} 的微弱隐性偏差即可推翻）"

        grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        rows = [(g, *_bounds(g)) for g in grid]
        pd.DataFrame(rows, columns=["Gamma", "p_lower", "p_upper"]).round(5).to_csv(
            d / "rosenbaum_bounds.csv", index=False, encoding="utf-8")
        files.append("rosenbaum_bounds.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gs = np.arange(1.0, 6.0001, 0.05)
            ups = [_bounds(float(g))[1] for g in gs]
            los = [_bounds(float(g))[0] for g in gs]
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.plot(gs, ups, color="#C44E52", label="upper-bound p")
            ax.plot(gs, los, color="#4C72B0", lw=1, ls="--", label="lower-bound p")
            ax.axhline(0.05, color="grey", ls=":", lw=1, label="alpha = 0.05")
            if gamma_crit == gamma_crit:
                ax.axvline(gamma_crit, color="black", lw=1)
                ax.text(gamma_crit, 0.5, f" Gamma*={gamma_crit:g}", fontsize=8,
                        rotation=90, va="center")
            ax.set_xlabel("Gamma (hidden-bias magnitude)")
            ax.set_ylabel("bounding p-value (one-sided)")
            ax.set_title("Rosenbaum sensitivity bounds (signed-rank)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "rosenbaum_bounds.png", dpi=150)
            plt.close(fig)
            files.append("rosenbaum_bounds.png")
        except Exception:
            pass

        gamma_txt = "未在 Γ≤6 内出现" if gamma_crit != gamma_crit else f"{gamma_crit:g}"
        (d / "rosenbaum_summary.txt").write_text(
            "Rosenbaum (2002) 隐性偏差敏感性边界（匹配对 Wilcoxon signed-rank）\n"
            f"结果 {outcome}，处理 {treatment}，按 {len(covs)} 个协变量匹配；"
            f"配对方式：{match_kind}\n"
            f"匹配成功 {len(raw_pairs)} 对（caliper 外丢弃 {n_dropped}）；"
            f"非零配对差 S={S}；处理使结果{dir_txt}（中位配对差={median_diff:.4f}）\n"
            f"signed-rank T={T:.2f}；Γ=1 下 p={base_p:.4f}\n"
            f"临界 Γ*（使上界 p 超过 0.05）= {gamma_txt}\n"
            f"判语：{verdict}\n"
            "解读：Γ 是「配对内两单位接受处理的几率之比」的上界——Γ=1 即无隐性偏差（随机化）。"
            "Γ* 越大，越需要强的未观测混杂才能推翻当前显著结论 ⇒ 越稳健。\n"
            "⚠ 关键前提：配对按【协变量】（倾向得分/标准化距离）而非结果——匹配质量决定可信度，"
            "协变量重叠差或残余不均衡会使界失真；signed-rank 假定配对差对称；"
            "用正态近似（S 较小时偏差大）；Rosenbaum 界只针对【隐性】（未观测）偏差，"
            "不修正可观测协变量的残余不均衡，也非混杂存在性检验。\n",
            encoding="utf-8",
        )
        files.append("rosenbaum_summary.txt")
        summary.append(
            f"{entry.method} 完成：按 {len(covs)} 协变量匹配 {len(raw_pairs)} 对"
            f"（丢 {n_dropped}），处理使 {outcome} {dir_txt}；signed-rank T={T:.1f}，"
            f"Γ=1 下 p={base_p:.4f}，临界 Γ*={gamma_txt}；判语：{verdict}。"
            " ⚠ 配对按**协变量**（倾向得分/标准化距离）非结果，匹配质量决定可信度；"
            "Γ 是配对内处理几率比的上界（Γ=1=无隐性偏差），Γ* 越大越稳健；"
            "仅针对未观测偏差、用正态近似、非混杂存在性检验。"
        )
        code += [
            "from scipy import stats  # Rosenbaum(2002) signed-rank sensitivity bounds",
            "# 1) match treated/control by COVARIATE distance (propensity NN + caliper)",
            "# 2) pair diffs d=y_t-y_c; signed-rank T=sum(rank|d| where d>0)",
            "# 3) per Gamma: mu+=Gamma/(1+Gamma)*Sq, var=p(1-p)*Sq2; p_upper=norm.sf((T-mu+)/sd)",
            "# Gamma* = smallest Gamma with upper-bound p > 0.05",
        ]
    except Exception as err:
        summary.append(f"Rosenbaum 边界失败：{err}")


