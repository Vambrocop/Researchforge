"""Causal family branch handler: ipw (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.profiler.semantics import (
    survival_event_column,
    treatment_never_named,
)
from researchforge.executor.run import resolve_outcome, resolve_treatment


@register("ipw")
def _branch_ipw(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    # H4d: fp.binary_columns is EVERY binary column, so [0] meant "first
    # binary in file order" — on [.., event, treatment, ..] that picked the
    # survival EVENT as the intervention. resolve_treatment applies the name
    # signal that roles.py already computed (and records the binding).
    treatment = resolve_treatment(fp, cfg, fp.binary_columns or bins, df=df)
    # H4c: bind the DETECTED outcome among the non-treatment continuous columns
    # (config > high-confidence outcome name > first non-treatment-named) instead of
    # plain column order — and record it, so the report can name what was modeled.
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        resolve_outcome(fp, cfg, [c for c in cont if c != treatment])
        if [c for c in cont if c != treatment] else None)
    if cfg.get("covariates"):
        covs = [c for c in cfg["covariates"] if c in df.columns and c not in {outcome, treatment}]
    else:
        # 处理后协变量: in a survival-shaped frame the event indicator is a descendant of
        # the duration (event = 1{T<=C}) and hence of the treatment, so adjusting for it is
        # post-treatment adjustment. Measured 33-73% attenuation of the ATT, and an outright
        # failure once the event rate saturates. AUTO set only — an explicit config
        # covariates list stays the user's call — and disclosed in the summary.
        _post = survival_event_column(fp)
        # A control/negated flag is a deterministic function of the treatment (control_arm
        # = 1 - treated), so it is never a confounder — and putting it in the propensity
        # model separates it perfectly, which surfaced as a naked "Singular matrix".
        _mirror = {c.name for c in fp.columns
                   if c.name != treatment and treatment_never_named(c.name)}
        covs = [c.name for c in fp.columns if c.kind in {"continuous", "binary", "count"}
                and c.name not in (_excl | {outcome, treatment} | _mirror
                                   | ({_post} if _post else set()))]
        if _mirror:
            summary.append(
                f"⚠ 已把 {'、'.join(sorted(_mirror))} 排除在自动协变量之外："
                "它标记的是未处理/对照，是处理变量的确定性函数（而非混杂因子），"
                "放进倾向模型会造成完全分离（此前表现为一句裸的 Singular matrix）。"
                "若确需纳入，用 config covariates 显式指定。"
            )
        # Cold review A MUST-FIX 5: the AUTO covariate set is "every numeric column that is
        # not the outcome/treatment" — it cannot tell a confounder from a MEDIATOR, and a
        # mediator is post-treatment adjustment. Measured on a randomised trial whose true
        # effect is -8, with a post-treatment mediator `adherence` carrying the full effect:
        #     IPW -3.207   PSM +1.197 (SIGN FLIPPED)   AIPW -1.358, 95% CI [-2.04, -0.68]
        # i.e. the AIPW interval does not cover the truth at all. Auto-DROPPING mediators is
        # not the fix (there is no name test that separates a mediator from a real
        # confounder, and dropping a confounder is the opposite bias) — but the set was never
        # even shown. Name it, and say what would make it wrong.
        if covs:
            summary.append(
                f"⚠ 自动选取的协变量：{'、'.join(covs)}。它们是按「非结果、非处理的数值列」"
                "挑的，**引擎无法分辨混杂因子与中介变量**。若其中任何一列是在处理之后"
                "测得的（依从性、实际剂量、随访时长、中间结局），那是处理后调整，"
                "会把效应吸走甚至翻号（实测中介占比 100% 时 PSM 由 −8 变成 +1.197、"
                "AIPW 的 95% CI 完全不覆盖真值）。请核对这份名单，必要时"
                '用 config={"covariates":[..]} 指定基线前（处理前）变量。'
            )
        if _post:
            summary.append(
                f"⚠ 已把 '{_post}' 排除在自动协变量之外：它是生存数据的事件指示列"
                "（event = 1{时长 ≤ 删失时间}），是时长、进而是处理变量的后代，"
                "放进倾向得分属于处理后调整（实测会把 ATT 衰减 33-73%，"
                "事件率饱和时甚至直接失败）。若确需纳入，用 config covariates 显式指定。"
            )
    # 常数协变量：zero variance cannot inform a propensity model and makes the design
    # matrix singular (surfaced by a saturated event rate, where the indicator collapses to
    # one value and the profiler types it `count`, so the survival-event exclusion above
    # cannot see it). Drop with disclosure instead of failing on "Singular matrix".
    _const = [c for c in covs if df[c].dropna().nunique() < 2]
    if _const:
        covs = [c for c in covs if c not in set(_const)]
        summary.append(
            f"⚠ 已剔除常数协变量 {_const}：取值唯一，无法为倾向得分提供信息，"
            "保留会让设计矩阵奇异（报 Singular matrix）。"
        )
    if treatment is None or outcome is None or not covs:
        summary.append('逆概率加权失败：需要 二值处理 + 连续结果 + ≥1 协变量。'
                       'config={"treatment":..,"outcome":..,"covariates":[..]}。')
        return

    sub = df[[outcome, treatment, *covs]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    treated_level = None
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            treated_level = sorted(tvals)[1]  # higher value = treated; disclose so the ATE sign is unambiguous
            sub[treatment] = (sub[treatment] == treated_level).astype(int)
        else:
            summary.append("逆概率加权失败：处理变量必须是二值（0/1 或恰两类）。")
            return
    sub[treatment] = sub[treatment].astype(int)
    n_t, n_c = int((sub[treatment] == 1).sum()), int((sub[treatment] == 0).sum())
    if n_t < 5 or n_c < 5:
        summary.append(f"逆概率加权失败：处理组 {n_t}、对照组 {n_c}，样本太少。")
        return

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        rhs = " + ".join(f'Q("{c}")' for c in covs)
        ps = smf.logit(f'Q("{treatment}") ~ {rhs}', data=sub).fit(disp=0).predict(sub).to_numpy()
        ps = np.clip(ps, 1e-3, 1 - 1e-3)  # bound to avoid exploding weights
        t = sub[treatment].to_numpy(dtype=float)
        y = sub[outcome].to_numpy(dtype=float)
        p_treat = float(t.mean())
        sw = np.where(t == 1, p_treat / ps, (1 - p_treat) / (1 - ps))  # stabilized weights

        # ATE via the marginal structural model: WLS of y ~ T weighted by stabilized weights
        wls = sm.WLS(y, sm.add_constant(t), weights=sw).fit(cov_type="HC1")
        ate, se, pval = float(wls.params[1]), float(wls.bse[1]), float(wls.pvalues[1])
        ess = float(sw.sum() ** 2 / (sw ** 2).sum())          # effective sample size
        extreme = float((((ps < 0.05) | (ps > 0.95)).mean()))  # poor-overlap fraction
        estimates.update({"ate": ate, "se": se, "pvalue": pval, "ess": ess,
                          "max_weight": float(sw.max()), "extreme_ps_frac": extreme,
                          "n": float(len(sub))})

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ps[t == 1], bins=20, alpha=0.5, label="treated", color="#C44E52")
            ax.hist(ps[t == 0], bins=20, alpha=0.5, label="control", color="#4C72B0")
            ax.set_xlabel("propensity score")
            ax.set_ylabel("count")
            ax.set_title("Propensity overlap (IPW positivity)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "propensity_overlap.png", dpi=150)
            plt.close(fig)
            files.append("propensity_overlap.png")
        except Exception:
            pass

        overlap = "良好" if extreme < 0.05 and ess > 0.5 * len(sub) else f"⚠ 重叠/正性存疑(极端倾向 {extreme:.0%}, ESS={ess:.0f}/{len(sub)})"
        sig = "显著" if (pval == pval and pval < 0.05) else "不显著"
        trt_note = f"（处理组 = {treatment}='{treated_level}'）" if treated_level is not None else ""
        summary.append(
            f"{entry.method} 完成：ATE={ate:.4f}（HC1 SE={se:.4f}, p={pval:.3g}，{sig}）{trt_note}；"
            f"稳定化权重，ESS={ess:.0f}/{len(sub)}，最大权重 {sw.max():.2f}；重叠 {overlap}。"
            " ⚠ IPW 估 **ATE**，因果有效仅在**可忽略性 + 正性/重叠**(无未观测混杂、处理概率不近 0/1)下成立——"
            "极端权重会放大方差/偏差(已报 ESS 与极端倾向占比);HC1 SE 未计倾向得分估计不确定性(自助/三明治更严)。"
        )
        code += [
            "import statsmodels.api as sm, statsmodels.formula.api as smf  # 逆概率加权(IPW/MSM)",
            f"# 稳定化权重 sw = T·P(T)/e + (1-T)·P(0)/(1-e); WLS({outcome}~T, weights=sw).T 系数 = ATE",
        ]
    except Exception as err:
        summary.append(f"逆概率加权失败：{err}")
