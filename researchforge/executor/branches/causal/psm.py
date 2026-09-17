"""Causal family branch handler: psm (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.profiler.semantics import (
    survival_event_column,
    treatment_never_named,
)
from researchforge.executor.run import resolve_outcome, resolve_treatment


@register("psm")
def _branch_psm(ctx: Ctx) -> None:
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
        summary.append('倾向得分匹配失败：需要 二值处理 + 连续结果 + ≥1 协变量。'
                       'config={"treatment":..,"outcome":..,"covariates":[..]}。')
        return

    sub = df[[outcome, treatment, *covs]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:  # map two arbitrary values to 0/1 (higher = treated)
            hi = sorted(tvals)[1]
            sub[treatment] = (sub[treatment] == hi).astype(int)
        else:
            summary.append("倾向得分匹配失败：处理变量必须是二值（0/1 或恰两类）。")
            return
    sub[treatment] = sub[treatment].astype(int)
    n_t, n_c = int((sub[treatment] == 1).sum()), int((sub[treatment] == 0).sum())
    if n_t < 5 or n_c < 5:
        summary.append(f"倾向得分匹配失败：处理组 {n_t}、对照组 {n_c}，样本太少。")
        return

    try:
        import statsmodels.formula.api as smf
        from scipy import stats as _st

        rhs = " + ".join(f'Q("{c}")' for c in covs)
        ps_model = smf.logit(f'Q("{treatment}") ~ {rhs}', data=sub).fit(disp=0)
        ps = ps_model.predict(sub).clip(1e-6, 1 - 1e-6)
        sub["_lp"] = np.log(ps / (1 - ps))  # match on the logit (linear predictor), per Austin
        caliper = 0.2 * float(sub["_lp"].std(ddof=1))
        treated = sub[sub[treatment] == 1]
        controls = sub[sub[treatment] == 0]
        ctrl_lp = controls["_lp"].to_dict()

        used: set = set()
        pairs: list[tuple] = []  # greedy 1:1 NN on _lp, no replacement, within caliper
        for ti, trow in treated.sort_values("_lp", ascending=False).iterrows():
            best, bestd = None, None  # nearest unused control WITHIN the caliper
            for ci, lp in ctrl_lp.items():
                if ci in used:
                    continue
                dlp = abs(trow["_lp"] - lp)
                if dlp <= caliper and (bestd is None or dlp < bestd):
                    bestd, best = dlp, ci
            if best is not None:
                used.add(best)
                pairs.append((ti, best))
        if len(pairs) < 3:
            summary.append(f"倾向得分匹配失败：卡尺内仅匹配到 {len(pairs)} 对（共同支撑不足）。")
            return

        t_idx = [p[0] for p in pairs]
        c_idx = [p[1] for p in pairs]
        diffs = sub.loc[t_idx, outcome].to_numpy(dtype=float) - sub.loc[c_idx, outcome].to_numpy(dtype=float)
        att = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        tstat = att / se if se and se > 0 else float("nan")
        df_t = len(diffs) - 1  # matched-pairs t-test df; small-n matches (as few as 3 pairs) need
        # the t reference distribution, not normal, or the p-value is anti-conservative.
        pval = (float(2 * _st.t.sf(abs(tstat), df=df_t)) if tstat == tstat and df_t >= 1
                else float("nan"))

        def _smd(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            return float((a.mean() - b.mean()) / sp) if sp > 1e-12 else 0.0

        bal = pd.DataFrame([
            {"covariate": c,
             "smd_before": round(_smd(treated[c], controls[c]), 3),
             "smd_after": round(_smd(sub.loc[t_idx, c], sub.loc[c_idx, c]), 3)}
            for c in covs
        ])
        bal.to_csv(d / "balance.csv", index=False, encoding="utf-8")
        files.append("balance.csv")
        max_smd_after = float(bal["smd_after"].abs().max())

        estimates.update({"att": att, "se": se, "pvalue": pval, "n_treated": float(n_t),
                          "n_matched_pairs": float(len(pairs)), "max_abs_smd_after": max_smd_after})

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(treated["_lp"], bins=20, alpha=0.5, label="treated", color="#C44E52")
            ax.hist(controls["_lp"], bins=20, alpha=0.5, label="control", color="#4C72B0")
            ax.set_xlabel("propensity (logit)")
            ax.set_ylabel("count")
            ax.set_title("Propensity overlap (common support)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "propensity_overlap.png", dpi=150)
            plt.close(fig)
            files.append("propensity_overlap.png")
        except Exception:
            pass

        bal_ok = "达标(|SMD|<0.1)" if max_smd_after < 0.1 else f"⚠ 残留不平衡(最大|SMD|={max_smd_after:.2f})"
        sig = "显著" if (pval == pval and pval < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成：ATT={att:.4f}（SE={se:.4f}, p={pval:.3g}，{sig}）；"
            f"匹配 {len(pairs)} 对（处理组 {n_t}/对照 {n_c}）；匹配后协变量平衡 {bal_ok}。"
            " ⚠ PSM 假定**可忽略性/选择仅基于可观测**（无未观测混杂）——不可检验的强假设，"
            "PSM 不能修正未观测混杂；估计的是 **ATT**（对处理组）非 ATE；需共同支撑 + 匹配后平衡。"
            " SE 为配对差简化估计（未计倾向得分估计不确定性、亦未计无放回匹配的依赖；Abadie-Imbens 方差更严）。"
        )
        code += [
            "import statsmodels.formula.api as smf  # 倾向得分匹配 (PSM)",
            f"# logit({treatment} ~ 协变量) -> 倾向得分 -> 线性预测子上 1:1 最近邻(卡尺 0.2σ) -> ATT",
        ]
    except Exception as err:
        summary.append(f"倾向得分匹配失败：{err}")
