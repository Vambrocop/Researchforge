"""Causal family branch handler: psm (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("psm")
def _branch_psm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins[0] if bins else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if cfg.get("covariates"):
        covs = [c for c in cfg["covariates"] if c in df.columns and c not in {outcome, treatment}]
    else:
        covs = [c.name for c in fp.columns if c.kind in {"continuous", "binary", "count"}
                and c.name not in (_excl | {outcome, treatment})]
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
