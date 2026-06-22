"""Causal family branch handler: staggered_did (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("staggered_did")
def _branch_staggered_did(ctx: Ctx) -> None:
    # Sun & Abraham (2021) interaction-weighted estimator: heterogeneity-robust event-study /
    # ATT under STAGGERED adoption. Fits cohort-specific CATT(g,e) (cohort x relative-time dummies,
    # e=-1 omitted, never-treated as the clean control), then aggregates ATT(e) = Σ_g (N_g/ΣN) CATT(g,e)
    # weighted by cohort sample shares — avoiding the negative-weight ("bad comparison") bias of pooled
    # TWFE event studies. SE via delta method on the cluster-robust covariance. Pure Python (no R did).
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    if not unit or not time:
        summary.append('交错DiD失败：需要面板数据(单位列+时间列)。config={"unit":..,"time":..}。')
        return
    _excl = {unit, time}
    bins_ = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    treatment = cfg.get("treatment") if cfg.get("treatment") in df.columns else (
        fp.treatment_candidates[0] if fp.treatment_candidates else (bins_[0] if bins_ else None))
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        next((c for c in cont if c != treatment), None))
    if treatment is None or outcome is None:
        summary.append('交错DiD失败：需要 二值处理(随时间开启) + 连续结果。config={"treatment":..,"outcome":..}。')
        return

    sub = df[[unit, time, treatment, outcome]].dropna().copy()
    tvals = set(pd.unique(sub[treatment].dropna()))
    if not (tvals <= {0, 1}):
        if len(tvals) == 2:
            sub[treatment] = (sub[treatment] == sorted(tvals)[1]).astype(int)
        else:
            summary.append("交错DiD失败：处理变量必须二值。")
            return
    sub[treatment] = sub[treatment].astype(int)
    sub[time] = pd.to_numeric(sub[time], errors="coerce")
    sub = sub.dropna(subset=[time])

    onset = sub[sub[treatment] == 1].groupby(unit)[time].min()
    if onset.empty:
        summary.append("交错DiD失败：没有任何单位被处理。")
        return
    sub["_cohort"] = sub[unit].map(onset)        # NaN = never-treated
    has_never = bool(sub["_cohort"].isna().any())
    # Sun-Abraham needs a CLEAN control group. We require never-treated units (the most robust
    # control): with none, the saturated cohort x period spec is collinear with unit+time FE (would
    # need last-cohort-as-control + period trimming) -> degrade honestly toward Callaway-Sant'Anna.
    if not has_never:
        summary.append("交错DiD失败：Sun-Abraham 交互加权估计需要『从未处理』对照组(本数据所有单位最终都被处理)；"
                       "请改用 R did(Callaway-Sant'Anna，可用『尚未处理』作对照)或提供含从未处理单位的面板。")
        return
    cohorts = sorted(float(c) for c in onset.unique())

    try:
        L = max(2, min(12, int(cfg.get("window", 5))))
    except (TypeError, ValueError):
        L = 5

    sub["_rel"] = sub[time] - sub["_cohort"]   # NaN for never-treated

    def _relbin(r):
        return np.nan if pd.isna(r) else int(max(-L, min(L, r)))

    sub["_relb"] = sub["_rel"].apply(_relbin)

    # cohort-specific relative-time dummies (e=-1 reference); never-treated & e=-1 are all-zero.
    # Column names are guaranteed-safe identifiers so patsy param names == column names (no quoting).
    dummy_map = {}   # (cohort, e) -> column name
    for gi, g in enumerate(cohorts):
        for e in range(-L, L + 1):
            if e == -1:
                continue
            mask = (sub["_cohort"] == g) & (sub["_relb"] == e)
            if int(mask.sum()) == 0:
                continue
            col = f"saD_{gi}_{e + L}"
            sub[col] = mask.astype(float)
            dummy_map[(g, e)] = col
    if not dummy_map:
        summary.append("交错DiD失败：处理单位的事件时间变化不足(需要前后多期)。")
        return

    try:
        import statsmodels.formula.api as smf
        from scipy.stats import norm

        rhs = " + ".join(dummy_map.values())
        formula = f'Q("{outcome}") ~ {rhs} + C(Q("{unit}")) + C(Q("{time}"))'
        model = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub[unit]})
        params, Vc = model.params, model.cov_params()
        # bare safe identifier names match patsy param names verbatim; fail LOUD if any coefficient is
        # missing (e.g. dropped for collinearity) rather than silently mislabeling a CATT.
        missing = [col for col in dummy_map.values() if col not in params.index]
        if missing:
            summary.append(f"交错DiD失败：{len(missing)} 个队列×事件期系数未进入模型(可能共线)，无法可靠聚合。")
            return
        pname = {ge: col for ge, col in dummy_map.items()}
        # Sun-Abraham weight for CATT(g,e): share of cohort g among units OBSERVED at relative time e
        # (per-(g,e) unit counts, not a fixed cohort size -> correct under UNBALANCED panels too;
        # for a balanced panel this reduces to the cohort-size share).
        treated_obs = sub[sub["_cohort"].notna()]
        ng_e = treated_obs.groupby(["_cohort", "_relb"])[unit].nunique()

        def _agg_vector(e):
            present = [g for g in cohorts if (g, e) in dummy_map]
            counts = {g: float(ng_e.get((g, e), 0.0)) for g in present}
            tot = float(sum(counts.values()))
            if tot <= 0:
                return None, present
            vec = pd.Series(0.0, index=params.index)
            for g in present:
                vec[pname[(g, e)]] = counts[g] / tot
            return vec, present

        def _att_se(vec):
            att = float(vec.values @ params.values)
            var = float(vec.values @ Vc.values @ vec.values)
            se = float(np.sqrt(max(var, 0.0)))
            z = att / se if se > 0 else float("nan")
            p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else float("nan")
            return att, se, p

        event_times = sorted({e for (_, e) in dummy_map})
        rows = [{"event_time": -1, "att": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                 "p": float("nan"), "n_cohorts": 0}]
        post_vecs = []
        for e in event_times:
            vec, present = _agg_vector(e)
            if vec is None:
                continue
            att, se, p = _att_se(vec)
            rows.append({"event_time": e, "att": att, "se": se, "ci_low": att - 1.96 * se,
                         "ci_high": att + 1.96 * se, "p": p, "n_cohorts": len(present)})
            if e >= 0:
                post_vecs.append(vec.values)
        es = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
        es.to_csv(d / "staggered_did.csv", index=False, encoding="utf-8")
        files.append("staggered_did.csv")

        # overall post ATT = simple average of post-period IW event-study ATTs (linear combo -> delta-method SE)
        if post_vecs:
            lpost = pd.Series(sum(post_vecs) / len(post_vecs), index=params.index)
            att_overall, se_overall, p_overall = _att_se(lpost)
        else:
            att_overall = se_overall = p_overall = float("nan")

        leads = es[es["event_time"] < -1]
        pretrend_bad = bool((leads["p"] < 0.05).any()) if len(leads) else False
        n_never = int(sub[sub["_cohort"].isna()][unit].nunique())
        estimates.update({
            "att_overall": round(att_overall, 4), "att_overall_se": round(se_overall, 4),
            "att_overall_p": round(p_overall, 4) if p_overall == p_overall else float("nan"),
            "n_cohorts": float(len(cohorts)), "n_treated_units": float(len(onset)),
            "n_never_treated": float(n_never), "pretrend_violation": 1.0 if pretrend_bad else 0.0,
        })

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(-0.5, color="red", ls="--", lw=0.8)
            ax.errorbar(es["event_time"], es["att"],
                        yerr=[es["att"] - es["ci_low"], es["ci_high"] - es["att"]],
                        fmt="o", capsize=3, color="#55A868")
            ax.set_xlabel("event time (relative to onset; ref = -1)")
            ax.set_ylabel(f"IW ATT on {outcome}")
            ax.set_title("Staggered DiD — Sun-Abraham interaction-weighted ATT")
            fig.tight_layout()
            fig.savefig(d / "staggered_did.png", dpi=150)
            plt.close(fig)
            files.append("staggered_did.png")
        except Exception:
            pass

        pt = "⚠ 检出预趋势(平行趋势存疑)" if pretrend_bad else "前置期 ATT 未见显著(支持平行趋势)"
        sig = "显著" if (p_overall == p_overall and p_overall < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成(Sun-Abraham 交互加权)：{len(cohorts)} 个处理队列 / {len(onset)} 个处理单位 / "
            f"{n_never} 个从未处理对照；总体处理后 ATT = {att_overall:.4f}"
            f"(SE {se_overall:.4f}，{sig}，p={p_overall:.3g})；事件期 IW-ATT 见 staggered_did.png。{pt}。"
            " ⚠ 交错采纳下纯 TWFE 事件研究会受『坏对照(已处理单位被当对照)』负权重污染——本估计先拟合队列特定 "
            "CATT(g,e)、再按队列样本份额加权聚合(Sun & Abraham 2021)，对异质处理效应稳健；识别仍依赖**平行趋势**"
            "(前置期 ATT 应≈0，已检；队列数少时该检验功效有限，无显著≠平行趋势成立)与从未处理对照干净；总体 ATT "
            "为处理后各事件期 IW-ATT 的**简单平均**(非按格元样本量加权，与 Callaway-Sant'Anna simple 聚合略异)；"
            "端点(±窗口)为合并累计期。"
        )
        code += [
            "import statsmodels.formula.api as smf  # 交错DiD(Sun-Abraham 交互加权)",
            '# y ~ Σ 1[cohort=g & (t-g)=e, e≠-1] + C(unit) + C(time); cluster by unit',
            "# ATT(e)=Σ_g (N_g/ΣN)·CATT(g,e); SE via L·cov·Lᵀ (delta method)",
        ]
    except Exception as err:
        summary.append(f"交错DiD失败：{err}")
