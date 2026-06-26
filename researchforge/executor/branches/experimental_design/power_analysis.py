"""Experimental-design family branch handler: power_analysis (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("power_analysis")
def _branch_power_analysis(ctx: Ctx) -> None:
    """DoE advisory: required replications / sample size for a one-way comparison.
    Reports required n per group for CONVENTIONAL effect sizes (small/medium/large
    Cohen's f) at 80% & 90% power — the statistically sound planning output — plus the
    pilot data's observed effect as context (NOT 'observed/post-hoc power', which is a
    deterministic function of the p-value and not a planning tool)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import math

    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    cats = [c.name for c in fp.columns if c.kind in {"categorical", "binary"} and c.name not in _excl]
    cats.sort(key=lambda name: int(df[name].nunique()))  # prefer low-cardinality group

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    group = cfg.get("group") if cfg.get("group") in df.columns else (cats[0] if cats else None)
    if y is None or group is None:
        summary.append(
            "功效/样本量失败：需要 1 个连续结果 + 1 个分组变量。"
            'config={"outcome":..,"group":..,"alpha":0.05} 指定。'
        )
        return
    try:
        alpha = min(0.5, max(1e-4, float(cfg.get("alpha", 0.05))))
    except (TypeError, ValueError):
        alpha = 0.05

    sub = df[[y, group]].dropna()
    k = int(sub[group].nunique())
    if k < 2 or len(sub) <= k:
        summary.append(f"功效/样本量失败：分组数={k}、有效行={len(sub)}，不足以估效应。")
        return

    try:
        from statsmodels.stats.power import FTestAnovaPower

        grp = sub.groupby(group, observed=True)[y]
        means, ns = grp.mean(), grp.count()
        grand = float(sub[y].to_numpy(dtype=float).mean())
        n_total = int(len(sub))
        ss_within = float(((sub[y] - sub[group].map(means)) ** 2).sum())
        sd_within = math.sqrt(ss_within / (n_total - k)) if n_total > k else float("nan")
        sigma_m = math.sqrt(float(((ns / n_total) * (means - grand) ** 2).sum()))
        f_obs = sigma_m / sd_within if sd_within and sd_within > 1e-12 else float("nan")

        analyzer = FTestAnovaPower()
        levels = [("small", 0.10), ("medium", 0.25), ("large", 0.40)]
        rows = []
        for label, f in levels:
            for pw in (0.80, 0.90):
                ntot = float(analyzer.solve_power(effect_size=f, alpha=alpha, power=pw, k_groups=k))
                per = int(math.ceil(ntot / k))
                rows.append({"effect": f"{label} (f={f})", "power": pw,
                             "n_per_group": per, "n_total": per * k})
                estimates[f"n_per_group_{label}_p{int(pw*100)}"] = float(per)
        tbl = pd.DataFrame(rows)
        tbl.to_csv(d / "required_sample_size.csv", index=False, encoding="utf-8")
        files.append("required_sample_size.csv")

        estimates["observed_f"] = float(f_obs) if f_obs == f_obs else float("nan")
        estimates["k_groups"] = float(k)
        estimates["n_current"] = float(n_total)
        estimates["alpha"] = float(alpha)

        # plot: required n/group vs effect size at 80% power
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [f for _, f in levels]
            ys = [int(math.ceil(float(analyzer.solve_power(effect_size=f, alpha=alpha, power=0.8, k_groups=k)) / k))
                  for _, f in levels]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, marker="o", color="#4C72B0")
            ax.set_xlabel("effect size (Cohen's f)")
            ax.set_ylabel("required n per group (80% power)")
            ax.set_title(f"Sample size vs effect — {k} groups, α={alpha}")
            fig.tight_layout()
            fig.savefig(d / "sample_size_curve.png", dpi=150)
            plt.close(fig)
            files.append("sample_size_curve.png")
        except Exception:
            pass

        med80 = estimates.get("n_per_group_medium_p80")
        fobs_txt = f"{f_obs:.3f}" if f_obs == f_obs else "不可估"
        summary.append(
            f"{entry.method} 完成：{k} 组比较（结果 {y}，分组 {group}，α={alpha}）。"
            f"**所需每组样本量**（单因素 ANOVA）：中等效应 f=0.25 → 每组 {int(med80) if med80 else '—'}（80% 功效）。"
            f"小/中/大(f=0.1/0.25/0.4) × 80%/90% 全表见 required_sample_size.csv。"
            f" 试点数据观测效应 f={fobs_txt}（当前每组≈{n_total//k}）。"
            " ⚠ 规划请用**有意义的目标效应**（小/中/大），别用观测效应当目标；"
            "「观测/事后功效」是 p 值的函数、非规划工具,故不作主输出。假定单因素均衡设计、正态/等方差。"
        )
        code += [
            "from statsmodels.stats.power import FTestAnovaPower",
            f"n_total = FTestAnovaPower().solve_power(effect_size=0.25, alpha={alpha}, power=0.8, k_groups={k})",
            f"print('每组样本量(中等效应,80%功效):', -(-int(n_total) // {k}))",
        ]
    except Exception as err:
        summary.append(f"功效/样本量失败：{err}")
