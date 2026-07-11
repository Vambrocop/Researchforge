"""Branch handler: group_comparison (statistics family).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _welch_anova(groups: list) -> tuple[float, float, float, float] | None:
    """Welch's F-test for one-way ANOVA with unequal variances (Satterthwaite-
    corrected df; Welch 1951). Groups i=1..k with size n_i, mean m_i, sample
    variance v_i (ddof=1). Returns (F, p, df1, df2), or None if any group has
    zero within-group variance (a constant group makes the weight w_i = n_i/v_i
    infinite — degenerate, so callers should skip honestly instead)."""
    import numpy as np
    from scipy import stats as _stats

    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups], dtype=float)
    variances = np.array([np.var(g, ddof=1) for g in groups], dtype=float)

    if np.any(variances == 0):
        return None

    w = ns / variances
    W = w.sum()
    m_bar = (w * means).sum() / W
    numer = (w * (means - m_bar) ** 2).sum() / (k - 1)
    A = ((1 - w / W) ** 2 / (ns - 1)).sum()
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * A
    F = numer / denom
    df1 = float(k - 1)
    df2 = (k**2 - 1) / (3 * A)
    p = float(_stats.f.sf(F, df1, df2))
    return float(F), p, df1, float(df2)


# Column-name hints for a block/replicate/site role — kept local to this family
# (not imported from executor/branches/experimental_design/_shared.py) to avoid
# cross-family coupling per CLAUDE.md's helper-placement convention. A group_col
# candidate whose name matches one of these is a *design* nuisance factor (block,
# replicate, batch, site) rather than the treatment/grouping factor the analyst
# actually wants compared — see docs/dogfood-findings.md #12.
_GROUP_BLOCK_HINTS = (
    "block", "blk", "rep", "replicate", "replication", "batch", "site", "field", "plot",
    "区组", "重复", "地块", "批次", "小区", "场地",
)


def _looks_block_named(name: str) -> bool:
    lname = name.lower()
    return any(h in lname for h in _GROUP_BLOCK_HINTS)


@register("group_comparison")
def _branch_group_comparison(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    from scipy import stats

    _excl = {fp.unit_col, fp.time_col}
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
    group_candidates = bin_cols + cat_cols
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
    outcome = cont_cols[0] if cont_cols else None

    group_override = cfg.get("group")
    guessed_group = True
    if group_override in df.columns:
        group_col = group_override
        guessed_group = False
    elif group_candidates:
        # Layer 1: candidates whose name does NOT look like a block/replicate/site
        # nuisance factor get first dibs — a block-hint name is demoted so a real
        # treatment/grouping factor is never shadowed by a block that happens to
        # have fewer levels (the old `sort by nunique` picked whichever column had
        # the fewest levels, which is backwards: it favored blocks over treatments).
        # Binary-before-categorical and ascending-nunique remain as tie-breakers
        # within a layer (the original nunique-sort's purpose — keep high-cardinality
        # id-like columns from winning — is preserved, just demoted to tie-break).
        def _rank(name: str) -> tuple:
            return (
                1 if _looks_block_named(name) else 0,
                0 if name in bin_cols else 1,
                int(df[name].nunique()),
            )

        group_col = sorted(group_candidates, key=_rank)[0]
    else:
        group_col = None

    if group_col is None or outcome is None:
        summary.append("组间比较失败：未找到分组变量或连续结果变量。")
    else:
        if guessed_group:
            summary.append(f"⚠ 自动选分组={group_col}（可用 config[\"group\"] 覆盖）")
        # Per-group means/counts
        group_means = df.groupby(group_col)[outcome].agg(["mean", "count", "std"])
        group_means.to_csv(d / "group_means.csv", encoding="utf-8")
        files.append("group_means.csv")

        # Split outcome by group levels, drop NaN
        levels = df[group_col].dropna().unique().tolist()
        groups = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
        n_groups = len(groups)

        # Guard: a group with <2 non-null observations makes var(ddof=1) NaN —
        # skip honestly instead of reporting "统计量=nan, p=nan" as a result.
        _too_small = [str(lv) for lv, g in zip(levels, groups) if len(g) < 2]
        if _too_small:
            summary.append(
                f"组间比较失败：分组变量 {group_col} 下的组 {', '.join(_too_small)} "
                "样本量 <2，组内方差不可估计，已跳过该比较（不报告 NaN 统计量）。"
            )
        else:
            var_note = ""
            welch_degenerate = False
            if n_groups == 2:
                stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = "Welch t-test"
            else:
                # k>=3: Welch's ANOVA is used unconditionally (variance-robust by
                # default — a two-stage "test Levene then pick a test" procedure
                # has poor error control; Delacre, Leys & Lakens 2019). Levene's
                # test is still computed and disclosed, but purely as a diagnostic
                # explaining *why* Welch is the default, not as a gate.
                welch_result = _welch_anova(groups)
                try:
                    lstat, lp = stats.levene(*groups)
                    levene_note = f"Levene 方差齐性检验：统计量={lstat:.4f}，p={lp:.3g}"
                except Exception as err:
                    levene_note = f"方差齐性检验(Levene)未能完成：{err}"

                if welch_result is None:
                    welch_degenerate = True
                else:
                    stat, p, welch_df1, welch_df2 = welch_result
                    test_name = "Welch 稳健单因素方差分析（不假定方差齐性）"
                    estimates["welch_df1"] = welch_df1
                    estimates["welch_df2"] = welch_df2
                    var_note = (
                        "⚠ 已默认用 Welch 稳健单因素方差分析（不假定方差齐性）；"
                        f"{levene_note}"
                        "（<0.05 时方差不齐，正是默认用 Welch 的原因；≥0.05 亦不改变默认）。"
                    )

            if welch_degenerate:
                summary.append(
                    f"组间比较失败：分组变量 {group_col} 下至少一组结果变量 {outcome} 组内方差为 0"
                    "（该组所有观测值相同），Welch 方差分析的组权重不可定义，"
                    "已跳过该比较（不报告 inf/NaN 统计量）。"
                )
            else:
                estimates["statistic"] = float(stat)
                estimates["pvalue"] = float(p)

                # Boxplot
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 4))
                    plot_data = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
                    ax.boxplot(plot_data, tick_labels=[str(lv) for lv in levels])
                    ax.set_xlabel(group_col)
                    ax.set_ylabel(outcome)
                    ax.set_title(f"{outcome} by {group_col}")
                    fig.tight_layout()
                    fig.savefig(d / "boxplot.png", dpi=150)
                    plt.close(fig)
                    files.append("boxplot.png")
                except Exception:
                    pass

                summary.append(
                    f"{entry.method} 完成：{outcome} 按 {group_col} 分 {n_groups} 组，"
                    f"统计量={stat:.4f}，p={p:.3g}"
                    + (f"。{var_note}" if var_note else "")
                )
                # Wave K-E5: 显著时把 stat/p 数字翻译成一句人话结论（哪组最高、比最低组高
                # 多少个百分点），别只甩统计量让用户自己算。仅比较族做，别泛化到别的分支。
                if p < 0.05:
                    _means = group_means["mean"]
                    top_level = _means.idxmax()
                    bottom_level = _means.idxmin()
                    top_val = float(_means[top_level])
                    bottom_val = float(_means[bottom_level])
                    if top_level != bottom_level:
                        if bottom_val != 0:
                            pct = (top_val - bottom_val) / abs(bottom_val) * 100
                            diff_note = f"较最低（{bottom_level} 组，{bottom_val:.4g}）高 ~{pct:.0f}%"
                        else:
                            diff_note = (
                                f"较最低（{bottom_level} 组，{bottom_val:.4g}）高 "
                                f"{top_val - bottom_val:.4g}（基准为 0，无法折算百分比）"
                            )
                        summary.append(
                            f"⚠ {top_level} 组均值最高（{outcome}={top_val:.4g}），"
                            f"{diff_note}，p={p:.3g}，差异显著。"
                        )
                code += [
                    "from scipy import stats",
                    f"groups = [df.loc[df['{group_col}'] == lv, '{outcome}'].dropna().values",
                    f"         for lv in df['{group_col}'].dropna().unique()]",
                    "stat, p = stats.ttest_ind(*groups[:2], equal_var=False)  # or Welch's ANOVA for k>=3",
                    "print(f'statistic={stat:.4f}, p={p:.3g}')",
                ]
