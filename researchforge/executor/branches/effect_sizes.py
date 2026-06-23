"""Branch handlers for the effect_sizes family.

Standardized effect sizes for a difference in a CONTINUOUS outcome between TWO
groups — the magnitude companions to a significance test (a p-value tells you
*whether*, an effect size tells you *how much*):

* ``cohens_d``     — Cohen's d (pooled-SD standardized mean difference) + analytic
                     95% CI, magnitude band, common-language effect size (CLES),
                     and Glass's delta (control-SD variant) reported alongside.
* ``hedges_g``     — Hedges' g = d × J small-sample correction (bias-corrected d,
                     preferred for small n) + CI + magnitude.
* ``cliffs_delta`` — Cliff's delta (nonparametric, ordinal-valid dominance) +
                     Romano magnitude band + P(X1>X2); related to Mann-Whitney
                     (δ = 2·AUC − 1).

Engine conventions (see CLAUDE.md「引擎约定」): each handler is
``@register("<id>") def _branch_<id>(ctx)``; it unpacks ctx into
df/fp/entry/cfg/d + files/summary/estimates/code and **mutates** them (never
rebinds). Outcome = first continuous column; group = a binary / lowest-cardinality
2-level column; both overridable via ``config["outcome"]`` / ``config["group"]``.

Honest degradation: ≠2 groups, too few per group, non-numeric outcome, or a
missing import → Chinese "<方法>跳过：<原因>" appended to summary, then RETURN —
never crash, never fabricate. Products (CSV + matplotlib-Agg PNG with ENGLISH
labels) are wrapped in try/except. Pure Python (numpy/scipy/pandas) — no R.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared two-group resolution (mirrors nonparametric._branch_permutation_test:
# outcome = first continuous column; group = lowest-cardinality binary/categorical;
# both config-overridable). Restricts to EXACTLY two non-empty levels, splits the
# outcome into the two numeric samples, and returns
#   (x1, x2, outcome, group_col, lvl1, lvl2, problem)
# where (when problem is None) x1 = sample of the FIRST level, x2 = the second.
# When `problem` is not None the caller appends it to summary and returns.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_two_groups(ctx: Ctx, label: str):
    import numpy as np

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    excl = {fp.unit_col, fp.time_col}
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in excl]
    cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in excl]
    cat_cols.sort(key=lambda name: int(df[name].nunique()))  # lowest-cardinality first
    group_candidates = bin_cols + cat_cols
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]

    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        cont_cols[0] if cont_cols else None
    )
    group_col = cfg.get("group") if cfg.get("group") in df.columns else (
        group_candidates[0] if group_candidates else None
    )

    none = (None, None, None, None, None, None)
    if outcome is None:
        return (*none, f"{label}跳过：未找到连续型结果变量（可用 config={{\"outcome\":\"<列>\"}} 指定）。")
    if group_col is None:
        return (*none, f"{label}跳过：未找到二分/分组变量（可用 config={{\"group\":\"<列>\"}} 指定）。")

    sub = df[[group_col, outcome]].dropna()
    yv = sub[outcome].to_numpy(dtype=float)
    sub = sub[np.isfinite(yv)]
    if sub.empty:
        return (*none, f"{label}跳过：结果变量 {outcome} 无有效数值观测。")

    levels = [lv for lv in sub[group_col].unique().tolist()
              if int((sub[group_col] == lv).sum()) > 0]
    if len(levels) != 2:
        return (*none,
                f"{label}跳过：分组变量 {group_col} 必须恰好 2 个水平（实际 {len(levels)} 个）；"
                f"可用 config={{\"group\":\"<二分列>\"}} 指定，或先把变量二分化。")

    lvl1, lvl2 = levels[0], levels[1]
    x1 = sub.loc[sub[group_col] == lvl1, outcome].to_numpy(dtype=float)
    x2 = sub.loc[sub[group_col] == lvl2, outcome].to_numpy(dtype=float)
    if x1.size < 2 or x2.size < 2:
        return (*none,
                f"{label}跳过：每组至少需 2 个观测（{lvl1}={x1.size}，{lvl2}={x2.size}）。")
    return x1, x2, outcome, group_col, lvl1, lvl2, None


def _pooled_sd(x1, x2):
    """Pooled standard deviation (the denominator of Cohen's d).
    sp = sqrt( ((n1-1) s1^2 + (n2-1) s2^2) / (n1+n2-2) ), with sample (ddof=1) variances."""
    import numpy as np

    n1, n2 = x1.size, x2.size
    v1 = float(np.var(x1, ddof=1))
    v2 = float(np.var(x2, ddof=1))
    return float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)))


def _cles(x1, x2):
    """Common-language effect size = P(X1 > X2) for a random pair (McGraw & Wong 1992),
    estimated nonparametrically from all n1·n2 pairs (ties count 0.5). This is the
    Mann-Whitney AUC, so it stays valid without the normality assumption."""
    import numpy as np

    n1, n2 = x1.size, x2.size
    # vectorised pairwise comparison; for the sizes we see this is fine.
    diff = x1[:, None] - x2[None, :]
    gt = float(np.count_nonzero(diff > 0))
    eq = float(np.count_nonzero(diff == 0))
    return (gt + 0.5 * eq) / (n1 * n2)


def _two_group_violin(d, x1, x2, lvl1, lvl2, outcome, title, annot):
    """Overlaid distributions of the two groups (violin + jittered points) with an
    effect-size annotation. matplotlib Agg, ENGLISH labels. Best-effort."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        parts = ax.violinplot([x1, x2], positions=[1, 2], showmeans=True, showextrema=False)
        for pc, col in zip(parts["bodies"], ("#4C72B0", "#C44E52")):
            pc.set_facecolor(col)
            pc.set_alpha(0.5)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
        rng = np.random.default_rng(0)  # fixed seed for jitter (cosmetic, disclosed)
        for pos, xs, col in ((1, x1, "#4C72B0"), (2, x2, "#C44E52")):
            jit = pos + rng.uniform(-0.08, 0.08, size=xs.size)
            ax.scatter(jit, xs, s=10, color=col, alpha=0.5, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"{lvl1} (n={x1.size})", f"{lvl2} (n={x2.size})"])
        ax.set_ylabel(str(outcome))
        ax.set_title(title)
        ax.text(0.98, 0.02, annot, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
        fig.tight_layout()
        fig.savefig(d / "group_distributions.png", dpi=150)
        plt.close(fig)
        return "group_distributions.png"
    except Exception:
        return None


def _band_smd(absd: float) -> str:
    """Cohen's small/medium/large bands for a standardized mean difference."""
    if absd < 0.2:
        return "极小/可忽略 (negligible, <0.2)"
    if absd < 0.5:
        return "小 (small, 0.2–0.5)"
    if absd < 0.8:
        return "中 (medium, 0.5–0.8)"
    return "大 (large, ≥0.8)"


# ─────────────────────────────────────────────────────────────────────────────
# 1. cohens_d — pooled-SD standardized mean difference + CI + CLES + Glass's delta
# ─────────────────────────────────────────────────────────────────────────────
@register("cohens_d")
def _branch_cohens_d(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    x1, x2, outcome, group_col, lvl1, lvl2, problem = _resolve_two_groups(ctx, "Cohen's d")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import pandas as pd

        n1, n2 = x1.size, x2.size
        m1, m2 = float(np.mean(x1)), float(np.mean(x2))
        sp = _pooled_sd(x1, x2)
        if sp == 0.0:
            summary.append("Cohen's d 跳过：合并标准差为 0（两组内部均为常数），效应量无定义。")
            return

        # Cohen's d = (mean1 - mean2) / pooled_sd. Sign: positive => group `lvl1` higher.
        d_val = (m1 - m2) / sp

        # Analytic 95% CI for d (Hedges & Olkin 1985 large-sample variance):
        #   var(d) = (n1+n2)/(n1 n2) + d^2 / (2 (n1+n2))   ; CI = d ± z_{.975} sqrt(var).
        from scipy import stats as _sps

        ci_level = float(cfg.get("ci", 0.95))
        if not (0.0 < ci_level < 1.0):
            ci_level = 0.95
        z = float(_sps.norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
        se_d = float(np.sqrt((n1 + n2) / (n1 * n2) + d_val ** 2 / (2.0 * (n1 + n2))))
        ci_low = d_val - z * se_d
        ci_high = d_val + z * se_d

        # Common-language effect size P(X1 > X2) (Mann-Whitney AUC; normality-free).
        cles = _cles(x1, x2)

        # Glass's delta: standardize by the CONTROL-group SD only (use lvl2 as the
        # reference/control). Preferred when group variances differ (then the pooled
        # SD mixes a heterogeneous scale).
        sd_ctrl = float(np.std(x2, ddof=1))
        glass = (m1 - m2) / sd_ctrl if sd_ctrl > 0 else float("nan")

        band = _band_smd(abs(d_val))

        # ---- products: effect-size table ------------------------------------- #
        tab = pd.DataFrame({
            "metric": ["cohens_d", "ci_low", "ci_high", "cles_P(X1>X2)",
                       "glass_delta", "mean_g1", "mean_g2", "pooled_sd", "n1", "n2"],
            "value": [d_val, ci_low, ci_high, cles, glass, m1, m2, sp, float(n1), float(n2)],
        })
        tab["value"] = tab["value"].round(5)
        tab.to_csv(d / "cohens_d.csv", index=False, encoding="utf-8")
        files.append("cohens_d.csv")

        annot = f"Cohen's d = {d_val:.2f}\n95% CI [{ci_low:.2f}, {ci_high:.2f}]\nCLES = {cles:.2f}"
        png = _two_group_violin(
            d, x1, x2, lvl1, lvl2, outcome,
            f"Cohen's d: {outcome} by {group_col}", annot,
        )
        if png:
            files.append(png)

        estimates["cohens_d"] = round(float(d_val), 5)
        estimates["ci_low"] = round(float(ci_low), 5)
        estimates["ci_high"] = round(float(ci_high), 5)
        estimates["cles"] = round(float(cles), 5)
        estimates["glass_delta"] = round(float(glass), 5) if glass == glass else float("nan")
        estimates["n1"] = float(n1)
        estimates["n2"] = float(n2)

        direction = (f"{lvl1} 高于 {lvl2}" if d_val > 0 else
                     (f"{lvl2} 高于 {lvl1}" if d_val < 0 else "两组持平"))
        summary.append(
            f"{entry.method} 完成：{outcome} 按 {group_col}（{lvl1} vs {lvl2}）的标准化均值差 "
            f"Cohen's d={d_val:.3f}（{band}），95% CI [{ci_low:.3f}, {ci_high:.3f}]；"
            f"{direction}。共同语言效应量 CLES=P({lvl1}>{lvl2})={cles:.3f}（随机各取一观测，{lvl1} 更大的概率）；"
            f"Glass's delta（以 {lvl2} 组标准差为基准）={glass:.3f}。组均值 {lvl1}={m1:.4g}、{lvl2}={m2:.4g}，"
            f"合并标准差={sp:.4g}（n1={n1}, n2={n2}）。"
        )
        summary.append(
            "⚠ 假定：Cohen's d 的解析 95% CI 假定两组方差大致相等且近正态；"
            "若方差明显不等，改看同时报告的 Glass's delta（以对照组标准差为基准）；"
            "若明显非正态/为序数数据，改用 cliffs_delta（分布无关）。小样本时 d 会高估 |效应|，可用 hedges_g。"
            f"小/中/大 阈值 0.2/0.5/0.8（Cohen 1988）。config 可指定 outcome/group（置信水平默认 0.95，本次 {ci_level}）。"
        )

        code += [
            "import numpy as np; from scipy import stats",
            f"sub = df[['{group_col}', '{outcome}']].dropna()",
            f"x1 = sub.loc[sub['{group_col}']=={lvl1!r}, '{outcome}'].to_numpy(float)",
            f"x2 = sub.loc[sub['{group_col}']=={lvl2!r}, '{outcome}'].to_numpy(float)",
            "n1, n2 = x1.size, x2.size",
            "sp = np.sqrt(((n1-1)*x1.var(ddof=1) + (n2-1)*x2.var(ddof=1)) / (n1+n2-2))  # pooled SD",
            "d = (x1.mean() - x2.mean()) / sp  # Cohen's d",
            "se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))  # Hedges-Olkin variance",
            "ci = d + np.array([-1, 1]) * stats.norm.ppf(0.975) * se  # analytic 95% CI",
            "glass = (x1.mean() - x2.mean()) / x2.std(ddof=1)  # control-SD variant",
        ]
    except Exception as err:
        summary.append(f"Cohen's d 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. hedges_g — small-sample bias-corrected SMD (d × J) + CI + magnitude
# ─────────────────────────────────────────────────────────────────────────────
@register("hedges_g")
def _branch_hedges_g(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    x1, x2, outcome, group_col, lvl1, lvl2, problem = _resolve_two_groups(ctx, "Hedges' g")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import pandas as pd
        from scipy import stats as _sps

        n1, n2 = x1.size, x2.size
        m1, m2 = float(np.mean(x1)), float(np.mean(x2))
        sp = _pooled_sd(x1, x2)
        if sp == 0.0:
            summary.append("Hedges' g 跳过：合并标准差为 0（两组内部均为常数），效应量无定义。")
            return

        d_val = (m1 - m2) / sp  # Cohen's d (uncorrected)
        # Hedges' small-sample correction J = 1 - 3/(4(n1+n2)-9) (Hedges 1981).
        # J<1 shrinks |d| toward 0, removing d's upward small-sample bias.
        N = n1 + n2
        j = 1.0 - 3.0 / (4.0 * N - 9.0)
        g_val = j * d_val

        # CI: scale the analytic d-CI by J (var(g) = J^2 var(d)); same equal-variance /
        # normality caveat as Cohen's d.
        ci_level = float(cfg.get("ci", 0.95))
        if not (0.0 < ci_level < 1.0):
            ci_level = 0.95
        z = float(_sps.norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
        se_d = float(np.sqrt(N / (n1 * n2) + d_val ** 2 / (2.0 * N)))
        se_g = j * se_d
        ci_low = g_val - z * se_g
        ci_high = g_val + z * se_g

        band = _band_smd(abs(g_val))

        tab = pd.DataFrame({
            "metric": ["hedges_g", "cohens_d", "ci_low", "ci_high", "correction_j",
                       "mean_g1", "mean_g2", "pooled_sd", "n1", "n2"],
            "value": [g_val, d_val, ci_low, ci_high, j, m1, m2, sp, float(n1), float(n2)],
        })
        tab["value"] = tab["value"].round(6)
        tab.to_csv(d / "hedges_g.csv", index=False, encoding="utf-8")
        files.append("hedges_g.csv")

        annot = f"Hedges' g = {g_val:.2f}\n(d = {d_val:.2f}, J = {j:.3f})\n95% CI [{ci_low:.2f}, {ci_high:.2f}]"
        png = _two_group_violin(
            d, x1, x2, lvl1, lvl2, outcome,
            f"Hedges' g: {outcome} by {group_col}", annot,
        )
        if png:
            files.append(png)

        estimates["hedges_g"] = round(float(g_val), 6)
        estimates["cohens_d"] = round(float(d_val), 6)
        estimates["ci_low"] = round(float(ci_low), 6)
        estimates["ci_high"] = round(float(ci_high), 6)
        estimates["correction_j"] = round(float(j), 6)
        estimates["n1"] = float(n1)
        estimates["n2"] = float(n2)

        direction = (f"{lvl1} 高于 {lvl2}" if g_val > 0 else
                     (f"{lvl2} 高于 {lvl1}" if g_val < 0 else "两组持平"))
        summary.append(
            f"{entry.method} 完成：{outcome} 按 {group_col}（{lvl1} vs {lvl2}）的偏差校正标准化均值差 "
            f"Hedges' g={g_val:.3f}（{band}），95% CI [{ci_low:.3f}, {ci_high:.3f}]；{direction}。"
            f"g = Cohen's d × J 小样本校正：d={d_val:.3f}，J={j:.4f}（n1={n1}, n2={n2}, N={N}）。"
        )
        summary.append(
            "⚠ Hedges' g 是小样本下偏差更小的标准化均值差：Cohen's d 在 n 小时会高估 |效应|，"
            "J=1−3/(4N−9) 把它收缩回去（N 大时 J→1、g≈d）；n 小时优先报告 g。"
            "CI 仍沿用与 d 相同的方差大致相等 + 近正态假定；非正态/序数请改用 cliffs_delta。"
            "小/中/大 阈值 0.2/0.5/0.8。config 可指定 outcome/group。"
        )

        code += [
            "import numpy as np; from scipy import stats",
            f"sub = df[['{group_col}', '{outcome}']].dropna()",
            f"x1 = sub.loc[sub['{group_col}']=={lvl1!r}, '{outcome}'].to_numpy(float)",
            f"x2 = sub.loc[sub['{group_col}']=={lvl2!r}, '{outcome}'].to_numpy(float)",
            "n1, n2 = x1.size, x2.size; N = n1 + n2",
            "sp = np.sqrt(((n1-1)*x1.var(ddof=1) + (n2-1)*x2.var(ddof=1)) / (N-2))",
            "d = (x1.mean() - x2.mean()) / sp",
            "J = 1 - 3/(4*N - 9)            # Hedges small-sample correction",
            "g = J * d                       # bias-corrected SMD",
            "se = J * np.sqrt(N/(n1*n2) + d**2/(2*N))",
            "ci = g + np.array([-1, 1]) * stats.norm.ppf(0.975) * se",
        ]
    except Exception as err:
        summary.append(f"Hedges' g 失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. cliffs_delta — nonparametric ordinal-valid dominance effect size
# ─────────────────────────────────────────────────────────────────────────────
def _band_cliff(absd: float) -> tuple[str, float]:
    """Romano et al. (2006) magnitude bands for |Cliff's delta|; returns
    (label, ordinal 0..3) for negligible/small/medium/large."""
    if absd < 0.147:
        return "极小/可忽略 (negligible, |δ|<0.147)", 0.0
    if absd < 0.33:
        return "小 (small, |δ|<0.33)", 1.0
    if absd < 0.474:
        return "中 (medium, |δ|<0.474)", 2.0
    return "大 (large, |δ|≥0.474)", 3.0


@register("cliffs_delta")
def _branch_cliffs_delta(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    x1, x2, outcome, group_col, lvl1, lvl2, problem = _resolve_two_groups(ctx, "Cliff's delta")
    if problem is not None:
        summary.append(problem)
        return
    try:
        import pandas as pd

        n1, n2 = x1.size, x2.size
        # Cliff's delta = ( #(x1>x2) - #(x1<x2) ) / (n1*n2)  over all pairs, ∈ [-1, 1].
        # Sign: positive => group `lvl1` tends to dominate (be larger than) lvl2.
        diff = x1[:, None] - x2[None, :]
        n_gt = float(np.count_nonzero(diff > 0))
        n_lt = float(np.count_nonzero(diff < 0))
        total = float(n1 * n2)
        delta = (n_gt - n_lt) / total
        # P(X1 > X2) with ties at 0.5 (= Mann-Whitney AUC). Identity check: δ = 2·AUC − 1.
        n_eq = total - n_gt - n_lt
        p_x1_gt = (n_gt + 0.5 * n_eq) / total

        band_label, band_ord = _band_cliff(abs(delta))

        tab = pd.DataFrame({
            "metric": ["cliffs_delta", "p_x1_gt_x2", "magnitude_ordinal_0to3",
                       "n_x1_gt_x2", "n_x1_lt_x2", "n_ties", "n1", "n2"],
            "value": [delta, p_x1_gt, band_ord, n_gt, n_lt, n_eq, float(n1), float(n2)],
        })
        tab["value"] = tab["value"].round(5)
        tab.to_csv(d / "cliffs_delta.csv", index=False, encoding="utf-8")
        files.append("cliffs_delta.csv")

        _en_band = ["negligible", "small", "medium", "large"][int(band_ord)]
        annot = f"Cliff's d = {delta:.2f}\nP(X1>X2) = {p_x1_gt:.2f}\n{_en_band}"
        png = _two_group_violin(
            d, x1, x2, lvl1, lvl2, outcome,
            f"Cliff's delta: {outcome} by {group_col}", annot,
        )
        if png:
            files.append(png)

        estimates["cliffs_delta"] = round(float(delta), 5)
        estimates["magnitude_negligible_to_large"] = float(band_ord)
        estimates["p_x1_gt_x2"] = round(float(p_x1_gt), 5)
        estimates["n1"] = float(n1)
        estimates["n2"] = float(n2)

        if delta > 0:
            dominance = f"{lvl1} 倾向于大于 {lvl2}（正向优势）"
        elif delta < 0:
            dominance = f"{lvl2} 倾向于大于 {lvl1}（负向优势）"
        else:
            dominance = "两组完全无优势差异（δ=0）"
        summary.append(
            f"{entry.method} 完成：{outcome} 按 {group_col}（{lvl1} vs {lvl2}）的非参数优势效应量 "
            f"Cliff's δ={delta:.3f}（{band_label}）；{dominance}。"
            f"δ = (#({lvl1}>{lvl2}) − #({lvl1}<{lvl2})) / (n1·n2)，∈[−1,1]；"
            f"随机各取一观测、{lvl1} 更大的概率 P(X1>X2)={p_x1_gt:.3f}（与 Mann-Whitney AUC 等价，δ=2·AUC−1）。"
            f"配对计数 大于={int(n_gt)}、小于={int(n_lt)}、平局={int(n_eq)}（n1={n1}, n2={n2}）。"
        )
        summary.append(
            "⚠ Cliff's delta 是分布无关（distribution-free）的优势效应量：当正态假定不成立、"
            "或数据为序数 (ordinal) 时优先使用（不依赖均值/标准差，只看成对优势）；"
            "它与 Mann-Whitney U 检验同源（δ=2·AUC−1）。Romano 阈值 |δ|<.147 可忽略 / <.33 小 / "
            "<.474 中 / 否则大。config 可指定 outcome/group。"
        )

        code += [
            "import numpy as np",
            f"sub = df[['{group_col}', '{outcome}']].dropna()",
            f"x1 = sub.loc[sub['{group_col}']=={lvl1!r}, '{outcome}'].to_numpy(float)",
            f"x2 = sub.loc[sub['{group_col}']=={lvl2!r}, '{outcome}'].to_numpy(float)",
            "diff = x1[:, None] - x2[None, :]            # all n1*n2 pairwise comparisons",
            "n_gt = (diff > 0).sum(); n_lt = (diff < 0).sum()",
            "delta = (n_gt - n_lt) / (x1.size * x2.size)  # Cliff's delta in [-1, 1]",
            "# |delta|: <.147 negligible, <.33 small, <.474 medium, else large (Romano)",
            "# related to Mann-Whitney: delta = 2*AUC - 1, AUC = P(X1 > X2)",
        ]
    except Exception as err:
        summary.append(f"Cliff's delta 失败：{err}")
