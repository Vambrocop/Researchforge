"""Branch handlers for the nonparametric_tests family — rank-based group tests.

Three classic distribution-free group-comparison tests (the nonparametric cousins
of one-way ANOVA, repeated-measures ANOVA, and the two-sample t-test). DISTINCT
from the `nonparametric` family (permutation/bootstrap/robust) — this one is the
rank-based group-difference suite:

  * kruskal_wallis — Kruskal-Wallis H test: a continuous/ordinal outcome across
    >=3 independent groups, with the eta-squared (η²[H]) effect size + Dunn's post-hoc.
  * friedman_test  — Friedman test: repeated measures across >=3 conditions
    (nonparametric RM-ANOVA), with Kendall's W + Nemenyi post-hoc.
  * mann_whitney   — Mann-Whitney U (Wilcoxon rank-sum): a continuous/ordinal
    outcome between EXACTLY 2 independent groups, with rank-biserial r + the
    Hodges-Lehmann median-shift estimate.

Engine conventions (see CLAUDE.md「引擎约定」): each handler is
``@register("<id>") def _branch_<id>(ctx)``; it unpacks ctx into
df/fp/entry/cfg/d + files/summary/estimates/code and **mutates** them (never
rebinds). Pure Python (scipy/numpy/pandas) — no R. Honest degrade: wrong number
of groups / too few per group / non-numeric outcome / incomplete blocks /
missing import -> append a Chinese "<方法>跳过：<原因>" message and RETURN; never
crash, never fabricate. Products: CSV + PNG (matplotlib Agg, ENGLISH plot
labels), float estimates, Chinese summary ending with ⚠ disclosures.

Column resolution (config overrides, else profiler convention):
  * outcome  = config["outcome"] else first continuous (else first count).
  * group    = config["group"]   else a categorical/binary column (KW: 3..~10
    levels; MWU: exactly 2 levels).
  * Friedman = long via config subject/within/outcome, OR wide (>=3 numeric
    repeated-measure columns auto-pivoted).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Reasonable upper bound on group levels for a categorical "grouping" column so a
# high-cardinality id-like column isn't mistaken for groups.
_MAX_GROUP_LEVELS = 12


# ----------------------------------------------------------------------------- #
# shared column helpers (profiler-convention resolution)                        #
# ----------------------------------------------------------------------------- #
def _numeric_outcome_cols(fp, df) -> list[str]:
    """Continuous-then-count columns (a rank test treats ordinal/count as ranks),
    excluding the panel unit/time id columns."""
    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]
    cnt = [c.name for c in fp.columns if c.kind == "count" and c.name not in excl]
    return cont + cnt


def _group_candidates(fp, df, exclude=()):
    """Categorical/binary columns that could be a grouping factor, lowest
    cardinality first. Excludes unit/time and any caller-excluded columns."""
    excl = {fp.unit_col, fp.time_col} | set(exclude)
    cats = [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary"} and c.name not in excl
    ]
    cats.sort(key=lambda name: int(df[name].nunique()))
    return cats


# ============================================================================= #
# 1. kruskal_wallis — Kruskal-Wallis H + eta-squared η²[H] + Dunn post-hoc      #
# ============================================================================= #
@register("kruskal_wallis")
def _branch_kruskal_wallis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats as sps
    except Exception as err:  # pragma: no cover - scipy is installed
        summary.append(f"Kruskal-Wallis 跳过：缺少依赖（{err}）。")
        return

    # ----- column resolution ------------------------------------------------- #
    cont = _numeric_outcome_cols(fp, df)
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        cont[0] if cont else None
    )
    group_col = cfg.get("group") if cfg.get("group") in df.columns else None
    if group_col is None:
        # prefer a categorical/binary column whose level count is 3..MAX
        for cand in _group_candidates(fp, df, exclude=(outcome,) if outcome else ()):
            k = int(df[cand].dropna().nunique())
            if 3 <= k <= _MAX_GROUP_LEVELS:
                group_col = cand
                break

    if outcome is None or group_col is None:
        summary.append(
            "Kruskal-Wallis 跳过：需要 1 个连续/有序结果变量 + 1 个 ≥3 水平的分组变量"
            "（用 config={\"outcome\":..,\"group\":..} 指定）。"
        )
        return

    sub = df[[group_col, outcome]].dropna()
    yvals = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub[np.isfinite(yvals.to_numpy(dtype=float))]
    if sub.empty:
        summary.append(f"Kruskal-Wallis 跳过：结果变量「{outcome}」非数值或全缺失。")
        return

    # keep levels with >=2 observations (a 1-obs group has no within-group rank info)
    counts = sub[group_col].value_counts()
    levels = [lv for lv in counts.index.tolist() if counts[lv] >= 2]
    dropped_levels = [lv for lv in counts.index.tolist() if counts[lv] < 2]
    # order levels by their first appearance for stable, readable labels
    seen, ordered = set(), []
    for lv in sub[group_col].tolist():
        if lv in levels and lv not in seen:
            seen.add(lv)
            ordered.append(lv)
    levels = ordered
    k = len(levels)
    if k < 3:
        dropped_note = (
            f"（另有 {len(dropped_levels)} 个因样本量<2 被剔除的分组："
            f"{', '.join(str(lv) for lv in dropped_levels)}）"
        ) if dropped_levels else ""
        summary.append(
            f"Kruskal-Wallis 跳过：有效分组（每组 ≥2 观测）少于 3 个（现 {k} 个{dropped_note}，"
            "n_groups<3 请改用 mann_whitney）。"
        )
        return

    groups = [pd.to_numeric(sub.loc[sub[group_col] == lv, outcome]).to_numpy(dtype=float)
              for lv in levels]
    n = int(sum(len(g) for g in groups))

    # ----- Kruskal-Wallis H (scipy ties-corrects automatically) -------------- #
    H, p = sps.kruskal(*groups)
    H, p = float(H), float(p)
    dfree = k - 1
    # eta-squared-from-H effect size: η²[H] = (H - k + 1) / (n - k). This is the
    # quantity rstatix::kruskal_effsize() returns as eta2[H] (Cohen) — NOT Tomczak's
    # epsilon-squared (= H/(n-1)), a different formula. Ranges ~0..1; >0.14 ≈ large.
    eta2h = (H - k + 1.0) / (n - k) if n > k else float("nan")
    eta2h = max(0.0, eta2h) if eta2h == eta2h else eta2h

    # ----- Dunn's post-hoc pairwise test ------------------------------------- #
    # Dunn (1964): pool ALL observations, rank once (mid-ranks for ties), and
    # compare mean ranks. For groups i,j:
    #   z_ij = (Rbar_i - Rbar_j) / sqrt( (N(N+1)/12 - tie_corr) * (1/n_i + 1/n_j) )
    # tie correction: subtract  sum(t^3 - t) / (12*(N-1))  from N(N+1)/12 inside
    # the SE (Dunn's tie-adjusted variance). Two-sided p from the normal; then
    # Bonferroni-multiply by the number of pairs m = k*(k-1)/2.
    allvals = np.concatenate(groups)
    ranks = sps.rankdata(allvals)  # average ranks for ties
    N = allvals.size
    # split ranks back per group (groups were concatenated in `levels` order)
    rank_groups, idx = [], 0
    for g in groups:
        rank_groups.append(ranks[idx:idx + len(g)])
        idx += len(g)
    mean_ranks = [float(rg.mean()) for rg in rank_groups]
    ns = [len(g) for g in groups]

    # tie correction term: sum over tie groups of (t^3 - t)
    _, tie_counts = np.unique(allvals, return_counts=True)
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
    # variance constant with tie adjustment (Dunn / Zar):
    #   sigma2_const = N(N+1)/12 - tie_term/(12(N-1))
    sigma2_const = N * (N + 1) / 12.0 - tie_term / (12.0 * (N - 1)) if N > 1 else 0.0

    m_pairs = k * (k - 1) // 2
    rows = []
    n_sig = 0
    for i in range(k):
        for j in range(i + 1, k):
            se = (sigma2_const * (1.0 / ns[i] + 1.0 / ns[j])) ** 0.5
            z = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
            p_raw = 2.0 * sps.norm.sf(abs(z))
            p_adj = min(1.0, p_raw * m_pairs)  # Bonferroni
            sig = p_adj < 0.05
            n_sig += int(sig)
            rows.append({
                "group_a": str(levels[i]),
                "group_b": str(levels[j]),
                "mean_rank_a": round(mean_ranks[i], 3),
                "mean_rank_b": round(mean_ranks[j], 3),
                "z": round(float(z), 4),
                "p_raw": round(float(p_raw), 5),
                "p_bonferroni": round(float(p_adj), 5),
                "significant_0.05": bool(sig),
            })

    # ----- estimates --------------------------------------------------------- #
    estimates["h_stat"] = round(H, 4)
    estimates["p_value"] = round(p, 5)
    estimates["eta_squared_h"] = round(float(eta2h), 4) if eta2h == eta2h else float("nan")
    estimates["n_groups"] = float(k)
    estimates["n"] = float(n)
    estimates["n_sig_pairs"] = float(n_sig)
    estimates["df"] = float(dfree)

    # ----- products: per-group table + Dunn pairwise CSV --------------------- #
    medians = [float(np.median(g)) for g in groups]
    grp_tab = pd.DataFrame({
        "group": [str(lv) for lv in levels],
        "n": ns,
        "median": [round(mv, 4) for mv in medians],
        "mean_rank": [round(mr, 3) for mr in mean_ranks],
    })
    grp_tab.to_csv(d / "kruskal_groups.csv", index=False, encoding="utf-8")
    files.append("kruskal_groups.csv")

    dunn_tab = pd.DataFrame(rows)
    dunn_tab.to_csv(d / "kruskal_dunn_posthoc.csv", index=False, encoding="utf-8")
    files.append("kruskal_dunn_posthoc.csv")

    # ----- plot: per-group boxplot ------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, 0.9 * k), 4.5))
        ax.boxplot(groups, tick_labels=[str(lv) for lv in levels], showmeans=True)
        ax.set_xlabel("group")
        ax.set_ylabel(str(outcome))
        ax.set_title(f"Kruskal-Wallis: {outcome} by {group_col} (H={H:.3g}, p={p:.3g})")
        if k > 6:
            for lab in ax.get_xticklabels():
                lab.set_rotation(40)
                lab.set_ha("right")
        fig.tight_layout()
        fig.savefig(d / "kruskal_boxplot.png", dpi=150)
        plt.close(fig)
        files.append("kruskal_boxplot.png")
    except Exception:
        pass

    # ----- summary ----------------------------------------------------------- #
    sig_pairs_str = "，".join(
        f"{r['group_a']}↔{r['group_b']}(p={r['p_bonferroni']:.3g})"
        for r in rows if r["significant_0.05"]
    ) or "无"
    eff_word = (
        "大效应" if eta2h == eta2h and eta2h >= 0.14
        else "中效应" if eta2h == eta2h and eta2h >= 0.06
        else "小/微效应"
    )
    summary.append(
        f"{entry.method} 完成：{outcome} 按 {group_col} 分 {k} 组（n={n}）；"
        f"H={H:.4g}（df={dfree}），p={p:.4g}；η²[H]={eta2h:.3g}（{eff_word}）。"
        f" Dunn 事后（Bonferroni）显著差异组对：{sig_pairs_str}。"
    )
    summary.append(
        "⚠ Kruskal-Wallis 检验的是组间**随机优势/分布位移**（stochastic dominance），"
        "并非单纯的中位数差——只有当各组分布形状相同时，它才退化为中位数检验；"
        "结存在时已做并列校正（scipy + Dunn 方差均含 tie 校正）；Dunn 事后用 Bonferroni 控制族错误率；"
        "结果是关联性的（非因果）。可用 config 指定 outcome/group。"
    )
    if dropped_levels:
        summary.append(
            f"⚠ 已剔除 {len(dropped_levels)} 个样本量<2 的分组（无组内秩信息，无法纳入检验）："
            f"{', '.join(str(lv) for lv in dropped_levels)}。"
        )

    code += [
        "import numpy as np; from scipy import stats",
        f"sub = df[['{group_col}', '{outcome}']].dropna()",
        f"groups = [sub.loc[sub['{group_col}']==lv, '{outcome}'].to_numpy(float) "
        "for lv in sub['" + str(group_col) + "'].unique()]",
        "H, p = stats.kruskal(*groups)  # ties-corrected",
        "eta2h = (H - k + 1) / (n - k)  # eta-squared-from-H effect size (rstatix eta2[H])",
        "# Dunn post-hoc: rank all values once, z=(Rbar_i-Rbar_j)/SE, Bonferroni*pairs",
    ]


# ============================================================================= #
# 2. friedman_test — Friedman Q + Kendall's W + Nemenyi post-hoc                #
# ============================================================================= #
def _friedman_resolve_blocks(ctx: Ctx):
    """Resolve a complete-block matrix (rows = subjects, cols = conditions).

    Long form: config subject/within/outcome (or sensible defaults).
    Wide form: >=3 numeric repeated-measure columns (auto-detected) used directly.
    Returns (matrix DataFrame [subjects x conditions], n_dropped_incomplete,
    layout_str, problem_msg). When problem_msg is not None, caller degrades."""
    import pandas as pd

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df

    subject = cfg.get("subject")
    within = cfg.get("within")
    outcome = cfg.get("outcome")

    # ----- explicit long form (all three given & valid) ---------------------- #
    if (subject in df.columns and within in df.columns and outcome in df.columns
            and len({subject, within, outcome}) == 3):
        long = df[[subject, within, outcome]].copy()
        long[outcome] = pd.to_numeric(long[outcome], errors="coerce")
        long = long.dropna()
        if long.empty:
            return None, 0, "long", "Friedman 跳过：long 数据结果列非数值或全缺失。"
        wide = long.pivot_table(index=subject, columns=within, values=outcome,
                                aggfunc="mean")
        before = wide.shape[0]
        wide = wide.dropna(axis=0, how="any")  # complete blocks only
        return wide, before - wide.shape[0], "long", None

    # ----- wide form: >=3 numeric repeated-measure columns ------------------- #
    excl = {fp.unit_col, fp.time_col}
    num_cols = [
        c.name for c in fp.columns
        if c.kind in {"continuous", "count"} and c.name not in excl
    ]
    cols = cfg.get("conditions") if isinstance(cfg.get("conditions"), list) else None
    if cols:
        cols = [c for c in cols if c in df.columns]
    else:
        cols = num_cols
    if len(cols) < 3:
        return None, 0, "wide", (
            "Friedman 跳过：需要 ≥3 个重复测量条件——"
            "宽表给 ≥3 个数值列，或用 config={\"subject\":..,\"within\":..,\"outcome\":..} 指定长表。"
        )
    wide = df[cols].apply(pd.to_numeric, errors="coerce")
    before = wide.shape[0]
    wide = wide.dropna(axis=0, how="any")  # complete blocks only
    return wide, before - wide.shape[0], "wide", None


@register("friedman_test")
def _branch_friedman_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats as sps
    except Exception as err:  # pragma: no cover
        summary.append(f"Friedman 跳过：缺少依赖（{err}）。")
        return

    wide, n_dropped, layout, problem = _friedman_resolve_blocks(ctx)
    if problem is not None:
        summary.append(problem)
        return

    conditions = [str(c) for c in wide.columns.tolist()]
    kc = len(conditions)
    n_subj = int(wide.shape[0])
    if kc < 3:
        summary.append(f"Friedman 跳过：有效条件少于 3 个（现 {kc} 个）。")
        return
    if n_subj < 2:
        summary.append(
            f"Friedman 跳过：完整区块（无缺失条件的受试）少于 2 个（现 {n_subj} 个）。"
        )
        return

    mat = wide.to_numpy(dtype=float)  # rows = subjects, cols = conditions
    samples = [mat[:, j] for j in range(kc)]

    # ----- Friedman chi-square (scipy) --------------------------------------- #
    Q, p = sps.friedmanchisquare(*samples)
    Q, p = float(Q), float(p)
    dfree = kc - 1
    # Kendall's W effect size: W = chi2 / (n*(k-1)). Ranges 0..1 (0=no agreement,
    # 1=perfect agreement / strongest condition effect).
    W = Q / (n_subj * (kc - 1)) if n_subj > 0 and kc > 1 else float("nan")
    W = min(1.0, max(0.0, W)) if W == W else W

    # ----- per-condition mean rank (within-subject ranks across conditions) -- #
    # Rank each subject's k condition values 1..k (average ranks for ties), then
    # average down columns -> the per-condition mean rank Friedman operates on.
    within_ranks = np.vstack([sps.rankdata(row) for row in mat])  # n_subj x kc
    mean_ranks = within_ranks.mean(axis=0)  # length kc

    # ----- Nemenyi post-hoc -------------------------------------------------- #
    # Nemenyi (1963): two conditions differ if |Rbar_i - Rbar_j| >= CD, where the
    # critical difference CD = q_alpha * sqrt( k(k+1) / (12 n) ) and q_alpha is the
    # studentized-range critical value (k means, df=inf). The matching p-value comes
    # from the studentized-range distribution evaluated at the standardized mean-rank
    # gap with the SAME scale:  stat = |Rbar_i - Rbar_j| / sqrt( k(k+1)/(12 n) ),
    # p = P(Q_range > stat | k, df=inf). At the boundary diff==CD, stat==q_alpha and
    # p==0.05, so the CD flag and the p-value are self-consistent. Conservative by
    # design (Nemenyi controls the family-wise error over all pairs jointly).
    se_rank = (kc * (kc + 1) / (12.0 * n_subj)) ** 0.5
    cd = float(sps.studentized_range.ppf(0.95, kc, np.inf) * se_rank)
    rows = []
    n_sig = 0
    for i in range(kc):
        for j in range(i + 1, kc):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            stat = diff / se_rank if se_rank > 0 else 0.0
            p_adj = float(sps.studentized_range.sf(stat, kc, np.inf))
            p_adj = min(1.0, max(0.0, p_adj))
            sig = diff >= cd
            n_sig += int(sig)
            rows.append({
                "condition_a": conditions[i],
                "condition_b": conditions[j],
                "mean_rank_a": round(float(mean_ranks[i]), 3),
                "mean_rank_b": round(float(mean_ranks[j]), 3),
                "rank_diff": round(float(diff), 3),
                "critical_diff": round(cd, 3),
                "p_nemenyi": round(p_adj, 5),
                "significant_0.05": bool(sig),
            })

    # ----- estimates --------------------------------------------------------- #
    estimates["chi2_stat"] = round(Q, 4)
    estimates["p_value"] = round(p, 5)
    estimates["kendalls_w"] = round(float(W), 4) if W == W else float("nan")
    estimates["n_subjects"] = float(n_subj)
    estimates["n_conditions"] = float(kc)
    estimates["n_sig_pairs"] = float(n_sig)
    estimates["df"] = float(dfree)

    # ----- products: per-condition table + Nemenyi pairwise ------------------ #
    cond_tab = pd.DataFrame({
        "condition": conditions,
        "mean_rank": [round(float(r), 3) for r in mean_ranks],
        "median": [round(float(np.median(samples[j])), 4) for j in range(kc)],
        "n": [n_subj] * kc,
    })
    cond_tab.to_csv(d / "friedman_conditions.csv", index=False, encoding="utf-8")
    files.append("friedman_conditions.csv")

    nem_tab = pd.DataFrame(rows)
    nem_tab.to_csv(d / "friedman_nemenyi_posthoc.csv", index=False, encoding="utf-8")
    files.append("friedman_nemenyi_posthoc.csv")

    # ----- plot: per-condition mean rank ------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, 0.9 * kc), 4.5))
        ax.bar(range(kc), [float(r) for r in mean_ranks], color="#6c8ebf",
               edgecolor="white")
        ax.set_xticks(range(kc))
        ax.set_xticklabels(conditions, rotation=40 if kc > 5 else 0,
                           ha="right" if kc > 5 else "center")
        ax.set_xlabel("condition")
        ax.set_ylabel("mean rank")
        ax.set_title(f"Friedman: mean rank by condition (Q={Q:.3g}, p={p:.3g})")
        fig.tight_layout()
        fig.savefig(d / "friedman_mean_ranks.png", dpi=150)
        plt.close(fig)
        files.append("friedman_mean_ranks.png")
    except Exception:
        pass

    # ----- summary ----------------------------------------------------------- #
    sig_pairs_str = "，".join(
        f"{r['condition_a']}↔{r['condition_b']}(p={r['p_nemenyi']:.3g})"
        for r in rows if r["significant_0.05"]
    ) or "无"
    drop_note = f"（已剔除 {n_dropped} 个含缺失条件的受试以保完整区块）" if n_dropped else ""
    eff_word = (
        "强一致/大效应" if W == W and W >= 0.5
        else "中等" if W == W and W >= 0.3
        else "弱/微"
    )
    summary.append(
        f"{entry.method} 完成（{layout} 输入）：{n_subj} 个受试 × {kc} 个条件{drop_note}；"
        f"χ²={Q:.4g}（df={dfree}），p={p:.4g}；Kendall's W={W:.3g}（{eff_word}）。"
        f" Nemenyi 事后显著差异条件对：{sig_pairs_str}。"
    )
    summary.append(
        "⚠ Friedman 需要**完整区块**——任一条件缺失的受试已被剔除（数量见上，已披露）；"
        "它检验的是整体条件差异（重复测量的非参方差分析）；"
        "Nemenyi 事后较为保守（基于 studentized-range 的临界差 CD）；"
        "结果是关联性的。可用 config 指定 subject/within/outcome（长表）或 conditions（宽表列）。"
    )

    code += [
        "import numpy as np; from scipy import stats",
        "# wide: rows=subjects, cols=conditions (drop rows with any missing condition)",
        "mat = wide.dropna().to_numpy(float); n, k = mat.shape",
        "Q, p = stats.friedmanchisquare(*[mat[:,j] for j in range(k)])",
        "W = Q / (n*(k-1))  # Kendall's W effect size",
        "ranks = np.vstack([stats.rankdata(r) for r in mat]); mean_ranks = ranks.mean(0)",
        "CD = stats.studentized_range.ppf(0.95, k, np.inf) * np.sqrt(k*(k+1)/(12*n))",
        "# Nemenyi: conditions differ if |mean_rank_i - mean_rank_j| >= CD",
    ]


# ============================================================================= #
# 3. mann_whitney — Mann-Whitney U + rank-biserial r + Hodges-Lehmann shift     #
# ============================================================================= #
@register("mann_whitney")
def _branch_mann_whitney(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats as sps
    except Exception as err:  # pragma: no cover
        summary.append(f"Mann-Whitney 跳过：缺少依赖（{err}）。")
        return

    # ----- column resolution ------------------------------------------------- #
    cont = _numeric_outcome_cols(fp, df)
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        cont[0] if cont else None
    )
    group_col = cfg.get("group") if cfg.get("group") in df.columns else None
    if group_col is None:
        for cand in _group_candidates(fp, df, exclude=(outcome,) if outcome else ()):
            if int(df[cand].dropna().nunique()) == 2:
                group_col = cand
                break

    if outcome is None or group_col is None:
        summary.append(
            "Mann-Whitney 跳过：需要 1 个连续/有序结果变量 + 1 个恰好 2 水平的分组变量"
            "（用 config={\"outcome\":..,\"group\":..} 指定）。"
        )
        return

    sub = df[[group_col, outcome]].dropna()
    yvals = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub[np.isfinite(yvals.to_numpy(dtype=float))]
    if sub.empty:
        summary.append(f"Mann-Whitney 跳过：结果变量「{outcome}」非数值或全缺失。")
        return

    counts = sub[group_col].value_counts()
    levels = [lv for lv in counts.index.tolist() if counts[lv] >= 1]
    # stable order by first appearance
    seen, ordered = set(), []
    for lv in sub[group_col].tolist():
        if lv in levels and lv not in seen:
            seen.add(lv)
            ordered.append(lv)
    levels = ordered
    if len(levels) != 2:
        summary.append(
            f"Mann-Whitney 跳过：需要恰好 2 个分组（现 {len(levels)} 个）；"
            "≥3 组请改用 kruskal_wallis。"
        )
        return

    g1 = pd.to_numeric(sub.loc[sub[group_col] == levels[0], outcome]).to_numpy(dtype=float)
    g2 = pd.to_numeric(sub.loc[sub[group_col] == levels[1], outcome]).to_numpy(dtype=float)
    n1, n2 = g1.size, g2.size
    if n1 < 2 or n2 < 2:
        summary.append(
            f"Mann-Whitney 跳过：每组至少需 2 个观测（{levels[0]}:{n1}，{levels[1]}:{n2}）。"
        )
        return

    # ----- Mann-Whitney U (two-sided; ties handled by scipy) ----------------- #
    # scipy returns U1 for the FIRST sample (g1): U1 = R1 - n1(n1+1)/2, so U1 is
    # large when g1 outranks g2. Reported as u_stat.
    U1, p = sps.mannwhitneyu(g1, g2, alternative="two-sided")
    U1, p = float(U1), float(p)
    # rank-biserial correlation (Kerby 2014; the simple-difference / common-language
    # effect size): r = 2*U1/(n1*n2) - 1, equivalently 1 - 2*U2/(n1*n2) where
    # U2 = n1*n2 - U1. r>0 => group1 stochastically dominates (tends higher); r<0 =>
    # group2 higher; |r| in [0,1] is the effect-size magnitude.
    r_rb = 2.0 * U1 / (n1 * n2) - 1.0

    # ----- Hodges-Lehmann median shift (median of cross-group differences) --- #
    # HL = median over all n1*n2 pairwise (g1_i - g2_j). Distribution-free point
    # estimate of the location shift. For tractable sizes also an APPROXIMATE
    # Moses/Lehmann-style CI via order statistics of the pairwise differences —
    # the order-statistic rank K is picked from a NORMAL APPROXIMATION to the
    # Wilcoxon rank-sum distribution, so this CI is asymptotic, not exact.
    hl = float("nan")
    hl_lo = hl_hi = float("nan")
    max_pairs = int(cfg.get("hl_max_pairs", 4_000_000))
    if n1 * n2 <= max_pairs:
        diffs = (g1[:, None] - g2[None, :]).ravel()
        diffs.sort()
        hl = float(np.median(diffs))
        # approximate CI: the K-th and (n1*n2+1-K)-th order statistics of the
        # pairwise diffs, K = U_{alpha/2} from the Wilcoxon rank-sum (normal approx
        # with continuity + tie correction). Lehmann (1975), Hollander & Wolfe.
        try:
            Np = n1 * n2
            z = float(sps.norm.ppf(0.975))
            # tie-corrected SD of the Mann-Whitney U
            allv = np.concatenate([g1, g2])
            _, tcnt = np.unique(allv, return_counts=True)
            Ntot = allv.size
            tie_corr = float(np.sum(tcnt ** 3 - tcnt))
            var_u = (n1 * n2 / 12.0) * (
                (Ntot + 1) - tie_corr / (Ntot * (Ntot - 1))
            ) if Ntot > 1 else 0.0
            sd_u = var_u ** 0.5
            if sd_u > 0:
                K = int(np.floor(Np / 2.0 - z * sd_u))
                if 0 <= K < Np // 2:
                    hl_lo = float(diffs[K])
                    hl_hi = float(diffs[Np - 1 - K])
        except Exception:
            pass

    # ----- estimates --------------------------------------------------------- #
    estimates["u_stat"] = round(U1, 3)
    estimates["p_value"] = round(p, 5)
    estimates["rank_biserial"] = round(float(r_rb), 4)
    estimates["hodges_lehmann"] = round(float(hl), 4) if hl == hl else float("nan")
    estimates["n1"] = float(n1)
    estimates["n2"] = float(n2)
    if hl_lo == hl_lo:
        estimates["hl_ci_low"] = round(float(hl_lo), 4)
        estimates["hl_ci_high"] = round(float(hl_hi), 4)

    # ----- products: per-group table ----------------------------------------- #
    grp_tab = pd.DataFrame({
        "group": [str(levels[0]), str(levels[1])],
        "n": [n1, n2],
        "median": [round(float(np.median(g1)), 4), round(float(np.median(g2)), 4)],
        "mean": [round(float(np.mean(g1)), 4), round(float(np.mean(g2)), 4)],
    })
    grp_tab.to_csv(d / "mannwhitney_groups.csv", index=False, encoding="utf-8")
    files.append("mannwhitney_groups.csv")

    stats_tab = pd.DataFrame({
        "metric": ["U_stat", "p_value", "rank_biserial", "hodges_lehmann",
                   "hl_ci_low_approx", "hl_ci_high_approx"],
        "value": [round(U1, 3), round(p, 5), round(float(r_rb), 4),
                  round(float(hl), 4) if hl == hl else float("nan"),
                  round(float(hl_lo), 4) if hl_lo == hl_lo else float("nan"),
                  round(float(hl_hi), 4) if hl_hi == hl_hi else float("nan")],
    })
    stats_tab.to_csv(d / "mannwhitney_stats.csv", index=False, encoding="utf-8")
    files.append("mannwhitney_stats.csv")

    # ----- plot: two-group boxplot ------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.boxplot([g1, g2], tick_labels=[str(levels[0]), str(levels[1])], showmeans=True)
        # jittered strip overlay
        rng = np.random.default_rng(0)
        for pos, g in ((1, g1), (2, g2)):
            jit = pos + (rng.random(g.size) - 0.5) * 0.15
            ax.scatter(jit, g, s=10, color="#888888", alpha=0.5)
        ax.set_xlabel("group")
        ax.set_ylabel(str(outcome))
        ax.set_title(f"Mann-Whitney U: {outcome} by {group_col} (p={p:.3g})")
        fig.tight_layout()
        fig.savefig(d / "mannwhitney_boxplot.png", dpi=150)
        plt.close(fig)
        files.append("mannwhitney_boxplot.png")
    except Exception:
        pass

    # ----- summary ----------------------------------------------------------- #
    direction = (
        f"{levels[0]} 倾向高于 {levels[1]}" if r_rb > 0
        else f"{levels[1]} 倾向高于 {levels[0]}" if r_rb < 0
        else "两组无方向性差异"
    )
    mag = (
        "大效应" if abs(r_rb) >= 0.5
        else "中效应" if abs(r_rb) >= 0.3
        else "小/微效应"
    )
    ci_str = (
        f"，95% CI（近似）=[{hl_lo:.4g}, {hl_hi:.4g}]" if hl_lo == hl_lo else "（CI 不可得）"
    )
    summary.append(
        f"{entry.method} 完成：{outcome} 在 {group_col} 两组（{levels[0]}:n={n1}，"
        f"{levels[1]}:n={n2}）；U={U1:.4g}，p={p:.4g}；秩二列相关 r={r_rb:+.3g}"
        f"（{mag}，{direction}）；Hodges-Lehmann 中位移={hl:.4g}{ci_str}。"
    )
    summary.append(
        "⚠ Mann-Whitney 检验的是**随机优势**（stochastic dominance）——只有当两组分布形状相同时，"
        "才等价于中位数比较；报告了秩二列相关 r 作为效应量（符号指向方向）；"
        "本法仅用于**恰好 2 组**（≥3 组请用 kruskal_wallis）；结存在时已做并列校正；"
        "Hodges-Lehmann 中位移的 95% CI 为**近似**（正态近似选取阶统计量秩次，非精确分布无 CI）；"
        "结果是关联性的。可用 config 指定 outcome/group。"
    )

    code += [
        "import numpy as np; from scipy import stats",
        f"g1 = df.loc[df['{group_col}']==L0, '{outcome}'].dropna().to_numpy(float)",
        f"g2 = df.loc[df['{group_col}']==L1, '{outcome}'].dropna().to_numpy(float)",
        "U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')  # ties-handled",
        "r_rb = 2*U/(g1.size*g2.size) - 1  # rank-biserial; r>0 => g1 higher",
        "HL = np.median(np.subtract.outer(g1, g2))  # Hodges-Lehmann median shift",
    ]
