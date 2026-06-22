"""Branch handlers for the categorical (contingency-table) family.

Three classic categorical-data-analysis methods that work on cross-tabulations
built from categorical / binary / ordinal columns (pure Python — numpy / scipy /
statsmodels / pandas; no R):

  * loglinear           — log-linear independence model for an r×c table; deviance
                          (G²) likelihood-ratio test of independence + expected
                          counts + standardized Pearson residuals.
  * cmh_test            — Cochran-Mantel-Haenszel test for a binary exposure ×
                          binary outcome stratified over K strata; Mantel-Haenszel
                          common OR + Breslow-Day test of OR homogeneity.
  * ordinal_association — Goodman-Kruskal γ, Kendall τ-b, Somers' D (both
                          directions) between two ordinal columns.

Each handler resolves its columns (config override else auto-pick from profiler
kinds), degrades honestly (too few cols/rows, sparse, import missing → append a
Chinese "<方法>跳过：<原因>" to summary and RETURN — never crash, never
fabricate), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float
`estimates`, appends a Chinese `summary` ending with ⚠ disclosures, and MUTATES
ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared column resolution. Categorical-ish = categorical / binary / count
# (Likert / coded factors profile as count); also accept "id" for small-cardinality
# integer-coded factors. Excludes the unit/time columns.
# ─────────────────────────────────────────────────────────────────────────────
def _categorical_like_cols(ctx: Ctx) -> list[str]:
    df, fp = ctx.df, ctx.fp
    excl = {fp.unit_col, fp.time_col}
    # A contingency-table factor must be LOW-cardinality: count/id columns can be
    # all-distinct integers (the profiler's "id trap") which are NOT factors. Cap
    # the distinct-value count so a near-continuous integer column is screened out
    # of AUTO-selection (config factors/var1/var2/stratum bypass this filter).
    cap = max(2, min(20, len(df) // 2))
    out = []
    for c in fp.columns:
        if c.name in excl or c.kind not in {"categorical", "binary", "count", "id"}:
            continue
        if 2 <= df[c.name].dropna().nunique() <= cap:
            out.append(c.name)
    return out


def _binary_cols(ctx: Ctx) -> list[str]:
    """Columns with exactly two distinct non-missing values (profiler 'binary',
    or any categorical/count column that happens to be 2-valued in the data)."""
    df, fp = ctx.df, ctx.fp
    excl = {fp.unit_col, fp.time_col}
    out = []
    for c in fp.columns:
        if c.name in excl or c.kind not in {"binary", "categorical", "count", "id"}:
            continue
        if df[c.name].dropna().nunique() == 2:
            out.append(c.name)
    return out


def _sorted_categories(series):
    """Ordered category set: numeric ascending if all numeric, else lexicographic.
    The ordering is what defines the ordinal axis — DISCLOSED to the user."""
    import pandas as pd

    cats = pd.unique(series.dropna())
    try:
        return sorted(cats, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(cats, key=lambda v: str(v))


# ─────────────────────────────────────────────────────────────────────────────
# 1. loglinear — log-linear independence model + G² test of independence
# ─────────────────────────────────────────────────────────────────────────────
@register("loglinear")
def _branch_loglinear(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cands = _categorical_like_cols(ctx)
    factors = cfg.get("factors")
    if isinstance(factors, list):
        factors = [c for c in factors if c in df.columns]
    else:
        factors = []
    if len(factors) < 2:
        rest = [c for c in cands if c not in factors]
        while len(factors) < 2 and rest:
            factors.append(rest.pop(0))
    if len(factors) < 2:
        summary.append(
            "对数线性模型跳过：需要 ≥2 个类别/二值列构成列联表。config 用 factors=[...] 指定。"
        )
        return
    # Use the first two factors for the 2-way table (G² test of independence).
    f1, f2 = factors[0], factors[1]
    if f1 == f2:
        summary.append("对数线性模型跳过：两个因子列相同，无法构成列联表。config 用 factors=[...] 指定。")
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        sub = df[[f1, f2]].dropna()
        if sub.shape[0] < 4:
            summary.append("对数线性模型跳过：成对删除缺失后样本不足 4 行。")
            return

        # Observed cell-count table (rows = f1, cols = f2).
        obs = pd.crosstab(sub[f1], sub[f2])
        r, c = obs.shape
        if r < 2 or c < 2:
            summary.append(
                f"对数线性模型跳过：列联表维度 {r}×{c}，每个因子至少需 2 个类别。"
            )
            return

        O = obs.to_numpy(dtype=float)
        n = float(O.sum())
        row_tot = O.sum(axis=1, keepdims=True)
        col_tot = O.sum(axis=0, keepdims=True)
        # Independence-model expected counts E_ij = row_i * col_j / n (the fitted
        # values of a Poisson GLM with main effects only — for a 2-way table this
        # has the closed form below, identical to the GLM MLE).
        E = row_tot @ col_tot / n

        # Deviance / likelihood-ratio statistic G² = 2 Σ O ln(O/E) (saturated vs
        # independence model). Cells with O=0 contribute 0 (lim x ln x = 0).
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(O > 0, O / E, 1.0)
            g2 = float(2.0 * np.sum(np.where(O > 0, O * np.log(ratio), 0.0)))
        dof = (r - 1) * (c - 1)
        p_g2 = float(stats.chi2.sf(g2, dof))

        # Pearson χ² for contrast (same df).
        pearson_chi2 = float(np.sum((O - E) ** 2 / E))
        p_pearson = float(stats.chi2.sf(pearson_chi2, dof))
        # ADJUSTED (standardized) Pearson residuals: (O-E)/sqrt(E·(1-row/n)·(1-col/n)).
        # These ARE ~N(0,1) under independence, so the |r|>2 cell-flagging rule is
        # valid. The plain (O-E)/sqrt(E) has variance <1 and is NOT N(0,1) (Agresti).
        with np.errstate(divide="ignore", invalid="ignore"):
            std_resid = (O - E) / np.sqrt(E * (1 - row_tot / n) * (1 - col_tot / n))

        # Optionally confirm via statsmodels Poisson GLM (same G² as residual
        # deviance of the main-effects model). Best-effort; never blocks.
        glm_dev = None
        try:
            import statsmodels.api as sm  # noqa: F401
            from statsmodels.genmod.generalized_linear_model import GLM
            from statsmodels.genmod.families import Poisson

            import warnings

            long = obs.stack().reset_index()
            long.columns = ["_f1", "_f2", "count"]
            X = pd.get_dummies(
                long[["_f1", "_f2"]].astype(str), drop_first=True
            ).astype(float)
            X.insert(0, "const", 1.0)
            with warnings.catch_warnings():
                # a perfectly-independent table can trigger PerfectSeparationWarning
                # in this confirmatory GLM; the load-bearing G² is the closed form.
                warnings.simplefilter("ignore")
                mod = GLM(long["count"].astype(float), X, family=Poisson()).fit()
            glm_dev = float(mod.deviance)  # residual deviance == G²
        except Exception:
            glm_dev = None

        # --- products: per-cell observed / expected / std-resid CSV ---
        long_rows = []
        rlabels = list(obs.index)
        clabels = list(obs.columns)
        for i, rl in enumerate(rlabels):
            for j, cl in enumerate(clabels):
                long_rows.append({
                    f1: rl, f2: cl,
                    "observed": round(float(O[i, j]), 4),
                    "expected": round(float(E[i, j]), 4),
                    "std_resid": round(float(std_resid[i, j]), 4),
                })
        cell_df = pd.DataFrame(long_rows)
        cell_df.to_csv(d / "loglinear_cells.csv", index=False, encoding="utf-8")
        files.append("loglinear_cells.csv")

        obs.to_csv(d / "loglinear_observed.csv", encoding="utf-8")
        files.append("loglinear_observed.csv")

        # sparse cells (expected < 5) — flag count (Cochran's rule of thumb).
        n_sparse = int(np.sum(E < 5))
        min_expected = float(E.min())

        # --- PNG: heatmap of standardized Pearson residuals ---
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(5, 0.7 * c + 3), max(4, 0.5 * r + 2)))
            vmax = float(np.nanmax(np.abs(std_resid))) if std_resid.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
            im = ax.imshow(std_resid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(c))
            ax.set_xticklabels([str(x) for x in clabels], rotation=45, ha="right")
            ax.set_yticks(range(r))
            ax.set_yticklabels([str(x) for x in rlabels])
            ax.set_xlabel(str(f2))
            ax.set_ylabel(str(f1))
            ax.set_title("Standardized Pearson residuals (O-E)/sqrt(E)")
            for i in range(r):
                for j in range(c):
                    ax.text(j, i, f"{std_resid[i, j]:.1f}", ha="center", va="center",
                            color="black", fontsize=8)
            fig.colorbar(im, ax=ax, label="std. residual")
            fig.tight_layout()
            fig.savefig(d / "loglinear_residuals.png", dpi=150)
            plt.close(fig)
            files.append("loglinear_residuals.png")
        except Exception:
            pass

        estimates["g2"] = round(g2, 4)
        estimates["df"] = float(dof)
        estimates["p_value"] = round(p_g2, 6)
        estimates["n"] = round(n, 1)
        estimates["pearson_chi2"] = round(pearson_chi2, 4)
        estimates["n_rows_table"] = float(r)
        estimates["n_cols_table"] = float(c)
        estimates["n_sparse_cells"] = float(n_sparse)
        estimates["min_expected"] = round(min_expected, 4)
        if glm_dev is not None:
            estimates["glm_residual_deviance"] = round(glm_dev, 4)

        verdict = (
            f"两变量不独立（G²={round(g2, 3)}, df={dof}, p={p_g2:.4g}<0.05）"
            if p_g2 < 0.05
            else f"不能拒绝独立性（G²={round(g2, 3)}, df={dof}, p={p_g2:.4g}≥0.05）"
        )
        sparse_note = (
            f"⚠ {n_sparse} 个单元格期望频数<5（最小期望 {round(min_expected, 2)}），"
            "渐近卡方近似可能不稳，建议合并稀疏类别或用精确检验。"
            if n_sparse > 0 else ""
        )
        (d / "loglinear_summary.txt").write_text(
            f"对数线性独立模型（{f1} × {f2}）：{r}×{c} 列联表，N={int(n)}\n"
            f"似然比 G² 独立性检验：G²=2ΣO·ln(O/E)={round(g2, 4)}，df=(r-1)(c-1)={dof}，p={p_g2:.4g}\n"
            f"Pearson χ²（对照，同 df）={round(pearson_chi2, 4)}，p={p_pearson:.4g}\n"
            + (f"statsmodels Poisson GLM 残差偏差（主效应模型）={round(glm_dev, 4)}（应≈G²）\n"
               if glm_dev is not None else "")
            + f"结论：{verdict}\n"
            f"稀疏单元格（期望<5）：{n_sparse} 个；最小期望频数 {round(min_expected, 3)}\n"
            "注：G² 与 Pearson χ² 渐近等价（大样本下相近），都检验「行列独立」（饱和模型 vs 主效应模型）；"
            "标准化 Pearson 残差 (O-E)/sqrt(E) 指出偏离独立最大的单元格（|残差|>2 提示该单元格观测显著偏离期望）；"
            "假定 Poisson/多项抽样；列由 config factors 选定。\n\n"
            "观测列联表：\n" + obs.to_string() + "\n\n"
            "标准化 Pearson 残差：\n"
            + pd.DataFrame(np.round(std_resid, 3), index=rlabels, columns=clabels).to_string(),
            encoding="utf-8",
        )
        files.append("loglinear_summary.txt")

        summary.append(
            f"{entry.method} 完成（{f1} × {f2}，{r}×{c} 表，N={int(n)}）："
            f"似然比 G²={round(g2, 3)}（df={dof}, p={p_g2:.4g}），Pearson χ²={round(pearson_chi2, 3)}；{verdict}。"
            + (sparse_note if sparse_note else "")
            + "⚠ G² 是对数线性饱和模型 vs 独立模型的偏差检验；标准化残差 (O-E)/√E 标出偏离独立的单元格（|·|>2 显著）；"
            "假定 Poisson/多项抽样；列由 config factors 选定。"
        )
        code += [
            "import pandas as pd, numpy as np",
            "from scipy import stats",
            f"obs = pd.crosstab(df[{f1!r}], df[{f2!r}])  # r×c contingency table",
            "O = obs.to_numpy(float); n = O.sum()",
            "E = O.sum(1, keepdims=True) @ O.sum(0, keepdims=True) / n  # independence expected",
            "G2 = 2*np.nansum(np.where(O>0, O*np.log(O/E), 0))  # likelihood-ratio (deviance)",
            "df = (obs.shape[0]-1)*(obs.shape[1]-1); p = stats.chi2.sf(G2, df)",
            "std_resid = (O - E)/np.sqrt(E)  # standardized Pearson residuals",
        ]
    except Exception as err:
        summary.append(f"对数线性模型失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. cmh_test — Cochran-Mantel-Haenszel + Mantel-Haenszel OR + Breslow-Day
# ─────────────────────────────────────────────────────────────────────────────
def _bd_expected_a(a_marg, b_marg, c_marg, n_k, psi):
    """Breslow-Day: expected count a_k in cell (1,1) of stratum k under the common
    odds ratio psi. Solve the quadratic for the MLE expectation E[a_k] given the
    fixed margins. a_marg=a, b_marg=b, c_marg=c so R1=a+b (exposed), C1=a+c (positive)."""
    # margins: row1 = a+b, col1 = a+c, n = total. Let x = E[a].
    # OR psi = x(n - row1 - col1 + x) / ((row1 - x)(col1 - x))
    # => (psi-1) x² - [psi(row1+col1) + (n - row1 - col1)] x + psi*row1*col1 = 0
    import math

    R1 = a_marg + b_marg  # exposed total (a+b)
    C1 = a_marg + c_marg  # outcome-positive total (a+c)
    N = n_k
    if abs(psi - 1.0) < 1e-12:
        return R1 * C1 / N if N else 0.0  # OR=1 -> independence expectation
    A = psi - 1.0
    B = -(psi * (R1 + C1) + (N - R1 - C1))
    C = psi * R1 * C1
    disc = B * B - 4 * A * C
    if disc < 0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    x1 = (-B - sqrt_disc) / (2 * A)
    x2 = (-B + sqrt_disc) / (2 * A)
    # pick the root in the feasible range [max(0, R1+C1-N), min(R1, C1)]
    lo = max(0.0, R1 + C1 - N)
    hi = min(R1, C1)
    for x in (x1, x2):
        if lo - 1e-9 <= x <= hi + 1e-9:
            return min(max(x, lo), hi)
    return min(max(x1, lo), hi)


@register("cmh_test")
def _branch_cmh_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    binc = _binary_cols(ctx)
    cats = _categorical_like_cols(ctx)

    exposure = cfg.get("exposure") if cfg.get("exposure") in df.columns else None
    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else None
    stratum = cfg.get("stratum") if cfg.get("stratum") in df.columns else None

    # auto-pick: two binary columns for exposure/outcome, one categorical for strata.
    if exposure is None:
        for c in binc:
            if c not in {outcome, stratum}:
                exposure = c
                break
    if outcome is None:
        for c in binc:
            if c not in {exposure, stratum}:
                outcome = c
                break
    if stratum is None:
        for c in cats:
            if c not in {exposure, outcome} and df[c].dropna().nunique() >= 2:
                stratum = c
                break

    if exposure is None or outcome is None or stratum is None or len({exposure, outcome, stratum}) < 3:
        summary.append(
            "CMH 检验跳过：需要二值暴露 + 二值结局 + 分层变量。"
            "config 用 exposure / outcome / stratum 指定。"
        )
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        sub = df[[exposure, outcome, stratum]].dropna()
        if sub.shape[0] < 4:
            summary.append("CMH 检验跳过：删除缺失后样本不足 4 行。")
            return

        # establish a consistent binary coding: "level 2" (the larger sorted label)
        # is the exposed / outcome-positive cell. Disclose this.
        e_levels = _sorted_categories(sub[exposure])
        o_levels = _sorted_categories(sub[outcome])
        if len(e_levels) != 2 or len(o_levels) != 2:
            summary.append(
                f"CMH 检验跳过：暴露需恰好 2 个水平（得 {len(e_levels)}）、"
                f"结局需恰好 2 个水平（得 {len(o_levels)}）。config 指定二值列。"
            )
            return
        e_pos, e_neg = e_levels[1], e_levels[0]  # exposed = higher label
        o_pos, o_neg = o_levels[1], o_levels[0]  # positive = higher label

        strata = _sorted_categories(sub[stratum])
        rows = []
        sum_a = sum_Ea = sum_Va = 0.0
        num_or = den_or = 0.0  # MH common-OR numerator/denominator
        usable = 0
        for sk in strata:
            t = sub[sub[stratum] == sk]
            a = float(((t[exposure] == e_pos) & (t[outcome] == o_pos)).sum())
            b = float(((t[exposure] == e_pos) & (t[outcome] == o_neg)).sum())
            c = float(((t[exposure] == e_neg) & (t[outcome] == o_pos)).sum())
            dd = float(((t[exposure] == e_neg) & (t[outcome] == o_neg)).sum())
            nk = a + b + c + dd
            if nk < 1:
                continue
            usable += 1
            # MH common OR pieces
            num_or += (a * dd / nk) if nk else 0.0
            den_or += (b * c / nk) if nk else 0.0
            # CMH expectation & variance of a_k under H0 (independence within stratum)
            Ea = (a + b) * (a + c) / nk
            if nk > 1:
                Va = (a + b) * (c + dd) * (a + c) * (b + dd) / (nk * nk * (nk - 1))
            else:
                Va = 0.0
            sum_a += a
            sum_Ea += Ea
            sum_Va += Va
            or_k = (a * dd) / (b * c) if (b * c) > 0 else float("inf")
            rows.append({
                "stratum": sk, "a": a, "b": b, "c": c, "d": dd, "n": nk,
                "OR": round(or_k, 4) if or_k != float("inf") else float("inf"),
                "E_a": round(Ea, 4), "V_a": round(Va, 4),
            })

        if usable < 1 or sum_Va <= 0:
            summary.append("CMH 检验跳过：无可用 2×2 分层（方差为 0 / 全空）。")
            return

        mh_or = (num_or / den_or) if den_or > 0 else float("inf")
        # CMH statistic with continuity correction.
        cmh_chi2 = (abs(sum_a - sum_Ea) - 0.5) ** 2 / sum_Va if sum_Va > 0 else float("nan")
        cmh_chi2 = max(cmh_chi2, 0.0)
        cmh_p = float(stats.chi2.sf(cmh_chi2, 1))

        # Breslow-Day test of OR homogeneity across strata.
        bd_chi2 = 0.0
        bd_ok = mh_or not in (float("inf"),) and mh_or > 0 and mh_or == mh_or
        if bd_ok:
            for row in rows:
                a, b, c, dd, nk = row["a"], row["b"], row["c"], row["d"], row["n"]
                Ea = _bd_expected_a(a, b, c, nk, mh_or)
                # variance of a under the BD fitted table
                Eb = (a + b) - Ea
                Ec = (a + c) - Ea
                Ed = nk - Ea - Eb - Ec
                if min(Ea, Eb, Ec, Ed) <= 0:
                    continue
                var_bd = 1.0 / (1.0 / Ea + 1.0 / Eb + 1.0 / Ec + 1.0 / Ed)
                bd_chi2 += (a - Ea) ** 2 / var_bd if var_bd > 0 else 0.0
            bd_df = max(usable - 1, 1)
            bd_p = float(stats.chi2.sf(bd_chi2, bd_df)) if usable > 1 else float("nan")
        else:
            bd_df = max(usable - 1, 1)
            bd_p = float("nan")

        strat_df = pd.DataFrame(rows)
        strat_df.to_csv(d / "cmh_strata.csv", index=False, encoding="utf-8")
        files.append("cmh_strata.csv")

        # --- PNG: forest plot of per-stratum OR + MH pooled (log scale) ---
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            finite = [r for r in rows if r["OR"] not in (float("inf"),) and r["OR"] > 0]
            labels = [str(r["stratum"]) for r in finite] + ["MH pooled"]
            ors = [r["OR"] for r in finite] + [mh_or if mh_or not in (float("inf"),) else float("nan")]
            ypos = list(range(len(labels)))[::-1]
            fig, ax = plt.subplots(figsize=(6.5, max(3, 0.5 * len(labels) + 1.5)))
            colors = ["#4C72B0"] * len(finite) + ["#C44E52"]
            ax.scatter(ors, ypos, color=colors, zorder=3,
                       s=[60] * len(finite) + [110])
            ax.axvline(1.0, ls="--", color="grey", lw=1)
            if mh_or not in (float("inf"),) and mh_or > 0:
                ax.axvline(mh_or, ls=":", color="#C44E52", lw=1, alpha=0.7)
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels)
            ax.set_xscale("log")
            ax.set_xlabel("Odds ratio (log scale)")
            ax.set_title(f"Per-stratum OR + Mantel-Haenszel pooled (MH OR={mh_or:.2f})")
            fig.tight_layout()
            fig.savefig(d / "cmh_forest.png", dpi=150)
            plt.close(fig)
            files.append("cmh_forest.png")
        except Exception:
            pass

        estimates["mh_or"] = round(float(mh_or), 4) if mh_or not in (float("inf"),) else float("inf")
        estimates["cmh_chi2"] = round(float(cmh_chi2), 4)
        estimates["cmh_p"] = round(float(cmh_p), 6)
        estimates["bd_chi2"] = round(float(bd_chi2), 4)
        estimates["bd_p"] = round(float(bd_p), 6) if bd_p == bd_p else float("nan")
        estimates["n_strata"] = float(usable)
        estimates["n"] = float(sub.shape[0])

        assoc = (
            f"暴露与结局显著相关（CMH χ²={round(cmh_chi2, 3)}, p={cmh_p:.4g}<0.05）"
            if cmh_p < 0.05
            else f"未见显著相关（CMH χ²={round(cmh_chi2, 3)}, p={cmh_p:.4g}≥0.05）"
        )
        homog_note = ""
        if bd_p == bd_p:  # not NaN
            homog_note = (
                f"⚠ Breslow-Day 拒绝 OR 同质性（χ²={round(bd_chi2, 3)}, p={bd_p:.4g}<0.05）——"
                "各层 OR 不一致，合并 OR 可疑，应分层报告。"
                if bd_p < 0.05 else
                f"Breslow-Day 未拒绝 OR 同质性（χ²={round(bd_chi2, 3)}, p={bd_p:.4g}≥0.05）——合并 OR 合理。"
            )
        bd_p_str = f"{bd_p:.4g}" if bd_p == bd_p else "NaN"
        (d / "cmh_summary.txt").write_text(
            f"Cochran-Mantel-Haenszel 检验：暴露 {exposure}（暴露={e_pos} vs {e_neg}）× "
            f"结局 {outcome}（阳性={o_pos} vs {o_neg}），分层于 {stratum}（{usable} 层）\n"
            f"Mantel-Haenszel 合并 OR = Σ(a·d/n)/Σ(b·c/n) = "
            f"{round(float(mh_or), 4) if mh_or not in (float('inf'),) else 'inf'}\n"
            f"CMH 统计量（含连续性校正）χ²=(|Σa-ΣE|-0.5)²/ΣV = {round(cmh_chi2, 4)}，df=1，p={cmh_p:.4g}\n"
            f"结论：{assoc}\n"
            f"Breslow-Day OR 同质性检验（未加 Tarone 校正）：χ²={round(bd_chi2, 4)}，df={bd_df}，p={bd_p_str}\n"
            "注：CMH 在控制分层混杂后检验暴露-结局相关，假定各层有一个共同 OR（Breslow-Day 检验此假定——"
            "若 bd_p<0.05 合并 OR 存疑，应分层报告）；Breslow-Day 用经典未校正式（Tarone 校正未加，"
            "与 R mantelhaen/BreslowDayTest 默认一致，因 ψ_MH 非精确条件 MLE 而略偏松）；"
            "需二值暴露+二值结局；编码取自排序标签（较大标签=暴露/阳性）；"
            "列由 config exposure/outcome/stratum 选定。\n\n"
            "各层 2×2（a=暴露+阳性, b=暴露+阴性, c=非暴露+阳性, d=非暴露+阴性）：\n"
            + strat_df.to_string(index=False),
            encoding="utf-8",
        )
        files.append("cmh_summary.txt")

        summary.append(
            f"{entry.method} 完成：暴露 {exposure} × 结局 {outcome}，分层于 {stratum}（{usable} 层，N={sub.shape[0]}）；"
            f"MH 合并 OR={round(float(mh_or), 3) if mh_or not in (float('inf'),) else 'inf'}，"
            f"CMH χ²={round(cmh_chi2, 3)}（p={cmh_p:.4g}）；{assoc}。"
            + (homog_note if homog_note else "")
            + "⚠ CMH 假定各层共同 OR（Breslow-Day 已检验，bd_p<0.05 则合并 OR 可疑）；"
            "需二值暴露+结局；编码取自排序标签（较大标签=暴露/阳性）；列由 config exposure/outcome/stratum 选定。"
        )
        code += [
            "import numpy as np, pandas as pd",
            "from scipy import stats",
            f"# per stratum k of {stratum}: build 2x2 [[a,b],[c,d]] (exposure {exposure} x outcome {outcome})",
            "mh_or = sum(a*d/n for ...)/sum(b*c/n for ...)  # Mantel-Haenszel common OR",
            "Ea = (a+b)*(a+c)/n; Va = (a+b)*(c+d)*(a+c)*(b+d)/(n**2*(n-1))",
            "cmh = (abs(sum_a - sum_Ea) - 0.5)**2 / sum_Va  # continuity-corrected, df=1",
            "p = stats.chi2.sf(cmh, 1)  # + Breslow-Day test of OR homogeneity across strata",
        ]
    except Exception as err:
        summary.append(f"CMH 检验失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ordinal_association — Goodman-Kruskal γ, Kendall τ-b, Somers' D
# ─────────────────────────────────────────────────────────────────────────────
def _concordance_counts(tab):
    """From an r×c contingency-count table return (C, D, T_x, T_y, total_pairs):
      C = concordant pairs, D = discordant pairs,
      T_x = pairs tied on x only (same row, different col),
      T_y = pairs tied on y only (same col, different row).
    Uses the standard cumulative formulation."""
    import numpy as np

    A = np.asarray(tab, dtype=float)
    r, c = A.shape
    C = D = 0.0
    for i in range(r):
        for j in range(c):
            nij = A[i, j]
            if nij == 0:
                continue
            # concordant: cells strictly greater in BOTH dims (down-right + up-left)
            larger = A[i + 1:, j + 1:].sum() + A[:i, :j].sum()
            # discordant: greater in one dim, smaller in the other
            smaller = A[i + 1:, :j].sum() + A[:i, j + 1:].sum()
            C += nij * larger
            D += nij * smaller
    C /= 2.0
    D /= 2.0
    row_tot = A.sum(axis=1)
    col_tot = A.sum(axis=0)
    n = A.sum()
    total_pairs = n * (n - 1) / 2.0
    tied_rows = float(np.sum(row_tot * (row_tot - 1) / 2.0))   # pairs in same row (tied on x)
    tied_cols = float(np.sum(col_tot * (col_tot - 1) / 2.0))   # pairs in same col (tied on y)
    tied_cells = float(np.sum(A * (A - 1) / 2.0))              # pairs in same cell (tied on both)
    T_x = tied_rows - tied_cells   # tied on x only (same row, different col)
    T_y = tied_cols - tied_cells   # tied on y only (same col, different row)
    return C, D, T_x, T_y, total_pairs


@register("ordinal_association")
def _branch_ordinal_association(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cands = _categorical_like_cols(ctx)
    v1 = cfg.get("var1") if cfg.get("var1") in df.columns else None
    v2 = cfg.get("var2") if cfg.get("var2") in df.columns else None
    if v1 is None or v2 is None:
        rest = [c for c in cands if c not in {v1, v2}]
        if v1 is None:
            v1 = rest.pop(0) if rest else None
        if v2 is None:
            v2 = rest.pop(0) if rest else None
    if v1 is None or v2 is None or v1 == v2:
        summary.append(
            "序数关联跳过：需要 2 个序数/类别列。config 用 var1 / var2 指定。"
        )
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        sub = df[[v1, v2]].dropna()
        if sub.shape[0] < 4:
            summary.append("序数关联跳过：删除缺失后样本不足 4 行。")
            return

        # ordered category axes (sorted labels — DISCLOSED).
        x_cats = _sorted_categories(sub[v1])
        y_cats = _sorted_categories(sub[v2])
        if len(x_cats) < 2 or len(y_cats) < 2:
            summary.append(
                f"序数关联跳过：每个变量至少需 2 个类别（{v1}:{len(x_cats)}, {v2}:{len(y_cats)}）。"
            )
            return

        tab = pd.crosstab(sub[v1], sub[v2])
        # reindex to the disclosed sorted order
        tab = tab.reindex(index=x_cats, columns=y_cats, fill_value=0)
        A = tab.to_numpy(dtype=float)

        C, D, T_x, T_y, total_pairs = _concordance_counts(A)

        # Goodman-Kruskal gamma
        gamma = (C - D) / (C + D) if (C + D) > 0 else float("nan")
        # Somers' D (asymmetric): D_yx normalizes by ties on y only; D_xy by ties on x only.
        somers_d_yx = (C - D) / (C + D + T_y) if (C + D + T_y) > 0 else float("nan")
        somers_d_xy = (C - D) / (C + D + T_x) if (C + D + T_x) > 0 else float("nan")

        # Kendall tau-b (+ p-value) on the numeric rank codes (scipy handles ties).
        x_code = {v: i for i, v in enumerate(x_cats)}
        y_code = {v: i for i, v in enumerate(y_cats)}
        xc = sub[v1].map(x_code).to_numpy()
        yc = sub[v2].map(y_code).to_numpy()
        tau_res = stats.kendalltau(xc, yc)
        tau_b = float(tau_res.correlation)
        tau_b_p = float(tau_res.pvalue)

        # Goodman-Kruskal large-sample (consistent, ASE1) SE for gamma — this is the
        # CONFIDENCE-INTERVAL SE (valid under the alternative):
        #   SE = 4/(C+D)^2 * sqrt( sum_ij n_ij (D·A_ij - C·B_ij)^2 )
        # where A_ij/B_ij are per-cell concordant/discordant neighbour sums.
        # NOTE: ASE1 must NOT be used as the null SE for an H0: gamma=0 z-test (that
        # would be mis-calibrated); the association significance test is τ-b's p
        # (γ and τ-b test the same concordance null). So we report a γ CI, not a γ p.
        gamma_se = float("nan")
        gamma_ci_lo = float("nan")
        gamma_ci_hi = float("nan")
        try:
            r, c = A.shape
            Amat = np.zeros_like(A)
            Bmat = np.zeros_like(A)
            for i in range(r):
                for j in range(c):
                    Amat[i, j] = A[i + 1:, j + 1:].sum() + A[:i, :j].sum()
                    Bmat[i, j] = A[i + 1:, :j].sum() + A[:i, j + 1:].sum()
            if (C + D) > 0:
                s = np.sum(A * (D * Amat - C * Bmat) ** 2)
                gamma_se = float(4.0 / (C + D) ** 2 * np.sqrt(s))
                if gamma_se == gamma_se:
                    gamma_ci_lo = float(gamma - 1.959963985 * gamma_se)
                    gamma_ci_hi = float(gamma + 1.959963985 * gamma_se)
        except Exception:
            pass

        # --- products ---
        tab.to_csv(d / "ordinal_table.csv", encoding="utf-8")
        files.append("ordinal_table.csv")
        meas = pd.DataFrame([
            {"measure": "Goodman-Kruskal gamma", "value": round(gamma, 4),
             "se": round(gamma_se, 4) if gamma_se == gamma_se else float("nan"),
             "ci95": (f"[{gamma_ci_lo:.3f}, {gamma_ci_hi:.3f}]"
                      if gamma_ci_lo == gamma_ci_lo else "NA"),
             "p_value": float("nan")},  # significance: see tau-b (same null)
            {"measure": "Kendall tau-b", "value": round(tau_b, 4), "se": float("nan"),
             "ci95": "NA", "p_value": round(tau_b_p, 6)},
            {"measure": "Somers' D (Y|X)", "value": round(somers_d_yx, 4),
             "se": float("nan"), "ci95": "NA", "p_value": float("nan")},
            {"measure": "Somers' D (X|Y)", "value": round(somers_d_xy, 4),
             "se": float("nan"), "ci95": "NA", "p_value": float("nan")},
        ])
        meas.to_csv(d / "ordinal_measures.csv", index=False, encoding="utf-8")
        files.append("ordinal_measures.csv")

        # --- PNG: contingency heatmap ---
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            r, c = A.shape
            fig, ax = plt.subplots(figsize=(max(5, 0.7 * c + 3), max(4, 0.5 * r + 2)))
            im = ax.imshow(A, cmap="Blues", aspect="auto")
            ax.set_xticks(range(c))
            ax.set_xticklabels([str(x) for x in y_cats], rotation=45, ha="right")
            ax.set_yticks(range(r))
            ax.set_yticklabels([str(x) for x in x_cats])
            ax.set_xlabel(f"{v2} (ordered)")
            ax.set_ylabel(f"{v1} (ordered)")
            ax.set_title(f"Contingency table (gamma={gamma:.2f}, tau-b={tau_b:.2f})")
            for i in range(r):
                for j in range(c):
                    ax.text(j, i, f"{int(A[i, j])}", ha="center", va="center",
                            color="black", fontsize=8)
            fig.colorbar(im, ax=ax, label="count")
            fig.tight_layout()
            fig.savefig(d / "ordinal_heatmap.png", dpi=150)
            plt.close(fig)
            files.append("ordinal_heatmap.png")
        except Exception:
            pass

        estimates["gamma"] = round(float(gamma), 4) if gamma == gamma else float("nan")
        estimates["tau_b"] = round(float(tau_b), 4)
        estimates["tau_b_p"] = round(float(tau_b_p), 6)
        estimates["somers_d_yx"] = round(float(somers_d_yx), 4) if somers_d_yx == somers_d_yx else float("nan")
        estimates["somers_d_xy"] = round(float(somers_d_xy), 4) if somers_d_xy == somers_d_xy else float("nan")
        if gamma_se == gamma_se:
            estimates["gamma_se"] = round(float(gamma_se), 4)
            estimates["gamma_ci_low"] = round(float(gamma_ci_lo), 4)
            estimates["gamma_ci_high"] = round(float(gamma_ci_hi), 4)
        estimates["concordant"] = round(float(C), 1)
        estimates["discordant"] = round(float(D), 1)
        estimates["n"] = float(sub.shape[0])

        strength = "强" if abs(gamma) >= 0.6 else ("中等" if abs(gamma) >= 0.3 else "弱")
        direction = "正向（同向）" if gamma > 0 else ("负向（反向）" if gamma < 0 else "无")
        (d / "ordinal_summary.txt").write_text(
            f"序数关联（{v1} × {v2}）：{A.shape[0]}×{A.shape[1]} 列联表，N={sub.shape[0]}\n"
            f"类别顺序（取自排序标签——请核对是否符合量表）：\n"
            f"  {v1}: {[str(x) for x in x_cats]}\n"
            f"  {v2}: {[str(x) for x in y_cats]}\n"
            f"同序对 C={int(C)}，异序对 D={int(D)}\n"
            f"Goodman-Kruskal γ=(C-D)/(C+D)={round(gamma, 4)}"
            + (f"（ASE1 SE={round(gamma_se, 4)}, 95% CI=[{gamma_ci_lo:.3f}, {gamma_ci_hi:.3f}]）\n"
               if gamma_se == gamma_se else "\n")
            + f"Kendall τ-b={round(tau_b, 4)}（p={tau_b_p:.4g}）\n"
            f"Somers' D(Y|X)={round(somers_d_yx, 4)}，Somers' D(X|Y)={round(somers_d_xy, 4)}\n"
            f"关联强度：{strength}，方向：{direction}\n"
            "注：这些都是序数关联度量（基于同序/异序对）——γ 忽略所有平局；τ-b 对行列平局做了调整；"
            "Somers' D 是非对称的（按其中一个变量的平局归一化，可作「以 X 预测 Y」的 PRE 解读）；"
            "类别顺序取自排序后的标签，请核对是否与真实量表一致；若变量为名义（无序）请改用 cramers_v / 卡方检验；"
            "γ 的 SE 是置信区间用的 ASE1（在备择下一致），不能拿来做 γ=0 的 z 检验——"
            "关联是否显著请看 τ-b 的 p（γ 与 τ-b 检验同一同序/异序 null）；"
            "列由 config var1/var2 选定。\n\n"
            "列联表（已按顺序重排）：\n" + tab.to_string() + "\n\n"
            "关联度量：\n" + meas.to_string(index=False),
            encoding="utf-8",
        )
        files.append("ordinal_summary.txt")

        summary.append(
            f"{entry.method} 完成（{v1} × {v2}，{A.shape[0]}×{A.shape[1]} 表，N={sub.shape[0]}）："
            f"γ={round(gamma, 3)}"
            + (f"（95% CI=[{gamma_ci_lo:.2f}, {gamma_ci_hi:.2f}]）" if gamma_ci_lo == gamma_ci_lo else "")
            + f"，Kendall τ-b={round(tau_b, 3)}（p={tau_b_p:.4g}，关联显著性以此为准），"
            f"Somers' D(Y|X)={round(somers_d_yx, 3)}，D(X|Y)={round(somers_d_xy, 3)}；"
            f"{strength}{direction}关联。"
            + "⚠ 类别顺序取自排序标签——请核对是否符合量表；名义（无序）变量请改用 cramers_v / 卡方；"
            "γ 忽略平局、τ-b 调整平局、Somers' D 非对称；列由 config var1/var2 选定。"
        )
        code += [
            "import pandas as pd, numpy as np",
            "from scipy import stats",
            f"tab = pd.crosstab(df[{v1!r}], df[{v2!r}])  # ordinal x ordinal (sorted category order)",
            "# C, D = concordant / discordant pairs from the table",
            "gamma = (C - D)/(C + D)  # Goodman-Kruskal gamma (ignores ties)",
            "tau_b, p = stats.kendalltau(x_codes, y_codes)  # Kendall tau-b (ties-adjusted) + p",
            "# Somers' D_yx=(C-D)/(C+D+T_y), D_xy=(C-D)/(C+D+T_x)  (asymmetric PRE measures)",
        ]
    except Exception as err:
        summary.append(f"序数关联失败：{err}")
