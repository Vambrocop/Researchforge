"""Branch handlers for the categorical_tests family.

Three CLASSICAL categorical hypothesis tests (pure Python — numpy / scipy /
statsmodels / pandas; NO R). These are DISTINCT from the `categorical` family
(loglinear / cmh_test / ordinal_association) — those model association strength;
these are the textbook independence / exact / paired hypothesis tests:

  * chi_square_test — Pearson chi-square test of independence for two categorical
                      columns (config var1/var2 else first two low-cardinality
                      categorical/binary), AND a goodness-of-fit mode (one
                      categorical column vs uniform or config `expected`).
                      Reports chi2, df, p; bias-corrected Cramer's V effect size;
                      expected counts; flags cells with expected<5 (+ Yates note
                      for 2x2).
  * fisher_exact  — Fisher's exact test for a 2x2 table (odds ratio + exact p,
                    1- and 2-sided via scipy.stats.fisher_exact). r×c degrades
                    honestly to a Monte-Carlo / chi-square approximation with a
                    note (scipy Fisher is 2x2 only).
  * mcnemar_test  — McNemar's test for PAIRED binary data (two binary measurements
                    on the same subjects). Builds the 2x2 paired table and uses
                    statsmodels.stats.contingency_tables.mcnemar (exact for small
                    discordant counts, else continuity-corrected chi-square).

Each handler resolves its columns (config override else auto-pick from profiler
kinds), degrades honestly (wrong #cols/kinds, non-binary for mcnemar, too few
rows, import missing → append a Chinese "<方法>跳过：<原因>" to summary and
RETURN — never crash, never fabricate), writes CSV + PNG (matplotlib Agg, ENGLISH
plot labels), fills float `estimates`, appends a Chinese `summary` ending with ⚠
disclosures, and MUTATES ctx (never rebinds). See executor/_branch_api.py and
CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ─────────────────────────────────────────────────────────────────────────────
# Shared column resolution — MIRRORS categorical.py's low-cardinality factor idea.
# A contingency-table factor must be LOW-cardinality: count/id columns can be
# all-distinct integers (profiler "id trap") which are NOT factors. Cap the
# distinct-value count so a near-continuous integer column is screened out of
# AUTO-selection (config var1/var2 bypass this filter).
# ─────────────────────────────────────────────────────────────────────────────
def _categorical_like_cols(ctx: Ctx) -> list[str]:
    df, fp = ctx.df, ctx.fp
    excl = {fp.unit_col, fp.time_col}
    cap = max(2, min(20, len(df) // 2))
    out = []
    for c in fp.columns:
        if c.name in excl or c.kind not in {"categorical", "binary", "count", "id"}:
            continue
        if 2 <= df[c.name].dropna().nunique() <= cap:
            out.append(c.name)
    return out


def _binary_cols(ctx: Ctx) -> list[str]:
    """Columns with exactly two distinct non-missing values."""
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
    """Ordered category set: numeric ascending if all numeric, else lexicographic."""
    import pandas as pd

    cats = pd.unique(series.dropna())
    try:
        return sorted(cats, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(cats, key=lambda v: str(v))


def _resolve_pair(ctx: Ctx):
    """config var1/var2 (any column, bypass filter) else first two low-cardinality
    categorical/binary columns. Returns (v1, v2) — either may be None."""
    df, cfg = ctx.df, ctx.cfg
    cands = _categorical_like_cols(ctx)
    v1 = cfg.get("var1") if cfg.get("var1") in df.columns else None
    v2 = cfg.get("var2") if cfg.get("var2") in df.columns else None
    if v1 is None or v2 is None:
        rest = [c for c in cands if c not in {v1, v2}]
        if v1 is None:
            v1 = rest.pop(0) if rest else None
        if v2 is None:
            v2 = rest.pop(0) if rest else None
    return v1, v2


# ─────────────────────────────────────────────────────────────────────────────
# 1. chi_square_test — Pearson chi-square test of independence + goodness-of-fit
# ─────────────────────────────────────────────────────────────────────────────
@register("chi_square_test")
def _branch_chi_square_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    v1, v2 = _resolve_pair(ctx)

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        from scipy.stats import chi2_contingency
    except Exception as err:
        summary.append(f"卡方检验跳过：缺少依赖（numpy/scipy/pandas）：{err}")
        return

    # ── Goodness-of-fit mode: exactly one usable categorical column (or only one
    # resolvable) → test observed vs uniform / config `expected`. ──
    two_way = v1 is not None and v2 is not None and v1 != v2
    if not two_way:
        gof_col = v1 or v2
        if gof_col is None:
            summary.append(
                "卡方检验跳过：需要 ≥1 个低基数类别/二值列。"
                "两列做独立性检验，单列做拟合优度检验。config 用 var1/var2 指定。"
            )
            return
        try:
            s = df[gof_col].dropna()
            cats = _sorted_categories(s)
            if len(cats) < 2:
                summary.append(
                    f"卡方检验跳过：列 {gof_col} 只有 {len(cats)} 个类别，拟合优度检验至少需 2 类。"
                )
                return
            obs = np.array([float((s == cat).sum()) for cat in cats])
            n = float(obs.sum())
            if n < 4:
                summary.append("卡方检验（拟合优度）跳过：删除缺失后样本不足 4 行。")
                return
            # expected: config dict {category: prob/count} else uniform
            exp_cfg = cfg.get("expected")
            if isinstance(exp_cfg, dict):
                raw = np.array([float(exp_cfg.get(cat, exp_cfg.get(str(cat), 0.0)))
                                for cat in cats])
                if raw.sum() <= 0:
                    exp = np.full(len(cats), n / len(cats))
                    exp_kind = "均匀（config expected 全 0，已回退均匀）"
                else:
                    exp = raw / raw.sum() * n  # normalize to total count
                    exp_kind = "config 指定的期望分布"
            else:
                exp = np.full(len(cats), n / len(cats))
                exp_kind = "均匀分布（每类等概率）"

            chi2 = float(np.sum((obs - exp) ** 2 / exp))
            dof = len(cats) - 1
            p = float(stats.chi2.sf(chi2, dof))
            n_sparse = int(np.sum(exp < 5))

            tab = pd.DataFrame({
                "category": [str(c) for c in cats],
                "observed": np.round(obs, 4),
                "expected": np.round(exp, 4),
            })
            try:
                tab.to_csv(d / "chi_square_gof_table.csv", index=False, encoding="utf-8")
                files.append("chi_square_gof_table.csv")
            except Exception:
                pass

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                x = np.arange(len(cats))
                w = 0.4
                fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(cats) + 2), 4))
                ax.bar(x - w / 2, obs, w, label="Observed", color="#4C72B0")
                ax.bar(x + w / 2, exp, w, label="Expected", color="#C44E52")
                ax.set_xticks(x)
                ax.set_xticklabels([str(c) for c in cats], rotation=45, ha="right")
                ax.set_ylabel("Count")
                ax.set_title(f"Chi-square goodness-of-fit ({gof_col})")
                ax.legend()
                fig.tight_layout()
                fig.savefig(d / "chi_square_gof.png", dpi=150)
                plt.close(fig)
                files.append("chi_square_gof.png")
            except Exception:
                pass

            estimates["chi2"] = round(chi2, 4)
            estimates["df"] = float(dof)
            estimates["p_value"] = round(p, 6)
            estimates["cramers_v"] = float("nan")  # V undefined for GoF (no 2-way table)
            estimates["n_sparse_cells"] = float(n_sparse)
            estimates["n"] = round(n, 1)

            verdict = (
                f"拒绝「与期望分布一致」（χ²={round(chi2, 3)}, df={dof}, p={p:.4g}<0.05）"
                if p < 0.05 else
                f"不能拒绝「与期望分布一致」（χ²={round(chi2, 3)}, df={dof}, p={p:.4g}≥0.05）"
            )
            sparse_note = (
                f"⚠ {n_sparse} 个类别期望频数<5，渐近卡方近似可能不稳。"
                if n_sparse > 0 else ""
            )
            summary.append(
                f"{entry.method} 完成（拟合优度：{gof_col} vs {exp_kind}，{len(cats)} 类，N={int(n)}）："
                f"χ²={round(chi2, 3)}（df={dof}, p={p:.4g}）；{verdict}。"
                + sparse_note
                + "⚠ 拟合优度检验观测频数是否符合给定期望分布（默认均匀，config expected 可指定）；"
                "需期望频数大多≥5；列由 config var1 指定。"
            )
            code += [
                "import numpy as np; from scipy import stats",
                f"s = df[{gof_col!r}].dropna(); cats = sorted(s.unique())",
                "obs = np.array([(s==c).sum() for c in cats], float); n = obs.sum()",
                "exp = np.full(len(cats), n/len(cats))  # uniform (or config expected)",
                "chi2 = ((obs-exp)**2/exp).sum(); df = len(cats)-1",
                "p = stats.chi2.sf(chi2, df)  # goodness-of-fit",
            ]
        except Exception as err:
            summary.append(f"卡方检验（拟合优度）失败：{err}")
        return

    # ── Independence mode: two categorical columns ──
    try:
        sub = df[[v1, v2]].dropna()
        if sub.shape[0] < 4:
            summary.append("卡方检验跳过：成对删除缺失后样本不足 4 行。")
            return
        obs = pd.crosstab(sub[v1], sub[v2])
        r, c = obs.shape
        if r < 2 or c < 2:
            summary.append(
                f"卡方检验跳过：列联表维度 {r}×{c}，每个变量至少需 2 个类别。"
            )
            return

        O = obs.to_numpy(dtype=float)
        n = float(O.sum())
        is_2x2 = (r == 2 and c == 2)
        # Yates continuity correction is scipy's default for 2x2; we report the
        # UNCORRECTED chi2 as the headline (matches Cramer's V / loglinear), and
        # also surface the Yates-corrected value for 2x2.
        chi2, p, dof, E = chi2_contingency(O, correction=False)
        chi2 = float(chi2); p = float(p); dof = int(dof)
        chi2_yates = p_yates = None
        if is_2x2:
            cy, py, _, _ = chi2_contingency(O, correction=True)
            chi2_yates = float(cy); p_yates = float(py)

        # Bias-corrected Cramer's V (Bergsma 2013): correct phi2 and the table dims
        # so V is not inflated in small / large-table samples.
        phi2 = chi2 / n
        phi2c = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        ccorr = c - (c - 1) ** 2 / (n - 1)
        denom = min(rcorr - 1, ccorr - 1)
        cramers_v = float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")

        n_sparse = int(np.sum(E < 5))
        min_expected = float(E.min())

        # ── products ──
        try:
            obs.to_csv(d / "chi_square_observed.csv", encoding="utf-8")
            files.append("chi_square_observed.csv")
            exp_df = pd.DataFrame(np.round(E, 4), index=obs.index, columns=obs.columns)
            exp_df.to_csv(d / "chi_square_expected.csv", encoding="utf-8")
            files.append("chi_square_expected.csv")
        except Exception:
            pass

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rlabels = list(obs.index)
            clabels = list(obs.columns)
            fig, ax = plt.subplots(figsize=(max(5, 0.7 * c + 3), max(4, 0.5 * r + 2)))
            im = ax.imshow(O, cmap="Blues", aspect="auto")
            ax.set_xticks(range(c))
            ax.set_xticklabels([str(x) for x in clabels], rotation=45, ha="right")
            ax.set_yticks(range(r))
            ax.set_yticklabels([str(x) for x in rlabels])
            ax.set_xlabel(str(v2))
            ax.set_ylabel(str(v1))
            ax.set_title(f"Contingency table (chi2={chi2:.1f}, Cramer's V={cramers_v:.2f})")
            for i in range(r):
                for j in range(c):
                    ax.text(j, i, f"{int(O[i, j])}\n({E[i, j]:.1f})", ha="center",
                            va="center", color="black", fontsize=8)
            fig.colorbar(im, ax=ax, label="observed count")
            fig.tight_layout()
            fig.savefig(d / "chi_square_heatmap.png", dpi=150)
            plt.close(fig)
            files.append("chi_square_heatmap.png")
        except Exception:
            pass

        estimates["chi2"] = round(chi2, 4)
        estimates["df"] = float(dof)
        estimates["p_value"] = round(p, 6)
        estimates["cramers_v"] = round(cramers_v, 4) if cramers_v == cramers_v else float("nan")
        estimates["n_sparse_cells"] = float(n_sparse)
        estimates["n"] = round(n, 1)
        if chi2_yates is not None:
            estimates["chi2_yates"] = round(chi2_yates, 4)
            estimates["p_value_yates"] = round(p_yates, 6)

        verdict = (
            f"两变量不独立（χ²={round(chi2, 3)}, df={dof}, p={p:.4g}<0.05）"
            if p < 0.05 else
            f"不能拒绝独立性（χ²={round(chi2, 3)}, df={dof}, p={p:.4g}≥0.05）"
        )
        vstrength = ("强" if cramers_v >= 0.5 else ("中等" if cramers_v >= 0.3
                     else ("弱" if cramers_v >= 0.1 else "极弱/无"))) \
            if cramers_v == cramers_v else "未定义"
        sparse_note = (
            f"⚠ {n_sparse} 个单元格期望频数<5（最小期望 {round(min_expected, 2)}），"
            "渐近卡方近似可能不稳，建议合并稀疏类别或改用 fisher_exact。"
            if n_sparse > 0 else ""
        )
        yates_note = (
            f"（2×2 表，Yates 连续性校正 χ²={round(chi2_yates, 3)}, p={p_yates:.4g}）"
            if chi2_yates is not None else ""
        )
        summary.append(
            f"{entry.method} 完成（独立性：{v1} × {v2}，{r}×{c} 表，N={int(n)}）："
            f"Pearson χ²={round(chi2, 3)}（df={dof}, p={p:.4g}）{yates_note}，"
            f"偏差校正 Cramér's V={round(cramers_v, 3) if cramers_v == cramers_v else 'NA'}（{vstrength}）；{verdict}。"
            + sparse_note
            + "⚠ 卡方检验需期望频数大多≥5（不满足→改用 fisher_exact）；只检验是否相关、不指明方向；"
            "Cramér's V 为效应量（偏差校正）；2×2 默认报告未校正 χ²、另附 Yates；列由 config var1/var2 指定。"
        )
        code += [
            "import numpy as np, pandas as pd",
            "from scipy.stats import chi2_contingency",
            f"obs = pd.crosstab(df[{v1!r}], df[{v2!r}])  # r×c contingency table",
            "chi2, p, dof, E = chi2_contingency(obs, correction=False)  # Pearson chi-square",
            "phi2 = chi2/obs.values.sum(); r,c = obs.shape; n = obs.values.sum()",
            "phi2c = max(0, phi2-(c-1)*(r-1)/(n-1))  # Bergsma bias-corrected Cramer's V",
            "V = (phi2c/min(r-(r-1)**2/(n-1)-1, c-(c-1)**2/(n-1)-1))**0.5",
        ]
    except Exception as err:
        summary.append(f"卡方检验失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. fisher_exact — Fisher's exact test (2x2 exact; r×c honest degrade)
# ─────────────────────────────────────────────────────────────────────────────
@register("fisher_exact")
def _branch_fisher_exact(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    v1, v2 = _resolve_pair(ctx)
    if v1 is None or v2 is None or v1 == v2:
        summary.append(
            "Fisher 精确检验跳过：需要 2 个低基数类别/二值列。config 用 var1/var2 指定。"
        )
        return

    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import fisher_exact, chi2_contingency
    except Exception as err:
        summary.append(f"Fisher 精确检验跳过：缺少依赖（numpy/scipy/pandas）：{err}")
        return

    try:
        sub = df[[v1, v2]].dropna()
        if sub.shape[0] < 4:
            summary.append("Fisher 精确检验跳过：成对删除缺失后样本不足 4 行。")
            return
        obs = pd.crosstab(sub[v1], sub[v2])
        r, c = obs.shape
        if r < 2 or c < 2:
            summary.append(
                f"Fisher 精确检验跳过：列联表维度 {r}×{c}，每个变量至少需 2 个类别。"
            )
            return

        O = obs.to_numpy(dtype=float)
        n = float(O.sum())
        is_2x2 = (r == 2 and c == 2)

        try:
            obs.to_csv(d / "fisher_table.csv", encoding="utf-8")
            files.append("fisher_table.csv")
        except Exception:
            pass

        if is_2x2:
            # scipy fisher_exact returns (odds_ratio, p) for 2x2. Sample (conditional
            # MLE) OR is the standard Fisher OR; also give the 1-sided p (greater).
            odds_ratio, p_two = fisher_exact(O, alternative="two-sided")
            _, p_greater = fisher_exact(O, alternative="greater")
            _, p_less = fisher_exact(O, alternative="less")
            odds_ratio = float(odds_ratio); p_two = float(p_two)
            p_greater = float(p_greater); p_less = float(p_less)

            estimates["odds_ratio"] = round(odds_ratio, 4) if np.isfinite(odds_ratio) else float("inf")
            estimates["p_value"] = round(p_two, 6)
            estimates["p_one_sided"] = round(min(p_greater, p_less), 6)
            estimates["n"] = round(n, 1)
            estimates["table_rows"] = float(r)
            estimates["table_cols"] = float(c)

            # PNG: grouped bar of the 2x2
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                rlabels = [str(x) for x in obs.index]
                clabels = [str(x) for x in obs.columns]
                x = np.arange(2)
                w = 0.38
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.bar(x - w / 2, O[:, 0], w, label=f"{v2}={clabels[0]}", color="#4C72B0")
                ax.bar(x + w / 2, O[:, 1], w, label=f"{v2}={clabels[1]}", color="#C44E52")
                ax.set_xticks(x)
                ax.set_xticklabels([f"{v1}={lab}" for lab in rlabels])
                ax.set_ylabel("Count")
                ax.set_title(f"Fisher exact 2x2 (OR={odds_ratio:.2f}, p={p_two:.3g})")
                ax.legend()
                fig.tight_layout()
                fig.savefig(d / "fisher_bar.png", dpi=150)
                plt.close(fig)
                files.append("fisher_bar.png")
            except Exception:
                pass

            or_dir = ("正向关联（OR>1）" if odds_ratio > 1 else
                      ("负向关联（OR<1）" if odds_ratio < 1 else "无关联（OR=1）"))
            verdict = (
                f"显著相关（精确 p={p_two:.4g}<0.05）" if p_two < 0.05
                else f"未见显著相关（精确 p={p_two:.4g}≥0.05）"
            )
            (d / "fisher_summary.txt").write_text(
                f"Fisher 精确检验（{v1} × {v2}，2×2，N={int(n)}）\n"
                f"条件最大似然比值比 OR={round(odds_ratio, 4)}\n"
                f"双侧精确 p={p_two:.4g}；单侧 greater p={p_greater:.4g}，less p={p_less:.4g}\n"
                f"结论：{verdict}，{or_dir}\n"
                "注：精确检验基于超几何分布，不依赖大样本近似——对小样本/稀疏 2×2 优于卡方；"
                "OR 为条件 MLE（与样本 ad/bc 略有差异）；列由 config var1/var2 选定。\n\n"
                "2×2 表：\n" + obs.to_string(),
                encoding="utf-8",
            )
            files.append("fisher_summary.txt")

            summary.append(
                f"{entry.method} 完成（{v1} × {v2}，2×2，N={int(n)}）："
                f"OR={round(odds_ratio, 3) if np.isfinite(odds_ratio) else 'inf'}，"
                f"双侧精确 p={p_two:.4g}（单侧 p={min(p_greater, p_less):.4g}）；{verdict}，{or_dir}。"
                "⚠ 精确检验不依赖大样本假定——小样本/稀疏 2×2 首选；OR 为条件 MLE；列由 config var1/var2 指定。"
            )
            code += [
                "from scipy.stats import fisher_exact",
                f"obs = pd.crosstab(df[{v1!r}], df[{v2!r}])  # must be 2x2",
                "odds_ratio, p = fisher_exact(obs, alternative='two-sided')  # exact (hypergeometric)",
            ]
        else:
            # r×c: scipy.stats.fisher_exact is 2x2 ONLY. Honest degrade — try a
            # Monte-Carlo permutation p (no large-sample assumption), else fall
            # back to the asymptotic chi-square. Either way: clearly noted.
            chi2, p_chi, dof, E = chi2_contingency(O, correction=False)
            chi2 = float(chi2); p_chi = float(p_chi); dof = int(dof)
            mc_p = float("nan")
            mc_iter = 0
            try:
                # Monte-Carlo: resample tables with the SAME margins, compare chi2.
                rng = np.random.default_rng(12345)
                row_tot = O.sum(axis=1).astype(int)
                col_tot = O.sum(axis=0).astype(int)
                Earr = np.asarray(E, dtype=float)
                obs_stat = float(np.sum((O - Earr) ** 2 / Earr))
                n_iter = 2000
                ge = 0
                # fix the row assignment, permute the column labels (preserves margins)
                col_labels = np.repeat(np.arange(c), col_tot)
                row_index = np.repeat(np.arange(r), row_tot)
                for _ in range(n_iter):
                    perm = rng.permutation(col_labels)
                    sim = np.zeros((r, c), dtype=float)
                    for ri, ci in zip(row_index, perm):
                        sim[ri, ci] += 1.0
                    rt = sim.sum(axis=1, keepdims=True)
                    ct = sim.sum(axis=0, keepdims=True)
                    Es = rt @ ct / sim.sum()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        stat = float(np.nansum(np.where(Es > 0, (sim - Es) ** 2 / Es, 0.0)))
                    if stat >= obs_stat - 1e-9:
                        ge += 1
                mc_iter = n_iter
                mc_p = (ge + 1) / (n_iter + 1)  # add-one (Davison-Hinkley) estimator
            except Exception:
                mc_p = float("nan")
                mc_iter = 0

            # Cramer's V (bias-corrected) as an effect size for the r×c table.
            phi2 = chi2 / n
            phi2c = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
            denom = min(r - (r - 1) ** 2 / (n - 1) - 1,
                        c - (c - 1) ** 2 / (n - 1) - 1)
            cramers_v = float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")

            chosen_p = mc_p if mc_p == mc_p else p_chi
            estimates["odds_ratio"] = float("nan")  # OR undefined for r×c
            estimates["p_value"] = round(chosen_p, 6)
            estimates["p_chi2_asymptotic"] = round(p_chi, 6)
            if mc_p == mc_p:
                estimates["p_monte_carlo"] = round(mc_p, 6)
            estimates["n"] = round(n, 1)
            estimates["table_rows"] = float(r)
            estimates["table_cols"] = float(c)

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                rlabels = list(obs.index)
                clabels = list(obs.columns)
                fig, ax = plt.subplots(figsize=(max(5, 0.7 * c + 3), max(4, 0.5 * r + 2)))
                im = ax.imshow(O, cmap="Blues", aspect="auto")
                ax.set_xticks(range(c))
                ax.set_xticklabels([str(x) for x in clabels], rotation=45, ha="right")
                ax.set_yticks(range(r))
                ax.set_yticklabels([str(x) for x in rlabels])
                ax.set_xlabel(str(v2))
                ax.set_ylabel(str(v1))
                ax.set_title(f"{r}x{c} table (Fisher 2x2-only; MC/chi-square fallback)")
                for i in range(r):
                    for j in range(c):
                        ax.text(j, i, f"{int(O[i, j])}", ha="center", va="center",
                                color="black", fontsize=8)
                fig.colorbar(im, ax=ax, label="count")
                fig.tight_layout()
                fig.savefig(d / "fisher_table_heatmap.png", dpi=150)
                plt.close(fig)
                files.append("fisher_table_heatmap.png")
            except Exception:
                pass

            method_used = (
                f"蒙特卡洛置换检验（{mc_iter} 次重采样，固定边际）" if mc_p == mc_p
                else "渐近 Pearson 卡方（蒙特卡洛失败时的回退）"
            )
            verdict = (
                f"显著相关（p={chosen_p:.4g}<0.05）" if chosen_p < 0.05
                else f"未见显著相关（p={chosen_p:.4g}≥0.05）"
            )
            (d / "fisher_summary.txt").write_text(
                f"Fisher 精确检验（{v1} × {v2}，{r}×{c}，N={int(n)}）\n"
                f"⚠ scipy.stats.fisher_exact 仅支持 2×2 表；本表 {r}×{c}，已诚实降级为：{method_used}\n"
                f"采用 p={chosen_p:.4g}（蒙特卡洛 p={mc_p if mc_p==mc_p else 'NA'}；"
                f"渐近卡方 p={p_chi:.4g}, χ²={round(chi2, 4)}, df={dof}）\n"
                f"偏差校正 Cramér's V（效应量）={round(cramers_v, 4) if cramers_v == cramers_v else 'NA'}\n"
                f"结论：{verdict}\n"
                "注：r×c 的精确检验计算昂贵，scipy 未提供；这里用固定边际的蒙特卡洛置换 p"
                "（无大样本假定）近似，失败则回退渐近卡方；OR 仅对 2×2 有定义（此处 NA）；"
                "列由 config var1/var2 选定。\n\n"
                "列联表：\n" + obs.to_string(),
                encoding="utf-8",
            )
            files.append("fisher_summary.txt")

            summary.append(
                f"{entry.method} 完成（{v1} × {v2}，{r}×{c}，N={int(n)}）："
                f"⚠ scipy Fisher 仅支持 2×2 → 已降级为{method_used}；"
                f"p={chosen_p:.4g}（渐近卡方 p={p_chi:.4g}），"
                f"Cramér's V={round(cramers_v, 3) if cramers_v == cramers_v else 'NA'}；{verdict}。"
                "⚠ r×c 精确检验昂贵且 scipy 未提供，已诚实降级为蒙特卡洛置换/卡方；"
                "OR 仅 2×2 有定义（此处 NA）；列由 config var1/var2 指定。"
            )
            code += [
                "from scipy.stats import fisher_exact, chi2_contingency",
                "# scipy fisher_exact is 2x2 ONLY; for r×c degrade to Monte-Carlo / chi-square",
                f"obs = pd.crosstab(df[{v1!r}], df[{v2!r}])",
                "chi2, p, dof, E = chi2_contingency(obs, correction=False)  # asymptotic fallback",
                "# Monte-Carlo: resample tables with fixed margins, compare chi2 statistics",
            ]
    except Exception as err:
        summary.append(f"Fisher 精确检验失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. mcnemar_test — McNemar's test for PAIRED binary data
# ─────────────────────────────────────────────────────────────────────────────
@register("mcnemar_test")
def _branch_mcnemar_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    binc = _binary_cols(ctx)
    v1 = cfg.get("var1") if cfg.get("var1") in df.columns else None
    v2 = cfg.get("var2") if cfg.get("var2") in df.columns else None
    if v1 is None or v2 is None:
        rest = [c for c in binc if c not in {v1, v2}]
        if v1 is None:
            v1 = rest.pop(0) if rest else None
        if v2 is None:
            v2 = rest.pop(0) if rest else None
    if v1 is None or v2 is None or v1 == v2:
        summary.append(
            "McNemar 检验跳过：需要 2 个二值列（同一批受试者上的两次测量，如前/后、评分者1/2）。"
            "config 用 var1/var2 指定。"
        )
        return

    try:
        import numpy as np
        import pandas as pd
    except Exception as err:
        summary.append(f"McNemar 检验跳过：缺少依赖（numpy/pandas）：{err}")
        return
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except Exception as err:
        summary.append(
            f"McNemar 检验跳过：需要 statsmodels（statsmodels.stats.contingency_tables.mcnemar）：{err}"
        )
        return

    try:
        sub = df[[v1, v2]].dropna()
        if sub.shape[0] < 4:
            summary.append("McNemar 检验跳过：成对删除缺失后配对样本不足 4 对。")
            return

        l1 = _sorted_categories(sub[v1])
        l2 = _sorted_categories(sub[v2])
        if len(l1) != 2 or len(l2) != 2:
            summary.append(
                f"McNemar 检验跳过：两列都须恰好 2 个水平（{v1}:{len(l1)}, {v2}:{len(l2)}）；"
                "McNemar 仅用于配对二值数据。config 指定二值列。"
            )
            return
        # consistent coding: larger sorted label = "positive" (1).
        pos1, neg1 = l1[1], l1[0]
        pos2, neg2 = l2[1], l2[0]

        # paired 2x2: rows = var1 (neg, pos), cols = var2 (neg, pos)
        nn = float(((sub[v1] == neg1) & (sub[v2] == neg2)).sum())
        np_ = float(((sub[v1] == neg1) & (sub[v2] == pos2)).sum())  # b: 0 -> 1
        pn = float(((sub[v1] == pos1) & (sub[v2] == neg2)).sum())   # c: 1 -> 0
        pp = float(((sub[v1] == pos1) & (sub[v2] == pos2)).sum())
        table = np.array([[nn, np_], [pn, pp]], dtype=float)
        n_pairs = float(table.sum())

        # Discordant pairs: b = (neg1, pos2) -> moved 0->1; c = (pos1, neg2) -> 1->0.
        b = np_
        c = pn
        n_discordant = b + c

        # statsmodels: exact (binomial) when discordant counts small, else
        # continuity-corrected chi-square (classic rule: exact if n_disc < 25).
        use_exact = n_discordant < 25
        res = mcnemar(table, exact=use_exact, correction=True)
        statistic = float(res.statistic)
        p_value = float(res.pvalue)
        test_kind = ("精确二项检验" if use_exact else "连续性校正 χ²（df=1）")

        # odds of change: b/c (how much more likely a 0->1 switch than 1->0).
        odds_change = (b / c) if c > 0 else (float("inf") if b > 0 else float("nan"))

        try:
            tab_df = pd.DataFrame(
                table,
                index=[f"{v1}={neg1}", f"{v1}={pos1}"],
                columns=[f"{v2}={neg2}", f"{v2}={pos2}"],
            )
            tab_df.to_csv(d / "mcnemar_paired_table.csv", encoding="utf-8")
            files.append("mcnemar_paired_table.csv")
        except Exception:
            pass

        # PNG: discordant-pairs bar (b vs c).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(
                [f"b: {neg1}->{pos2}\n({v1}=0, {v2}=1)", f"c: {pos1}->{neg2}\n({v1}=1, {v2}=0)"],
                [b, c], color=["#4C72B0", "#C44E52"],
            )
            ax.set_ylabel("Discordant pair count")
            ax.set_title(f"McNemar discordant pairs (stat={statistic:.2f}, p={p_value:.3g})")
            for rect, val in zip(bars, [b, c]):
                ax.text(rect.get_x() + rect.get_width() / 2, val, f"{int(val)}",
                        ha="center", va="bottom")
            fig.tight_layout()
            fig.savefig(d / "mcnemar_discordant.png", dpi=150)
            plt.close(fig)
            files.append("mcnemar_discordant.png")
        except Exception:
            pass

        estimates["statistic"] = round(statistic, 4)
        estimates["p_value"] = round(p_value, 6)
        estimates["n_discordant"] = float(n_discordant)
        estimates["b"] = float(b)
        estimates["c"] = float(c)
        estimates["n_pairs"] = float(n_pairs)
        if odds_change == odds_change and odds_change not in (float("inf"),):
            estimates["odds_of_change"] = round(float(odds_change), 4)

        verdict = (
            f"配对边际分布显著改变（{test_kind}：stat={round(statistic, 3)}, p={p_value:.4g}<0.05）"
            if p_value < 0.05 else
            f"未见配对边际分布改变（{test_kind}：stat={round(statistic, 3)}, p={p_value:.4g}≥0.05）"
        )
        if b > c:
            direction = (
                f"0→1 的变化多于 1→0（b={int(b)} vs c={int(c)}，变化比 b/c="
                f"{round(odds_change, 3) if odds_change == odds_change and odds_change not in (float('inf'),) else 'inf'}）"
            )
        elif c > b:
            direction = f"1→0 的变化多于 0→1（b={int(b)} vs c={int(c)}）"
        else:
            direction = f"两方向变化相等（b=c={int(b)}）"
        disc_note = (
            f"⚠ 不一致配对很少（b+c={int(n_discordant)}<10），检验功效低，p 值仅供参考。"
            if n_discordant < 10 else ""
        )
        oc_str = (round(odds_change, 4) if odds_change == odds_change
                  and odds_change not in (float("inf"),) else "inf")
        (d / "mcnemar_summary.txt").write_text(
            f"McNemar 配对检验（{v1} × {v2}，{int(n_pairs)} 对；阳性={pos1}/{pos2}，阴性={neg1}/{neg2}）\n"
            f"配对 2×2：一致 (0,0)={int(nn)}、(1,1)={int(pp)}；不一致 b(0→1)={int(b)}、c(1→0)={int(c)}\n"
            f"{test_kind}：统计量={round(statistic, 4)}，p={p_value:.4g}\n"
            f"变化比 odds of change b/c={oc_str}\n"
            f"结论：{verdict}；{direction}\n"
            "注：McNemar 检验配对二值数据的「边际同质性」（两次测量的阳性率是否一致），"
            "不是独立性检验——只有不一致配对 (b, c) 携带信息，一致配对 (a, d) 被忽略；"
            "需同一批单位上的两个二值测量；编码取自排序标签（较大标签=阳性）；"
            "不一致数<25 用精确二项检验、否则连续性校正 χ²；列由 config var1/var2 选定。\n\n"
            "配对 2×2 表：\n"
            + pd.DataFrame(table, index=[f"{v1}={neg1}", f"{v1}={pos1}"],
                           columns=[f"{v2}={neg2}", f"{v2}={pos2}"]).to_string(),
            encoding="utf-8",
        )
        files.append("mcnemar_summary.txt")

        summary.append(
            f"{entry.method} 完成（{v1} × {v2}，{int(n_pairs)} 对，不一致 b+c={int(n_discordant)}）："
            f"{test_kind} 统计量={round(statistic, 3)}（p={p_value:.4g}）；b（0→1）={int(b)}, c（1→0）={int(c)}；{verdict}。"
            + disc_note
            + "⚠ McNemar 检验配对二值数据的边际同质性（非独立性）——只有不一致配对 (b,c) 起作用；"
            "需同一批单位的两个二值测量；编码较大标签=阳性；不一致数<25 用精确检验否则校正 χ²；列由 config var1/var2 指定。"
        )
        code += [
            "import numpy as np",
            "from statsmodels.stats.contingency_tables import mcnemar",
            f"# paired 2x2 from two binary columns {v1!r} (rows) x {v2!r} (cols)",
            "table = np.array([[a, b], [c, d]])  # b,c = discordant pairs",
            "res = mcnemar(table, exact=(b+c < 25), correction=True)  # exact or CC chi-square",
            "stat, p = res.statistic, res.pvalue  # tests MARGINAL homogeneity (paired)",
        ]
    except Exception as err:
        summary.append(f"McNemar 检验失败：{err}")
