"""Branch handlers for the multivariate (classical multivariate statistics) family.

Four methods, all pure Python (statsmodels / scikit-learn / scipy / numpy):

- ``manova``                — multivariate ANOVA (statsmodels MANOVA), four test stats.
- ``discriminant_analysis`` — LDA / QDA with stratified-CV accuracy (sklearn).
- ``canonical_correlation`` — CCA between two variable sets + Bartlett/Wilks sequential test.
- ``hotelling_t2``          — two-sample Hotelling's T-squared (hand-rolled T²→F).

Each handler unpacks ctx into the same local names run_analysis uses and MUTATES
summary/estimates/files/code (never rebinds). See executor/_branch_api.py. This
family file is auto-registered by branches/__init__.py (pkgutil.walk_packages).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ---------------------------------------------------------------------------
# Shared column-role helpers (local to this family — generic split rules).
# ---------------------------------------------------------------------------

def _continuous_cols(fp) -> list[str]:
    """Continuous columns, excluding the profiled unit/time columns."""
    return [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]


def _group_candidates(fp, df) -> list[str]:
    """Categorical/binary group columns, lowest-cardinality first.

    Binary first, then categoricals sorted by nunique() so a high-cardinality
    id/unit column is never auto-picked as the grouping factor.
    """
    _excl = {fp.unit_col, fp.time_col}
    bins = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    cats = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
    cats.sort(key=lambda name: int(df[name].nunique()))
    return bins + cats


# ===========================================================================
# 1. MANOVA — Multivariate analysis of variance
# ===========================================================================

@register("manova")
def _branch_manova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    cont = _continuous_cols(fp)
    # outcomes (DVs): config override else all continuous columns
    forced_out = [c for c in (cfg.get("outcomes") or []) if c in df.columns]
    outcomes = forced_out if forced_out else cont
    # factor(s): config override (group / factors) else lowest-cardinality categorical
    cands = _group_candidates(fp, df)
    forced_f = cfg.get("factors") or ([cfg["group"]] if cfg.get("group") else None)
    factors = [c for c in (forced_f or []) if c in df.columns] if forced_f else (
        [cands[0]] if cands else []
    )

    if len(outcomes) < 2:
        summary.append("MANOVA 跳过：需要 ≥2 个连续因变量（DV）。")
        return
    if not factors:
        summary.append("MANOVA 跳过：未找到分类因子（自变量）。设 config group/factors。")
        return

    # Identifier-guard every column name that enters the statsmodels formula.
    cols = outcomes + factors
    if not all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in cols):
        summary.append(
            "MANOVA 跳过：列名需为标识符式（字母/数字/. _），statsmodels formula 要求。"
        )
        return

    sub = df[cols].dropna()
    # Need n > p (DV count) and >=2 levels per factor for a non-singular fit.
    first_factor = factors[0]
    if sub[first_factor].nunique() < 2:
        summary.append("MANOVA 跳过：主因子需要 ≥2 个分组水平。")
        return
    if len(sub) <= len(outcomes) + 1:
        summary.append(f"MANOVA 跳过：样本太少（n={len(sub)} ≤ DV 数+1）。")
        return

    try:
        from statsmodels.multivariate.manova import MANOVA
        import numpy as np
        import pandas as pd

        lhs = " + ".join(outcomes)
        # Treat each factor categorically via C(...).
        rhs = " + ".join(f"C({f})" for f in factors)
        formula = f"{lhs} ~ {rhs}"
        mv = MANOVA.from_formula(formula, data=sub)
        test = mv.mv_test()

        # Pull the four multivariate stats for the FIRST factor's effect.
        # statsmodels names the effect "C(<factor>)"; the result table has rows
        # Wilks' lambda / Pillai's trace / Hotelling-Lawley trace / Roy's greatest root,
        # columns Value / Num DF / Den DF / F Value / Pr > F.
        effect_key = f"C({first_factor})"
        if effect_key not in test.results:
            # fall back: pick the first effect key that mentions the factor
            # (skips the "Intercept" row), else the last non-intercept key.
            cand = [
                kk for kk in test.results
                if kk != "Intercept" and first_factor in kk
            ] or [kk for kk in test.results if kk != "Intercept"]
            effect_key = cand[0] if cand else next(iter(test.results))
        tbl = test.results[effect_key]["stat"]
        # Standard statsmodels row labels.
        row_map = {
            "Wilks' lambda": "wilks",
            "Pillai's trace": "pillai",
            "Hotelling-Lawley trace": "hotelling_lawley",
            "Roy's greatest root": "roy",
        }
        rows = []
        for label, key in row_map.items():
            if label not in tbl.index:
                continue
            val = float(tbl.loc[label, "Value"])
            f_val = float(tbl.loc[label, "F Value"])
            num_df = float(tbl.loc[label, "Num DF"])
            den_df = float(tbl.loc[label, "Den DF"])
            p_val = float(tbl.loc[label, "Pr > F"])
            rows.append({
                "statistic": label,
                "value": val,
                "F": f_val,
                "num_df": num_df,
                "den_df": den_df,
                "p_value": p_val,
            })
            estimates[f"{key}_value"] = round(val, 6)
            estimates[f"{key}_F"] = round(f_val, 4)
            estimates[f"{key}_p"] = round(p_val, 6)

        res_df = pd.DataFrame(rows)
        res_df.to_csv(d / "manova_tests.csv", index=False, encoding="utf-8")
        files.append("manova_tests.csv")

        # Per-group DV means table (context for follow-up univariate ANOVAs).
        gmeans = sub.groupby(first_factor)[outcomes].mean()
        gmeans.to_csv(d / "manova_group_means.csv", encoding="utf-8")
        files.append("manova_group_means.csv")

        wilks_p = next((r["p_value"] for r in rows if r["statistic"] == "Wilks' lambda"), float("nan"))
        pillai_p = next((r["p_value"] for r in rows if r["statistic"] == "Pillai's trace"), float("nan"))

        # Discriminant-style scatter of the first two DVs coloured by group.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            levels = sub[first_factor].dropna().unique().tolist()
            for lv in levels:
                m = sub[first_factor] == lv
                ax.scatter(sub.loc[m, outcomes[0]], sub.loc[m, outcomes[1]],
                           s=18, alpha=0.7, label=str(lv))
            ax.set_xlabel(outcomes[0])
            ax.set_ylabel(outcomes[1])
            ax.set_title(f"MANOVA: {outcomes[0]} vs {outcomes[1]} by {first_factor}")
            ax.legend(title=first_factor, fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "manova_scatter.png", dpi=150)
            plt.close(fig)
            files.append("manova_scatter.png")
        except Exception:
            pass

        sig = "显著" if (pillai_p == pillai_p and pillai_p < 0.05) else "不显著"
        summary.append(
            f"{entry.method} 完成：{len(outcomes)} 个因变量 ~ 因子 {first_factor}"
            f"（{sub[first_factor].nunique()} 组，n={len(sub)}）。"
            f"四个多元检验统计量见 manova_tests.csv：Wilks' Λ p={wilks_p:.3g}、"
            f"Pillai's trace p={pillai_p:.3g}（Pillai 最稳健，{sig}）。"
            f"⚠ 假定多元正态 + 协方差齐性（Box's M）；不平衡/奇异协方差会失真；"
            f"Pillai 对违背假定最稳健，建议以它判读；多元显著后应跟进单变量 ANOVA / 判别分析定位是哪些 DV。"
        )
        code += [
            "from statsmodels.multivariate.manova import MANOVA",
            f"mv = MANOVA.from_formula({formula!r}, data=df)",
            "print(mv.mv_test())  # Wilks / Pillai / Hotelling-Lawley / Roy",
        ]
    except Exception as err:
        summary.append(f"MANOVA 失败：{err}")


# ===========================================================================
# 2. Discriminant analysis — LDA & QDA
# ===========================================================================

@register("discriminant_analysis")
def _branch_discriminant_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont = _continuous_cols(fp)
    # predictors: config override else all continuous columns
    forced_p = [c for c in (cfg.get("predictors") or []) if c in df.columns]
    predictors = forced_p if forced_p else cont
    # group: config override else lowest-cardinality categorical/binary
    cands = _group_candidates(fp, df)
    group = cfg["group"] if cfg.get("group") in df.columns else (cands[0] if cands else None)
    # drop a predictor that collides with the group column
    predictors = [c for c in predictors if c != group]

    if group is None:
        summary.append("判别分析跳过：未找到分类分组变量。设 config group。")
        return
    if len(predictors) < 2:
        summary.append("判别分析跳过：需要 ≥2 个连续预测变量。")
        return

    sub = df[[group] + predictors].dropna()
    y = sub[group]
    if y.nunique() < 2:
        summary.append("判别分析跳过：分组变量需要 ≥2 类。")
        return

    # n > predictors per class is needed for a stable within-class covariance.
    min_class = int(y.value_counts().min())
    if min_class <= len(predictors):
        summary.append(
            f"判别分析跳过：最小类样本 {min_class} ≤ 预测变量数 {len(predictors)}"
            f"（每类需 n > 预测变量数）。"
        )
        return

    try:
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis,
            QuadraticDiscriminantAnalysis,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import accuracy_score, confusion_matrix
        import numpy as np
        import pandas as pd

        X = sub[predictors].values
        yv = y.values

        n_classes = int(y.nunique())
        # k folds: cap by smallest class size, between 2 and 5.
        k = max(2, min(5, min_class))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

        # Honest performance: stratified-CV accuracy (NOT resubstitution).
        lda_pred_cv = cross_val_predict(LinearDiscriminantAnalysis(), X, yv, cv=skf)
        lda_cv_acc = float(accuracy_score(yv, lda_pred_cv))
        try:
            qda_pred_cv = cross_val_predict(QuadraticDiscriminantAnalysis(), X, yv, cv=skf)
            qda_cv_acc = float(accuracy_score(yv, qda_pred_cv))
        except Exception:
            qda_cv_acc = float("nan")

        # Fit LDA on the full data for transform / explained ratio / confusion matrix.
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, yv)
        # Confusion matrix from the CV predictions (honest, not resubstitution).
        labels_sorted = sorted(y.unique().tolist(), key=lambda v: str(v))
        cm = confusion_matrix(yv, lda_pred_cv, labels=labels_sorted)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{lvl}" for lvl in labels_sorted],
            columns=[f"pred_{lvl}" for lvl in labels_sorted],
        )
        cm_df.to_csv(d / "lda_confusion_matrix.csv", encoding="utf-8")
        files.append("lda_confusion_matrix.csv")

        # Explained discriminant ratio per LD axis (only present for multi-class LDA).
        evr = getattr(lda, "explained_variance_ratio_", None)
        n_ld = 0
        if evr is not None and len(evr) > 0:
            n_ld = len(evr)
            evr_df = pd.DataFrame({
                "LD": [f"LD{i+1}" for i in range(n_ld)],
                "explained_variance_ratio": np.asarray(evr, dtype=float),
            })
            evr_df.to_csv(d / "lda_explained_variance.csv", index=False, encoding="utf-8")
            files.append("lda_explained_variance.csv")
            estimates["ld1_explained_ratio"] = float(evr[0])

        estimates["lda_cv_accuracy"] = round(lda_cv_acc, 4)
        if qda_cv_acc == qda_cv_acc:
            estimates["qda_cv_accuracy"] = round(qda_cv_acc, 4)
        estimates["n_classes"] = float(n_classes)
        estimates["chance_accuracy"] = round(float(y.value_counts(normalize=True).max()), 4)

        # LD1 x LD2 projection scatter (or LD1 strip if only one discriminant axis).
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            Z = lda.transform(X)
            fig, ax = plt.subplots(figsize=(6, 5))
            for lvl in labels_sorted:
                m = yv == lvl
                if Z.shape[1] >= 2:
                    ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.7, label=str(lvl))
                else:
                    ax.scatter(Z[m, 0], np.zeros(m.sum()), s=18, alpha=0.7, label=str(lvl))
            ax.set_xlabel("LD1")
            ax.set_ylabel("LD2" if Z.shape[1] >= 2 else "")
            ax.set_title(f"LDA projection ({group})")
            ax.legend(title=group, fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "lda_projection.png", dpi=150)
            plt.close(fig)
            files.append("lda_projection.png")
        except Exception:
            pass

        chance = float(y.value_counts(normalize=True).max())
        better = "LDA" if (not (qda_cv_acc == qda_cv_acc) or lda_cv_acc >= qda_cv_acc) else "QDA"
        qda_txt = f"QDA CV 准确率={qda_cv_acc:.3f}" if qda_cv_acc == qda_cv_acc else "QDA 不可用（类内样本不足）"
        summary.append(
            f"{entry.method} 完成：{len(predictors)} 个预测变量判别 {group}"
            f"（{n_classes} 类，n={len(sub)}，{k}-折分层 CV）。"
            f"LDA CV 准确率={lda_cv_acc:.3f}（基线/最大类占比={chance:.3f}）；{qda_txt}；"
            f"较优={better}；判别轴解释比 {('LD1='+format(evr[0],'.3f')) if n_ld else '（二分类只有1轴）'}。"
            f"⚠ CV 准确率是诚实的性能估计（重代入 resubstitution 会高估，故未采用）；"
            f"LDA 假定各类协方差相等，QDA 放松此假定——已对比两者；每类需 n > 预测变量数。"
        )
        code += [
            "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis",
            "from sklearn.model_selection import StratifiedKFold, cross_val_predict",
            "from sklearn.metrics import accuracy_score",
            f"skf = StratifiedKFold(n_splits={k}, shuffle=True, random_state=0)",
            "pred = cross_val_predict(LinearDiscriminantAnalysis(), X, y, cv=skf)",
            "print('LDA CV accuracy:', accuracy_score(y, pred))",
        ]
    except Exception as err:
        summary.append(f"判别分析失败：{err}")


# ===========================================================================
# 3. Canonical correlation analysis (CCA)
# ===========================================================================

@register("canonical_correlation")
def _branch_canonical_correlation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont = _continuous_cols(fp)
    set_x = [c for c in (cfg.get("set_x") or []) if c in df.columns]
    set_y = [c for c in (cfg.get("set_y") or []) if c in df.columns]
    auto_split = not (set_x and set_y)
    if auto_split:
        # Default: split the continuous columns into two halves (DISCLOSED as arbitrary).
        half = len(cont) // 2
        set_x = cont[:half]
        set_y = cont[half:]

    if len(set_x) < 1 or len(set_y) < 1 or (len(set_x) + len(set_y)) < 3:
        summary.append(
            "典型相关分析跳过：需要两组连续变量（每组 ≥1，合计 ≥3）。设 config set_x/set_y。"
        )
        return

    sub = df[set_x + set_y].dropna()
    p, q = len(set_x), len(set_y)
    n = len(sub)
    n_pairs = min(p, q)
    # Need n > p + q + 1 for a non-degenerate covariance / sequential test.
    if n <= p + q + 1:
        summary.append(f"典型相关分析跳过：样本太少（n={n} ≤ 变量总数+1）。")
        return

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        Xc = sub[set_x].values.astype(float)
        Yc = sub[set_y].values.astype(float)
        # Centre each set (canonical correlations are invariant to centring/scaling).
        Xc = Xc - Xc.mean(axis=0)
        Yc = Yc - Yc.mean(axis=0)

        # Canonical correlations via the standard whitened-cross-covariance SVD.
        # Whiten X and Y by their (symmetric) inverse-sqrt covariance, then the
        # singular values of the whitened cross-covariance are the canonical
        # correlations rho_k in [0, 1]. (Mardia, Kent & Bibby 1979, Ch. 10.)
        Sxx = (Xc.T @ Xc) / (n - 1)
        Syy = (Yc.T @ Yc) / (n - 1)
        Sxy = (Xc.T @ Yc) / (n - 1)

        def _inv_sqrt(S):
            w, V = np.linalg.eigh(S)
            w = np.clip(w, 1e-12, None)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

        Sxx_is = _inv_sqrt(Sxx)
        Syy_is = _inv_sqrt(Syy)
        K = Sxx_is @ Sxy @ Syy_is
        U_, svals, Vt_ = np.linalg.svd(K, full_matrices=False)
        rho = np.clip(svals[:n_pairs], 0.0, 1.0)

        # ----- Bartlett / Wilks sequential test ---------------------------
        # H0(k): canonical correlations rho_{k+1}, ..., rho_m are all zero, i.e.
        # at most k significant pairs.  Wilks' Lambda_k = prod_{i=k+1}^{m} (1 - rho_i^2).
        # Bartlett's chi-square approximation (m = min(p,q)):
        #   chi2_k = -(n - 1 - (p + q + 1)/2) * ln(Lambda_k)
        # with df_k = (p - k) * (q - k).   (Bartlett 1941; Mardia et al. 1979 §10.2.)
        # k runs 0..m-1; k=0 tests "all pairs zero".
        m = n_pairs
        const = (n - 1) - (p + q + 1) / 2.0
        rho2 = rho ** 2
        seq_rows = []
        for k in range(m):
            lam = float(np.prod(1.0 - rho2[k:]))  # product over i = k..m-1 (0-based)
            lam = max(lam, 1e-300)
            chi2 = -const * np.log(lam)
            dfk = (p - k) * (q - k)
            pval = float(stats.chi2.sf(chi2, dfk)) if dfk > 0 else float("nan")
            seq_rows.append({
                "test": f"pairs {k+1}..{m} (after removing first {k})",
                "canonical_corr": float(rho[k]),
                "wilks_lambda": lam,
                "chi_square": float(chi2),
                "df": int(dfk),
                "p_value": pval,
            })

        seq_df = pd.DataFrame(seq_rows)
        seq_df.to_csv(d / "cca_sequential_test.csv", index=False, encoding="utf-8")
        files.append("cca_sequential_test.csv")

        # Canonical correlations table.
        corr_df = pd.DataFrame({
            "pair": [f"CC{i+1}" for i in range(m)],
            "canonical_correlation": rho,
            "squared": rho2,
        })
        corr_df.to_csv(d / "cca_correlations.csv", index=False, encoding="utf-8")
        files.append("cca_correlations.csv")

        # Canonical variate scores for the FIRST pair: U1 = Xc @ a1, V1 = Yc @ b1
        # where a1 = Sxx^{-1/2} u1, b1 = Syy^{-1/2} v1 (first SVD vectors).
        a1 = Sxx_is @ U_[:, 0]
        b1 = Syy_is @ Vt_[0, :]
        U1 = Xc @ a1
        V1 = Yc @ b1

        estimates["first_canonical_corr"] = round(float(rho[0]), 6)
        estimates["n_pairs"] = float(m)
        estimates["first_pair_p"] = round(float(seq_rows[0]["p_value"]), 6)
        # number of significant successive pairs (p<0.05 sequentially from the top)
        n_sig = 0
        for r in seq_rows:
            if r["p_value"] == r["p_value"] and r["p_value"] < 0.05:
                n_sig += 1
            else:
                break
        estimates["n_significant_pairs"] = float(n_sig)

        # Scatter of first canonical variate pair U1 vs V1.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(U1, V1, s=18, alpha=0.7)
            ax.set_xlabel("U1 (X canonical variate)")
            ax.set_ylabel("V1 (Y canonical variate)")
            ax.set_title(f"First canonical variate pair (r={rho[0]:.3f})")
            fig.tight_layout()
            fig.savefig(d / "cca_first_pair.png", dpi=150)
            plt.close(fig)
            files.append("cca_first_pair.png")
        except Exception:
            pass

        split_note = (
            "⚠ 默认把连续列对半切成两组 set_x/set_y 是任意的，应按概念用 config set_x/set_y 指定两个变量集。"
            if auto_split else ""
        )
        summary.append(
            f"{entry.method} 完成：X 组 {p} 变量 vs Y 组 {q} 变量（n={n}，{m} 对典型变量）。"
            f"第一典型相关 r={rho[0]:.3f}（p={seq_rows[0]['p_value']:.3g}，Bartlett/Wilks 序贯检验）；"
            f"序贯检验判定 {n_sig} 对显著（详见 cca_sequential_test.csv）。"
            f"{split_note}"
            f"⚠ CCA 找两组的线性组合使其相关最大；通常只有前几对可解释；"
            f"序贯检验依赖多元正态假定。"
        )
        code += [
            "import numpy as np; from scipy import stats",
            "# whiten X,Y by inverse-sqrt covariance; SVD of cross-cov -> canonical corrs",
            "# Wilks Lambda_k = prod_{i>k}(1-rho_i^2); chi2 = -(n-1-(p+q+1)/2)*ln(Lambda_k)",
            "# df_k = (p-k)*(q-k)  # Bartlett 1941 sequential test",
        ]
    except Exception as err:
        summary.append(f"典型相关分析失败：{err}")


# ===========================================================================
# 4. Hotelling's T-squared — two-sample test of mean vectors
# ===========================================================================

@register("hotelling_t2")
def _branch_hotelling_t2(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont = _continuous_cols(fp)
    forced_out = [c for c in (cfg.get("outcomes") or []) if c in df.columns]
    outcomes = forced_out if forced_out else cont
    cands = _group_candidates(fp, df)
    group = cfg["group"] if cfg.get("group") in df.columns else (cands[0] if cands else None)
    outcomes = [c for c in outcomes if c != group]

    if group is None:
        summary.append("Hotelling T² 跳过：未找到 2 水平分组变量。设 config group。")
        return
    if len(outcomes) < 2:
        summary.append("Hotelling T² 跳过：需要 ≥2 个连续结果变量。")
        return

    sub = df[[group] + outcomes].dropna()
    levels = sub[group].dropna().unique().tolist()
    if len(levels) != 2:
        if len(levels) > 2:
            summary.append(
                f"Hotelling T² 跳过：分组有 {len(levels)} 个水平（>2）；多组请用 MANOVA。"
                f" 可用 config group 选 2 水平变量。"
            )
        else:
            summary.append("Hotelling T² 跳过：分组变量需恰好 2 个水平。")
        return

    g1 = sub.loc[sub[group] == levels[0], outcomes].values.astype(float)
    g2 = sub.loc[sub[group] == levels[1], outcomes].values.astype(float)
    n1, n2 = len(g1), len(g2)
    p = len(outcomes)
    # Pooled covariance is invertible only if n1 + n2 - 2 >= p.
    if n1 < 2 or n2 < 2:
        summary.append("Hotelling T² 跳过：每组至少需 2 个观测。")
        return
    if (n1 + n2 - 2) < p:
        summary.append(
            f"Hotelling T² 跳过：n1+n2-2={n1 + n2 - 2} < p={p}，合并协方差不可逆。"
        )
        return

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        mean1 = g1.mean(axis=0)
        mean2 = g2.mean(axis=0)
        diff = mean1 - mean2

        # Pooled within-group covariance (unbiased, ddof=1 per group):
        #   S_pooled = ((n1-1) S1 + (n2-1) S2) / (n1 + n2 - 2)
        S1 = np.cov(g1, rowvar=False, ddof=1)
        S2 = np.cov(g2, rowvar=False, ddof=1)
        S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
        S_inv = np.linalg.inv(S_pooled)

        # Two-sample Hotelling's T-squared:
        #   T^2 = (n1*n2 / (n1+n2)) * diff^T S_pooled^{-1} diff
        # F transform (Anderson 2003; Mardia et al. 1979 §3.6):
        #   F = T^2 * (n1+n2-p-1) / ((n1+n2-2)*p),   df = (p, n1+n2-p-1)
        t2 = (n1 * n2 / (n1 + n2)) * float(diff @ S_inv @ diff)
        df1 = p
        df2 = n1 + n2 - p - 1
        f_stat = t2 * df2 / ((n1 + n2 - 2) * p)
        p_value = float(stats.f.sf(f_stat, df1, df2))

        estimates["T2"] = round(float(t2), 6)
        estimates["F"] = round(float(f_stat), 6)
        estimates["df1"] = float(df1)
        estimates["df2"] = float(df2)
        estimates["p_value"] = round(p_value, 6)

        # Per-variable mean differences (context: which DV drives the separation).
        md_df = pd.DataFrame({
            "variable": outcomes,
            f"mean_{levels[0]}": mean1,
            f"mean_{levels[1]}": mean2,
            "mean_diff": diff,
        })
        md_df.to_csv(d / "hotelling_mean_diffs.csv", index=False, encoding="utf-8")
        files.append("hotelling_mean_diffs.csv")

        test_df = pd.DataFrame([{
            "T2": t2, "F": f_stat, "df1": df1, "df2": df2,
            "p_value": p_value, "n1": n1, "n2": n2, "p_vars": p,
        }])
        test_df.to_csv(d / "hotelling_test.csv", index=False, encoding="utf-8")
        files.append("hotelling_test.csv")

        # Scatter of the first two outcomes coloured by group, with group means.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(g1[:, 0], g1[:, 1], s=18, alpha=0.6, label=str(levels[0]))
            ax.scatter(g2[:, 0], g2[:, 1], s=18, alpha=0.6, label=str(levels[1]))
            ax.scatter([mean1[0]], [mean1[1]], marker="X", s=120, c="black")
            ax.scatter([mean2[0]], [mean2[1]], marker="X", s=120, c="black")
            ax.set_xlabel(outcomes[0])
            ax.set_ylabel(outcomes[1])
            ax.set_title(f"Hotelling T2: {outcomes[0]} vs {outcomes[1]} by {group}")
            ax.legend(title=group, fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "hotelling_scatter.png", dpi=150)
            plt.close(fig)
            files.append("hotelling_scatter.png")
        except Exception:
            pass

        verdict = "拒绝" if p_value < 0.05 else "不拒绝"
        summary.append(
            f"{entry.method} 完成：{p} 个结果变量在 {group} 两组"
            f"（{levels[0]} n={n1} vs {levels[1]} n={n2}）的均值向量检验。"
            f"T²={t2:.4f}，F={f_stat:.4f}，df=({df1},{df2})，p={p_value:.3g}；"
            f"{verdict}「两组均值向量相等」的原假设（α=0.05）。"
            f"⚠ 假定多元正态 + 两组协方差相等；>2 组请用 MANOVA；"
            f"需 n1+n2-2 ≥ p 以保证合并协方差可逆。"
        )
        code += [
            "import numpy as np; from scipy import stats",
            "diff = g1.mean(0) - g2.mean(0)",
            "Sp = ((n1-1)*np.cov(g1,rowvar=False)+(n2-1)*np.cov(g2,rowvar=False))/(n1+n2-2)",
            "T2 = (n1*n2/(n1+n2)) * diff @ np.linalg.inv(Sp) @ diff",
            "F = T2 * (n1+n2-p-1) / ((n1+n2-2)*p)  # df = (p, n1+n2-p-1)",
            "p_value = stats.f.sf(F, p, n1+n2-p-1)",
        ]
    except Exception as err:
        summary.append(f"Hotelling T² 失败：{err}")
