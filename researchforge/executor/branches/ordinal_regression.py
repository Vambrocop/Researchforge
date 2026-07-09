"""Branch handlers for the ordinal-outcome regression family.

Three methods for an ORDINAL outcome (a small ordered scale, 3..~10 levels — e.g.
Likert satisfaction, adoption-intensity tiers), all via statsmodels'
``OrderedModel`` and binary-logit Brant approximation. PURE PYTHON (statsmodels /
numpy / scipy); NO R.

  * proportional_odds_logit — proportional-odds ordinal logistic regression
        (cumulative logit). Per-predictor LOG-ODDS + ODDS RATIOS exp(coef) with
        95% CI + p, the K-1 thresholds, log-likelihood, McFadden pseudo-R², and an
        overall LR test vs the null. Forest plot of odds ratios (log scale).
  * ordered_probit — same model with ``distr="probit"``; coefficients are on the
        latent-normal scale (coef + CI + p, NOT odds ratios), thresholds, loglik,
        pseudo-R². Coefficient plot.
  * brant_test — approximate Brant (1990) test of the proportional-odds
        (parallel-lines) assumption via SEPARATE binary logits 1[y > level_k];
        per-predictor slope spread across cut points + omnibus chi-square + per-
        predictor p (small p ⇒ proportional-odds VIOLATED). Slope-by-cutpoint plot.

NOTE on ids: the engine already ships a basic ``ordered_logit`` (in
``branches/statistics.py``). To avoid the duplicate-handler collision this richer
proportional-odds logit is registered as ``proportional_odds_logit``.

Each handler resolves the ordinal outcome (config["outcome"] else a count/categorical
column with 3..10 ordered levels) and continuous/binary predictors, degrades honestly
(outcome not ordinal / <3 levels / too many levels / too few rows / sparse cut /
singular / import missing -> append a Chinese "<方法>跳过：<原因>" and return — never
crash/fabricate), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels), fills float
``estimates``, appends a Chinese ``summary`` ending with ⚠ disclosures, and MUTATES ctx
(never rebinds). See executor/_branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_MIN_ROWS = 30
_MIN_LEVELS = 3
_MAX_LEVELS = 10
_MAX_PREDICTORS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Shared outcome/predictors resolution. Mirrors the regression-family convention
# but the outcome must be ORDINAL (a small ordered scale). Returns
# (outcome, predictors, problem_msg). When problem_msg is not None the caller
# appends it to summary and returns (honest degrade).
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_ordinal(ctx: Ctx, method: str):
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    excl = {fp.unit_col, fp.time_col}

    if fp.n_rows < _MIN_ROWS:
        return None, None, f"{method}跳过：样本量过小（{fp.n_rows} 行 < {_MIN_ROWS}），有序回归不可靠。"

    # outcome: config override (validate it is genuinely low-cardinality ordered),
    # else the first count/categorical column with 3..10 ordered levels (count first
    # — its numeric level order is unambiguous). NOTE: profiler's "id trap" puts a
    # distinct-integer column in kind=="id"; an outcome with 3..10 levels won't be id.
    outcome = None
    cfg_out = cfg.get("outcome")
    if cfg_out is not None:
        if cfg_out not in df.columns:
            return None, None, f"{method}跳过：config 指定的结果列 {cfg_out!r} 不在数据中。"
        nun = int(df[cfg_out].dropna().nunique())
        if nun < _MIN_LEVELS:
            return None, None, f"{method}跳过：结果 {cfg_out} 仅 {nun} 个等级（<{_MIN_LEVELS}），不是有序结果。"
        if nun > _MAX_LEVELS:
            return None, None, (
                f"{method}跳过：结果 {cfg_out} 有 {nun} 个不同值（>{_MAX_LEVELS}），"
                "更像连续/ID 而非有序量表（如确为有序请先粗化分级）。"
            )
        outcome = cfg_out
    else:
        ord_cols = [
            c for c in fp.columns
            if c.kind in {"count", "categorical"}
            and _MIN_LEVELS <= c.n_unique <= _MAX_LEVELS
            and c.name not in excl
        ]
        ord_cols.sort(key=lambda c: 0 if c.kind == "count" else 1)
        # shared resolver on the sorted candidates: high-confidence detected outcome >
        # first non-treatment-named (an ordinal 'dose' is the regressor, not the outcome
        # — {dose, severity} models severity) > first. config already handled above.
        if ord_cols:
            from researchforge.executor.run import resolve_outcome

            outcome = resolve_outcome(fp, cfg, [c.name for c in ord_cols])
        else:
            outcome = None
        if outcome is None:
            return None, None, (
                f"{method}跳过：未找到有序结果变量（{_MIN_LEVELS}–{_MAX_LEVELS} 个有序等级，"
                "如 Likert 量表）。可用 config={\"outcome\":\"<列>\"} 指定。"
            )

    # predictors: config override (filter to existing, non-outcome columns) else
    # continuous/binary/count columns. Same convention as the regression family.
    exclude = {outcome, fp.unit_col, fp.time_col}
    cfg_pred = cfg.get("predictors")
    if cfg_pred:
        predictors = [p for p in cfg_pred if p in df.columns and p not in exclude]
    else:
        predictors = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
        ]
    predictors = predictors[:_MAX_PREDICTORS]
    if not predictors:
        return None, None, f"{method}跳过：没有可用的连续/二值预测变量。"

    return outcome, predictors, None


def _ordered_design(df, outcome, predictors):
    """Drop NA rows on outcome+predictors, build an ordered Categorical y and a
    numeric float predictor frame X. Raises if too few rows or a predictor is
    constant (singular). Returns (y_cat, levels, X, n)."""
    import pandas as pd

    sub = df[[outcome] + predictors].dropna()
    n = len(sub)
    if n < _MIN_ROWS:
        raise RuntimeError(f"删除缺失后仅 {n} 行（<{_MIN_ROWS}）")
    yc = pd.Categorical(sub[outcome], ordered=True)
    levels = list(yc.categories)
    if len(levels) < _MIN_LEVELS:
        raise RuntimeError(f"删除缺失后结果仅 {len(levels)} 个等级")
    X = sub[predictors].apply(pd.to_numeric, errors="coerce").astype(float)
    if X.isna().any().any():
        raise RuntimeError("预测变量含非数值，无法转为数值矩阵")
    bad = [p for p in predictors if float(X[p].std(ddof=0)) == 0.0]
    if bad:
        raise RuntimeError(f"预测变量常数（无方差）：{bad}")
    return yc, levels, X, n


def _mcfadden(model, y_cat, n_levels):
    """McFadden pseudo-R² and LR test vs the intercept-only (proportions) null.
    Returns (pseudo_r2, llr, llr_p, df_diff). Best-effort; NaN on failure."""
    import numpy as np
    from scipy import stats

    ll_full = float(model.llf)
    # Null log-likelihood: only the K-1 thresholds (= empirical level proportions).
    counts = np.array([np.sum(np.asarray(y_cat.codes) == k) for k in range(n_levels)], dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    ll_null = float(np.sum(counts[counts > 0] * np.log(p)))
    pseudo_r2 = 1.0 - ll_full / ll_null if ll_null != 0 else float("nan")
    df_diff = int(len(model.params) - (n_levels - 1))  # = number of predictor slopes
    llr = 2.0 * (ll_full - ll_null)
    llr_p = float(stats.chi2.sf(llr, df_diff)) if df_diff > 0 else float("nan")
    return pseudo_r2, float(llr), llr_p, df_diff


# ─────────────────────────────────────────────────────────────────────────────
# (A) proportional_odds_logit — proportional-odds ordinal logistic regression
# ─────────────────────────────────────────────────────────────────────────────
@register("proportional_odds_logit")
def _branch_proportional_odds_logit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "比例优势有序 Logit"

    outcome, predictors, problem = _resolve_ordinal(ctx, method)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        yc, levels, X, n = _ordered_design(df, outcome, predictors)
        n_levels = len(levels)
        model = OrderedModel(yc, X, distr="logit").fit(method="bfgs", disp=False)

        params, bse, pvals = model.params, model.bse, model.pvalues
        ci = model.conf_int()  # DataFrame [0,1] = 2.5% / 97.5%
        ci.columns = ["ci_low", "ci_high"]

        # Slope rows are indexed by the bare predictor names; the remaining rows are
        # the K-1 thresholds (statsmodels labels them like "1/2", "2/3" ...).
        slope_idx = [p for p in predictors if p in params.index]
        thresh_idx = [i for i in params.index if i not in slope_idx]

        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")

        # Per-predictor coefficient table: log-odds + odds ratios (exp) with 95% CI + p.
        rows = []
        for v in slope_idx:
            rows.append({
                "term": v,
                "log_odds": float(params[v]),
                "odds_ratio": float(np.exp(params[v])),
                "or_ci_low": float(np.exp(ci.loc[v, "ci_low"])),
                "or_ci_high": float(np.exp(ci.loc[v, "ci_high"])),
                "std_err": float(bse[v]),
                "z": float(model.tvalues[v]),
                "p_value": float(pvals[v]),
            })
        coef_df = pd.DataFrame(rows)
        # Threshold/cutpoint rows. statsmodels OrderedModel stores thresholds in an
        # UNCONSTRAINED optimizer space ([c1, log(c2-c1), log(c3-c2), …]) to enforce
        # monotonicity, so .params/.bse for threshold rows are in THAT space — only the
        # first is a real cutpoint, the rest are log-increments. Transform to the true
        # ordered cutpoints. The library SE/z/p are on the increment scale (not the
        # cutpoint scale), so we report the cutpoint VALUE only (a cutpoint SE would need
        # the delta method) — disclosed in the summary.
        raw_thr = np.asarray(params[thresh_idx], dtype=float)
        try:
            cuts_full = np.asarray(model.model.transform_threshold_params(raw_thr), dtype=float)
            cutpoints = [c for c in cuts_full if np.isfinite(c)]
        except Exception:
            cutpoints = list(raw_thr)
        if len(cutpoints) != len(thresh_idx):
            cutpoints = list(raw_thr)
        thr_rows = []
        for t, c in zip(thresh_idx, cutpoints):
            thr_rows.append({
                "term": f"cutpoint[{t}]",
                "log_odds": float(c),  # the true ordered cutpoint (latent-logit scale)
                "odds_ratio": float("nan"),
                "or_ci_low": float("nan"),
                "or_ci_high": float("nan"),
                "std_err": float("nan"),
                "z": float("nan"),
                "p_value": float("nan"),
            })
        out_df = pd.concat([coef_df, pd.DataFrame(thr_rows)], ignore_index=True)
        out_df.to_csv(d / "coefficients.csv", index=False, encoding="utf-8")
        files.append("coefficients.csv")

        pseudo_r2, llr, llr_p, _df = _mcfadden(model, yc, n_levels)

        # Forest plot of odds ratios on a LOG scale (ENGLISH labels).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            terms = list(coef_df["term"])
            orr = coef_df["odds_ratio"].to_numpy()
            lo = coef_df["or_ci_low"].to_numpy()
            hi = coef_df["or_ci_high"].to_numpy()
            ypos = np.arange(len(terms))
            fig, ax = plt.subplots(figsize=(6, 0.6 * len(terms) + 1.6))
            ax.errorbar(orr, ypos, xerr=[orr - lo, hi - orr], fmt="o", color="#4C72B0", capsize=3)
            ax.axvline(1.0, color="grey", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_yticks(ypos)
            ax.set_yticklabels(terms)
            ax.set_xlabel("odds ratio (log scale, 95% CI)")
            ax.set_title("Proportional-odds ordinal logit: odds ratios")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(d / "odds_ratios.png", dpi=150)
            plt.close(fig)
            files.append("odds_ratios.png")
        except Exception:
            pass

        max_abs_or_log = float(np.max(np.abs([params[v] for v in slope_idx]))) if slope_idx else 0.0
        estimates["loglik"] = float(model.llf)
        estimates["llr_p"] = float(llr_p)
        estimates["pseudo_r2"] = float(pseudo_r2)
        estimates["n"] = float(n)
        estimates["n_thresholds"] = float(len(thresh_idx))
        estimates["max_abs_or_log"] = max_abs_or_log
        # namespaced with a "coef_" prefix so a predictor literally named e.g. "n" or
        # "loglik" cannot collide with (overwrite) the bookkeeping keys above.
        for v in slope_idx:  # per-predictor log-odds for downstream/sign checks
            estimates[f"coef_{v}"] = float(params[v])

        is_text = df[outcome].dtype == object or str(df[outcome].dtype) == "string"
        lvl_note = f"（等级顺序按排序取为 {levels}）" if is_text else "（等级按排序取序）"
        key = ""
        if slope_idx:
            v0 = slope_idx[0]
            key = (f"，关键 {v0} OR={float(np.exp(params[v0])):.3f} "
                   f"(p={float(pvals[v0]):.3g})")
        summary.append(
            f"{entry.method} 完成：有序结果 {outcome}（{n_levels} 级，n={n}），"
            f"{len(slope_idx)} 个预测变量{lvl_note}{key}；"
            f"McFadden 伪R²={pseudo_r2:.3f}，整体 LR 检验 p={llr_p:.3g}。"
            "⚠ 比例优势（平行线）假定：所有阈值共用一组斜率——请用 brant_test 检验是否成立；"
            "⚠ 等级顺序按排序取定（非外部验证，若顺序不符请重编码）；config 可指定 outcome/predictors。"
        )
        code += [
            "from statsmodels.miscmodels.ordinal_model import OrderedModel",
            f"yc = pd.Categorical(df[{outcome!r}], ordered=True)",
            f"m = OrderedModel(yc, df[{predictors!r}], distr='logit').fit(method='bfgs')",
            "import numpy as np; OR = np.exp(m.params)  # odds ratios; thresholds = K-1 cutpoints",
        ]
    except Exception as err:
        summary.append(f"{method}跳过：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) ordered_probit — same model, latent-normal (probit) link
# ─────────────────────────────────────────────────────────────────────────────
@register("ordered_probit")
def _branch_ordered_probit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "有序 Probit"

    outcome, predictors, problem = _resolve_ordinal(ctx, method)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        yc, levels, X, n = _ordered_design(df, outcome, predictors)
        n_levels = len(levels)
        model = OrderedModel(yc, X, distr="probit").fit(method="bfgs", disp=False)

        params, bse, pvals = model.params, model.bse, model.pvalues
        ci = model.conf_int()
        ci.columns = ["ci_low", "ci_high"]
        slope_idx = [p for p in predictors if p in params.index]
        thresh_idx = [i for i in params.index if i not in slope_idx]

        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")

        # Latent-scale coefficients (NOT odds ratios) + 95% CI + p, then thresholds.
        rows = []
        for v in slope_idx:
            rows.append({
                "term": v,
                "coef": float(params[v]),
                "ci_low": float(ci.loc[v, "ci_low"]),
                "ci_high": float(ci.loc[v, "ci_high"]),
                "std_err": float(bse[v]),
                "z": float(model.tvalues[v]),
                "p_value": float(pvals[v]),
            })
        coef_df = pd.DataFrame(rows)
        # threshold rows -> true ordered cutpoints (OrderedModel stores them as
        # [c1, log(c2-c1), …]; transform). Library SE/z/p are increment-scale, omitted.
        raw_thr = np.asarray(params[thresh_idx], dtype=float)
        try:
            cuts_full = np.asarray(model.model.transform_threshold_params(raw_thr), dtype=float)
            cutpoints = [c for c in cuts_full if np.isfinite(c)]
        except Exception:
            cutpoints = list(raw_thr)
        if len(cutpoints) != len(thresh_idx):
            cutpoints = list(raw_thr)
        thr_rows = [{
            "term": f"cutpoint[{t}]",
            "coef": float(c),  # true ordered cutpoint (latent-normal scale)
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "std_err": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
        } for t, c in zip(thresh_idx, cutpoints)]
        out_df = pd.concat([coef_df, pd.DataFrame(thr_rows)], ignore_index=True)
        out_df.to_csv(d / "coefficients.csv", index=False, encoding="utf-8")
        files.append("coefficients.csv")

        pseudo_r2, llr, llr_p, _df = _mcfadden(model, yc, n_levels)

        # Coefficient plot (latent scale, 95% CI, ENGLISH labels).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            terms = list(coef_df["term"])
            co = coef_df["coef"].to_numpy()
            lo = coef_df["ci_low"].to_numpy()
            hi = coef_df["ci_high"].to_numpy()
            ypos = np.arange(len(terms))
            fig, ax = plt.subplots(figsize=(6, 0.6 * len(terms) + 1.6))
            ax.errorbar(co, ypos, xerr=[co - lo, hi - co], fmt="o", color="#55A868", capsize=3)
            ax.axvline(0.0, color="grey", ls="--", lw=1)
            ax.set_yticks(ypos)
            ax.set_yticklabels(terms)
            ax.set_xlabel("coefficient (latent-normal scale, 95% CI)")
            ax.set_title("Ordered probit: latent-scale coefficients")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(d / "coefficients.png", dpi=150)
            plt.close(fig)
            files.append("coefficients.png")
        except Exception:
            pass

        max_abs_coef = float(np.max(np.abs([params[v] for v in slope_idx]))) if slope_idx else 0.0
        estimates["loglik"] = float(model.llf)
        estimates["llr_p"] = float(llr_p)
        estimates["pseudo_r2"] = float(pseudo_r2)
        estimates["n"] = float(n)
        estimates["n_thresholds"] = float(len(thresh_idx))
        estimates["max_abs_coef"] = max_abs_coef
        # namespaced with a "coef_" prefix so a predictor literally named e.g. "n" or
        # "loglik" cannot collide with (overwrite) the bookkeeping keys above.
        for v in slope_idx:
            estimates[f"coef_{v}"] = float(params[v])

        key = ""
        if slope_idx:
            v0 = slope_idx[0]
            key = f"，关键 {v0} 系数={float(params[v0]):.3f} (p={float(pvals[v0]):.3g})"
        summary.append(
            f"{entry.method} 完成：有序结果 {outcome}（{n_levels} 级，n={n}），"
            f"{len(slope_idx)} 个预测变量（等级按排序取序）{key}；"
            f"McFadden 伪R²={pseudo_r2:.3f}，整体 LR 检验 p={llr_p:.3g}。"
            "⚠ Probit 系数在隐变量（latent-normal）尺度上，不是优势比（OR），符号/显著性可解读、量级不可直接当 OR；"
            "⚠ 同样依赖比例/平行线假定（用 brant_test 查）；等级顺序按排序取定；config 可指定 outcome/predictors。"
        )
        code += [
            "from statsmodels.miscmodels.ordinal_model import OrderedModel",
            f"yc = pd.Categorical(df[{outcome!r}], ordered=True)",
            f"m = OrderedModel(yc, df[{predictors!r}], distr='probit').fit(method='bfgs')",
            "print(m.summary())  # coefficients are on the latent-normal scale (NOT odds ratios)",
        ]
    except Exception as err:
        summary.append(f"{method}跳过：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) brant_test — approximate Brant (1990) parallel-lines test via binary logits
# ─────────────────────────────────────────────────────────────────────────────
@register("brant_test")
def _branch_brant_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "Brant 比例优势检验"

    outcome, predictors, problem = _resolve_ordinal(ctx, method)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from scipy import stats

        yc, levels, X, n = _ordered_design(df, outcome, predictors)
        n_levels = len(levels)
        codes = np.asarray(yc.codes)  # 0..K-1
        Xmat = sm.add_constant(X.to_numpy(), has_constant="add")
        cuts = list(range(n_levels - 1))  # cut k -> 1[y > level_k]

        # Fit a separate binary logit at each cut point; collect per-cut slopes
        # (excluding intercept) and their covariance. Degrade if any cut is too
        # sparse to fit (a side with too few observations / no variation).
        slopes = {}      # cut -> np.array of predictor slopes
        vcovs = {}       # cut -> covariance of slopes
        sparse_cuts = []
        for k in cuts:
            ybin = (codes > k).astype(int)
            n1 = int(ybin.sum())
            n0 = int(len(ybin) - n1)
            if min(n0, n1) < max(10, len(predictors) + 2):
                sparse_cuts.append((k, n0, n1))
                continue
            try:
                fit = sm.Logit(ybin, Xmat).fit(disp=False, method="bfgs", maxiter=200)
                beta = np.asarray(fit.params)[1:]            # drop intercept
                cov = np.asarray(fit.cov_params())[1:, 1:]
                if not np.all(np.isfinite(beta)):
                    sparse_cuts.append((k, n0, n1))
                    continue
                slopes[k] = beta
                vcovs[k] = cov
            except Exception:
                sparse_cuts.append((k, n0, n1))

        usable = sorted(slopes.keys())
        if len(usable) < 2:
            summary.append(
                f"{method}跳过：可用切点 < 2（需要至少 2 个非稀疏切点比较斜率）。"
                "比例优势检验对每个切点的两组都需足够样本——某些切点过稀疏。"
            )
            return

        p = len(predictors)
        # Per-predictor Wald test that the cut-specific slopes are all equal:
        # for predictor j, stack its slope across usable cuts and test equality
        # (chi² with df = #cuts - 1) using a diagonal (independent-fit) approximation
        # of the slope variances. This is the simpler per-predictor variant of Brant.
        per_pred_rows = []
        per_pred_chi2 = {}
        per_pred_p = {}
        for j, name in enumerate(predictors):
            bj = np.array([slopes[k][j] for k in usable])
            vj = np.array([vcovs[k][j, j] for k in usable])
            vj = np.where(vj > 0, vj, np.nan)
            w = 1.0 / vj
            if not np.all(np.isfinite(w)):
                chi2 = float("nan"); pval = float("nan"); dfj = len(usable) - 1
            else:
                bbar = float(np.sum(w * bj) / np.sum(w))   # precision-weighted mean
                chi2 = float(np.sum(w * (bj - bbar) ** 2))  # ~ chi²(#cuts-1)
                dfj = len(usable) - 1
                pval = float(stats.chi2.sf(chi2, dfj)) if dfj > 0 else float("nan")
            per_pred_chi2[name] = chi2
            per_pred_p[name] = pval
            row = {"predictor": name, "chi2": chi2, "df": float(len(usable) - 1), "p_value": pval}
            for k in usable:
                row[f"slope_cut_{k}"] = float(slopes[k][j])
            per_pred_rows.append(row)

        slope_df = pd.DataFrame(per_pred_rows)
        slope_df.to_csv(d / "brant_slopes.csv", index=False, encoding="utf-8")
        files.append("brant_slopes.csv")

        # Omnibus test: sum the per-predictor chi² (independent under H0) -> global
        # chi² with df = p * (#cuts - 1). Small global p ⇒ proportional odds VIOLATED.
        valid = [c for c in per_pred_chi2.values() if c == c]  # drop NaN
        global_chi2 = float(np.sum(valid)) if valid else float("nan")
        global_df = float(len(valid) * (len(usable) - 1))
        global_p = float(stats.chi2.sf(global_chi2, global_df)) if (valid and global_df > 0) else float("nan")
        n_violations = float(sum(1 for v in per_pred_p.values() if v == v and v < 0.05))

        # Plot: per-predictor slopes across cut points (ENGLISH labels).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 4))
            for j, name in enumerate(predictors):
                ys = [slopes[k][j] for k in usable]
                ax.plot(usable, ys, marker="o", label=name)
            ax.set_xlabel("cut point k  (binary logit of 1[y > level_k])")
            ax.set_ylabel("estimated slope")
            ax.set_xticks(usable)
            ax.set_title("Brant test: per-predictor slopes across cut points")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "brant_slopes.png", dpi=150)
            plt.close(fig)
            files.append("brant_slopes.png")
        except Exception:
            pass

        estimates["global_chi2"] = global_chi2
        estimates["global_p"] = global_p
        estimates["global_df"] = global_df
        estimates["n_violations"] = n_violations
        estimates["n_predictors"] = float(p)

        verdict = (
            "比例优势假定被拒绝（p<0.05）——平行线不成立" if (global_p == global_p and global_p < 0.05)
            else "未拒绝比例优势假定（p≥0.05）——平行线大体成立" if global_p == global_p
            else "全局检验不可用"
        )
        sparse_note = ""
        if sparse_cuts:
            sl = ", ".join(f"切点{k}(0组{n0}/1组{n1})" for k, n0, n1 in sparse_cuts)
            sparse_note = f"⚠ 已跳过稀疏切点：{sl}（两组样本不足，Brant 二元 logit 近似需每侧足够样本）；"

        (d / "brant_summary.txt").write_text(
            f"Brant (1990) 比例优势（平行线）检验——分切点二元 logit 近似：有序结果 {outcome}"
            f"（{n_levels} 级，n={n}），{p} 个预测变量\n"
            f"用到的切点：{usable}（每个切点拟合 1[y>level_k] 的二元 logit）\n"
            f"全局卡方={round(global_chi2, 4)}，df={int(global_df)}，p={round(global_p, 4) if global_p == global_p else 'NaN'}"
            f" —— {verdict}\n"
            f"违反比例优势的预测变量数（p<0.05）：{int(n_violations)}/{p}\n"
            + (sparse_note + "\n" if sparse_note else "")
            + "注：这是 Brant 的近似实现——对每个切点独立拟合二元 logit，比较各预测变量斜率是否跨切点相等"
            "（逐预测变量 Wald，用独立拟合的对角方差近似；全局检验为各预测变量卡方之和）。"
            "若检验显著（拒绝），说明该预测变量的效应随切点变化，应改用非比例/广义有序 logit（partial/generalized）或多项 logit。\n\n"
            "逐预测变量斜率（跨切点）：\n" + slope_df.round(4).to_string(index=False),
            encoding="utf-8",
        )
        files.append("brant_summary.txt")

        viol_names = [k for k, v in per_pred_p.items() if v == v and v < 0.05]
        summary.append(
            f"{entry.method} 完成：有序结果 {outcome}（{n_levels} 级，n={n}），{p} 个预测变量；"
            f"全局卡方={round(global_chi2, 3)}（df={int(global_df)}），p={round(global_p, 4) if global_p == global_p else 'NaN'}"
            f"——{verdict}"
            + (f"；违反的预测变量：{viol_names}" if viol_names else "")
            + ". "
            + sparse_note
            + "⚠ 这是分切点二元 logit 的 Brant 近似（非精确 Brant 联合协方差），需每个切点两组样本充足；"
            "⚠ 显著（小 p）⇒ 比例优势被违反，应改用非比例/广义有序 logit 或多项 logit；config 可指定 outcome/predictors。"
        )
        code += [
            "import numpy as np, statsmodels.api as sm; from scipy import stats",
            "# Brant approx: for each cut k fit binary logit of 1[y>level_k] on predictors",
            "# then test per-predictor whether the cut-specific slopes are equal (chi^2).",
            "# small global p => proportional-odds assumption VIOLATED.",
        ]
    except Exception as err:
        summary.append(f"{method}跳过：{err}")
