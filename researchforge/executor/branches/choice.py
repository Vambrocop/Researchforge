"""Branch handlers for the DISCRETE-CHOICE / CHOICE-MODELING family (economics).

Two complementary choice models for unordered/qualitative choice — the workhorses of
applied microeconometrics, transport, marketing, environmental & health economics:

  * mnl_choice          — unordered multinomial logit (statsmodels MNLogit). Predictors
        vary by CASE (case-specific, not alternative-specific): one outcome category is
        chosen per observation. Reports per-alternative coefficients (relative to the
        base category), RELATIVE RISK RATIOS RRR = exp(coef), McFadden pseudo-R²,
        an overall likelihood-ratio (LR) test vs the null, and a predicted-probability
        summary. Forest plot of RRRs.
  * conditional_logit   — McFadden's conditional (alternative-specific) logit on LONG
        data: each row is a (chooser, alternative) pair with ATTRIBUTES, exactly one
        chosen per choice situation. Hand-rolled MLE (scipy BFGS) with analytic
        gradient; SEs from the inverse observed-information (Hessian). Reports each
        attribute's β/SE/z/p, willingness-to-pay WTP_k = −β_k/β_cost (if a cost
        attribute is given), a point own-elasticity at means, and McFadden pseudo-R².

WHY a NEW id ``mnl_choice`` (not ``multinomial_logit``)?  The engine already ships a
basic ``multinomial_logit`` (statistics family, ``branches/statistics.py``). Re-using
that id would (a) raise ``ValueError("duplicate branch handler")`` at import — breaking
the whole registry — and (b) duplicate a catalog id. So this richer economics-family
MNL (RRR with the base category named, LR test, config outcome/predictors, forest plot)
is registered as ``mnl_choice``. Reported in STOP-AND-REPORT.

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法> 跳过：<原因>" appended to summary + return
    (never crash/fabricate). MNL needs an unordered outcome with ≥3 levels + ≥1 numeric
    predictor + enough rows; conditional logit needs a choice_id grouping + a 0/1 chosen
    + ≥1 attribute + each choice set having exactly one chosen alternative.
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    float ``estimates`` dict (plain floats only; nan for N/A — never tuples/strings),
    Chinese ``summary`` with ⚠ assumption/bias disclosures. MUTATE ctx, never rebind.

Pure Python (statsmodels / numpy / scipy / pandas / matplotlib). NO R.
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register

_MNL_MIN_ROWS = 30
_MNL_MIN_LEVELS = 3
_MNL_MAX_LEVELS = 12
_MNL_MAX_PREDICTORS = 6
_CLOGIT_MIN_SETS = 5
_CLOGIT_MAX_ATTRS = 8


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _sanitize(s: str) -> str:
    """Make a column name safe as an estimates-dict key fragment (alnum + underscore)."""
    out = re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_")
    return out or "x"


def _save_fig(d, fname, files, build):
    """best-effort matplotlib figure (Agg). build(plt) draws on the current figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        build(plt)
        plt.tight_layout()
        plt.savefig(d / fname, dpi=150)
        plt.close("all")
        files.append(fname)
    except Exception:
        pass


# ===========================================================================
# 1) mnl_choice — unordered multinomial logit (case-specific predictors)
#    Model:  P(y=j | x) = exp(x'β_j) / Σ_{k=0}^{J-1} exp(x'β_k), with β_base = 0.
#    RRR_jk = exp(β_jk): multiplicative change in the odds of category j vs the base
#             per 1-unit rise in predictor k.
#    pseudo-R² (McFadden) = 1 − llf/llnull.   LR = 2(llf − llnull) ~ chi²(df).
#    Refs: McFadden (1974); Greene "Econometric Analysis"; Train "Discrete Choice
#          Methods with Simulation".
# ===========================================================================
@register("mnl_choice")
def _branch_mnl_choice(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "多项 Logit（无序选择）"

    excl = {fp.unit_col, fp.time_col}

    if fp.n_rows < _MNL_MIN_ROWS:
        summary.append(
            f"{method} 跳过：样本量过小（{fp.n_rows} 行 < {_MNL_MIN_ROWS}），多项 logit 不可靠。"
        )
        return

    # ---- resolve the unordered categorical outcome (≥3 levels) ---------------
    outcome = None
    cfg_out = cfg.get("outcome")
    if cfg_out is not None:
        if cfg_out not in df.columns:
            summary.append(f"{method} 跳过：config 指定的结果列 {cfg_out!r} 不在数据中。")
            return
        nun = int(df[cfg_out].dropna().nunique())
        if nun < _MNL_MIN_LEVELS:
            summary.append(
                f"{method} 跳过：结果 {cfg_out} 仅 {nun} 个类别（<{_MNL_MIN_LEVELS}）；"
                "二分类请用 logistic_regression。"
            )
            return
        if nun > _MNL_MAX_LEVELS:
            summary.append(
                f"{method} 跳过：结果 {cfg_out} 有 {nun} 个不同值（>{_MNL_MAX_LEVELS}），"
                "更像连续/ID 而非名义类别。"
            )
            return
        outcome = cfg_out
    else:
        # auto: a categorical/binary/count column with 3..MAX levels (prefer categorical).
        cands = [
            c for c in fp.columns
            if c.kind in {"categorical", "binary", "count"}
            and _MNL_MIN_LEVELS <= c.n_unique <= _MNL_MAX_LEVELS
            and c.name not in excl
        ]
        cands.sort(key=lambda c: 0 if c.kind == "categorical" else 1)
        outcome = cands[0].name if cands else None
        if outcome is None:
            summary.append(
                f"{method} 跳过：未找到无序的多类别结果（{_MNL_MIN_LEVELS}–{_MNL_MAX_LEVELS} 类）。"
                "可用 config={\"outcome\":\"<列>\"} 指定。"
            )
            return

    # ---- resolve case-specific numeric predictors ---------------------------
    exclude = {outcome, fp.unit_col, fp.time_col}
    cfg_pred = cfg.get("predictors")
    if cfg_pred:
        predictors = [p for p in cfg_pred if p in df.columns and p not in exclude]
    else:
        predictors = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ]
    predictors = predictors[:_MNL_MAX_PREDICTORS]
    if not predictors:
        summary.append(f"{method} 跳过：没有可用的数值预测变量（连续/计数/二值）。")
        return

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from scipy import stats

        sub = df[[outcome] + predictors].dropna()
        n = len(sub)
        if n < _MNL_MIN_ROWS:
            summary.append(f"{method} 跳过：删除缺失后仅 {n} 行（<{_MNL_MIN_ROWS}）。")
            return

        # numeric predictor matrix (coerce; drop predictors that become non-numeric)
        Xnum = sub[predictors].apply(pd.to_numeric, errors="coerce")
        good_pred = [p for p in predictors if not Xnum[p].isna().any()]
        if not good_pred:
            summary.append(f"{method} 跳过：预测变量无法全部转为数值。")
            return
        Xnum = Xnum[good_pred].astype(float)
        # drop constant predictors (singular design)
        good_pred = [p for p in good_pred if float(Xnum[p].std(ddof=0)) > 0.0]
        if not good_pred:
            summary.append(f"{method} 跳过：预测变量均为常数（无方差），无法识别。")
            return
        Xnum = Xnum[good_pred]

        # outcome -> integer codes; statsmodels MNLogit uses the FIRST level (code 0)
        # as the base category (we keep the label and name it explicitly).
        codes, cats = pd.factorize(sub[outcome], sort=True)
        n_alt = int(len(cats))
        if n_alt < _MNL_MIN_LEVELS:
            summary.append(f"{method} 跳过：删除缺失后结果仅 {n_alt} 类（<{_MNL_MIN_LEVELS}）。")
            return
        base_level = str(cats[0])

        X = sm.add_constant(Xnum.to_numpy(), has_constant="add")
        Xcols = ["const"] + good_pred
        model = sm.MNLogit(codes, X).fit(disp=False, method="newton", maxiter=100)

        (d / "mnl_summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("mnl_summary.txt")

        params = np.asarray(model.params)   # shape (n_params, J-1) ; rows=Xcols
        pvals = np.asarray(model.pvalues)
        bse = np.asarray(model.bse)
        zvals = np.asarray(model.tvalues)
        rrr = np.exp(params)

        # McFadden pseudo-R² and overall LR test vs the intercept-only null.
        llf = float(model.llf)
        try:
            llnull = float(model.llnull)
        except Exception:
            # fallback: intercept-only model from class proportions
            counts = np.bincount(codes, minlength=n_alt).astype(float)
            p = counts / counts.sum()
            llnull = float(np.sum(counts[counts > 0] * np.log(p[counts > 0])))
        pseudo_r2 = 1.0 - llf / llnull if llnull != 0 else float("nan")
        lr_stat = 2.0 * (llf - llnull)
        # df = number of slope params freed = (#predictors) * (J-1)
        lr_df = int(len(good_pred) * (n_alt - 1))
        lr_p = float(stats.chi2.sf(lr_stat, lr_df)) if lr_df > 0 else float("nan")

        # coefficient / RRR table — one block per non-base alternative.
        rows = []
        for j in range(n_alt - 1):  # column j -> alternative cats[j+1] vs base cats[0]
            lvl = str(cats[j + 1])
            for r, term in enumerate(Xcols):
                rows.append({
                    "alternative_vs_base": f"{lvl}_vs_{base_level}",
                    "term": term,
                    "coef": round(float(params[r, j]), 6),
                    "rrr": round(float(rrr[r, j]), 6),
                    "std_err": round(float(bse[r, j]), 6),
                    "z": round(float(zvals[r, j]), 6),
                    "p_value": round(float(pvals[r, j]), 6),
                })
        coef_df = pd.DataFrame(rows)
        coef_df.to_csv(d / "mnl_coefficients.csv", index=False, encoding="utf-8")
        files.append("mnl_coefficients.csv")

        # predicted-probability summary (mean predicted P per alternative).
        try:
            pred = np.asarray(model.predict(X))         # (n, J)
            mean_pred = pred.mean(axis=0)
            acc = float((pred.argmax(axis=1) == codes).mean())
            pd.DataFrame({
                "alternative": [str(c) for c in cats],
                "observed_share": [float(np.mean(codes == k)) for k in range(n_alt)],
                "mean_pred_prob": [float(x) for x in mean_pred],
            }).to_csv(d / "mnl_pred_prob.csv", index=False, encoding="utf-8")
            files.append("mnl_pred_prob.csv")
        except Exception:
            acc = float("nan")

        # estimates: plain floats only.
        estimates.update({
            "pseudo_r2": round(float(pseudo_r2), 6) if pseudo_r2 == pseudo_r2 else float("nan"),
            "lr_stat": round(float(lr_stat), 6),
            "lr_p": round(float(lr_p), 6) if lr_p == lr_p else float("nan"),
            "lr_df": float(lr_df),
            "n_obs": float(n),
            "n_alternatives": float(n_alt),
            "n_predictors": float(len(good_pred)),
            "in_sample_accuracy": round(acc, 6) if acc == acc else float("nan"),
            "loglik": round(llf, 6),
        })
        # per non-base level × predictor RRR (sanitized keys; guard collisions).
        for j in range(n_alt - 1):
            lvl = _sanitize(str(cats[j + 1]))
            for r, term in enumerate(Xcols):
                if term == "const":
                    continue
                key = f"rrr__{lvl}__{_sanitize(term)}"
                if key in estimates:  # collision guard (sanitization can clash)
                    suffix = 2
                    while f"{key}_{suffix}" in estimates:
                        suffix += 1
                    key = f"{key}_{suffix}"
                estimates[key] = round(float(rrr[r, j]), 6)

        # forest plot of RRRs (log scale), excluding the constant.
        def _plot(plt):
            labels, vals = [], []
            for j in range(n_alt - 1):
                lvl = str(cats[j + 1])
                for r, term in enumerate(Xcols):
                    if term == "const":
                        continue
                    labels.append(f"{term} | {lvl}")
                    vals.append(float(rrr[r, j]))
            ypos = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(7, max(2.4, 0.42 * len(labels) + 1.2)))
            ax.scatter(vals, ypos, color="#4C72B0", zorder=3)
            ax.axvline(1.0, color="grey", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("relative risk ratio RRR = exp(coef), log scale")
            ax.set_title(f"MNL relative risk ratios (base = {base_level})")
            ax.invert_yaxis()

        _save_fig(d, "mnl_rrr_forest.png", files, _plot)

        # a short key-finding line
        key_txt = ""
        if good_pred and n_alt >= 2:
            r0 = Xcols.index(good_pred[0])
            key_txt = (f"，例：预测变量 {good_pred[0]} 对「{str(cats[1])} vs {base_level}」"
                       f"的 RRR={float(rrr[r0, 0]):.3f}（p={float(pvals[r0, 0]):.3g}）")
        acc_txt = f"，类内准确率={acc:.1%}" if acc == acc else ""
        summary.append(
            f"{entry.method} 完成：无序结果 {outcome}（{n_alt} 类，基准={base_level}，n={n}），"
            f"{len(good_pred)} 个案例层(case-specific)预测变量{key_txt}；"
            f"McFadden 伪R²={pseudo_r2:.3f}，整体 LR 检验 χ²={lr_stat:.3f}"
            f"（df={lr_df}，p={lr_p:.3g}）{acc_txt}。系数/RRR 表见 mnl_coefficients.csv、森林图见 mnl_rrr_forest.png。"
            " ⚠ 系数与 RRR 均相对于基准类别（base）解读：RRR=exp(系数) 是「某预测变量+1 单位时，"
            f"该类 vs 基准类「{base_level}」相对发生比」的乘数；基准类的系数恒为 0。"
            " ⚠ IIA 假定（无关方案独立性）：MNL 假定增删某一备选不改变其余各对备选的相对几率；"
            "若该假定被违反（如存在相似/替代性强的备选），应改用嵌套 logit 或混合(mixed) logit。"
            " 本法为案例层预测变量（每观测一选择），非备选层属性——后者请用 conditional_logit。"
        )
        code += [
            "import statsmodels.api as sm; import numpy as np",
            f"codes, cats = pd.factorize(df[{outcome!r}], sort=True)  # base = cats[0]",
            f"X = sm.add_constant(df[{good_pred!r}].astype(float))",
            "m = sm.MNLogit(codes, X).fit()      # 多项 logit（基准=cats[0]）",
            "RRR = np.exp(m.params)              # 相对风险比 RRR=exp(系数)",
            "pseudo_r2 = 1 - m.llf/m.llnull      # McFadden 伪 R²; LR=2(llf-llnull)~chi2",
        ]
    except Exception as err:
        summary.append(f"{method} 跳过：{err}")


# ===========================================================================
# 2) conditional_logit — McFadden conditional logit (alternative-specific attrs)
#    LONG data: row = (choice situation s, alternative j) with attributes x_jk.
#    Within choice set s:  P(chosen=j) = exp(Σ_k β_k x_jk) / Σ_{j'} exp(Σ_k β_k x_j'k)
#    log-likelihood  LL(β) = Σ_s Σ_j chosen_sj · log P_sj ; maximize over β.
#    Analytic gradient:  ∂LL/∂β_k = Σ_s Σ_j (chosen_sj − P_sj) · x_sjk.
#    SEs from the inverse observed-information (negative Hessian of LL) — see below.
#    WTP_k = −β_k / β_cost.    own-elasticity (continuous attr, at means):
#            E_k ≈ β_k · x̄_k · (1 − P̄).
#    Refs: McFadden (1974) "Conditional logit analysis of qualitative choice
#          behavior"; Train (2009); Greene "Econometric Analysis".
# ===========================================================================
@register("conditional_logit")
def _branch_conditional_logit(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "条件 Logit（McFadden）"

    excl = {fp.unit_col, fp.time_col}

    # ---- resolve the choice-situation grouping id ---------------------------
    choice_id = cfg.get("choice_id")
    if choice_id is None:
        choice_id = fp.unit_col if (fp.unit_col and fp.unit_col in df.columns) else None
    if choice_id is None or choice_id not in df.columns:
        summary.append(
            f"{method} 跳过：需要选择情景分组列 choice_id（每组是一组备选）。"
            "请用 config={\"choice_id\":\"<列>\"} 指定（长表：每行=一个「选择情景 × 备选」对）。"
        )
        return

    # ---- resolve the 0/1 chosen indicator ----------------------------------
    chosen_col = cfg.get("chosen")
    if chosen_col is None:
        # auto: a binary column whose value set is exactly {0,1}.
        import pandas as pd  # local for the scan
        for c in df.columns:
            if c in {choice_id} | excl:
                continue
            vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if set(vals).issubset({0.0, 1.0}) and len(set(vals)) == 2:
                chosen_col = c
                break
    if chosen_col is None or chosen_col not in df.columns:
        summary.append(
            f"{method} 跳过：需要 0/1 的选中指示列 chosen（每个选择情景恰有一行=1）。"
            "请用 config={\"chosen\":\"<列>\"} 指定。"
        )
        return

    # ---- resolve attribute columns (alternative-specific) ------------------
    cost_name = cfg.get("cost") or cfg.get("price")
    cfg_attrs = cfg.get("attributes")
    drop = {choice_id, chosen_col, fp.unit_col, fp.time_col}
    if cfg_attrs:
        attrs = [a for a in cfg_attrs if a in df.columns and a not in drop]
    else:
        attrs = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in drop
        ]
        # the cost column (if named) must be included as an attribute even if it
        # would otherwise be excluded; ensure it is present.
        if cost_name and cost_name in df.columns and cost_name not in attrs and cost_name not in drop:
            attrs.append(cost_name)
    attrs = attrs[:_CLOGIT_MAX_ATTRS]
    if not attrs:
        summary.append(f"{method} 跳过：没有可用的备选层属性列（数值）。")
        return
    if cost_name and cost_name not in attrs:
        # cost named but not among modelled attributes (filtered out) -> no WTP
        cost_name = None

    try:
        import numpy as np
        import pandas as pd
        from scipy import optimize, stats

        keep = [choice_id, chosen_col] + attrs
        sub = df[keep].copy()
        sub[attrs] = sub[attrs].apply(pd.to_numeric, errors="coerce")
        sub[chosen_col] = pd.to_numeric(sub[chosen_col], errors="coerce")
        sub = sub.dropna(subset=[chosen_col] + attrs)

        # validate the 0/1 chosen and one-chosen-per-set structure.
        ch = sub[chosen_col].to_numpy(dtype=float)
        if not set(np.unique(ch)).issubset({0.0, 1.0}):
            summary.append(f"{method} 跳过：chosen 列 {chosen_col} 不是 0/1 指示。")
            return

        # group rows by choice situation; keep only well-formed sets:
        #   ≥2 alternatives AND exactly one chosen (==1).
        groups = []          # list of (idx array) per usable choice set
        n_bad_chosen = 0
        n_singleton = 0
        for _, g in sub.groupby(choice_id, sort=False):
            idx = g.index.to_numpy()
            csum = float(g[chosen_col].sum())
            if len(idx) < 2:
                n_singleton += 1
                continue
            if abs(csum - 1.0) > 1e-9:
                n_bad_chosen += 1
                continue
            groups.append(idx)

        n_sets = len(groups)
        if n_sets < _CLOGIT_MIN_SETS:
            summary.append(
                f"{method} 跳过：合格选择情景过少（{n_sets} < {_CLOGIT_MIN_SETS}）。"
                f"（合格=每组≥2备选且恰有一个 chosen==1；剔除 {n_singleton} 个单备选组、"
                f"{n_bad_chosen} 个选中数≠1 的组）"
            )
            return

        # drop attributes with NO within-set variation in ANY set (unidentified):
        # an attribute that is constant across alternatives within every choice set
        # contributes nothing (differences out of the conditional likelihood).
        usable_attrs = []
        for a in attrs:
            has_var = False
            for idx in groups:
                col = sub.loc[idx, a].to_numpy(dtype=float)
                if float(np.std(col)) > 1e-12:
                    has_var = True
                    break
            if has_var:
                usable_attrs.append(a)
        if not usable_attrs:
            summary.append(
                f"{method} 跳过：所有属性在每个选择情景内都无组内变异——"
                "条件 logit 需要属性在同一情景的备选间有差异（否则被差分掉、不可识别）。"
            )
            return
        if cost_name and cost_name not in usable_attrs:
            cost_name = None
        attrs = usable_attrs
        K = len(attrs)

        # Build per-set design matrices and chosen index.
        Xs = [sub.loc[idx, attrs].to_numpy(dtype=float) for idx in groups]   # (J_s, K)
        ys = [int(np.argmax(sub.loc[idx, chosen_col].to_numpy())) for idx in groups]

        def _neg_ll_grad(beta):
            """Negative log-likelihood and its gradient (analytic)."""
            nll = 0.0
            grad = np.zeros(K, dtype=float)
            for Xg, yj in zip(Xs, ys):
                u = Xg @ beta                       # utilities (J_s,)
                u -= u.max()                        # numerical stability
                ex = np.exp(u)
                P = ex / ex.sum()                   # choice probabilities (J_s,)
                nll -= np.log(max(P[yj], 1e-300))
                # gradient of -LL: Σ_j (P_j − chosen_j) x_j  (chosen = e_{yj})
                d_choose = P.copy()
                d_choose[yj] -= 1.0
                grad += Xg.T @ d_choose
            return nll, grad

        beta0 = np.zeros(K, dtype=float)
        res = optimize.minimize(
            _neg_ll_grad, beta0, jac=True, method="BFGS",
            options={"maxiter": 500, "gtol": 1e-7},
        )
        beta = np.asarray(res.x, dtype=float)
        # BFGS res.success is unreliable here — it often reports False from line-search
        # precision loss even AT the optimum of a flat likelihood. For an unconstrained
        # MLE the real convergence criterion is a near-zero gradient, so check that.
        _, _gfin = _neg_ll_grad(beta)
        converged = bool(res.success) or float(np.linalg.norm(_gfin)) < 1e-4

        # log-likelihoods for McFadden pseudo-R².
        ll_full = -float(res.fun)
        # null model: β=0 -> each alt equally likely within its set -> LL0 = Σ_s -log(J_s)
        ll_null = -float(np.sum([np.log(len(idx)) for idx in groups]))
        pseudo_r2 = 1.0 - ll_full / ll_null if ll_null != 0 else float("nan")

        # SEs from the inverse observed information = inverse of the Hessian of the
        # NEGATIVE log-likelihood at the optimum. scipy BFGS keeps an APPROXIMATE
        # inverse-Hessian (res.hess_inv) but it is unreliable for SEs; we compute an
        # analytic Hessian of -LL directly (the conditional logit Hessian is the sum
        # over sets of X_s'(diag(P)-P P')X_s), then invert it. (STOP-AND-REPORT: we use
        # the analytic Hessian rather than numdifftools to avoid an extra dependency.)
        H = np.zeros((K, K), dtype=float)
        for Xg in Xs:
            u = Xg @ beta
            u -= u.max()
            ex = np.exp(u)
            P = ex / ex.sum()
            # Σ_j P_j x_j x_j'  −  (Σ_j P_j x_j)(Σ_j P_j x_j)'
            PX = P[:, None] * Xg                       # (J_s, K)
            term1 = Xg.T @ PX                          # Σ P_j x_j x_j'
            mbar = PX.sum(axis=0)                      # Σ P_j x_j
            H += term1 - np.outer(mbar, mbar)
        # H is the (positive-definite) information matrix; cov = H^{-1}.
        ill_conditioned = False
        try:
            if float(np.linalg.cond(H)) > 1e10:  # near-collinear attributes -> unstable SEs
                ill_conditioned = True
            cov = np.linalg.inv(H)
            se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        except np.linalg.LinAlgError:
            cov = np.full((K, K), np.nan)
            se = np.full(K, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = beta / se
            pval = 2.0 * stats.norm.sf(np.abs(z))

        # WTP_k = −β_k / β_cost  (if a cost attribute is given). delta-method CI noted.
        wtp = {}
        if cost_name is not None:
            ci = attrs.index(cost_name)
            b_cost = float(beta[ci])
            if abs(b_cost) > 1e-12:
                for k, a in enumerate(attrs):
                    if a == cost_name:
                        continue
                    wtp[a] = -float(beta[k]) / b_cost

        # point own-elasticity at means for each (continuous) attribute:
        #   E_k ≈ β_k · x̄_k · (1 − P̄),  P̄ = mean across sets of the chosen alt's prob.
        # Disclosed approximation (uses the average choice prob, not alt-specific).
        all_x = np.vstack(Xs)
        xbar = all_x.mean(axis=0)
        # mean predicted probability of an "average" alternative ~ 1/Jbar; use the mean
        # chosen-alt probability as P̄ for a representative point elasticity.
        pbars = []
        for Xg, yj in zip(Xs, ys):
            u = Xg @ beta
            u -= u.max()
            ex = np.exp(u)
            P = ex / ex.sum()
            pbars.append(float(P[yj]))
        Pbar = float(np.mean(pbars)) if pbars else float("nan")
        elasticity = {a: float(beta[k] * xbar[k] * (1.0 - Pbar)) for k, a in enumerate(attrs)}

        # coefficient table.
        rows = []
        for k, a in enumerate(attrs):
            rows.append({
                "attribute": a,
                "beta": round(float(beta[k]), 6),
                "std_err": round(float(se[k]), 6) if se[k] == se[k] else float("nan"),
                "z": round(float(z[k]), 6) if z[k] == z[k] else float("nan"),
                "p_value": round(float(pval[k]), 6) if pval[k] == pval[k] else float("nan"),
                "wtp": round(float(wtp[a]), 6) if a in wtp else float("nan"),
                "own_elasticity_at_means": round(float(elasticity[a]), 6),
            })
        coef_df = pd.DataFrame(rows)
        coef_df.to_csv(d / "clogit_coefficients.csv", index=False, encoding="utf-8")
        files.append("clogit_coefficients.csv")

        # estimates: plain floats only.
        estimates.update({
            "pseudo_r2": round(float(pseudo_r2), 6) if pseudo_r2 == pseudo_r2 else float("nan"),
            "n_choice_sets": float(n_sets),
            "n_alternatives": round(float(np.mean([len(idx) for idx in groups])), 3),
            "n_attributes": float(K),
            "loglik": round(float(ll_full), 6),
            "loglik_null": round(float(ll_null), 6),
            "converged": 1.0 if converged else 0.0,
            "p_chosen_mean": round(Pbar, 6) if Pbar == Pbar else float("nan"),
        })
        for k, a in enumerate(attrs):
            sa = _sanitize(a)
            estimates[f"beta__{sa}"] = round(float(beta[k]), 6)
            estimates[f"se__{sa}"] = round(float(se[k]), 6) if se[k] == se[k] else float("nan")
            if a in wtp:
                estimates[f"wtp__{sa}"] = round(float(wtp[a]), 6)

        # coefficient plot (β ± 1.96 SE, ENGLISH labels).
        def _plot(plt):
            yp = np.arange(K)
            err = 1.96 * np.where(np.isfinite(se), se, 0.0)
            fig, ax = plt.subplots(figsize=(6.5, 0.6 * K + 1.6))
            ax.errorbar(beta, yp, xerr=err, fmt="o", color="#4C72B0", capsize=3)
            ax.axvline(0.0, color="grey", ls="--", lw=1)
            ax.set_yticks(yp)
            ax.set_yticklabels(attrs)
            ax.set_xlabel("conditional-logit coefficient beta (95% CI)")
            ax.set_title("Conditional logit: attribute coefficients")
            ax.invert_yaxis()

        _save_fig(d, "clogit_coefficients.png", files, _plot)

        # narrative
        sign_txt = "、".join(
            f"{a}: β={float(beta[k]):.3f}{'*' if (pval[k] == pval[k] and pval[k] < 0.05) else ''}"
            for k, a in enumerate(attrs)
        )
        wtp_txt = ""
        if wtp:
            wtp_txt = (" 边际支付意愿 WTP（相对成本 " + cost_name + "）："
                       + "、".join(f"{a}={v:.4g}" for a, v in wtp.items()) + "；")
        conv_txt = "已收敛" if converged else "⚠ 未收敛（结果不可靠，请检查数据/共线性/标度）"
        if ill_conditioned:
            conv_txt += "；⚠ 信息矩阵接近奇异（属性近似共线），SE/p 不稳定"
        summary.append(
            f"{entry.method} 完成：{n_sets} 个选择情景、{K} 个备选层属性（{conv_txt}）；"
            f"系数 {sign_txt}；McFadden 伪R²={pseudo_r2:.3f}。{wtp_txt}"
            "系数/WTP/弹性表见 clogit_coefficients.csv、系数图见 clogit_coefficients.png。"
            " ⚠ 条件 logit 同样假定 IIA（无关方案独立性）——若备选间存在相似/替代结构，"
            "应改用嵌套或混合(mixed) logit。"
            " ⚠ WTP=−β_属性/β_成本 为比值，其置信区间需用 delta 法（本实现未给 WTP 的 CI，"
            "仅给点估计）。 ⚠ 点弹性按公式 E_k≈β_k·x̄_k·(1−P̄)（在均值处、用平均被选概率 P̄ 近似），"
            "是局部近似，非每备选精确弹性。"
            " ⚠ 需要足够多的选择情景，且属性在同一情景的备选间须有变异方能识别（无组内变异的属性已剔除）。"
        )
        code += [
            "import numpy as np; from scipy import optimize, stats",
            "# 长表：每行=(选择情景 s, 备选 j)；choice_id 分组、chosen∈{0,1} 每组恰一个 1",
            "# 条件 logit: P_sj = exp(x_sj'β)/Σ_j' exp(x_sj'β); LL=Σ_s log P_{s,chosen}",
            "# 解析梯度 ∂LL/∂β = Σ_s Σ_j (chosen_j − P_j) x_j; BFGS 求最大似然",
            "# SE: 解析 Hessian H=Σ_s X_s'(diag(P)−P P')X_s 的逆 (cov=H^{-1})",
            "# WTP_k = −β_k/β_cost ; 点弹性 E_k≈β_k·x̄_k·(1−P̄)",
        ]
    except Exception as err:
        summary.append(f"{method} 跳过：{err}")
