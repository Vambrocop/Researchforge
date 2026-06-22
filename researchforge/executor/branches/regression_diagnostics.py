"""Branch handlers for the OLS regression-diagnostics family.

Three post-fit diagnostics for an OLS model — they all fit ``OLS(outcome ~ predictors)``
(engine regression convention: outcome = first continuous column, predictors = the
remaining continuous/binary/count, both overridable via ``config['outcome']`` /
``config['predictors']``) and then interrogate the fit:

* ``vif_multicollinearity`` — VIF per predictor + condition number (collinearity).
* ``heteroskedasticity_test`` — Breusch-Pagan + White tests (non-constant variance).
* ``influence_diagnostics`` — leverage / studentized resid / Cook's D / DFFITS (outliers).

Engine conventions (see CLAUDE.md「引擎约定」): each handler is
``@register("<id>") def _branch_<id>(ctx)``; it unpacks ctx into
df/fp/entry/cfg/d + files/summary/estimates/code and **mutates** them (never rebinds).
Honest degrade: too few rows / <2 predictors (VIF) / non-numeric / singular design /
import missing -> append a Chinese "<方法>跳过：<原因>" to summary and RETURN; never
crash or fabricate. Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels) in
try/except; float ``estimates``; Chinese ``summary`` ending with ⚠ disclosures.

Pure Python (statsmodels / numpy / scipy) — no R.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ----------------------------------------------------------------------------- #
# Shared: resolve outcome + predictors (engine regression convention) and fit   #
# OLS on the cleaned numeric subset. Returns a dict; on failure returns a dict   #
# with key "error" holding the Chinese skip reason (caller appends + returns).   #
# ----------------------------------------------------------------------------- #
def _fit_ols(ctx: Ctx, label: str, min_predictors: int = 1):
    import numpy as np
    import statsmodels.api as sm

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    excl = {fp.unit_col, fp.time_col}
    cont_cols = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]

    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else (
        cont_cols[0] if cont_cols else None
    )
    if outcome is None:
        return {"error": f"{label}跳过：未找到连续型结果变量。"}

    exclude = {outcome, fp.unit_col, fp.time_col}
    if cfg.get("predictors"):
        predictors = [c for c in cfg["predictors"] if c in df.columns and c not in exclude]
    else:
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
        ][:8]
    if len(predictors) < min_predictors:
        why = "需要至少 2 个预测变量" if min_predictors >= 2 else "未找到预测变量"
        return {"error": f"{label}跳过：{why}（当前 {len(predictors)} 个）。"}

    sub = df[[outcome] + predictors].dropna()
    for c in [outcome] + predictors:
        try:
            sub = sub[np.isfinite(sub[c].to_numpy(dtype=float))]
        except (TypeError, ValueError):
            return {"error": f"{label}跳过：列 {c} 含非数值，无法做诊断。"}
    n = int(sub.shape[0])
    if n < len(predictors) + 2:
        return {"error": f"{label}跳过：有效样本不足（n={n}，预测变量 {len(predictors)} 个）。"}

    y = sub[outcome].to_numpy(dtype=float)
    X = sub[predictors].to_numpy(dtype=float)
    X_const = sm.add_constant(X, has_constant="add")  # columns: [const, *predictors]
    # singular / rank-deficient design -> degrade honestly
    if np.linalg.matrix_rank(X_const) < X_const.shape[1]:
        return {"error": f"{label}跳过：设计矩阵奇异（列完全共线 / 常数列），无法拟合。"}
    try:
        model = sm.OLS(y, X_const).fit()
    except Exception as err:  # noqa: BLE001 — degrade, never crash
        return {"error": f"{label}跳过：OLS 拟合失败（{err}）。"}

    return {
        "outcome": outcome,
        "predictors": predictors,
        "n": n,
        "y": y,
        "X": X,
        "X_const": X_const,
        "model": model,
    }


# ----------------------------------------------------------------------------- #
# (A) vif_multicollinearity — VIF per predictor + condition number              #
# ----------------------------------------------------------------------------- #
@register("vif_multicollinearity")
def _branch_vif_multicollinearity(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    fit = _fit_ols(ctx, "多重共线性诊断", min_predictors=2)  # VIF needs >=2 predictors
    if "error" in fit:
        summary.append(fit["error"])
        return
    predictors, X, X_const = fit["predictors"], fit["X"], fit["X_const"]

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception as err:  # noqa: BLE001
        summary.append(f"多重共线性诊断跳过：导入失败（{err}）。")
        return

    # VIF_j = 1/(1-R^2_j), R^2_j from regressing predictor j on the OTHER predictors.
    # statsmodels computes this on the design WITH a constant; the predictor columns
    # are at indices 1..p in X_const (index 0 is the constant, skipped).
    p = len(predictors)
    vifs: list[float] = []
    for j in range(1, p + 1):
        try:
            v = float(variance_inflation_factor(X_const, j))
        except Exception:  # noqa: BLE001 — perfect collinearity -> inf
            v = float("inf")
        vifs.append(v)
    vif_arr = np.array(vifs, dtype=float)
    tol = np.where(vif_arr > 0, 1.0 / vif_arr, np.nan)  # tolerance = 1/VIF

    # Condition number from eigenvalues of the scaled (unit-length columns) X'X.
    # Belsley-Kuh-Welsch (1980) condition number: scale each column of the FULL design
    # INCLUDING the intercept to unit length (no centering — centering would hide
    # intercept-related collinearity), then kappa = sqrt(lambda_max / lambda_min). The
    # >30 cutoff is calibrated for this with-intercept, unit-scaled design.
    norms = np.linalg.norm(X_const, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    Xs = X_const / norms
    XtX = Xs.T @ Xs
    eigvals = np.linalg.eigvalsh(XtX)
    eigvals = np.clip(eigvals, 0.0, None)
    lam_max = float(eigvals.max())
    lam_min = float(eigvals.min())
    if lam_min <= 0 or not np.isfinite(lam_min):
        cond_number = float("inf")
        cond_indices = np.full_like(eigvals, np.inf, dtype=float)
    else:
        cond_number = float(np.sqrt(lam_max / lam_min))
        cond_indices = np.sqrt(lam_max / eigvals)

    finite_vif = vif_arr[np.isfinite(vif_arr)]
    max_vif = float(vif_arr.max()) if vif_arr.size else float("nan")
    mean_vif = float(finite_vif.mean()) if finite_vif.size else float("inf")
    n_high_vif = float(int(np.sum(vif_arr > 10)))  # severe
    n_moderate = int(np.sum((vif_arr > 5) & (vif_arr <= 10)))

    estimates["max_vif"] = max_vif
    estimates["mean_vif"] = mean_vif
    estimates["n_high_vif"] = n_high_vif
    estimates["condition_number"] = cond_number
    estimates["n_predictors"] = float(p)

    # --- CSV: per-predictor VIF + tolerance --------------------------------- #
    import pandas as pd

    flags = [
        "severe(>10)" if v > 10 else ("moderate(>5)" if v > 5 else "ok") for v in vifs
    ]
    tab = pd.DataFrame(
        {
            "predictor": predictors,
            "VIF": vifs,
            "tolerance": [float(t) for t in tol],
            "flag": flags,
        }
    )
    tab.to_csv(d / "vif.csv", index=False, encoding="utf-8")
    files.append("vif.csv")

    # condition indices alongside (eigenvalue diagnostics)
    cond_tab = pd.DataFrame(
        {
            "eigenvalue": [float(e) for e in eigvals],
            "condition_index": [float(c) for c in cond_indices],
        }
    )
    cond_tab.to_csv(d / "condition_indices.csv", index=False, encoding="utf-8")
    files.append("condition_indices.csv")

    # --- PNG: bar of VIF per predictor + 5 / 10 thresholds ------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_vif = [min(v, 50.0) if np.isfinite(v) else 50.0 for v in vifs]  # cap inf for display
        fig, ax = plt.subplots(figsize=(max(5, 0.7 * p + 2), 4))
        colors = ["#b85450" if v > 10 else ("#d6b656" if v > 5 else "#6c8ebf") for v in vifs]
        ax.bar(range(p), plot_vif, color=colors)
        ax.axhline(5, color="#d6b656", ls="--", lw=1.2, label="VIF = 5 (moderate)")
        ax.axhline(10, color="#b85450", ls="--", lw=1.2, label="VIF = 10 (severe)")
        ax.set_xticks(range(p))
        ax.set_xticklabels(predictors, rotation=45, ha="right")
        ax.set_ylabel("VIF (capped at 50 for display)")
        ax.set_title("Variance Inflation Factor per predictor")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "vif.png", dpi=150)
        plt.close(fig)
        files.append("vif.png")
    except Exception:  # noqa: BLE001 — plotting is best-effort
        pass

    cond_flag = "（条件数 >30，存在共线性迹象）" if cond_number > 30 else ""
    sev = (
        f"，其中 {int(n_high_vif)} 个 VIF>10（严重）" if n_high_vif else ""
    ) + (f"、{n_moderate} 个 5<VIF≤10（中度）" if n_moderate else "")
    summary.append(
        f"{entry.method} 完成：因变量 {fit['outcome']}，{p} 个预测变量。"
        f" 最大 VIF={max_vif:.3g}，平均 VIF={mean_vif:.3g}{sev}；"
        f" 条件数={cond_number:.3g}{cond_flag}。明细见 vif.csv。"
    )
    summary.append(
        "⚠ 假定：VIF 需 ≥2 个预测变量；高 VIF / 高条件数会膨胀系数标准误、使系数不稳定，"
        "但并不使系数有偏（OLS 仍无偏）；条件数对量纲敏感——本处已把各列缩放到单位长度后"
        "再算（去掉了平凡的量纲依赖）；config 可覆盖 outcome/predictors。"
    )

    code += [
        "import statsmodels.api as sm; import numpy as np",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor",
        f"X = sm.add_constant(df[{predictors}].dropna().to_numpy(float))",
        "vif = [variance_inflation_factor(X, j) for j in range(1, X.shape[1])]  # skip const",
        "Xs = X[:,1:] / np.linalg.norm(X[:,1:], axis=0)  # unit-length columns",
        "ev = np.linalg.eigvalsh(Xs.T @ Xs); cond = np.sqrt(ev.max()/ev.min())",
    ]


# ----------------------------------------------------------------------------- #
# (B) heteroskedasticity_test — Breusch-Pagan + White                          #
# ----------------------------------------------------------------------------- #
@register("heteroskedasticity_test")
def _branch_heteroskedasticity_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    fit = _fit_ols(ctx, "异方差检验", min_predictors=1)
    if "error" in fit:
        summary.append(fit["error"])
        return
    predictors, X_const, model, n = fit["predictors"], fit["X_const"], fit["model"], fit["n"]

    try:
        from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    except Exception as err:  # noqa: BLE001
        summary.append(f"异方差检验跳过：导入失败（{err}）。")
        return

    resid = model.resid

    # Breusch-Pagan: regress squared residuals on the regressors -> LM = n*R^2 ~ chi2(p).
    # (Breusch & Pagan 1979; Koenker's studentized version is what statsmodels returns.)
    try:
        bp_lm, bp_p, bp_f, bp_fp = (float(v) for v in het_breuschpagan(resid, X_const))
    except Exception as err:  # noqa: BLE001
        summary.append(f"异方差检验跳过：Breusch-Pagan 失败（{err}）。")
        return

    # White: auxiliary regression on regressors, their squares, and cross-products.
    # More general (no functional-form assumption) but consumes df fast and needs
    # enough rows; statsmodels raises when the auxiliary design is rank-deficient.
    white_ok = True
    try:
        w_lm, w_p, w_f, w_fp = (float(v) for v in het_white(resid, X_const))
    except Exception:  # noqa: BLE001 — too few rows / singular aux design
        white_ok = False
        w_lm = w_p = w_f = w_fp = float("nan")

    estimates["bp_lm_stat"] = bp_lm
    estimates["bp_p"] = bp_p
    estimates["white_lm_stat"] = w_lm
    estimates["white_p"] = w_p
    estimates["n"] = float(n)
    estimates["n_predictors"] = float(len(predictors))

    # --- CSV: each test's stat / p / df ------------------------------------- #
    import pandas as pd

    p = len(predictors)
    # White auxiliary df = the rank of [const, regressors, their squares, cross-products]
    # minus the constant. Computing the ACTUAL rank (not the nominal p+p+p(p-1)/2) is
    # correct when statsmodels drops collinear auxiliary terms — e.g. a binary
    # predictor's square equals itself, so that column adds no df.
    if white_ok:
        import itertools

        Xp = np.asarray(X_const)[:, 1:]  # predictor columns (drop the constant)
        _aux = [np.ones(Xp.shape[0]), *[Xp[:, j] for j in range(p)]]
        _aux += [Xp[:, j] ** 2 for j in range(p)]
        _aux += [Xp[:, a] * Xp[:, b] for a, b in itertools.combinations(range(p), 2)]
        white_df = int(np.linalg.matrix_rank(np.column_stack(_aux))) - 1
    else:
        white_df = float("nan")
    tab = pd.DataFrame(
        {
            "test": ["Breusch-Pagan", "White"],
            "LM_stat": [bp_lm, w_lm],
            "LM_p": [bp_p, w_p],
            "LM_df": [p, white_df],
            "F_stat": [bp_f, w_f],
            "F_p": [bp_fp, w_fp],
        }
    )
    tab.to_csv(d / "heteroskedasticity_tests.csv", index=False, encoding="utf-8")
    files.append("heteroskedasticity_tests.csv")

    # --- PNG: residuals vs fitted (canonical heteroskedasticity diagnostic) -- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fitted = model.fittedvalues
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(fitted, resid, s=16, color="#6c8ebf", alpha=0.6, edgecolor="white")
        ax.axhline(0, color="#b85450", ls="--", lw=1.2)
        ax.set_xlabel("fitted values")
        ax.set_ylabel("residuals")
        ax.set_title(f"Residuals vs fitted (BP p={bp_p:.3g})")
        fig.tight_layout()
        fig.savefig(d / "residuals_vs_fitted.png", dpi=150)
        plt.close(fig)
        files.append("residuals_vs_fitted.png")
    except Exception:  # noqa: BLE001
        pass

    verdict = "存在异方差迹象" if bp_p < 0.05 else "未检出明显异方差"
    white_note = (
        f"，White LM={w_lm:.4g}（p={w_p:.3g}）" if white_ok else "，White 检验因样本/自由度不足跳过"
    )
    summary.append(
        f"{entry.method} 完成：因变量 {fit['outcome']}，n={n}。"
        f" Breusch-Pagan LM={bp_lm:.4g}（p={bp_p:.3g}）{white_note}。"
        f" 结论：{verdict}（α=0.05）。明细见 heteroskedasticity_tests.csv。"
    )
    summary.append(
        "⚠ 解读：检验显著 ⇒ OLS 普通标准误有偏（点估计仍无偏）→ 改用稳健(HC)标准误或 WLS；"
        "White 更一般（不设方差函数形式）但功效较低、预测变量多时迅速耗尽自由度；"
        "检验只识别方差结构、不指明正确的补救方法；config 可覆盖 outcome/predictors。"
    )

    code += [
        "import statsmodels.api as sm",
        "from statsmodels.stats.diagnostic import het_breuschpagan, het_white",
        f"X = sm.add_constant(df[{predictors}].dropna().to_numpy(float))",
        f"m = sm.OLS(df['{fit['outcome']}'].dropna().to_numpy(float), X).fit()",
        "bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(m.resid, X)  # LM ~ chi2(p)",
        "w_lm, w_p, w_f, w_fp = het_white(m.resid, X)  # general, df-hungry",
    ]


# ----------------------------------------------------------------------------- #
# (C) influence_diagnostics — leverage / studentized resid / Cook's D / DFFITS  #
# ----------------------------------------------------------------------------- #
@register("influence_diagnostics")
def _branch_influence_diagnostics(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    fit = _fit_ols(ctx, "影响点诊断", min_predictors=1)
    if "error" in fit:
        summary.append(fit["error"])
        return
    predictors, model, n = fit["predictors"], fit["model"], fit["n"]
    p = len(predictors) + 1  # parameters incl. intercept (for thresholds)

    try:
        infl = model.get_influence()
        leverage = np.asarray(infl.hat_matrix_diag, dtype=float)
        student = np.asarray(infl.resid_studentized_external, dtype=float)  # externally
        cooks_d = np.asarray(infl.cooks_distance[0], dtype=float)
        dffits = np.asarray(infl.dffits[0], dtype=float)
    except Exception as err:  # noqa: BLE001
        summary.append(f"影响点诊断跳过：影响量计算失败（{err}）。")
        return

    # Standard heuristic thresholds (Belsley/Kuh/Welsch 1980; Cook & Weisberg 1982):
    cooks_thr = 4.0 / n
    dffits_thr = 2.0 * np.sqrt(p / n)
    lev_thr = 2.0 * p / n

    flag_cooks = cooks_d > cooks_thr
    flag_dffits = np.abs(dffits) > dffits_thr
    flag_lev = leverage > lev_thr
    flagged = flag_cooks | flag_dffits | flag_lev

    estimates["max_cooks_d"] = float(np.nanmax(cooks_d)) if cooks_d.size else float("nan")
    estimates["n_influential_cooks"] = float(int(np.sum(flag_cooks)))
    estimates["max_leverage"] = float(np.nanmax(leverage)) if leverage.size else float("nan")
    estimates["n_high_leverage"] = float(int(np.sum(flag_lev)))
    estimates["max_abs_dffits"] = float(np.nanmax(np.abs(dffits))) if dffits.size else float("nan")
    estimates["n"] = float(n)

    # --- CSV: per-observation diagnostics ----------------------------------- #
    import pandas as pd

    tab = pd.DataFrame(
        {
            "obs": np.arange(n),
            "leverage": leverage,
            "studentized_resid": student,
            "cooks_d": cooks_d,
            "dffits": dffits,
            "flag": flagged,
        }
    )
    tab.to_csv(d / "influence.csv", index=False, encoding="utf-8")
    files.append("influence.csv")

    # --- PNG: influence plot (leverage vs studentized resid, size ∝ Cook's D) #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cd = np.nan_to_num(cooks_d, nan=0.0)
        cd_max = float(cd.max()) if cd.size and cd.max() > 0 else 1.0
        sizes = 30.0 + 600.0 * (cd / cd_max)
        colors = np.where(flagged, "#b85450", "#6c8ebf")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(leverage, student, s=sizes, c=colors, alpha=0.6, edgecolor="white")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhline(2, color="#d6b656", ls="--", lw=1.0, label="|studentized| = 2")
        ax.axhline(-2, color="#d6b656", ls="--", lw=1.0)
        ax.axvline(lev_thr, color="#82b366", ls=":", lw=1.2, label=f"leverage = 2p/n = {lev_thr:.3g}")
        ax.set_xlabel("leverage (hat diag)")
        ax.set_ylabel("externally studentized residual")
        ax.set_title("Influence plot (point size proportional to Cook's D)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "influence_plot.png", dpi=150)
        plt.close(fig)
        files.append("influence_plot.png")
    except Exception:  # noqa: BLE001
        pass

    n_flagged = int(np.sum(flagged))
    summary.append(
        f"{entry.method} 完成：因变量 {fit['outcome']}，n={n}（参数 {p} 个）。"
        f" 最大 Cook's D={estimates['max_cooks_d']:.3g}（阈值 4/n={cooks_thr:.3g}，"
        f"超阈 {int(estimates['n_influential_cooks'])} 个）；最大杠杆={estimates['max_leverage']:.3g}"
        f"（阈值 2p/n={lev_thr:.3g}，超阈 {int(estimates['n_high_leverage'])} 个）；"
        f" 共 {n_flagged} 个观测被任一准则标记。明细见 influence.csv。"
    )
    summary.append(
        "⚠ 假定：阈值（4/n、2·√(p/n)、2p/n）是经验启发式，被标记 ≠ 证明该点是错误——"
        "应人工检查被标记点，切勿自动删除；高杠杆 ≠ 高影响（Cook's D 同时综合了杠杆与残差，"
        "更全面）；config 可覆盖 outcome/predictors。"
    )

    code += [
        "import statsmodels.api as sm; import numpy as np",
        f"X = sm.add_constant(df[{predictors}].dropna().to_numpy(float))",
        f"m = sm.OLS(df['{fit['outcome']}'].dropna().to_numpy(float), X).fit()",
        "infl = m.get_influence()",
        "lev = infl.hat_matrix_diag; cooks = infl.cooks_distance[0]; dffits = infl.dffits[0]",
        "stud = infl.resid_studentized_external",
        "flag = (cooks > 4/len(cooks)) | (np.abs(dffits) > 2*np.sqrt(X.shape[1]/len(cooks)))",
    ]
