"""Branch handlers for the MISSING-DATA family (principled missing-data handling).

The engine elsewhere only does simple cleaning fills; this family adds the gold-
standard statistical machinery for missing data:

  - mice_imputation       — Multiple Imputation by Chained Equations + Rubin's rules
                            (pool a regression across M imputed datasets; report FMI,
                            and the complete-case regression alongside for contrast).
  - missingness_diagnosis — missing-data patterns + an MCAR association screen.

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法>跳过：<原因>" appended to summary + return
    (never crash / fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    a float-only ``estimates`` dict (plain floats / float("nan"), never tuples/strings),
    a Chinese ``summary`` with ⚠ assumption/bias disclosures, reproducible ``code``.
  * Column resolution from ``fp.columns`` (.name/.kind/.n_unique/.n_missing), excluding
    fp.unit_col / fp.time_col; coercion via pd.to_numeric(errors="coerce").

Pure Python: numpy / scipy / pandas / statsmodels / sklearn / matplotlib only. NO R.

NOTES on the two ambiguous choices (per STOP-AND-REPORT):
  * Rubin df: we use the SIMPLE Rubin (1987) formula
        df = (M-1) * (1 + Ubar / ((1 + 1/M) * B))^2
    guarded against B==0 (then df -> a large finite number, i.e. no between-imputation
    variability, so the pooled SE is essentially the within-imputation SE). The
    Barnard-Rubin small-sample correction is NOT applied (it needs the complete-data
    df, which is well-defined here but the simpler form is the standard default and is
    what statsmodels' own MICE reports for large samples). Disclosed in code comments.
  * MCAR test: we implement an MCAR ASSOCIATION SCREEN (labelled "MCAR 关联筛查",
    NOT "Little's test"): for each column WITH missing values we regress its missing-
    indicator on the OTHER numeric columns via logistic regression and combine the
    per-column likelihood-ratio chi-squares into one overall chi-square (df summed).
    A significant result is evidence AGAINST MCAR (missingness is associated with
    observed values -> at best MAR). This is a defensible, robust alternative to
    Little's (1988) statistic (which needs pattern-group ML means / a pooled covariance
    that is fragile with many sparse patterns). The choice is disclosed in the summary.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _numeric_frame(ctx: Ctx):
    """Return (numeric_df, names) of analysis columns coerced to float.

    Excludes the panel unit / time columns. Accepts continuous / count / binary /
    id kinds (an integer column of all-distinct values profiles as ``id`` per the
    CLAUDE.md「id 陷阱」, but is a legitimate number). Drops all-missing columns.
    Coercion is via pd.to_numeric(errors="coerce") — never .astype(float).
    """
    import pandas as pd

    df, fp = ctx.df, ctx.fp
    excl = {fp.unit_col, fp.time_col}
    names: list[str] = []
    for c in fp.columns:
        if c.name in excl or c.name not in df.columns:
            continue
        if c.kind in ("continuous", "count", "binary", "id"):
            names.append(c.name)
    cols = {}
    kept: list[str] = []
    for n in names:
        s = pd.to_numeric(df[n], errors="coerce")
        if s.notna().sum() == 0:  # all-missing after coercion -> useless
            continue
        cols[n] = s
        kept.append(n)
    num = pd.DataFrame(cols) if cols else pd.DataFrame()
    return num, kept


def _bar_plot(labels, values, title, xlabel, path, errs=None):
    """Best-effort horizontal bar plot (ENGLISH labels). Never raises."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, max(2.4, 0.5 * n + 1.2)))
        ypos = list(range(n))
        if errs is not None:
            ax.barh(ypos, values, xerr=errs, color="#4C72B0", capsize=3)
        else:
            ax.barh(ypos, values, color="#4C72B0")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 1) mice_imputation — MICE + Rubin's rules
# --------------------------------------------------------------------------- #
@register("mice_imputation")
def _branch_mice_imputation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    num, names = _numeric_frame(ctx)
    if num.empty or len(names) < 2:
        summary.append("多重插补跳过：可用数值列不足 2（需结果 + 至少 1 个预测变量）。")
        return

    n_total = int(len(num))
    # missingness on the analysis columns
    n_missing_total = int(num.isna().sum().sum())
    if n_missing_total == 0:
        summary.append(
            "多重插补跳过：所选数值列无缺失值，无需多重插补；可直接用 ols_regression。"
        )
        return

    # --- resolve outcome / predictors (regression-family convention) ----------
    # outcome = first continuous column (config outcome override); predictors =
    # remaining numeric columns (config predictors override).
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name in names]
    outcome = cfg.get("outcome") if cfg.get("outcome") in names else (cont[0] if cont else names[0])
    forced = [c for c in (cfg.get("predictors") or []) if c in names and c != outcome]
    if forced:
        predictors = forced[:12]
    else:
        predictors = [c for c in names if c != outcome][:12]
    if not predictors:
        summary.append("多重插补跳过：除结果变量外无可用预测变量。")
        return

    model_cols = [outcome] + predictors
    work = num[model_cols].copy()

    # --- M imputations --------------------------------------------------------
    try:
        m = int(cfg.get("m_imputations", 10))
    except Exception:
        m = 10
    m = max(2, min(m, 100))  # Rubin's rules need M>=2; cap to keep runtime sane
    try:
        seed = int(cfg.get("seed", 12345))
    except Exception:
        seed = 12345

    completed, backend = _run_mice(work, m, seed)
    if not completed:
        summary.append("多重插补跳过：MICE 后端不可用或插补失败（statsmodels / sklearn 均未产出）。")
        return

    # --- fit OLS on each completed dataset ------------------------------------
    import statsmodels.api as sm

    param_names = ["const"] + predictors
    coefs = {p: [] for p in param_names}
    ses = {p: [] for p in param_names}
    n_used = 0
    for comp in completed:
        try:
            y = comp[outcome].to_numpy(dtype=float)
            X = comp[predictors].to_numpy(dtype=float)
            X = sm.add_constant(X, has_constant="add")
            res = sm.OLS(y, X).fit()
            params = np.asarray(res.params, dtype=float)
            bse = np.asarray(res.bse, dtype=float)
            if params.shape[0] != len(param_names):
                continue
            for i, p in enumerate(param_names):
                coefs[p].append(float(params[i]))
                ses[p].append(float(bse[i]))
            n_used += 1
        except Exception:
            continue

    if n_used < 2:
        summary.append("多重插补跳过：插补后回归在 <2 个数据集上成功，无法用 Rubin 规则合并。")
        return

    # --- pool via Rubin's rules ----------------------------------------------
    # qbar = mean of M estimates
    # Ubar = mean of M squared SEs (within-imputation variance)
    # B    = sample variance of the M estimates (between-imputation variance, ddof=1)
    # T    = Ubar + (1 + 1/M) * B  (total variance)
    # SE   = sqrt(T)
    # FMI  = (1 + 1/M) * B / T
    # df   = (M-1) * (1 + Ubar / ((1+1/M)*B))^2   [Rubin 1987; guard B==0]
    Mu = float(n_used)
    rows = []
    pooled = {}  # name -> (coef, se, t, dfree, fmi)
    for p in param_names:
        est = np.asarray(coefs[p], dtype=float)
        se_arr = np.asarray(ses[p], dtype=float)
        qbar = float(np.mean(est))
        Ubar = float(np.mean(se_arr ** 2))
        B = float(np.var(est, ddof=1)) if Mu > 1 else 0.0
        infl = (1.0 + 1.0 / Mu) * B
        T = Ubar + infl
        se = float(np.sqrt(T)) if T > 0 else float("nan")
        fmi = float(infl / T) if T > 0 else float("nan")
        if infl > 0 and se > 0:
            dfree = (Mu - 1.0) * (1.0 + Ubar / infl) ** 2
        else:
            # no between-imputation variance: SE ≈ within-imputation SE, treat df as large
            dfree = 1.0e6
        tval = float(qbar / se) if se and se > 0 else float("nan")
        pooled[p] = (qbar, se, tval, dfree, fmi)
        rows.append({"parameter": p, "pooled_coef": qbar, "pooled_se": se,
                     "t": tval, "df": dfree, "fmi": fmi})

    # --- complete-case (listwise-deletion) regression, for contrast ----------
    cc = work.dropna()
    n_complete_case = int(len(cc))
    cc_coef = {p: float("nan") for p in param_names}
    cc_ok = False
    if n_complete_case >= len(param_names) + 1:
        try:
            ycc = cc[outcome].to_numpy(dtype=float)
            Xcc = sm.add_constant(cc[predictors].to_numpy(dtype=float), has_constant="add")
            rescc = sm.OLS(ycc, Xcc).fit()
            cp = np.asarray(rescc.params, dtype=float)
            if cp.shape[0] == len(param_names):
                for i, p in enumerate(param_names):
                    cc_coef[p] = float(cp[i])
                cc_ok = True
        except Exception:
            cc_ok = False

    # --- products: coefficients CSV ------------------------------------------
    coef_df = pd.DataFrame(rows)
    coef_df["complete_case_coef"] = [cc_coef[p] for p in coef_df["parameter"]]
    coef_df = coef_df[["parameter", "pooled_coef", "pooled_se", "t", "df",
                       "fmi", "complete_case_coef"]]
    coef_df.to_csv(d / "mice_coefficients.csv", index=False, encoding="utf-8")
    files.append("mice_coefficients.csv")

    # --- products: forest/bar PNG of pooled coefs (predictors only) ----------
    plot_names = predictors  # skip the intercept for readability
    pcoef = [pooled[p][0] for p in plot_names]
    pse = [pooled[p][1] for p in plot_names]
    if _bar_plot(plot_names, pcoef,
                 f"MICE-pooled OLS coefficients (outcome: {outcome})",
                 "Pooled coefficient", d / "mice_pooled_coefficients.png",
                 errs=pse):
        files.append("mice_pooled_coefficients.png")

    # --- estimates (plain floats only) ---------------------------------------
    for p in predictors:
        qbar, se, tval, dfree, fmi = pooled[p]
        estimates[f"coef__{p}"] = round(qbar, 5)
        estimates[f"se__{p}"] = round(se, 5) if se == se else float("nan")
        estimates[f"fmi__{p}"] = round(fmi, 5) if fmi == fmi else float("nan")
    qb0, se0, _, _, fmi0 = pooled["const"]
    estimates["coef__const"] = round(qb0, 5)
    estimates["n_imputations"] = float(n_used)
    estimates["n_complete_case"] = float(n_complete_case)
    estimates["n_total"] = float(n_total)
    estimates["n_missing_cells"] = float(n_missing_total)
    estimates["max_fmi"] = round(
        float(np.nanmax([pooled[p][4] for p in predictors])), 5
    )

    # --- Chinese summary with ⚠ disclosures ----------------------------------
    key = predictors[0]
    kqb, kse, _, _, kfmi = pooled[key]
    dropped = n_total - n_complete_case
    summary.append(
        f"多重插补（MICE，{backend}）完成：在 M={n_used} 个插补数据集上拟合 OLS "
        f"（结果变量 {outcome}，预测变量 {len(predictors)} 个），按 Rubin 规则合并。"
        f"关键预测变量 {key}：合并系数 {kqb:.4f}（SE {kse:.4f}，FMI {kfmi:.2f}）。"
    )
    summary.append(
        f"完整个案（列表删除）回归对照：保留 {n_complete_case}/{n_total} 行"
        f"（删除 {dropped} 行含缺失），"
        + (f"{key} 完整个案系数 {cc_coef[key]:.4f}。" if cc_ok else "样本不足、未拟合。")
    )
    summary.append(
        "⚠ MICE 假设数据为 MAR（随机缺失，可由观测变量解释）；插补模型即其余变量的链式方程，"
        "模型设定错误会使插补有偏。FMI = 缺失信息比例：FMI 高（如 >0.5）表示结论对插补不确定性敏感、需谨慎。"
    )

    # --- reproducible code ----------------------------------------------------
    code += [
        "import numpy as np, pandas as pd, statsmodels.api as sm",
        "# MICE multiple imputation + Rubin's rules pooling of an OLS regression",
        f"model_cols = {model_cols!r}",
        "work = df[model_cols].apply(pd.to_numeric, errors='coerce')",
        "# --- M imputations (statsmodels MICEData if available, else sklearn IterativeImputer) ---",
        f"M = {n_used}",
        "completed = []  # list of M completed DataFrames (see _run_mice in missing_data.py)",
        "# fit OLS on each, collect coef + SE per parameter, then pool:",
        "#   qbar=mean(est); Ubar=mean(se**2); B=var(est,ddof=1)",
        "#   T=Ubar+(1+1/M)*B; SE=sqrt(T); FMI=(1+1/M)*B/T",
        "#   df=(M-1)*(1+Ubar/((1+1/M)*B))**2   # Rubin 1987 (guard B==0)",
        f"outcome, predictors = {outcome!r}, {predictors!r}",
    ]


def _run_mice(work, m: int, seed: int):
    """Produce M completed (imputed) copies of ``work`` (a numeric DataFrame).

    Backend order (CLAUDE.md R-bridge-style optional + graceful degrade):
      1. statsmodels MICE + MICEData (the chained-equations gold standard), OR
      2. sklearn IterativeImputer with M random-seeded fits (each a stochastic draw).

    Returns (list_of_completed_DataFrames, backend_name) or ([], "") on failure.
    """
    import numpy as np
    import pandas as pd

    cols = list(work.columns)

    # --- backend 1: statsmodels MICEData (true chained equations) -------------
    try:
        from statsmodels.imputation import mice as sm_mice

        # MICEData needs a DataFrame; it imputes in place across update cycles.
        completed = []
        # statsmodels uses the global numpy RNG inside MICEData; seed it for repro.
        np.random.seed(seed)
        base = work.reset_index(drop=True).copy()
        imp = sm_mice.MICEData(base)
        # burn-in then collect M imputations spaced by a few update cycles
        for _ in range(5):
            imp.update_all()
        for _k in range(m):
            for _ in range(3):
                imp.update_all()
            completed.append(imp.data[cols].copy())
        # If statsmodels left any residual NaN (rare), reject this backend.
        if completed and all(c.notna().all().all() for c in completed):
            return completed, "statsmodels"
    except Exception:
        pass

    # --- backend 2: sklearn IterativeImputer (M random-seeded draws) ----------
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        arr = work.to_numpy(dtype=float)
        completed = []
        for k in range(m):
            imp = IterativeImputer(
                max_iter=20, sample_posterior=True, random_state=seed + k
            )
            filled = imp.fit_transform(arr)
            completed.append(pd.DataFrame(filled, columns=cols))
        if completed and all(c.notna().all().all() for c in completed):
            return completed, "sklearn IterativeImputer"
    except Exception:
        pass

    return [], ""


# --------------------------------------------------------------------------- #
# 2) missingness_diagnosis — patterns + MCAR association screen
# --------------------------------------------------------------------------- #
@register("missingness_diagnosis")
def _branch_missingness_diagnosis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    # Scan ALL analysis columns (exclude unit/time); keep original dtypes for the
    # per-column missing counts, but coerce to numeric for the MCAR screen.
    fp_excl = {fp.unit_col, fp.time_col}
    scan_cols = [c.name for c in fp.columns
                 if c.name not in fp_excl and c.name in df.columns]
    if not scan_cols:
        summary.append("缺失诊断跳过：无可分析列。")
        return

    work = df[scan_cols]
    n_total = int(len(work))
    if n_total == 0:
        summary.append("缺失诊断跳过：数据为空。")
        return

    miss = work.isna()
    per_col_n = miss.sum().astype(int)
    per_col_rate = (per_col_n / n_total).astype(float)
    cols_with_missing = [c for c in scan_cols if int(per_col_n[c]) > 0]
    overall_rate = float(miss.to_numpy().mean())

    if not cols_with_missing:
        # Honest: nothing missing. Still emit a per-column table so the report exists.
        col_df = pd.DataFrame({
            "column": scan_cols,
            "n_missing": [int(per_col_n[c]) for c in scan_cols],
            "missing_rate": [float(per_col_rate[c]) for c in scan_cols],
        })
        col_df.to_csv(d / "missingness_by_column.csv", index=False, encoding="utf-8")
        files.append("missingness_by_column.csv")
        estimates["overall_missing_rate"] = 0.0
        estimates["n_patterns"] = 1.0
        estimates["mcar_stat"] = float("nan")
        estimates["mcar_p"] = float("nan")
        estimates["n_cols_with_missing"] = 0.0
        estimates["n_total"] = float(n_total)
        summary.append("缺失诊断：数据无缺失值（缺失率 0%）。无需插补，可直接分析。")
        return

    # --- per-column missingness table ----------------------------------------
    col_df = pd.DataFrame({
        "column": scan_cols,
        "n_missing": [int(per_col_n[c]) for c in scan_cols],
        "missing_rate": [float(per_col_rate[c]) for c in scan_cols],
    }).sort_values("n_missing", ascending=False)
    col_df.to_csv(d / "missingness_by_column.csv", index=False, encoding="utf-8")
    files.append("missingness_by_column.csv")

    # --- distinct missingness patterns + frequencies -------------------------
    # A pattern = the tuple of (missing?) flags across columns (1=missing).
    pat = miss.astype(int)
    pat_counts = pat.value_counts()  # Series indexed by pattern tuple
    pattern_rows = []
    for i, (pattern, cnt) in enumerate(pat_counts.items()):
        flags = pattern if isinstance(pattern, tuple) else (pattern,)
        missing_in = [scan_cols[j] for j, f in enumerate(flags) if f == 1]
        pattern_rows.append({
            "pattern_id": i,
            "n_rows": int(cnt),
            "frequency": float(cnt) / n_total,
            "n_missing_cols": int(sum(flags)),
            "missing_columns": ", ".join(missing_in) if missing_in else "(complete)",
        })
    pat_df = pd.DataFrame(pattern_rows)
    pat_df.to_csv(d / "missingness_patterns.csv", index=False, encoding="utf-8")
    files.append("missingness_patterns.csv")
    n_patterns = int(len(pat_df))

    # --- MCAR association screen (NOT Little's test; disclosed) ---------------
    mcar_stat, mcar_p, mcar_df, mcar_note = _mcar_assoc_screen(work, cols_with_missing)

    # --- products: per-column missing-rate bar PNG ---------------------------
    if _bar_plot(list(col_df["column"]), list(col_df["missing_rate"]),
                 "Missing rate by column", "Missing rate",
                 d / "missingness_by_column.png"):
        files.append("missingness_by_column.png")
    # heatmap of the missingness matrix (rows x columns), best-effort
    if _missing_heatmap(miss, d / "missingness_heatmap.png"):
        files.append("missingness_heatmap.png")

    # --- estimates (plain floats only) ---------------------------------------
    estimates["overall_missing_rate"] = round(overall_rate, 5)
    estimates["n_patterns"] = float(n_patterns)
    estimates["mcar_stat"] = round(float(mcar_stat), 5) if mcar_stat == mcar_stat else float("nan")
    estimates["mcar_p"] = round(float(mcar_p), 5) if mcar_p == mcar_p else float("nan")
    estimates["mcar_df"] = float(mcar_df) if mcar_df == mcar_df else float("nan")
    estimates["n_cols_with_missing"] = float(len(cols_with_missing))
    estimates["n_total"] = float(n_total)

    # --- Chinese summary with ⚠ disclosures ----------------------------------
    summary.append(
        f"缺失诊断完成：{len(cols_with_missing)}/{len(scan_cols)} 列含缺失，"
        f"整体缺失率 {overall_rate * 100:.1f}%，共 {n_patterns} 种缺失模式。"
    )
    if mcar_p == mcar_p:  # not NaN
        verdict = "拒绝 MCAR（缺失与观测值相关，至多为 MAR）" if mcar_p < 0.05 else "未能拒绝 MCAR"
        summary.append(
            f"MCAR 关联筛查（非 Little 检验，见下）：χ²={mcar_stat:.2f}，df={int(mcar_df)}，"
            f"p={mcar_p:.4f} → {verdict}。{mcar_note}"
        )
    else:
        summary.append(f"MCAR 关联筛查未能计算：{mcar_note}")
    summary.append(
        "⚠ 缺失机制三分：MCAR（完全随机）/ MAR（可由观测变量解释）/ MNAR（与未观测的自身取值相关，"
        "不可检验）。本筛查用「缺失指示量 ~ 其余数值列」的 logistic 回归似然比检验组合而成，"
        "标注为「MCAR 关联筛查」而非 Little 检验。⚠ 检验不显著并不能证明 MCAR（可能只是功效不足）；"
        "MNAR 无法从数据本身检验。"
    )

    # --- reproducible code ----------------------------------------------------
    code += [
        "import numpy as np, pandas as pd",
        f"scan_cols = {scan_cols!r}",
        "miss = df[scan_cols].isna()",
        "per_col = miss.sum(); overall = miss.to_numpy().mean()",
        "patterns = miss.astype(int).value_counts()  # distinct missingness patterns",
        "# MCAR association screen: for each col with missing, logistic-regress its",
        "# missing-indicator on the OTHER numeric columns; combine LR chi-squares.",
        "# A significant overall chi-square is evidence AGAINST MCAR (-> at best MAR).",
    ]


def _mcar_assoc_screen(work, cols_with_missing):
    """MCAR ASSOCIATION SCREEN (NOT Little's MCAR test — disclosed).

    For each column with missing values, fit a logistic regression of its
    missing-indicator (1=missing) on the OTHER numeric columns (mean-imputed so the
    predictor matrix is complete), and take the likelihood-ratio chi-square vs the
    intercept-only model. Sum the per-column LR chi-squares and dfs into one overall
    chi-square test (independence across the per-column tests is assumed — disclosed).

    Significant overall => missingness is associated with observed values => the data
    are NOT MCAR (at best MAR). Returns (stat, p_value, df, note); NaN stat/p if no
    test is feasible.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # numeric predictor pool (mean-imputed for use as RHS)
    num = work.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, [c for c in num.columns if num[c].notna().sum() > 0]]
    if num.shape[1] < 2:
        return float("nan"), float("nan"), float("nan"), "数值列不足 2，无法做关联筛查。"

    try:
        import statsmodels.api as sm
    except Exception:
        return float("nan"), float("nan"), float("nan"), "statsmodels 不可用。"

    total_stat = 0.0
    total_df = 0
    n_tested = 0
    n_skipped = 0
    for tgt in cols_with_missing:
        if tgt not in num.columns:
            continue
        y = work[tgt].isna().astype(float).to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue  # no variation in the missing indicator
        others = [c for c in num.columns if c != tgt]
        if not others:
            continue
        Xo = num[others].copy()
        # mean-impute predictors so the design matrix is complete; drop constant cols
        Xo = Xo.fillna(Xo.mean(numeric_only=True))
        keep = [c for c in others if Xo[c].nunique(dropna=True) > 1]
        if not keep:
            continue
        Xo = Xo[keep]
        try:
            import warnings

            Xc = sm.add_constant(Xo.to_numpy(dtype=float), has_constant="add")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # silence separation overflow noise
                full = sm.Logit(y, Xc).fit(disp=0, maxiter=100)
                null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0, maxiter=100)
            # skip non-converged / (near-)separated fits — their LR is inflated and the
            # combined χ² would over-reject MCAR exactly where separation bites.
            if not (full.mle_retvals.get("converged") and null.mle_retvals.get("converged")):
                n_skipped += 1
                continue
            lr = float(2.0 * (full.llf - null.llf))
            ddf = int(Xc.shape[1] - 1)
            if lr >= 0 and ddf >= 1 and np.isfinite(lr):
                total_stat += lr
                total_df += ddf
                n_tested += 1
        except Exception:
            n_skipped += 1
            continue

    if n_tested == 0 or total_df < 1:
        return float("nan"), float("nan"), float("nan"), "无可行的 logistic 关联检验。"
    pval = float(stats.chi2.sf(total_stat, total_df))
    skip_txt = f"，跳过 {n_skipped} 个不收敛/近完全分离列" if n_skipped else ""
    note = (f"基于 {n_tested} 个含缺失列的「缺失指示 ~ 其余数值列」logistic 似然比检验合并"
            f"（假设各列检验独立；合并检验偏宽松{skip_txt}）。")
    return float(total_stat), pval, float(total_df), note


def _missing_heatmap(miss_df, path):
    """Best-effort missingness heatmap (rows x columns; ENGLISH labels). Never raises."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mat = miss_df.astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(max(4, 0.5 * mat.shape[1] + 2), 4))
        ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(list(miss_df.columns), rotation=90)
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title("Missingness map (dark = missing)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False
