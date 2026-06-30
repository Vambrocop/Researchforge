"""Auto-diagnose the data, then recommend WITH A PLAN (smarter auto-selection, v1.5).

This layer sits ON TOP of the rigor/score recommend layer and does **not** change
any run-time defaults. The column-only ``DataFingerprint`` can't see value-level
structure, so here we run a handful of cheap statistical diagnostics on the actual
data and turn each finding into an actionable model-choice nudge — e.g. "the count
outcome is overdispersed → prefer negative_binomial_regression over poisson_regression".

The outcome column used for the diagnostics reuses the slice-1 semantic role hint
(``fp.likely_outcome``) when present, so the two smarter-selection slices compound:
roles.py says *which* column is the outcome, diagnostics.py says *which model* it wants.

All method ids referenced in ``prefer`` / ``over`` are real catalog ids; ``build_plan``
additionally filters them to the loaded catalog so a future rename can't dangle.
Everything degrades honestly: a diagnostic that can't be computed (too few rows,
wrong column types, missing scipy) is simply omitted — never guessed, never crashes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from researchforge.profiler.fingerprint import DataFingerprint


class Diagnostic(BaseModel):
    """One value-level finding and the model choice it argues for."""

    code: str                 # machine code, e.g. "overdispersion"
    finding: str              # Chinese human-readable headline
    detail: str = ""          # the numbers behind it
    prefer: list[str] = Field(default_factory=list)  # method ids this finding favors
    over: list[str] = Field(default_factory=list)    # method ids it argues against


class DiagnosticPlan(BaseModel):
    """The diagnostic read on a dataset plus the column it treated as the outcome."""

    outcome: Optional[str] = None      # column used as the analysis outcome (if any)
    diagnostics: list[Diagnostic] = Field(default_factory=list)


# ── helpers ──────────────────────────────────────────────────────────────────
def _num(s: pd.Series) -> pd.Series:
    """Coerce to numeric, dropping anything unparseable (never raises)."""
    return pd.to_numeric(s, errors="coerce").dropna()


def _ols_resid(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS via lstsq with an intercept. Returns (beta, residuals)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta, y - Xc @ beta


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² of regressing y on X (with intercept)."""
    _, resid = _ols_resid(y, X)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - float(np.sum(resid ** 2)) / ss_tot


def _analysis_cols(fp: DataFingerprint) -> list:
    """Columns eligible as analysis variables (exclude unit/time/id/geo)."""
    skip = {fp.unit_col, fp.time_col}
    return [c for c in fp.columns if c.name not in skip and c.kind not in {"id", "geo", "datetime"}]


def _pick_outcome(df: pd.DataFrame, fp: DataFingerprint) -> Optional[str]:
    """Outcome for diagnostics: the slice-1 role hint, else first continuous, else
    first count — restricted to columns actually present in the frame."""
    cols = _analysis_cols(fp)
    present = {c.name for c in cols} & set(df.columns)
    if fp.likely_outcome in present:
        return fp.likely_outcome
    for kind in ("continuous", "count"):
        for c in cols:
            if c.kind == kind and c.name in present:
                return c.name
    return None


def _numeric_predictors(df: pd.DataFrame, fp: DataFingerprint, outcome: str) -> list[str]:
    """Continuous/count columns other than the outcome (the regressors)."""
    return [
        c.name for c in _analysis_cols(fp)
        if c.kind in {"continuous", "count"} and c.name != outcome and c.name in df.columns
    ]


# ── individual diagnostics ───────────────────────────────────────────────────
def _diag_count(df: pd.DataFrame, fp: DataFingerprint, outcome: Optional[str]) -> list[Diagnostic]:
    """Overdispersion + zero-inflation for a count outcome (Poisson-fit Pearson φ
    and excess-zero comparison vs the Poisson-implied zero rate)."""
    from scipy import stats

    out: list[Diagnostic] = []
    # the count column: the role-hinted outcome if it's count, else first count column
    col = None
    if outcome and (ci := fp.column(outcome)) and ci.kind == "count":
        col = outcome
    else:
        for c in _analysis_cols(fp):
            if c.kind == "count" and c.name in df.columns:
                col = c.name
                break
    if col is None:
        return out

    y = _num(df[col]).to_numpy()
    n = len(y)
    if n < 10 or y.min() < 0 or y.mean() <= 0:
        return out
    mean = float(y.mean())
    var = float(y.var(ddof=1))
    phi = var / mean  # = Pearson X²/(n−1) for a Poisson fit to the mean
    x2 = phi * (n - 1)
    p = float(stats.chi2.sf(x2, n - 1))
    overdispersed = phi > 1.5 and p < 0.05
    if overdispersed:
        out.append(Diagnostic(
            code="overdispersion",
            finding=f"计数结果「{col}」过离散（方差 ≫ 均值）",
            detail=f"离散系数 φ=var/mean={phi:.2f}（Poisson 应≈1）, Pearson χ²p={p:.3g}"
                   "（无条件检验，未扣协变量；真实回归扣掉协变量后离散可能减小）",
            prefer=["negative_binomial_regression", "zero_inflated_negbin", "tweedie_glm"],
            over=["poisson_regression"],
        ))

    # Excess zeros vs the Poisson-implied zero rate exp(−mean). NOTE: overdispersion
    # alone (e.g. plain NB) also produces more zeros than a same-mean Poisson, so the
    # honest headline is "more zeros than Poisson", not a structural-zero claim — and
    # zero_inflated_negbin (which absorbs both) leads the suggestions.
    p0_obs = float(np.mean(y == 0))
    p0_exp = float(np.exp(-mean))
    if p0_obs > 0.05 and p0_obs - p0_exp > 0.10 and p0_obs > 1.2 * p0_exp:
        out.append(Diagnostic(
            code="zero_inflation",
            finding=f"计数结果「{col}」零的比例高于 Poisson 预期（可能零膨胀，也可能仅过离散）",
            detail=f"观测零比例={p0_obs:.2f}, Poisson 预期≈{p0_exp:.2f}"
                   + ("；已检出过离散，多余的零可能来自过离散而非结构零" if overdispersed else ""),
            prefer=["zero_inflated_negbin", "zero_inflated_poisson"],
            over=["poisson_regression"],
        ))
    return out


def _diag_normality(df: pd.DataFrame, fp: DataFingerprint, outcome: Optional[str]) -> list[Diagnostic]:
    """Non-normal / heavy-skew continuous outcome → nonparametric / robust / bootstrap.

    Gated on a *practically meaningful* departure (|skew|>1 or |excess kurtosis|>2),
    not only a significance test — at large n a normality test rejects on trivial
    departures, and at small n it has no power. The test p is reported as support."""
    from scipy import stats

    if not outcome or not (ci := fp.column(outcome)) or ci.kind != "continuous":
        return []
    y = _num(df[outcome]).to_numpy()
    n = len(y)
    if n < 8 or np.allclose(y, y[0]):
        return []
    skew = float(stats.skew(y, bias=False))
    kurt = float(stats.kurtosis(y, fisher=True, bias=False))  # excess kurtosis
    try:
        p = float(stats.normaltest(y).pvalue) if n >= 20 else float(stats.shapiro(y).pvalue)
    except Exception:
        p = float("nan")
    if abs(skew) > 1.0 or abs(kurt) > 2.0:
        return [Diagnostic(
            code="non_normal_outcome",
            finding=f"连续结果「{outcome}」明显偏离正态（偏态/厚尾）",
            detail=f"偏度={skew:.2f}, 超额峰度={kurt:.2f}, 正态检验 p={p:.3g}",
            prefer=["mann_whitney", "kruskal_wallis", "robust_regression",
                    "quantile_regression", "bootstrap_ci"],
            over=["ols_regression", "anova_oneway"],
        )]
    return []


def _diag_heteroskedasticity(df: pd.DataFrame, fp: DataFingerprint, outcome: Optional[str]) -> list[Diagnostic]:
    """Koenker's studentized Breusch-Pagan: regress squared OLS residuals on the
    predictors, LM = n·R²_aux ~ χ²(k)."""
    from scipy import stats

    if not outcome or not (ci := fp.column(outcome)) or ci.kind != "continuous":
        return []
    preds = _numeric_predictors(df, fp, outcome)
    if not preds:
        return []
    sub = df[[outcome] + preds].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(sub), len(preds)
    if n < 3 * k + 10:
        return []
    y = sub[outcome].to_numpy(float)
    X = sub[preds].to_numpy(float)
    if np.isclose(X.std(axis=0), 0.0).any():
        return []
    _, resid = _ols_resid(y, X)
    lm = n * _r2(resid ** 2, X)  # Koenker robust BP statistic
    p = float(stats.chi2.sf(lm, k))
    if p < 0.05:
        return [Diagnostic(
            code="heteroskedasticity",
            finding=f"回归残差异方差（结果「{outcome}」的离散随预测变量变化）",
            detail=f"Koenker BP LM={lm:.2f}, df={k}, p={p:.3g}",
            prefer=["heteroskedasticity_test", "robust_regression", "quantile_regression"],
            over=[],
        )]
    return []


def _diag_multicollinearity(df: pd.DataFrame, fp: DataFingerprint, outcome: Optional[str]) -> list[Diagnostic]:
    """Variance Inflation Factor among the numeric predictors; flag max VIF > 10."""
    preds = _numeric_predictors(df, fp, outcome or "")
    if len(preds) < 2:
        return []
    sub = df[preds].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < len(preds) + 5:
        return []
    cols = [c for c in preds if sub[c].std() > 0]
    if len(cols) < 2:
        return []
    M = sub[cols].to_numpy(float)
    worst_col, worst_vif = None, 0.0
    for j, name in enumerate(cols):
        others = np.delete(M, j, axis=1)
        r2 = _r2(M[:, j], others)
        vif = 1.0 / max(1e-12, 1.0 - r2)
        if vif > worst_vif:
            worst_col, worst_vif = name, vif
    if worst_vif > 10:
        return [Diagnostic(
            code="multicollinearity",
            finding=f"预测变量多重共线（「{worst_col}」可被其他变量线性预测）",
            detail=f"最大 VIF={worst_vif:.1f}（>10 为强共线）",
            prefer=["vif_multicollinearity", "regularized_regression",
                    "pls_regression", "dominance_analysis"],
            over=["ols_regression"],
        )]
    return []


def _diag_outliers(fp: DataFingerprint) -> list[Diagnostic]:
    """Reuse the profiler's outlier findings — argue for robust/influence tools."""
    cols = [iss.column for iss in fp.issues if iss.kind == "outliers" and iss.column]
    if not cols:
        return []
    shown = "、".join(dict.fromkeys(cols))  # de-dup, keep order
    return [Diagnostic(
        code="outliers",
        finding=f"检测到离群值（列：{shown}）",
        detail="离群值会拉偏最小二乘估计与经典 SE",
        prefer=["robust_regression", "influence_diagnostics"],
        over=[],
    )]


def _diag_small_sample(fp: DataFingerprint) -> list[Diagnostic]:
    """Small n → resampling / exact inference over large-sample approximations."""
    if fp.n_rows >= 30:
        return []
    return [Diagnostic(
        code="small_sample",
        finding=f"样本量偏小（n={fp.n_rows}）",
        detail="大样本近似（正态/卡方）在小样本下不可靠",
        prefer=["bootstrap_ci", "permutation_test"],
        over=[],
    )]


# ── non-GLM diagnostics (Stage 5: time-series / survival / missingness) ───────
def _diag_timeseries(df: pd.DataFrame, fp: DataFingerprint) -> list[Diagnostic]:
    """Time-series structure → the right temporal model. Non-stationarity (ADF unit
    root) argues for differencing/ARIMA over methods that treat the series as iid;
    volatility clustering (ARCH effect = autocorrelated squared series) argues for GARCH."""
    if not getattr(fp, "is_timeseries", False):
        return []
    col = next((c.name for c in _analysis_cols(fp)
                if c.kind in {"continuous", "count"} and c.name in df.columns and c.name != fp.time_col),
               None)
    if col is None:
        return []
    y = _num(df[col]).to_numpy()
    n = len(y)
    if n < 20 or np.allclose(y, y[0]):
        return []
    out: list[Diagnostic] = []

    adf_p = float("nan")
    try:
        from statsmodels.tsa.stattools import adfuller

        adf_p = float(adfuller(y, autolag="AIC")[1])
    except Exception:
        pass
    if adf_p == adf_p and adf_p > 0.05:
        out.append(Diagnostic(
            code="nonstationary",
            finding=f"时间序列「{col}」可能非平稳（趋势/单位根）",
            detail=f"ADF p={adf_p:.3g}（>0.05 未拒绝单位根）；需差分/去趋势后建模",
            prefer=["arima", "exponential_smoothing", "theta_method", "unobserved_components"],
            over=["correlation", "ols_regression"],
        ))

    lb_p = float("nan")
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        r2 = (y - y.mean()) ** 2
        lb_p = float(acorr_ljungbox(r2, lags=[min(10, n // 5)], return_df=True)["lb_pvalue"].iloc[0])
    except Exception:
        pass
    if lb_p == lb_p and lb_p < 0.05:
        out.append(Diagnostic(
            code="volatility_clustering",
            finding=f"时间序列「{col}」存在波动聚集（ARCH 效应）",
            detail=f"平方序列 Ljung-Box p={lb_p:.3g}（<0.05 表条件异方差/波动随时间成簇）",
            prefer=["garch"],
            over=[],
        ))
    return out


def _diag_survival(df: pd.DataFrame, fp: DataFingerprint) -> list[Diagnostic]:
    """Time-to-event data (a duration column + an event/censoring indicator) → survival
    models. Ordinary regression on a censored outcome is biased (it can't tell a short
    follow-up from a short survival)."""
    from researchforge.recommender.affinity import data_signals

    if not data_signals(fp)["has_survival"]:
        return []
    return [Diagnostic(
        code="survival_data",
        finding="检测到生存/时间-事件数据（时长列 + 事件/删失指示）",
        detail="对删失数据用普通回归会有偏（无法区分随访短与生存短）；应同时建模时长与删失",
        prefer=["survival_analysis", "cox_ph_diagnostics", "parametric_survival",
                "stratified_cox", "competing_risks", "rmst"],
        over=["ols_regression", "logistic_regression"],
    )]


def _diag_missingness(df: pd.DataFrame, fp: DataFingerprint) -> list[Diagnostic]:
    """Non-trivial missingness → multiple imputation. Listwise deletion drops cases and
    can bias estimates when data are not missing completely at random."""
    miss_cols = [iss.column for iss in fp.issues if iss.kind == "missing" and iss.column]
    total = int(df.size)
    if not miss_cols or total == 0:
        return []
    miss_rate = float(df.isna().sum().sum()) / total
    if miss_rate < 0.05:
        return []
    shown = "、".join(dict.fromkeys(miss_cols))
    return [Diagnostic(
        code="missing_data",
        finding=f"存在缺失数据（列：{shown}；整体缺失率 {miss_rate:.0%}）",
        detail="列表删除（listwise）丢样本且在非 MCAR 下有偏；多重插补按 Rubin 规则合并插补不确定性",
        prefer=["mice_imputation", "missingness_diagnosis"],
        over=[],
    )]


# ── orchestration ────────────────────────────────────────────────────────────
def diagnose_data(df: pd.DataFrame, fp: DataFingerprint) -> list[Diagnostic]:
    """Run every cheap diagnostic and collect the findings (order = severity-ish).

    Each sub-diagnostic is independently wrapped so a genuine statistical edge case
    (degenerate scipy input, a singular design, a missing optional import) can't sink
    the whole plan. The catch is deliberately NARROW — wiring bugs (AttributeError /
    TypeError / NameError) are left to surface in tests, not silently swallowed
    (an earlier `.any()`-on-scalar bug hid exactly this way)."""
    _expected = (ValueError, KeyError, IndexError, ZeroDivisionError,
                 np.linalg.LinAlgError, ImportError)
    outcome = _pick_outcome(df, fp)
    out: list[Diagnostic] = []
    for fn in (
        lambda: _diag_count(df, fp, outcome),
        lambda: _diag_normality(df, fp, outcome),
        lambda: _diag_heteroskedasticity(df, fp, outcome),
        lambda: _diag_multicollinearity(df, fp, outcome),
        lambda: _diag_outliers(fp),
        lambda: _diag_small_sample(fp),
        lambda: _diag_timeseries(df, fp),
        lambda: _diag_survival(df, fp),
        lambda: _diag_missingness(df, fp),
    ):
        try:
            out.extend(fn())
        except _expected:
            continue
    return out


def build_plan(fp: DataFingerprint, df: Optional[pd.DataFrame] = None, catalog=None) -> DiagnosticPlan:
    """Top-level entry: diagnose the data and return an actionable plan.

    ``df`` is read from ``fp.path`` (via the robust ingest reader) when not given,
    so callers that only hold a fingerprint still work. When a ``catalog`` is
    supplied, ``prefer`` / ``over`` are filtered to ids it actually contains."""
    if df is None:
        try:
            from researchforge.profiler.ingest import read_table

            df = read_table(fp.path)
        except Exception:
            return DiagnosticPlan(outcome=None, diagnostics=[])

    outcome = _pick_outcome(df, fp)
    diags = diagnose_data(df, fp)

    if catalog is not None:
        ids = {e.id for e in catalog.all()}
        for dgn in diags:
            dgn.prefer = [m for m in dgn.prefer if m in ids]
            dgn.over = [m for m in dgn.over if m in ids]
        diags = [dgn for dgn in diags if dgn.prefer]  # drop findings with no live method

    return DiagnosticPlan(outcome=outcome, diagnostics=diags)
