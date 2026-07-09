"""Branch handler for GAUSSIAN-PROCESS REGRESSION (Bayesian nonparametric).

A Gaussian process (GP) places a prior over FUNCTIONS and conditions on the data to
return a full posterior PREDICTIVE distribution — a mean and a variance at every
input. It is genuinely Bayesian: with the kernel
``ConstantKernel() * RBF() + WhiteKernel()`` the signal scale, the RBF length-scale
(how wiggly the function may be) and the observation-noise level are learned by
empirical Bayes (maximising the log MARGINAL likelihood — the prior integrated over
all functions), and predictions are the posterior mean ± posterior SD.

scikit-learn is used (robust, always installed). The method is INFERENCE-BEARING in
the sense that it produces a calibrated predictive distribution, but the headline
numbers we report are HONEST out-of-sample performance on a held-out test split:

  * held-out R² and RMSE (generalisation, not in-sample over-fit),
  * 95% predictive-interval COVERAGE = fraction of test y inside mean ± 1.96·σ
    (≈ 0.95 if the GP's uncertainty is calibrated),
  * the learned length-scale(s), noise level and log-marginal-likelihood.

Engine conventions (CLAUDE.md「引擎约定」): ``@register("<id>") def _branch_<id>(ctx)``;
unpack ctx and MUTATE summary/estimates/files/code (never rebind). Products: a
predictions CSV + a PNG (matplotlib Agg, ENGLISH plot labels, best-effort
try/except), float-only ``estimates``, a Chinese ``summary`` with ⚠ disclosures, and
an honest skip + return when the data is unusable (no continuous outcome / no numeric
predictor / < 20 rows) — never crashes / fabricates.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome

_MAX_PREDICTORS = 8  # GP scales O(n^3) in rows; keep the input dimension modest too


def _resolve_outcome_predictors(ctx: Ctx):
    """Resolve (outcome, predictors, problem) by the regression-family convention.

    outcome = config['outcome'] if present, else the first continuous column.
    predictors = config['predictors'] (kept if present in df, != outcome) else the
    remaining numeric (continuous/count/binary) columns, capped at _MAX_PREDICTORS.
    The panel unit/time columns are excluded from auto-selection. ``problem`` is a
    Chinese skip message (caller appends + returns) when no usable roles are found.
    """
    import pandas as pd

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]

    forced_y = cfg.get("outcome")
    if forced_y in df.columns:
        outcome = forced_y
    elif cont:
        outcome = resolve_outcome(fp, cfg, cont)
    else:
        return None, [], "高斯过程回归 跳过：未找到连续结果变量（需 ≥1 个连续列）。"

    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome]
    if forced:
        preds = forced
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, fp.unit_col, fp.time_col}
        ]
    # keep only columns with at least one numeric value
    preds = [c for c in preds if pd.to_numeric(df[c], errors="coerce").notna().any()]
    preds = preds[:_MAX_PREDICTORS]
    if not preds:
        return None, [], "高斯过程回归 跳过：未找到可用的数值预测变量（需 ≥1 个）。"
    return outcome, preds, None


def _clean(df, outcome, preds):
    """Numeric-coerce X (predictors) and y (outcome), drop rows with any NaN.

    Returns (X, y, preds, problem). ``problem`` set (and X/y None) when fewer than
    20 complete rows remain — a GP on a tiny sample over-fits and the held-out split
    becomes meaningless.
    """
    import pandas as pd

    X = df[preds].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    y = pd.to_numeric(df[outcome], errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if len(y) < 20:
        return None, None, None, f"高斯过程回归 跳过：有效样本不足（去缺失后 n={len(y)}<20）。"
    return X.to_numpy(float), y.to_numpy(float), list(preds), None


@register("gaussian_process_regression")
def _branch_gaussian_process_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np

    outcome, preds, problem = _resolve_outcome_predictors(ctx)
    if problem:
        summary.append(problem)
        return
    X, y, preds, problem = _clean(df, outcome, preds)
    if problem:
        summary.append(problem)
        return

    try:
        try:
            random_state = int(cfg.get("random_state", 42))
        except (TypeError, ValueError):
            random_state = 42

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.model_selection import train_test_split

        n = int(len(y))
        n_predictors = int(X.shape[1])

        # honest out-of-sample split (75/25, seeded). Guard a workable test size.
        test_frac = 0.25
        n_test = int(round(n * test_frac))
        if n_test < 3:
            n_test = 3
        if n - n_test < 5:  # too few to train; fall back to a smaller test set
            n_test = max(3, n // 5)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=n_test, random_state=random_state
        )

        # ---- standardize X and y on the TRAIN split (no test leakage) ----------
        x_mu = X_tr.mean(axis=0)
        x_sd = X_tr.std(axis=0, ddof=0)
        x_sd = np.where(x_sd < 1e-12, 1.0, x_sd)
        y_mu = float(y_tr.mean())
        y_sd = float(y_tr.std(ddof=0))
        if y_sd < 1e-12:
            summary.append("高斯过程回归 跳过：结果变量在训练集上无变异（常数列）。")
            return

        Xz_tr = (X_tr - x_mu) / x_sd
        Xz_te = (X_te - x_mu) / x_sd
        yz_tr = (y_tr - y_mu) / y_sd

        # ---- kernel: signal scale × RBF (length-scale) + white noise -----------
        # ARD: one length-scale per predictor (length_scale is a vector).
        ls0 = np.ones(n_predictors)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=ls0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,  # we standardize y ourselves to back-transform cleanly
            n_restarts_optimizer=5,
            random_state=random_state,
        )
        gp.fit(Xz_tr, yz_tr)

        # ---- posterior predictive on the held-out test set ---------------------
        yz_mean, yz_std = gp.predict(Xz_te, return_std=True)
        # back-transform predictions + predictive SD to the raw y scale
        pred_mean = yz_mean * y_sd + y_mu
        pred_std = yz_std * y_sd
        lower = pred_mean - 1.96 * pred_std
        upper = pred_mean + 1.96 * pred_std

        # held-out performance
        ss_res = float(np.sum((y_te - pred_mean) ** 2))
        ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
        r2_heldout = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        rmse_heldout = float(np.sqrt(np.mean((y_te - pred_mean) ** 2)))
        coverage_95 = float(np.mean((y_te >= lower) & (y_te <= upper)))

        log_ml = float(gp.log_marginal_likelihood_value_)

        # ---- recover learned hyper-parameters from the fitted kernel -----------
        # Kernel structure is (Constant * RBF) + WhiteKernel. Length-scale lives on
        # the RBF; noise_level on the WhiteKernel. Names are version-stable in
        # sklearn's kernel-parameter dict; we also fall back to attribute walking.
        params = gp.kernel_.get_params()
        noise_level = float("nan")
        lengthscale = float("nan")
        ls_vec = None
        for key, val in params.items():
            if key.endswith("noise_level") and np.isscalar(val):
                noise_level = float(val)
            if key.endswith("length_scale") and not key.endswith("length_scale_bounds"):
                arr = np.atleast_1d(np.asarray(val, dtype=float))
                ls_vec = arr
                lengthscale = float(np.mean(arr))  # mean length-scale if ARD
        # WhiteKernel noise_level is on the STANDARDIZED y scale; report on raw scale
        noise_level_raw = noise_level * (y_sd ** 2) if noise_level == noise_level else float("nan")

        estimates.update({
            "r2_heldout": round(r2_heldout, 6) if r2_heldout == r2_heldout else float("nan"),
            "rmse_heldout": round(rmse_heldout, 6),
            "coverage_95": round(coverage_95, 6),
            "log_marginal_likelihood": round(log_ml, 6),
            "noise_level": round(noise_level_raw, 8) if noise_level_raw == noise_level_raw else float("nan"),
            "lengthscale": round(lengthscale, 6) if lengthscale == lengthscale else float("nan"),
            "n": float(n),
            "n_predictors": float(n_predictors),
        })

        # ---- products: held-out predictions CSV --------------------------------
        import pandas as pd

        pred_df = pd.DataFrame({
            "actual": np.round(y_te, 8),
            "pred_mean": np.round(pred_mean, 8),
            "lower_95": np.round(lower, 8),
            "upper_95": np.round(upper, 8),
        })
        pred_df.to_csv(d / "gpr_predictions.csv", index=False, encoding="utf-8")
        files.append("gpr_predictions.csv")

        # ---- PNG: 1 predictor → GP mean + 95% band over data; else pred-vs-actual
        def _plot(plt):
            if n_predictors == 1:
                fig, ax = plt.subplots(figsize=(8, 4.6))
                # dense grid across the observed predictor range for a smooth curve
                xmin = float(np.min(X[:, 0]))
                xmax = float(np.max(X[:, 0]))
                grid = np.linspace(xmin, xmax, 300).reshape(-1, 1)
                gz = (grid - x_mu) / x_sd
                gz_mean, gz_std = gp.predict(gz, return_std=True)
                g_mean = gz_mean * y_sd + y_mu
                g_std = gz_std * y_sd
                ax.scatter(X_tr[:, 0], y_tr, s=18, color="#4C72B0", alpha=0.6,
                           label="train data")
                ax.scatter(X_te[:, 0], y_te, s=22, color="#C44E52", alpha=0.8,
                           marker="^", label="test data")
                ax.plot(grid[:, 0], g_mean, color="#55A868", lw=1.8, label="GP mean")
                ax.fill_between(grid[:, 0], g_mean - 1.96 * g_std, g_mean + 1.96 * g_std,
                                color="#55A868", alpha=0.18, label="95% predictive band")
                ax.set_xlabel(str(preds[0]))
                ax.set_ylabel(str(outcome))
                ax.set_title("Gaussian-process regression: posterior mean + 95% band")
                ax.legend(fontsize=8)
            else:
                fig, ax = plt.subplots(figsize=(6.2, 6.0))
                ax.errorbar(y_te, pred_mean, yerr=1.96 * pred_std, fmt="o",
                            color="#4C72B0", ecolor="#cccccc", ms=5, alpha=0.8,
                            capsize=2, label="test (mean ± 1.96σ)")
                lo = float(min(np.min(y_te), np.min(pred_mean)))
                hi = float(max(np.max(y_te), np.max(pred_mean)))
                ax.plot([lo, hi], [lo, hi], color="#C44E52", ls="--", lw=1.2,
                        label="y = x")
                ax.set_xlabel("actual (held-out)")
                ax.set_ylabel("GP predicted mean")
                ax.set_title(f"GP predicted vs actual (held-out R2={r2_heldout:.3f})")
                ax.legend(fontsize=8)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _plot(plt)
            plt.tight_layout()
            plt.savefig(d / "gpr_fit.png", dpi=140)
            plt.close("all")
            files.append("gpr_fit.png")
        except Exception:
            pass

        # ---- Chinese summary with ⚠ disclosures --------------------------------
        r2_txt = f"{r2_heldout:.4f}" if r2_heldout == r2_heldout else "不可估"
        ls_txt = (f"{lengthscale:.4g}" if lengthscale == lengthscale else "不可估")
        ard_note = ""
        if ls_vec is not None and n_predictors > 1:
            ard_note = ("（ARD：每个预测变量各一长度尺度，报告为均值；越小说明该维上函数越「敏感/多变」，"
                        "越大越「平滑/无关」）")
        summary.append(
            f"{entry.method} 完成：结果={outcome}、预测变量 {n_predictors} 个"
            f"（{', '.join(map(str, preds))}），n={n}（75/25 留出，random_state={random_state}）。"
            f"留出集 R²={r2_txt}、RMSE={rmse_heldout:.4g}；95% 预测区间覆盖率={coverage_95:.3f}"
            "（校准良好应≈0.95——低于则低估不确定性、高于则过保守）。"
            f"学得 RBF 长度尺度={ls_txt}{ard_note}、观测噪声方差（原始尺度）="
            f"{(estimates['noise_level'] if estimates['noise_level']==estimates['noise_level'] else float('nan'))}、"
            f"对数边际似然={log_ml:.4g}。留出预测见 gpr_predictions.csv、图见 gpr_fit.png。"
            " ⚠ 高斯过程给出的是「函数的完整后验预测分布」（贝叶斯方法）：每个输入处都有均值与方差，"
            "区间是对预测的直接概率陈述，而非频率派置信区间。"
            " ⚠ 此处用 RBF 核 ⇒ 隐含「平滑函数先验」，长度尺度即「摆动程度」——若真函数有突变/不连续，"
            "RBF 会过度平滑；可换 Matérn/周期核等以匹配结构。"
            " ⚠ 超参数（信号尺度/长度尺度/噪声）由「经验贝叶斯」即最大化边际似然点估计得到，"
            "并非对超参数做完整后验采样——其不确定性未传播进预测区间。"
            " ⚠ GP 训练复杂度约 O(n³)，大样本会很慢/吃内存；输入维度已上限 "
            f"{_MAX_PREDICTORS} 个预测变量。"
            " ⚠ 外推到训练数据范围之外时，预测会回归到先验均值且区间迅速变宽——"
            "远离数据处不要轻信均值。"
        )

        code += [
            "import numpy as np",
            "from sklearn.gaussian_process import GaussianProcessRegressor",
            "from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel",
            "from sklearn.model_selection import train_test_split",
            "# 标准化 X、y（仅用训练集统计量，避免测试泄漏）",
            "# 核 = 信号尺度·RBF(长度尺度, ARD) + 白噪声; 经验贝叶斯最大化边际似然",
            "kernel = ConstantKernel()*RBF(length_scale=np.ones(p)) + WhiteKernel()",
            "gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5).fit(Xz_tr, yz_tr)",
            "mean, std = gp.predict(Xz_te, return_std=True)   # 后验预测均值+标准差",
            "coverage = np.mean((y_te >= mean-1.96*std) & (y_te <= mean+1.96*std))  # ≈0.95 即校准",
        ]
    except Exception as exc:
        summary.append(f"高斯过程回归 计算失败：{exc}")
