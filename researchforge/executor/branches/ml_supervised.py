"""Branch handlers for the ml_supervised family — supervised machine learning
with HONEST cross-validated performance (pure Python, scikit-learn; no R).

Three predictive (non-causal) methods, each resolving outcome + predictors with
the regression-family convention (outcome = first continuous column; predictors =
the rest of the numeric columns), config-overridable, mirroring ml.py:

  * regularized_regression — penalized linear regression (lasso / ridge / elasticnet),
    penalty chosen by cross-validation; reports CV R²/RMSE and the standardized
    coefficients shrunk toward zero (sparsity).
  * svm_model              — support vector machine; SVC if the outcome is
    binary/categorical, SVR if continuous; reports k-fold CV performance and the
    number of support vectors.
  * gradient_boosting      — sklearn GradientBoosting (Classifier / Regressor by
    outcome kind); reports k-fold CV performance and PERMUTATION importance.

Engine conventions (see CLAUDE.md, executor/_branch_api.py):
  * each handler is ``@register("<id>") def _branch_<id>(ctx)`` — unpack ctx, then
    MUTATE summary/estimates/files/code (never rebind);
  * honest degrade — too few rows / <1 predictor / non-numeric predictors that can't
    be used / a class smaller than the CV fold count / sklearn missing → append a
    Chinese "<方法>跳过：<原因>" message and RETURN (never crash, never fabricate);
  * products in try/except — CSV + PNG (matplotlib Agg, ENGLISH plot labels), float
    `estimates`, Chinese `summary` ending with ⚠ disclosures;
  * report CROSS-VALIDATED (out-of-sample) metrics as the headline, NEVER in-sample;
  * fixed random_state (disclosed) for reproducibility.

scikit-learn / numpy / pandas are installed; this module imports nothing from R.
"""

from __future__ import annotations

import importlib.util

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor._helpers.diagnostics import suspicious_fit_warnings
from researchforge.executor.branches.ml import _resolve_ml_outcome

_SEED = 0  # fixed random_state, disclosed in every summary


# ─────────────────────────────────────────────────────────────────────────────
# Shared outcome / predictor resolution (regression-family convention, like ml.py).
#
# Returns (outcome, is_clf, predictors, problem_msg). When problem_msg is not None
# the caller should append it (prefixed with the method name) and RETURN — honest
# degrade. is_clf is True when the chosen outcome is binary/categorical (classify),
# False when continuous (regress).
#
# Preference order mirrors ml.py's random_forest/xgboost (via _resolve_ml_outcome,
# reused directly — see Wave L comment below): explicit config["outcome"] wins; else
# a HIGH-confidence detected outcome ACROSS kinds (an event-named binary like churn is
# the target even next to a continuous column); else a CONTINUOUS column (a lone binary
# is usually a flag *feature*, not the target); else a binary/categorical column becomes
# a classification target.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_xy(ctx: Ctx, method: str, min_rows: int):
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df

    if importlib.util.find_spec("sklearn") is None:
        return None, False, [], f"{method}跳过：未检测到 scikit-learn（安装：pip install scikit-learn）。"

    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    cat = [
        c.name for c in fp.columns
        if c.kind in {"binary", "categorical"} and c.name not in {fp.unit_col, fp.time_col}
    ]

    # config outcome override decides the task by that column's kind (Wave K F1 —
    # untouched by this Wave L change).
    forced_y = cfg.get("outcome")
    if forced_y in df.columns:
        outcome = forced_y
        is_clf = forced_y not in cont
    else:
        # Wave L — delegate the no-config fallback to ml.py's _resolve_ml_outcome so
        # random_forest/xgboost and gradient_boosting/svm/regularized_regression share
        # ONE outcome-tier ladder instead of two copies that can drift apart. Its tier 2
        # (checked before cont-first) binds a HIGH-confidence detected outcome ACROSS
        # kinds — an event-named binary (churn/died, via roles._BIN_OUTCOME_RE) IS the
        # prediction target even when a continuous column is also present, closing the
        # same gap dogfooding ③ found in ml.py (no-config path silently regressing a
        # continuous feature like tenure instead of classifying churn). A design-factor
        # column (arm/exposed) is treatment-named, so roles routes it to likely_treatment
        # and it never reaches likely_outcome here — can't be mis-bound. Tiers 3/4 below
        # (cont-first, then binary/categorical) reproduce this function's prior behavior
        # exactly, so this is a pure additive fix, not a behavior change to the fallback.
        outcome, is_clf = _resolve_ml_outcome(fp, cfg, cont, cat)
        if outcome is None:
            return None, False, [], f"{method}跳过：未找到结果变量（需要 1 个连续列做回归，或二值/分类列做分类）。"

    # predictors: config override (must exist, drop the outcome), else all numeric
    # (continuous/count/binary) columns except the outcome / unit / time.
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome]
    if forced:
        preds = forced[:30]
    else:
        preds = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary"}
            and c.name not in {outcome, fp.unit_col, fp.time_col}
        ][:30]
    if not preds:
        return None, is_clf, [], f"{method}跳过：未找到可用的预测变量（≥1 个数值列）。"

    # numeric-coerce predictors; drop any that are entirely non-numeric (can't encode)
    import pandas as pd

    usable = [c for c in preds if pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not usable:
        return None, is_clf, [], f"{method}跳过：预测变量无法转为数值（需要数值型预测变量）。"

    return outcome, is_clf, usable, None


def _clean_xy(df, outcome, preds, is_clf):
    """Numeric-coerce, drop rows with any NaN, return (X, y). For classification y
    is kept as-is (labels). Raises ValueError for the caller to degrade on."""
    import numpy as np
    import pandas as pd

    X = df[preds].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    if is_clf:
        y = df[outcome]
    else:
        y = pd.to_numeric(df[outcome], errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if y.nunique() < 2:
        raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")
    return X.to_numpy(dtype=float), (y.to_numpy() if is_clf else y.to_numpy(dtype=float)), np.asarray(preds)


def _cv_folds(n_samples, is_clf, y, default=5):
    """Pick a safe k for k-fold CV: <= default, and (classification) <= the smallest
    class size so every fold can hold each class. Returns k (>=2) or None when even
    2 folds are impossible (smallest class < 2 / n < 4)."""
    import numpy as np

    if is_clf:
        _, counts = np.unique(y, return_counts=True)
        min_class = int(counts.min())
        if min_class < 2:
            return None
        return max(2, min(default, min_class))
    if n_samples < 4:
        return None
    return max(2, min(default, n_samples // 2))


# ─────────────────────────────────────────────────────────────────────────────
# (A) regularized_regression — penalized linear regression (continuous outcome).
# ─────────────────────────────────────────────────────────────────────────────
@register("regularized_regression")
def _branch_regularized_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    method = "正则化回归"
    outcome, is_clf, preds, prob = _resolve_xy(ctx, method, min_rows=20)
    if prob:
        summary.append(prob)
        return
    if is_clf:
        summary.append(f"{method}跳过：结果变量为分类型，正则化线性回归需要连续结果（分类请用 svm_model / gradient_boosting）。")
        return

    chosen = str(cfg.get("method", "elasticnet")).lower()
    if chosen not in {"lasso", "ridge", "elasticnet"}:
        chosen = "elasticnet"

    try:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y, names = _clean_xy(df, outcome, preds, is_clf=False)
        n = int(X.shape[0])
        if n < 20:
            summary.append(f"{method}跳过：有效样本 {n}<20，正则化回归 + 交叉验证不可靠。")
            return

        k = _cv_folds(n, is_clf=False, y=y, default=5)
        if k is None:
            summary.append(f"{method}跳过：有效样本太少，无法做交叉验证。")
            return

        # build the CV-penalty estimator (alpha picked by inner CV inside *CV)
        inner = KFold(n_splits=k, shuffle=True, random_state=_SEED)
        alphas = cfg.get("alphas")
        alphas = list(alphas) if alphas else None  # None -> sklearn's own alpha path
        if chosen == "lasso":
            est = LassoCV(alphas=alphas, cv=inner, random_state=_SEED, max_iter=20000)
        elif chosen == "ridge":
            # RidgeCV needs an explicit alpha grid
            ridge_alphas = alphas if alphas else np.logspace(-3, 3, 25)
            est = RidgeCV(alphas=ridge_alphas, cv=inner)
        else:  # elasticnet
            try:
                l1r = float(cfg.get("l1_ratio", 0.5))
            except (TypeError, ValueError):
                l1r = 0.5
            l1r = min(1.0, max(0.01, l1r))
            est = ElasticNetCV(l1_ratio=l1r, alphas=alphas, cv=inner, random_state=_SEED, max_iter=20000)

        pipe = Pipeline([("scale", StandardScaler()), ("model", est)])

        # HONEST out-of-sample R²/RMSE via cross_val_predict on the WHOLE pipeline
        # (scaler + CV-penalty estimator refit inside each outer fold).
        outer = KFold(n_splits=k, shuffle=True, random_state=_SEED)
        y_oof = cross_val_predict(pipe, X, y, cv=outer)
        cv_r2 = float(r2_score(y, y_oof))
        cv_rmse = float(np.sqrt(mean_squared_error(y, y_oof)))

        # fit once on all data to report the chosen alpha + standardized coefficients
        pipe.fit(X, y)
        fitted = pipe.named_steps["model"]
        coefs = np.asarray(fitted.coef_, dtype=float).ravel()
        alpha = float(getattr(fitted, "alpha_", np.nan))
        l1_ratio = float(getattr(fitted, "l1_ratio_", (1.0 if chosen == "lasso" else (0.0 if chosen == "ridge" else np.nan))))
        n_selected = int(np.sum(np.abs(coefs) > 1e-10))

        order = np.argsort(-np.abs(coefs))
        coef_df = pd.DataFrame({
            "predictor": names[order],
            "std_coef": np.round(coefs[order], 6),
            "abs_coef": np.round(np.abs(coefs[order]), 6),
            "selected": (np.abs(coefs[order]) > 1e-10),
        })

        estimates["cv_r2"] = round(cv_r2, 4)
        estimates["cv_rmse"] = round(cv_rmse, 4)
        estimates["alpha"] = round(alpha, 6)
        estimates["l1_ratio"] = round(l1_ratio, 4)
        estimates["n_selected"] = float(n_selected)
        estimates["n_predictors"] = float(len(names))
        estimates["n"] = float(n)

        try:
            coef_df.to_csv(d / "regularized_coefficients.csv", index=False, encoding="utf-8")
            files.append("regularized_coefficients.csv")
        except Exception:
            pass

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = coef_df.head(20)
            colors = ["#999999" if (not s) else "#2c7fb8" for s in top["selected"]]
            fig, ax = plt.subplots(figsize=(7, max(3, len(top) * 0.4)))
            ax.barh(top["predictor"][::-1], top["std_coef"][::-1], color=colors[::-1])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("standardized coefficient")
            ax.set_title(f"{chosen} coefficients (grey = shrunk to 0) — {outcome}")
            fig.tight_layout()
            fig.savefig(d / "regularized_coefficients.png", dpi=150)
            plt.close(fig)
            files.append("regularized_coefficients.png")
        except Exception:
            pass

        l1_txt = f"，l1_ratio={l1_ratio:.2f}" if chosen == "elasticnet" else ""
        summary.append(
            f"{entry.method} 完成（{chosen}，{k}-折交叉验证，seed={_SEED}）：{outcome} ~ {len(names)} 个预测变量；"
            f"交叉验证 R²={cv_r2:.3f}、RMSE={cv_rmse:.3f}（样本外，诚实泛化估计）；"
            f"选中惩罚 α={alpha:.4g}{l1_txt}；{n_selected}/{len(names)} 个系数非零（其余被收缩至 0）；n={n}。"
            "⚠ 系数基于**标准化**特征（量纲可比，非原始单位）；lasso 在预测变量相关时变量选择不稳定；"
            "以交叉验证 R² 评估泛化（样本内偏乐观）；可用 config 设 outcome/predictors/method/alphas。"
        )
        code += [
            "from sklearn.linear_model import ElasticNetCV  # 或 LassoCV / RidgeCV",
            "from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler",
            "from sklearn.model_selection import KFold, cross_val_predict",
            f"X = df[{list(names)!r}].apply(pd.to_numeric, errors='coerce'); y = df['{outcome}']",
            f"pipe = Pipeline([('scale', StandardScaler()), ('model', ElasticNetCV(cv=KFold({k}, shuffle=True, random_state={_SEED})))])",
            f"yhat = cross_val_predict(pipe, X, y, cv=KFold({k}, shuffle=True, random_state={_SEED}))  # 样本外预测",
            "pipe.fit(X, y); print(pipe.named_steps['model'].alpha_, pipe.named_steps['model'].coef_)",
        ]
    except Exception as err:  # pragma: no cover - safety net, never crash the run
        summary.append(f"{method}失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) svm_model — support vector machine (SVC if categorical/binary, else SVR).
# ─────────────────────────────────────────────────────────────────────────────
@register("svm_model")
def _branch_svm_model(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    method = "支持向量机"
    outcome, is_clf, preds, prob = _resolve_xy(ctx, method, min_rows=20)
    if prob:
        summary.append(prob)
        return

    kernel = str(cfg.get("kernel", "rbf")).lower()
    if kernel not in {"rbf", "linear", "poly", "sigmoid"}:
        kernel = "rbf"
    try:
        C = float(cfg.get("C", 1.0))
    except (TypeError, ValueError):
        C = 1.0
    gamma = cfg.get("gamma", "scale")
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        try:
            gamma = float(gamma)
        except (TypeError, ValueError):
            gamma = "scale"

    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            confusion_matrix, mean_squared_error,
        )
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, SVR

        X, y, names = _clean_xy(df, outcome, preds, is_clf)
        n = int(X.shape[0])
        if n < 20:
            summary.append(f"{method}跳过：有效样本 {n}<20，SVM + 交叉验证不可靠。")
            return
        k = _cv_folds(n, is_clf, y, default=5)
        if k is None:
            summary.append(f"{method}跳过：{'某一类样本数<2，分类' if is_clf else '样本太少，'}无法做交叉验证。")
            return

        est = (SVC(kernel=kernel, C=C, gamma=gamma, random_state=_SEED)
               if is_clf else SVR(kernel=kernel, C=C, gamma=gamma))
        pipe = Pipeline([("scale", StandardScaler()), ("model", est)])

        # count support vectors from a single full-data fit (descriptive)
        pipe.fit(X, y)
        sv = getattr(pipe.named_steps["model"], "support_vectors_", None)
        n_sv = int(sv.shape[0]) if sv is not None else 0

        if is_clf:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=_SEED)
            acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
            y_oof = cross_val_predict(pipe, X, y, cv=cv)
            cv_acc = float(np.mean(acc_scores))
            cv_f1 = float(np.mean(f1_scores))
            labels = np.unique(y)
            cm = confusion_matrix(y, y_oof, labels=labels)
            counts = pd.Series(y).value_counts()
            imbalance = float(counts.min() / counts.max())

            estimates["cv_accuracy"] = round(cv_acc, 4)
            estimates["cv_f1_macro"] = round(cv_f1, 4)
            estimates["n_support_vectors"] = float(n_sv)
            estimates["n_classes"] = float(len(labels))
            estimates["n"] = float(n)

            try:
                fold_df = pd.DataFrame({
                    "fold": np.arange(1, k + 1),
                    "accuracy": np.round(acc_scores, 6),
                    "f1_macro": np.round(f1_scores, 6),
                })
                fold_df.to_csv(d / "svm_fold_scores.csv", index=False, encoding="utf-8")
                files.append("svm_fold_scores.csv")
                cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])
                cm_df.to_csv(d / "svm_confusion_matrix.csv", encoding="utf-8")
                files.append("svm_confusion_matrix.csv")
            except Exception:
                pass

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.8), max(3.5, len(labels) * 0.8)))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(labels))); ax.set_xticklabels([str(c) for c in labels], rotation=45, ha="right")
                ax.set_yticks(range(len(labels))); ax.set_yticklabels([str(c) for c in labels])
                ax.set_xlabel("predicted"); ax.set_ylabel("actual")
                ax.set_title(f"Confusion matrix (CV) — {outcome}")
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(d / "svm_confusion_matrix.png", dpi=150)
                plt.close(fig)
                files.append("svm_confusion_matrix.png")
            except Exception:
                pass

            bal_note = "（类别较均衡）" if imbalance >= 0.5 else f"（类别不均衡，最小/最大类比={imbalance:.2f}）"
            # Wave K-F3: 可疑地完美诊断（SVM 无特征重要度，仅准确率信号）
            _svm_base = float(pd.Series(y).value_counts(normalize=True).max())
            _svm_warn = suspicious_fit_warnings(cv_accuracy=cv_acc, baseline_rate=_svm_base)
            _oos = "交叉验证，见下警告" if _svm_warn else "样本外"
            summary.append(
                f"{entry.method} 完成（SVC，{kernel} 核，C={C:g}，{k}-折分层交叉验证，seed={_SEED}）："
                f"分类 {outcome}（{len(labels)} 类）{bal_note}；交叉验证 准确率={cv_acc:.3f}、macro-F1={cv_f1:.3f}（{_oos}）；"
                f"支持向量 {n_sv} 个（占 {n_sv / n:.0%}）；n={n}。"
                "⚠ SVM 需特征标准化（已做）；rbf 的 C/gamma 影响大（除非 config 指定，否则用默认，不再自动调参以保证速度）；"
                "报告的是交叉验证表现（非样本内）；类别不均衡会抬高准确率，故同时报 macro-F1；可用 config 设 outcome/predictors/kernel/C/gamma。"
            )
            for _w in _svm_warn:
                summary.append(_w)
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=_SEED)
            r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            y_oof = cross_val_predict(pipe, X, y, cv=cv)
            cv_r2 = float(np.mean(r2_scores))
            cv_rmse = float(np.sqrt(mean_squared_error(y, y_oof)))

            estimates["cv_r2"] = round(cv_r2, 4)
            estimates["cv_rmse"] = round(cv_rmse, 4)
            estimates["n_support_vectors"] = float(n_sv)
            estimates["n"] = float(n)

            try:
                fold_df = pd.DataFrame({"fold": np.arange(1, k + 1), "r2": np.round(r2_scores, 6)})
                fold_df.to_csv(d / "svm_fold_scores.csv", index=False, encoding="utf-8")
                files.append("svm_fold_scores.csv")
            except Exception:
                pass

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5.5, 5))
                ax.scatter(y, y_oof, s=18, alpha=0.6, color="#2c7fb8")
                lo, hi = float(min(y.min(), y_oof.min())), float(max(y.max(), y_oof.max()))
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set_xlabel("actual"); ax.set_ylabel("predicted (CV)")
                ax.set_title(f"Predicted vs actual (CV) — {outcome}")
                fig.tight_layout()
                fig.savefig(d / "svm_pred_vs_actual.png", dpi=150)
                plt.close(fig)
                files.append("svm_pred_vs_actual.png")
            except Exception:
                pass

            summary.append(
                f"{entry.method} 完成（SVR，{kernel} 核，C={C:g}，{k}-折交叉验证，seed={_SEED}）："
                f"回归 {outcome} ~ {len(names)} 个预测变量；交叉验证 R²={cv_r2:.3f}、RMSE={cv_rmse:.3f}（样本外）；"
                f"支持向量 {n_sv} 个（占 {n_sv / n:.0%}）；n={n}。"
                "⚠ SVM 需特征标准化（已做）；rbf 的 C/gamma 影响大（除非 config 指定，否则用默认，不再自动调参以保证速度）；"
                "报告的是交叉验证表现（非样本内）；可用 config 设 outcome/predictors/kernel/C/gamma。"
            )

        task = "SVC" if is_clf else "SVR"
        code += [
            f"from sklearn.svm import {task}",
            "from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler",
            "from sklearn.model_selection import cross_val_score, cross_val_predict",
            f"X = df[{list(names)!r}].apply(pd.to_numeric, errors='coerce'); y = df['{outcome}']",
            f"pipe = Pipeline([('scale', StandardScaler()), ('model', {task}(kernel='{kernel}', C={C}))])",
            f"print(cross_val_score(pipe, X, y, cv={k}).mean())  # 样本外交叉验证表现",
        ]
    except Exception as err:  # pragma: no cover - safety net
        summary.append(f"{method}失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) gradient_boosting — sklearn GradientBoosting + permutation importance.
# ─────────────────────────────────────────────────────────────────────────────
@register("gradient_boosting")
def _branch_gradient_boosting(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    method = "梯度提升树"
    outcome, is_clf, preds, prob = _resolve_xy(ctx, method, min_rows=20)
    if prob:
        summary.append(prob)
        return

    try:
        n_estimators = max(10, int(cfg.get("n_estimators", 100)))
    except (TypeError, ValueError):
        n_estimators = 100
    try:
        learning_rate = float(cfg.get("learning_rate", 0.1))
    except (TypeError, ValueError):
        learning_rate = 0.1
    try:
        max_depth = max(1, int(cfg.get("max_depth", 3)))
    except (TypeError, ValueError):
        max_depth = 3

    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import (
            KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
        )

        X, y, names = _clean_xy(df, outcome, preds, is_clf)
        n = int(X.shape[0])
        if n < 20:
            summary.append(f"{method}跳过：有效样本 {n}<20，梯度提升 + 交叉验证不可靠。")
            return
        k = _cv_folds(n, is_clf, y, default=5)
        if k is None:
            summary.append(f"{method}跳过：{'某一类样本数<2，分类' if is_clf else '样本太少，'}无法做交叉验证。")
            return

        common = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                      max_depth=max_depth, random_state=_SEED)
        model = (GradientBoostingClassifier(**common) if is_clf
                 else GradientBoostingRegressor(**common))

        # headline CV (out-of-sample) performance
        if is_clf:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=_SEED)
            cv_acc = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="accuracy")))
            cv_f1 = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="f1_macro")))
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=_SEED)
            cv_r2 = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="r2")))
            y_oof = cross_val_predict(model, X, y, cv=cv)
            cv_rmse = float(np.sqrt(mean_squared_error(y, y_oof)))

        # PERMUTATION importance on a HELD-OUT split (model-agnostic, less biased
        # than impurity importance which favours high-cardinality features).
        strat = y if (is_clf and int(pd.Series(y).value_counts().min()) >= 2) else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=_SEED, stratify=strat)
        model.fit(X_tr, y_tr)
        scoring = "accuracy" if is_clf else "r2"
        perm = permutation_importance(
            model, X_te, y_te, n_repeats=10, random_state=_SEED, scoring=scoring)
        imp_mean = np.asarray(perm.importances_mean, dtype=float)
        imp_sd = np.asarray(perm.importances_std, dtype=float)
        order = np.argsort(-imp_mean)
        imp_df = pd.DataFrame({
            "predictor": names[order],
            "perm_importance_mean": np.round(imp_mean[order], 6),
            "perm_importance_sd": np.round(imp_sd[order], 6),
        })
        top_importance = float(imp_mean[order][0]) if len(order) else 0.0

        if is_clf:
            estimates["cv_accuracy"] = round(cv_acc, 4)
            estimates["cv_f1_macro"] = round(cv_f1, 4)
        else:
            estimates["cv_r2"] = round(cv_r2, 4)
            estimates["cv_rmse"] = round(cv_rmse, 4)
        estimates["top_importance"] = round(top_importance, 6)
        estimates["n_predictors"] = float(len(names))
        estimates["n"] = float(n)

        try:
            imp_df.to_csv(d / "gbm_permutation_importance.csv", index=False, encoding="utf-8")
            files.append("gbm_permutation_importance.csv")
        except Exception:
            pass

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = imp_df.head(20)
            fig, ax = plt.subplots(figsize=(7, max(3, len(top) * 0.4)))
            ax.barh(top["predictor"][::-1], top["perm_importance_mean"][::-1],
                    xerr=top["perm_importance_sd"][::-1], color="#31a354")
            ax.set_xlabel("permutation importance (mean drop in score)")
            ax.set_title(f"Permutation importance — {outcome}")
            fig.tight_layout()
            fig.savefig(d / "gbm_permutation_importance.png", dpi=150)
            plt.close(fig)
            files.append("gbm_permutation_importance.png")
        except Exception:
            pass

        top_name = str(imp_df.iloc[0]["predictor"]) if len(imp_df) else "?"
        if is_clf:
            perf_txt = f"准确率={cv_acc:.3f}、macro-F1={cv_f1:.3f}"
            task_txt = f"分类 {outcome}"
        else:
            perf_txt = f"R²={cv_r2:.3f}、RMSE={cv_rmse:.3f}"
            task_txt = f"回归 {outcome} ~ {len(names)} 个预测变量"
        # Wave K-F3: 可疑地完美/泄漏事后诊断 — 命中则抹掉背书措辞（诚实泛化估计/样本外）、改一等 ⚠
        _base = float(pd.Series(y).value_counts(normalize=True).max()) if is_clf else None
        _warn = (suspicious_fit_warnings(cv_accuracy=cv_acc, baseline_rate=_base,
                                         importances=imp_mean, feature_names=list(names))
                 if is_clf else [])
        _perf_tag = "交叉验证" if _warn else "样本外，诚实泛化估计"
        _honest_note = "" if _warn else "交叉验证表现为诚实估计（样本内偏乐观）；"
        summary.append(
            f"{entry.method} 完成（GradientBoosting，{n_estimators} 树，lr={learning_rate:g}，depth={max_depth}，"
            f"{k}-折交叉验证，seed={_SEED}）：{task_txt}；交叉验证 {perf_txt}（{_perf_tag}）；"
            f"置换重要性最高的预测变量={top_name}（{top_importance:.4f}）；n={n}。"
            "⚠ 用**置换重要性**（非杂质重要性，后者偏向高基数特征）；" + _honest_note +
            "GBM 在小样本上易过拟合（已披露 n）；除非 config 指定否则用默认（n_estimators/learning_rate/max_depth）；可用 config 设 outcome/predictors。"
        )
        for _w in _warn:
            summary.append(_w)
        task = "GradientBoostingClassifier" if is_clf else "GradientBoostingRegressor"
        code += [
            f"from sklearn.ensemble import {task}",
            "from sklearn.inspection import permutation_importance",
            "from sklearn.model_selection import cross_val_score, train_test_split",
            f"X = df[{list(names)!r}].apply(pd.to_numeric, errors='coerce'); y = df['{outcome}']",
            f"model = {task}(n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, random_state={_SEED})",
            f"print(cross_val_score(model, X, y, cv={k}).mean())  # 样本外交叉验证表现",
            "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0); model.fit(Xtr, ytr)",
            "imp = permutation_importance(model, Xte, yte, n_repeats=10, random_state=0)  # 置换重要性",
        ]
    except Exception as err:  # pragma: no cover - safety net
        summary.append(f"{method}失败：{err}")
