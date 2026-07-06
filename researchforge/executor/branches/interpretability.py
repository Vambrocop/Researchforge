"""Branch handlers for the interpretability family — model-agnostic post-hoc
explainers for a fitted predictive model (sklearn / shap; no R). "Open the black
box": which features drive the prediction, in what shape, and how strongly.

  * partial_dependence        — Partial Dependence (PDP) + ICE curves for the most
                                influential features (sklearn.inspection). Shows the
                                marginal effect of a feature on the prediction.
  * shap_values               — SHAP (Shapley additive explanations) global feature
                                attribution via TreeExplainer: mean(|SHAP|) importance
                                + the sign of each feature's average effect.
  * accumulated_local_effects — ALE (Apley & Zhu 2020): the correlation-robust
                                alternative to PDP — accumulates LOCAL prediction
                                differences within feature quantile bins, so it is not
                                biased by correlated features the way PDP is.

All fit a default tree model (gradient boosting, or random forest via config model)
on a continuous outcome (regression) — or a binary outcome (classification) when no
continuous column exists — mirroring ml.py's role convention; config outcome/predictors
override. Each degrades honestly (no outcome/features / too few rows / <2 classes /
missing import -> append a Chinese "<method>跳过/失败：<reason>" and RETURN), writes
CSV + PNG (matplotlib Agg, ENGLISH labels), fills float `estimates`, appends a Chinese
`summary` with ⚠ disclosures, and MUTATES ctx. See _branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_FEAT_KINDS = {"continuous", "count", "binary"}
_TOP_N = 4  # how many top features to explain by default


def _build_model(ctx: Ctx, label: str, min_rows: int = 30):
    """Resolve roles, fit a tree model. Returns a dict on success, else (None, msg)."""
    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        return None, f"{label}跳过：需要 scikit-learn 包（未检测到）。"
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    binary = [c.name for c in fp.columns if c.kind == "binary"]

    # outcome: config wins; else the shared resolver within each tier (high-confidence
    # detected outcome > first non-treatment-named > first) — continuous before binary.
    from researchforge.executor.run import resolve_outcome

    outcome = cfg.get("outcome") if cfg.get("outcome") in df.columns else None
    if outcome is not None:
        is_clf = fp.column(outcome) is not None and fp.column(outcome).kind in {"binary", "categorical"}
    elif cont:
        outcome, is_clf = resolve_outcome(fp, cfg, cont), False
    elif binary:
        outcome, is_clf = resolve_outcome(fp, cfg, binary), True
    else:
        return None, f"{label}跳过：未找到结果变量（需连续或二值列），config outcome 指定。"

    excl = {outcome, fp.unit_col, fp.time_col}
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != outcome]
    features = forced or [c.name for c in fp.columns if c.kind in _FEAT_KINDS and c.name not in excl]
    if not features:
        return None, f"{label}跳过：未找到可用特征列。"

    import numpy as np
    import pandas as pd

    mask = df[features].notna().all(axis=1) & df[outcome].notna()
    X = df.loc[mask, features].astype(float)
    y = df.loc[mask, outcome]
    if len(X) < min_rows:
        return None, f"{label}跳过：有效行 {len(X)} < {min_rows}。"
    if y.nunique() < 2:
        return None, f"{label}跳过：结果变量 {outcome} 取值不足两类。"
    # multiclass classification would make the positive-class slice (predict_proba[:,1])
    # arbitrary — explain only binary classification (or a continuous outcome).
    if is_clf and y.nunique() > 2:
        return None, (f"{label}跳过：分类结果暂仅支持二值（{outcome} 有 {int(y.nunique())} 类）——"
                      "请二值化，或改用连续结果做回归解释。")

    model_name = str(cfg.get("model", "gbm")).lower()
    try:
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
            RandomForestClassifier, RandomForestRegressor,
        )
    except Exception as e:  # pragma: no cover
        return None, f"{label}跳过：sklearn 导入失败：{e}"

    if model_name == "rf":
        model = (RandomForestClassifier(n_estimators=200, random_state=0) if is_clf
                 else RandomForestRegressor(n_estimators=200, random_state=0))
    else:
        model = (GradientBoostingClassifier(random_state=0) if is_clf
                 else GradientBoostingRegressor(random_state=0))
        model_name = "gbm"
    model.fit(X, y)
    return {
        "model": model, "X": X, "y": y, "features": features, "outcome": outcome,
        "is_clf": is_clf, "model_name": model_name, "np": np, "pd": pd,
    }, None


def _top_features(info, k=_TOP_N):
    """Top-k features by the model's own importance (fallback: first k)."""
    m = info["model"]
    feats = info["features"]
    imp = getattr(m, "feature_importances_", None)
    if imp is None:
        return feats[:k]
    order = sorted(range(len(feats)), key=lambda i: -imp[i])
    return [feats[i] for i in order[:k]]


def _predict_fn(info):
    """A scalar prediction function: P(class 1) for classifiers, else the regressor."""
    m, is_clf = info["model"], info["is_clf"]
    if is_clf:
        return lambda Xarr: m.predict_proba(Xarr)[:, 1]
    return lambda Xarr: m.predict(Xarr)


# ---------------------------------------------------------------------------
# 1. partial_dependence — PDP + ICE
# ---------------------------------------------------------------------------
@register("partial_dependence")
def _branch_partial_dependence(ctx: Ctx) -> None:
    d, entry = ctx.d, ctx.entry
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    info, err = _build_model(ctx, "部分依赖分析")
    if err:
        summary.append(err)
        return
    try:
        from sklearn.inspection import partial_dependence

        np, pd = info["np"], info["pd"]
        X, model = info["X"], info["model"]
        cfg = ctx.cfg
        chosen = [c for c in (cfg.get("features") or []) if c in info["features"]] or _top_features(info)

        rows = []
        pd_ranges = {}
        for feat in chosen:
            pdr = partial_dependence(model, X, [X.columns.get_loc(feat)], kind="average",
                                     grid_resolution=int(cfg.get("grid_resolution", 30)))
            grid = np.asarray(pdr["grid_values"][0], dtype=float)
            avg = np.asarray(pdr["average"][0], dtype=float)
            rng = float(avg.max() - avg.min())
            pd_ranges[feat] = rng
            for g, a in zip(grid, avg):
                rows.append({"feature": feat, "value": float(g), "partial_dependence": float(a)})
        tab = pd.DataFrame(rows)
        tab.to_csv(d / "partial_dependence.csv", index=False, encoding="utf-8")
        files.append("partial_dependence.csv")

        ranked = sorted(pd_ranges.items(), key=lambda kv: -kv[1])
        for feat, rng in ranked:
            estimates[f"pd_range_{feat}"] = round(float(rng), 6)
        estimates["n"] = float(len(X))
        estimates["n_features_explained"] = float(len(chosen))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            nplt = len(chosen)
            ncol = min(2, nplt)
            nrow = (nplt + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow), squeeze=False)
            for i, feat in enumerate(chosen):
                ax = axes[i // ncol][i % ncol]
                sub = tab[tab["feature"] == feat]
                ax.plot(sub["value"], sub["partial_dependence"], color="#4C72B0")
                ax.set_xlabel(feat)
                ax.set_ylabel("partial dependence")
                ax.set_title(feat)
            for j in range(nplt, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(f"Partial dependence — {info['outcome']}")
            fig.tight_layout()
            fig.savefig(d / "partial_dependence.png", dpi=150)
            plt.close(fig)
            files.append("partial_dependence.png")
        except Exception:
            pass

        code += [
            "from sklearn.inspection import partial_dependence",
            "# PDP: marginal effect of each feature on the model prediction",
        ]
        top = ranked[0][0] if ranked else "?"
        summary.append(
            f"{entry.method}（{info['model_name'].upper()} 基模型，{'分类' if info['is_clf'] else '回归'}）："
            f"结果={info['outcome']}，解释了 {len(chosen)} 个特征的部分依赖。"
            f"边际效应最强（PD 跨度最大）的特征：{top}（跨度={pd_ranges.get(top, float('nan')):.4f}）。"
            " ⚠ PDP 假定被绘特征与其余特征**独立**——特征相关时会外推到不真实的组合而产生偏差"
            "（此时优先看 accumulated_local_effects / ALE）；PDP 展示关联性边际效应，非因果效应。"
        )
    except Exception as e:
        summary.append(f"部分依赖分析失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 2. shap_values — SHAP global attribution (TreeExplainer)
# ---------------------------------------------------------------------------
@register("shap_values")
def _branch_shap_values(ctx: Ctx) -> None:
    d, entry = ctx.d, ctx.entry
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import importlib.util
    if importlib.util.find_spec("shap") is None:
        summary.append("SHAP 分析跳过：需要 shap 包（未检测到）。pip install shap。"
                       "可用替代：random_forest / gradient_boosting 的特征重要性、partial_dependence。")
        return
    info, err = _build_model(ctx, "SHAP 分析")
    if err:
        summary.append(err)
        return
    try:
        import shap

        np, pd = info["np"], info["pd"]
        X, model, feats = info["X"], info["model"], info["features"]

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # classifiers may return a list per class (or a 3-D array) — take the positive class
        if isinstance(sv, list):
            sv = sv[1] if (info["is_clf"] and len(sv) > 1) else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, p, classes)
            sv = sv[:, :, 1] if (info["is_clf"] and sv.shape[2] > 1) else sv[:, :, 0]

        mean_abs = np.abs(sv).mean(axis=0)
        # direction: sign of corr(shap_j, x_j); +1 means higher feature -> higher prediction
        direction = []
        for j in range(len(feats)):
            xj = X.iloc[:, j].to_numpy(float)
            s = sv[:, j]
            if np.std(xj) < 1e-12 or np.std(s) < 1e-12:
                direction.append(0.0)
            else:
                direction.append(float(np.sign(np.corrcoef(xj, s)[0, 1])))

        imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs,
                            "direction": direction}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(d / "shap_importance.csv", index=False, encoding="utf-8")
        files.append("shap_importance.csv")

        for _, r in imp.iterrows():
            estimates[f"mean_abs_shap_{r['feature']}"] = round(float(r["mean_abs_shap"]), 6)
        # base value must be the SAME class the SHAP values were sliced to (class 1
        # for binary classification), else base + Σshap = prediction breaks.
        ev = np.ravel(explainer.expected_value)
        cls = 1 if (info["is_clf"] and ev.size > 1) else 0
        estimates["base_value"] = round(float(ev[cls]), 6)
        estimates["n"] = float(len(X))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            topk = imp.head(15)[::-1]
            fig, ax = plt.subplots(figsize=(6, max(3, len(topk) * 0.4)))
            colors = ["#C44E52" if ddirection > 0 else "#4C72B0" if ddirection < 0 else "grey"
                      for ddirection in topk["direction"]]
            ax.barh(topk["feature"], topk["mean_abs_shap"], color=colors)
            ax.set_xlabel("mean(|SHAP value|)")
            ax.set_title(f"SHAP feature importance — {info['outcome']}\n(red=+ / blue=- direction)")
            fig.tight_layout()
            fig.savefig(d / "shap_importance.png", dpi=150)
            plt.close(fig)
            files.append("shap_importance.png")
        except Exception:
            pass

        code += [
            "import shap  # SHAP TreeExplainer global attribution",
            "explainer = shap.TreeExplainer(model); sv = explainer.shap_values(X)",
            "import numpy as np; mean_abs = np.abs(sv).mean(axis=0)  # global importance",
        ]
        top = imp.iloc[0]
        dir_txt = {1.0: "正向", -1.0: "负向", 0.0: "方向不定"}.get(top["direction"], "")
        summary.append(
            f"{entry.method}（{info['model_name'].upper()} + TreeExplainer，{'分类' if info['is_clf'] else '回归'}）："
            f"结果={info['outcome']}，按 mean(|SHAP|) 最重要的特征：{top['feature']}"
            f"（{top['mean_abs_shap']:.4f}，{dir_txt}）。SHAP 值可加性地分解每个预测=基准值+各特征贡献。"
            " ⚠ SHAP 解释的是**模型**学到的关联（含其偏差/混杂），非数据中的因果效应；"
            "TreeExplainer 用条件期望，强相关特征间的归因可在彼此间转移。"
        )
    except Exception as e:
        summary.append(f"SHAP 分析失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 3. accumulated_local_effects — ALE (Apley & Zhu 2020), 1-D
# ---------------------------------------------------------------------------
@register("accumulated_local_effects")
def _branch_accumulated_local_effects(ctx: Ctx) -> None:
    d, entry, cfg = ctx.d, ctx.entry, ctx.cfg
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    info, err = _build_model(ctx, "累积局部效应分析")
    if err:
        summary.append(err)
        return
    try:
        np, pd = info["np"], info["pd"]
        X, feats = info["X"], info["features"]
        predict = _predict_fn(info)

        # choose continuous-ish features (>= ~10 distinct values) to ALE
        chosen = [c for c in (cfg.get("features") or []) if c in feats]
        if not chosen:
            cont_feats = [c for c in _top_features(info, k=len(feats)) if X[c].nunique() >= 10]
            chosen = cont_feats[:_TOP_N] or _top_features(info)
        n_bins = max(4, int(cfg.get("n_bins", 20)))

        all_rows = []
        ale_ranges = {}
        for feat in chosen:
            xj = X[feat].to_numpy(float)
            # quantile bin edges (unique); need >=2 intervals
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(xj, qs))
            if edges.size < 3:
                continue
            K = edges.size - 1
            jloc = X.columns.get_loc(feat)
            # assign each point to an interval (1..K); points at the min go to bin 1
            idx = np.clip(np.searchsorted(edges, xj, side="left"), 1, K)
            # Empty bins contribute NO local effect (standard ALEPlot behavior): we
            # skip them entirely so they inject no spurious 0-slope segment. We keep
            # only non-empty bins; accumulation runs over that reduced sequence, and
            # the x-axis nodes are the edges bordering the non-empty bins.
            local_list = []   # mean local diff per kept (non-empty) bin
            count_list = []   # its sample count (for count-weighted centering)
            node_edges = []   # edge values of the kept ALE curve (length = #kept + 1)
            first = True
            for k in range(1, K + 1):
                m = idx == k
                nk = int(m.sum())
                if nk == 0:
                    continue   # skip empty bin: no local effect, no node
                Xlo = X[m].copy(); Xhi = X[m].copy()
                Xlo.iloc[:, jloc] = edges[k - 1]
                Xhi.iloc[:, jloc] = edges[k]
                # pass DataFrames (keep feature names) so sklearn doesn't warn
                diff = predict(Xhi) - predict(Xlo)
                local_list.append(float(np.mean(diff)))
                count_list.append(nk)
                if first:
                    node_edges.append(float(edges[k - 1]))   # left edge of first kept bin
                    first = False
                node_edges.append(float(edges[k]))           # right edge of this kept bin
            if not local_list:
                continue
            local = np.asarray(local_list, dtype=float)
            counts = np.asarray(count_list, dtype=float)
            node_edges = np.asarray(node_edges, dtype=float)
            # accumulate, then center by the count-weighted mean (so E[ALE]=0)
            acc = np.concatenate([[0.0], np.cumsum(local)])   # length kept+1, at the nodes
            # value at each node; center using midpoint values weighted by bin counts
            mid = 0.5 * (acc[:-1] + acc[1:])
            total = counts.sum()
            cbar = float(np.sum(mid * counts) / total) if total > 0 else 0.0
            ale = acc - cbar
            ale_ranges[feat] = float(ale.max() - ale.min())
            for e, a in zip(node_edges, ale):
                all_rows.append({"feature": feat, "value": float(e), "ale": float(a)})

        if not all_rows:
            summary.append("累积局部效应分析跳过：所选特征离散值太少，无法分箱（需连续型特征）。")
            return
        tab = pd.DataFrame(all_rows)
        tab.to_csv(d / "accumulated_local_effects.csv", index=False, encoding="utf-8")
        files.append("accumulated_local_effects.csv")
        for feat, rng in sorted(ale_ranges.items(), key=lambda kv: -kv[1]):
            estimates[f"ale_range_{feat}"] = round(float(rng), 6)
        estimates["n"] = float(len(X))
        estimates["n_bins"] = float(n_bins)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ch = [f for f in chosen if f in ale_ranges]
            ncol = min(2, len(ch)); nrow = (len(ch) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow), squeeze=False)
            for i, feat in enumerate(ch):
                ax = axes[i // ncol][i % ncol]
                sub = tab[tab["feature"] == feat]
                ax.plot(sub["value"], sub["ale"], marker="o", ms=3, color="#55A868")
                ax.axhline(0, color="grey", lw=0.8)
                ax.set_xlabel(feat); ax.set_ylabel("ALE")
                ax.set_title(feat)
            for j in range(len(ch), nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(f"Accumulated local effects — {info['outcome']}")
            fig.tight_layout()
            fig.savefig(d / "accumulated_local_effects.png", dpi=150)
            plt.close(fig)
            files.append("accumulated_local_effects.png")
        except Exception:
            pass

        code += [
            "# ALE (Apley & Zhu): accumulate mean local prediction diffs within quantile bins,",
            "# then center to mean 0. Robust to correlated features (unlike PDP).",
        ]
        ranked = sorted(ale_ranges.items(), key=lambda kv: -kv[1])
        top = ranked[0][0] if ranked else "?"
        summary.append(
            f"{entry.method}（{info['model_name'].upper()} 基模型，{'分类' if info['is_clf'] else '回归'}）："
            f"结果={info['outcome']}，ALE 解释了 {len(ranked)} 个特征。效应跨度最大的特征：{top}"
            f"（ALE 跨度={ale_ranges.get(top, float('nan')):.4f}）。"
            " ⚠ ALE 用**局部**（箱内）预测差分并累积，故对相关特征**比 PDP 更稳健**（不外推到不真实组合）；"
            "曲线已中心化（均值 0），读作相对平均预测的偏移；空箱（无观测的分位区间）按标准 ALEPlot 约定**跳过**"
            "（不贡献局部效应、不注入伪 0 斜率段）；仍是模型关联非因果。"
        )
    except Exception as e:
        summary.append(f"累积局部效应分析失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4. quantile_intervals — GBM quantile-regression prediction intervals
# ---------------------------------------------------------------------------
@register("quantile_intervals")
def _branch_quantile_intervals(ctx: Ctx) -> None:
    d, entry, cfg = ctx.d, ctx.entry, ctx.cfg
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    info, err = _build_model(ctx, "分位数预测区间")
    if err:
        summary.append(err)
        return
    if info["is_clf"]:
        summary.append("分位数预测区间跳过：仅适用于连续结果（回归）；分类不确定性见 calibration / conformal。")
        return
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        X, y = info["X"], info["y"]
        lo_q = min(0.49, max(0.001, float(cfg.get("lower", 0.05))))
        hi_q = max(0.51, min(0.999, float(cfg.get("upper", 0.95))))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        preds = {}
        for q in (lo_q, 0.5, hi_q):
            m = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=0)
            m.fit(Xtr, ytr)
            preds[q] = m.predict(Xte)
        lo = np.minimum(preds[lo_q], preds[hi_q])   # guard against quantile crossing
        hi = np.maximum(preds[lo_q], preds[hi_q])
        med = preds[0.5]
        yt = yte.to_numpy(float)
        coverage = float(np.mean((yt >= lo) & (yt <= hi)))
        width = float(np.mean(hi - lo))
        nominal = hi_q - lo_q

        estimates.update({
            "nominal_coverage": round(nominal, 4), "empirical_coverage": round(coverage, 4),
            "mean_interval_width": round(width, 6),
            "lower_quantile": lo_q, "upper_quantile": hi_q,
            "n_test": float(len(yt)), "n": float(len(X)),
        })

        order = np.argsort(yt)
        pd.DataFrame({"actual": yt, "pred_median": med, "lower": lo, "upper": hi,
                      "covered": (yt >= lo) & (yt <= hi)}).to_csv(
            d / "quantile_intervals.csv", index=False, encoding="utf-8")
        files.append("quantile_intervals.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = np.arange(len(yt))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(xs, lo[order], hi[order], alpha=0.3, color="#4C72B0",
                            label=f"{int(nominal*100)}% interval")
            ax.plot(xs, med[order], color="#4C72B0", lw=1, label="median pred")
            ax.scatter(xs, yt[order], s=10, color="#C44E52", label="actual", zorder=3)
            ax.set_xlabel("test points (sorted by actual)")
            ax.set_ylabel(info["outcome"])
            ax.set_title(f"Quantile prediction intervals (coverage={coverage:.0%} vs {nominal:.0%})")
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(d / "quantile_intervals.png", dpi=150)
            plt.close(fig); files.append("quantile_intervals.png")
        except Exception:
            pass

        code += [
            "from sklearn.ensemble import GradientBoostingRegressor",
            "# fit quantile regressors at lower/0.5/upper; interval = [q_lo, q_hi]; check holdout coverage",
        ]
        verdict = ("接近名义" if abs(coverage - nominal) < 0.1
                   else ("偏窄（覆盖不足）" if coverage < nominal else "偏宽（覆盖过度）"))
        summary.append(
            f"{entry.method}（GBM 分位数回归）：结果={info['outcome']}，{int(nominal*100)}% 预测区间"
            f"在留出集的**经验覆盖率={coverage:.1%}**（名义 {nominal:.0%}，{verdict}），平均区间宽={width:.4f}。"
            f" ⚠ 经验覆盖率来自**单次随机留出**（test_size=0.25, random_state=0，n_test={int(len(yt))}），"
            "本身有抽样噪声——小留出集上的覆盖率估计可能偏离名义值数个百分点，换种子/做 CV 会有波动。"
            " ⚠ 分位数回归区间是**条件分位的模型估计**，覆盖率依赖模型拟合优度与分位交叉处理"
            "（已取 min/max 防交叉）；非分布无关保证（要严格覆盖用 conformal_prediction）；反映模型而非真值。"
        )
    except Exception as e:
        summary.append(f"分位数预测区间失败：{type(e).__name__}: {e}")


def _pd_at_points(model, eval_df, base_df, cols, predict):
    """Centered partial dependence evaluated AT each row's values of `cols`
    (Friedman's definition): for each eval row, fix `cols` to that row's values,
    average the model prediction over all of base_df. Returns a centered vector."""
    import numpy as np

    out = np.empty(len(eval_df))
    base = base_df
    for i in range(len(eval_df)):
        tmp = base.copy()
        for c in cols:
            tmp[c] = eval_df.iloc[i][c]
        out[i] = float(np.mean(predict(tmp)))
    return out - out.mean()


# ---------------------------------------------------------------------------
# 5. feature_interaction — Friedman's H-statistic (pairwise)
# ---------------------------------------------------------------------------
@register("feature_interaction")
def _branch_feature_interaction(ctx: Ctx) -> None:
    d, entry, cfg = ctx.d, ctx.entry, ctx.cfg
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    info, err = _build_model(ctx, "特征交互(H 统计量)")
    if err:
        summary.append(err)
        return
    try:
        import numpy as np
        import pandas as pd

        X = info["X"]
        predict = _predict_fn(info)
        requested = [c for c in (cfg.get("features") or []) if c in info["features"]]
        feats = requested or _top_features(info)
        n_requested = len(feats)
        feats = feats[:4]   # cap pairwise cost at C(4,2)=6 pairs
        truncated = n_requested > 4   # config gave more than we use -> disclose
        if len(feats) < 2:
            summary.append("特征交互跳过：需要 ≥2 个特征。")
            return
        # subsample to bound the O(m^2) partial-dependence-at-points cost
        m = min(len(X), int(cfg.get("sample", 120)))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), m, replace=False) if len(X) > m else np.arange(len(X))
        ev = X.iloc[idx]
        base = X.iloc[idx]

        pd1 = {f: _pd_at_points(info["model"], ev, base, [f], predict) for f in feats}
        rows = []
        tiny_den_pairs = []   # pairs whose joint-PD magnitude is negligible -> H noisy
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                fa, fb = feats[a], feats[b]
                pjk = _pd_at_points(info["model"], ev, base, [fa, fb], predict)
                num = float(np.sum((pjk - pd1[fa] - pd1[fb]) ** 2))
                den = float(np.sum(pjk ** 2))
                if den < 1e-6:   # Σpjk² ~ 0: H is a ratio of two tiny numbers -> unstable
                    tiny_den_pairs.append(f"{fa}×{fb}")
                h = float(np.sqrt(num / den)) if den > 1e-12 else 0.0
                rows.append({"feature_1": fa, "feature_2": fb,
                             "H_statistic": round(min(h, 1.0), 6)})
        tab = pd.DataFrame(rows).sort_values("H_statistic", ascending=False)
        tab.to_csv(d / "feature_interaction_H.csv", index=False, encoding="utf-8")
        files.append("feature_interaction_H.csv")
        for _, r in tab.iterrows():
            estimates[f"H_{r['feature_1']}__{r['feature_2']}"] = float(r["H_statistic"])
        estimates["max_H"] = float(tab["H_statistic"].max())
        estimates["n_pairs"] = float(len(tab))
        estimates["n"] = float(len(X))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = tab.head(10)[::-1]
            labels = [f"{a}×{b}" for a, b in zip(top["feature_1"], top["feature_2"])]
            fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.4)))
            ax.barh(labels, top["H_statistic"], color="#8172B3")
            ax.set_xlabel("Friedman's H (0=no interaction, 1=pure interaction)")
            ax.set_title(f"Pairwise feature interaction — {info['outcome']}")
            fig.tight_layout(); fig.savefig(d / "feature_interaction.png", dpi=150)
            plt.close(fig); files.append("feature_interaction.png")
        except Exception:
            pass

        code += [
            "# Friedman's H_jk = sqrt( Σ(PD_jk - PD_j - PD_k)^2 / Σ PD_jk^2 ), centered PDs",
            "# evaluated at the data points (partial dependence at observed feature values)",
        ]
        top = tab.iloc[0]
        trunc_note = (f" ⚠ config 提供了 {n_requested} 个特征，仅用本模型重要性最高的前 4 个"
                      f"（{'、'.join(feats)}）控制成对成本。" if truncated else "")
        tiny_note = (f" ⚠ 以下特征对的联合部分依赖幅度极小（Σpjk²≈0）：{'、'.join(tiny_den_pairs)}——"
                     "其 H 是两个微小量之比，数值不稳/噪声大，不应据以排序交互强弱。"
                     if tiny_den_pairs else "")
        summary.append(
            f"{entry.method}（{info['model_name'].upper()} 基模型）：结果={info['outcome']}，"
            f"交互最强的特征对：{top['feature_1']}×{top['feature_2']}（H={top['H_statistic']:.3f}）。"
            f"H≈0 表示该对无交互（效应可加），越接近 1 交互越主导。"
            f"{trunc_note}{tiny_note}"
            " ⚠ H 统计量基于模型的部分依赖（在数据点上估计、已子采样以控成本），反映**模型**学到的交互"
            "而非数据中的因果交互；相关特征下 PD 外推可使 H 偏高；H 是相对量纲。"
        )
    except Exception as e:
        summary.append(f"特征交互失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 6. surrogate_model — interpretable global surrogate (shallow tree) + fidelity
# ---------------------------------------------------------------------------
@register("surrogate_model")
def _branch_surrogate_model(ctx: Ctx) -> None:
    d, entry, cfg = ctx.d, ctx.entry, ctx.cfg
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    info, err = _build_model(ctx, "代理模型")
    if err:
        summary.append(err)
        return
    try:
        import numpy as np
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

        X, feats, is_clf = info["X"], info["features"], info["is_clf"]
        max_depth = max(2, min(6, int(cfg.get("max_depth", 3))))
        bb = info["model"]

        if is_clf:
            target = bb.predict(X)                 # mimic the black box's labels
            surr = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, target)
            from sklearn.metrics import accuracy_score
            fidelity = float(accuracy_score(target, surr.predict(X)))
            fid_name = "fidelity_accuracy"
        else:
            target = bb.predict(X)
            surr = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, target)
            ss_res = float(np.sum((target - surr.predict(X)) ** 2))
            ss_tot = float(np.sum((target - np.mean(target)) ** 2))
            fidelity = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
            fid_name = "fidelity_r2"

        rules = export_text(surr, feature_names=list(feats))
        (d / "surrogate_tree_rules.txt").write_text(rules, encoding="utf-8")
        files.append("surrogate_tree_rules.txt")

        imp = pd.DataFrame({"feature": feats, "surrogate_importance": surr.feature_importances_}
                           ).sort_values("surrogate_importance", ascending=False)
        imp.to_csv(d / "surrogate_importance.csv", index=False, encoding="utf-8")
        files.append("surrogate_importance.csv")

        estimates.update({
            fid_name: round(fidelity, 6), "surrogate_max_depth": float(max_depth),
            "surrogate_n_leaves": float(surr.get_n_leaves()), "n": float(len(X)),
        })
        for _, r in imp.iterrows():
            estimates[f"surrogate_imp_{r['feature']}"] = round(float(r["surrogate_importance"]), 6)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree

            fig, ax = plt.subplots(figsize=(min(16, 3 + 2 * max_depth), 6))
            plot_tree(surr, feature_names=list(feats), filled=True, fontsize=7,
                      max_depth=max_depth, ax=ax, impurity=False)
            ax.set_title(f"Global surrogate tree — {info['outcome']} ({fid_name}={fidelity:.2f})")
            fig.tight_layout(); fig.savefig(d / "surrogate_tree.png", dpi=140)
            plt.close(fig); files.append("surrogate_tree.png")
        except Exception:
            pass

        code += [
            "from sklearn.tree import DecisionTreeRegressor, export_text",
            "# fit a shallow tree to the BLACK BOX's predictions; fidelity = how well it mimics",
        ]
        trust = ("高保真，可信地用树解释黑箱" if fidelity >= 0.8
                 else "中等保真，树仅近似" if fidelity >= 0.6 else "**低保真，树不能代表黑箱**")
        summary.append(
            f"{entry.method}（深度≤{max_depth} 决策树代理 {info['model_name'].upper()} 黑箱，"
            f"{'分类' if is_clf else '回归'}）：结果={info['outcome']}，"
            f"保真度（{fid_name}）={fidelity:.3f} —— {trust}。规则见 surrogate_tree_rules.txt。"
            " ⚠ 代理树解释的是**黑箱模型**（非真实数据关系），且**仅在保真度高时**可信；"
            "低保真说明黑箱含树无法捕捉的交互/非线性——勿用树过度简化解读。"
        )
    except Exception as e:
        summary.append(f"代理模型失败：{type(e).__name__}: {e}")
