"""Executor: run the chosen analysis and persist code / tables / figures / report
to outputs/<timestamp>_<analysis>/. Reuses the empirical-analysis-python stack
(statsmodels + matplotlib)."""

from __future__ import annotations

import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.profiler.profile import read_table

_REGRESSION = {"ols_regression", "panel_fixed_effects", "did"}


class RunResult(BaseModel):
    analysis_id: str
    method: str
    output_dir: str
    files: list[str] = Field(default_factory=list)
    report_path: str
    summary: str = ""
    estimates: dict[str, float] = Field(default_factory=dict)


def _run_dir(root: str, entry_id: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    d = Path(root) / f"{ts}_{entry_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_did_treatment(df, fp: DataFingerprint) -> list[str]:
    """The DID treatment is the binary that varies WITHIN units over time (a
    treatment that switches on), not a fixed group flag. Returns [] if none vary."""
    if not (fp.unit_col and fp.time_col):
        return fp.treatment_candidates[:1]
    best = None
    for name in fp.treatment_candidates:
        frac = float((df.groupby(fp.unit_col)[name].nunique() > 1).mean())
        if frac > 0 and (best is None or frac > best[0]):
            best = (frac, name)
    return [best[1]] if best else []


def _regression(df, fp: DataFingerprint, entry: AnalysisEntry):
    import statsmodels.formula.api as smf

    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    if not cont:
        raise ValueError("没有连续型因变量，无法回归")
    y = cont[0]
    exclude = {y, fp.unit_col, fp.time_col}

    fe_terms: list[str] = []
    if entry.id in {"panel_fixed_effects", "did"} and fp.unit_col and fp.time_col:
        fe_terms = [f"C(Q('{fp.unit_col}'))", f"C(Q('{fp.time_col}'))"]

    if entry.id == "did" and fp.treatment_candidates:
        rhs_vars = _pick_did_treatment(df, fp) or fp.treatment_candidates[:1]
    else:
        rhs_vars = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ][:5]

    rhs = [f"Q('{v}')" for v in rhs_vars] + fe_terms
    formula = f"Q('{y}') ~ " + (" + ".join(rhs) if rhs else "1")
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
    return y, rhs_vars, formula, model


def _heatmap(corr, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception:
        pass


def _coef_plot(model, rhs_vars, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [f"Q('{v}')" for v in rhs_vars if f"Q('{v}')" in model.params.index]
        if not names:
            return
        coefs = model.params[names]
        errs = model.bse[names]
        labels = [v for v in rhs_vars if f"Q('{v}')" in model.params.index]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.errorbar(coefs.values, range(len(names)), xerr=1.96 * errs.values, fmt="o")
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("coefficient (95% CI)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception:
        pass


def _report(entry, fp, summary, files, override) -> str:
    lines = [
        f"# ResearchForge 分析报告：{entry.method}",
        "",
        f"- 数据：`{fp.path}`（{fp.n_rows} 行 × {fp.n_cols} 列）",
        f"- 分析：{entry.method}（{entry.family} / {entry.goal}）",
        "",
    ]
    if override:
        lines += ["> ⚠️ **知情覆盖**：该分析部分前提未满足，结果仅供参考、需谨慎解读。", ""]
    lines += ["## 结果摘要", *[f"- {s}" for s in summary], ""]
    if entry.biases:
        lines += ["## 偏差提醒（需读者判断）", *[f"- {b}" for b in entry.biases], ""]
    lines += ["## 产物文件", *[f"- `{f}`" for f in files]]
    return "\n".join(lines)


def run_analysis(
    fp: DataFingerprint,
    entry: AnalysisEntry,
    output_root: str = "outputs",
    override: bool = False,
) -> RunResult:
    df = read_table(Path(fp.path))
    d = _run_dir(output_root, entry.id)
    files: list[str] = []
    summary: list[str] = []
    estimates: dict[str, float] = {}
    code: list[str] = ["import pandas as pd", f"df = pd.read_csv(r'{fp.path}')", ""]

    if entry.id == "descriptive_stats":
        df.describe(include="all").transpose().to_csv(d / "table_describe.csv", encoding="utf-8")
        files.append("table_describe.csv")
        summary.append(f"描述统计完成：{df.shape[0]} 行 × {df.shape[1]} 列")
        high_card = [c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.n_unique > 50]
        if high_card:
            summary.append(f"注意：{len(high_card)} 个高基数列（如 {high_card[0]}）描述统计意义有限。")
        code.append("df.describe(include='all').transpose().to_csv('table_describe.csv')")

    elif entry.id == "correlation":
        num = df.select_dtypes(include="number")
        corr = num.corr()
        corr.to_csv(d / "correlation.csv", encoding="utf-8")
        files.append("correlation.csv")
        _heatmap(corr, d / "correlation_heatmap.png")
        files.append("correlation_heatmap.png")
        summary.append(f"相关分析完成：{num.shape[1]} 个数值变量")
        code += ["num = df.select_dtypes(include='number')", "num.corr().to_csv('correlation.csv')"]

    elif entry.id in _REGRESSION:
        y, rhs_vars, formula, model = _regression(df, fp, entry)
        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")
        model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
        files.append("coefficients.csv")
        _coef_plot(model, rhs_vars, d / "coefficients.png")
        files.append("coefficients.png")
        for v in rhs_vars:
            kn = f"Q('{v}')"
            if kn in model.params.index:
                estimates[v] = float(model.params[kn])
        key = ""
        if rhs_vars:
            kname = f"Q('{rhs_vars[0]}')"
            if kname in model.params.index:
                key = f"，关键系数 {rhs_vars[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
        n_cont = sum(1 for c in fp.columns if c.kind == "continuous")
        dv_note = f"（数据有 {n_cont} 个连续列，默认取 {y} 为因变量）" if n_cont > 1 else ""
        summary.append(f"{entry.method} 完成：因变量 {y}{key}{dv_note}")
        if not rhs_vars:
            summary.append("⚠️ 无可用解释变量，仅拟合了截距模型，结果无解释意义。")
        if entry.id == "did" and rhs_vars and fp.unit_col:
            if int(df.groupby(fp.unit_col)[rhs_vars[0]].nunique().max()) <= 1:
                summary.append(
                    f"⚠️ 处理变量 {rhs_vars[0]} 在每个单位内不随时间变化，可能不是有效的 DID 处理。"
                )
        code += [
            "import statsmodels.formula.api as smf",
            f'model = smf.ols("{formula}", data=df).fit(cov_type="HC1")',
            "print(model.summary())",
        ]

    elif entry.id == "group_comparison":
        from scipy import stats

        _excl = {fp.unit_col, fp.time_col}
        bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
        cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
        # prefer a binary group; otherwise the lowest-cardinality categorical, so a
        # high-cardinality unit/id column is never picked as the grouping variable.
        cat_cols.sort(key=lambda name: int(df[name].nunique()))
        group_candidates = bin_cols + cat_cols
        cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
        group_col = group_candidates[0] if group_candidates else None
        outcome = cont_cols[0] if cont_cols else None

        if group_col is None or outcome is None:
            summary.append("组间比较失败：未找到分组变量或连续结果变量。")
        else:
            # Per-group means/counts
            group_means = df.groupby(group_col)[outcome].agg(["mean", "count", "std"])
            group_means.to_csv(d / "group_means.csv", encoding="utf-8")
            files.append("group_means.csv")

            # Split outcome by group levels, drop NaN
            levels = df[group_col].dropna().unique().tolist()
            groups = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
            n_groups = len(groups)

            if n_groups == 2:
                stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = "Welch t-test"
            else:
                stat, p = stats.f_oneway(*groups)
                test_name = "one-way ANOVA"

            estimates["statistic"] = float(stat)
            estimates["pvalue"] = float(p)

            # Boxplot
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 4))
                plot_data = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
                ax.boxplot(plot_data, tick_labels=[str(lv) for lv in levels])
                ax.set_xlabel(group_col)
                ax.set_ylabel(outcome)
                ax.set_title(f"{outcome} by {group_col}")
                fig.tight_layout()
                fig.savefig(d / "boxplot.png", dpi=120)
                plt.close(fig)
                files.append("boxplot.png")
            except Exception:
                pass

            summary.append(
                f"{entry.method} 完成：{outcome} 按 {group_col} 分 {n_groups} 组，"
                f"统计量={stat:.4f}，p={p:.3g}"
            )
            code += [
                "from scipy import stats",
                f"groups = [df.loc[df['{group_col}'] == lv, '{outcome}'].dropna().values",
                f"         for lv in df['{group_col}'].dropna().unique()]",
                "stat, p = stats.ttest_ind(*groups[:2], equal_var=False)  # or f_oneway(*groups)",
                "print(f'statistic={stat:.4f}, p={p:.3g}')",
            ]

    elif entry.id == "random_forest":
        cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
        binary_cols = [c.name for c in fp.columns if c.kind == "binary"]

        # Prefer a continuous outcome (regression). Classify a binary outcome only
        # when there is no continuous column — a lone binary is usually a
        # treatment / flag *feature*, not the prediction target. This prevents
        # silently running the wrong analysis on the common "outcome + indicator" shape.
        if cont_cols:
            outcome, is_clf = cont_cols[0], False
        elif binary_cols:
            outcome, is_clf = binary_cols[0], True
        else:
            outcome, is_clf = None, False

        exclude = {outcome, fp.unit_col, fp.time_col}
        features = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ]

        if outcome is None:
            summary.append("随机森林失败：未找到合适的结果变量（需要连续型或二值列）。")
        elif not features:
            summary.append("随机森林失败：未找到可用的特征列。")
        else:
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.model_selection import train_test_split

                mask = df[features].notna().all(axis=1) & df[outcome].notna()
                X = df.loc[mask, features]
                y = df.loc[mask, outcome]

                if y.nunique() < 2:
                    raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")

                split_kwargs = {"test_size": 0.25, "random_state": 0}
                if is_clf and int(y.value_counts().min()) >= 2:
                    split_kwargs["stratify"] = y
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

                model = (
                    RandomForestClassifier(n_estimators=200, random_state=0)
                    if is_clf
                    else RandomForestRegressor(n_estimators=200, random_state=0)
                )

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                import pandas as pd
                imp_df = pd.DataFrame(
                    {"feature": features, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
                imp_df.to_csv(d / "feature_importances.csv", index=False, encoding="utf-8")
                files.append("feature_importances.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
                    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                    ax.set_xlabel("importance")
                    ax.set_title(f"Feature importances — {outcome}")
                    fig.tight_layout()
                    fig.savefig(d / "feature_importances.png", dpi=120)
                    plt.close(fig)
                    files.append("feature_importances.png")
                except Exception:
                    pass

                estimates["test_score"] = float(score)
                score_label = "accuracy" if is_clf else "R²"
                task_label = "分类" if is_clf else "回归"
                summary.append(
                    f"{entry.method} 完成：{task_label}预测 {outcome}，"
                    f"测试集得分={score:.4f}（{score_label}）"
                )
                code += [
                    "from sklearn.ensemble import "
                    + ("RandomForestClassifier" if is_clf else "RandomForestRegressor"),
                    "from sklearn.model_selection import train_test_split",
                    f"features = {features!r}",
                    f"X = df[features].dropna()",
                    f"y = df.loc[X.index, '{outcome}'].dropna()",
                    f"X = X.loc[y.index]",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)",
                    "model = "
                    + ("RandomForestClassifier" if is_clf else "RandomForestRegressor")
                    + "(n_estimators=200, random_state=0)",
                    "model.fit(X_train, y_train)",
                    f"print('score:', model.score(X_test, y_test))",
                ]
            except Exception as err:
                summary.append(f"随机森林执行失败：{err}")

    elif entry.id == "xgboost":
        cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
        binary_cols = [c.name for c in fp.columns if c.kind == "binary"]

        # Prefer a continuous outcome (regression). Classify a binary outcome only
        # when there is no continuous column — a lone binary is usually a
        # treatment / flag *feature*, not the prediction target. This prevents
        # silently running the wrong analysis on the common "outcome + indicator" shape.
        if cont_cols:
            outcome, is_clf = cont_cols[0], False
        elif binary_cols:
            outcome, is_clf = binary_cols[0], True
        else:
            outcome, is_clf = None, False

        exclude = {outcome, fp.unit_col, fp.time_col}
        features = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ]

        if outcome is None:
            summary.append("XGBoost 失败：未找到合适的结果变量（需要连续型或二值列）。")
        elif not features:
            summary.append("XGBoost 失败：未找到可用的特征列。")
        else:
            try:
                from xgboost import XGBClassifier, XGBRegressor
                from sklearn.model_selection import train_test_split

                mask = df[features].notna().all(axis=1) & df[outcome].notna()
                X = df.loc[mask, features]
                y = df.loc[mask, outcome]

                if y.nunique() < 2:
                    raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")

                split_kwargs = {"test_size": 0.25, "random_state": 0}
                if is_clf and int(y.value_counts().min()) >= 2:
                    split_kwargs["stratify"] = y
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

                model = (
                    XGBClassifier(n_estimators=200, random_state=0, verbosity=0)
                    if is_clf
                    else XGBRegressor(n_estimators=200, random_state=0, verbosity=0)
                )

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                import pandas as pd
                imp_df = pd.DataFrame(
                    {"feature": features, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
                imp_df.to_csv(d / "feature_importances.csv", index=False, encoding="utf-8")
                files.append("feature_importances.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
                    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                    ax.set_xlabel("importance")
                    ax.set_title(f"Feature importances — {outcome}")
                    fig.tight_layout()
                    fig.savefig(d / "feature_importances.png", dpi=120)
                    plt.close(fig)
                    files.append("feature_importances.png")
                except Exception:
                    pass

                estimates["test_score"] = float(score)
                score_label = "accuracy" if is_clf else "R²"
                task_label = "分类" if is_clf else "回归"
                summary.append(
                    f"{entry.method} 完成：{task_label}预测 {outcome}，"
                    f"测试集得分={score:.4f}（{score_label}）"
                )
                code += [
                    "from xgboost import "
                    + ("XGBClassifier" if is_clf else "XGBRegressor"),
                    "from sklearn.model_selection import train_test_split",
                    f"features = {features!r}",
                    f"X = df[features].dropna()",
                    f"y = df.loc[X.index, '{outcome}'].dropna()",
                    f"X = X.loc[y.index]",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)",
                    "model = "
                    + ("XGBClassifier" if is_clf else "XGBRegressor")
                    + "(n_estimators=200, random_state=0, verbosity=0)",
                    "model.fit(X_train, y_train)",
                    f"print('score:', model.score(X_test, y_test))",
                ]
            except Exception as err:
                summary.append(f"XGBoost 执行失败：{err}")

    elif entry.id == "kmeans_clustering":
        features = [
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ]
        X = df[features].dropna()  # keep X.index for alignment
        if len(features) < 2 or len(X) < 4:
            summary.append("K-means 跳过：连续特征不足或有效样本太少。")
        else:
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                from sklearn.decomposition import PCA
                import numpy as np
                import pandas as pd

                Xs = StandardScaler().fit_transform(X)
                n = len(Xs)
                k_max = max(2, min(6, n // 10))
                best = None  # (score, k, labels)
                for k in range(2, min(k_max, n - 1) + 1):
                    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
                    if len(set(labels)) < 2:
                        continue
                    score = silhouette_score(Xs, labels)
                    if best is None or score > best[0]:
                        best = (score, k, labels)

                if best is None:
                    summary.append("K-means 未能形成有效聚类（数据可能近常数）。")
                else:
                    score, k, labels = best
                    k = len(set(labels))  # actual cluster count (KMeans may collapse on duplicate points)
                    assign = pd.DataFrame({"row": X.index, "cluster": labels})
                    assign.to_csv(d / "cluster_assignments.csv", index=False, encoding="utf-8")
                    files.append("cluster_assignments.csv")

                    profile_out = X.groupby(labels).mean()
                    size = X.groupby(labels).size()
                    profile_out["size"] = size.values
                    profile_out.to_csv(d / "cluster_profile.csv", encoding="utf-8")
                    files.append("cluster_profile.csv")

                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        n_components = min(2, len(features))
                        pca_coords = PCA(n_components=n_components).fit_transform(Xs)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        if n_components == 2:
                            ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10", s=20)
                        else:
                            ax.scatter(pca_coords[:, 0], [0] * len(pca_coords), c=labels, cmap="tab10", s=20)
                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2" if n_components == 2 else "")
                        ax.set_title(f"K-means (k={k}) — PCA projection")
                        fig.tight_layout()
                        fig.savefig(d / "pca_scatter.png", dpi=120)
                        plt.close(fig)
                        files.append("pca_scatter.png")
                    except Exception:
                        pass

                    estimates["silhouette"] = float(score)
                    estimates["k"] = float(k)
                    summary.append(
                        f"{entry.method} 完成：在 {len(features)} 个连续特征上聚成 {k} 类，silhouette={score:.4f}"
                    )
                    code += [
                        "from sklearn.preprocessing import StandardScaler",
                        "from sklearn.cluster import KMeans",
                        "from sklearn.metrics import silhouette_score",
                        f"features = {features!r}",
                        "X = df[features].dropna()",
                        "Xs = StandardScaler().fit_transform(X)",
                        f"labels = KMeans(n_clusters={k}, random_state=0, n_init=10).fit_predict(Xs)",
                        "print('silhouette:', silhouette_score(Xs, labels))",
                    ]
            except Exception as err:
                summary.append(f"K-means 执行失败：{err}")

    elif entry.id == "pca":
        features = [
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ]
        X = df[features].dropna()
        if len(features) < 2 or len(X) < 3:
            summary.append("PCA 跳过：连续特征不足或样本太少。")
        else:
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                import numpy as np
                import pandas as pd

                Xs = StandardScaler().fit_transform(X)
                n_comp = min(len(features), 10, len(X) - 1)
                pca = PCA(n_components=n_comp).fit(Xs)
                evr = pca.explained_variance_ratio_

                # explained_variance.csv: component (PC1..), explained_variance_ratio, cumulative
                ev_df = pd.DataFrame({
                    "component": [f"PC{i+1}" for i in range(n_comp)],
                    "explained_variance_ratio": evr,
                    "cumulative": np.cumsum(evr),
                })
                ev_df.to_csv(d / "explained_variance.csv", index=False, encoding="utf-8")
                files.append("explained_variance.csv")

                # loadings.csv: rows=features, cols=PC1..n
                load_df = pd.DataFrame(
                    pca.components_.T,
                    index=features,
                    columns=[f"PC{i+1}" for i in range(n_comp)],
                )
                load_df.to_csv(d / "loadings.csv", encoding="utf-8")
                files.append("loadings.csv")

                # scree plot (bar of evr) -> pca_scree.png
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar([f"PC{i+1}" for i in range(n_comp)], evr)
                    ax.set_xlabel("component")
                    ax.set_ylabel("explained variance ratio")
                    ax.set_title("PCA scree plot")
                    fig.tight_layout()
                    fig.savefig(d / "pca_scree.png", dpi=120)
                    plt.close(fig)
                    files.append("pca_scree.png")
                except Exception:
                    pass

                estimates["pc1_explained_ratio"] = float(evr[0])
                estimates["n_components"] = float(n_comp)
                estimates["cum_explained_top2"] = float(np.cumsum(evr)[min(1, n_comp - 1)])
                summary.append(
                    f"{entry.method} 完成：{len(features)} 个连续特征 -> {n_comp} 个主成分，"
                    f"PC1 解释方差={evr[0]:.1%}"
                )
                code += [
                    "from sklearn.preprocessing import StandardScaler",
                    "from sklearn.decomposition import PCA",
                    "import numpy as np",
                    f"features = {features!r}",
                    "X = df[features].dropna()",
                    "Xs = StandardScaler().fit_transform(X)",
                    f"pca = PCA(n_components={n_comp}).fit(Xs)",
                    "print('explained variance ratio:', pca.explained_variance_ratio_)",
                ]
            except Exception as err:
                summary.append(f"PCA 执行失败：{err}")

    elif entry.id == "arima":
        time_col = fp.time_col
        # value_col: forecast the first continuous column. Time columns are
        # datetime/id/count kind (never continuous), so they are never picked here.
        value_col = next((c.name for c in fp.columns if c.kind == "continuous"), None)

        if time_col is None or value_col is None:
            summary.append(
                "ARIMA 失败：未找到时间列或连续值列，请检查数据结构。"
            )
        else:
            try:
                from statsmodels.tsa.arima.model import ARIMA

                sorted_df = df.sort_values(time_col)
                dup = int(sorted_df[time_col].duplicated().sum())
                if dup:
                    sorted_df = sorted_df.drop_duplicates(subset=time_col, keep="first")
                    summary.append(f"注意：{dup} 个重复时间点已去重（保留首次）。")
                y = sorted_df[value_col].astype(float).reset_index(drop=True)
                if y.nunique() < 2 or len(y) < 10:
                    raise ValueError(f"序列有效观测不足或近常数（n={len(y)}），无法拟合 ARIMA")

                model = ARIMA(y, order=(1, 1, 1)).fit()

                (d / "model_summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("model_summary.txt")

                steps = 10
                fc = model.forecast(steps=steps)
                import pandas as _pd
                fc_df = _pd.DataFrame({"step": list(range(1, steps + 1)), "forecast": fc.tolist()})
                fc_df.to_csv(d / "forecast.csv", index=False, encoding="utf-8")
                files.append("forecast.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(len(y)), y, label="observed")
                    fc_x = list(range(len(y), len(y) + steps))
                    ax.plot(fc_x, fc.tolist(), color="red", linestyle="--", label="forecast")
                    ax.set_xlabel("period index")
                    ax.set_ylabel(value_col)
                    ax.set_title(f"ARIMA(1,1,1) — {value_col}")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(d / "forecast.png", dpi=120)
                    plt.close(fig)
                    files.append("forecast.png")
                except Exception:
                    pass

                estimates["aic"] = float(model.aic)
                summary.append(
                    f"{entry.method} 完成：对 {value_col} 拟合 ARIMA(1,1,1)，"
                    f"AIC={model.aic:.2f}，预测未来 {steps} 期"
                )
                code += [
                    "from statsmodels.tsa.arima.model import ARIMA",
                    f"y = df.sort_values('{time_col}')['{value_col}'].astype(float).reset_index(drop=True)",
                    "model = ARIMA(y, order=(1, 1, 1)).fit()",
                    "print(model.summary())",
                    f"fc = model.forecast(steps={steps})",
                ]
            except Exception as err:
                summary.append(f"ARIMA 拟合失败：{err}")

    elif entry.id == "logistic_regression":
        import statsmodels.formula.api as smf

        # identify outcome (first binary column, excluding unit/time) and predictors
        binary_cols = [
            c.name
            for c in fp.columns
            if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col}
        ]
        outcome = binary_cols[0] if binary_cols else None
        exclude = {outcome, fp.unit_col, fp.time_col}
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count"} and c.name not in exclude
        ][:5]

        if outcome is None:
            summary.append("逻辑回归失败：未找到二值结果变量。")
        else:
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            try:
                model = smf.logit(formula, data=df).fit(disp=False)
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                key = ""
                if predictors:
                    kname = f"Q('{predictors[0]}')"
                    if kname in model.params.index:
                        key = f"，关键系数 {predictors[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
                amb = (
                    f"（数据有 {len(binary_cols)} 个二值列，已取 {outcome}；若它实为处理/标志变量请改选）"
                    if len(binary_cols) > 1
                    else ""
                )
                summary.append(f"{entry.method} 完成：结果变量 {outcome}{key}{amb}")
                code += [
                    "import statsmodels.formula.api as smf",
                    f'model = smf.logit("{formula}", data=df).fit(disp=False)',
                    "print(model.summary())",
                ]
            except Exception as err:
                summary.append(f"逻辑回归未收敛/失败：{err}")

    elif entry.id == "poisson_regression":
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        import numpy as np

        _excl = {fp.unit_col, fp.time_col}
        count_cols = [
            c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
        ]
        outcome = count_cols[0] if count_cols else None

        if outcome is None:
            summary.append("泊松回归失败：未找到计数型结果变量。")
        else:
            amb = (
                f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
                if len(count_cols) > 1
                else ""
            )
            exclude = {outcome, fp.unit_col, fp.time_col}
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in exclude
            ][:5]
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            recipe = (
                "import statsmodels.formula.api as smf\n"
                "import statsmodels.api as sm\n"
                f'model = smf.glm("{formula}", data=df, family=sm.families.Poisson()).fit()\n'
                "print(model.summary())"
            )
            try:
                model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                tab = model.summary2().tables[1].copy()
                tab["rate_ratio"] = np.exp(model.params.values)
                tab.to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                summary.append(
                    f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}"
                )
                code += [recipe]
            except Exception as err:
                summary.append(f"泊松回归失败：{err}")

    elif entry.id == "negative_binomial_regression":
        import statsmodels.formula.api as smf
        import numpy as np

        _excl = {fp.unit_col, fp.time_col}
        count_cols = [
            c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
        ]
        outcome = count_cols[0] if count_cols else None

        if outcome is None:
            summary.append("负二项回归失败：未找到计数型结果变量。")
        else:
            amb = (
                f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
                if len(count_cols) > 1
                else ""
            )
            exclude = {outcome, fp.unit_col, fp.time_col}
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in exclude
            ][:5]
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            recipe = (
                "import statsmodels.formula.api as smf\n"
                f'model = smf.negativebinomial("{formula}", data=df).fit(disp=False)\n'
                "print(model.summary())"
            )
            try:
                model = smf.negativebinomial(formula, data=df).fit(disp=False)
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                tab = model.summary2().tables[1].copy()
                # model.params includes an 'alpha' (dispersion) row at the end;
                # summary2().tables[1] also includes it — lengths always match,
                # so exp() of all rows is safe (exp(alpha) is a positive scalar,
                # harmless alongside the log-rate coefficients).
                tab["rate_ratio"] = np.exp(model.params.values)
                tab.to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                summary.append(
                    f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}"
                )
                code += [recipe]
            except Exception as err:
                summary.append(f"负二项回归失败：{err}")

    elif entry.id == "mixed_effects":
        import statsmodels.formula.api as smf

        # outcome: first continuous column
        outcome = next((c.name for c in fp.columns if c.kind == "continuous"), None)
        if outcome is None:
            summary.append("混合模型失败：未找到连续结果变量。")
        else:
            # group_col: prefer unit_col; else first categorical/binary that is not outcome
            if fp.unit_col:
                group_col = fp.unit_col
            else:
                group_col = next(
                    (
                        c.name
                        for c in fp.columns
                        if c.kind in {"categorical", "binary"} and c.name != outcome
                    ),
                    None,
                )
            if group_col is None:
                summary.append("混合模型失败：未找到分组变量(随机效应)。")
            else:
                predictors = [
                    c.name
                    for c in fp.columns
                    if c.kind in {"continuous", "count", "binary"}
                    and c.name not in {outcome, group_col, fp.unit_col, fp.time_col}
                ][:5]
                rhs = [f"Q('{v}')" for v in predictors]
                # Control for time on panel data — otherwise a staggered treatment is
                # confounded with the time trend (mirrors _regression's FE handling).
                if fp.time_col and fp.time_col != group_col:
                    rhs.append(f"C(Q('{fp.time_col}'))")
                formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
                try:
                    model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit()
                    (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                    files.append("summary.txt")
                    try:
                        import pandas as pd
                        pd.DataFrame(model.summary().tables[1]).to_csv(
                            d / "coefficients.csv", encoding="utf-8"
                        )
                    except Exception:
                        import pandas as pd
                        model.params.to_frame(name="coef").to_csv(
                            d / "coefficients.csv", encoding="utf-8"
                        )
                    files.append("coefficients.csv")
                    for v in predictors:
                        kn = f"Q('{v}')"
                        if kn in model.params.index:
                            estimates[v] = float(model.params[kn])
                    summary.append(
                        f"{entry.method} 完成：结果变量 {outcome}，随机效应分组 {group_col}，"
                        f"固定效应 {len(predictors)} 个"
                    )
                    code += [
                        "import statsmodels.formula.api as smf",
                        f'model = smf.mixedlm("{formula}", data=df, groups=df["{group_col}"]).fit()',
                        "print(model.summary())",
                    ]
                except Exception as err:
                    summary.append(f"混合模型未收敛/失败：{err}")

    elif entry.id == "iv_regression":
        summary.append(
            "工具变量回归（2SLS）需要你指定外生工具变量（instrument），引擎无法自动识别。"
            "请在指定工具变量后手动运行；或先用 panel_fixed_effects / did 作为可自动执行的替代。"
        )

    else:
        summary.append(f"{entry.method} 暂未接入执行器（需补依赖/封装），仅生成占位报告。")

    (d / "analysis_code.py").write_text("\n".join(code), encoding="utf-8")
    files.append("analysis_code.py")

    (d / "report.md").write_text(_report(entry, fp, summary, files, override), encoding="utf-8")
    files.append("report.md")

    return RunResult(
        analysis_id=entry.id,
        method=entry.method,
        output_dir=str(d),
        files=files,
        report_path=str(d / "report.md"),
        summary="\n".join(summary),
        estimates=estimates,
    )
