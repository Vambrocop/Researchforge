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


def _run_dir(root: str, entry_id: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    d = Path(root) / f"{ts}_{entry_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        rhs_vars = fp.treatment_candidates[:1]
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
    code: list[str] = ["import pandas as pd", f"df = pd.read_csv(r'{fp.path}')", ""]

    if entry.id == "descriptive_stats":
        df.describe(include="all").transpose().to_csv(d / "table_describe.csv", encoding="utf-8")
        files.append("table_describe.csv")
        summary.append(f"描述统计完成：{df.shape[0]} 行 × {df.shape[1]} 列")
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
        key = ""
        if rhs_vars:
            kname = f"Q('{rhs_vars[0]}')"
            if kname in model.params.index:
                key = f"，关键系数 {rhs_vars[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
        summary.append(f"{entry.method} 完成：因变量 {y}{key}")
        code += [
            "import statsmodels.formula.api as smf",
            f'model = smf.ols("{formula}", data=df).fit(cov_type="HC1")',
            "print(model.summary())",
        ]

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
    )
