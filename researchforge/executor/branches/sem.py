"""Branch handlers for the sem family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _sem_latents,
    _sem_via_lavaan,
    _sem_via_semopy,
)


@register("bayesian_sem")
def _branch_bayesian_sem(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    n_cont = sum(1 for c in fp.columns if c.kind == "continuous" and c.name not in _excl)
    has_blavaan = rbridge.r_available() and rbridge.r_package_available("blavaan")
    # Bayesian SEM needs blavaan + a JAGS or Stan backend (a C++ toolchain), AND a
    # theory-driven measurement model — none auto-inferable. Honest-degrade to the
    # runnable frequentist `sem` rather than trigger a heavy/fragile toolchain install.
    if n_cont < 3:
        summary.append("贝叶斯 SEM 跳过：需要 ≥3 个连续指标变量。")
    else:
        backend = (
            "已检测到 blavaan，但仍需 JAGS 或 Stan(C++ 编译工具链) 后端运行，且需你提供测量模型"
            if has_blavaan
            else "未检测到 R 包 blavaan（且需 JAGS 或 Stan/RTools 编译后端）"
        )
        summary.append(
            f"贝叶斯 SEM 暂以诚实降级提示（{backend}）。安装路径：install.packages('blavaan') + "
            "装 JAGS（jags 官网二进制）或 Stan/RTools；并需理论测量模型。"
            "**可直接运行的替代**：① `sem`（频率派 CB-SEM，经 lavaan/semopy，"
            "用 config={\"model_spec\":\"<lavaan 语法>\"} 指定结构，给点估计 + CFI/TLI/RMSEA）；"
            "② `efa`（探索因子结构）。贝叶斯 SEM 的后验分布/可信区间待后端就绪后接。"
        )
        code += [
            "# 贝叶斯 SEM（待 blavaan + JAGS/Stan 后端）",
            "library(blavaan)  # bsem(model, data, target='stan'|'jags')  —— 需测量模型 + 编译后端",
        ]



@register("pls_sem")
def _branch_pls_sem(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    summary.append(
        "PLS-SEM（偏最小二乘结构方程）需要你指定**测量模型**（哪些指标→哪个潜变量）与结构路径；"
        "引擎无法自动推断（随意分组会产出无意义结果，故不自动跑）。请指定测量模型后用 plspm / SmartPLS 运行；"
        "或先用 **SEM**（CB-SEM，自动单因子 CFA，经 lavaan/semopy）/ **EFA**（探索因子结构）作可自动执行的替代。"
    )



@register("sem")
def _branch_sem(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    _excl = {fp.unit_col, fp.time_col}
    indicators = [
        c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl
    ]
    # config={"model_spec": "<lavaan/semopy syntax>"} lets the user supply their
    # theoretical structure (multi-factor CFA / paths) instead of the auto
    # single-factor template. Columns are taken from those named in the spec.
    user_spec = cfg.get("model_spec") or cfg.get("sem_spec")
    if user_spec:
        used = [
            c for c in df.columns
            if re.search(rf"(?<![\w.]){re.escape(str(c))}(?![\w.])", user_spec)
        ]
        spec = user_spec
    else:
        used = indicators[:8]
        spec = "F =~ " + " + ".join(used)
    if not user_spec and len(indicators) < 3:
        summary.append("SEM 失败：需要 ≥3 个连续指标变量（单因子模型识别要求）。")
    elif user_spec and len(used) < 2:
        summary.append("SEM 失败：config model_spec 中未匹配到 ≥2 个数据列名。")
    else:
        import pandas as pd

        from researchforge.executor import rbridge

        inds = used
        sub = df[inds].dropna()
        # prefer lavaan (R, gold standard — also gives SRMR) when available;
        # fall back to pure-Python semopy so the analysis runs anywhere.
        # Only use the R backend with identifier-safe column names: names go
        # into the R model string, so a name with quotes/commas could break
        # parsing or inject R — semopy takes the names as data, no eval.
        names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in inds)
        # the spec is interpolated into cfa("...") as an R string literal; a stray
        # double-quote/backslash would break out, so a custom spec with those is
        # routed to semopy instead (takes the spec as a Python string, no R eval).
        spec_safe = '"' not in spec and "\\" not in spec
        result = None
        if names_safe and spec_safe and rbridge.r_available() and rbridge.r_package_available("lavaan"):
            csv = d / "_sem_input.csv"
            sub.to_csv(csv, index=False)
            try:
                result = _sem_via_lavaan(csv, spec)
            except Exception:
                result = None
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass
        if result is None:
            try:
                result = _sem_via_semopy(sub, spec)
            except Exception as err:
                summary.append(f"SEM 拟合失败：{err}")
        if result is not None:
            load = result["loadings"]
            fit = result["fit"]
            (d / "summary.txt").write_text(result["summary"], encoding="utf-8")
            files.append("summary.txt")
            load.to_csv(d / "loadings.csv", index=False, encoding="utf-8")
            files.append("loadings.csv")
            pd.DataFrame([fit]).to_csv(d / "fit_indices.csv", index=False, encoding="utf-8")
            files.append("fit_indices.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 3.2))
                _ylab = (
                    load["indicator"].astype(str) + " ← " + load["factor"].astype(str)
                    if "factor" in load.columns and load["factor"].nunique() > 1
                    else load["indicator"].astype(str)
                )
                ax.barh(_ylab, load["std_loading"], color="#4C72B0")
                ax.set_xlabel("standardised loading")
                ax.set_title("SEM measurement loadings")
                fig.tight_layout()
                fig.savefig(d / "loadings.png", dpi=150)
                plt.close(fig)
                files.append("loadings.png")
            except Exception:
                pass
            cfi, tli, rmsea = fit["cfi"], fit["tli"], fit["rmsea"]
            chi2, dof, srmr = fit["chi2"], fit["dof"], fit.get("srmr", float("nan"))
            for kk, vv in (("cfi", cfi), ("tli", tli), ("rmsea", rmsea), ("chi2", chi2), ("dof", dof)):
                estimates[kk] = round(vv, 4)
            if dof <= 0:
                # 3 indicators -> just-identified (df=0): CFI/RMSEA perfect by
                # construction, say nothing about fit (Opus double-review catch).
                verdict = "恰好识别(df=0)，拟合指数无意义(CFI/RMSEA 必完美)；需 ≥4 指标才能评估拟合"
            elif cfi >= 0.95 and rmsea <= 0.06:
                verdict = "拟合良好"
            else:
                verdict = "拟合一般/欠佳"
            srmr_txt = f" SRMR={srmr:.3f}" if srmr == srmr else ""  # NaN-safe
            _n_factors = len(set(_sem_latents(spec))) or 1
            _model_desc = (
                f"自定义模型（{_n_factors} 因子，按 config model_spec）"
                if user_spec
                else "单因子 CFA"
            )
            _tail = (
                "" if user_spec
                else "（此为探索性模板；可用 config={\"model_spec\": \"lavaan语法\"} 按理论结构改写后重跑）"
            )
            summary.append(
                f"{entry.method} 完成（后端：{result['backend']}）：{_model_desc} over "
                f"{len(inds)} 个指标（df={dof:.0f}）；CFI={cfi:.3f} TLI={tli:.3f} "
                f"RMSEA={rmsea:.3f}{srmr_txt} → {verdict}" + _tail
            )
            code += [
                "# SEM single-factor CFA — prefers R/lavaan, falls back to semopy",
                f'spec = "{spec}"',
                "# lavaan: cfa(spec, data=df, std.lv=TRUE); semopy: semopy.Model(spec).fit(df)",
            ]

