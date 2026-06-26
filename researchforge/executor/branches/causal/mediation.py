"""Causal family branch handler: mediation (split from causal.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("mediation")
def _branch_mediation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    y_col = cont[0] if cont else None
    cand = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in _excl | {y_col}
    ]
    if y_col is None or len(cand) < 2:
        summary.append("中介分析失败：需要连续结果变量 Y + ≥2 个变量（自变量 X、中介 M）。")
    else:
        x_col, m_col = cand[0], cand[1]  # default by column order; X→M→Y assumption
        try:
            import statsmodels.api as sm
            from statsmodels.stats.mediation import Mediation

            sub = df[[y_col, x_col, m_col]].dropna().rename(
                columns={y_col: "_y", x_col: "_x", m_col: "_m"}
            )
            om = sm.OLS.from_formula("_y ~ _x + _m", sub)  # outcome: Y ~ X + M
            mm = sm.OLS.from_formula("_m ~ _x", sub)  # mediator: M ~ X
            med = Mediation(om, mm, "_x", "_m").fit(n_rep=1000)
            s = med.summary()
            s.to_csv(d / "mediation_summary.csv", encoding="utf-8")
            files.append("mediation_summary.csv")

            def _row(label):
                return s.loc[label] if label in s.index else None

            acme = _row("ACME (average)")
            ade = _row("ADE (average)")
            tot = _row("Total effect")
            pm = _row("Prop. mediated (average)")
            indirect = float(acme["Estimate"])
            direct = float(ade["Estimate"])
            total = float(tot["Estimate"])
            prop = float(pm["Estimate"]) if pm is not None else float("nan")
            acme_p = float(acme["P-value"])
            estimates["indirect_effect_ACME"] = round(indirect, 4)
            estimates["direct_effect_ADE"] = round(direct, 4)
            estimates["total_effect"] = round(total, 4)
            estimates["prop_mediated"] = round(prop, 4)
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                labels = ["indirect (ACME)", "direct (ADE)", "total"]
                est = [indirect, direct, total]
                lo = [float(acme["Lower CI bound"]), float(ade["Lower CI bound"]), float(tot["Lower CI bound"])]
                hi = [float(acme["Upper CI bound"]), float(ade["Upper CI bound"]), float(tot["Upper CI bound"])]
                err = [[e - l for e, l in zip(est, lo)], [h - e for e, h in zip(est, hi)]]
                fig, ax = plt.subplots(figsize=(5.5, 3.2))
                ax.errorbar(est, range(3), xerr=err, fmt="o", capsize=4)
                ax.axvline(0, color="grey", ls="--")
                ax.set_yticks(range(3))
                ax.set_yticklabels(labels)
                ax.set_xlabel("effect (95% CI)")
                ax.set_title(f"Mediation {x_col} → {m_col} → {y_col}")
                fig.tight_layout()
                fig.savefig(d / "mediation_effects.png", dpi=150)
                plt.close(fig)
                files.append("mediation_effects.png")
            except Exception:
                pass
            verdict = "存在显著中介" if acme_p < 0.05 else "中介效应不显著"
            # prop. mediated is meaningless under suppression (opposite signs) or
            # near-zero total effect — flag rather than print a misleading % (Opus catch).
            suppression = abs(total) < 0.05 or (direct * indirect < 0)
            prop_txt = "不稳定（抑制效应/总效应近零，比例无意义）" if suppression else f"{prop:.1%}"
            summary.append(
                f"{entry.method} 完成：路径 {x_col} → {m_col} → {y_col}；"
                f"间接效应 ACME={indirect:.4f}（p={acme_p:.3g}，{verdict}），"
                f"直接效应 ADE={direct:.4f}，总效应={total:.4f}，中介比例={prop_txt}（Monte-Carlo CI）。"
                "⚠ X/M/Y 按列序默认（首连续=Y，其后=X、M），且 **X↔M 不对称**——交换二者是不同模型、"
                "列序只是选了一个假设而非事实，请核对你的理论路径；中介推断需 no-unmeasured-confounding 假定（非纯相关即因果）。"
            )
            code += [
                "from statsmodels.stats.mediation import Mediation",
                f"# OLS('{y_col}~{x_col}+{m_col}') + OLS('{m_col}~{x_col}'); Mediation(...).fit(n_rep=1000)",
            ]
        except Exception as err:
            summary.append(f"中介分析失败：{err}")
