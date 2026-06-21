"""Branch handlers for the configurational family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _cna_via_r,
    _csqca_via_r,
    _fsqca_via_r,
    _nca_ceiling,
    _nca_plot,
    _panel_qca_via_r,
    _qca_anchors,
    _qca_incl_cut,
    _qca_necessity_via_r,
)


@register("cna")
def _branch_cna(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    # CNA's factors ARE binary/continuous conditions — do NOT drop binary
    # treatment_candidates (they're exactly the configurational factors we need).
    _excl = {fp.unit_col, fp.time_col}
    binc = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
    contc = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    factors = (binc + contc)[:8]
    fuzzy = bool(contc)  # any continuous factor -> fuzzy-calibrate; else crisp 0/1
    outcome = cfg["outcome"] if cfg.get("outcome") in factors else None
    con = _qca_incl_cut({"incl_cut": cfg.get("con")} if cfg.get("con") else {}, 0.8)
    cov = _qca_incl_cut({"incl_cut": cfg.get("cov")} if cfg.get("cov") else {}, 0.8)
    anchors = _qca_anchors(cfg)
    names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in factors)
    if len(factors) < 3:
        summary.append("CNA 失败：需要 ≥3 个因子（二值或连续条件/结果列）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("cna")):
        summary.append("CNA 需要 R 的 cna 包（未检测到）。安装：install.packages('cna')；或用 fsqca（单结果）。")
    elif not names_safe:
        summary.append("CNA 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        import pandas as pd

        sub = df[factors].dropna()
        csv = d / "_cna_input.csv"
        sub.to_csv(csv, index=False)
        try:
            asf, n_csf = _cna_via_r(csv, factors, outcome, con, cov, anchors, fuzzy)
            asf = asf.sort_values(["consistency", "coverage"], ascending=False).reset_index(drop=True)
            asf.to_csv(d / "cna_solutions.csv", index=False, encoding="utf-8")
            files.append("cna_solutions.csv")
            cal_txt = (
                f"模糊校准(分位锚点 {anchors[0]}/{anchors[1]}/{anchors[2]})"
                if fuzzy else "清晰集(0/1 直接用)"
            )
            outs = sorted(asf["outcome"].unique().tolist())
            top = asf.iloc[0]
            estimates["n_solutions"] = float(len(asf))
            estimates["n_outcomes"] = float(len(outs))
            estimates["max_consistency"] = round(float(asf["consistency"].max()), 4)
            estimates["n_complex_structures"] = float(n_csf)
            (d / "cna_solutions.txt").write_text(
                f"巧合分析 CNA（R/cna，{cal_txt}，con≥{con}，cov≥{cov}）\n"
                f"发现 {len(asf)} 个原子解(asf)，涉及结果 {outs}；复杂结构(csf) {n_csf} 个。\n"
                "记号：* =与(AND)，+ =或(OR)，<-> 左侧为右侧结果的(配置性)原因；"
                "con=一致性(充分性)，cov=覆盖率(必要性)。\n"
                "CNA 不预设单一结果，可揭示多结果因果链；与 QCA 互补。\n\n"
                + asf.to_string(index=False),
                encoding="utf-8",
            )
            files.append("cna_solutions.txt")
            summary.append(
                f"{entry.method} 完成（R/cna，{cal_txt}）：{len(asf)} 个原子解、"
                f"结果变量 {outs}、复杂结构 {n_csf} 个；最强解 [{top['condition']}]"
                f"（con={top['consistency']:.3f}, cov={top['coverage']:.3f}）。"
                "⚠ 配置性因果≠净效应；解依赖 con/cov 阈值与校准锚点；有限多样性下慎读。"
            )
            code += [
                "library(cna)  # 巧合分析(多结果配置性因果)",
                f"# cna(d, type='{'fs' if fuzzy else 'cs'}', con={con}, cov={cov}); asf()/csf()",
            ]
        except Exception as err:
            summary.append(f"CNA 失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("csqca")
def _branch_csqca(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cont[0] if cont else None
    conditions = cont[1:6]
    names_safe = outcome is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
    )
    if outcome is None or len(conditions) < 2:
        summary.append("csQCA 失败：需要 1 个结果变量 + ≥2 个条件变量（均连续，按中位数二分）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("QCA")):
        summary.append(
            "csQCA 需要 R 的 QCA 包（未检测到）。安装：install.packages('QCA')；"
            "或先用 NCA（必要条件分析，纯 Python，无需 R）。"
        )
    elif not names_safe:
        summary.append("csQCA 失败：列名需为标识符式（字母/数字/. _）。")
    else:
        sub = df[[outcome, *conditions]].dropna()
        csv = d / "_qca_input.csv"
        sub.to_csv(csv, index=False)
        try:
            _ic = _qca_incl_cut(cfg, 0.8)
            sol_str, tab = _csqca_via_r(csv, outcome, conditions, incl_cut=_ic)
            tab.to_csv(d / "csqca_solution.csv", index=False, encoding="utf-8")
            files.append("csqca_solution.csv")
            (d / "solution.txt").write_text(
                f"清晰集 QCA 充分性解（complex solution） → {outcome}:\n  {sol_str}\n\n"
                f"清晰校准(连续条件列按中位数二分为 0/1；⚠ 当前仅自动选连续列,二值列暂未自动纳为条件),incl.cut={_ic}；* =AND, + =OR\n"
                "说明：① complex 解(不纳入反事实)；② 中位数二分丢失信息——连续条件通常 fsQCA 更优,"
                "且偏态/离散数据的中位数二分可能极不均衡(近恒值)；③ 充分性≠因果。\n\n"
                + tab.to_string(index=False),
                encoding="utf-8",
            )
            files.append("solution.txt")
            estimates["n_configurations"] = float(len(tab))
            estimates["min_consistency"] = round(float(tab["consistency"].min()), 4)
            estimates["total_unique_coverage"] = round(float(tab["unique_coverage"].sum()), 4)
            summary.append(
                f"{entry.method} 完成（R/QCA，complex 解）：充分配置 [{sol_str}] → {outcome}；"
                f"{len(tab)} 个配置，一致性 {tab['consistency'].min():.3f}–{tab['consistency'].max():.3f}"
                "（* =AND, + =OR；连续条件按中位数二分=信息损失,连续数据建议改用 fsQCA；充分性≠因果）"
                + ("（incl.cut 按 config 指定）" if cfg.get("incl_cut") else f"（incl.cut={_ic} 可配）")
            )
            code += [
                "library(QCA)  # 清晰集 QCA: 中位数二分 -> 真值表 -> 布尔最小化",
                f'# truthTable(outcome="{outcome}", conditions={conditions}); minimize(incl.cut={_ic})',
            ]
        except Exception as err:
            summary.append(f"csQCA 失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("fsqca")
def _branch_fsqca(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cont[0] if cont else None
    conditions = cont[1:6]  # outcome=cont[0]; up to 5 conditions (truth table 2^k)
    names_safe = outcome is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
    )
    if outcome is None or len(conditions) < 2:
        summary.append("fsQCA 失败：需要 1 个结果变量 + ≥2 个条件变量（均连续）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("QCA")):
        summary.append(
            "fsQCA 需要 R 的 QCA 包（未检测到）。安装：在 R 里 install.packages('QCA')；"
            "或先用 NCA（必要条件分析，纯 Python，无需 R）。"
        )
    elif not names_safe:
        summary.append("fsQCA 失败：列名需为标识符式（字母/数字/. _），R 后端要求。")
    else:
        sub = df[[outcome, *conditions]].dropna()
        csv = d / "_qca_input.csv"
        sub.to_csv(csv, index=False)
        try:
            _anch = _qca_anchors(cfg)
            _ic = _qca_incl_cut(cfg, 0.8)
            sol_str, tab = _fsqca_via_r(csv, outcome, conditions, anchors=_anch, incl_cut=_ic)
            tab.to_csv(d / "fsqca_solution.csv", index=False, encoding="utf-8")
            files.append("fsqca_solution.csv")
            (d / "solution.txt").write_text(
                f"充分性解（complex solution，sufficient configurations） → {outcome}:\n"
                f"  {sol_str}\n\n"
                f"直接校准(百分位锚点 {_anch[0]}/{_anch[1]}/{_anch[2]})，incl.cut={_ic}；"
                "* = 逻辑与（AND）, + = 或（OR）\n"
                "说明：① 这是 complex 解（不纳入反事实/remainders，最保守）；"
                "② crossover 锚点取中位数是机械设定，偏态数据会失真，请按理论设锚点；"
                "③ fsQCA 显示集合关系上的充分性，不等于因果证明。\n\n"
                + tab.to_string(index=False),
                encoding="utf-8",
            )
            files.append("solution.txt")
            estimates["n_configurations"] = float(len(tab))
            estimates["min_consistency"] = round(float(tab["consistency"].min()), 4)
            estimates["total_unique_coverage"] = round(float(tab["unique_coverage"].sum()), 4)
            _anch_note = (
                "（锚点/incl.cut 按 config 指定）"
                if (cfg.get("anchors") or cfg.get("incl_cut"))
                else f"（锚点 {_anch[0]}/{_anch[1]}/{_anch[2]}、incl.cut={_ic} 为机械起点，"
                "可用 config anchors/incl_cut 按理论设定）"
            )
            summary.append(
                f"{entry.method} 完成（R/QCA，complex 解）：充分配置 [{sol_str}] → {outcome}；"
                f"{len(tab)} 个配置，一致性 {tab['consistency'].min():.3f}–{tab['consistency'].max():.3f}"
                "（* =AND, + =OR；充分性≠因果证明）" + _anch_note
            )
            code += [
                "library(QCA)  # 直接校准 -> 真值表 -> 布尔最小化",
                f'# calibrate({[outcome, *conditions]}); truthTable(outcome="{outcome}"); minimize(incl.cut={_ic})',
            ]
        except Exception as err:
            summary.append(f"fsQCA 失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("nca")
def _branch_nca(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cont[0] if cont else None
    if outcome is None or len(cont) < 2:
        summary.append("NCA 失败：需要连续结果变量 + ≥1 个连续条件变量。")
    else:
        import pandas as pd

        predictors = cont[1:6]  # outcome is cont[0]; up to 5 conditions
        sub = df[[outcome] + predictors].dropna()
        if len(sub) < 20:
            summary.append("NCA 失败：有效样本不足（<20）。")
        else:
            y = sub[outcome].to_numpy(dtype=float)

            def _bucket(dv: float) -> str:
                if dv < 0.1:
                    return "negligible"
                if dv < 0.3:
                    return "medium"
                if dv < 0.5:
                    return "large"
                return "very large"

            ceilings = {}
            rows = []
            for p in predictors:
                x = sub[p].to_numpy(dtype=float)
                dv, xs, cmax = _nca_ceiling(x, y)  # `d` is the output dir — don't shadow
                ceilings[p] = (xs, cmax, dv)
                rows.append((p, round(dv, 4), _bucket(dv)))
                estimates[p] = round(dv, 4)

            tab = pd.DataFrame(rows, columns=["condition", "effect_size_d", "necessity"])
            tab.to_csv(d / "nca_effect_sizes.csv", index=False, encoding="utf-8")
            files.append("nca_effect_sizes.csv")
            _nca_plot(sub, outcome, predictors, ceilings, d / "nca_ceiling.png")
            if (d / "nca_ceiling.png").exists():
                files.append("nca_ceiling.png")

            strong = [r for r in rows if r[1] >= 0.1]
            top = max(rows, key=lambda r: r[1])
            summary.append(
                f"{entry.method} 完成：结果 {outcome}，{len(predictors)} 个条件；"
                f"最强必要条件 {top[0]}（d={top[1]}，{top[2]}）；"
                f"{len(strong)} 个达到有意义阈值 d≥0.1（d=空白区/总域面积，CE-FDH 天花板）"
            )
            code += [
                "import numpy as np  # NCA (Dul 2016), CE-FDH ceiling",
                "# c(x)=max{y: x_i<=x}; d = empty_zone_area / scope_area per condition",
            ]



@register("panel_qca")
def _branch_panel_qca(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    unit, time = fp.unit_col, fp.time_col
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {unit, time}]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced = [c for c in (cfg.get("predictors") or cfg.get("conditions") or []) if c in cont and c != outcome]
    conditions = forced[:5] if forced else [c for c in cont if c != outcome][:5]
    anchors = _qca_anchors(cfg)
    incl_cut = _qca_incl_cut(cfg, 0.8)
    names_safe = outcome is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
    )
    if not (unit and time):
        summary.append("面板 QCA 失败：需要面板数据（单位列 + 时间列）。")
    elif outcome is None or len(conditions) < 2:
        summary.append("面板 QCA 失败：需要 1 个结果 + ≥2 个条件（均连续，将模糊校准）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("SetMethods") and rbridge.r_package_available("QCA")):
        summary.append("面板 QCA 需要 R 的 SetMethods + QCA 包（未检测到）。安装：install.packages(c('QCA','SetMethods'))；或用 fsqca（截面）。")
    elif not names_safe:
        summary.append("面板 QCA 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        import pandas as pd

        sub = df[[outcome, *conditions, unit]].dropna()
        csv = d / "_pqca_input.csv"
        sub.to_csv(csv, index=False)
        try:
            sol_str, terms = _panel_qca_via_r(csv, outcome, conditions, unit, anchors, incl_cut)
            terms = terms.sort_values("pooled_consistency", ascending=False).reset_index(drop=True)
            terms.to_csv(d / "panel_qca_terms.csv", index=False, encoding="utf-8")
            files.append("panel_qca_terms.csv")
            # large between/within distance => the configuration is NOT uniform
            # across units / over time (clustered heterogeneity)
            het = terms[(terms["dist_between"] > 0.2) | (terms["dist_within"] > 0.2)]["term"].tolist()
            estimates["n_terms"] = float(len(terms))
            estimates["max_pooled_consistency"] = round(float(terms["pooled_consistency"].max()), 4)
            estimates["max_dist_between"] = round(float(terms["dist_between"].max()), 4)
            estimates["max_dist_within"] = round(float(terms["dist_within"].max()), 4)
            het_txt = (
                f"⚠ 跨单位/时间不稳定项：{het}（between/within→pooled 距离>0.2，配置在子总体间不一致）"
                if het else "各项 between/within 距离均小（配置在单位/时间间较稳定）"
            )
            (d / "panel_qca.txt").write_text(
                f"面板/聚类 fsQCA（SetMethods cluster，分位锚点 {anchors[0]}/{anchors[1]}/{anchors[2]}，incl.cut={incl_cut}）\n"
                f"汇总(pooled)充分性解 → {outcome}:  {sol_str}\n"
                f"按单位 {unit} 聚类，分解每个解项的 一致性：汇总(POCOS) vs 组间(between) vs 组内(within)；"
                "dBP/dWP=组间/组内到汇总的距离(越大越不一致)。\n"
                f"{het_txt}\n\n" + terms.to_string(index=False),
                encoding="utf-8",
            )
            files.append("panel_qca.txt")
            summary.append(
                f"{entry.method} 完成（R/SetMethods）：汇总解 [{sol_str}] → {outcome}（按 {unit} 聚类）；"
                f"{len(terms)} 个解项，最高汇总一致性 {terms['pooled_consistency'].max():.3f}，"
                f"最大组间距离 {terms['dist_between'].max():.3f}、组内 {terms['dist_within'].max():.3f}。{het_txt}。"
                "⚠ 配置性充分≠因果；距离大说明 pooled 解掩盖了子总体差异。"
            )
            code += [
                "library(SetMethods)  # 面板/聚类 fsQCA",
                f"# minimize(...) -> cluster(results=sol, unit_id, cluster_id='{unit}'); POCOS/dBP/dWP",
            ]
        except Exception as err:
            summary.append(f"面板 QCA 失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("qca_necessity")
def _branch_qca_necessity(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    outcome = cont[0] if cont else None
    conditions = cont[1:6]
    names_safe = outcome is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
    )
    if outcome is None or len(conditions) < 2:
        summary.append("QCA 必要性分析失败：需要 1 个结果变量 + ≥2 个条件变量（均连续）。")
    elif not (rbridge.r_available() and rbridge.r_package_available("QCA")):
        summary.append(
            "QCA 必要性分析需要 R 的 QCA 包（未检测到）。安装：install.packages('QCA')；"
            "或先用 NCA（必要条件分析，纯 Python，无需 R）。"
        )
    elif not names_safe:
        summary.append("QCA 必要性分析失败：列名需为标识符式（字母/数字/. _）。")
    else:
        sub = df[[outcome, *conditions]].dropna()
        csv = d / "_qca_input.csv"
        sub.to_csv(csv, index=False)
        try:
            _anch = _qca_anchors(cfg)
            _ic = _qca_incl_cut(cfg, 0.9)
            tab = _qca_necessity_via_r(csv, outcome, conditions, anchors=_anch, incl_cut=_ic)
            tab = tab.sort_values("consistency_inclN", ascending=False).reset_index(drop=True)
            tab.to_csv(d / "necessity.csv", index=False, encoding="utf-8")
            files.append("necessity.csv")
            (d / "necessity.txt").write_text(
                f"必要性分析（superSubset） → {outcome}（fuzzy 校准 {_anch[0]}/{_anch[1]}/{_anch[2]}，incl.cut={_ic}）:\n"
                "inclN=必要性一致性；RoN=必要性相关度(越高越非琐碎)；covN=覆盖度；"
                "~X=非 X，+ =或。\n注意：inclN 高但 RoN 低 = 琐碎必要（条件几乎恒为高）；"
                "必要性≠因果。\n\n" + tab.to_string(index=False),
                encoding="utf-8",
            )
            files.append("necessity.txt")
            top = tab.iloc[0]
            estimates["max_inclN"] = round(float(tab["consistency_inclN"].max()), 4)
            estimates["n_necessary_expr"] = float(len(tab))
            summary.append(
                f"{entry.method} 完成（R/QCA）：最强必要项 [{top['expression']}]"
                f"（inclN={top['consistency_inclN']:.3f}, RoN={top['RoN']:.3f}, "
                f"covN={top['coverage_covN']:.3f}）；共 {len(tab)} 项"
                "（RoN 低=琐碎必要；必要性≠因果证明）"
                + ("（锚点/incl.cut 按 config 指定）" if (cfg.get("anchors") or cfg.get("incl_cut")) else f"（incl.cut={_ic} 可配 anchors/incl_cut）")
            )
            code += [
                f"library(QCA)  # 必要性: 模糊校准 -> superSubset(incl.cut={_ic}, cov.cut=0.5)",
                f'# superSubset(cal, outcome="{outcome}", conditions={conditions})',
            ]
        except Exception as err:
            summary.append(f"QCA 必要性分析失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass

