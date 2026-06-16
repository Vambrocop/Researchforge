"""Branch handlers for the meta family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _meta_regression_via_r,
    _meta_via_r,
)


@register("meta_analysis")
def _branch_meta_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import pandas as pd

    from researchforge.executor import rbridge

    low = {c.name.lower(): c.name for c in fp.columns}

    def _pick(*names):
        return next((low[n] for n in names if n in low), None)

    # role resolution: config overrides, else detect by column name. Three
    # input formats — pre-computed effect sizes, raw two-group means, 2x2 counts.
    measure = str(cfg.get("measure") or "").upper()
    method = str(cfg.get("method") or "REML").upper()  # REML/DL random, FE fixed
    study_col = cfg.get("study") or _pick(
        "study", "study_id", "studyid", "label", "author", "trial", "name", "id", "source"
    )
    yi = cfg.get("effect") or _pick("yi", "effect", "es", "effect_size", "smd", "logor", "d", "g", "lnrr")
    vi = cfg.get("variance") or _pick("vi", "var", "variance", "v", "samp_var")
    sei = cfg.get("se") or _pick("sei", "se", "std_err", "stderr", "se_effect")
    m1, sd1, n1 = _pick("m1", "m1i", "mean1", "mean_t", "mt"), _pick("sd1", "sd1i", "sd_t", "sdt"), _pick("n1", "n1i", "nt", "n_t")
    m2, sd2, n2 = _pick("m2", "m2i", "mean2", "mean_c", "mc"), _pick("sd2", "sd2i", "sd_c", "sdc"), _pick("n2", "n2i", "nc", "n_c")
    ai, bi, ci_, di = _pick("ai", "events1", "a"), _pick("bi", "b"), _pick("ci", "events2", "c"), _pick("di", "d")

    roles, used = {}, []
    if measure in ("", "GEN") and yi and (vi or sei):
        measure = "GEN"
        roles = {"yi": yi, "vi": vi, "sei": sei}
        used = [yi, vi or sei]
    elif measure in ("", "SMD", "MD") and all([m1, sd1, n1, m2, sd2, n2]):
        measure = measure if measure in ("SMD", "MD") else "SMD"
        roles = {"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2}
        used = [m1, sd1, n1, m2, sd2, n2]
    elif measure in ("", "OR", "RR", "RD") and all([ai, bi, ci_, di]):
        measure = measure if measure in ("OR", "RR", "RD") else "OR"
        roles = {"ai": ai, "bi": bi, "ci": ci_, "di": di}
        used = [ai, bi, ci_, di]

    cols_all = [c for c in [*used, study_col] if c]
    names_safe = all(('"' not in c and "]" not in c and "\\" not in c) for c in cols_all)
    if not roles:
        summary.append(
            "Meta 分析失败：未识别到效应量数据。需以下任一格式（列名可用 config 指定）："
            "① 预算效应量 yi + 方差 vi（或标准误 sei）；"
            "② 两组原始均值 m1,sd1,n1,m2,sd2,n2（→标准化均差 SMD）；"
            "③ 2×2 计数 ai,bi,ci,di（→比值比 OR）。详见 docs/meta-analysis.md。"
        )
    elif not (rbridge.r_available() and rbridge.r_package_available("metafor")):
        summary.append(
            "Meta 分析需要 R 的 metafor 包（未检测到）。安装：install.packages('metafor')。"
        )
    elif not names_safe:
        summary.append("Meta 分析失败：相关列名含特殊字符（\" ] \\），请重命名。")
    else:
        sub = df[cols_all].dropna()
        if len(sub) < 2:
            summary.append("Meta 分析失败：有效研究数 <2，无法合并。")
        else:
            csv = d / "_meta_input.csv"
            sub.to_csv(csv, index=False)
            try:
                meta, study = _meta_via_r(
                    csv, measure=measure, roles=roles, study_col=study_col,
                    method=method, forest_png=d / "forest.png", funnel_png=d / "funnel.png",
                )
                study.to_csv(d / "study_effects.csv", index=False, encoding="utf-8")
                files.append("study_effects.csv")
                for png in ("forest.png", "funnel.png"):
                    if (d / png).exists():
                        files.append(png)
                import math

                est, lb, ub = meta["estimate"], meta["ci_lb"], meta["ci_ub"]
                i2, tau2, k = meta["I2"], meta["tau2"], int(meta["k"])
                qp, pval = meta["Qp"], meta["pval"]
                # OR/RR are pooled on the log scale -> exp() the WHOLE interval for
                # display (point AND CI), not just the point estimate.
                log_scale = measure in ("OR", "RR")
                if log_scale:
                    de, dlb, dub = math.exp(est), math.exp(lb), math.exp(ub)
                    shown_est = f"{de:.3f}（{measure}, 由 log 尺度还原）"
                    ci_str = f"[{dlb:.3f}, {dub:.3f}]"
                    estimates["pooled_effect"] = round(de, 4)
                    estimates["ci_lb"] = round(dlb, 4)
                    estimates["ci_ub"] = round(dub, 4)
                    estimates["pooled_log_effect"] = round(est, 4)
                else:
                    shown_est = f"{est:.3f}"
                    ci_str = f"[{lb:.3f}, {ub:.3f}]"
                    estimates["pooled_effect"] = round(est, 4)
                    estimates["ci_lb"] = round(lb, 4)
                    estimates["ci_ub"] = round(ub, 4)
                estimates["I2_percent"] = round(i2, 2)
                estimates["tau2"] = round(tau2, 4)
                estimates["k_studies"] = float(k)
                het = "高" if i2 >= 75 else ("中" if i2 >= 50 else ("低" if i2 >= 25 else "极低"))
                is_fe = method == "FE"
                eg_txt = ""
                if "egger_p" in meta:
                    estimates["egger_p"] = round(meta["egger_p"], 4)
                    if k < 10:
                        # Egger / funnel-asymmetry tests are underpowered at k<10
                        # (Cochrane) — report but flag, don't over-reassure (Opus catch).
                        eg_txt = f"；Egger 检验 p={meta['egger_p']:.3g}（⚠ k<10，偏倚检验功效不足、不可靠）"
                    else:
                        eg_txt = (
                            f"；Egger 检验 p={meta['egger_p']:.3g}"
                            f"（{'⚠ 漏斗图不对称、可能有发表偏倚' if meta['egger_p'] < 0.05 else '未见明显不对称'}）"
                        )
                # under fixed-effect, tau2/I2 are 0 by assumption (not estimated) —
                # the Q test is the meaningful heterogeneity signal (Opus catch).
                het_txt = (
                    f"（固定效应假定同质，I²/τ² 不估计；看 Q 检验 p={qp:.3g}）"
                    if is_fe
                    else f"异质性 I²={i2:.1f}%（{het}）、τ²={tau2:.4f}"
                )
                mlabel = {"FE": "固定效应", "DL": "随机效应(DL)"}.get(method, "随机效应(REML)")
                (d / "meta_summary.txt").write_text(
                    f"Meta 分析（metafor，{mlabel}，measure={measure}，k={k} 研究）\n"
                    f"合并效应 = {shown_est}，95% CI {ci_str}，p={pval:.4g}\n"
                    f"异质性：I²={i2:.1f}%（{het}），τ²={tau2:.4f}，Q 检验 p={qp:.4g}\n"
                    f"{'森林图 forest.png、漏斗图 funnel.png' }\n"
                    "注：I²>50% 提示研究间异质性较大，合并需谨慎、宜探究调节变量；"
                    "随机效应不假定各研究共享同一真效应。\n",
                    encoding="utf-8",
                )
                files.append("meta_summary.txt")
                sig = "显著" if pval < 0.05 else "不显著"
                summary.append(
                    f"{entry.method} 完成（R/metafor，{mlabel}）：合并 {k} 项研究，"
                    f"合并效应={shown_est}，95% CI {ci_str}（{sig}，p={pval:.3g}）；"
                    f"{het_txt}{eg_txt}。"
                    "⚠ I²>50% 宜查调节变量；合并不能修正原始研究的偏倚。"
                )
                code += [
                    "library(metafor)  # 随机效应 meta 分析",
                    f"# escalc(measure='{measure}', ...) → rma(yi, vi, method='{method}'); forest()/funnel()/regtest()",
                ]
            except Exception as err:
                summary.append(f"Meta 分析失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass



@register("meta_regression")
def _branch_meta_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import pandas as pd

    from researchforge.executor import rbridge

    low = {c.name.lower(): c.name for c in fp.columns}

    def _pick(*names):
        return next((low[n] for n in names if n in low), None)

    measure = str(cfg.get("measure") or "").upper()
    method = str(cfg.get("method") or "REML").upper()
    study = cfg.get("study") or _pick("study", "study_id", "studyid", "label", "author", "trial", "name", "id", "source")
    yi = cfg.get("effect") or _pick("yi", "effect", "es", "effect_size", "smd", "logor", "d", "g", "lnrr")
    vi = cfg.get("variance") or _pick("vi", "var", "variance", "v", "samp_var")
    sei = cfg.get("se") or _pick("sei", "se", "std_err", "stderr", "se_effect")
    m1, sd1, n1 = _pick("m1", "m1i", "mean1", "mean_t", "mt"), _pick("sd1", "sd1i", "sd_t", "sdt"), _pick("n1", "n1i", "nt", "n_t")
    m2, sd2, n2 = _pick("m2", "m2i", "mean2", "mean_c", "mc"), _pick("sd2", "sd2i", "sd_c", "sdc"), _pick("n2", "n2i", "nc", "n_c")
    ai, bi, ci_, di = _pick("ai", "events1", "a"), _pick("bi", "b"), _pick("ci", "events2", "c"), _pick("di", "d")
    roles, es_cols = {}, []
    if measure in ("", "GEN") and yi and (vi or sei):
        measure, roles, es_cols = "GEN", {"yi": yi, "vi": vi, "sei": sei}, [yi, vi or sei]
    elif measure in ("", "SMD", "MD") and all([m1, sd1, n1, m2, sd2, n2]):
        measure = measure if measure in ("SMD", "MD") else "SMD"
        roles, es_cols = {"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2}, [m1, sd1, n1, m2, sd2, n2]
    elif measure in ("", "OR", "RR", "RD") and all([ai, bi, ci_, di]):
        measure = measure if measure in ("OR", "RR", "RD") else "OR"
        roles, es_cols = {"ai": ai, "bi": bi, "ci": ci_, "di": di}, [ai, bi, ci_, di]
    # moderators: config, else numeric/categorical columns not used as effect-size/study.
    # exclude ALL resolved role columns (consumed or not) so a leftover precision column
    # (e.g. an unused sei when vi was picked) can't sneak in as a moderator (Opus catch).
    used = {c for c in (yi, vi, sei, m1, sd1, n1, m2, sd2, n2, ai, bi, ci_, di, study) if c}
    forced_mods = [c for c in (cfg.get("moderators") or cfg.get("predictors") or []) if c in df.columns and c not in used]
    if forced_mods:
        moderators = forced_mods[:5]
    else:
        moderators = [
            c.name for c in fp.columns
            if c.kind in {"continuous", "count", "binary", "categorical"} and c.name not in used
            and c.name not in {fp.unit_col, fp.time_col}
        ][:4]
    # drop constant moderators — metafor silently drops them and would mislabel the
    # omnibus as a moderator test (Opus catch); filter in Python first.
    const_mods = [m for m in moderators if df[m].dropna().nunique() <= 1]
    moderators = [m for m in moderators if m not in const_mods]
    _num_kind = {c.name: c.kind for c in fp.columns}
    bubble_mod = next((m for m in moderators if _num_kind.get(m) in {"continuous", "count"}), None)
    cols_all = [c for c in [*es_cols, study, *moderators] if c]
    names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in moderators) if moderators else False
    if not roles:
        summary.append(
            "Meta 回归失败：未识别到效应量数据（需 yi+vi/sei，或两组 m/sd/n，或 2×2 ai/bi/ci/di）。详见 docs/meta-analysis.md。"
        )
    elif not moderators:
        summary.append("Meta 回归失败：需要 ≥1 个研究层调节变量（解释异质性的协变量）。用 config['moderators'] 指定。")
    elif not (rbridge.r_available() and rbridge.r_package_available("metafor")):
        summary.append("Meta 回归需要 R 的 metafor 包（未检测到）。安装：install.packages('metafor')。")
    elif not names_safe:
        summary.append("Meta 回归失败：调节变量列名需为标识符式（字母/数字/. _），R 公式要求。")
    else:
        sub = df[cols_all].dropna()
        if len(sub) < len(moderators) + 3:
            summary.append(f"Meta 回归失败：有效研究数 {len(sub)} 太少（需 > 调节变量数 {len(moderators)}+2）。")
        else:
            csv = d / "_metareg_input.csv"
            sub.to_csv(csv, index=False)
            try:
                coef, st = _meta_regression_via_r(
                    csv, measure=measure, roles=roles, moderators=moderators,
                    study=study, method=method, bubble_png=d / "bubble.png", bubble_mod=bubble_mod,
                )
                surviving = st.get("_surviving", [])
                if not surviving:
                    raise RuntimeError("所有调节变量被 metafor 剔除（常量/完全共线），无可解释的调节项")
                coef.to_csv(d / "meta_regression_coef.csv", index=False, encoding="utf-8")
                files.append("meta_regression_coef.csv")
                if (d / "bubble.png").exists():
                    files.append("bubble.png")
                qm, qmp = st["QM"], st["QMp"]
                r2 = float("nan") if method == "FE" else st.get("R2", float("nan"))
                i2r, k = st["I2_resid"], int(st["k"])
                # disclose moderators dropped as constant (Python) or collinear (metafor)
                drop_note = f"；⚠ 常量调节变量已剔除：{const_mods}" if const_mods else ""
                if len(surviving) < len(moderators):
                    drop_note += "；⚠ 部分调节变量因共线被 metafor 剔除（见系数表实际项）"
                estimates["QM_pvalue"] = round(qmp, 4)
                estimates["R2_percent"] = round(r2, 2) if r2 == r2 else float("nan")
                estimates["I2_residual"] = round(i2r, 2)
                estimates["k_studies"] = float(k)
                for _, r in coef.iterrows():
                    if r["term"] != "intrcpt":
                        estimates[f"beta_{r['term']}"] = round(float(r["estimate"]), 4)
                sig_mods = [r["term"] for _, r in coef.iterrows() if r["term"] != "intrcpt" and float(r["p_value"]) < 0.05]
                r2_txt = f"，调节变量解释了 {r2:.0f}% 的研究间异质性" if r2 == r2 else ""
                (d / "meta_regression.txt").write_text(
                    f"Meta 回归（metafor rma，{method}，k={k} 研究，measure={measure}）\n"
                    f"调节变量：{moderators}\n"
                    f"omnibus 检验 QM={qm:.3f}，p={qmp:.4g}"
                    f"（{'调节变量整体显著' if qmp < 0.05 else '调节变量整体不显著'}）\n"
                    f"残差异质性 I²={i2r:.1f}%{r2_txt}\n"
                    f"显著调节变量（p<0.05）：{sig_mods}\n"
                    "注：meta 回归是观察性的（研究层混杂、生态谬误风险）；"
                    "调节变量少、研究数少时易过拟合/假阳性（建议每 10 项研究 ≤1 个调节变量）。\n\n"
                    + coef.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("meta_regression.txt")
                summary.append(
                    f"{entry.method} 完成（R/metafor，{method}）：合并 {k} 项研究、调节变量 {moderators}；"
                    f"omnibus p={qmp:.3g}（{'整体显著' if qmp < 0.05 else '整体不显著'}）"
                    f"{r2_txt}；残差 I²={i2r:.1f}%；显著项 {sig_mods}。"
                    "⚠ 研究层观察性关联（非个体因果，慎防生态谬误）；研究少易过拟合。" + drop_note
                )
                code += [
                    "library(metafor)  # meta 回归（调节变量解释异质性）",
                    f"# rma(yi, vi, mods = ~ {' + '.join(moderators)}, method='{method}'); QM 检验 + R²",
                ]
            except Exception as err:
                summary.append(f"Meta 回归失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

