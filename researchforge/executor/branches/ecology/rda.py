"""rda — redundancy analysis (constrained ordination) via R vegan::rda."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


def _rda_via_r(csv_path, comm_cols: list[str], pred_cols: list[str]):
    """Redundancy analysis via R vegan::rda (constrained ordination) + anova.cca for
    global and per-axis/term significance. Community matrix constrained by
    environmental predictors. Returns a dict with variance partition, scores, and
    significance. Column names are identifier-guarded upstream. Raises so the caller
    can degrade honestly."""
    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    comm_r = ", ".join(f'"{c}"' for c in comm_cols)
    pred_r = " + ".join(pred_cols)
    rcode = (
        "suppressMessages(library(vegan))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"comm <- d[, c({comm_r})]\n"
        f"m <- rda(comm ~ {pred_r}, data = d)\n"
        "tot <- m$tot.chi\n"
        "constr <- if (is.null(m$CCA)) 0 else m$CCA$tot.chi\n"
        "unconstr <- if (is.null(m$CA)) 0 else m$CA$tot.chi\n"
        # adjusted R^2 (Ezekiel) — honest fit measure; raw constrained% inflates with #predictors.
        # NA when n is too small relative to constraints; emit -999 sentinel for that case.
        "r2 <- tryCatch(RsquareAdj(m)$adj.r.squared, error = function(e) NA)\n"
        "set.seed(0)\n"
        "ag <- anova(m, permutations = 999)\n"
        "set.seed(0)\n"
        "aax <- tryCatch(anova(m, by = 'axis', permutations = 999), error = function(e) NULL)\n"
        "set.seed(0)\n"
        "atm <- tryCatch(anova(m, by = 'terms', permutations = 999), error = function(e) NULL)\n"
        'cat("##VAR\\n")\n'
        'cat(sprintf("total|%.6f\\n", tot))\n'
        'cat(sprintf("constrained|%.6f\\n", constr))\n'
        'cat(sprintf("unconstrained|%.6f\\n", unconstr))\n'
        'cat(sprintf("r2adj|%.6f\\n", ifelse(is.na(r2), -999, r2)))\n'
        'cat("##GLOBAL\\n")\n'
        'cat(sprintf("F|%.6f\\np|%.6g\\n", ag$F[1], ag$"Pr(>F)"[1]))\n'
        'cat("##EIG\\n")\n'
        "eig <- if (is.null(m$CCA)) NULL else m$CCA$eig\n"
        "if (!is.null(eig)) for (i in seq_along(eig)) "
        'cat(sprintf("%s|%.6f\\n", names(eig)[i], eig[i]))\n'
        'cat("##AXIS\\n")\n'
        "if (!is.null(aax)) { nm <- rownames(aax); "
        "for (i in seq_len(nrow(aax)-1)) "
        'cat(sprintf("%s|%.6f|%.6g\\n", nm[i], aax$F[i], aax$"Pr(>F)"[i])) }\n'
        'cat("##TERMS\\n")\n'
        "if (!is.null(atm)) { nm <- rownames(atm); "
        "for (i in seq_len(nrow(atm)-1)) "
        'cat(sprintf("%s|%.6f|%.6g\\n", nm[i], atm$F[i], atm$"Pr(>F)"[i])) }\n'
        "nax <- if (is.null(eig)) 1 else min(2, max(1, length(eig)))\n"
        'cat("##SITES\\n")\n'
        "sc <- as.matrix(scores(m, display = 'sites', choices = 1:nax))\n"
        "for (i in seq_len(nrow(sc))) "
        'cat(sprintf("%d|%s\\n", i, paste(sprintf("%.6f", sc[i, ]), collapse="|")))\n'
        'cat("##BP\\n")\n'
        "bp <- tryCatch(as.matrix(scores(m, display = 'bp', choices = 1:nax)), "
        "error = function(e) NULL)\n"
        "if (!is.null(bp)) for (i in seq_len(nrow(bp))) "
        'cat(sprintf("%s|%s\\n", rownames(bp)[i], paste(sprintf("%.6f", bp[i, ]), collapse="|")))\n'
    )
    out = rbridge.run_r(rcode, timeout=300)
    parsed: dict = {
        "var": {},
        "global": {},
        "eig": [],
        "axis": [],
        "terms": [],
        "sites": [],
        "bp": [],
    }
    section = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("##"):
            section = s[2:]
            continue
        if "|" not in s:
            continue
        parts = s.split("|")
        if section == "VAR":
            parsed["var"][parts[0]] = float(parts[1])
        elif section == "GLOBAL":
            parsed["global"][parts[0]] = float(parts[1])
        elif section == "EIG":
            parsed["eig"].append((parts[0], float(parts[1])))
        elif section == "AXIS":
            parsed["axis"].append((parts[0], float(parts[1]), float(parts[2])))
        elif section == "TERMS":
            parsed["terms"].append((parts[0], float(parts[1]), float(parts[2])))
        elif section == "SITES":
            parsed["sites"].append([float(x) for x in parts[1:]])
        elif section == "BP":
            parsed["bp"].append((parts[0], [float(x) for x in parts[1:]]))
    if not parsed["var"]:
        raise RuntimeError("vegan::rda 未返回方差分解")
    return parsed


@register("rda")
def _branch_rda(ctx: Ctx) -> None:
    """Redundancy analysis (RDA) — constrained ordination via R vegan::rda. Community
    matrix constrained by environmental predictors; reports constrained vs
    unconstrained variance, global + axis/term significance (anova.cca), and a
    triplot. Optional R bridge with honest graceful degrade to nmds/permanova."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    import numpy as np
    import pandas as pd

    from researchforge.executor import rbridge

    _excl = {fp.unit_col, fp.time_col}
    auto_comm = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    auto_pred = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "geo"} and c.name not in _excl
    ]
    comm_cols = list(cfg.get("community") or cfg.get("species") or auto_comm)
    pred_cols = list(cfg.get("predictors") or cfg.get("env") or auto_pred)
    comm_cols = [c for c in comm_cols if c in df.columns]
    pred_cols = [c for c in pred_cols if c in df.columns and c not in comm_cols]

    # identifier guard before column names enter an R formula (injection / parse safety)
    _ident = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")
    bad = [c for c in comm_cols + pred_cols if not _ident.fullmatch(str(c))]

    if len(comm_cols) < 2 or len(pred_cols) < 1:
        summary.append(
            "RDA 跳过：需要 ≥2 个群落/物种列（计数）与 ≥1 个环境预测变量（连续）。"
            f"（自动检出 community={comm_cols[:3]}，predictors={pred_cols}；可用 config "
            "community/predictors 指定。）"
        )
        return

    sub = df[comm_cols + pred_cols].dropna()
    if len(sub) < max(6, len(pred_cols) + 2):
        summary.append(
            f"RDA 跳过：去缺失后有效样点 {len(sub)} 个，对 {len(pred_cols)} 个约束变量过少"
            "（需 n > 预测数 + 2）。"
        )
        return

    degrade_to = "可改用 nmds（无约束排序）或 permanova（分组成分检验）作为纯 Python 替代。"
    if bad:
        summary.append(
            f"RDA 跳过：列名 {bad} 含非法字符（R 公式守卫拦截）。请重命名为 "
            "字母/点开头、仅含字母数字._ 的标识符。" + degrade_to
        )
        return

    if not (rbridge.r_available() and rbridge.r_package_available("vegan")):
        summary.append(
            "⚠ RDA 需要 R 包 vegan（约束排序金标准），当前环境未检测到 R 或 vegan，"
            "已诚实跳过（装：install.packages('vegan')）。" + degrade_to
        )
        return

    _csv = d / "_rda_input.csv"
    try:
        sub.to_csv(_csv, index=False, encoding="utf-8")
        parsed = _rda_via_r(_csv, comm_cols, pred_cols)
    except Exception as err:
        summary.append(
            f"⚠ vegan::rda 运行失败（{err}），已跳过。" + degrade_to
        )
        return
    finally:
        try:
            _csv.unlink()
        except OSError:
            pass

    try:
        tot = parsed["var"].get("total", 0.0)
        constr = parsed["var"].get("constrained", 0.0)
        unconstr = parsed["var"].get("unconstrained", 0.0)
        pct_constr = 100.0 * constr / tot if tot > 0 else 0.0
        pct_unconstr = 100.0 * unconstr / tot if tot > 0 else 0.0
        r2adj_raw = parsed["var"].get("r2adj", -999.0)
        r2adj = r2adj_raw if r2adj_raw > -900.0 else float("nan")  # -999 sentinel = NA in R
        g_F = parsed["global"].get("F", float("nan"))
        g_p = parsed["global"].get("p", float("nan"))

        # variance / significance tables
        rows = [
            {"component": "constrained (RDA)", "variance": round(constr, 4),
             "pct_of_total": round(pct_constr, 2)},
            {"component": "unconstrained (PCA residual)", "variance": round(unconstr, 4),
             "pct_of_total": round(pct_unconstr, 2)},
        ]
        pd.DataFrame(rows).to_csv(
            d / "rda_variance.csv", index=False, encoding="utf-8"
        )
        files.append("rda_variance.csv")

        sig_rows = [{"test": "global", "F": round(g_F, 4), "p_value": round(g_p, 4)}]
        for nm, F, p in parsed["axis"]:
            sig_rows.append({"test": f"axis:{nm}", "F": round(F, 4), "p_value": round(p, 4)})
        for nm, F, p in parsed["terms"]:
            sig_rows.append({"test": f"term:{nm}", "F": round(F, 4), "p_value": round(p, 4)})
        pd.DataFrame(sig_rows).to_csv(
            d / "rda_significance.csv", index=False, encoding="utf-8"
        )
        files.append("rda_significance.csv")

        estimates["constrained_variance_pct"] = round(pct_constr, 2)
        estimates["unconstrained_variance_pct"] = round(pct_unconstr, 2)
        import math as _math

        if _math.isfinite(r2adj):
            estimates["adjusted_r_squared"] = round(float(r2adj), 4)
        estimates["global_F"] = round(float(g_F), 4)
        estimates["global_p"] = round(float(g_p), 4)
        n_sig_terms = sum(1 for _, _, p in parsed["terms"] if p < 0.05)
        estimates["n_significant_predictors"] = float(n_sig_terms)

        # triplot: site scores + predictor biplot arrows (first 2 constrained axes)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sites = np.asarray(parsed["sites"], dtype=float)
            if sites.ndim == 2 and sites.shape[1] >= 2 and len(sites) > 0:
                fig, ax = plt.subplots(figsize=(6, 5.5))
                ax.scatter(sites[:, 0], sites[:, 1], s=22, c="#4C72B0",
                           alpha=0.7, label="sites")
                bp2 = [(nm, vec) for nm, vec in parsed["bp"] if len(vec) >= 2]
                if bp2:
                    smax = float(np.abs(sites[:, :2]).max()) or 1.0
                    arr = np.array([vec[:2] for _, vec in bp2], dtype=float)
                    bmax = float(np.abs(arr).max()) or 1.0
                    scale = 0.85 * smax / bmax
                    for name, vec in bp2:
                        ax.arrow(0, 0, vec[0] * scale, vec[1] * scale,
                                 color="#C44E52", width=0.0, head_width=0.04 * smax,
                                 length_includes_head=True)
                        ax.text(vec[0] * scale * 1.08, vec[1] * scale * 1.08,
                                name, color="#C44E52", fontsize=8)
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                ax.axvline(0, color="grey", lw=0.6, ls="--")
                ax.set_xlabel("RDA1")
                ax.set_ylabel("RDA2")
                ax.set_title(f"RDA triplot (constrained={pct_constr:.1f}%)")
                fig.tight_layout()
                fig.savefig(d / "rda_triplot.png", dpi=150)
                plt.close(fig)
                files.append("rda_triplot.png")
        except Exception:
            pass

        sig = "显著" if (np.isfinite(g_p) and g_p < 0.05) else "不显著"
        r2adj_txt = (
            f"，调整 R²={r2adj:.3f}（Ezekiel 校正预测变量数后的诚实拟合）"
            if _math.isfinite(r2adj) else "（调整 R² 不可估：样点对约束变量过少）"
        )
        summary.append(
            f"{entry.method} 完成（R vegan::rda）：{len(comm_cols)} 物种 × {len(sub)} 样点，"
            f"受 {len(pred_cols)} 个环境变量约束。约束方差占比 {pct_constr:.1f}%、"
            f"非约束（残差）{pct_unconstr:.1f}%{r2adj_txt}；全局检验 F={g_F:.3f}，p={g_p:.3f}"
            f"（999 次置换，{sig}）；{n_sig_terms}/{len(pred_cols)} 个预测变量显著。"
            "⚠ RDA 是线性约束排序，假定物种对梯度线性响应——长环境梯度（单峰响应）应优先 CCA；"
            "⚠ 原始约束方差占比随预测变量数膨胀（过拟合风险），应以上面的**调整 R²**为准；"
            "⚠ 统计关联、非因果；预测变量按连续列自动选取，可用 config predictors 覆盖。"
        )
        code += [
            "library(vegan)  # R; redundancy analysis (constrained ordination)",
            f"# m <- rda(comm ~ {' + '.join(pred_cols)}, data = d)",
            "# anova(m, permutations=999) 全局; by='axis' 轴显著; by='terms' 各预测变量",
        ]
    except Exception as err:
        summary.append(f"RDA 结果解析失败：{err}。" + degrade_to)
