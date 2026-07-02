"""R-bridge meta-analysis delegators (metafor: meta-analysis, meta-regression)."""

from __future__ import annotations


def _meta_via_r(csv_path, *, measure, roles, study_col, method, forest_png, funnel_png):
    """Random/fixed-effects meta-analysis via R metafor. `measure` is "GEN"
    (pre-computed effect sizes yi + vi/sei) or an escalc measure ("SMD"/"MD"/
    "OR"/"RR"/"RD") computed from raw study columns. Column names are passed as
    d[["name"]] vectors (no R symbol eval → injection-safe). Writes forest + funnel
    PNGs. Returns (meta dict, per-study DataFrame). Raises so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    fp_r = str(forest_png).replace("\\", "/")
    fn_r = str(funnel_png).replace("\\", "/")
    slab = f'd[["{study_col}"]]' if study_col else "NULL"
    if measure == "GEN":
        vi = f'd[["{roles["vi"]}"]]' if roles.get("vi") else f'(d[["{roles["sei"]}"]])^2'
        esc = f'yi <- d[["{roles["yi"]}"]]; vi <- {vi}\n'
    elif measure in ("SMD", "MD"):
        esc = (
            f'es <- escalc(measure="{measure}", m1i=d[["{roles["m1"]}"]], '
            f'sd1i=d[["{roles["sd1"]}"]], n1i=d[["{roles["n1"]}"]], '
            f'm2i=d[["{roles["m2"]}"]], sd2i=d[["{roles["sd2"]}"]], n2i=d[["{roles["n2"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    else:  # OR / RR / RD from a 2x2 table
        esc = (
            f'es <- escalc(measure="{measure}", ai=d[["{roles["ai"]}"]], '
            f'bi=d[["{roles["bi"]}"]], ci=d[["{roles["ci"]}"]], di=d[["{roles["di"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    rcode = (
        "suppressMessages(library(metafor))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        + esc
        + f'm <- rma(yi, vi, method="{method}", slab={slab})\n'
        'cat("##M\\n")\n'
        'cat(sprintf("estimate|%.6f\\nci_lb|%.6f\\nci_ub|%.6f\\npval|%.6g\\n", '
        "m$b[1], m$ci.lb, m$ci.ub, m$pval))\n"
        'cat(sprintf("Q|%.6f\\nQp|%.6g\\nI2|%.4f\\ntau2|%.6f\\nk|%d\\n", '
        "m$QE, m$QEp, m$I2, m$tau2, m$k))\n"
        # Egger's regression test for funnel asymmetry (needs k>=3); guard failures
        "eg <- tryCatch(regtest(m), error=function(e) NULL)\n"
        'if (!is.null(eg)) cat(sprintf("egger_z|%.4f\\negger_p|%.6g\\n", eg$zval, eg$pval))\n'
        'cat("##S\\n")\n'
        "sl <- if (is.null(m$slab)) seq_along(yi) else as.character(m$slab)\n"
        'for (i in seq_along(yi)) cat(sprintf("%s|%.6f|%.6f\\n", sl[i], yi[i], vi[i]))\n'
        f'png("{fp_r}", width=920, height=max(380, 70+34*length(yi)), res=120); '
        "forest(m); dev.off()\n"
        f'png("{fn_r}", width=720, height=620, res=120); funnel(m); dev.off()\n'
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, meta, srows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##M":
            section = "M"
        elif s == "##S":
            section = "S"
        elif "|" in s and section == "M":
            k, v = s.split("|", 1)
            meta[k] = float(v)
        elif "|" in s and section == "S":
            srows.append(s.rsplit("|", 2))
    if "estimate" not in meta or not srows:
        raise RuntimeError("metafor 未返回结果（检查效应量/方差列）")
    study = pd.DataFrame(srows, columns=["study", "yi", "vi"])
    for c in ("yi", "vi"):
        study[c] = pd.to_numeric(study[c], errors="coerce")
    return meta, study


def _meta_regression_via_r(csv_path, *, measure, roles, moderators, study, method, bubble_png, bubble_mod=None):
    """Meta-regression via R metafor — rma(yi, vi, mods = ~ moderators): explains
    between-study heterogeneity with study-level moderators. `measure` is "GEN"
    (pre-computed yi + vi/sei) or an escalc measure ("SMD"/"MD"/"OR"/"RR"/"RD").
    Reports the moderator coefficient table, the omnibus moderator test (QM),
    residual heterogeneity (I2/tau2), and R2 (% heterogeneity explained). Writes a
    bubble plot for the first moderator. Returns (coef_df, stats dict). Column
    names go through an identifier guard upstream. Raises so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    bp_r = str(bubble_png).replace("\\", "/")
    if measure == "GEN":
        vi = f'd[["{roles["vi"]}"]]' if roles.get("vi") else f'(d[["{roles["sei"]}"]])^2'
        esc = f'yi <- d[["{roles["yi"]}"]]; vi <- {vi}\n'
    elif measure in ("SMD", "MD"):
        esc = (
            f'es <- escalc(measure="{measure}", m1i=d[["{roles["m1"]}"]], '
            f'sd1i=d[["{roles["sd1"]}"]], n1i=d[["{roles["n1"]}"]], '
            f'm2i=d[["{roles["m2"]}"]], sd2i=d[["{roles["sd2"]}"]], n2i=d[["{roles["n2"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    else:  # OR / RR / RD
        esc = (
            f'es <- escalc(measure="{measure}", ai=d[["{roles["ai"]}"]], '
            f'bi=d[["{roles["bi"]}"]], ci=d[["{roles["ci"]}"]], di=d[["{roles["di"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    mods_r = "~ " + " + ".join(moderators)
    rcode = (
        "suppressMessages(library(metafor))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        + esc
        + f"m <- rma(yi, vi, mods = {mods_r}, data=d, method=\"{method}\")\n"
        'cat("##C\\n")\n'
        "for (i in seq_len(nrow(m$beta))) cat(sprintf('%s|%.6f|%.6f|%.6g|%.6f|%.6f\\n', "
        "rownames(m$beta)[i], m$beta[i], m$se[i], m$pval[i], m$ci.lb[i], m$ci.ub[i]))\n"
        'cat("##S\\n")\n'
        'cat(sprintf("QM|%.6f\\nQMp|%.6g\\nR2|%.4f\\nI2_resid|%.4f\\ntau2_resid|%.6f\\nk|%d\\n", '
        "m$QM, m$QMp, ifelse(is.null(m$R2),NA,m$R2), m$I2, m$tau2, m$k))\n"
        # also report the surviving moderator terms so the caller can detect drops
        'cat("##T\\n"); for (t in rownames(m$beta)) cat(t, "\\n")\n'
        # bubble plot for a NUMERIC moderator only (regplot needs continuous); guard failures
        + (
            f'tryCatch({{png("{bp_r}", width=760, height=600, res=120); '
            f'regplot(m, mod="{bubble_mod}", shade=TRUE); dev.off()}}, error=function(e) NULL)\n'
            if bubble_mod else ""
        )
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, crows, stats, surviving = None, [], {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##C":
            section = "C"
        elif s == "##S":
            section = "S"
        elif s == "##T":
            section = "T"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 5))
        elif "|" in s and section == "S":
            k, v = s.split("|", 1)
            try:
                stats[k] = float(v)  # NA (constant/collinear moderator) -> NaN, not a crash
            except ValueError:
                stats[k] = float("nan")
        elif section == "T" and s:
            surviving.append(s)
    if not crows or "QM" not in stats:
        raise RuntimeError("metafor meta-regression 未返回结果（检查效应量/调节变量列）")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "se", "p_value", "ci_lb", "ci_ub"])
    for c in ("estimate", "se", "p_value", "ci_lb", "ci_ub"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    stats["_surviving"] = [t for t in surviving if t != "intrcpt"]
    return coef, stats


