"""R-bridge causal delegators (qte changes-in-changes, gsynth)."""

from __future__ import annotations


def _cic_via_r(csv_path, outcome, treat, time, t_post, t_pre, probs, qte_png):
    """Changes-in-changes (Athey & Imbens 2006) via R qte::CiC — a distributional
    generalization of DID: estimates quantile treatment effects (QTE) across the
    outcome distribution plus the overall ATT, relaxing parallel-trends to a
    monotonicity/rank-invariance assumption. Returns a dict (ATT + CI, per-quantile
    QTE + CI). Writes a QTE-by-quantile plot. Raises so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(qte_png).replace("\\", "/")
    probs_r = ", ".join(f"{p:.4f}" for p in probs)
    # t_post / t_pre are interpolated as numeric literals (validated numeric upstream)
    rcode = (
        "suppressMessages(library(qte))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        'd[[".rid"]] <- seq_len(nrow(d))\n'
        # seed the empirical-bootstrap SE/CI (iters=100) so the ATT/QTE confidence bounds
        # are reproducible run-to-run — unseeded they jitter and a tight CI can flip
        # across runs (a flaky test surfaced exactly this under parallel execution).
        "set.seed(20240607)\n"
        f'r <- CiC({outcome} ~ {treat}, t={t_post}, tmin1={t_pre}, tname="{time}", '
        'data=d, idname=".rid", panel=FALSE, '
        f"probs=c({probs_r}), se=TRUE, iters=100)\n"
        'cat("##A\\n")\n'
        'cat(sprintf("ate|%.6f\\nate_lb|%.6f\\nate_ub|%.6f\\nate_se|%.6f\\n", '
        "r$ate, r$ate.lower, r$ate.upper, r$ate.se))\n"
        'cat("##Q\\n")\n'
        "for (i in seq_along(r$probs)) cat(sprintf('%.4f|%.6f|%.6f|%.6f|%.6f\\n', "
        "r$probs[i], r$qte[i], r$qte.se[i], r$qte.lower[i], r$qte.upper[i]))\n"
        f'png("{png_r}", width=760, height=520, res=120)\n'
        "plot(r$probs, r$qte, type='b', pch=19, ylim=range(c(r$qte.lower, r$qte.upper)), "
        'xlab="quantile", ylab="quantile treatment effect (QTE)", main="Changes-in-changes: QTE by quantile")\n'
        "arrows(r$probs, r$qte.lower, r$probs, r$qte.upper, length=0.03, angle=90, code=3, col='grey')\n"
        "abline(h=r$ate, lty=2, col='red'); abline(h=0, lty=3, col='grey')\n"
        "dev.off()\n"
    )
    out = rbridge.run_r(rcode, timeout=240)
    section, meta, qrows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##A":
            section = "A"
        elif s == "##Q":
            section = "Q"
        elif "|" in s and section == "A":
            k, v = s.split("|", 1)
            meta[k] = float(v)
        elif "|" in s and section == "Q":
            qrows.append(s.split("|"))
    if "ate" not in meta or not qrows:
        raise RuntimeError("qte CiC 未返回结果（检查分组/时间/结果列与两期设置）")
    qte = pd.DataFrame(qrows, columns=["quantile", "qte", "se", "ci_lb", "ci_ub"])
    for c in ("quantile", "qte", "se", "ci_lb", "ci_ub"):
        qte[c] = pd.to_numeric(qte[c], errors="coerce")
    return meta, qte


def _gsynth_via_r(csv_path, outcome, treat, unit, time, n_boots, png_path, seed=2024):
    """Generalized synthetic control via R gsynth (Xu 2017): an interactive
    fixed-effects counterfactual that handles MULTIPLE treated units and staggered
    adoption (vs the single-treated classic synthetic control). The number of
    latent factors is chosen by cross-validation. Returns (meta dict, att-by-time
    DataFrame). Writes a dynamic-ATT plot. Raises so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(png_path).replace("\\", "/")
    rcode = (
        "suppressMessages(library(gsynth))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'out <- gsynth({outcome} ~ {treat}, data=d, index=c("{unit}","{time}"), '
        'force="two-way", CV=TRUE, r=c(0,5), se=TRUE, inference="parametric", '
        # seed the CV + parametric bootstrap — unseeded, the factor count and CI flip
        # run-to-run (significance not reproducible; Opus catch, same as DML/causal_forest).
        f"nboots={int(n_boots)}, parallel=FALSE, seed={int(seed)})\n"
        "ea <- out$est.avg\n"
        'cat("##A\\n")\n'
        'cat(sprintf("att|%.6f\\nse|%.6f\\nci_lb|%.6f\\nci_ub|%.6f\\npval|%.6g\\n", '
        'ea[1,"ATT.avg"], ea[1,"S.E."], ea[1,"CI.lower"], ea[1,"CI.upper"], ea[1,"p.value"]))\n'
        'cat(sprintf("r_cv|%d\\nn_treated|%d\\nn_control|%d\\nn_periods|%d\\n", '
        "out$r.cv, out$Ntr, out$Nco, out$T))\n"
        'cat("##T\\n"); et <- out$est.att\n'
        'for (i in seq_len(nrow(et))) cat(sprintf("%s|%.6f|%.6f|%.6f\\n", '
        'rownames(et)[i], et[i,"ATT"], et[i,"CI.lower"], et[i,"CI.upper"]))\n'
        f'png("{png_r}", width=820, height=520, res=120)\n'
        "tt <- as.numeric(rownames(et))\n"
        'plot(tt, et[,"ATT"], type="l", lwd=2, col="#C44E52", ylim=range(c(et[,"CI.lower"],et[,"CI.upper"])), '
        'xlab="time", ylab="ATT (treated - synthetic)", main="Generalized synthetic control: dynamic ATT")\n'
        'polygon(c(tt,rev(tt)), c(et[,"CI.lower"],rev(et[,"CI.upper"])), col=rgb(0.3,0.3,0.3,0.15), border=NA)\n'
        'abline(h=0, lty=2, col="grey"); dev.off()\n'
    )
    out = rbridge.run_r(rcode, timeout=300)
    section, meta, trows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##A":
            section = "A"
        elif s == "##T":
            section = "T"
        elif "|" in s and section == "A":
            k, v = s.split("|", 1)
            meta[k] = float(v)
        elif "|" in s and section == "T":
            trows.append(s.rsplit("|", 3))
    if "att" not in meta or not trows:
        raise RuntimeError("gsynth 未返回结果（检查面板/处理结构与收敛）")
    att_t = pd.DataFrame(trows, columns=["time", "att", "ci_lb", "ci_ub"])
    for c in ("time", "att", "ci_lb", "ci_ub"):
        att_t[c] = pd.to_numeric(att_t[c], errors="coerce")
    return meta, att_t


