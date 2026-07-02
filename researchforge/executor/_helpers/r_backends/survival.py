"""R-bridge joint longitudinal-survival delegator (JM/JMbayes)."""

from __future__ import annotations


def _jm_via_r(csv_path, idc, timec, marker, surv_time, event, covariates):
    """Joint longitudinal–survival model via R JM (Rizopoulos): a linear mixed
    model for the longitudinal marker + a Cox/Weibull-PH survival submodel, linked
    by the ASSOCIATION parameter (how the current marker value shifts the event
    hazard). Returns (event_df, long_df, fit dict). Column names are identifier-
    guarded upstream. Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    covf = " + ".join(covariates) if covariates else "1"
    rcode = (
        "suppressMessages(library(JM))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'd <- d[order(d[["{idc}"]], d[["{timec}"]]), ]\n'
        f"lmeFit <- lme({marker} ~ {timec}, random = ~ {timec} | {idc}, data = d, "
        'control = lmeControl(opt="optim", returnObject=TRUE))\n'
        f'sid <- d[!duplicated(d[["{idc}"]]), ]\n'
        f"coxFit <- coxph(Surv({surv_time}, {event}) ~ {covf}, data = sid, x = TRUE)\n"
        f'jf <- jointModel(lmeFit, coxFit, timeVar = "{timec}", method = "weibull-PH-aGH")\n'
        "s <- summary(jf)\n"
        'cat("##E\\n"); ev <- s[["CoefTable-Event"]]\n'
        'for (i in seq_len(nrow(ev))) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", '
        "rownames(ev)[i], ev[i,1], ev[i,2], ev[i,4]))\n"
        'cat("##L\\n"); lo <- s[["CoefTable-Long"]]\n'
        'for (i in seq_len(nrow(lo))) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", '
        "rownames(lo)[i], lo[i,1], lo[i,2], lo[i,4]))\n"
        'cat("##F\\n")\n'
        # count subjects/events directly from the per-subject data (s$d is not the event count)
        f'nev <- sum(as.numeric(sid[["{event}"]]) == 1)\n'
        "cat(sprintf(\"aic|%.4f\\nn_subjects|%d\\nn_events|%d\\nconv|%d\\n\", "
        "s$AIC, nrow(sid), as.integer(nev), as.integer(jf$conv)))\n"
    )
    out = rbridge.run_r(rcode, timeout=300)
    section, erows, lrows, fit = None, [], [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##E":
            section = "E"
        elif s == "##L":
            section = "L"
        elif s == "##F":
            section = "F"
        elif "|" in s and section == "E":
            erows.append(s.rsplit("|", 3))
        elif "|" in s and section == "L":
            lrows.append(s.rsplit("|", 3))
        elif "|" in s and section == "F":
            k, v = s.split("|", 1)
            fit[k] = float(v)
    if not erows or "aic" not in fit:
        raise RuntimeError("JM jointModel 未返回结果（检查纵向/生存结构与收敛）")
    # JM does NOT error on non-convergence — it returns numbers with conv!=0 (Opus
    # catch). Reject them: a non-converged association/p-value is not trustworthy.
    if fit.get("conv", 0) != 0:
        raise RuntimeError("JM 联合模型未收敛（conv≠0），结果不可信——可换更简单的轨迹或加迭代")
    ev = pd.DataFrame(erows, columns=["term", "value", "std_err", "p_value"])
    lo = pd.DataFrame(lrows, columns=["term", "value", "std_err", "p_value"])
    for t in (ev, lo):
        for c in ("value", "std_err", "p_value"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return ev, lo, fit


