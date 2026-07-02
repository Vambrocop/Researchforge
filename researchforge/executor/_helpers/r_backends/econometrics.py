"""R-bridge dynamic-panel GMM delegator (plm)."""

from __future__ import annotations


def _dynamic_gmm_via_r(
    csv_path, unit: str, time: str, y: str, predictors: list[str],
    endogenous=None, gmm_lags=(2, 4),
):
    """Arellano-Bond difference-GMM dynamic panel via R plm::pgmm. Returns
    (coef DataFrame [term, estimate, std_err, p_value], diagnostics dict with
    sargan_stat/p and ar1_p/ar2_p). Raises so the caller can degrade honestly.

    By default every covariate is treated as strictly exogenous and only the
    lagged dependent is GMM-instrumented (lags `gmm_lags`). If `endogenous` names
    covariates, those join the GMM-instrument block (instrumented with their own
    lags `gmm_lags`, the Arellano-Bond treatment for endogenous regressors) and a
    third "normal instruments" formula part lists the remaining exogenous
    covariates as their own instruments."""
    import pandas as pd

    from researchforge.executor import rbridge

    endo = [p for p in (endogenous or []) if p in predictors]
    exog = [p for p in predictors if p not in endo]
    lo, hi = int(gmm_lags[0]), int(gmm_lags[1])
    # The lagged dependent must use lag>=2 in the differenced equation (y_{t-1}
    # correlates with Δε_t, so lag 1 is an INVALID instrument; Opus catch). Floor
    # its block at 2 even if the user lowers `lo` for predetermined covariates.
    y_lo = max(2, lo)
    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join([f"lag({y}, 1)", *predictors])
    # cap GMM instruments at lags lo:hi (not 2:99) to curb instrument proliferation,
    # which otherwise inflates/weakens the Sargan test (Roodman; Opus catch).
    gmm_inst = " + ".join([f"lag({y}, {y_lo}:{hi})", *(f"lag({p}, {lo}:{hi})" for p in endo)])
    if endo:
        # 3-part pgmm formula is REQUIRED here: pgmm auto-instruments omitted
        # covariates as their own (normal) instruments, which would wrongly treat an
        # endogenous regressor as exogenous. Listing only `exog` in part 3 (verified:
        # 2-part≡3-part for purely-exogenous models) confines self-instrumentation to
        # the exogenous set; if all covariates are endogenous, no part 3 is needed.
        formula = f"{y} ~ {rhs} | {gmm_inst}" + (f" | {' + '.join(exog)}" if exog else "")
    else:
        formula = f"{y} ~ {rhs} | {gmm_inst}"  # default: pgmm self-instruments all exog
    rcode = (
        "suppressMessages(library(plm))\n"
        f'd <- read.csv("{csv_r}")\n'
        f'pd <- pdata.frame(d, index=c("{unit}","{time}"))\n'
        f'm <- pgmm({formula}, data=pd, effect="individual", model="twosteps")\n'
        "s <- summary(m, robust=TRUE); ct <- s$coefficients\n"
        'cat("##COEF\\n")\n'
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,4]))\n'
        "a1 <- mtest(m, order=1); a2 <- mtest(m, order=2)\n"
        'cat("##DIAG\\n")\n'
        'cat(sprintf("sargan_stat|%.6f\\nsargan_p|%.6f\\n", s$sargan$statistic, s$sargan$p.value))\n'
        'cat(sprintf("ar1_p|%.6f\\nar2_p|%.6f\\n", a1$p.value, a2$p.value))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, crows, diag = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##DIAG":
            section = "D"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))  # term may contain no |; split last 3 fields
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            diag[k] = float(v)
    if not crows:
        raise RuntimeError("pgmm 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return coef, diag


