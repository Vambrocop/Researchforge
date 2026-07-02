"""R-bridge stochastic-frontier delegator (frontier/sfa)."""

from __future__ import annotations


def _sfa_via_r(csv_path, output: str, inputs: list[str]):
    """Stochastic Frontier Analysis via R frontier: Cobb-Douglas production
    frontier log(y) ~ Σ log(x), ML with composed error v−u. Returns
    (coef dict incl. sigmaSq & gamma, per-row technical-efficiency array)."""
    import numpy as np

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(f"log({c})" for c in inputs)
    rcode = (
        "suppressMessages(library(frontier))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"m <- sfa(log({output}) ~ {rhs}, data=d)\n"
        "co <- coef(m)\n"
        # one-sided LR test of γ=0 (no inefficiency) vs OLS; boundary mixed χ² ⇒ ½ weight
        f"ols <- lm(log({output}) ~ {rhs}, data=d)\n"
        "lr <- max(0, 2*(as.numeric(logLik(m)) - as.numeric(logLik(ols))))\n"
        "pval <- 0.5 * pchisq(lr, df=1, lower.tail=FALSE)\n"
        'cat("##COEF\\n"); for (nm in names(co)) cat(sprintf("%s|%.6f\\n", nm, co[nm]))\n'
        'cat(sprintf("lr_stat|%.6f\\nlr_pvalue|%.6f\\n", lr, pval))\n'
        "eff <- efficiencies(m, asInData=TRUE)\n"
        'cat("##EFF\\n"); for (i in seq_along(eff)) cat(sprintf("%.6f\\n", eff[i]))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, coef, te = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##EFF":
            section = "E"
        elif section == "C" and "|" in s:
            k, v = s.split("|", 1)
            coef[k] = float(v)
        elif section == "E" and s and "|" not in s:
            try:
                te.append(float(s))
            except ValueError:
                pass
    if not coef or not te:
        raise RuntimeError("frontier sfa 未返回结果")
    return coef, np.array(te)


