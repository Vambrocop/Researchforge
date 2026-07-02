"""R-bridge SEM delegator (lavaan)."""

from __future__ import annotations


def _sem_via_lavaan(csv_path, spec: str) -> dict:
    """Fit a CFA via R's lavaan (gold standard) through the R bridge. Returns a
    backend-agnostic result dict, or raises so the caller falls back to semopy."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rcode = (
        "suppressMessages(library(lavaan))\n"
        f'd <- read.csv("{csv_r}")\n'
        f'fit <- cfa("{spec}", data=d, std.lv=TRUE)\n'
        'std <- standardizedSolution(fit); ld <- std[std$op == "=~", ]\n'
        'cat("##LOADINGS\\n")\n'
        'for (i in seq_len(nrow(ld))) cat(sprintf("%s,%s,%.6f,%.6f,%.6f\\n", '
        "ld$rhs[i], ld$lhs[i], ld$est.std[i], ifelse(is.na(ld$se[i]),0,ld$se[i]), "
        "ifelse(is.na(ld$pvalue[i]),0,ld$pvalue[i])))\n"
        'fm <- fitMeasures(fit, c("chisq","df","cfi","tli","rmsea","srmr","aic","bic"))\n'
        'cat("##FIT\\n"); for (nm in names(fm)) cat(sprintf("%s,%.6f\\n", nm, fm[nm]))\n'
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, lrows, frows = None, [], []
    for line in out.splitlines():
        s = line.strip()
        if s == "##LOADINGS":
            section = "L"
        elif s == "##FIT":
            section = "F"
        elif "," in s:
            (lrows if section == "L" else frows).append(s)
    if not lrows or not frows:
        raise RuntimeError("lavaan produced no parseable output")
    load = pd.DataFrame(
        [r.split(",") for r in lrows],
        columns=["indicator", "factor", "std_loading", "std_err", "p_value"],
    )
    for c in ("std_loading", "std_err", "p_value"):
        load[c] = load[c].astype(float)
    fm = {k: float(v) for k, v in (r.split(",", 1) for r in frows)}
    return {
        "loadings": load,
        "fit": {
            "cfi": fm.get("cfi", float("nan")),
            "tli": fm.get("tli", float("nan")),
            "rmsea": fm.get("rmsea", float("nan")),
            "chi2": fm.get("chisq", float("nan")),
            "dof": fm.get("df", float("nan")),
            "srmr": fm.get("srmr", float("nan")),
        },
        "summary": out,
        "backend": "lavaan (R)",
    }


