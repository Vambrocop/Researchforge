"""Executor backend delegators (R bridge / econml / doubleml / semopy) — heavy
external-estimator wrappers moved out of the run.py monolith. Re-exported by run.py.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from researchforge.catalog.schema import AnalysisEntry
from researchforge.executor._helpers.core import _sem_latents
from researchforge.profiler.fingerprint import DataFingerprint


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


def _sem_via_semopy(sub, spec: str) -> dict:
    """Fit a CFA/SEM model spec with pure-Python semopy (portable fallback).
    Extracts standardised loadings for ALL latent factors in the spec (not just a
    single factor named F), so custom multi-factor specs are handled."""
    import semopy

    latents = _sem_latents(spec) or ["F"]
    model = semopy.Model(spec)
    model.fit(sub)
    ins = model.inspect(std_est=True)
    load = ins[(ins["op"] == "~") & (ins["rval"].isin(latents))][
        ["lval", "rval", "Est. Std", "Std. Err", "p-value"]
    ].copy()
    load.columns = ["indicator", "factor", "std_loading", "std_err", "p_value"]
    stats = semopy.calc_stats(model)
    return {
        "loadings": load,
        "fit": {
            "cfi": float(stats["CFI"].iloc[0]),
            "tli": float(stats["TLI"].iloc[0]),
            "rmsea": float(stats["RMSEA"].iloc[0]),
            "chi2": float(stats["chi2"].iloc[0]),
            "dof": float(stats["DoF"].iloc[0]),
            "srmr": float("nan"),
        },
        "summary": str(ins),
        "backend": "semopy (Python)",
    }


def _csqca_via_r(csv_path, outcome: str, conditions: list[str], incl_cut=0.8):
    """Crisp-set QCA via R QCA: dichotomise (binary kept; continuous cut at the
    median) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x) {\n"
        "  nn <- x[!is.na(x)]; u <- unique(nn)\n"
        "  if (length(u) <= 2 && all(u %in% c(0,1))) return(as.integer(x))\n"
        "  as.integer(x > median(nn))\n"  # crisp threshold at the median
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), incl.cut={float(incl_cut)}, show.cases=FALSE)\n'
        "sol <- minimize(tt, details=TRUE)\n"
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##IC\\n"); ic <- sol$IC$incl.cov\n'
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclS[i], ic$covS[i], ic$covU[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##IC":
            section = "I"
        elif section == "S" and s:
            sol_str = s
        elif section == "I" and "|" in s:
            rows.append(s.split("|"))
    if not sol_str or not rows:
        raise RuntimeError("QCA 未返回充分配置（无解 / 有限多样性）")
    tab = pd.DataFrame(rows, columns=["configuration", "consistency", "raw_coverage", "unique_coverage"])
    for c in ("consistency", "raw_coverage", "unique_coverage"):
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return sol_str, tab


def _fsqca_via_r(csv_path, outcome: str, conditions: list[str], anchors=(0.1, 0.5, 0.9), incl_cut=0.8):
    """Run fsQCA with R's QCA package: direct fuzzy calibration (percentile
    anchors) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report an honest message."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        # skewed/discrete columns can tie the anchor quantiles; calibrate
        # needs strictly increasing anchors, so nudge ties by a tiny epsilon.
        "calib <- function(x) {\n"
        f"  thr <- as.numeric(quantile(x, c({a0},{a1},{a2}), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib))\n"
        "names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), '
        f"incl.cut={float(incl_cut)}, show.cases=FALSE)\n"
        "sol <- minimize(tt, details=TRUE)\n"
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##IC\\n"); ic <- sol$IC$incl.cov\n'
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclS[i], ic$covS[i], ic$covU[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##IC":
            section = "I"
        elif section == "S" and s:
            sol_str = s
        elif section == "I" and "|" in s:
            rows.append(s.split("|"))
    if not sol_str or not rows:
        raise RuntimeError("QCA 未返回充分配置（可能无解 / 有限多样性）")
    tab = pd.DataFrame(
        rows, columns=["configuration", "consistency", "raw_coverage", "unique_coverage"]
    )
    for c in ("consistency", "raw_coverage", "unique_coverage"):
        # R prints NA (e.g. unique coverage of a single-term solution) -> NaN
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return sol_str, tab


def _kriging_via_r(csv_path, lon: str, lat: str, value: str):
    """Ordinary kriging via R gstat: auto-fit a variogram (Sph/Exp/Gau), LOO
    cross-validation RMSE, and predict a 40x40 grid with kriging variance.
    Returns (meta dict, grid DataFrame[lon,lat,pred,var]). Raises on failure."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rcode = (
        "suppressMessages({library(gstat); library(sp)})\n"
        f'd <- read.csv("{csv_r}")\n'
        f"coordinates(d) <- ~{lon}+{lat}\n"
        f"v <- variogram({value} ~ 1, d)\n"
        # explicit initial psill/range/nugget so fit.variogram doesn't silently
        # return a poor/zero-nugget fit from bare defaults (Opus catch).
        f'vy <- var(d@data[["{value}"]], na.rm=TRUE)\n'
        # Spherical/Exponential only — the Gaussian model gives near-singular
        # kriging matrices and unstable (blown-up) predictions (Opus catch via test).
        'vm <- fit.variogram(v, vgm(psill=0.9*vy, model=c("Sph","Exp"), '
        "range=max(v$dist)/2, nugget=0.1*vy))\n"
        f"cv <- krige.cv({value} ~ 1, d, model=vm, nfold=nrow(d@data), verbose=FALSE)\n"
        "rmse <- sqrt(mean(cv$residual^2, na.rm=TRUE))\n"
        "bb <- d@bbox\n"
        f"g <- expand.grid({lon}=seq(bb[1,1],bb[1,2],length.out=40), "
        f"{lat}=seq(bb[2,1],bb[2,2],length.out=40)); coordinates(g) <- ~{lon}+{lat}\n"
        f"k <- krige({value} ~ 1, d, g, model=vm, debug.level=0)\n"
        'cat("##META\\n")\n'
        "n <- nrow(vm)\n"
        'cat(sprintf("model|%s\\nrange|%.5f\\nloo_rmse|%.5f\\n", as.character(vm$model[n]), vm$range[n], rmse))\n'
        'cat("##GRID\\n"); gd <- as.data.frame(k)\n'
        f'for (i in seq_len(nrow(gd))) cat(sprintf("%.5f|%.5f|%.5f|%.5f\\n", gd${lon}[i], gd${lat}[i], gd$var1.pred[i], gd$var1.var[i]))\n'
    )
    out = rbridge.run_r(rcode, timeout=240)
    section, meta, rows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##META":
            section = "M"
        elif s == "##GRID":
            section = "G"
        elif section == "M" and "|" in s:
            kk, vv = s.split("|", 1)
            meta[kk] = vv if kk == "model" else float(vv)
        elif section == "G" and "|" in s:
            rows.append(s.split("|"))
    if not rows or "loo_rmse" not in meta:
        raise RuntimeError("gstat kriging 未返回结果（变异函数拟合可能失败）")
    grid = pd.DataFrame(rows, columns=[lon, lat, "prediction", "kriging_variance"])
    for c in (lon, lat, "prediction", "kriging_variance"):
        grid[c] = pd.to_numeric(grid[c], errors="coerce")
    return meta, grid


def _spatial_reg_via_r(csv_path, outcome: str, predictors: list[str], lon: str, lat: str, k: int = 6):
    """Spatial regression via R spdep/spatialreg: OLS residual Moran test, then
    SAR (lag) and SEM (error) on k-NN weights, AIC model choice. Reports the
    PREFERRED model's variable effects — SEM betas are marginal effects; for SAR
    the betas are NOT marginal, so impacts() direct/indirect/total are reported
    instead. Returns (diagnostics dict, preferred str, effects DataFrame)."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(predictors)
    preds_r = ", ".join(f'"{p}"' for p in predictors)
    rcode = (
        "suppressMessages({library(spdep); library(spatialreg)})\n"
        f'd <- read.csv("{csv_r}")\n'
        f'coords <- as.matrix(d[, c("{lon}","{lat}")])\n'
        f"lw <- nb2listw(knn2nb(knearneigh(coords, k=min({int(k)}, nrow(d)-1))), style=\"W\")\n"
        f'f <- as.formula("{outcome} ~ {rhs}")\n'
        "ols <- lm(f, data=d); mt <- lm.morantest(ols, lw)\n"
        "sar <- lagsarlm(f, data=d, listw=lw); sem <- errorsarlm(f, data=d, listw=lw)\n"
        'pref <- if (AIC(sar) <= AIC(sem)) "SAR" else "SEM"\n'
        'cat("##DIAG\\n")\n'
        'cat(sprintf("resid_moran_p|%.5g\\n", mt$p.value))\n'
        'cat(sprintf("ols_aic|%.3f\\n", AIC(ols)))\n'
        'cat(sprintf("sar_aic|%.3f\\nsar_rho|%.4f\\n", AIC(sar), sar$rho))\n'
        'cat(sprintf("sem_aic|%.3f\\nsem_lambda|%.4f\\n", AIC(sem), sem$lambda))\n'
        'cat("##PREF\\n"); cat(pref, "\\n")\n'
        f"preds <- c({preds_r})\n"
        'cat("##COEF\\n")\n'
        'if (pref == "SAR") {\n'
        "  imp <- impacts(sar, listw=lw)\n"
        "  for (i in seq_along(preds)) cat(sprintf('%s|%.5f|%.5f|%.5f\\n', "
        "preds[i], imp$direct[i], imp$indirect[i], imp$total[i]))\n"
        "} else {\n"
        "  co <- summary(sem)$Coef\n"
        "  for (nm in preds) cat(sprintf('%s|%.5f|%.5f|%.5g\\n', nm, co[nm,1], co[nm,2], co[nm,4]))\n"
        "}\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, diag, pref, crows = None, {}, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##DIAG":
            section = "D"
        elif s == "##PREF":
            section = "P"
        elif s == "##COEF":
            section = "C"
        elif section == "P" and s:
            pref = s
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            diag[k] = float(v)
        elif "|" in s and section == "C":
            crows.append(s.split("|"))
    if not crows or not pref:
        raise RuntimeError("spatialreg 未返回系数")
    cols = (
        ["term", "direct", "indirect", "total"]
        if pref == "SAR"
        else ["term", "estimate", "std_err", "p_value"]
    )
    coef = pd.DataFrame(crows, columns=cols)
    for c in cols[1:]:
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return diag, pref, coef


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


def _diff_abundance_aldex2_via_r(csv_path, taxa: list[str], group: str):
    """Differential abundance via R ALDEx2 (compositional gold standard): CLR with
    Monte-Carlo Dirichlet sampling of the counts, Welch t per taxon over the MC
    instances, BH-FDR. ALDEx2 expects features (taxa) as ROWS, samples as COLUMNS.
    Returns a DataFrame [taxon, effect, diff_btw, p_value, q_value]. Raises so the
    caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    taxa_r = ", ".join(f'"{t}"' for t in taxa)
    rcode = (
        "suppressMessages(library(ALDEx2))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"taxa <- c({taxa_r})\n"
        "counts <- t(as.matrix(d[, taxa]))\n"  # features x samples
        f'conds <- as.character(d[["{group}"]])\n'
        # 128 MC Dirichlet instances (ALDEx2 default); Welch t with effect sizes
        'x <- aldex(round(counts), conds, mc.samples=128, test="t", effect=TRUE, denom="all", verbose=FALSE)\n'
        'cat("##R\\n")\n'
        'for (i in seq_len(nrow(x))) cat(sprintf("%s|%.6f|%.6f|%.6g|%.6g\\n", '
        "rownames(x)[i], x$effect[i], x$diff.btw[i], x$we.ep[i], x$we.eBH[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=300)
    rows = []
    section = None
    for line in out.splitlines():
        s = line.strip()
        if s == "##R":
            section = "R"
        elif section == "R" and "|" in s:
            rows.append(s.split("|"))
    if not rows:
        raise RuntimeError("ALDEx2 未返回结果")
    res = pd.DataFrame(rows, columns=["taxon", "effect", "diff_btw", "p_value", "q_value"])
    for c in ("effect", "diff_btw", "p_value", "q_value"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


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


def _panel_qca_via_r(csv_path, outcome, conditions, unit, anchors, incl_cut):
    """Panel/clustered fsQCA via R SetMethods::cluster (Garcia-Castro & Ariño):
    fuzzy-calibrate, derive the pooled fsQCA solution, then decompose each
    solution term's consistency into POOLED vs BETWEEN-unit vs WITHIN-unit, with
    the between→pooled (dBP) and within→pooled (dWP) distances. A large distance
    means the configuration does NOT hold uniformly across units / over time
    (clustered heterogeneity). Returns (solution_str, terms_df). Raises on no
    result. Column names go through an identifier guard upstream."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages({library(SetMethods); library(QCA)})\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x){ thr<-as.numeric(quantile(x,c("
        f"{a0},{a1},{a2}),na.rm=TRUE)); rng<-diff(range(x,na.rm=TRUE)); "
        "eps<-if(rng>0) rng*1e-6 else 1e-6; if(thr[2]<=thr[1]) thr[2]<-thr[1]+eps; "
        'if(thr[3]<=thr[2]) thr[3]<-thr[2]+eps; calibrate(x,type="fuzzy",thresholds=thr) }\n'
        "cal <- as.data.frame(lapply(d[,cols], calib)); names(cal) <- cols\n"
        f'cal[["UNIT"]] <- as.character(d[["{unit}"]])\n'
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), incl.cut={float(incl_cut)})\n'
        "sol <- minimize(tt, details=TRUE)\n"
        f'cl <- cluster(data=cal, results=sol, outcome="{outcome}", unit_id="UNIT", cluster_id="UNIT")\n'
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##TERMS\\n")\n'
        "for (t in names(cl$output)) { o <- cl$output[[t]]; "
        'cat(sprintf("%s|%.4f|%.4f|%.4f|%.4f\\n", t, o$POCOS, o$Coverages$pooled, o$dBP, o$dWP)) }\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##TERMS":
            section = "T"
        elif section == "S" and s:
            sol_str = s
        elif "|" in s and section == "T":
            rows.append(s.rsplit("|", 4))
    if not rows:
        raise RuntimeError("SetMethods cluster 未返回分解（可能无解/单解项/有限多样性）")
    terms = pd.DataFrame(
        rows, columns=["term", "pooled_consistency", "pooled_coverage", "dist_between", "dist_within"]
    )
    for c in ("pooled_consistency", "pooled_coverage", "dist_between", "dist_within"):
        terms[c] = pd.to_numeric(terms[c], errors="coerce")
    return sol_str, terms


def _cna_via_r(csv_path, factors, outcome, con, cov, anchors, fuzzy):
    """Coincidence Analysis via R `cna` — finds configurational causal structures
    (Boolean solution formulas) and, unlike QCA, can recover structures with
    MULTIPLE outcomes (it does not require pre-designating one). Crisp (0/1) data
    is used directly (type="cs"); continuous factors are fuzzy-calibrated by
    percentile anchors (type="fs"). Returns (asf_df, n_csf). asf columns:
    outcome, condition, consistency, coverage, complexity. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    facs_r = ", ".join(f'"{c}"' for c in factors)
    out_arg = f', outcome=c("{outcome}")' if outcome else ""
    typ = "fs" if fuzzy else "cs"
    calib = (
        # fuzzy-calibrate non-binary columns to [0,1]; leave 0/1 columns as-is
        "calib <- function(x){ u<-unique(x[!is.na(x)]); "
        "if(length(u)<=2 && all(u %in% c(0,1))) return(x); "
        f"thr<-as.numeric(quantile(x,c({a0},{a1},{a2}),na.rm=TRUE)); "
        "rng<-diff(range(x,na.rm=TRUE)); eps<-if(rng>0) rng*1e-6 else 1e-6; "
        "if(thr[2]<=thr[1]) thr[2]<-thr[1]+eps; if(thr[3]<=thr[2]) thr[3]<-thr[2]+eps; "
        'calibrate(x,type="fuzzy",thresholds=thr) }\n'
        "d <- as.data.frame(lapply(d[,facs], calib)); names(d)<-facs\n"
        if fuzzy
        else "d <- d[, facs]\n"
    )
    rcode = (
        "suppressMessages(library(cna))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"facs <- c({facs_r})\n"
        + calib
        + f'x <- cna(d, type="{typ}", con={float(con)}, cov={float(cov)}{out_arg})\n'
        "a <- asf(x)\n"
        'cat("##ASF\\n")\n'
        "if (nrow(a)) for (i in seq_len(nrow(a))) cat(sprintf('%s|%s|%.4f|%.4f|%d\\n', "
        "a$outcome[i], a$condition[i], a$con[i], a$cov[i], a$complexity[i]))\n"
        'cat("##CSF\\n"); cat(sprintf("n|%d\\n", nrow(csf(x))))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, rows, n_csf = None, [], 0
    for line in out.splitlines():
        s = line.strip()
        if s == "##ASF":
            section = "A"
        elif s == "##CSF":
            section = "C"
        elif "|" in s and section == "A":
            rows.append(s.rsplit("|", 4))
        elif s.startswith("n|") and section == "C":
            n_csf = int(s.split("|", 1)[1])
    if not rows:
        raise RuntimeError("cna 未返回解（可能 con/cov 阈值过高或无配置性结构）")
    asf = pd.DataFrame(rows, columns=["outcome", "condition", "consistency", "coverage", "complexity"])
    for c in ("consistency", "coverage", "complexity"):
        asf[c] = pd.to_numeric(asf[c], errors="coerce")
    return asf, n_csf


def _dml_via_doubleml(df, outcome, treatment, controls, n_folds, discrete, plot_path, seed=0):
    """Double/debiased machine learning (Chernozhukov et al.) via the doubleml
    package: ML-learn the nuisance functions (outcome + treatment/propensity),
    cross-fit, and estimate an orthogonalized average treatment effect. Binary
    treatment -> IRM (ATE, no functional-form on the effect); continuous -> PLR
    (partially-linear). RandomForest learners. Returns a dict (incl. an overlap
    flag, the treatment 0/1 encoding, and the estimand label). Writes an ATE
    point+CI plot. Raises so the caller can degrade honestly."""
    import warnings

    import numpy as np
    import doubleml as dml
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    sub = df[[outcome, treatment, *controls]].dropna().copy()
    # IRM requires the treatment coded exactly {0,1}; normalise any 2-valued
    # encoding ({1,2}, {"Treated","Control"}, …) with a disclosed mapping (Opus catch).
    treat_map = None
    if discrete:
        vals = sorted(sub[treatment].unique().tolist(), key=lambda v: str(v))
        if set(vals) != {0, 1}:
            treat_map = {vals[0]: 0, vals[1]: 1}
            sub[treatment] = sub[treatment].map(treat_map).astype(int)
    # pin the cross-fitting sample split — it is drawn from the global NumPy RNG,
    # which the learners' own random_state does NOT control (Opus catch: ATE was
    # varying run-to-run without this).
    np.random.seed(int(seed))
    data = dml.DoubleMLData(sub, y_col=outcome, d_cols=treatment, x_cols=list(controls))
    reg = RandomForestRegressor(n_estimators=100, random_state=0)
    if discrete:
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        model = dml.DoubleMLIRM(data, ml_g=reg, ml_m=clf, n_folds=n_folds)
        kind = "IRM"
    else:
        model = dml.DoubleMLPLR(
            data, ml_l=reg, ml_m=RandomForestRegressor(n_estimators=100, random_state=1), n_folds=n_folds
        )
        kind = "PLR"
    overlap_warn = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit()
        overlap_warn = any("close to zero or one" in str(w.message) for w in caught)
    ci = model.confint(level=0.95)
    out = {
        "ate": float(model.coef[0]),
        "se": float(model.se[0]),
        "ci_lb": float(ci.iloc[0, 0]),
        "ci_ub": float(ci.iloc[0, 1]),
        "p_value": float(model.pval[0]),
        "model": kind,
        "n": int(sub.shape[0]),
        "overlap_warn": overlap_warn,
        "treat_map": treat_map,
        "estimand": "ATE" if discrete else "PLR_coefficient",
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.errorbar([out["ate"]], [0], xerr=[[out["ate"] - out["ci_lb"]], [out["ci_ub"] - out["ate"]]],
                    fmt="o", color="#4C72B0", capsize=5)
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_yticks([])
        ax.set_xlabel(f"ATE of {treatment} on {outcome} (95% CI)")
        ax.set_title(f"Double ML ({kind}) — ATE = {out['ate']:.3f}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return out


def _causal_forest_via_econml(df, outcome, treatment, modifiers, n_folds, discrete, seed, hist_png, scatter_png):
    """Heterogeneous treatment effects (CATE) via econml CausalForestDML — a
    causal-forest DML estimator: residualize Y and T on the covariates (ML), then
    grow a causal forest over the effect-modifiers X to estimate effect(x). Reports
    the overall ATE, the CATE distribution, the share of individuals with a
    significant effect, and which modifiers drive heterogeneity. Binary treatment
    is normalized to {0,1} (disclosed). Returns a dict; writes a CATE histogram and
    a CATE-vs-top-modifier scatter. Raises so the caller can degrade honestly."""
    import numpy as np
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    sub = df[[outcome, treatment, *modifiers]].dropna().copy()
    treat_map = None
    if discrete:
        vals = sorted(sub[treatment].unique().tolist(), key=lambda v: str(v))
        if set(vals) != {0, 1}:
            treat_map = {vals[0]: 0, vals[1]: 1}
            sub[treatment] = sub[treatment].map(treat_map).astype(int)
    np.random.seed(int(seed))
    Y = sub[outcome].astype(float).to_numpy()
    T = sub[treatment].to_numpy()
    X = sub[modifiers].astype(float).to_numpy()
    model_t = (
        RandomForestClassifier(n_estimators=100, random_state=int(seed))
        if discrete
        else RandomForestRegressor(n_estimators=100, random_state=int(seed))
    )
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=int(seed)),
        model_t=model_t, discrete_treatment=discrete, n_estimators=300,
        random_state=int(seed), cv=n_folds,
    )
    est.fit(Y, T, X=X)
    cate = np.asarray(est.effect(X), dtype=float)
    ate = float(est.ate(X))
    a_lb, a_ub = (float(v) for v in est.ate_interval(X, alpha=0.05))
    lb, ub = est.effect_interval(X, alpha=0.05)  # per-row CI
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    frac_sig = float(np.mean((lb > 0) | (ub < 0)))  # share with a significant individual effect
    imp = np.asarray(est.feature_importances_, dtype=float)
    order = np.argsort(imp)[::-1]
    drivers = [(modifiers[i], round(float(imp[i]), 3)) for i in order]
    out = {
        "ate": ate, "ate_lb": a_lb, "ate_ub": a_ub,
        "cate_mean": float(cate.mean()), "cate_sd": float(cate.std()),
        "cate_p10": float(np.percentile(cate, 10)), "cate_p90": float(np.percentile(cate, 90)),
        "frac_significant": frac_sig, "drivers": drivers, "n": int(sub.shape[0]),
        "treat_map": treat_map,
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(cate, bins=30, color="#4C72B0", edgecolor="white")
        ax.axvline(ate, color="#C44E52", ls="--", label=f"ATE={ate:.3f}")
        ax.axvline(0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel(f"individual treatment effect (CATE) of {treatment} on {outcome}")
        ax.set_ylabel("count")
        ax.set_title("CATE distribution (heterogeneous effects)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(hist_png, dpi=150)
        plt.close(fig)
        top = order[0]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(X[:, top], cate, s=10, alpha=0.4, edgecolor="none", color="#55A868")
        ax2.axhline(0, color="grey", ls=":", lw=0.8)
        ax2.set_xlabel(f"{modifiers[top]} (top effect-modifier)")
        ax2.set_ylabel("CATE")
        ax2.set_title(f"CATE vs {modifiers[top]}")
        fig2.tight_layout()
        fig2.savefig(scatter_png, dpi=150)
        plt.close(fig2)
    except Exception:
        pass
    return out


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


def _rdd_via_rdrobust(df, outcome, running, cutoff, plot_path):
    """Sharp regression discontinuity (Calonico–Cattaneo–Titiunik) via rdrobust:
    local-linear estimate of the outcome jump at the cutoff, MSE-optimal bandwidth,
    robust bias-corrected inference. Returns a dict; writes a binned RD plot.
    Raises so the caller can degrade honestly."""
    import numpy as np
    from rdrobust import rdrobust

    sub = df[[outcome, running]].copy()
    sub[outcome] = sub[outcome].astype(float)
    sub[running] = sub[running].astype(float)
    sub = sub.dropna()
    y, x = sub[outcome], sub[running]
    r = rdrobust(y=y, x=x, c=float(cutoff))
    out = {
        "rd_robust": float(r.coef.loc["Robust", "Coeff"]),
        "rd_conventional": float(r.coef.loc["Conventional", "Coeff"]),
        "ci_lb": float(r.ci.loc["Robust", "CI Lower"]),
        "ci_ub": float(r.ci.loc["Robust", "CI Upper"]),
        "p_robust": float(r.pv.loc["Robust", "P>|z|"]),
        "bw_h": float(r.bws.loc["h", "left"]),
        "n_left": int(r.N_h[0]),
        "n_right": int(r.N_h[1]),
        "bwselect": str(r.bwselect),
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = out["bw_h"]
        xv, yv = x.to_numpy(), y.to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        # binned means within ~3 bandwidths each side of the cutoff
        for lo, hi, col in [(cutoff - 3 * h, cutoff, "#4C72B0"), (cutoff, cutoff + 3 * h, "#C44E52")]:
            sel = (xv >= lo) & (xv < hi) if lo < cutoff else (xv >= lo) & (xv <= hi)
            if sel.sum() > 5:
                edges = np.linspace(lo, hi, 11)
                idx = np.digitize(xv[sel], edges)
                bx = [xv[sel][idx == k].mean() for k in range(1, len(edges)) if (idx == k).any()]
                by = [yv[sel][idx == k].mean() for k in range(1, len(edges)) if (idx == k).any()]
                ax.scatter(bx, by, s=26, c=col)
        ax.axvline(cutoff, color="grey", ls="--", lw=1)
        ax.set_xlabel(f"{running} (cutoff = {cutoff:g})")
        ax.set_ylabel(f"{outcome} (binned mean)")
        ax.set_title(f"Regression discontinuity — jump = {out['rd_robust']:.3f}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return out


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


def _glmm_via_r(csv_path, outcome: str, predictors: list[str], group: str, family: str):
    """Generalized linear mixed model via R lme4::glmer — a fixed-effect part plus
    a random intercept (1|group), for a binary (family="binomial", logit) or count
    (family="poisson", log) outcome. Returns (fixed_df[term,estimate,se,z,p],
    re dict{group_var, group_sd, icc, n_groups, n_obs, aic, overdispersion}).
    Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(predictors) or "1"
    fam = "binomial" if family == "binomial" else "poisson"
    rcode = (
        "suppressMessages(library(lme4))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'm <- glmer({outcome} ~ {rhs} + (1|{group}), data=d, family={fam})\n'
        "s <- summary(m); ct <- s$coefficients\n"
        'cat("##FX\\n")\n'
        'for (i in seq_len(nrow(ct))) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", '
        "rownames(ct)[i], ct[i,1], ct[i,2], ct[i,4]))\n"
        "vc <- as.data.frame(VarCorr(m)); gv <- vc$vcov[1]\n"
        # latent-scale ICC: binomial logit residual var = pi^2/3; poisson left as NA
        f'icc <- if ("{fam}" == "binomial") gv/(gv + (pi^2)/3) else NA\n'
        "od <- sum(residuals(m, type='pearson')^2) / df.residual(m)\n"
        'cat("##RE\\n")\n'
        'cat(sprintf("group_var|%.6f\\ngroup_sd|%.6f\\n", gv, sqrt(gv)))\n'
        'if (!is.na(icc)) cat(sprintf("icc|%.6f\\n", icc))\n'
        'cat(sprintf("n_groups|%d\\nn_obs|%d\\naic|%.4f\\noverdispersion|%.4f\\n", '
        "as.integer(ngrps(m)[1]), as.integer(nobs(m)), AIC(m), od))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, fx, re_d = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##FX":
            section = "FX"
        elif s == "##RE":
            section = "RE"
        elif "|" in s and section == "FX":
            fx.append(s.rsplit("|", 3))
        elif "|" in s and section == "RE":
            k, v = s.split("|", 1)
            re_d[k] = float(v)
    if not fx or "group_var" not in re_d:
        raise RuntimeError("lme4 glmer 未返回结果（可能不收敛）")
    fixed = pd.DataFrame(fx, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        fixed[c] = pd.to_numeric(fixed[c], errors="coerce")
    return fixed, re_d


def _gamm_via_r(csv_path, outcome, smooth_terms, linear_terms, group, png_path):
    """Generalized additive mixed model via R mgcv: GAM smooth (penalised-spline)
    terms s(x) for continuous predictors + a random intercept for `group` via the
    "re" smooth basis s(group, bs="re") (Wood's single-object GAMM, REML). Returns
    (smooth_df[term,edf,F,p] for covariate smooths, param_df, re dict{edf,p,sd},
    fit{dev_expl,r_sq,n}). Writes partial-effect plots for the covariate smooths.
    Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(png_path).replace("\\", "/")
    re_term = f's({group}, bs="re")'
    rhs = " + ".join([f"s({t})" for t in smooth_terms] + list(linear_terms) + [re_term])
    n_panels = max(1, len(smooth_terms))
    rcode = (
        "suppressMessages(library(mgcv))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'd[["{group}"]] <- as.factor(d[["{group}"]])\n'
        f'm <- gam({outcome} ~ {rhs}, data=d, method="REML")\n'
        "s <- summary(m)\n"
        f're_row <- "s({group})"\n'
        'cat("##S\\n")\n'
        "if (!is.null(s$s.table)) for (i in seq_len(nrow(s$s.table))) "
        "if (rownames(s$s.table)[i] != re_row) "
        'cat(sprintf("%s|%.4f|%.4f|%.6g\\n", rownames(s$s.table)[i], '
        's$s.table[i,"edf"], s$s.table[i,"F"], s$s.table[i,"p-value"]))\n'
        'cat("##P\\n")\n'
        "if (!is.null(s$p.table)) for (i in seq_len(nrow(s$p.table))) "
        'cat(sprintf("%s|%.6f|%.6f|%.6g\\n", rownames(s$p.table)[i], '
        "s$p.table[i,1], s$p.table[i,2], s$p.table[i,4]))\n"
        'cat("##R\\n")\n'
        "ri <- which(rownames(s$s.table) == re_row)\n"
        'if (length(ri)) cat(sprintf("re_edf|%.4f\\nre_p|%.6g\\n", s$s.table[ri,"edf"], s$s.table[ri,"p-value"]))\n'
        # random-effect SD closed form for bs="re": sqrt(scale / smoothing-param)
        f're_sd <- tryCatch(as.numeric(sqrt(m$sig2 / m$sp["s({group})"])), error=function(e) NA)\n'
        'if (!is.na(re_sd)) cat(sprintf("re_sd|%.6f\\n", re_sd))\n'
        'cat("##F\\n")\n'
        'cat(sprintf("dev_expl|%.4f\\nr_sq|%.4f\\nn|%d\\n", s$dev.expl, s$r.sq, s$n))\n'
        # plot only the covariate smooths (skip the RE term, which is select index n_panels+1)
        f'png("{png_r}", width=900, height=max(350, 320*ceiling({n_panels}/2)), res=120)\n'
        f"par(mfrow=c(ceiling({n_panels}/2), min(2,{n_panels})))\n"
        f"for (i in 1:{n_panels}) plot(m, select=i, shade=TRUE, seWithMean=TRUE)\n"
        "dev.off()\n"
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, srows, prows, re_d, fit = None, [], [], {}, {}
    for line in out.splitlines():
        s = line.strip()
        if s in ("##S", "##P", "##R", "##F"):
            section = s
        elif "|" in s and section == "##S":
            srows.append(s.rsplit("|", 3))
        elif "|" in s and section == "##P":
            prows.append(s.rsplit("|", 3))
        elif "|" in s and section == "##R":
            k, v = s.split("|", 1)
            re_d[k] = float(v)
        elif "|" in s and section == "##F":
            k, v = s.split("|", 1)
            fit[k] = float(v)
    if "dev_expl" not in fit:
        raise RuntimeError("mgcv GAMM 未返回结果")
    smooth_df = pd.DataFrame(srows, columns=["term", "edf", "F", "p_value"]) if srows else pd.DataFrame(columns=["term", "edf", "F", "p_value"])
    param_df = pd.DataFrame(prows, columns=["term", "estimate", "std_err", "p_value"]) if prows else pd.DataFrame(columns=["term", "estimate", "std_err", "p_value"])
    for col in ("edf", "F", "p_value"):
        if len(smooth_df):
            smooth_df[col] = pd.to_numeric(smooth_df[col], errors="coerce")
    for col in ("estimate", "std_err", "p_value"):
        if len(param_df):
            param_df[col] = pd.to_numeric(param_df[col], errors="coerce")
    return smooth_df, param_df, re_d, fit


def _bart_via_r(csv_path, outcome: str, predictors: list[str], ntree: int, seed: int, png_path):
    """Bayesian Additive Regression Trees (Chipman et al.) via R dbarts: a
    sum-of-trees Bayesian nonparametric regression for a continuous outcome.
    Returns (metrics dict: r2/rmse/sigma/n, variable-importance DataFrame from
    posterior split counts). Writes a variable-importance bar plot. Raises so the
    caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(png_path).replace("\\", "/")
    preds_r = ", ".join(f'"{p}"' for p in predictors)
    rcode = (
        "suppressMessages(library(dbarts))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"preds <- c({preds_r})\n"
        f'y <- d[["{outcome}"]]; X <- d[, preds, drop=FALSE]\n'
        f"set.seed({int(seed)})\n"
        f"fit <- bart(x.train=X, y.train=y, ntree={int(ntree)}, keeptrees=FALSE, verbose=FALSE)\n"
        "yhat <- fit$yhat.train.mean\n"
        "r2 <- 1 - sum((y-yhat)^2)/sum((y-mean(y))^2); rmse <- sqrt(mean((y-yhat)^2))\n"
        'cat("##M\\n")\n'
        'cat(sprintf("r2|%.6f\\nrmse|%.6f\\nsigma|%.6f\\nn|%d\\n", r2, rmse, mean(fit$sigma), length(y)))\n'
        # label from the columns dbarts ACTUALLY used (it drops constant cols / expands
        # factors), NOT the input preds vector — else labels misalign (Opus catch).
        "vc <- colMeans(fit$varcount); vc <- vc/sum(vc); nm <- colnames(fit$varcount)\n"
        'cat("##V\\n")\n'
        "ord <- order(vc, decreasing=TRUE)\n"
        'for (i in ord) cat(sprintf("%s|%.6f\\n", nm[i], vc[i]))\n'
        f'png("{png_r}", width=720, height=max(300, 36*length(nm)), res=120)\n'
        "par(mar=c(4,8,2,1)); barplot(rev(vc[ord]), names.arg=rev(nm[ord]), horiz=TRUE, las=1, "
        'col="#4C72B0", xlab="variable split share", main="BART variable importance")\n'
        "dev.off()\n"
    )
    out = rbridge.run_r(rcode, timeout=240)
    section, meta, vrows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##M":
            section = "M"
        elif s == "##V":
            section = "V"
        elif "|" in s and section == "M":
            k, v = s.split("|", 1)
            meta[k] = float(v)
        elif "|" in s and section == "V":
            vrows.append(s.rsplit("|", 1))
    if "r2" not in meta or not vrows:
        raise RuntimeError("dbarts BART 未返回结果")
    varimp = pd.DataFrame(vrows, columns=["predictor", "split_share"])
    varimp["split_share"] = pd.to_numeric(varimp["split_share"], errors="coerce")
    return meta, varimp


def _gam_via_r(csv_path, outcome: str, smooth_terms: list[str], linear_terms: list[str], png_path):
    """Generalized additive model via R mgcv (gold standard): smooth (penalised
    spline) terms s(x) for continuous predictors, parametric terms for the rest,
    fit by REML. Returns (smooth_df[term,edf,F,p], param_df[term,est,se,p],
    fit{dev_expl,r_sq,n}). Writes the partial-effect smooth panels to png_path.
    Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(png_path).replace("\\", "/")
    rhs = " + ".join([f"s({t})" for t in smooth_terms] + list(linear_terms)) or "1"
    n_panels = max(1, len(smooth_terms))
    rcode = (
        "suppressMessages(library(mgcv))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'm <- gam({outcome} ~ {rhs}, data=d, method="REML")\n'
        "s <- summary(m)\n"
        'cat("##S\\n")\n'
        "if (!is.null(s$s.table)) for (i in seq_len(nrow(s$s.table))) "
        'cat(sprintf("%s|%.4f|%.4f|%.6g\\n", rownames(s$s.table)[i], '
        's$s.table[i,"edf"], s$s.table[i,"F"], s$s.table[i,"p-value"]))\n'
        'cat("##P\\n")\n'
        "if (!is.null(s$p.table)) for (i in seq_len(nrow(s$p.table))) "
        'cat(sprintf("%s|%.6f|%.6f|%.6g\\n", rownames(s$p.table)[i], '
        "s$p.table[i,1], s$p.table[i,2], s$p.table[i,4]))\n"
        'cat("##F\\n")\n'
        'cat(sprintf("dev_expl|%.4f\\nr_sq|%.4f\\nn|%d\\n", s$dev.expl, s$r.sq, s$n))\n'
        # worst-case concurvity (nonlinear analogue of collinearity); deterministic,
        # no refit. >~0.8 means a smooth is well-approximated by the others (Opus).
        'cw <- tryCatch(max(concurvity(m, full=TRUE)["worst", ], na.rm=TRUE), error=function(e) NA)\n'
        'if (!is.na(cw)) cat(sprintf("concurvity|%.4f\\n", cw))\n'
        f'png("{png_r}", width=900, height=max(350, 320*ceiling({n_panels}/2)), res=120)\n'
        f"par(mfrow=c(ceiling({n_panels}/2), min(2,{n_panels})))\n"
        "plot(m, shade=TRUE, seWithMean=TRUE, residuals=FALSE); dev.off()\n"
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, srows, prows, fit = None, [], [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##S":
            section = "S"
        elif s == "##P":
            section = "P"
        elif s == "##F":
            section = "F"
        elif "|" in s and section == "S":
            srows.append(s.rsplit("|", 3))
        elif "|" in s and section == "P":
            prows.append(s.rsplit("|", 3))
        elif "|" in s and section == "F":
            k, v = s.split("|", 1)
            fit[k] = float(v)
    if "dev_expl" not in fit:
        raise RuntimeError("mgcv gam 未返回结果")
    smooth_df = pd.DataFrame(srows, columns=["term", "edf", "F", "p_value"]) if srows else pd.DataFrame(
        columns=["term", "edf", "F", "p_value"]
    )
    param_df = pd.DataFrame(prows, columns=["term", "estimate", "std_err", "p_value"]) if prows else pd.DataFrame(
        columns=["term", "estimate", "std_err", "p_value"]
    )
    for col in ("edf", "F", "p_value"):
        if len(smooth_df):
            smooth_df[col] = pd.to_numeric(smooth_df[col], errors="coerce")
    for col in ("estimate", "std_err", "p_value"):
        if len(param_df):
            param_df[col] = pd.to_numeric(param_df[col], errors="coerce")
    return smooth_df, param_df, fit


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


def _qca_necessity_via_r(csv_path, outcome: str, conditions: list[str], anchors=(0.1, 0.5, 0.9), incl_cut=0.9):
    """QCA necessity analysis via R superSubset on fuzzy-calibrated data. Returns
    a DataFrame [expression, inclN(consistency), RoN, covN(coverage)]. RoN flags
    trivially-necessary (always-high) conditions. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x) {\n"
        f"  thr <- as.numeric(quantile(x, c({a0},{a1},{a2}), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'ss <- superSubset(cal, outcome="{outcome}", conditions=c({conds_r}), '
        f"incl.cut={float(incl_cut)}, cov.cut=0.5)\n"
        "ic <- ss$incl.cov\n"
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclN[i], ic$RoN[i], ic$covN[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    rows = [s.split("|") for s in out.splitlines() if "|" in s and not s.strip().startswith("#")]
    if not rows:
        raise RuntimeError("superSubset 未返回必要条件（无满足一致性阈值的必要项）")
    tab = pd.DataFrame(rows, columns=["expression", "consistency_inclN", "RoN", "coverage_covN"])
    for c in ("consistency_inclN", "RoN", "coverage_covN"):
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return tab
