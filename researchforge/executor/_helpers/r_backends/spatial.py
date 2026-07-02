"""R-bridge spatial delegators (gstat kriging, spdep spatial regression)."""

from __future__ import annotations


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


