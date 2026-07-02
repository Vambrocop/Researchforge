"""R-bridge GLMM / GAM(M) / BART delegators (lme4, mgcv, dbarts)."""

from __future__ import annotations


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


def _gamm_via_r(csv_path, outcome, smooth_terms, linear_terms, group, png_path, family="gaussian"):
    """Generalized additive mixed model via R mgcv: GAM smooth (penalised-spline)
    terms s(x) for continuous predictors + a random intercept for `group` via the
    "re" smooth basis s(group, bs="re") (Wood's single-object GAMM, REML). `family`
    selects the response distribution/link: gaussian (identity), binomial (logit),
    or poisson (log). Returns (smooth_df[term,edf,stat,p] (stat=F gaussian / Chi.sq otherwise),
    param_df, re dict{edf,p,sd}, fit{dev_expl,r_sq,n}). Writes partial-effect plots
    for the covariate smooths. Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    family = family if family in {"gaussian", "binomial", "poisson"} else "gaussian"
    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(png_path).replace("\\", "/")
    re_term = f's({group}, bs="re")'
    rhs = " + ".join([f"s({t})" for t in smooth_terms] + list(linear_terms) + [re_term])
    n_panels = max(1, len(smooth_terms))
    rcode = (
        "suppressMessages(library(mgcv))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'd[["{group}"]] <- as.factor(d[["{group}"]])\n'
        f'm <- gam({outcome} ~ {rhs}, data=d, family={family}, method="REML")\n'
        "s <- summary(m)\n"
        f're_row <- "s({group})"\n'
        'cat("##S\\n")\n'
        "if (!is.null(s$s.table)) for (i in seq_len(nrow(s$s.table))) "
        "if (rownames(s$s.table)[i] != re_row) "
        # col 3 = test stat: F (gaussian, estimated scale) or Chi.sq (binomial/poisson, fixed scale)
        'cat(sprintf("%s|%.4f|%.4f|%.6g\\n", rownames(s$s.table)[i], '
        's$s.table[i,"edf"], s$s.table[i,3], s$s.table[i,"p-value"]))\n'
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
    smooth_df = pd.DataFrame(srows, columns=["term", "edf", "stat", "p_value"]) if srows else pd.DataFrame(columns=["term", "edf", "stat", "p_value"])
    param_df = pd.DataFrame(prows, columns=["term", "estimate", "std_err", "p_value"]) if prows else pd.DataFrame(columns=["term", "estimate", "std_err", "p_value"])
    for col in ("edf", "stat", "p_value"):
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
        # honest out-of-sample R²: a single 80/20 holdout (dbarts x.test gives test predictions).
        # set.seed right before the split so it is reproducible regardless of prior RNG use.
        "hold_r2 <- NA; n_test <- 0L; n_all <- length(y)\n"
        f"if (n_all >= 25) {{ set.seed({int(seed)}); idx <- sample(n_all, floor(0.8*n_all));\n"
        f"  fit2 <- bart(x.train=X[idx,,drop=FALSE], y.train=y[idx], x.test=X[-idx,,drop=FALSE], ntree={int(ntree)}, keeptrees=FALSE, verbose=FALSE);\n"
        "  yte <- y[-idx]; sst <- sum((yte-mean(yte))^2);\n"
        "  if (sst > 0) { hold_r2 <- 1 - sum((yte-fit2$yhat.test.mean)^2)/sst; n_test <- length(yte) } }\n"
        'cat("##M\\n")\n'
        'cat(sprintf("r2|%.6f\\nrmse|%.6f\\nsigma|%.6f\\nn|%d\\n", r2, rmse, mean(fit$sigma), length(y)))\n'
        'if (!is.na(hold_r2)) cat(sprintf("holdout_r2|%.6f\\nn_test|%d\\n", hold_r2, n_test))\n'
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


