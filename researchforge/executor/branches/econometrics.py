"""Branch handlers for the econometrics family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _dynamic_gmm_via_r,
    _gmm_lags,
    _knn_k,
)


@register("dynamic_panel_gmm")
def _branch_dynamic_panel_gmm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    if not (fp.unit_col and fp.time_col):
        summary.append("动态面板 GMM 失败：需要面板数据（单位列 + 时间列）。")
    else:
        y = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
            None,
        )
        preds = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:5]
        names_safe = y is not None and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c))
            for c in [y, *preds, fp.unit_col, fp.time_col]
        )
        n_periods = int(df[fp.time_col].nunique())
        if y is None or not preds:
            summary.append("动态面板 GMM 失败：需要连续结果变量 + ≥1 个预测变量。")
        elif n_periods < 3:
            summary.append("动态面板 GMM 失败：需要 ≥3 个时间期（差分 GMM 与 AR 检验要求）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("plm")):
            summary.append(
                "动态面板 GMM 需要 R 的 plm 包（未检测到）。安装：install.packages('plm')；"
                "或用 panel_fixed_effects / random_effects。"
            )
        elif not names_safe:
            summary.append("动态面板 GMM 失败：列名需为标识符式（字母/数字/. _）。")
        else:
            sub = df[[fp.unit_col, fp.time_col, y, *preds]].dropna()
            csv = d / "_gmm_input.csv"
            sub.to_csv(csv, index=False)
            try:
                import numpy as np

                _endo = [p for p in (cfg.get("endogenous") or []) if p in preds]
                _lo, _hi = _gmm_lags(cfg)
                coef, diag = _dynamic_gmm_via_r(
                    csv, fp.unit_col, fp.time_col, y, preds,
                    endogenous=_endo, gmm_lags=(_lo, _hi),
                )
                coef["term"] = [
                    f"lag_{y}" if str(t).startswith("lag(") else str(t) for t in coef["term"]
                ]
                coef.to_csv(d / "gmm_coefficients.csv", index=False, encoding="utf-8")
                files.append("gmm_coefficients.csv")
                sargan_p = diag.get("sargan_p", float("nan"))
                ar1_p = diag.get("ar1_p", float("nan"))
                ar2_p = diag.get("ar2_p", float("nan"))
                sargan_ok = sargan_p > 0.05
                ar2_ok = ar2_p > 0.05
                _endo_note = (
                    f"内生变量（用 lag {_lo}:{_hi} 工具）：{_endo}"
                    if _endo
                    else "全部协变量设为严格外生（可用 config endogenous 标出内生变量）"
                )
                if _lo < 2:
                    _endo_note += (
                        "；⚠ gmm_lags 起始<2 仅对前定(predetermined)变量有效，"
                        "滞后被解释变量已强制 lag≥2（差分方程中 lag1 为无效工具）"
                    )
                (d / "diagnostics.txt").write_text(
                    "动态面板 GMM (Arellano-Bond 差分 GMM, twosteps, Windmeijer 稳健 SE)\n"
                    f"工具集限 lag {_lo}-{_hi}（抑制工具过度增殖；多工具会抬高/弱化 Sargan）。\n"
                    f"{_endo_note}。\n"
                    f"Sargan 过度识别检验 p = {sargan_p:.4g}（注：Sargan 非稳健,非 Hansen J）"
                    f"（{'工具有效（不拒）' if sargan_ok else '被拒 → 工具集可疑'}）\n"
                    f"AR(1) p = {ar1_p:.4g}（差分后通常显著，正常）\n"
                    f"AR(2) p = {ar2_p:.4g} "
                    f"（{'无二阶自相关 → GMM 一致' if ar2_ok else '⚠ 有二阶自相关 → GMM 不一致'}）\n\n"
                    + coef.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("diagnostics.txt")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.errorbar(
                        coef["estimate"], range(len(coef)),
                        xerr=1.96 * coef["std_err"], fmt="o",
                    )
                    ax.axvline(0, color="grey", ls="--")
                    ax.set_yticks(range(len(coef)))
                    ax.set_yticklabels(coef["term"])
                    ax.set_xlabel("GMM coefficient (95% CI)")
                    ax.set_title("Dynamic panel GMM (Arellano-Bond)")
                    fig.tight_layout()
                    fig.savefig(d / "coefficients.png", dpi=150)
                    plt.close(fig)
                    files.append("coefficients.png")
                except Exception:
                    pass
                lag_rows = coef[coef["term"].str.startswith("lag_")]
                persistence = float(lag_rows.iloc[0]["estimate"]) if len(lag_rows) else float("nan")
                estimates["persistence_lag_coef"] = round(persistence, 4)
                estimates["sargan_p"] = round(sargan_p, 4)
                estimates["ar2_p"] = round(ar2_p, 4)
                for _, r in coef.iterrows():
                    if not str(r["term"]).startswith("lag_"):
                        estimates[str(r["term"])] = round(float(r["estimate"]), 4)
                valid = sargan_ok and ar2_ok
                summary.append(
                    f"{entry.method} 完成（R/plm，{n_periods} 期）：滞后被解释变量系数（持续性）"
                    f"={persistence:.3f}；Sargan p={sargan_p:.3g}"
                    f"（{'工具有效' if sargan_ok else '工具可疑'}），AR(2) p={ar2_p:.3g}"
                    f"（{'无二阶自相关' if ar2_ok else '⚠有二阶自相关、GMM不一致'}）"
                    f"{'' if valid else ' —— ⚠ 诊断未全通过,结果存疑'}。系数见 gmm_coefficients.csv。"
                    + (f"（内生变量 {_endo}、工具滞后 {_lo}:{_hi} 按 config 指定）" if (_endo or cfg.get("gmm_lags")) else "")
                )
                code += [
                    "library(plm)  # Arellano-Bond 差分 GMM",
                    f"# pgmm({y} ~ lag({y},1) + ... | lag({y},{_lo}:{_hi})[+ lag(endo)], model='twosteps')",
                ]
            except Exception as err:
                summary.append(f"动态面板 GMM 失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass



@register("random_effects")
def _branch_random_effects(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    if not (fp.unit_col and fp.time_col):
        summary.append("随机效应模型失败：需要面板数据（单位列 + 时间列）。")
    else:
        import numpy as np
        import pandas as pd
        from scipy.stats import chi2

        y = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
            None,
        )
        preds = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
        ][:6]
        if y is None or not preds:
            summary.append("随机效应模型失败：需要连续结果变量 + ≥1 个预测变量。")
        else:
            try:
                from linearmodels.panel import PanelOLS, RandomEffects

                dd = (
                    df[[fp.unit_col, fp.time_col, y, *preds]]
                    .dropna()
                    .drop_duplicates([fp.unit_col, fp.time_col])
                    .set_index([fp.unit_col, fp.time_col])
                )
                fe = PanelOLS(dd[y], dd[preds], entity_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                re = RandomEffects(dd[y], dd[preds].assign(const=1.0)).fit(
                    cov_type="clustered", cluster_entity=True
                )
                # classic Hausman needs the EFFICIENT (unadjusted) covariances so
                # V_FE - V_RE is PSD; clustered/robust covs break the χ² basis (Opus
                # catch). Use unadjusted fits for the test; keep clustered for SEs.
                fe_u = PanelOLS(dd[y], dd[preds], entity_effects=True).fit()
                re_u = RandomEffects(dd[y], dd[preds].assign(const=1.0)).fit()
                common = [p for p in preds if p in fe.params.index and p in re.params.index]
                diff = (fe_u.params[common] - re_u.params[common]).values
                vdiff = (fe_u.cov.loc[common, common] - re_u.cov.loc[common, common]).values
                h_stat = max(0.0, float(diff @ np.linalg.pinv(vdiff) @ diff))
                h_p = float(chi2.sf(h_stat, len(common)))
                use_fe = h_p < 0.05
                rec = "FE（固定效应）" if use_fe else "RE（随机效应）"
                tab = pd.DataFrame(
                    {
                        "term": common,
                        "FE_coef": [round(float(fe.params[c]), 4) for c in common],
                        "FE_p": [round(float(fe.pvalues[c]), 4) for c in common],
                        "RE_coef": [round(float(re.params[c]), 4) for c in common],
                        "RE_p": [round(float(re.pvalues[c]), 4) for c in common],
                    }
                )
                tab.to_csv(d / "fe_re_coefficients.csv", index=False, encoding="utf-8")
                files.append("fe_re_coefficients.csv")
                chosen = fe if use_fe else re
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    se = chosen.std_errors[common]
                    co = chosen.params[common]
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.errorbar(co.values, range(len(common)), xerr=1.96 * se.values, fmt="o")
                    ax.axvline(0, color="grey", ls="--")
                    ax.set_yticks(range(len(common)))
                    ax.set_yticklabels(common)
                    ax.set_xlabel(f"{rec[:2]} coefficient (95% CI)")
                    fig.tight_layout()
                    fig.savefig(d / "coefficients.png", dpi=150)
                    plt.close(fig)
                    files.append("coefficients.png")
                except Exception:
                    pass
                estimates["hausman_stat"] = round(h_stat, 4)
                estimates["hausman_p"] = round(h_p, 4)
                for c in common:
                    estimates[c] = round(float(chosen.params[c]), 4)
                summary.append(
                    f"{entry.method} 完成：面板 {dd.index.get_level_values(0).nunique()} 单位 × "
                    f"{dd.index.get_level_values(1).nunique()} 期；结果 {y}，{len(common)} 个预测变量。"
                    f"Hausman H={h_stat:.3f}, p={h_p:.3g} → 推荐 {rec}"
                    f"（p<0.05 表示随机效应与回归元相关、RE 不一致，应用 FE）。系数对比见 fe_re_coefficients.csv（聚类稳健 SE）。"
                )
                code += [
                    "from linearmodels.panel import PanelOLS, RandomEffects  # 面板 RE + Hausman",
                    "# FE=PanelOLS(entity_effects); RE=RandomEffects; H=(b_fe-b_re)'pinv(Vfe-Vre)(b_fe-b_re)",
                ]
            except Exception as err:
                summary.append(f"随机效应模型失败：{err}")


def _spatial_panel_via_r(
    csv_path, unit: str, time: str, outcome: str, predictors: list[str],
    lon: str, lat: str, model: str = "lag", k: int = 6,
):
    """Spatial panel model via R splm (Millo-Piras, the gold standard). Builds a
    row-standardised k-nearest-neighbour W from each unit's (lon,lat), then fits a
    fixed-effects ("within", individual) spatial panel:

      * model="lag"   -> SAR : y = rho*W y + X b + mu + eps         (spml lag=TRUE)
      * model="error" -> SEM : y = X b + u, u = lambda*W u + eps    (spml spatial.error="b")
      * model="sdm"   -> SDM : SAR + spatially-lagged X (W x)        (Durbin terms)

    Reports rho (lag/sdm) or lambda (error), and for SAR/SDM the direct / indirect
    (spillover) / total impacts. We compute the LeSage-Pace impacts ANALYTICALLY
    from the fitted rho + betas (+ Durbin thetas for SDM): the effects matrix for
    regressor k is S_k = (I - rho W)^{-1} (I b_k + W theta_k); average direct =
    mean(diag S_k), average total = mean(rowSums S_k), indirect = total - direct.
    This is exact and version-robust (splm's own impacts() is fragile across
    spdep/spatialreg versions: e.g. trW / as_dgRMatrix_listw moved/renamed). LM
    tests (slmtest lml / lme) on the pooled model advise lag-vs-error. Column names
    are identifier-guarded upstream. SDM's spatially-lagged regressors are
    pre-computed as W%*%x columns per period (splm has no Durbin switch and inline
    slag() needs a panel index). Returns (coef_df, diag dict, impacts_df or None).
    Raises so the caller can degrade honestly. No network in R."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    preds_r = ", ".join(f'"{p}"' for p in predictors)
    is_lag = model in ("lag", "sdm")
    is_sdm = model == "sdm"
    sperr = '"none"' if is_lag else '"b"'
    lag_flag = "TRUE" if is_lag else "FALSE"
    # SDM = SAR plus spatially-lagged regressors (W x). splm has no Durbin switch and
    # inline slag() inside the formula needs a panel index, so we pre-build columns
    # Wlag_<p> = (W %*% x) per time period and add them to the RHS (the standard route).
    wlag_names = [f"Wlag_{p}" for p in predictors] if is_sdm else []
    rhs = " + ".join(predictors + wlag_names)
    rcode = (
        # plm provides pdata.frame/index (splm imports but does not attach it)
        "suppressMessages({library(splm); library(spdep); library(plm)})\n"
        f'd <- read.csv("{csv_r}")\n'
        # one (lon,lat) per unit -> W on the unit cross-section, in unit order
        f'uc <- unique(d[, c("{unit}","{lon}","{lat}")])\n'
        f'uc <- uc[order(uc[["{unit}"]]), ]\n'
        f'uord <- uc[["{unit}"]]\n'
        f'coords <- as.matrix(uc[, c("{lon}","{lat}")])\n'
        f"nb <- knn2nb(knearneigh(coords, k=min({int(k)}, nrow(uc)-1)))\n"
        'lw <- nb2listw(nb, style="W")\n'
        "Wm <- listw2mat(lw)\n"  # dense row-standardised W in unit order
        # splm wants the data sorted unit-major, time-minor
        f'd <- d[order(d[["{unit}"]], d[["{time}"]]), ]\n'
        f"preds <- c({preds_r})\n"
    )
    if is_sdm:
        # build W%*%x per period (units within a period are reordered to W's unit order)
        rcode += (
            f'tms <- sort(unique(d[["{time}"]]))\n'
            "for (p in preds) {\n"
            '  wl <- rep(NA_real_, nrow(d))\n'
            "  for (tt in tms) {\n"
            f'    idx <- which(d[["{time}"]] == tt)\n'
            f'    ord <- match(uord, d[idx, "{unit}"])\n'  # rows of this period in unit order
            "    xv <- d[idx[ord], p]\n"
            "    wl[idx[ord]] <- as.numeric(Wm %*% xv)\n"
            "  }\n"
            '  d[[paste0("Wlag_", p)]] <- wl\n'
            "}\n"
        )
    rcode += (
        f'pd <- pdata.frame(d, index=c("{unit}","{time}"))\n'
        f'f <- as.formula("{outcome} ~ {rhs}")\n'
        # individual fixed effects ("within"); SAR/SEM/SDM per flags above
        f'm <- spml(f, data=pd, listw=lw, model="within", '
        f"spatial.error={sperr}, lag={lag_flag})\n"
        "s <- summary(m)\n"
        'cat("##COEF\\n")\n'
        "ct <- s$CoefTable\n"
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,4]))\n'
        'cat("##DIAG\\n")\n'
        # splm naming is version-dependent: the SAR lag parameter is reported as
        # "lambda" in m$coefficients, the SEM error parameter as "rho" in m$errcomp.
        # Grab both robustly and tag by the model the caller requested.
        'getp <- function(v, nms) { for (n in nms) if (!is.null(v) && n %in% names(v)) '
        'return(as.numeric(v[n])); NA }\n'
        'sp <- m$coefficients; spe <- m$errcomp\n'
        'lagp <- getp(sp, c("lambda","rho","psi"))\n'
        'errp <- getp(spe, c("rho","lambda","phi"))\n'
        'if (is.na(errp)) errp <- getp(sp, c("rho_err","lambda_err"))\n'
        'cat(sprintf("lag_param|%.6f\\n", ifelse(is.na(lagp), 0, lagp)))\n'
        'cat(sprintf("err_param|%.6f\\n", ifelse(is.na(errp), 0, errp)))\n'
        # LM tests on the pooled (OLS) model to advise lag vs error
        f'fb <- as.formula("{outcome} ~ {" + ".join(predictors)}")\n'
        'lm_lag <- tryCatch(slmtest(fb, data=pd, listw=lw, test="lml")$p.value, error=function(e) NA)\n'
        'lm_err <- tryCatch(slmtest(fb, data=pd, listw=lw, test="lme")$p.value, error=function(e) NA)\n'
        'cat(sprintf("lm_lag_p|%.6g\\n", ifelse(is.na(lm_lag), -1, lm_lag)))\n'
        'cat(sprintf("lm_err_p|%.6g\\n", ifelse(is.na(lm_err), -1, lm_err)))\n'
    )
    if is_lag:
        # analytic LeSage-Pace impacts: S_k = (I - rho W)^{-1}(I b_k + W theta_k)
        rcode += (
            'cat("##IMPACTS\\n")\n'
            "rho_hat <- ifelse(is.na(lagp), 0, lagp)\n"
            "n <- nrow(Wm)\n"
            "A <- solve(diag(n) - rho_hat * Wm)\n"
            "for (p in preds) {\n"
            '  bk <- getp(sp, p); if (is.na(bk)) next\n'
            + (
                '  tk <- getp(sp, paste0("Wlag_", p)); if (is.na(tk)) tk <- 0\n'
                "  Sk <- A %*% (diag(n) * bk + Wm * tk)\n"
                if is_sdm
                else "  Sk <- A * bk\n"
            )
            + "  dir <- mean(diag(Sk)); tot <- mean(rowSums(Sk)); ind <- tot - dir\n"
            "  cat(sprintf('%s|%.6f|%.6f|%.6f\\n', p, dir, ind, tot))\n"
            "}\n"
        )
    out = rbridge.run_r(rcode, timeout=240)
    section, crows, diag, irows = None, [], {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##DIAG":
            section = "D"
        elif s == "##IMPACTS":
            section = "I"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))
        elif "|" in s and section == "D":
            kk, v = s.split("|", 1)
            diag[kk] = float(v)
        elif "|" in s and section == "I":
            irows.append(s.rsplit("|", 3))
    if not crows:
        raise RuntimeError("splm 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    impacts = None
    if irows:
        impacts = pd.DataFrame(irows, columns=["term", "direct", "indirect", "total"])
        for c in ("direct", "indirect", "total"):
            impacts[c] = pd.to_numeric(impacts[c], errors="coerce")
    return coef, diag, impacts


@register("spatial_panel")
def _branch_spatial_panel(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    # --- column roles: config overrides, else auto-detect ---------------------
    unit = cfg.get("unit") or fp.unit_col
    time = cfg.get("time") or fp.time_col
    geo = [c.name for c in fp.columns if c.kind == "geo"]
    lon = cfg.get("lon") or next(
        (g for g in geo if "lon" in g.lower() or "lng" in g.lower()), geo[-1] if geo else None
    )
    lat = cfg.get("lat") or next(
        (g for g in geo if "lat" in g.lower()), (geo[0] if geo else None)
    )
    _exc = {unit, time, lon, lat}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _exc]
    outcome = cfg.get("outcome") or (cont[0] if cont else None)
    predictors = cfg.get("predictors") or [
        c.name for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in _exc and c.name != outcome
    ][:5]
    model = str(cfg.get("model") or "lag").lower()
    if model not in {"lag", "error", "sdm"}:
        model = "lag"

    names_safe = (
        outcome is not None and unit and time and lon and lat
        and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c))
            for c in [outcome, *predictors, unit, time, lon, lat]
        )
    )
    # splm needs plm for pdata.frame/index; require both
    have_r = (
        rbridge.r_available()
        and rbridge.r_package_available("splm")
        and rbridge.r_package_available("plm")
    )

    if not (unit and time):
        summary.append("空间面板失败：需要面板数据（单位列 + 时间列）。可用 panel_fixed_effects。")
        return
    if outcome is None or not predictors:
        summary.append("空间面板失败：需要连续结果变量 + ≥1 个预测变量。")
        return
    if not (lon and lat):
        summary.append(
            "空间面板失败：需要每个单位的经纬度（config lon/lat 指定）。无坐标可用 panel_fixed_effects。"
        )
        return
    if not have_r:
        summary.append(
            "空间面板（SAR/SEM/SDM）需要 R 的 splm 包（未检测到）。"
            "安装：install.packages('splm')。"
            "无 R 时可用 spatial_regression（截面 SAR/SEM）或 panel_fixed_effects（非空间面板）。"
        )
        return
    if not names_safe:
        summary.append("空间面板失败：列名需为标识符式（字母/数字/. _）。")
        return

    k = _knn_k(cfg, max(1, int(df[unit].nunique()) - 1), default=6)
    sub = df[[unit, time, lon, lat, outcome, *predictors]].dropna()
    csv = d / "_spml_input.csv"
    sub.to_csv(csv, index=False)
    try:
        coef, diag, impacts = _spatial_panel_via_r(
            csv, unit, time, outcome, predictors, lon, lat, model=model, k=k
        )
        coef.to_csv(d / "spatial_panel_coefficients.csv", index=False, encoding="utf-8")
        files.append("spatial_panel_coefficients.csv")
        is_lag = model in ("lag", "sdm")
        model_name = {"lag": "SAR（空间滞后）", "error": "SEM（空间误差）", "sdm": "SDM（空间杜宾）"}[model]
        # splm names are version-dependent; the R helper already disambiguated into
        # lag_param (SAR/SDM autoregressive ρ) and err_param (SEM error λ).
        sp_coef = diag.get("lag_param", 0.0) if is_lag else diag.get("err_param", 0.0)
        sp_name = "ρ (空间滞后)" if is_lag else "λ (空间误差)"
        lm_lag_p = diag.get("lm_lag_p", -1.0)
        lm_err_p = diag.get("lm_err_p", -1.0)
        lm_note = ""
        if lm_lag_p >= 0 and lm_err_p >= 0:
            advise = (
                "→ LM 检验偏向 lag（空间滞后）" if lm_lag_p < lm_err_p
                else "→ LM 检验偏向 error（空间误差）"
            )
            lm_note = f"LM(lag) p={lm_lag_p:.3g}, LM(error) p={lm_err_p:.3g} {advise}"
        else:
            lm_note = "LM 检验不可用（splm 版本/数据所限），默认按 config model 拟合"

        if impacts is not None:
            impacts.to_csv(d / "spatial_panel_impacts.csv", index=False, encoding="utf-8")
            files.append("spatial_panel_impacts.csv")

        diag_lines = [
            f"空间面板模型：{model_name}（个体固定效应 within，k-NN 空间权重 k={k}，行标准化）",
            f"{sp_name} = {sp_coef:.4f}"
            + ("（>0：正空间依赖/集聚）" if sp_coef > 0 else "（<=0：负/无空间依赖）"),
            lm_note,
            "",
            "系数表（FE within）：",
            coef.to_string(index=False),
        ]
        if impacts is not None:
            diag_lines += [
                "",
                "效应分解（impacts，LeSage-Pace 解析式 S=(I-ρW)^-1(Ib+Wθ)）："
                "direct 直接 / indirect 溢出(spillover) / total 总效应",
                "⚠ 空间滞后/杜宾模型的系数本身不是边际效应，应看 total/direct/indirect；"
                "此处报点估计（解析均值），未附 impacts 的模拟 SE。",
                impacts.to_string(index=False),
            ]
        (d / "diagnostics.txt").write_text("\n".join(diag_lines), encoding="utf-8")
        files.append("diagnostics.txt")

        # --- plot (best-effort, English labels) -------------------------------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            if impacts is not None and len(impacts):
                ind = range(len(impacts))
                ax.barh([i + 0.2 for i in ind], impacts["direct"], height=0.4,
                        color="#4C72B0", label="direct")
                ax.barh([i - 0.2 for i in ind], impacts["indirect"], height=0.4,
                        color="#DD8452", label="indirect (spillover)")
                ax.set_yticks(list(ind))
                ax.set_yticklabels(impacts["term"])
                ax.set_xlabel("impact on outcome")
                ax.legend(fontsize=8)
                ax.set_title(f"{model.upper()} panel direct vs indirect effects")
            else:
                cp = coef[~coef["term"].str.contains("Intercept", case=False)]
                ax.errorbar(cp["estimate"], range(len(cp)),
                            xerr=1.96 * cp["std_err"], fmt="o")
                ax.set_yticks(range(len(cp)))
                ax.set_yticklabels(cp["term"])
                ax.set_xlabel(f"{model.upper()} panel coefficient (95% CI)")
                ax.set_title(f"{model.upper()} spatial panel (FE) — {outcome}")
            ax.axvline(0, color="grey", ls="--")
            fig.tight_layout()
            fig.savefig(d / "coefficients.png", dpi=150)
            plt.close(fig)
            files.append("coefficients.png")
        except Exception:
            pass

        # --- estimates --------------------------------------------------------
        estimates["spatial_coef"] = round(float(sp_coef), 4)
        estimates["rho" if is_lag else "lambda"] = round(float(sp_coef), 4)
        if lm_lag_p >= 0:
            estimates["lm_lag_p"] = round(lm_lag_p, 4)
        if lm_err_p >= 0:
            estimates["lm_err_p"] = round(lm_err_p, 4)
        if impacts is not None:
            for _, r in impacts.iterrows():
                estimates[f"{r['term']}_total"] = round(float(r["total"]), 4)
                estimates[f"{r['term']}_indirect"] = round(float(r["indirect"]), 4)
        else:
            for _, r in coef.iterrows():
                if "Intercept" not in str(r["term"]):
                    estimates[str(r["term"])] = round(float(r["estimate"]), 4)

        n_units = int(sub[unit].nunique())
        n_time = int(sub[time].nunique())
        eff_note = (
            "已报 impacts（direct/indirect/total，溢出效应）" if impacts is not None
            else "SEM 系数即边际效应（无溢出分解）"
        )
        summary.append(
            f"{entry.method} 完成（R/splm，{model_name}，{n_units} 单位 × {n_time} 期，"
            f"个体固定效应）：{sp_name}={sp_coef:.3f}"
            f"（{'正空间依赖/集聚' if sp_coef > 0 else '负/无空间依赖'}）；{lm_note}。{eff_note}。"
            f"系数见 spatial_panel_coefficients.csv。"
            "\n⚠ 披露：(1) 假定存在空间依赖（同一时间相邻单位相关）；"
            "(2) 空间权重 W（这里 k-NN, k={k}, 行标准化）是建模选择，**会驱动结果**，"
            "换 k/距离阈值/邻接定义可能改变 ρ/λ 与溢出；"
            "(3) indirect=溢出效应，是经 W 反馈到其他单位的间接影响，**非结构性因果**，"
            "解释为「在该 W 与 FE 设定下的关联式溢出」；"
            "(4) within(个体FE) 吸收了时不变的单位异质，但**未控时变混淆**，"
            "且假定 W 外生、空间过程平稳；(5) 这是相关性/模型设定下的估计，非结构因果。".replace("{k}", str(k))
        )
        _lag_r = "TRUE" if is_lag else "FALSE"
        _sperr_r = '"none"' if is_lag else '"b"'
        _durbin_rhs = (
            " + " + " + ".join(f"Wlag_{p}" for p in predictors) if model == "sdm" else ""
        )
        _durbin_note = "  # Wlag_*=W%*%x 预算空间滞后回归元 (Durbin)" if model == "sdm" else ""
        code += [
            "library(splm); library(spdep)  # 空间面板 SAR/SEM/SDM, 个体FE (within)",
            f"# W=nb2listw(knn2nb(knearneigh(coords,k={k})),'W'); "
            f"spml({outcome} ~ {' + '.join(predictors)}{_durbin_rhs}, model='within', "
            f"lag={_lag_r}, spatial.error={_sperr_r}){_durbin_note}",
            "# impacts: S=(I-rho*W)^-1 (I*b + W*theta); direct=mean(diag(S)), total=mean(rowSums(S))",
        ]
    except Exception as err:
        summary.append(f"空间面板失败：{err}")
    finally:
        try:
            csv.unlink()
        except OSError:
            pass

