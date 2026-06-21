"""Branch handler for Geographically Weighted Regression (GWR).

Geographically Weighted Regression (Brunsdon, Fotheringham & Charlton 1996) fits a
LOCALLY weighted OLS at every observation location i, weighting each observation by a
spatial kernel of its distance to i:

    β_i = (Xᵀ W_i X)⁻¹ Xᵀ W_i y,    W_i = diag(kernel(d_ij, bw))

so the coefficients become *surfaces* over space, revealing spatial non-stationarity
in the relationship. The bandwidth is selected by minimising AICc (the standard GWR
criterion). This is a PURE-PYTHON (numpy/scipy) implementation — mgwr/spgwr/GWmodel are
NOT required.

Honest scope: GWR is EXPLORATORY, not confirmatory. Bandwidth choice drives smoothness;
local multicollinearity + multiple testing across locations inflate apparent variation;
lon/lat Euclidean distance is only an approximation of true planar distance.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Hand-rolled GWR core (kept module-local; numpy only).                        #
# The risky math lives here: local WLS, hat matrix tr(S), AICc, OLS limit.     #
# --------------------------------------------------------------------------- #
def _gwr_kernel(d, bw, kind):
    """Spatial weights for distances `d` (1-D array) at bandwidth `bw`.

    Gaussian:  w = exp(-0.5 (d/bw)²)                  (never exactly 0)
    Bisquare:  w = (1 - (d/bw)²)²  for d < bw, else 0 (compact support)

    bw must be > 0. Returns weights aligned with d.
    """
    import numpy as np

    bw = max(float(bw), 1e-12)
    r = d / bw
    if kind == "gaussian":
        return np.exp(-0.5 * r * r)
    # bisquare (default): compact support
    w = (1.0 - r * r) ** 2
    w[r >= 1.0] = 0.0
    return w


def _gwr_fit(coords, X, y, bw_vec, kind):
    """Fit GWR given a per-location bandwidth vector `bw_vec` (length n).

    Returns (betas, yhat, S_diag, trS, local_r2) where
      betas    : (n, p) local coefficients (row i = β_i)
      yhat     : (n,)   local fitted value at each focal point ŷ_i = x_i β_i
      S_diag   : (n,)   diagonal of the hat matrix  (s_ii)
      trS      : float  trace(S) = effective number of parameters
      local_r2 : (n,)   geographically-weighted local R² at each location

    Hat matrix row:  s_i = x_i (Xᵀ W_i X)⁻¹ Xᵀ W_i   so  ŷ_i = s_i y  and
    trace(S) = Σ_i s_ii. σ̂² = RSS/(n − trS) and AICc use trS as the effective df.
    """
    import numpy as np

    n, p = X.shape
    betas = np.empty((n, p), dtype=float)
    yhat = np.empty(n, dtype=float)
    s_diag = np.empty(n, dtype=float)
    local_r2 = np.empty(n, dtype=float)

    # pairwise euclidean distances once (n×n); n is moderate per CLAUDE guidance.
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt((diff * diff).sum(-1))

    for i in range(n):
        w = _gwr_kernel(dmat[i], bw_vec[i], kind)
        Xtw = X.T * w  # (p, n): each column j scaled by w_j
        XtWX = Xtw @ X  # (p, p)
        XtWy = Xtw @ y  # (p,)
        try:
            beta_i = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # rank-deficient local design (too few non-zero weights) -> least squares
            beta_i = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        betas[i] = beta_i
        yhat[i] = X[i] @ beta_i
        # hat row s_i = x_i (XtWX)^-1 Xtw ; we only need s_ii = (s_i)[i]
        try:
            ci = np.linalg.solve(XtWX, X[i])  # (XtWX)^-1 x_i  (p,)
        except np.linalg.LinAlgError:
            ci = np.linalg.lstsq(XtWX, X[i], rcond=None)[0]
        # full hat row s_i = (X (XtWX)^-1 x_i) ⊙ w ; s_ii is element i
        s_diag[i] = float((X[i] @ ci) * w[i])

        # geographically-weighted local R² at focal i
        wsum = w.sum()
        if wsum > 0:
            ybar_w = float((w * y).sum() / wsum)
            full_yhat = X @ beta_i  # local model's prediction everywhere
            tss = float((w * (y - ybar_w) ** 2).sum())
            rss = float((w * (y - full_yhat) ** 2).sum())
            local_r2[i] = 1.0 - rss / tss if tss > 0 else 0.0
        else:
            local_r2[i] = 0.0

    trS = float(s_diag.sum())
    return betas, yhat, s_diag, trS, local_r2


def _gwr_aicc(y, yhat, trS):
    """AICc for a GWR fit (Fotheringham et al. 2002, eqn 2.33):

        AICc = 2n·ln(σ̂) + n·ln(2π) + n·(n + trS) / (n − 2 − trS)

    with σ̂² = RSS/n  (the GWR convention uses RSS/n inside ln σ̂, while the
    effective-df penalty carries trS). Returns (aicc, sigma2_n, rss). Falls back
    to +inf when the penalty denominator is non-positive (overfit: trS ≥ n−2).
    """
    import numpy as np

    n = len(y)
    rss = float(((y - yhat) ** 2).sum())
    sigma2 = rss / n  # GWR ML-style estimate used inside the AICc log term
    if sigma2 <= 0:
        return float("inf"), sigma2, rss
    denom = n - 2.0 - trS
    if denom <= 0:
        return float("inf"), sigma2, rss
    aicc = (
        2.0 * n * np.log(np.sqrt(sigma2))
        + n * np.log(2.0 * np.pi)
        + n * (n + trS) / denom
    )
    return float(aicc), sigma2, rss


def _adaptive_bw_vec(coords, k):
    """Per-location adaptive bandwidth = distance to the k-th nearest neighbour.

    The sorted-distance row has self (distance 0) at column 0; the k-th nearest
    neighbour (self excluded) is therefore at index k. We clip bw to a positive
    floor so the kernel stays well-defined for coincident points.
    """
    import numpy as np

    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt((diff * diff).sum(-1))
    dsort = np.sort(dmat, axis=1)  # column 0 is self (distance 0)
    idx = min(k, dsort.shape[1] - 1)
    bw = dsort[:, idx].astype(float)
    floor = float(max(bw.max(), 1e-6))
    bw[bw <= 0] = floor  # coincident-point guard
    return bw


@register("gwr")
def _branch_gwr(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    # ---- resolve geo columns (lon/lat) ------------------------------------ #
    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    lon = cfg.get("lon") or next(
        (g for g in geo if "lon" in g.lower() or "lng" in g.lower()),
        geo[-1] if geo else None,
    )
    lat = cfg.get("lat") or next((g for g in geo if g != lon), geo[0] if geo else None)

    # ---- resolve outcome + predictors (continuous/binary) ----------------- #
    _exc = {fp.unit_col, fp.time_col, lon, lat}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _exc]
    feat = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "binary"} and c.name not in _exc
    ]
    outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    forced_pred = [
        c
        for c in (cfg.get("predictors") or [])
        if c in df.columns and c not in {outcome, lon, lat, fp.unit_col, fp.time_col}
    ]
    predictors = forced_pred[:6] if forced_pred else [c for c in feat if c != outcome][:6]

    # ---- honest gates ----------------------------------------------------- #
    if len(geo) < 2 or lon is None or lat is None:
        summary.append("GWR 失败：需要经纬度坐标（两个 geo 列）。")
        return
    if outcome is None or not predictors:
        summary.append("GWR 失败：需要一个连续结果变量 + ≥1 个连续/二值预测变量。")
        return

    sub = df[[lon, lat, outcome, *predictors]].dropna()
    n = len(sub)
    p = len(predictors) + 1  # + intercept
    # need enough points that the smallest local window can still fit p params
    if n < max(30, 3 * p):
        summary.append(
            f"GWR 失败：有效样本不足（n={n} < {max(30, 3 * p)}）——"
            "局部加权回归每个窗口需足够点拟合系数。"
        )
        return

    coords = sub[[lon, lat]].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    Xpred = sub[predictors].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), Xpred])  # design with intercept

    if float(y.var()) == 0:
        summary.append("GWR 失败：结果变量为常数（无可解释方差）。")
        return

    kernel = str(cfg.get("kernel", "bisquare")).lower()
    if kernel not in {"bisquare", "gaussian"}:
        kernel = "bisquare"

    # ---- global OLS (comparison baseline + non-stationarity reference SE) -- #
    # β_ols = (XᵀX)⁻¹ Xᵀy ; SE from σ²(XᵀX)⁻¹ with σ²=RSS/(n−p).
    XtX = X.T @ X
    beta_ols = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    ols_resid = y - X @ beta_ols
    ols_rss = float((ols_resid**2).sum())
    ols_sigma2 = ols_rss / max(n - p, 1)
    try:
        XtX_inv = np.linalg.inv(XtX)
        ols_se = np.sqrt(np.maximum(np.diag(XtX_inv) * ols_sigma2, 0.0))
    except np.linalg.LinAlgError:
        ols_se = np.full(p, np.nan)
    ols_ss_tot = float(((y - y.mean()) ** 2).sum())
    ols_r2 = 1.0 - ols_rss / ols_ss_tot if ols_ss_tot > 0 else 0.0

    # ---- bandwidth selection by AICc -------------------------------------- #
    # Fixed bandwidth path if cfg["bw"] is a positive number; else adaptive k.
    fixed_bw = cfg.get("bw")
    try:
        fixed_bw = float(fixed_bw) if fixed_bw is not None else None
    except (TypeError, ValueError):
        fixed_bw = None

    if fixed_bw is not None and fixed_bw > 0:
        bw_vec = np.full(n, fixed_bw, dtype=float)
        betas, yhat, s_diag, trS, local_r2 = _gwr_fit(coords, X, y, bw_vec, kernel)
        aicc, sigma2_n, rss = _gwr_aicc(y, yhat, trS)
        selected = {"mode": "fixed", "bw": fixed_bw, "aicc": aicc, "trS": trS}
        bw_report = f"固定带宽 bw={fixed_bw:.4g}"
    else:
        # adaptive: search a small grid of candidate k (neighbour counts), pick min AICc.
        forced_k = cfg.get("adaptive_k")
        try:
            forced_k = int(forced_k) if forced_k is not None else None
        except (TypeError, ValueError):
            forced_k = None
        if forced_k is not None:
            cand_k = [max(p + 2, min(forced_k, n - 1))]
        else:
            kmin = max(p + 2, int(0.10 * n))
            kmax = min(n - 1, int(0.80 * n))
            if kmax <= kmin:
                cand_k = [min(max(p + 2, n // 2), n - 1)]
            else:
                cand_k = sorted({int(round(v)) for v in np.linspace(kmin, kmax, 8)})
        best = None  # (aicc, k, fit_tuple, bw_vec)
        for k in cand_k:
            bw_vec_k = _adaptive_bw_vec(coords, k)
            fit_k = _gwr_fit(coords, X, y, bw_vec_k, kernel)
            aicc_k = _gwr_aicc(y, fit_k[1], fit_k[3])[0]  # fit_k[1]=yhat, fit_k[3]=trS
            if best is None or aicc_k < best[0]:
                best = (aicc_k, k, fit_k, bw_vec_k)
        aicc, sel_k, (betas, yhat, s_diag, trS, local_r2), bw_vec = best
        selected = {"mode": "adaptive", "k": sel_k, "aicc": aicc, "trS": trS}
        bw_report = (
            f"自适应带宽 k={sel_k} 近邻（中位带宽距离={np.median(bw_vec):.4g}，"
            f"候选 k={cand_k}）"
        )

    # ---- per-location coefficient surfaces -> CSV ------------------------- #
    import pandas as pd

    coef_cols = {lon: coords[:, 0], lat: coords[:, 1], "intercept": betas[:, 0].round(6)}
    for j, name in enumerate(predictors, start=1):
        coef_cols[f"beta_{name}"] = betas[:, j].round(6)
    coef_cols["local_r2"] = local_r2.round(4)
    pd.DataFrame(coef_cols).to_csv(
        d / "gwr_coefficients.csv", index=False, encoding="utf-8"
    )
    files.append("gwr_coefficients.csv")

    # ---- coefficient ranges + a DESCRIPTIVE wide-spread indicator ---------- #
    # NOT a formal non-stationarity test. We flag coefficients whose local IQR
    # exceeds 2·(global OLS SE) as having "wide local spread". This is a rough
    # exploratory indicator that TENDS TO OVER-FLAG: the local IQR carries the
    # sampling noise of small k-NN windows, while the global SE uses all n points,
    # so the two are mis-scaled and the flag fires even on stationary data. A
    # formal test (Monte-Carlo permutation / Leung et al. 2000) is NOT performed.
    range_rows = []
    nonstat_flags = {}
    for j, name in enumerate(predictors, start=1):
        col = betas[:, j]
        q1, med, q3 = (float(np.percentile(col, q)) for q in (25, 50, 75))
        iqr = q3 - q1
        se = float(ols_se[j]) if j < len(ols_se) else float("nan")
        nonstat = bool(np.isfinite(se) and se > 0 and iqr > 2.0 * se)
        nonstat_flags[name] = nonstat
        range_rows.append(
            {
                "predictor": name,
                "beta_min": round(float(col.min()), 6),
                "beta_q25": round(q1, 6),
                "beta_median": round(med, 6),
                "beta_q75": round(q3, 6),
                "beta_max": round(float(col.max()), 6),
                "local_iqr": round(iqr, 6),
                "global_ols_beta": round(float(beta_ols[j]), 6),
                "global_ols_se": round(se, 6),
                "wide_local_spread": nonstat,  # descriptive (over-flags); NOT a formal test
            }
        )
    pd.DataFrame(range_rows).to_csv(
        d / "gwr_coefficient_ranges.csv", index=False, encoding="utf-8"
    )
    files.append("gwr_coefficient_ranges.csv")

    # ---- local-R² + coefficient surface map (PNG, best-effort) ------------ #
    lon_is_x = "lon" in str(lon).lower() or "lng" in str(lon).lower()
    xcoord, ycoord = (coords[:, 0], coords[:, 1]) if lon_is_x else (coords[:, 1], coords[:, 0])
    xlab, ylab = (lon, lat) if lon_is_x else (lat, lon)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # pick the predictor with the widest local spread for the coefficient panel
        spreads = [betas[:, j].max() - betas[:, j].min() for j in range(1, p)]
        wj = int(np.argmax(spreads)) + 1 if spreads else 1
        wname = predictors[wj - 1]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sc0 = axes[0].scatter(
            xcoord, ycoord, c=local_r2, cmap="viridis", s=28,
            edgecolor="#333333", linewidth=0.3,
        )
        fig.colorbar(sc0, ax=axes[0], label="local R^2")
        axes[0].set_title(f"GWR local R^2 — {outcome}")
        sc1 = axes[1].scatter(
            xcoord, ycoord, c=betas[:, wj], cmap="RdBu_r", s=28,
            edgecolor="#333333", linewidth=0.3,
        )
        fig.colorbar(sc1, ax=axes[1], label=f"local beta ({wname})")
        axes[1].set_title(f"GWR coefficient surface — {wname}")
        for ax in axes:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
        fig.tight_layout()
        fig.savefig(d / "gwr_surfaces.png", dpi=150)
        plt.close(fig)
        files.append("gwr_surfaces.png")
    except Exception:
        pass

    # ---- estimates (floats) ----------------------------------------------- #
    estimates["aicc"] = round(float(aicc), 4)
    estimates["effective_params_trS"] = round(float(trS), 4)
    estimates["n"] = float(n)
    estimates["mean_local_r2"] = round(float(np.mean(local_r2)), 4)
    estimates["global_ols_r2"] = round(float(ols_r2), 4)
    if selected.get("mode") == "adaptive":
        estimates["selected_k"] = float(selected["k"])
    else:
        estimates["selected_bw"] = round(float(selected["bw"]), 6)
    for j, name in enumerate(predictors, start=1):
        col = betas[:, j]
        estimates[f"beta_{name}_min"] = round(float(col.min()), 6)
        estimates[f"beta_{name}_median"] = round(float(np.median(col)), 6)
        estimates[f"beta_{name}_max"] = round(float(col.max()), 6)
        estimates[f"beta_{name}_ols"] = round(float(beta_ols[j]), 6)

    # ---- Chinese summary with ⚠ disclosures ------------------------------- #
    n_nonstat = sum(nonstat_flags.values())
    nonstat_names = [k for k, v in nonstat_flags.items() if v]
    nonstat_txt = (
        f"局部系数「宽幅」(局部 IQR > 2·全局 OLS SE) 的预测变量 {n_nonstat}/{len(predictors)} 个"
        + (f"（{', '.join(nonstat_names)}）" if nonstat_names else "")
    )
    summary.append(
        f"{entry.method} 完成：n={n}，结果变量 {outcome}，预测变量 "
        f"{', '.join(predictors)}；核={'双平方(bisquare)' if kernel == 'bisquare' else '高斯(gaussian)'}，"
        f"{bw_report}（按 AICc={aicc:.1f} 选择）；有效参数 tr(S)={trS:.2f}；"
        f"平均局部 R²={np.mean(local_r2):.3f}（全局 OLS R²={ols_r2:.3f}）；{nonstat_txt}"
        "（⚠ 这是**描述性指标、非正式检验**，倾向高估：局部窗噪声放大 IQR；正式非平稳检验需 Monte-Carlo/Leung，未做）。"
        " ⚠ GWR 是探索性而非验证性方法：系数随空间变化揭示关系异质性，但"
        "带宽选择决定平滑度（已报告，AICc 选）；局部多重共线性 + 跨位置多重检验会夸大表观变异；"
        "经纬度欧氏距离仅为近似（应使用投影/平面坐标使距离有意义）。"
    )

    # ---- runnable-code sketch -------------------------------------------- #
    code += [
        "import numpy as np  # Geographically Weighted Regression (Brunsdon-Fotheringham-Charlton 1996)",
        f"# coords=({lon},{lat}); outcome='{outcome}'; predictors={predictors}; kernel='{kernel}'",
        "# per location i: W_i = diag(kernel(d_ij, bw)); beta_i = (X^T W_i X)^-1 X^T W_i y",
        "# hat row s_i = x_i (X^T W_i X)^-1 X^T W_i ; trS = sum(s_ii) = effective params",
        "# AICc = 2n*ln(sigma_hat) + n*ln(2pi) + n*(n+trS)/(n-2-trS); pick bw minimising AICc",
        f"# non-stationarity: local-IQR(beta) > 2 * global OLS SE  (selected: {selected})",
    ]
