"""Branch handlers for the spatial_extra family — point-pattern & local/categorical
spatial autocorrelation, hand-rolled in pure NumPy/SciPy (NO R, NO libpysal/esda/
pointpats — those are not installed).

  * ripleys_k  — Ripley's K / L function for a 2-D point pattern, with an approximate
                 bounding-box edge correction and a Monte-Carlo CSR envelope.
  * getis_ord  — Getis-Ord Gi* local hotspot z-scores for a value field, plus the
                 global G statistic with its analytical z / p (Getis & Ord 1992).
  * join_count — join-count statistics (BB / WW / BW) for a binary field over a
                 symmetrised k-NN contiguity graph, with exact free-sampling moments.

Each handler resolves coordinate columns (config x/y → name-hinted geo lon/lat →
first two continuous columns), degrades honestly (too few points / no coordinates /
non-binary attribute / missing import → append a Chinese "<方法>跳过：<原因>" to
summary and RETURN — never crash, never fabricate), writes CSV + PNG (matplotlib
Agg, ENGLISH plot labels — the default font has no CJK), fills float `estimates`,
appends a Chinese `summary` ending with ⚠ disclosures, and MUTATES ctx (never
rebinds summary/estimates/files/code). Monte-Carlo uses a fixed, disclosed seed.

See executor/_branch_api.py (Ctx) and CLAUDE.md「引擎约定」.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Fixed Monte-Carlo seed (disclosed in every summary that uses it).
_SEED = 12345


# ─────────────────────────────────────────────────────────────────────────────
# Shared coordinate resolution.
#   1. config x / y (explicit column names)
#   2. name-hinted geo columns (lon/lat — profiler kind == "geo"), lon→x, lat→y
#   3. first two continuous columns (a plain planar point pattern)
# Returns (xname, yname) or (None, None) if two distinct numeric coords can't be
# found. The caller degrades honestly when None.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_xy(ctx: Ctx):
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df

    cx, cy = cfg.get("x"), cfg.get("y")
    if cx and cy and cx in df.columns and cy in df.columns and cx != cy:
        return cx, cy

    geo = [c.name for c in fp.columns if c.kind == "geo"]
    if len(geo) >= 2:
        lon = next((g for g in geo if "lon" in g.lower() or "lng" in g.lower()), None)
        lat = next((g for g in geo if "lat" in g.lower()), None)
        if lon and lat and lon != lat:
            return lon, lat  # x = longitude, y = latitude
        return geo[0], geo[1]

    cont = [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]
    if len(cont) >= 2:
        return cont[0], cont[1]
    return None, None


def _knn_binary_weights(coords, k):
    """Symmetric binary k-NN contiguity. Returns (W, neighbour-index array).

    For each point, its k nearest neighbours (Euclidean) get weight 1; the matrix
    is then symmetrised (i~j if i is a kNN of j OR j is a kNN of i) so join counts
    and BB/BW moments are well defined. Self-weights are 0 (no self-loops)."""
    import numpy as np

    n = len(coords)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), nn.ravel()] = 1.0
    W = np.maximum(W, W.T)  # symmetrise contiguity
    np.fill_diagonal(W, 0.0)
    return W, nn


# ─────────────────────────────────────────────────────────────────────────────
# (A) ripleys_k — Ripley's K / L with a bounding-box edge correction + CSR envelope
# ─────────────────────────────────────────────────────────────────────────────
@register("ripleys_k")
def _branch_ripleys_k(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
    except ImportError:
        summary.append("Ripley's K 跳过：缺少 numpy。")
        return

    xn, yn = _resolve_xy(ctx)
    if xn is None:
        summary.append(
            "Ripley's K 跳过：需要两个坐标列（config={\"x\":\"<列>\",\"y\":\"<列>\"}，"
            "或经纬度 geo 列，或两个连续列）。"
        )
        return

    sub = df[[xn, yn]].dropna()
    pts = sub.to_numpy(dtype=float)
    n = len(pts)
    if n < 20:
        summary.append(f"Ripley's K 跳过：点数不足（{n}<20），点模式统计不稳。")
        return

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        summary.append("Ripley's K 跳过：点退化为一条线/一个点（包围盒面积为 0）。")
        return
    area = float(bw * bh)

    # Radius grid: up to ~1/4 of the smaller box side (beyond that the estimator
    # is dominated by edge correction and is unreliable).
    r_max = float(min(bw, bh) / 4.0)
    if r_max <= 0:
        summary.append("Ripley's K 跳过：包围盒过窄，无法选取半径栅格。")
        return
    n_r = 25
    radii = np.linspace(r_max / n_r, r_max, n_r)

    # Pairwise distances (n is modest, so the full n×n is fine).
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)  # exclude self-pairs

    def _edge_weights(points, dmat):
        """w_ij = 1 / (fraction of the circle of radius d_ij centred at i that lies
        inside the bounding box). Approximate (axis-aligned-box) isotropic edge
        correction: estimate the inside-fraction by sampling angles on the circle.
        Returns the per-pair weight matrix."""
        n_pts = len(points)
        n_ang = 36
        ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
        ca, sa = np.cos(ang), np.sin(ang)
        finite = np.isfinite(dmat)
        xi = points[:, 0][:, None]
        yi = points[:, 1][:, None]
        rr = np.where(finite, dmat, 0.0)
        inside_count = np.zeros((n_pts, n_pts))
        for cx_, sy_ in zip(ca, sa):  # 36 cheap vectorised passes, keeps mem n×n
            px = xi + rr * cx_
            py = yi + rr * sy_
            inside_count += (
                (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
            ).astype(float)
        frac = inside_count / n_ang
        frac = np.clip(frac, 1.0 / n_ang, 1.0)  # avoid div-by-0 for tiny fractions
        w = 1.0 / frac
        w[~finite] = 0.0
        return w

    w_edge = _edge_weights(pts, dist)

    def _K_of_r(dmat, wmat):
        # K(r) = (|A| / (n(n-1))) Σ_{i≠j} w_ij 1[d_ij ≤ r]
        Ks = np.empty(n_r)
        for ri, r in enumerate(radii):
            ind = (dmat <= r).astype(float)
            Ks[ri] = area / (n * (n - 1)) * float((wmat * ind).sum())
        return Ks

    K_obs = _K_of_r(dist, w_edge)
    L_obs = np.sqrt(np.maximum(K_obs, 0.0) / np.pi)
    Lmr_obs = L_obs - radii  # centred L; 0 under CSR, >0 cluster, <0 dispersion

    # Monte-Carlo CSR envelope: simulate n_sim patterns of n uniform points in the
    # same box; pointwise 2.5 / 97.5 percentiles of L(r)-r.
    n_sim = cfg.get("n_sim", 99)
    try:
        n_sim = int(n_sim)
    except (TypeError, ValueError):
        n_sim = 99
    n_sim = max(19, min(n_sim, 999))
    rng = np.random.default_rng(_SEED)
    sims = np.empty((n_sim, n_r))
    for s in range(n_sim):
        sp = np.column_stack(
            [rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)]
        )
        sd = np.sqrt(((sp[:, None, :] - sp[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(sd, np.inf)
        sw = _edge_weights(sp, sd)
        Ks = _K_of_r(sd, sw)
        sims[s] = np.sqrt(np.maximum(Ks, 0.0) / np.pi) - radii
    env_low = np.percentile(sims, 2.5, axis=0)
    env_high = np.percentile(sims, 97.5, axis=0)

    clustered = Lmr_obs > env_high          # observed above the CSR band
    dispersed = Lmr_obs < env_low
    frac_clustered = float(clustered.mean())
    exits = clustered | dispersed
    max_dev_idx = int(np.argmax(np.abs(Lmr_obs)))
    r_at_max_dev = float(radii[max_dev_idx])
    max_L_minus_r = float(Lmr_obs[max_dev_idx])
    largest_exit_r = float(radii[exits][-1]) if exits.any() else 0.0

    import pandas as pd

    try:
        pd.DataFrame(
            {
                "r": np.round(radii, 6),
                "K": np.round(K_obs, 6),
                "L": np.round(L_obs, 6),
                "L_minus_r": np.round(Lmr_obs, 6),
                "env_low": np.round(env_low, 6),
                "env_high": np.round(env_high, 6),
            }
        ).to_csv(d / "ripleys_k.csv", index=False, encoding="utf-8")
        files.append("ripleys_k.csv")
    except Exception:
        pass

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.fill_between(
            radii, env_low, env_high, color="#cccccc", alpha=0.6,
            label=f"CSR envelope ({n_sim} sims, 2.5-97.5%)",
        )
        ax.plot(radii, Lmr_obs, color="#C44E52", lw=1.6, label="observed L(r) - r")
        ax.axhline(0, color="#444444", ls="--", lw=0.8)
        ax.set_xlabel("radius r")
        ax.set_ylabel("L(r) - r")
        ax.set_title("Ripley's L (centred) with CSR envelope")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(d / "ripleys_l.png", dpi=150)
        plt.close(fig)
        files.append("ripleys_l.png")
    except Exception:
        pass

    estimates["max_L_minus_r"] = round(max_L_minus_r, 6)
    estimates["r_at_max_dev"] = round(r_at_max_dev, 6)
    estimates["n_points"] = float(n)
    estimates["max_radius"] = round(r_max, 6)
    estimates["frac_radii_clustered"] = round(frac_clustered, 4)

    if frac_clustered > 0:
        verdict = (
            f"在 {int(clustered.sum())}/{n_r} 个半径上观测 L(r)-r 超出 CSR 上包络 → 聚集"
            f"（最大偏离尺度 r≈{r_at_max_dev:.4g}，最远显著尺度 r≈{largest_exit_r:.4g}）"
        )
    elif dispersed.any():
        verdict = f"观测落在 CSR 下包络之下 → 规则/离散（最大偏离尺度 r≈{r_at_max_dev:.4g}）"
    else:
        verdict = "观测全程落在 CSR 包络带内 → 与完全空间随机(CSR)无异"

    summary.append(
        f"{entry.method} 完成：{n} 个点，坐标=({xn},{yn})，半径栅格 0–{r_max:.4g}（{n_r} 档）；{verdict}。"
        f"⚠ 边缘校正为近似（基于轴对齐包围盒的圆内比例，非真各向异性校正）；"
        f"CSR 包络为蒙特卡洛（{n_sim} 次模拟，固定种子 seed={_SEED}）；"
        f"需≥20 点；某尺度的聚集可能掩盖另一尺度的离散；config 可设 x/y/n_sim。"
    )
    code += [
        "import numpy as np  # Ripley's K / L for a 2-D point pattern",
        f"# coords=({xn},{yn}), |A|=bbox area, K(r)=|A|/(n(n-1)) * sum_{{i!=j}} w_ij*1[d<=r]",
        "# w_ij = 1 / (fraction of circle radius d_ij inside bbox)  [approx edge correction]",
        f"# L(r)=sqrt(K/pi); CSR envelope = pointwise 2.5/97.5% of L-r over {n_sim} uniform sims",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# (B) getis_ord — Getis-Ord Gi* local hotspots + global G (Getis & Ord 1992)
# ─────────────────────────────────────────────────────────────────────────────
@register("getis_ord")
def _branch_getis_ord(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
    except ImportError:
        summary.append("Getis-Ord 跳过：缺少 numpy。")
        return

    xn, yn = _resolve_xy(ctx)
    if xn is None:
        summary.append(
            "Getis-Ord 跳过：需要两个坐标列（config x/y，或经纬度 geo 列，或两个连续列）。"
        )
        return

    value = cfg.get("value")
    if not (value and value in df.columns):
        value = next(
            (
                c.name
                for c in fp.columns
                if c.kind == "continuous"
                and c.name not in {fp.unit_col, fp.time_col, xn, yn}
            ),
            None,
        )
    if value is None:
        summary.append("Getis-Ord 跳过：需要一个数值属性列（config={\"value\":\"<列>\"}）。")
        return

    sub = df[[xn, yn, value]].dropna()
    coords = sub[[xn, yn]].to_numpy(dtype=float)
    x = sub[value].to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        summary.append(f"Getis-Ord 跳过：有效样本不足（{n}<10）。")
        return

    from researchforge.executor.run import _knn_k

    # Gi* includes self → k+1 ≤ n; n-2 keeps the variance term strictly positive.
    k = _knn_k(cfg, n - 2)  # config={"knn_k": N}

    xbar = float(x.mean())
    # population SD (Getis-Ord uses S = sqrt(sum x^2 / n - xbar^2)).
    S = float(np.sqrt(max((x**2).mean() - xbar**2, 0.0)))
    if S == 0:
        summary.append("Getis-Ord 跳过：值变量为常数（标准差为 0）。")
        return

    # k-NN binary weights INCLUDING self (Gi* / star).
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), nn.ravel()] = 1.0
    np.fill_diagonal(W, 1.0)  # Gi* includes the focal point

    Wsum = W.sum(axis=1)
    Wsq = (W**2).sum(axis=1)
    num = W @ x - xbar * Wsum
    den = S * np.sqrt(np.maximum((n * Wsq - Wsum**2) / (n - 1), 1e-300))
    gi = num / den  # asymptotically standard-normal z-score per location
    hot = gi > 1.96
    cold = gi < -1.96

    # Global G statistic (Getis & Ord 1992): excludes self-pairs.
    Wg = W.copy()
    np.fill_diagonal(Wg, 0.0)
    Wtot = float(Wg.sum())
    cross = float(x @ (Wg @ x))            # Σ_i Σ_{j≠i} w_ij x_i x_j
    denom_all = float(np.outer(x, x).sum() - (x**2).sum())  # Σ_{i≠j} x_i x_j
    g_stat = float("nan")
    g_z = float("nan")
    g_p = float("nan")
    # Getis-Ord global G is DEFINED ONLY for a non-negative variable (it measures the
    # concentration of high POSITIVE values); with negative/centered x the denominator
    # Σ_{i≠j}x_i x_j and the moment formulas are not interpretable. Skip + disclose then.
    g_nonneg = float(x.min()) >= 0.0
    if denom_all != 0 and Wtot != 0 and g_nonneg:
        g_stat = cross / denom_all
        EG = Wtot / (n * (n - 1))
        S1 = 0.5 * float(((Wg + Wg.T) ** 2).sum())
        rowsum = Wg.sum(axis=1)
        colsum = Wg.sum(axis=0)
        S2 = float(((rowsum + colsum) ** 2).sum())
        m1 = float(x.sum())
        m2 = float((x**2).sum())
        m3 = float((x**3).sum())
        m4 = float((x**4).sum())
        B0 = (n**2 - 3 * n + 3) * S1 - n * S2 + 3 * Wtot**2
        B1 = -((n**2 - n) * S1 - 2 * n * S2 + 6 * Wtot**2)
        B2 = -(2 * n * S1 - (n + 3) * S2 + 6 * Wtot**2)
        B3 = 4 * (n - 1) * S1 - 2 * (n + 1) * S2 + 8 * Wtot**2
        B4 = S1 - S2 + Wtot**2
        denom_var = (m1**2 - m2) ** 2 * n * (n - 1) * (n - 2) * (n - 3)
        if denom_var != 0:
            EG2 = (
                B0 * m2 * m2 + B1 * m4 + B2 * m1 * m1 * m2 + B3 * m1 * m3 + B4 * m1**4
            ) / denom_var
            VG = EG2 - EG**2
            if VG > 0:
                g_z = (g_stat - EG) / float(np.sqrt(VG))
                from scipy import stats as _st

                g_p = float(2 * _st.norm.sf(abs(g_z)))

    import pandas as pd

    try:
        pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "value": x,
                "gi_z": np.round(gi, 4),
                "class": np.where(hot, "hotspot", np.where(cold, "coldspot", "ns")),
            }
        ).to_csv(d / "getis_ord_local.csv", index=False, encoding="utf-8")
        files.append("getis_ord_local.csv")
    except Exception:
        pass

    # map orientation: longitude on x, latitude on y when detectable
    lon_i = 1 if ("lon" in str(yn).lower() or "lng" in str(yn).lower()) else 0
    lat_i = 1 - lon_i
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            coords[:, lon_i], coords[:, lat_i], c=gi, cmap="RdBu_r",
            vmin=-3, vmax=3, s=30, edgecolor="#444444", linewidth=0.3,
        )
        fig.colorbar(sc, label="Gi* z-score")
        ax.set_xlabel([xn, yn][lon_i])
        ax.set_ylabel([xn, yn][lat_i])
        ax.set_title(f"Getis-Ord Gi* hotspots - {value}")
        fig.tight_layout()
        fig.savefig(d / "getis_ord_map.png", dpi=150)
        plt.close(fig)
        files.append("getis_ord_map.png")
    except Exception:
        pass

    estimates["global_g_z"] = round(g_z, 4) if np.isfinite(g_z) else float("nan")
    estimates["global_g_p"] = round(g_p, 4) if np.isfinite(g_p) else float("nan")
    estimates["n_hotspots"] = float(int(hot.sum()))
    estimates["n_coldspots"] = float(int(cold.sum()))
    estimates["max_gi_z"] = round(float(gi.max()), 4)
    estimates["min_gi_z"] = round(float(gi.min()), 4)
    estimates["n"] = float(n)

    if np.isfinite(g_z):
        gtxt = f"全局 G z={g_z:.3f}（p={g_p:.4g}）"
    elif not g_nonneg:
        gtxt = "全局 G 已跳过（需非负变量；本列含负值——可改用 moran_i 测全局自相关）"
    else:
        gtxt = "全局 G 不可估"
    summary.append(
        f"{entry.method} 完成：变量 {value}，{int(hot.sum())} 个热点 / {int(cold.sum())} 个冷点"
        f"（|Gi*|>1.96，k-NN={k}+自身）；{gtxt}。Gi* 为每点 z 分数，正=高值聚集、负=低值聚集。"
        f"⚠ Gi* 适用于任意实数变量；**全局 G 仅对非负变量有定义**（含负值时已跳过）；"
        f"Gi* 依赖权重设定（k-NN k={k} 已披露）；z 分数为渐近正态近似（小 n / 聚集权重下偏粗）；"
        f"跨位置多重检验未校正（报告原始 z）；config 可设 value/x/y/knn_k。"
    )
    code += [
        "import numpy as np  # Getis-Ord Gi* (star, includes focal point) + global G",
        f"# coords=({xn},{yn}), value='{value}', k={k}, binary kNN+self weights",
        "# Gi* = (Wx - xbar*sum_w) / (S*sqrt((n*sum_w2 - sum_w^2)/(n-1)))",
        "# global G = sum_{i!=j} w_ij x_i x_j / sum_{i!=j} x_i x_j ; z via Getis-Ord 1992 moments",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# (C) join_count — BB/WW/BW join counts for a binary field (free sampling)
# ─────────────────────────────────────────────────────────────────────────────
@register("join_count")
def _branch_join_count(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
    except ImportError:
        summary.append("Join-count 跳过：缺少 numpy。")
        return

    xn, yn = _resolve_xy(ctx)
    if xn is None:
        summary.append(
            "Join-count 跳过：需要两个坐标列（config x/y，或经纬度 geo 列，或两个连续列）。"
        )
        return

    # Resolve a binary attribute: config value (if present), else a binary col.
    value = cfg.get("value") if cfg.get("value") in df.columns else None
    if value is None:
        value = next(
            (
                c.name
                for c in fp.columns
                if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col, xn, yn}
            ),
            None,
        )
    if value is None:
        summary.append(
            "Join-count 跳过：需要一个二值属性列（两类皆出现；config={\"value\":\"<列>\"}）。"
        )
        return

    sub = df[[xn, yn, value]].dropna()
    coords = sub[[xn, yn]].to_numpy(dtype=float)
    raw = sub[value].to_numpy()
    uniq = np.unique(raw)
    if len(uniq) != 2:
        summary.append(
            f"Join-count 跳过：属性 {value} 不是恰好两类（现 {len(uniq)} 类），无法做二值 join-count。"
        )
        return
    # Map the two classes to {0,1}; the larger label is "black" (1).
    b = (raw == uniq.max()).astype(int)
    n = len(b)
    n1 = int(b.sum())   # blacks
    n0 = n - n1         # whites
    if n < 10:
        summary.append(f"Join-count 跳过：有效样本不足（{n}<10）。")
        return
    if n1 == 0 or n0 == 0:
        summary.append("Join-count 跳过：二值属性只有一类，无法计数 join。")
        return

    from researchforge.executor.run import _knn_k

    k = _knn_k(cfg, n - 1, default=6)  # config={"knn_k": N}
    W, _ = _knn_binary_weights(coords, k)  # symmetric binary contiguity

    # Edge list (i<j) from the symmetric binary matrix.
    iu, ju = np.triu_indices(n, k=1)
    edge = W[iu, ju] > 0
    ei, ej = iu[edge], ju[edge]
    J = int(edge.sum())  # number of joins (undirected edges, counted once)
    if J == 0:
        summary.append("Join-count 跳过：邻接图无边（k 太小或点完全分散）。")
        return

    bi, bj = b[ei], b[ej]
    bb = int((bi & bj).sum())                 # 1-1 joins
    ww = int(((1 - bi) & (1 - bj)).sum())     # 0-0 joins
    bw = J - bb - ww                          # 1-0 joins

    # Free-sampling moments (x_i iid Bernoulli(p), independent). Derived from
    # first principles via the indicator approach (exact):
    #   E[BB] = J p^2 ; Var[BB] = J(p^2 - p^4) + (p^3 - p^4) * sum_i L_i(L_i-1)
    #   E[BW] = 2 J p q ; Var[BW] = 2 J pq(1-2pq) + (pq - 4 p^2 q^2) * sum_i L_i(L_i-1)
    # where L_i = degree of node i and sum_i L_i(L_i-1) counts ordered pairs of
    # distinct edges sharing a vertex.
    p = n1 / n
    q = n0 / n
    deg = W.sum(axis=1)                        # node degrees (symmetric binary)
    m = float((deg * (deg - 1)).sum())         # sum_i L_i(L_i-1)

    bb_exp = J * p * p
    bb_var = J * (p**2 - p**4) + (p**3 - p**4) * m
    bw_exp = 2 * J * p * q
    bw_var = 2 * J * p * q * (1 - 2 * p * q) + (p * q - 4 * p**2 * q**2) * m
    ww_exp = J * q * q
    ww_var = J * (q**2 - q**4) + (q**3 - q**4) * m   # symmetric to BB (p<->q)

    def _z(obs, exp, var):
        if var <= 0:
            return float("nan")
        return (obs - exp) / float(np.sqrt(var))

    from scipy import stats as _st

    bb_z = _z(bb, bb_exp, bb_var)
    bw_z = _z(bw, bw_exp, bw_var)
    ww_z = _z(ww, ww_exp, ww_var)
    # Report two-sided p for symmetry with the engine's other tests; the sign of z
    # carries direction (BB z>0 = like-clustering).
    bb_p = float(2 * _st.norm.sf(abs(bb_z))) if np.isfinite(bb_z) else float("nan")
    bw_p = float(2 * _st.norm.sf(abs(bw_z))) if np.isfinite(bw_z) else float("nan")
    ww_p = float(2 * _st.norm.sf(abs(ww_z))) if np.isfinite(ww_z) else float("nan")

    import pandas as pd

    try:
        pd.DataFrame(
            {
                "join_type": ["BB (1-1)", "WW (0-0)", "BW (1-0)"],
                "observed": [bb, ww, bw],
                "expected": [round(bb_exp, 4), round(ww_exp, 4), round(bw_exp, 4)],
                "z": [
                    round(bb_z, 4) if np.isfinite(bb_z) else None,
                    round(ww_z, 4) if np.isfinite(ww_z) else None,
                    round(bw_z, 4) if np.isfinite(bw_z) else None,
                ],
                "p_value": [
                    round(bb_p, 4) if np.isfinite(bb_p) else None,
                    round(ww_p, 4) if np.isfinite(ww_p) else None,
                    round(bw_p, 4) if np.isfinite(bw_p) else None,
                ],
            }
        ).to_csv(d / "join_count.csv", index=False, encoding="utf-8")
        files.append("join_count.csv")
    except Exception:
        pass

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["BB (1-1)", "WW (0-0)", "BW (1-0)"]
        obs = [bb, ww, bw]
        exp = [bb_exp, ww_exp, bw_exp]
        xpos = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(xpos - 0.2, obs, width=0.4, color="#4C72B0", label="observed")
        ax.bar(xpos + 0.2, exp, width=0.4, color="#cccccc", label="expected (CSR)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("join count")
        ax.set_title("Join counts: observed vs expected (free sampling)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "join_count.png", dpi=150)
        plt.close(fig)
        files.append("join_count.png")
    except Exception:
        pass

    estimates["bb_count"] = float(bb)
    estimates["bb_expected"] = round(bb_exp, 4)
    estimates["bb_z"] = round(bb_z, 4) if np.isfinite(bb_z) else float("nan")
    estimates["bb_p"] = round(bb_p, 4) if np.isfinite(bb_p) else float("nan")
    estimates["bw_count"] = float(bw)
    estimates["bw_z"] = round(bw_z, 4) if np.isfinite(bw_z) else float("nan")
    estimates["bw_p"] = round(bw_p, 4) if np.isfinite(bw_p) else float("nan")
    estimates["n"] = float(n)

    if np.isfinite(bb_z) and bb_z > 1.96:
        verdict = "BB 显著偏多 → 同类聚集（正空间自相关）"
    elif np.isfinite(bb_z) and bb_z < -1.96:
        verdict = "BB 显著偏少 → 同类回避/棋盘格（负空间自相关）"
    else:
        verdict = "BB 与自由抽样期望无显著差异 → 无明显分类空间自相关"

    summary.append(
        f"{entry.method} 完成：二值属性 {value}（1={n1}/0={n0}），k-NN={k} 对称邻接共 {J} 条边；"
        f"BB={bb}(期望 {bb_exp:.2f}, z={bb_z:.2f}) / WW={ww}(期望 {ww_exp:.2f}) / "
        f"BW={bw}(期望 {bw_exp:.2f}, z={bw_z:.2f})；{verdict}。"
        f"⚠ 采用自由抽样(free sampling)假定，即视两类各点数为独立同分布抽样；"
        f"这比非自由抽样（固定类别总数下的无放回排列/置换检验）方差更小，"
        f"z 检验偏乐观（更容易显著），尤其在小样本或类别不均衡时；"
        f"需更保守推断可改用置换检验（打乱标签重算 BB/WW/BW 得经验零分布）。"
        f"权重来自 k-NN（k={k} 已披露，已对称化）；需属性两类皆出现；"
        f"config 可设 value/x/y/knn_k。"
    )
    code += [
        "import numpy as np  # join-count statistics for a binary spatial field",
        f"# coords=({xn},{yn}), binary='{value}', symmetric k-NN (k={k}) contiguity",
        "# BB=sum_{(i,j) in E} x_i x_j ; BW=sum (x_i + x_j - 2 x_i x_j) ; WW=J-BB-BW",
        "# free sampling: E[BB]=J p^2, Var[BB]=J(p^2-p^4)+(p^3-p^4) sum_i L_i(L_i-1)",
        "# z=(obs-exp)/sqrt(var); BB z>0 => like-clustering (positive autocorrelation)",
    ]
