"""Branch handlers for the spatial_dependence family — spatial association &
regionalization (pure Python: numpy / scipy, no R / no pysal). Complements the
existing spatial methods (moran_i, local_moran, getis_ord, spatial_regression).

  * bivariate_moran  — bivariate global Moran's I: association between variable X at a
                       location and the spatial lag of variable Y at its neighbours,
                       with a permutation pseudo p-value. "Is high-X near high-Y?"
  * local_geary      — Local Geary's C (a LISA): per-location c_i = Σ_j w_ij (z_i−z_j)²,
                       flagging local DISSIMILARITY (spatial outliers) vs local_moran's
                       similarity, via conditional-permutation pseudo p-values.
  * skater           — SKATER regionalization: spatially-CONTIGUOUS clustering by pruning
                       the minimum spanning tree of a k-NN graph to maximise within-region
                       homogeneity (sum-of-squares reduction). Contiguity-constrained.

Coordinates come from config x/y, else lon/lat geo columns, else the first two
continuous columns; analysis variables are the remaining continuous columns (config
overridable). Each degrades honestly (too few cols / rows / constant variable ->
append a Chinese "<method>跳过：<reason>" and RETURN), writes CSV + PNG (matplotlib
Agg, ENGLISH labels), fills float `estimates`, appends a Chinese `summary` with ⚠
disclosures, and MUTATES ctx. See _branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_SEED = 20240607


def _coords(ctx: Ctx):
    """Resolve (x_col, y_col) coordinate columns: config x/y, else geo lon/lat,
    else first two continuous. Returns (cx, cy) or (None, None)."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    cx, cy = cfg.get("x"), cfg.get("y")
    if cx in df.columns and cy in df.columns and cx != cy:
        return cx, cy
    geo = [c.name for c in fp.columns if c.kind == "geo"]
    if len(geo) >= 2:
        lon = next((g for g in geo if "lon" in g.lower() or "lng" in g.lower()), geo[0])
        lat = next((g for g in geo if "lat" in g.lower()), geo[1])
        return (lon, lat) if lon != lat else (geo[0], geo[1])
    cont = [c.name for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
    if len(cont) >= 2:
        return cont[0], cont[1]
    return None, None


def _attrs(ctx: Ctx, exclude):
    """Continuous analysis variables, excluding the coordinate columns."""
    fp = ctx.fp
    return [c.name for c in fp.columns
            if c.kind == "continuous" and c.name not in exclude
            and c.name not in {fp.unit_col, fp.time_col}]


def _knn_binary(coords, k):
    """Symmetric binary k-NN contiguity (no self-loops)."""
    import numpy as np

    n = len(coords)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), nn.ravel()] = 1.0
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def _row_std(W):
    rs = W.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def _knn_k(cfg, n):
    try:
        k = int(cfg.get("knn_k", 0))
    except (TypeError, ValueError):
        k = 0
    if k <= 0:
        k = max(2, min(8, int(round(n ** 0.5))))
    return max(1, min(k, n - 1))


# ---------------------------------------------------------------------------
# 1. bivariate_moran
# ---------------------------------------------------------------------------
@register("bivariate_moran")
def _branch_bivariate_moran(ctx: Ctx) -> None:
    df, entry, cfg, d = ctx.df, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cx, cy = _coords(ctx)
    if cx is None:
        summary.append("双变量 Moran 跳过：需要坐标（config x/y 或经纬度 geo 列，或≥2 连续列）。")
        return
    attrs = _attrs(ctx, {cx, cy})
    v1 = cfg.get("var1") if cfg.get("var1") in df.columns else (attrs[0] if attrs else None)
    v2 = cfg.get("var2") if cfg.get("var2") in df.columns else (attrs[1] if len(attrs) > 1 else None)
    if v1 is None or v2 is None or v1 == v2:
        summary.append("双变量 Moran 跳过：需要 2 个不同的分析变量（config var1/var2，或坐标外≥2 连续列）。")
        return

    sub = df[[cx, cy, v1, v2]].dropna()
    try:
        sub = sub.astype(float)
    except (TypeError, ValueError):
        summary.append("双变量 Moran 跳过：所选列存在非数值。")
        return
    n = len(sub)
    if n < 10:
        summary.append(f"双变量 Moran 跳过：有效点 {n} < 10。")
        return

    try:
        import numpy as np

        coords = sub[[cx, cy]].to_numpy(float)
        x = sub[v1].to_numpy(float)
        y = sub[v2].to_numpy(float)
        if x.std() < 1e-12 or y.std() < 1e-12:
            summary.append("双变量 Moran 跳过：分析变量为常数。")
            return
        zx = (x - x.mean()) / x.std()
        zy = (y - y.mean()) / y.std()
        k = _knn_k(cfg, n)
        Wrs = _row_std(_knn_binary(coords, k))

        def _I(zy_vec):
            return float((zx @ (Wrs @ zy_vec)) / (zx @ zx))

        I_obs = _I(zy)
        rng = np.random.default_rng(_SEED)
        n_perm = max(99, int(cfg.get("n_perm", 999)))
        perm = np.empty(n_perm)
        for i in range(n_perm):
            perm[i] = _I(zy[rng.permutation(n)])
        # two-sided pseudo p-value (Hope) — how often a permuted |I| >= observed
        p = float((np.sum(np.abs(perm) >= abs(I_obs)) + 1) / (n_perm + 1))
        z_score = float((I_obs - perm.mean()) / perm.std()) if perm.std() > 1e-12 else float("nan")

        estimates.update({
            "bivariate_moran_I": round(I_obs, 6), "p_value": round(p, 6),
            "z_score": round(z_score, 6), "k_neighbors": float(k),
            "n_perm": float(n_perm), "n": float(n),
        })

        import pandas as pd
        lag_y = Wrs @ zy
        pd.DataFrame({cx: coords[:, 0], cy: coords[:, 1], f"z_{v1}": zx,
                      f"lag_z_{v2}": lag_y}).to_csv(
            d / "bivariate_moran_points.csv", index=False, encoding="utf-8")
        files.append("bivariate_moran_points.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(zx, lag_y, s=18, alpha=0.7, color="#4C72B0")
            b = np.polyfit(zx, lag_y, 1)
            xs = np.linspace(zx.min(), zx.max(), 50)
            ax.plot(xs, b[0] * xs + b[1], color="#C44E52",
                    label=f"slope=I={I_obs:.3f}")
            ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
            ax.set_xlabel(f"z({v1})"); ax.set_ylabel(f"spatial lag z({v2})")
            ax.set_title(f"Bivariate Moran scatter ({v1} vs lagged {v2})")
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(d / "bivariate_moran.png", dpi=150)
            plt.close(fig); files.append("bivariate_moran.png")
        except Exception:
            pass

        code += [
            "import numpy as np  # bivariate Moran: I = zx' W zy / (zx'zx), W row-standardized k-NN",
            "# permutation: shuffle zy, recompute I; pseudo-p two-sided",
        ]
        sig = "显著" if p < 0.05 else "不显著"
        rel = "正向（高-高/低-低空间共现）" if I_obs > 0 else "负向（高-低空间错配）"
        summary.append(
            f"{entry.method}：{v1} 与 {v2} 的空间滞后 的双变量 Moran I={I_obs:.4f}"
            f"（{rel}），置换 pseudo-p={p:.4g}（{sig}，z={z_score:.2f}，k-NN={k}，{n_perm} 次置换）。"
            " ⚠ 双变量 Moran 混合了 X-Y 的（非空间）相关与 Y 的空间结构，**不分解**二者，"
            "也不含因果；依赖权重设定（k-NN，已披露）；pseudo-p 为置换近似。"
        )
    except Exception as e:
        summary.append(f"双变量 Moran 失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 2. local_geary — Local Geary's C (LISA), conditional permutation
# ---------------------------------------------------------------------------
@register("local_geary")
def _branch_local_geary(ctx: Ctx) -> None:
    df, entry, cfg, d = ctx.df, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cx, cy = _coords(ctx)
    if cx is None:
        summary.append("局部 Geary 跳过：需要坐标（config x/y 或经纬度 geo 列，或≥2 连续列）。")
        return
    attrs = _attrs(ctx, {cx, cy})
    val = cfg.get("value") if cfg.get("value") in df.columns else (attrs[0] if attrs else None)
    if val is None:
        summary.append("局部 Geary 跳过：需要 1 个分析变量（config value，或坐标外≥1 连续列）。")
        return

    sub = df[[cx, cy, val]].dropna()
    try:
        sub = sub.astype(float)
    except (TypeError, ValueError):
        summary.append("局部 Geary 跳过：所选列存在非数值。")
        return
    n = len(sub)
    if n < 10:
        summary.append(f"局部 Geary 跳过：有效点 {n} < 10。")
        return

    try:
        import numpy as np

        coords = sub[[cx, cy]].to_numpy(float)
        x = sub[val].to_numpy(float)
        if x.std() < 1e-12:
            summary.append("局部 Geary 跳过：分析变量为常数。")
            return
        z = (x - x.mean()) / x.std()
        k = _knn_k(cfg, n)
        W = _knn_binary(coords, k)
        Wrs = _row_std(W)

        # observed local Geary c_i = Σ_j w_ij (z_i - z_j)^2  (row-standardized w)
        diff2 = (z[:, None] - z[None, :]) ** 2
        c = (Wrs * diff2).sum(1)

        # conditional permutation: for each i, draw its neighbours' values from the
        # other n-1 z's, recompute c_i under the null.
        rng = np.random.default_rng(_SEED)
        n_perm = max(99, int(cfg.get("n_perm", 999)))
        pvals = np.empty(n)
        exp = np.empty(n)   # E[c_i] under each point's own conditional permutation null
        for i in range(n):
            wi = Wrs[i]
            nb = np.where(wi > 0)[0]
            if nb.size == 0:
                pvals[i] = 1.0
                exp[i] = c[i]
                continue
            w_nb = wi[nb]
            others = np.delete(z, i)
            sims = np.empty(n_perm)
            for p_ in range(n_perm):
                samp = rng.choice(others, size=nb.size, replace=False)
                sims[p_] = float(np.sum(w_nb * (z[i] - samp) ** 2))
            exp[i] = float(sims.mean())
            # low c_i = local similarity (extreme low tail); two-sided pseudo-p
            ge = np.sum(sims <= c[i])
            le = np.sum(sims >= c[i])
            pvals[i] = float((min(ge, le) + 1) / (n_perm + 1))

        sig = pvals < 0.05
        # label by each point's OWN null mean (esda convention): c_i below its expected
        # value = locally more similar than chance; above = spatial outlier.
        labels = np.where(~sig, "ns",
                          np.where(c < exp, "low-C (similar cluster)", "high-C (spatial outlier)"))

        estimates.update({
            "n_significant": float(int(sig.sum())),
            "n_similar_clusters": float(int(np.sum(sig & (c < exp)))),
            "n_spatial_outliers": float(int(np.sum(sig & (c >= exp)))),
            "mean_local_geary": round(float(c.mean()), 6),
            "k_neighbors": float(k), "n_perm": float(n_perm), "n": float(n),
        })

        import pandas as pd
        pd.DataFrame({cx: coords[:, 0], cy: coords[:, 1], val: x,
                      "local_geary_c": np.round(c, 6), "p_value": np.round(pvals, 6),
                      "label": labels}).to_csv(
            d / "local_geary.csv", index=False, encoding="utf-8")
        files.append("local_geary.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cmap = {"ns": "#CCCCCC", "low-C (similar cluster)": "#4C72B0",
                    "high-C (spatial outlier)": "#C44E52"}
            fig, ax = plt.subplots(figsize=(6, 5))
            for lab, col in cmap.items():
                m = labels == lab
                if m.any():
                    ax.scatter(coords[m, 0], coords[m, 1], s=24, c=col, label=lab, alpha=0.8)
            ax.set_xlabel(cx); ax.set_ylabel(cy)
            ax.set_title(f"Local Geary's C — {val} (p<0.05)")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout(); fig.savefig(d / "local_geary.png", dpi=150)
            plt.close(fig); files.append("local_geary.png")
        except Exception:
            pass

        code += [
            "import numpy as np  # local Geary c_i = Σ_j w_ij (z_i - z_j)^2 (row-std W)",
            "# conditional permutation per i -> pseudo-p (two-sided)",
        ]
        summary.append(
            f"{entry.method}（{val}）：{int(sig.sum())} 个显著位点（p<0.05，{n_perm} 次条件置换，k-NN={k}）——"
            f"其中 {int(np.sum(sig & (c < exp)))} 个低-C（局部相似聚集）、"
            f"{int(np.sum(sig & (c >= exp)))} 个高-C（空间异常点）。"
            " ⚠ 局部 Geary 测局部**差异**（低 C=邻里相似、高 C=异常），与 local_moran 互补；"
            "多重比较未校正（逐点 pseudo-p）；依赖权重设定（k-NN，已披露）；非因果。"
        )
    except Exception as e:
        summary.append(f"局部 Geary 失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 3. skater — SKATER regionalization (MST pruning, contiguity-constrained)
# ---------------------------------------------------------------------------
@register("skater")
def _branch_skater(ctx: Ctx) -> None:
    df, entry, cfg, d = ctx.df, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cx, cy = _coords(ctx)
    if cx is None:
        summary.append("SKATER 跳过：需要坐标（config x/y 或经纬度 geo 列，或≥2 连续列）。")
        return
    attrs = _attrs(ctx, {cx, cy})
    feats = [c for c in (cfg.get("features") or []) if c in df.columns and c not in {cx, cy}] or attrs
    if len(feats) < 1:
        summary.append("SKATER 跳过：需要 ≥1 个分析变量（config features，或坐标外连续列）。")
        return

    sub = df[[cx, cy] + feats].dropna()
    try:
        sub = sub.astype(float)
    except (TypeError, ValueError):
        summary.append("SKATER 跳过：所选列存在非数值。")
        return
    n = len(sub)
    n_clusters = max(2, int(cfg.get("n_clusters", 5)))
    if n < n_clusters + 2 or n < 8:
        summary.append(f"SKATER 跳过：有效点 {n} 太少（需 ≥{max(8, n_clusters + 2)}）。")
        return

    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

        coords = sub[[cx, cy]].to_numpy(float)
        A = sub[feats].to_numpy(float)
        # standardize attributes so no feature dominates the dissimilarity
        std = A.std(0); std[std < 1e-12] = 1.0
        As = (A - A.mean(0)) / std
        k = _knn_k(cfg, n)
        W = _knn_binary(coords, k)

        # edge dissimilarity = Euclidean distance in standardized attribute space
        ii, jj = np.where(np.triu(W, 1) > 0)
        if ii.size == 0:
            summary.append("SKATER 跳过：邻接图无边（增大 knn_k）。")
            return
        dissim = np.sqrt(((As[ii] - As[jj]) ** 2).sum(1))
        # floor at a tiny positive value: scipy's MST treats an explicit zero weight as
        # a non-edge, which would silently drop a contiguity link between identical units.
        dissim = np.maximum(dissim, 1e-12)
        G = csr_matrix((dissim, (ii, jj)), shape=(n, n))
        mst = minimum_spanning_tree(G)              # minimal total dissimilarity forest
        mst = mst + mst.T                            # symmetric
        m = mst.tocoo()
        # undirected edge list from the MST (upper triangle)
        edges = [(int(a), int(b)) for a, b in zip(m.row, m.col) if a < b]

        # adjacency for the current forest (we'll remove edges greedily)
        from collections import defaultdict

        def _components(edge_set):
            adj = defaultdict(list)
            for a, b in edge_set:
                adj[a].append(b); adj[b].append(a)
            seen = np.full(n, -1)
            lab = 0
            for s in range(n):
                if seen[s] != -1:
                    continue
                stack = [s]; seen[s] = lab
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if seen[v] == -1:
                            seen[v] = lab; stack.append(v)
                lab += 1
            return seen, lab

        def _ssd(idx):
            if idx.size == 0:
                return 0.0
            return float(((As[idx] - As[idx].mean(0)) ** 2).sum())

        def _split_sides(edge_set, drop):
            """nodes on each side of `drop` within its component (drop ∉ edge_set)."""
            a, b = drop
            adj = defaultdict(list)
            for e in edge_set:
                adj[e[0]].append(e[1]); adj[e[1]].append(e[0])
            # BFS from a (drop already excluded) -> side_a
            seen = {a}; stack = [a]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            return np.array(sorted(seen)), b in seen

        cur = set(edges)
        seen, ncomp = _components(cur)
        # greedily cut until we reach n_clusters components (or run out of edges)
        target = min(n_clusters, n)
        guard = 0
        while ncomp < target and cur and guard < n:
            guard += 1
            best_edge, best_gain = None, -np.inf
            comp_lab = seen
            for e in list(cur):
                # parent component node indices
                par = np.where(comp_lab == comp_lab[e[0]])[0]
                rest = cur - {e}
                side_a, b_in = _split_sides(rest, e)
                if b_in:  # removing e did not disconnect (shouldn't happen in a tree)
                    continue
                side_b = np.setdiff1d(par, side_a)
                gain = _ssd(par) - _ssd(side_a) - _ssd(side_b)
                if gain > best_gain:
                    best_gain, best_edge = gain, e
            if best_edge is None:
                break
            cur.discard(best_edge)
            seen, ncomp = _components(cur)

        labels = seen
        # within-cluster SSD total (homogeneity)
        ssd_total = float(sum(_ssd(np.where(labels == g)[0]) for g in range(ncomp)))
        sizes = np.bincount(labels, minlength=ncomp)

        estimates.update({
            "n_regions": float(ncomp), "requested_regions": float(n_clusters),
            "within_region_ssd": round(ssd_total, 6),
            "k_neighbors": float(k), "n": float(n),
            "min_region_size": float(int(sizes.min())), "max_region_size": float(int(sizes.max())),
        })

        import pandas as pd
        pd.DataFrame({cx: coords[:, 0], cy: coords[:, 1], "region": labels}).to_csv(
            d / "skater_regions.csv", index=False, encoding="utf-8")
        files.append("skater_regions.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=28)
            ax.set_xlabel(cx); ax.set_ylabel(cy)
            ax.set_title(f"SKATER regionalization ({ncomp} regions, k-NN={k})")
            try:
                fig.colorbar(sc, ax=ax, label="region")
            except Exception:
                pass
            fig.tight_layout(); fig.savefig(d / "skater_regions.png", dpi=150)
            plt.close(fig); files.append("skater_regions.png")
        except Exception:
            pass

        code += [
            "from scipy.sparse.csgraph import minimum_spanning_tree",
            "# SKATER: MST of k-NN graph (attr-dissimilarity weights), then greedily",
            "# prune edges maximising within-region SSD reduction until n_clusters regions",
        ]
        adj_note = (f"（请求 {n_clusters} 个，因邻接图连通分量限制实得 {ncomp} 个）"
                    if ncomp != n_clusters else "")
        summary.append(
            f"{entry.method}：基于 {len(feats)} 个变量在空间**连续**约束下分出 {ncomp} 个区域{adj_note}，"
            f"区内总平方和={ssd_total:.2f}（越小越同质），区大小 {int(sizes.min())}–{int(sizes.max())}（k-NN={k}）。"
            " ⚠ SKATER 强制每区空间连续（剪 MST），与普通聚类不同；结果依赖邻接定义（k-NN，已披露）"
            "与变量标准化；贪心剪枝是启发式（非全局最优）；区数为用户设定。"
        )
    except Exception as e:
        summary.append(f"SKATER 失败：{type(e).__name__}: {e}")
