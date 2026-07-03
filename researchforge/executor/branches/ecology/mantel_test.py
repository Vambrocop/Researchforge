"""mantel_test — correlation between two distance matrices with a permutation
p-value (community/species vs environmental/spatial distances). Pure numpy/scipy."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("mantel_test")
def _branch_mantel_test(ctx: Ctx) -> None:
    """Mantel test: correlation between two distance matrices with a permutation
    p-value. Community/species distances (Bray-Curtis or Euclidean) vs
    environmental/spatial distances (Euclidean), correlated on the lower triangles
    with a label-permutation null. Pure numpy/scipy."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    _excl = {fp.unit_col, fp.time_col}
    # Auto-defaults: community = count columns (species abundances); env/spatial =
    # continuous columns (measured gradients / coordinates). Overridable via config.
    auto_comm = [
        c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
    ]
    auto_env = [
        c.name
        for c in fp.columns
        if c.kind in {"continuous", "geo"} and c.name not in _excl
    ]
    comm_cols = list(cfg.get("community") or cfg.get("species") or auto_comm)
    env_cols = list(cfg.get("env") or cfg.get("spatial") or auto_env)
    # config metric for the community matrix: "braycurtis" (default for counts) or
    # "euclidean". The second (env/spatial) matrix is always Euclidean.
    comm_metric = str(cfg.get("metric") or "braycurtis").lower()
    if comm_metric not in {"braycurtis", "euclidean"}:
        comm_metric = "braycurtis"
    method = str(cfg.get("corr") or "pearson").lower()
    if method not in {"pearson", "spearman"}:
        method = "pearson"
    try:
        n_perm = int(cfg.get("n_perm") or 999)
    except (TypeError, ValueError):
        n_perm = 999
    n_perm = max(99, min(n_perm, 9999))

    # keep only columns that actually exist
    comm_cols = [c for c in comm_cols if c in df.columns]
    env_cols = [c for c in env_cols if c in df.columns]

    if len(comm_cols) < 2 or len(env_cols) < 1:
        summary.append(
            "Mantel 检验跳过：需要 ≥2 个群落/物种列（计数）与 ≥1 个环境/空间列（连续）。"
            f"（自动检出 community={comm_cols}，env={env_cols}；可用 config "
            "community/env 指定。）"
        )
        return

    sub = df[comm_cols + env_cols].dropna()
    if len(sub) < 4:
        summary.append("Mantel 检验跳过：去缺失后有效样点 <4，距离矩阵自由度不足。")
        return

    comm_mat = sub[comm_cols].astype(float).to_numpy()
    if comm_metric == "braycurtis":
        comm_mat = np.clip(comm_mat, 0, None)
        keep = comm_mat.sum(axis=1) > 0  # Bray-Curtis undefined for empty rows
        comm_mat = comm_mat[keep]
        env_mat = sub[env_cols].astype(float).to_numpy()[keep]
    else:
        env_mat = sub[env_cols].astype(float).to_numpy()
    n = len(comm_mat)
    if n < 4:
        summary.append("Mantel 检验跳过：有效（非空）样点 <4。")
        return

    try:
        D_comm = squareform(pdist(comm_mat, metric=comm_metric))
        # z-score each env column before euclidean pdist: unstandardized columns on
        # different scales let the large-magnitude column dominate the distance,
        # making r/p unit-dependent (guard std==0 -> leave that column centered at 0).
        env_std = env_mat.std(axis=0)
        env_std_safe = np.where(env_std > 0, env_std, 1.0)
        env_z = (env_mat - env_mat.mean(axis=0)) / env_std_safe
        D_env = squareform(pdist(env_z, metric="euclidean"))

        iu = np.triu_indices(n, k=1)
        x = D_comm[iu]
        y = D_env[iu]

        if method == "spearman":
            from scipy.stats import rankdata

            x = rankdata(x)
            y = rankdata(y)

        # standardized Mantel statistic r = corr(lower-triangle vectors)
        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            return float((a * b).sum() / denom) if denom > 0 else float("nan")

        r_obs = _corr(x, y)
        if not np.isfinite(r_obs):
            summary.append("Mantel 检验失败：距离矩阵无变异（常数距离），相关无定义。")
            return

        # permutation null: jointly permute rows/cols of one matrix (Mantel 1967),
        # two-sided test for association (|r| >= |r_obs|).
        rng = np.random.default_rng(0)
        ge = 0
        r_abs = abs(r_obs)
        for _ in range(n_perm):
            perm = rng.permutation(n)
            xp = D_comm[np.ix_(perm, perm)][iu]
            if method == "spearman":
                from scipy.stats import rankdata as _rd

                xp = _rd(xp)
            if abs(_corr(xp, y)) >= r_abs:
                ge += 1
        p_value = (ge + 1) / (n_perm + 1)

        pd.DataFrame(
            [{
                "mantel_r": round(r_obs, 4),
                "p_value": round(p_value, 4),
                "correlation": method,
                "community_metric": comm_metric,
                "env_metric": "euclidean",
                "n_sites": n,
                "n_perm": n_perm,
            }]
        ).to_csv(d / "mantel_result.csv", index=False, encoding="utf-8")
        files.append("mantel_result.csv")

        estimates["mantel_r"] = round(r_obs, 4)
        estimates["p_value"] = round(p_value, 4)
        estimates["n_sites"] = float(n)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(D_comm[iu], D_env[iu], s=14, alpha=0.6, c="#4C72B0")
            ax.set_xlabel(f"community distance ({comm_metric})")
            ax.set_ylabel("environmental/spatial distance (euclidean)")
            ax.set_title(f"Mantel test (r={r_obs:.3f}, p={p_value:.3f})")
            fig.tight_layout()
            fig.savefig(d / "mantel_scatter.png", dpi=150)
            plt.close(fig)
            files.append("mantel_scatter.png")
        except Exception:
            pass

        sig = "显著" if p_value < 0.05 else "不显著"
        summary.append(
            f"{entry.method} 完成：{n} 个样点，群落距离({comm_metric}) vs "
            f"环境/空间距离(euclidean) 的 Mantel {method} r={r_obs:.3f}，"
            f"p={p_value:.3f}（{n_perm} 次置换，{sig}）。"
            "⚠ Mantel 检验的是两个距离矩阵的相关、非因果；"
            "⚠ 距离度量选择会改变结论（计数群落默认 Bray-Curtis，可 config metric/corr 切换）；"
            "⚠ 空间自相关会使 Mantel r 偏高/p 偏小（样点非独立）——若 env 即地理距离，"
            "考虑偏 Mantel（控制空间）或谨慎解读；"
            "⚠ 环境/空间变量在计算欧氏距离前已逐列标准化（z-score），避免量纲/尺度不同时"
            "大数值列主导距离矩阵；"
            f"⚠ 列分组按 community={comm_cols[:3]}{'...' if len(comm_cols) > 3 else ''} / "
            f"env={env_cols} 自动划分，可用 config 覆盖。"
        )
        code += [
            "import numpy as np",
            "from scipy.spatial.distance import pdist, squareform",
            f"D_comm = squareform(pdist(df[{comm_cols!r}].values, metric={comm_metric!r}))",
            f"env_mat = df[{env_cols!r}].values",
            "env_z = (env_mat - env_mat.mean(axis=0)) / env_mat.std(axis=0)  # z-score env cols",
            "D_env = squareform(pdist(env_z, metric='euclidean'))",
            "iu = np.triu_indices(len(D_comm), k=1); x, y = D_comm[iu], D_env[iu]",
            "r_obs = np.corrcoef(x, y)[0, 1]  # Mantel statistic (Pearson on triangles)",
            "# permutation: jointly permute rows/cols of one matrix, recompute |r|, count >= |r_obs|",
        ]
    except Exception as err:
        summary.append(f"Mantel 检验失败：{err}")
