"""Branch handlers for the compositional family — closed data, where ordinary statistics lie.

Dogfood measurement that motivated this family, on a sand / silt / clay frame (rows summing
to 100) with the catalog as it stood:

    catalog methods for compositional data    0 of 306
    `correlation`                             rank 2 of 306, no warning of any kind
    measured raw correlations                 corr(sand,silt) = -0.292
                                              corr(sand,clay) = -0.657

Both negatives are artefacts. When the parts sum to a constant, every row satisfies
Σx = k, so for each component Σ_j cov(x_i, x_j) = 0: the covariances of any one part with
the others are FORCED to sum to zero, and most of them come out negative no matter what
relationship exists in the underlying quantities. Pearson described this in 1897; the engine
was handing it over as its top substantive method.

The Aitchison (1986) answer is to leave the simplex before doing any statistics:

  * compositional_profile — closure check, the centered log-ratio (CLR) transform, the
    VARIATION MATRIX τ_ij = Var(ln(x_i/x_j)) (scale-invariant, closure-free: τ→0 means two
    parts move proportionally), total variance, the Aitchison (closed geometric) mean, and a
    side-by-side of the raw correlations against the log-ratio ones so the spurious negatives
    are visible rather than merely described.
  * aitchison_pca — PCA on CLR coordinates: the standard compositional biplot. Ordinary PCA
    on raw parts inherits the same singular, closure-driven covariance structure (the last
    component is degenerate BY CONSTRUCTION, since the parts sum to a constant).

Zeros are the one genuine obstacle: ln(0) is undefined and a composition with a true zero is
not in the open simplex at all. Handled by multiplicative replacement (Martín-Fernández et
al. 2003) when zeros are rare, and by an honest refusal when they are not — never by quietly
adding a constant, which changes every ratio in the table.

Engine conventions (CLAUDE.md, executor/_branch_api.py): `@register("<id>") def _branch_<id>(ctx)`,
unpack ctx then MUTATE summary/estimates/files/code; honest degrade appends
"<方法>跳过：<原因>" and returns; products in try/except; ENGLISH plot labels; float estimates.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_MIN_ROWS = 20
_MAX_ZERO_FRAC = 0.10   # above this, replacement stops being a repair and becomes invention


def _components(fp, cfg, df, method: str):
    """The closed part-set. config `components` wins; else the profiler's shape fact."""
    cfg = cfg or {}
    forced = [c for c in (cfg.get("components") or []) if c in df.columns]
    comps = forced or list(getattr(fp, "closed_components", None) or [])
    comps = [c for c in comps if c in df.columns]
    if len(comps) < 3:
        return None, (
            f"{method}跳过：未检测到闭合成分（需要 ≥3 个非负数值列、其行和为常数）。"
            '若这确是成分数据，用 config={"components":[..]} 指定。'
        )
    return comps, None


def _clr(df, comps, summary, method: str):
    """(clr_matrix, closed_parts, problem). Zeros handled or honestly refused."""
    import numpy as np

    X = df[comps].apply(lambda s: s.astype(float)).dropna()
    if len(X) < _MIN_ROWS:
        return None, None, f"{method}跳过：有效行 {len(X)} < {_MIN_ROWS}"
    A = X.to_numpy(dtype=float)
    if (A < 0).any():
        return None, None, f"{method}跳过：成分含负值，不是组成部分"
    zero_frac = float((A == 0).mean())
    if zero_frac > _MAX_ZERO_FRAC:
        return None, None, (
            f"{method}跳过：{zero_frac:.1%} 的成分值为 0，超过 {_MAX_ZERO_FRAC:.0%}。"
            "对数比变换要求严格为正；这个比例下的零值替换是在**编造**数据而不是修补它"
            "（请先处理零值：合并稀有成分，或改用零值稳健的方法）。"
        )
    if zero_frac > 0:
        # multiplicative replacement (Martín-Fernández 2003): put a small delta in the zeros
        # and shrink the others proportionally, so every ROW still closes and the ratios among
        # the observed parts are preserved. Additive replacement would not preserve them.
        pos = A[A > 0]
        delta = 0.65 * float(pos.min())
        tot = A.sum(axis=1, keepdims=True)
        n_zero = (A == 0).sum(axis=1, keepdims=True)
        B = np.where(A == 0, delta, A * (1.0 - n_zero * delta / np.maximum(tot, 1e-12)))
        summary.append(
            f"⚠ {zero_frac:.1%} 的成分值为 0，已用**乘法替换**（δ=0.65×最小正值）处理："
            "零位填入 δ、其余按比例缩小，使每行仍然闭合且已观测部分之间的比值不变"
            "（加法替换做不到这一点）。零值处理会影响对数比结果，请知悉。"
        )
        A = B
    A = A / A.sum(axis=1, keepdims=True)          # close to 1 — CLR is scale-invariant anyway
    g = np.exp(np.log(A).mean(axis=1, keepdims=True))
    return np.log(A / g), A, None


def _variation_matrix(A):
    """τ_ij = Var(ln(x_i / x_j)) — Aitchison's closure-free measure of association."""
    import numpy as np

    L = np.log(A)
    D = L.shape[1]
    T = np.zeros((D, D))
    for i in range(D):
        for j in range(i + 1, D):
            v = float(np.var(L[:, i] - L[:, j], ddof=1))
            T[i, j] = T[j, i] = v
    return T


# ─────────────────────────────────────────────────────────────────────────────
@register("compositional_profile")
def _branch_compositional_profile(ctx: Ctx) -> None:
    df, fp, cfg, d = ctx.df, ctx.fp, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "成分数据结构画像"

    comps, problem = _components(fp, cfg, df, method)
    if problem:
        summary.append(problem)
        return
    import numpy as np

    Z, A, problem = _clr(df, comps, summary, method)
    if problem:
        summary.append(problem)
        return

    D = len(comps)
    T = _variation_matrix(A)
    total_var = float(T.sum() / (2 * D))
    gmean = np.exp(np.log(A).mean(axis=0))
    gmean = gmean / gmean.sum()

    raw = np.corrcoef(df[comps].dropna().to_numpy(dtype=float), rowvar=False)
    clr_corr = np.corrcoef(Z, rowvar=False)
    _off = [(raw[i, j], comps[i], comps[j]) for i in range(D) for j in range(i + 1, D)]
    n_neg = sum(1 for v, _, _ in _off if v < 0)

    estimates.update({
        "n_components": float(D),
        "n_rows": float(A.shape[0]),
        "total_variance": round(total_var, 6),
        "raw_corr_negative_pairs": float(n_neg),
        "raw_corr_pairs": float(len(_off)),
    })
    for i, c in enumerate(comps):
        estimates[f"aitchison_mean_{c}"] = round(float(gmean[i]), 6)
    _pairs = sorted(((T[i, j], comps[i], comps[j])
                     for i in range(D) for j in range(i + 1, D)))
    for v, a, b in _pairs[:3]:
        estimates[f"variation_{a}__{b}"] = round(float(v), 6)

    try:
        import pandas as pd

        pd.DataFrame(T, index=comps, columns=comps).to_csv(d / "variation_matrix.csv",
                                                           encoding="utf-8")
        files.append("variation_matrix.csv")
        pd.DataFrame({"component": comps, "aitchison_mean": gmean.round(6),
                      "arithmetic_mean": df[comps].mean().to_numpy().round(6)}).to_csv(
            d / "components.csv", index=False, encoding="utf-8")
        files.append("components.csv")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        im0 = axes[0].imshow(raw, cmap="coolwarm", vmin=-1, vmax=1)
        axes[0].set_title("Raw correlation (distorted by closure)")
        im1 = axes[1].imshow(T, cmap="viridis")
        axes[1].set_title("Variation matrix (closure-free)")
        for ax, im in ((axes[0], im0), (axes[1], im1)):
            ax.set_xticks(range(D))
            ax.set_xticklabels(comps, rotation=30, ha="right")
            ax.set_yticks(range(D))
            ax.set_yticklabels(comps)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(d / "compositional_structure.png", dpi=140)
        plt.close(fig)
        files.append("compositional_structure.png")
    except Exception:
        pass

    code += [
        "import numpy as np",
        f"parts = df[{comps!r}].astype(float).dropna()",
        "A = parts.to_numpy(); A = A / A.sum(axis=1, keepdims=True)",
        "clr = np.log(A / np.exp(np.log(A).mean(axis=1, keepdims=True)))  # CLR 坐标",
        "L = np.log(A)",
        "tau = np.array([[np.var(L[:,i]-L[:,j], ddof=1) for j in range(A.shape[1])]"
        " for i in range(A.shape[1])])  # 变差矩阵",
    ]

    strongest = _pairs[0] if _pairs else None
    summary.append(
        f"{method}完成：{D} 个闭合成分（{'、'.join(comps)}），n={A.shape[0]}，"
        f"行和恒定。总变差={total_var:.4f}；Aitchison（闭合几何）均值 = "
        + "、".join(f"{c}={gmean[i]:.3f}" for i, c in enumerate(comps)) + "。"
        + (f"变差最小的一对是 {strongest[1]}–{strongest[2]}（τ={strongest[0]:.4f}，"
           "τ 越接近 0 表示两成分越接近等比例变动）。" if strongest else "")
    )
    summary.append(
        f"⚠ **这组列上的普通相关系数不可解读**：行和恒定意味着对每个成分都有 "
        f"Σ_j cov(xᵢ,xⱼ)=0，各成分与其余成分的协方差被**强制**加总为零，于是多数成对相关"
        f"被推向负值——与底层量之间真实的关系无关。本数据 {len(_off)} 对里有 **{n_neg} 对为负**。"
        "这是 Pearson 1897 年就描述过的伪相关陷阱。请读**变差矩阵**"
        "（τ=Var(ln(xᵢ/xⱼ))，尺度不变、不受闭合影响），不要读左图。"
    )
    summary.append(
        "⚠ 对数比方法要求成分严格为正；成分数据的距离应使用 Aitchison 距离而非欧氏距离；"
        "子成分（只取部分成分重新闭合）会改变除变差矩阵之外的大多数统计量。"
    )


# ─────────────────────────────────────────────────────────────────────────────
@register("aitchison_pca")
def _branch_aitchison_pca(ctx: Ctx) -> None:
    df, fp, cfg, d = ctx.df, ctx.fp, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = "Aitchison PCA（CLR 坐标主成分）"

    comps, problem = _components(fp, cfg, df, method)
    if problem:
        summary.append(problem)
        return
    import numpy as np

    Z, A, problem = _clr(df, comps, summary, method)
    if problem:
        summary.append(problem)
        return

    D = len(comps)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = (S ** 2) / max(1, Zc.shape[0] - 1)
    ratio = var / var.sum() if var.sum() > 0 else var

    # raw-parts PCA, for the comparison that makes the point: its last component is
    # degenerate BY CONSTRUCTION, because the parts sum to a constant.
    R = df[comps].dropna().to_numpy(dtype=float)
    Rc = R - R.mean(axis=0, keepdims=True)
    raw_S = np.linalg.svd(Rc, compute_uv=False)
    raw_var = (raw_S ** 2) / max(1, Rc.shape[0] - 1)
    raw_ratio = raw_var / raw_var.sum() if raw_var.sum() > 0 else raw_var

    estimates.update({
        "n_components": float(D),
        "n_rows": float(A.shape[0]),
        "pc1_explained": round(float(ratio[0]), 6),
        "pc2_explained": round(float(ratio[1]), 6) if len(ratio) > 1 else 0.0,
        "pc1_pc2_explained": round(float(ratio[:2].sum()), 6),
        "clr_last_eigenvalue": round(float(var[-1]), 10),
        "raw_last_eigenvalue": round(float(raw_var[-1]), 10),
    })
    for i, c in enumerate(comps):
        estimates[f"pc1_loading_{c}"] = round(float(Vt[0, i]), 6)

    try:
        import pandas as pd

        pd.DataFrame(Vt[:min(3, len(Vt))].T, index=comps,
                     columns=[f"PC{i + 1}" for i in range(min(3, len(Vt)))]).to_csv(
            d / "clr_loadings.csv", encoding="utf-8")
        files.append("clr_loadings.csv")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sc = U[:, :2] * S[:2]
        ax.scatter(sc[:, 0], sc[:, 1], s=12, alpha=0.5, color="#4C72B0")
        k = 0.9 * max(abs(sc).max(), 1e-9)
        for i, c in enumerate(comps):
            ax.arrow(0, 0, k * Vt[0, i], k * Vt[1, i], color="#C44E52",
                     head_width=0.02 * k, length_includes_head=True)
            ax.text(k * Vt[0, i] * 1.08, k * Vt[1, i] * 1.08, c, color="#C44E52", fontsize=9)
        ax.set_xlabel(f"PC1 ({ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ratio[1]:.1%})" if len(ratio) > 1 else "PC2")
        ax.set_title("Compositional biplot (CLR coordinates)")
        ax.axhline(0, lw=0.5, color="#999")
        ax.axvline(0, lw=0.5, color="#999")
        fig.tight_layout()
        fig.savefig(d / "aitchison_biplot.png", dpi=140)
        plt.close(fig)
        files.append("aitchison_biplot.png")
    except Exception:
        pass

    code += [
        "import numpy as np",
        f"parts = df[{comps!r}].astype(float).dropna()",
        "A = parts.to_numpy(); A = A / A.sum(axis=1, keepdims=True)",
        "clr = np.log(A / np.exp(np.log(A).mean(axis=1, keepdims=True)))",
        "Zc = clr - clr.mean(axis=0, keepdims=True)",
        "U, S, Vt = np.linalg.svd(Zc, full_matrices=False)  # PCA on CLR coordinates",
    ]

    summary.append(
        f"{method}完成：{D} 个闭合成分、n={A.shape[0]}。"
        f"PC1 解释 {ratio[0]:.1%}"
        + (f"、PC1+PC2 解释 {ratio[:2].sum():.1%}" if len(ratio) > 1 else "")
        + "。PC1 载荷 = " + "、".join(f"{c}={Vt[0, i]:+.3f}" for i, c in enumerate(comps)) + "。"
    )
    summary.append(
        "⚠ 这是在 **CLR 坐标**上做的 PCA，不是在原始成分上。对闭合数据直接做 PCA 会继承"
        "闭合造成的奇异协方差结构——**最后一个主成分按构造就是退化的**（各部分之和为常数）。"
        f"本数据实测：原始成分的最小特征值 = {raw_var[-1]:.3g}。"
        "CLR 坐标本身也有一个零和约束，所以 CLR 的最小特征值同样接近 0"
        f"（实测 {var[-1]:.3g}）——这是变换的已知性质，"
        "解释时只看前 D−1 个方向即可。"
    )
    summary.append(
        "⚠ 双标图中箭头之间的**夹角**反映对数比关联，箭头长度反映该成分在所选平面上的变差；"
        "点之间的距离是 Aitchison 距离，不能按欧氏距离解读。"
    )
