"""Branch handlers for the distribution_extra family.

Three UNIVARIATE density / tail estimation methods on ONE numeric column. These
are DISTINCT from the `distribution` family (which fits NAMED distributions by MLE
and runs goodness-of-fit) and from the multivariate `gaussian_mixture` clustering
method — here we describe the *shape* of a single column's distribution and its
upper-tail behaviour:

  * finite_mixture — 1-D finite Gaussian mixture by EM, choose k=1..K by BIC;
                     reports component weights/means/sds + a multimodality verdict
  * kernel_density — Gaussian KDE; Silverman & Scott rule-of-thumb bandwidths,
                     mode detection (local maxima on a fine grid)
  * tail_index     — heavy-tail estimation via the Hill estimator on the upper
                     (or lower) tail; Hill plot + stable-region alpha + tail index

Each handler resolves the numeric column (cfg.get("column") else first continuous),
degrades honestly (no scipy/sklearn / too few rows / non-numeric / constant /
non-positive for Hill -> append a Chinese ⚠ message and return), writes CSV + PNG
(matplotlib Agg, ENGLISH plot labels), fills float `estimates`, appends a Chinese
`summary` with ⚠ disclosures, and mutates ctx (never rebinds).
See executor/_branch_api.py and CLAUDE.md.

Pure Python: numpy / scipy / scikit-learn / pandas (no R). EM uses a fixed seed.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Fixed RNG seed for EM (GaussianMixture) so runs are reproducible (disclosed).
_SEED = 0


def _resolve_column(ctx: Ctx):
    """Pick the numeric column to analyse (cfg['column'] else first continuous).

    Returns (column_name, values_ndarray, problem_msg). When problem_msg is not None
    the caller should append it to summary and return (honest degrade). Mirrors the
    column-resolution idiom of executor/branches/distribution.py."""
    import importlib.util

    if importlib.util.find_spec("scipy") is None:
        return None, None, "密度/尾部分析需要 scipy 包（未检测到）。安装：pip install scipy。"

    import numpy as np

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    col = cfg.get("column")
    if col is not None:
        if col not in df.columns:
            return None, None, f"密度/尾部分析失败：指定列 {col!r} 不在数据中。"
    else:
        cont = [c.name for c in fp.columns if c.kind == "continuous"]
        numeric = [c.name for c in fp.columns if c.kind in {"continuous", "count", "id"}]
        pick = cont or numeric
        if not pick:
            return None, None, "密度/尾部分析失败：未找到可用的数值列。用 config={\"column\":\"<列>\"} 指定。"
        col = pick[0]

    s = df[col]
    if not np.issubdtype(s.dropna().to_numpy().dtype, np.number):
        try:
            s = s.astype(float)
        except Exception:
            return None, None, f"密度/尾部分析失败：列 {col!r} 不是数值列。"
    x = s.dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None, None, f"密度/尾部分析跳过：列 {col!r} 有效样本不足（n={x.size}<8）。"
    if np.nanstd(x) == 0:
        return None, None, f"密度/尾部分析跳过：列 {col!r} 为常数列（无方差）。"
    return col, x, None


# ─────────────────────────────────────────────────────────────────────────────
# 1. finite_mixture — 1-D finite Gaussian mixture (EM), k chosen by BIC
# ─────────────────────────────────────────────────────────────────────────────
@register("finite_mixture")
def _branch_finite_mixture(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        summary.append("有限混合模型需要 scikit-learn 包（未检测到）。安装：pip install scikit-learn。")
        return

    col, x, problem = _resolve_column(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from sklearn.mixture import GaussianMixture

        n = x.size
        # cap K at the number of distinct values and at n//2 so EM has data per comp
        max_k = int(cfg.get("max_k", 5))
        max_k = max(1, max_k)
        n_unique = int(np.unique(x).size)
        k_cap = max(1, min(max_k, n_unique, n // 2))
        n_init = 5  # EM is init-sensitive; multiple restarts, best-LL kept (disclosed)

        X = x.reshape(-1, 1)
        bic_rows = []
        models = {}
        for k in range(1, k_cap + 1):
            try:
                gm = GaussianMixture(
                    n_components=k, covariance_type="full",
                    n_init=n_init, random_state=_SEED, max_iter=300,
                )
                gm.fit(X)
                bic = float(gm.bic(X))
                if not np.isfinite(bic):
                    continue
                models[k] = gm
                bic_rows.append({"k": k, "BIC": round(bic, 4),
                                 "AIC": round(float(gm.aic(X)), 4),
                                 "converged": bool(gm.converged_)})
            except Exception:
                continue

        if not models:
            summary.append(f"有限混合模型跳过：列 {col!r} 没有任何 k 拟合成功。")
            return

        bic_df = pd.DataFrame(bic_rows).sort_values("k").reset_index(drop=True)
        best_k = int(min(models, key=lambda k: models[k].bic(X)))
        best = models[best_k]
        best_bic = float(best.bic(X))
        bic_k1 = float(models[1].bic(X)) if 1 in models else float("nan")
        delta_bic_vs_k1 = (bic_k1 - best_bic) if np.isfinite(bic_k1) else float("nan")

        # per-component weight / mean / sd, sorted by mean for stable reporting
        weights = best.weights_.ravel()
        means = best.means_.ravel()
        sds = np.sqrt(best.covariances_.ravel())
        order = np.argsort(means)
        comp_df = pd.DataFrame({
            "component": np.arange(1, best_k + 1),
            "weight": np.round(weights[order], 6),
            "mean": np.round(means[order], 6),
            "sd": np.round(sds[order], 6),
        })
        largest_weight = float(np.max(weights))
        multimodal = best_k > 1

        comp_df.to_csv(d / "finite_mixture_components.csv", index=False, encoding="utf-8")
        files.append("finite_mixture_components.csv")
        bic_df.to_csv(d / "finite_mixture_bic.csv", index=False, encoding="utf-8")
        files.append("finite_mixture_bic.csv")

        # histogram (density) + fitted mixture pdf + each component
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.stats import norm as _norm

            grid = np.linspace(float(np.min(x)), float(np.max(x)), 500)
            mix_pdf = np.zeros_like(grid)
            for w, m, sdv in zip(weights, means, sds):
                mix_pdf += w * _norm.pdf(grid, m, max(sdv, 1e-12))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.hist(x, bins="auto", density=True, alpha=0.4, color="#999999",
                     label="data (density)")
            ax1.plot(grid, mix_pdf, color="#C44E52", lw=2.4,
                     label=f"mixture (k={best_k})")
            for ci, (w, m, sdv) in enumerate(zip(weights[order], means[order], sds[order]), 1):
                ax1.plot(grid, w * _norm.pdf(grid, m, max(sdv, 1e-12)),
                         lw=1.2, ls="--", alpha=0.8, label=f"comp {ci} (w={w:.2f})")
            ax1.set_xlabel(f"{col}")
            ax1.set_ylabel("density")
            ax1.set_title(f"Finite Gaussian mixture: {col} (k={best_k} by BIC)")
            ax1.legend(fontsize=8)

            ax2.plot(bic_df["k"], bic_df["BIC"], marker="o", color="#4C72B0", label="BIC")
            ax2.axvline(best_k, color="#C44E52", ls="--", lw=1.5, label=f"chosen k={best_k}")
            ax2.set_xlabel("number of components k")
            ax2.set_ylabel("BIC (lower = better)")
            ax2.set_title("BIC by k")
            ax2.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "finite_mixture.png", dpi=150)
            plt.close(fig)
            files.append("finite_mixture.png")
        except Exception:
            pass

        estimates["best_k"] = float(best_k)
        estimates["best_bic"] = round(best_bic, 4)
        estimates["delta_bic_vs_k1"] = (round(delta_bic_vs_k1, 4)
                                        if np.isfinite(delta_bic_vs_k1) else float("nan"))
        estimates["largest_weight"] = round(largest_weight, 6)
        estimates["n"] = float(n)

        dbic_txt = (f"{round(delta_bic_vs_k1, 2)}" if np.isfinite(delta_bic_vs_k1) else "NA")
        modal_note = (
            f"k>1 被选中 → 数据**多峰**（潜在子群）" if multimodal else
            "k=1 被选中 → 数据**单峰**（无证据支持子群）"
        )
        strong = np.isfinite(delta_bic_vs_k1) and delta_bic_vs_k1 > 10

        (d / "finite_mixture_summary.txt").write_text(
            f"有限高斯混合（EM，sklearn.mixture.GaussianMixture）：列 {col}，n={n}\n"
            f"在 k=1..{k_cap} 上按 BIC 选择，n_init={n_init}（多次随机重启取最优似然），随机种子={_SEED}\n"
            f"选中 k={best_k}（BIC={round(best_bic, 3)}）；与 k=1 的 BIC 差 ΔBIC={dbic_txt}"
            f"（>10 表示强烈支持多峰）—— {modal_note}\n"
            f"最大成分权重={round(largest_weight, 4)}\n"
            "注：EM 对初始化敏感（已用 n_init 多次重启 + 固定种子降低风险，但不保证全局最优）；"
            "BIC 选的是候选 k 之间的**相对**拟合，不证明绝对真值；这是**一维**密度形状/潜在子群分析，"
            "不是多维数据聚类（多维聚类请用 gaussian_mixture）；列/最大成分数可用 "
            "config={\"column\":\"...\",\"max_k\":K} 指定。\n\n"
            "各成分（按均值排序）：\n" + comp_df.to_string(index=False)
            + "\n\nBIC by k：\n" + bic_df.to_string(index=False),
            encoding="utf-8",
        )
        files.append("finite_mixture_summary.txt")

        summary.append(
            f"{entry.method} 完成（EM/BIC）：列 {col}，n={n}；在 k=1..{k_cap} 中选 k={best_k}"
            f"（BIC={round(best_bic, 2)}，ΔBIC vs k=1={dbic_txt}{'，强烈支持多峰' if strong else ''}）；"
            f"{modal_note}；最大成分权重={round(largest_weight, 3)}。"
            "⚠ EM 对初始化敏感（已用 n_init=5 重启+固定种子 0，但不保证全局最优）；BIC 只比较 k 间相对拟合，"
            "不证明绝对真值；这是一维密度/子群分析，不是多维聚类（多维用 gaussian_mixture）；列/max_k 可经 config 指定。"
        )
        code += [
            "import numpy as np",
            "from sklearn.mixture import GaussianMixture",
            f"x = df[{col!r}].dropna().to_numpy(float).reshape(-1, 1)",
            "models = {}",
            f"for k in range(1, {k_cap}+1):",
            f"    gm = GaussianMixture(n_components=k, n_init={n_init}, random_state={_SEED}).fit(x)",
            "    models[k] = gm  # compare gm.bic(x) across k; pick the smallest (lower BIC = better)",
            "best_k = min(models, key=lambda k: models[k].bic(x))",
        ]
    except Exception as err:
        summary.append(f"有限混合模型失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. kernel_density — Gaussian KDE, Silverman/Scott bandwidths, mode detection
# ─────────────────────────────────────────────────────────────────────────────
@register("kernel_density")
def _branch_kernel_density(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    col, x, problem = _resolve_column(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import gaussian_kde

        n = x.size
        std = float(np.std(x, ddof=1))
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        # rule-of-thumb bandwidths (Gaussian-kernel, in DATA units)
        # silverman_bw = Silverman's ROBUST rule (0.9·min(sd, IQR/1.349)·n^-1/5);
        # scott_bw = Scott's rule (1.0·sd·n^-1/5) = scipy gaussian_kde's actual "scott"
        # default factor·sd. (The 1.06·sd·n^-1/5 normal-reference rule is also Silverman,
        # NOT Scott — we report the genuine Scott so the name matches scipy's default.)
        sigma_robust = min(std, iqr / 1.349) if iqr > 0 else std
        silverman_bw = float(0.9 * sigma_robust * n ** (-1.0 / 5.0)) if sigma_robust > 0 else float("nan")
        scott_bw = float(std * n ** (-1.0 / 5.0)) if std > 0 else float("nan")

        # build the KDE — allow a numeric bandwidth override (config["bandwidth"])
        bw_override = cfg.get("bandwidth")
        if bw_override is not None:
            try:
                bw_val = float(bw_override)
                # scipy bw_method is a FACTOR on the data std; convert a data-unit bw
                bw_method = (bw_val / std) if std > 0 else None
            except Exception:
                bw_method = None
        else:
            bw_method = None  # scipy default = Scott's factor

        kde = gaussian_kde(x, bw_method=bw_method)
        used_bw_factor = float(kde.factor)
        used_bw_data = float(used_bw_factor * std) if std > 0 else float("nan")

        lo, hi = float(np.min(x)), float(np.max(x))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        grid = np.linspace(lo - pad, hi + pad, 1000)
        dens = kde(grid)

        # mode detection: strict interior local maxima of the KDE on the fine grid
        modes_idx = [i for i in range(1, len(grid) - 1)
                     if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]]
        # filter tiny ripples: keep modes whose density >= 1% of the global max
        if modes_idx:
            dmax = float(np.max(dens))
            modes_idx = [i for i in modes_idx if dens[i] >= 0.01 * dmax]
        mode_locs = [float(grid[i]) for i in modes_idx]
        mode_dens = [float(dens[i]) for i in modes_idx]
        n_modes = len(mode_locs)
        if n_modes == 0:  # plateau / extreme bw — fall back to the global argmax
            gi = int(np.argmax(dens))
            mode_locs = [float(grid[gi])]
            mode_dens = [float(dens[gi])]
            n_modes = 1
        primary_mode = float(mode_locs[int(np.argmax(mode_dens))])

        grid_df = pd.DataFrame({"x": np.round(grid, 6), "density": np.round(dens, 8)})
        grid_df.to_csv(d / "kernel_density_grid.csv", index=False, encoding="utf-8")
        files.append("kernel_density_grid.csv")
        modes_df = pd.DataFrame({
            "mode": np.arange(1, n_modes + 1),
            "location": np.round(mode_locs, 6),
            "density": np.round(mode_dens, 8),
        })
        modes_df.to_csv(d / "kernel_density_modes.csv", index=False, encoding="utf-8")
        files.append("kernel_density_modes.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(x, bins="auto", density=True, alpha=0.35, color="#999999",
                    label="data (density)")
            ax.plot(grid, dens, color="#4C72B0", lw=2.2, label="KDE")
            for ci, (loc, dn) in enumerate(zip(mode_locs, mode_dens), 1):
                ax.plot([loc], [dn], marker="v", color="#C44E52", ms=10,
                        label="mode" if ci == 1 else None)
                ax.axvline(loc, color="#C44E52", ls=":", lw=1, alpha=0.6)
            ax.set_xlabel(f"{col}")
            ax.set_ylabel("density")
            ax.set_title(f"Kernel density: {col} ({n_modes} mode(s))")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "kernel_density.png", dpi=150)
            plt.close(fig)
            files.append("kernel_density.png")
        except Exception:
            pass

        estimates["n_modes"] = float(n_modes)
        estimates["silverman_bw"] = round(silverman_bw, 6) if silverman_bw == silverman_bw else float("nan")
        estimates["scott_bw"] = round(scott_bw, 6) if scott_bw == scott_bw else float("nan")
        estimates["primary_mode"] = round(primary_mode, 6)
        estimates["n"] = float(n)

        modal_note = ("**多峰**（≥2 个局部极大）" if n_modes >= 2 else "**单峰**")
        bw_note = (f"用户指定带宽={float(bw_override)}（数据单位）" if bw_override is not None
                   else f"默认 Scott 带宽（数据单位≈{round(used_bw_data, 4) if used_bw_data == used_bw_data else 'NaN'}）")

        (d / "kernel_density_summary.txt").write_text(
            f"核密度估计（scipy.stats.gaussian_kde，高斯核）：列 {col}，n={n}\n"
            f"经验法则带宽（数据单位）：Silverman={round(silverman_bw, 5) if silverman_bw == silverman_bw else 'NaN'}，"
            f"Scott={round(scott_bw, 5) if scott_bw == scott_bw else 'NaN'}\n"
            f"实际使用：{bw_note}\n"
            f"在 1000 点网格上检测到 {n_modes} 个峰（局部极大）—— {modal_note}\n"
            f"主峰位置（密度最高）={round(primary_mode, 5)}\n"
            f"峰位置：{', '.join(f'{m:.4g}' for m in mode_locs)}\n"
            "注：KDE 形状高度依赖带宽（同时给出 Silverman 与 Scott 两种法则——过度平滑会抹掉峰、"
            "平滑不足会无中生有造出峰）；峰数量也依赖网格分辨率与带宽（已用 1000 点网格 + 1% 高度阈值滤波，"
            "属启发式，非严格检验）；列/带宽可用 config={\"column\":\"...\",\"bandwidth\":<数值>} 指定。\n\n"
            "峰：\n" + modes_df.to_string(index=False),
            encoding="utf-8",
        )
        files.append("kernel_density_summary.txt")

        summary.append(
            f"{entry.method} 完成：列 {col}，n={n}；Silverman 带宽="
            f"{round(silverman_bw, 4) if silverman_bw == silverman_bw else 'NaN'}、"
            f"Scott 带宽={round(scott_bw, 4) if scott_bw == scott_bw else 'NaN'}（{bw_note}）；"
            f"检测到 {n_modes} 个峰（{modal_note}），主峰≈{round(primary_mode, 4)}。"
            "⚠ KDE 形状取决于带宽（两种法则均给出——过平滑抹峰、欠平滑造峰）；峰数依赖网格/带宽（启发式，非检验）；"
            "列/带宽可经 config 指定。"
        )
        code += [
            "import numpy as np",
            "from scipy.stats import gaussian_kde",
            f"x = df[{col!r}].dropna().to_numpy(float); n = len(x)",
            "std = np.std(x, ddof=1); iqr = np.subtract(*np.percentile(x, [75, 25]))",
            "silverman = 0.9 * min(std, iqr/1.349) * n**(-1/5)  # rule-of-thumb bandwidths",
            "scott = 1.06 * std * n**(-1/5)",
            "kde = gaussian_kde(x)  # bw_method=<factor> to override",
            "grid = np.linspace(x.min(), x.max(), 1000); dens = kde(grid)",
            "modes = [i for i in range(1, len(grid)-1) if dens[i] > dens[i-1] and dens[i] > dens[i+1]]",
        ]
    except Exception as err:
        summary.append(f"核密度估计失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. tail_index — heavy-tail estimation via the Hill estimator
# ─────────────────────────────────────────────────────────────────────────────
@register("tail_index")
def _branch_tail_index(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    col, x, problem = _resolve_column(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        n_all = x.size
        if n_all < 50:
            summary.append(
                f"尾部指数（Hill）跳过：列 {col!r} 样本不足（n={n_all}<50），Hill 估计不可靠。"
            )
            return

        tail = str(cfg.get("tail", "upper")).lower()
        if tail not in {"upper", "lower"}:
            tail = "upper"

        # Hill needs a Pareto-type POSITIVE upper tail. For the lower tail we flip
        # the sign so the same machinery estimates the lower-tail index. Then we
        # require positivity; if non-positive remain, degrade honestly (do NOT
        # silently shift — that changes the tail).
        if tail == "lower":
            y = -x
            tail_desc = "下尾(lower)"
            tail_en = "lower tail"
        else:
            y = x.copy()
            tail_desc = "上尾(upper)"
            tail_en = "upper tail"

        if np.any(y <= 0):
            n_pos = int(np.sum(y > 0))
            if n_pos < 50:
                side = "正值" if tail == "upper" else "负值"
                summary.append(
                    f"尾部指数（Hill）跳过：{tail_desc}需要 Pareto 型正值尾，但列 {col!r} 的"
                    f"{side}样本不足（仅 {n_pos} 个）。Hill 假定正值尾，请改 config={{\"tail\":\"...\"}} "
                    "或先变换数据（如取绝对值/平移），不应静默平移以免改变尾部。"
                )
                return
            # keep only the positive side of the chosen tail
            y = y[y > 0]

        y_sorted = np.sort(y)[::-1]  # descending
        n = y_sorted.size

        # Hill estimator alpha_hat(k) for k = 1 .. k_max order statistics.
        # alpha_hat(k) = 1 / [ (1/k) * sum_{i=1}^{k} ln(X_(i) / X_(k+1)) ]
        # k_max bounded by k_frac of the sample (default 0.10) and < n-1.
        k_frac = float(cfg.get("k_frac", 0.10))
        k_frac = min(max(k_frac, 0.02), 0.5)
        k_max = max(5, min(int(k_frac * n), n - 2))

        log_x = np.log(y_sorted)
        cum = np.cumsum(log_x)  # cum[k-1] = sum of top-k logs
        ks = np.arange(1, k_max + 1)
        # mean of top-k logs minus log of the (k+1)-th order stat (the threshold)
        mean_top = cum[:k_max] / ks
        thresh_log = log_x[ks]  # X_(k+1) is index k (0-based) in descending order
        gamma = mean_top - thresh_log  # = 1/alpha (the tail index xi)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.where(gamma > 0, 1.0 / gamma, np.nan)

        hill_df = pd.DataFrame({
            "k": ks,
            "alpha_hat": np.round(alpha, 6),
            "xi_tail_index": np.round(gamma, 6),
        })
        hill_df.to_csv(d / "tail_index_hill.csv", index=False, encoding="utf-8")
        files.append("tail_index_hill.csv")

        # Stable-region heuristic: take the MIDDLE band of k (the central 20%-60%
        # quantiles of the k range) and use the MEDIAN alpha there. This avoids the
        # high-variance small-k region and the bias-prone large-k region. This is a
        # HEURISTIC, not an optimal threshold selector (e.g. Danielsson's bootstrap).
        lo_k = max(1, int(0.2 * k_max))
        hi_k = max(lo_k + 1, int(0.6 * k_max))
        band = alpha[lo_k - 1:hi_k]
        band = band[np.isfinite(band)]
        if band.size == 0:
            finite_alpha = alpha[np.isfinite(alpha)]
            if finite_alpha.size == 0:
                summary.append(f"尾部指数（Hill）跳过：列 {col!r} 的 Hill 估计全为非有限值（尾部退化）。")
                return
            alpha_hat = float(np.median(finite_alpha))
            k_used = int(np.median(ks))
        else:
            alpha_hat = float(np.median(band))
            k_used = int(round((lo_k + hi_k) / 2))
        xi = float(1.0 / alpha_hat) if alpha_hat > 0 else float("nan")
        heavy = 1.0 if (alpha_hat == alpha_hat and alpha_hat < 4.0) else 0.0

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ks, alpha, color="#4C72B0", lw=1.6, label="Hill alpha-hat(k)")
            ax.axvspan(lo_k, hi_k, color="#55A868", alpha=0.18,
                       label="stable-region band")
            ax.axhline(alpha_hat, color="#C44E52", ls="--", lw=1.5,
                       label=f"alpha-hat={alpha_hat:.3g}")
            ax.set_xlabel("number of upper order statistics k")
            ax.set_ylabel("Hill alpha-hat(k)  (tail index = 1/alpha)")
            ax.set_title(f"Hill plot: {col} ({tail_en})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "tail_index.png", dpi=150)
            plt.close(fig)
            files.append("tail_index.png")
        except Exception:
            pass

        estimates["alpha_hat"] = round(alpha_hat, 6)
        estimates["xi_tail_index"] = round(xi, 6) if xi == xi else float("nan")
        estimates["k_used"] = float(k_used)
        estimates["heavy_tail"] = float(heavy)
        estimates["n"] = float(n)

        heavy_note = (
            f"α̂≈{round(alpha_hat, 3)} < 4 → **重尾**（高阶矩可能不存在：α<4 时四阶矩发散，"
            "α<2 时方差发散，α<1 时均值发散）"
            if heavy else
            f"α̂≈{round(alpha_hat, 3)} ≥ 4 → 尾部不算重（前几阶矩存在）"
        )

        (d / "tail_index_summary.txt").write_text(
            f"尾部指数 / 重尾估计（Hill 估计量）：列 {col}，{tail_desc}，n（该尾正值）={n}\n"
            f"Hill 估计 α̂(k) 在 k=1..{k_max}（k_frac={round(k_frac, 3)}）上计算\n"
            f"稳定区间启发式：取 k∈[{lo_k},{hi_k}]（k 范围中段 20%–60%）的中位 α̂ = {round(alpha_hat, 4)}\n"
            f"隐含尾部指数 ξ = 1/α̂ = {round(xi, 4) if xi == xi else 'NaN'}；代表 k_used≈{k_used}\n"
            f"重尾判定：{heavy_note}\n"
            "注：Hill 估计**假定上尾为 Pareto 型且数据为正值**（下尾经符号翻转处理；含非正值时按上述诚实降级，"
            "不静默平移）；估计对 k 的选择高度敏感（本处用稳定区间**启发式**——中段中位数，"
            "非最优阈值选择如 Danielsson 自助法）；需要足够样本（n≥50）；α̂<4 提示重尾、高阶矩可能发散；"
            "列/尾向/k 比例可用 config={\"column\":\"...\",\"tail\":\"upper|lower\",\"k_frac\":0.1} 指定。\n\n"
            "Hill 表（前 25 行）：\n" + hill_df.head(25).to_string(index=False),
            encoding="utf-8",
        )
        files.append("tail_index_summary.txt")

        summary.append(
            f"{entry.method} 完成（Hill）：列 {col}，{tail_desc}，n={n}；"
            f"稳定区间 k∈[{lo_k},{hi_k}] 的中位 α̂={round(alpha_hat, 3)}，尾部指数 ξ=1/α̂="
            f"{round(xi, 3) if xi == xi else 'NaN'}（k_used≈{k_used}）；{heavy_note}。"
            "⚠ Hill 假定上尾 Pareto 型且数据为正（下尾翻号、非正值诚实降级而非静默平移）；"
            "对 k 选择高度敏感（用中段中位数**启发式**，非最优阈值如 Danielsson 自助）；需 n≥50；"
            "列/尾向/k_frac 可经 config 指定。"
        )
        code += [
            "import numpy as np",
            f"x = df[{col!r}].dropna().to_numpy(float)",
            "y = np.sort(x[x > 0])[::-1]  # upper tail, descending, positive (Pareto-type)",
            "logx = np.log(y); cum = np.cumsum(logx)",
            "k = np.arange(1, int(0.1*len(y)))  # range of order statistics",
            "gamma = cum[:len(k)]/k - logx[k]   # xi = 1/alpha (Hill estimator)",
            "alpha = 1.0/gamma                  # alpha < 4 => heavy tail",
            "alpha_hat = np.median(alpha[int(0.2*len(k)):int(0.6*len(k))])  # stable-region heuristic",
        ]
    except Exception as err:
        summary.append(f"尾部指数（Hill）失败：{err}")
