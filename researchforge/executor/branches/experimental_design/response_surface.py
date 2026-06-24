"""Experimental-design family branch handler: response_surface (split from experimental_design.py)."""
from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("response_surface")
def _branch_response_surface(ctx: Ctx) -> None:
    """RSM = Response Surface Methodology. Fits a second-order polynomial
    response ~ x1 + x2 + ... + x1² + ... + x1:x2 + ..., locates the stationary point
    (∇=0 → solve 2B·x = -b), and classifies it via the Hessian (2B) eigenvalues
    (all<0 max / all>0 min / mixed saddle). Draws a contour plot over the first two
    factors. Config: outcome + factors (list of ≥2 continuous columns)."""
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import itertools

    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]

    y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
    # factors: config list (any continuous columns ≠ outcome), else remaining continuous
    fac_cfg = cfg.get("factors")
    if isinstance(fac_cfg, (list, tuple)):
        factors = [c for c in fac_cfg if c in df.columns and c != y]
    else:
        factors = []
    guessed = not factors
    if not factors:
        factors = [c for c in cont if c != y]

    if y is None or len(factors) < 2:
        summary.append(
            "响应面 RSM 失败：需要 1 个连续结果 + ≥2 个连续因子列。"
            '用 config={"outcome":..,"factors":["x1","x2",...]} 指定。'
        )
        return

    sub = df[[y] + factors].dropna()
    # need enough rows + variation in each factor for a second-order fit
    n_params = 1 + 2 * len(factors) + len(factors) * (len(factors) - 1) // 2  # const+lin+quad+cross
    nuniq = {f: int(sub[f].nunique()) for f in factors}
    if len(sub) <= n_params or any(v < 3 for v in nuniq.values()):
        summary.append(
            f"响应面 RSM 失败：二阶模型需 {n_params} 个参数、有效行 {len(sub)}（需 > 参数数），"
            f"且每个因子至少 3 个不同水平（当前 {nuniq}）。请提供设计过的因子水平（如 CCD/Box-Behnken）。"
        )
        return

    try:
        import statsmodels.formula.api as smf

        # center factors to stabilize the quadratic fit; map back to raw scale for the
        # stationary point. Build a design matrix manually (robust to odd column names).
        Xc = sub[factors].to_numpy(dtype=float)
        ctr = Xc.mean(axis=0)
        Xcen = Xc - ctr[None, :]
        yv = sub[y].to_numpy(dtype=float)
        kf = len(factors)

        # design: intercept, linear (k), quadratic (k), cross terms (k choose 2)
        cols = [np.ones(len(sub))]
        names = ["const"]
        for i in range(kf):
            cols.append(Xcen[:, i]); names.append(f"L{i}")
        for i in range(kf):
            cols.append(Xcen[:, i] ** 2); names.append(f"Q{i}")
        cross_pairs = list(itertools.combinations(range(kf), 2))
        for (i, j) in cross_pairs:
            cols.append(Xcen[:, i] * Xcen[:, j]); names.append(f"X{i}_{j}")
        D = np.column_stack(cols)
        Ddf = pd.DataFrame(D, columns=names)
        Ddf["__y__"] = yv
        model = smf.ols("__y__ ~ " + " + ".join(names[1:]), data=Ddf).fit()
        r2 = float(model.rsquared)
        if not np.isfinite(r2):
            summary.append("响应面 RSM 失败：二阶模型不可估（设计矩阵奇异/共线）。需设计过的因子水平。")
            return

        beta = model.params
        b = np.array([float(beta.get(f"L{i}", 0.0)) for i in range(kf)])     # linear coefs
        # Hessian / B matrix: diagonal = 2*quad? In y = b0 + b'x + x'B x, quad coef = B_ii,
        # cross coef = 2 B_ij. Stationary point: x_s = -0.5 * B^{-1} b.
        B = np.zeros((kf, kf))
        for i in range(kf):
            B[i, i] = float(beta.get(f"Q{i}", 0.0))
        for (i, j) in cross_pairs:
            half = 0.5 * float(beta.get(f"X{i}_{j}", 0.0))
            B[i, j] = half
            B[j, i] = half

        # gradient = b + 2 B x = 0 → x_s = -0.5 B^{-1} b (in centered coords)
        eig = np.linalg.eigvalsh(B)
        stationary_ok = bool(np.all(np.abs(eig) > 1e-9))
        if stationary_ok:
            x_s_cen = np.linalg.solve(2.0 * B, -b)
            x_s = x_s_cen + ctr
            # predicted response at stationary point: b0 + b'x_s + x_s' B x_s (centered)
            b0 = float(beta.get("const", beta.get("Intercept", 0.0)))
            y_s = b0 + float(b @ x_s_cen) + float(x_s_cen @ B @ x_s_cen)
        else:
            x_s_cen = np.full(kf, np.nan)
            x_s = np.full(kf, np.nan)
            y_s = float("nan")

        # classify via Hessian (2B) eigenvalues — same sign pattern as B
        if np.all(eig < -1e-9):
            kind = "maximum"; kind_zh = "极大值(凸顶)"
        elif np.all(eig > 1e-9):
            kind = "minimum"; kind_zh = "极小值(凹底)"
        elif stationary_ok:
            kind = "saddle"; kind_zh = "鞍点(混合曲率)"
        else:
            kind = "ridge/degenerate"; kind_zh = "脊/退化(近零特征值)"

        # in-region check: is the stationary point inside the observed factor box?
        in_region = bool(np.all((x_s >= Xc.min(axis=0)) & (x_s <= Xc.max(axis=0)))) if stationary_ok else False

        rows = []
        for i, f in enumerate(factors):
            rows.append({"factor": f,
                         "stationary_point": float(x_s[i]) if stationary_ok else float("nan"),
                         "factor_min": float(Xc[:, i].min()),
                         "factor_max": float(Xc[:, i].max())})
            estimates[f"stationary_{f}"] = float(x_s[i]) if stationary_ok else float("nan")
        pd.DataFrame(rows).to_csv(d / "rsm_stationary_point.csv", index=False, encoding="utf-8")
        files.append("rsm_stationary_point.csv")

        coef_tbl = pd.DataFrame({"term": list(model.params.index),
                                 "coef": [float(v) for v in model.params.values],
                                 "p_value": [float(v) for v in model.pvalues.values]})
        coef_tbl.to_csv(d / "rsm_coefficients.csv", index=False, encoding="utf-8")
        files.append("rsm_coefficients.csv")

        estimates["r_squared"] = r2
        estimates["n_factors"] = float(kf)
        estimates["stationary_response"] = float(y_s) if y_s == y_s else float("nan")
        estimates["stationary_in_region"] = 1.0 if in_region else 0.0
        for i, ev in enumerate(eig):
            estimates[f"hessian_eig{i+1}"] = float(ev)

        # contour plot over the first two factors (others held at their stationary/centered value)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            i0, i1 = 0, 1
            x0 = np.linspace(Xc[:, i0].min(), Xc[:, i0].max(), 60)
            x1 = np.linspace(Xc[:, i1].min(), Xc[:, i1].max(), 60)
            XX, YY = np.meshgrid(x0, x1)
            # build grid design rows; other factors fixed at center (=0 in centered coords)
            base = np.zeros(kf)
            if stationary_ok:
                base = x_s_cen.copy()
            grid_cen = np.zeros((XX.size, kf))
            grid_cen[:] = base[None, :]
            grid_cen[:, i0] = XX.ravel() - ctr[i0]
            grid_cen[:, i1] = YY.ravel() - ctr[i1]
            gcols = {f"L{i}": grid_cen[:, i] for i in range(kf)}
            gcols.update({f"Q{i}": grid_cen[:, i] ** 2 for i in range(kf)})
            for (i, j) in cross_pairs:
                gcols[f"X{i}_{j}"] = grid_cen[:, i] * grid_cen[:, j]
            Zpred = model.predict(pd.DataFrame(gcols)).to_numpy().reshape(XX.shape)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            cs = ax.contourf(XX, YY, Zpred, levels=15, cmap="viridis")
            fig.colorbar(cs, ax=ax, label=f"predicted {y}")
            ax.scatter(Xc[:, i0], Xc[:, i1], c="white", s=12, edgecolors="k", lw=0.4, label="design points")
            if stationary_ok and in_region:
                ax.scatter([x_s[i0]], [x_s[i1]], c="red", s=80, marker="*", label=f"stationary ({kind})")
            ax.set_xlabel(factors[i0])
            ax.set_ylabel(factors[i1])
            ax.set_title(f"Response surface — {y}")
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(d / "rsm_contour.png", dpi=150)
            plt.close(fig)
            files.append("rsm_contour.png")
        except Exception:
            pass

        role_note = "（factors 自动取全部连续列，建议 config 明确 factors 列表）" if guessed else ""
        if stationary_ok:
            sp_txt = "、".join(f"{f}={x_s[i]:.4g}" for i, f in enumerate(factors))
            region_txt = "在设计区域内" if in_region else "⚠ 落在设计区域外（外推，不可信）"
            sp_summary = f"驻点 {kind_zh} 于 {sp_txt}（预测 {y}≈{y_s:.4g}，{region_txt}）。"
        else:
            sp_summary = "驻点不可解（Hessian 近奇异 → 脊系统/退化曲面，无唯一最优）。"
        summary.append(
            f"{entry.method} 完成{role_note}：{y} ~ 二阶多项式({kf} 因子：线性+平方+交互项)；R²={r2:.3f}。"
            f"{sp_summary}Hessian 特征值={np.array2string(eig, precision=3)}。"
            " ⚠ RSM 是设计区域内的**局部二次近似**；驻点仅在因子水平覆盖的区域内有效（区域外为外推、不可信）；"
            "需**设计过的因子水平**(如中心复合 CCD / Box-Behnken)，观测性数据的因子常共线导致曲面不可估或脊；"
            "等高线图固定其余因子于驻点/中心，残差正态/等方差假定。"
        )
        code += [
            "import numpy as np, statsmodels.formula.api as smf",
            "# fit y ~ x1+x2+...+ x1^2+...+ x1:x2+...  (centered factors)",
            "# stationary point: x_s = -0.5 * B^{-1} b ; classify via eig(B): <0 max / >0 min / mixed saddle",
        ]
    except Exception as err:
        summary.append(f"响应面 RSM 失败：{err}")
