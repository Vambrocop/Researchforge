"""Branch handlers for the spatial family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _knn_k,
    _kriging_via_r,
    _spatial_reg_via_r,
)


@register("getis_ord_gi")
def _branch_getis_ord_gi(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    value = next(
        (
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ),
        None,
    )
    if len(geo) < 2 or value is None:
        summary.append("Getis-Ord Gi* 失败：需要经纬度坐标 + 一个连续值变量。")
    else:
        sub = df[[geo[0], geo[1], value]].dropna()
        coords = sub[[geo[0], geo[1]]].to_numpy(dtype=float)
        x = sub[value].to_numpy(dtype=float)
        n = len(x)
        xbar = x.mean()
        S = float(np.sqrt((x**2).mean() - xbar**2))
        if n < 10 or S == 0:
            summary.append("Getis-Ord Gi* 失败：样本不足（<10）或值变量为常数。")
        else:
            # k+1 (neighbours + self) must stay < n, else the Gi* variance term
            # n·Σw² − (Σw)² collapses to 0 (every point a neighbour of every
            # point → no spatial contrast); n-2 keeps it strictly positive.
            k = _knn_k(cfg, n - 2)  # config={"knn_k": N} 覆盖近邻数
            d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = np.argsort(d2, axis=1)[:, :k]
            W = np.zeros((n, n))
            W[np.repeat(np.arange(n), k), nn.ravel()] = 1.0
            np.fill_diagonal(W, 1.0)  # Gi* (star) includes the focal point
            Wsum = W.sum(axis=1)
            Wsq = (W**2).sum(axis=1)
            num = W @ x - xbar * Wsum
            den = S * np.sqrt((n * Wsq - Wsum**2) / (n - 1))
            gi = num / den  # asymptotically standard-normal z-score per location
            hot, cold = gi > 1.96, gi < -1.96

            import pandas as pd

            pd.DataFrame(
                {
                    geo[0]: coords[:, 0],
                    geo[1]: coords[:, 1],
                    value: x,
                    "gi_star": np.round(gi, 4),
                    "class": np.where(hot, "hotspot", np.where(cold, "coldspot", "ns")),
                }
            ).to_csv(d / "getis_ord.csv", index=False, encoding="utf-8")
            files.append("getis_ord.csv")

            # map orientation: longitude on x, latitude on y when detectable
            lon_i = 1 if ("lon" in geo[1].lower() or "lng" in geo[1].lower()) else 0
            lat_i = 1 - lon_i
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(
                    coords[:, lon_i], coords[:, lat_i], c=gi, cmap="RdBu_r",
                    vmin=-3, vmax=3, s=28, edgecolor="#444444", linewidth=0.3,
                )
                fig.colorbar(sc, label="Gi* z-score")
                ax.set_xlabel(geo[lon_i])
                ax.set_ylabel(geo[lat_i])
                ax.set_title(f"Getis-Ord Gi* hotspots — {value}")
                fig.tight_layout()
                fig.savefig(d / "hotspot_map.png", dpi=150)
                plt.close(fig)
                files.append("hotspot_map.png")
            except Exception:
                pass

            estimates["n_hotspots"] = float(int(hot.sum()))
            estimates["n_coldspots"] = float(int(cold.sum()))
            estimates["max_gi"] = round(float(gi.max()), 4)
            estimates["min_gi"] = round(float(gi.min()), 4)
            summary.append(
                f"{entry.method} 完成：变量 {value}，{int(hot.sum())} 个热点 / "
                f"{int(cold.sum())} 个冷点（|Gi*|>1.96，k-NN={k}）；"
                "Gi* 为每点 z 分数，正=高值聚集、负=低值聚集"
            )
            code += [
                "import numpy as np  # Getis-Ord Gi* (star, includes focal point)",
                f"# coords={geo}, value='{value}', k={k}, binary kNN+self weights",
                "# Gi* = (Wx - xbar*sum_w) / (S*sqrt((n*sum_w2 - sum_w^2)/(n-1)))",
            ]



@register("idw_interpolation")
def _branch_idw_interpolation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    value = next(
        (
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ),
        None,
    )
    if len(geo) < 2 or value is None:
        summary.append("IDW 插值失败：需要经纬度坐标 + 一个连续值变量。")
    else:
        sub = df[[geo[0], geo[1], value]].dropna()
        coords = sub[[geo[0], geo[1]]].to_numpy(dtype=float)
        v = sub[value].to_numpy(dtype=float)
        n = len(v)
        if n < 5:
            summary.append("IDW 插值失败：有效样本不足（<5）。")
        else:
            power = 2.0
            # leave-one-out cross-validation RMSE (honest accuracy estimate)
            errs = []
            for i in range(n):
                mask = np.arange(n) != i
                di = np.maximum(np.sqrt(((coords[i] - coords[mask]) ** 2).sum(1)), 1e-12)
                w = 1.0 / di**power
                errs.append((w * v[mask]).sum() / w.sum() - v[i])
            rmse = float(np.sqrt(np.mean(np.asarray(errs) ** 2)))

            lon_i = 1 if ("lon" in geo[1].lower() or "lng" in geo[1].lower()) else 0
            lat_i = 1 - lon_i
            pts = coords[:, [lon_i, lat_i]]  # (lon, lat)
            G = 60
            glon = np.linspace(pts[:, 0].min(), pts[:, 0].max(), G)
            glat = np.linspace(pts[:, 1].min(), pts[:, 1].max(), G)
            gx, gy = np.meshgrid(glon, glat)
            gp = np.stack([gx.ravel(), gy.ravel()], axis=1)
            dist = np.maximum(np.sqrt(((gp[:, None, :] - pts[None, :, :]) ** 2).sum(-1)), 1e-12)
            wgrid = 1.0 / dist**power
            surf = ((wgrid @ v) / wgrid.sum(axis=1)).reshape(G, G)

            import pandas as pd

            pd.DataFrame(
                {geo[lon_i]: gp[:, 0], geo[lat_i]: gp[:, 1], f"{value}_idw": surf.ravel().round(4)}
            ).to_csv(d / "idw_surface.csv", index=False, encoding="utf-8")
            files.append("idw_surface.csv")

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(
                    surf, extent=[glon.min(), glon.max(), glat.min(), glat.max()],
                    origin="lower", cmap="YlOrBr", aspect="auto",
                )
                ax.scatter(
                    pts[:, 0], pts[:, 1], c=v, cmap="YlOrBr", s=22,
                    edgecolor="#222222", linewidth=0.4,
                )
                fig.colorbar(im, label=value)
                ax.set_xlabel(geo[lon_i])
                ax.set_ylabel(geo[lat_i])
                ax.set_title(f"IDW interpolated surface — {value}")
                fig.tight_layout()
                fig.savefig(d / "idw_surface.png", dpi=150)
                plt.close(fig)
                files.append("idw_surface.png")
            except Exception:
                pass

            estimates["loo_rmse"] = round(rmse, 4)
            estimates["power"] = power
            estimates["n_points"] = float(n)
            vrange = float(v.max() - v.min())
            rel = f"（≈值域 {vrange:.3g} 的 {100*rmse/vrange:.1f}%）" if vrange > 0 else ""
            summary.append(
                f"{entry.method} 完成：{value} 在 {n} 个采样点上插值为 {G}×{G} 栅格面，"
                f"留一交叉验证 RMSE={rmse:.4g}{rel}（幂={power:g}）"
            )
            code += [
                "import numpy as np  # Inverse Distance Weighting (power=2)",
                "# surf = sum(v_i / d_i^p) / sum(1/d_i^p); LOO-CV for RMSE",
            ]



@register("kriging")
def _branch_kriging(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    value = next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
        None,
    )
    lon = next((g for g in geo if "lon" in g.lower() or "lng" in g.lower()), geo[-1] if geo else None)
    lat = next((g for g in geo if g != lon), geo[0] if geo else None)
    names_safe = value is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [value, *geo]
    )
    if len(geo) < 2 or value is None:
        summary.append("克里金失败：需要经纬度坐标 + 一个连续值变量。")
    elif not (rbridge.r_available() and rbridge.r_package_available("gstat")):
        summary.append(
            "克里金需要 R 的 gstat 包（未检测到）。安装：install.packages('gstat')；"
            "或用 idw_interpolation（纯 Python，无需 R）。"
        )
    elif not names_safe:
        summary.append("克里金失败：列名需为标识符式（字母/数字/. _）。")
    else:
        import numpy as np

        sub = df[[*geo, value]].dropna()
        csv = d / "_krig_input.csv"
        sub.to_csv(csv, index=False)
        try:
            meta, grid = _kriging_via_r(csv, lon, lat, value)
            grid["kriging_variance"] = grid["kriging_variance"].clip(lower=0.0)  # numerical guard
            grid.to_csv(d / "kriged_surface.csv", index=False, encoding="utf-8")
            files.append("kriged_surface.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                piv_p = grid.pivot(index=lat, columns=lon, values="prediction")
                piv_v = grid.pivot(index=lat, columns=lon, values="kriging_variance")
                ext = [grid[lon].min(), grid[lon].max(), grid[lat].min(), grid[lat].max()]
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                im0 = axes[0].imshow(piv_p.values, extent=ext, origin="lower", cmap="YlOrBr", aspect="auto")
                axes[0].scatter(sub[lon], sub[lat], c=sub[value], cmap="YlOrBr", s=16, edgecolor="#222", linewidth=0.3)
                fig.colorbar(im0, ax=axes[0], label=value)
                axes[0].set_title(f"Kriged surface — {value}")
                im1 = axes[1].imshow(piv_v.values, extent=ext, origin="lower", cmap="Purples", aspect="auto")
                axes[1].scatter(sub[lon], sub[lat], c="black", s=6)
                fig.colorbar(im1, ax=axes[1], label="kriging variance")
                axes[1].set_title("Kriging variance (uncertainty)")
                for ax in axes:
                    ax.set_xlabel(lon)
                    ax.set_ylabel(lat)
                fig.tight_layout()
                fig.savefig(d / "kriging.png", dpi=150)
                plt.close(fig)
                files.append("kriging.png")
            except Exception:
                pass
            rmse = float(meta.get("loo_rmse", float("nan")))
            vrng = float(sub[value].max() - sub[value].min())
            # active stationarity check: OK assumes a constant mean; if a linear
            # lon/lat trend explains much variance, universal kriging is better.
            amat = np.column_stack(
                [np.ones(len(sub)), sub[lon].to_numpy(float), sub[lat].to_numpy(float)]
            )
            yv = sub[value].to_numpy(float)
            cf, *_ = np.linalg.lstsq(amat, yv, rcond=None)
            ss_tot = float(((yv - yv.mean()) ** 2).sum())
            trend_r2 = 1.0 - float(((yv - amat @ cf) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
            estimates["loo_rmse"] = round(rmse, 4)
            estimates["variogram_range"] = round(float(meta.get("range", float("nan"))), 4)
            estimates["trend_r2"] = round(trend_r2, 4)
            estimates["n_points"] = float(len(sub))
            rel = f"（≈值域 {vrng:.3g} 的 {100 * rmse / vrng:.1f}%）" if vrng > 0 else ""
            trend_warn = (
                f"；⚠ 检测到强空间趋势（经纬度 OLS R²={trend_r2:.2f}>0.5），普通克里金的平稳假定会有偏，"
                f"建议改用泛克里金（{value} ~ 经度+纬度）"
                if trend_r2 > 0.5
                else ""
            )
            summary.append(
                f"{entry.method} 完成（R/gstat）：{value} 在 {len(sub)} 点上普通克里金插值为 40×40 面；"
                f"变异函数模型={meta.get('model')}（range={meta.get('range', 0):.3g}）；"
                f"留一交叉验证 RMSE={rmse:.4g}{rel}；另出克里金方差(不确定性)图{trend_warn}。"
                "⚠ 默认首连续列为值；经纬度按欧氏近似。"
            )
            code += [
                "library(gstat); library(sp)  # 普通克里金",
                f"# variogram({value}~1) -> fit.variogram(Sph/Exp/Gau) -> krige + krige.cv(LOO)",
            ]
        except Exception as err:
            summary.append(f"克里金失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass



@register("local_moran")
def _branch_local_moran(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    value = next(
        (
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ),
        None,
    )
    if len(geo) < 2 or value is None:
        summary.append("局部 Moran (LISA) 失败：需要经纬度坐标 + 一个连续值变量。")
    else:
        sub = df[[geo[0], geo[1], value]].dropna()
        coords = sub[[geo[0], geo[1]]].to_numpy(dtype=float)
        x = sub[value].to_numpy(dtype=float)
        n = len(x)
        m2 = float(((x - x.mean()) ** 2).mean())
        if n < 10 or m2 == 0:
            summary.append("局部 Moran (LISA) 失败：样本不足（<10）或值变量为常数。")
        else:
            k = _knn_k(cfg, n - 2)  # config={"knn_k": N} 覆盖近邻数
            d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = np.argsort(d2, axis=1)[:, :k]
            z = x - x.mean()
            lag = z[nn].mean(axis=1)  # row-standardised lag = mean of k neighbours
            ii = (z / m2) * lag  # local Moran's I per location
            # conditional permutation: hold z_i, resample its k neighbours from the rest
            R = 499
            rng = np.random.default_rng(0)
            p = np.empty(n)
            for i in range(n):
                pool = np.delete(z, i)
                topk = np.argsort(rng.random((R, n - 1)), axis=1)[:, :k]
                ip = (z[i] / m2) * pool[topk].mean(axis=1)
                p[i] = (int(np.sum(np.abs(ip) >= abs(ii[i]))) + 1) / (R + 1)
            sig = p < 0.05
            quad = np.where(
                z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL")
            )
            cluster = np.where(sig, quad, "ns")

            import pandas as pd

            pd.DataFrame(
                {
                    geo[0]: coords[:, 0],
                    geo[1]: coords[:, 1],
                    value: x,
                    "local_I": np.round(ii, 4),
                    "p_value": np.round(p, 4),
                    "cluster": cluster,
                }
            ).to_csv(d / "lisa.csv", index=False, encoding="utf-8")
            files.append("lisa.csv")

            lon_i = 1 if ("lon" in geo[1].lower() or "lng" in geo[1].lower()) else 0
            lat_i = 1 - lon_i
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                colors = {
                    "HH": "#C44E52", "LL": "#4C72B0", "HL": "#DD8452",
                    "LH": "#55A868", "ns": "#cccccc",
                }
                fig, ax = plt.subplots(figsize=(6, 5))
                for cl, col in colors.items():
                    m = cluster == cl
                    if m.any():
                        ax.scatter(
                            coords[m, lon_i], coords[m, lat_i], c=col, s=26,
                            edgecolor="#444444", linewidth=0.3, label=f"{cl} ({int(m.sum())})",
                        )
                ax.set_xlabel(geo[lon_i])
                ax.set_ylabel(geo[lat_i])
                ax.set_title(f"LISA cluster map — {value}")
                ax.legend(fontsize=7, loc="best")
                fig.tight_layout()
                fig.savefig(d / "lisa_map.png", dpi=150)
                plt.close(fig)
                files.append("lisa_map.png")
            except Exception:
                pass

            for cl in ("HH", "LL", "HL", "LH"):
                estimates[f"n_{cl}"] = float(int(np.sum((cluster == cl))))
            summary.append(
                f"{entry.method} 完成：变量 {value}，显著局部簇 "
                f"HH={int(np.sum(cluster=='HH'))} LL={int(np.sum(cluster=='LL'))} "
                f"HL={int(np.sum(cluster=='HL'))} LH={int(np.sum(cluster=='LH'))}"
                f"（p<0.05，999→{R} 条件置换，k-NN={k}）；HH/LL=聚集，HL/LH=空间离群"
            )
            code += [
                "import numpy as np  # Local Moran's I (LISA, Anselin 1995)",
                "# I_i = (z_i/m2) * mean(z over kNN neighbours); conditional permutation p",
            ]



@register("moran_i")
def _branch_moran_i(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    value = next(
        (
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ),
        None,
    )
    if len(geo) < 2 or value is None:
        summary.append("Moran's I 失败：需要经纬度坐标 + 一个连续值变量。")
    else:
        sub = df[[geo[0], geo[1], value]].dropna()
        coords = sub[[geo[0], geo[1]]].to_numpy(dtype=float)
        x = sub[value].to_numpy(dtype=float)
        n = len(x)
        if n < 10:
            summary.append("Moran's I 失败：有效样本不足（<10）。")
        else:
            k = _knn_k(cfg, n - 1)  # config={"knn_k": N} 覆盖近邻数
            # pairwise squared euclidean distance on (lat, lon); fine for
            # ranking k nearest neighbours at moderate spatial extents.
            d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = np.argsort(d2, axis=1)[:, :k]
            W = np.zeros((n, n))
            W[np.repeat(np.arange(n), k), nn.ravel()] = 1.0 / k  # row-standardised
            z = x - x.mean()
            den = float((z**2).sum())
            Wsum = float(W.sum())

            def morans(zv):
                return (n / Wsum) * float(zv @ (W @ zv)) / den

            moran = morans(z)
            expected = -1.0 / (n - 1)
            rng = np.random.default_rng(0)
            perm = np.array([morans(rng.permutation(z)) for _ in range(999)])
            p = (int(np.sum(np.abs(perm - expected) >= abs(moran - expected))) + 1) / 1000.0

            lag = W @ z  # spatial lag of standardised value
            (d / "moran.txt").write_text(
                f"Moran's I = {moran:.4f}\nExpected (no autocorr) = {expected:.4f}\n"
                f"permutation p = {p:.4f} (999 perms)\nn = {n}, k-NN = {k}\n",
                encoding="utf-8",
            )
            files.append("moran.txt")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(z, lag, s=16, alpha=0.6, edgecolor="none")
                # Moran scatterplot slope is Moran's I
                xs = np.array([z.min(), z.max()])
                ax.plot(xs, moran * xs, color="#C44E52", lw=1.4, label=f"slope = I = {moran:.3f}")
                ax.axhline(0, color="grey", ls="--", lw=0.7)
                ax.axvline(0, color="grey", ls="--", lw=0.7)
                ax.set_xlabel(f"{value} (standardised)")
                ax.set_ylabel("spatial lag (W·z)")
                ax.set_title("Moran scatterplot")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(d / "moran_scatter.png", dpi=150)
                plt.close(fig)
                files.append("moran_scatter.png")
            except Exception:
                pass

            estimates["moran_i"] = round(moran, 4)
            estimates["p_value"] = round(p, 4)
            estimates["expected_i"] = round(expected, 4)
            verdict = "显著空间聚集" if (p < 0.05 and moran > expected) else "无显著空间自相关"
            summary.append(
                f"{entry.method} 完成：变量 {value}，Moran's I={moran:.4f}"
                f"（期望 {expected:.4f}），p={p:.4f}（999 置换，k-NN={k}）→ {verdict}"
            )
            code += [
                "import numpy as np  # Moran's I with k-NN row-standardised weights",
                f"# coords={geo}, value='{value}', k={k}",
                "# I = (n/W) * z'Wz / z'z ; permutation p over 999 shuffles of z",
            ]



@register("spatial_regression")
def _branch_spatial_regression(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import re

    from researchforge.executor import rbridge

    geo = [c.name for c in fp.columns if c.kind == "geo"][:2]
    _exc = {fp.unit_col, fp.time_col, *geo}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _exc]
    outcome = cont[0] if cont else None
    predictors = cont[1:6]
    lon = next((g for g in geo if "lon" in g.lower() or "lng" in g.lower()), geo[-1] if geo else None)
    lat = next((g for g in geo if g != lon), geo[0] if geo else None)
    names_safe = outcome is not None and all(
        re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *predictors, *geo]
    )
    have_r = (
        rbridge.r_available()
        and rbridge.r_package_available("spdep")
        and rbridge.r_package_available("spatialreg")
    )
    if len(geo) < 2 or outcome is None or not predictors:
        summary.append("空间回归失败：需要经纬度 + 连续结果变量 + ≥1 个连续预测变量。")
    elif not have_r:
        summary.append(
            "空间回归需要 R 的 spdep + spatialreg 包（未检测到）。"
            "安装：install.packages(c('spdep','spatialreg'))。"
        )
    elif not names_safe:
        summary.append("空间回归失败：列名需为标识符式（字母/数字/. _）。")
    else:
        sub = df[[*geo, outcome, *predictors]].dropna()
        csv = d / "_sar_input.csv"
        sub.to_csv(csv, index=False)
        try:
            diag, pref, coef = _spatial_reg_via_r(
                csv, outcome, predictors, lon, lat, k=_knn_k(cfg, len(sub) - 1, default=6)
            )
            coef.to_csv(d / "spatial_coefficients.csv", index=False, encoding="utf-8")
            files.append("spatial_coefficients.csv")
            preferred = "SAR（空间滞后）" if pref == "SAR" else "SEM（空间误差）"
            # report the PREFERRED model's effects (Opus catch): SEM betas ARE
            # marginal effects; SAR betas are NOT — report impacts() instead.
            is_sar = pref == "SAR"
            effect_col = "total" if is_sar else "estimate"
            effect_note = (
                "SAR 优选 → 报告 impacts（direct 直接 / indirect 溢出 / total 总效应）；"
                "注意：空间滞后模型系数本身不是边际效应，total 才是。"
                if is_sar
                else "SEM 优选 → 系数即边际效应（estimate ± 1.96·se）。"
            )
            pmoran = diag.get("resid_moran_p", float("nan"))
            (d / "diagnostics.txt").write_text(
                "空间回归诊断（k-NN 空间权重，k≈6）\n"
                f"OLS 残差 Moran's I p = {pmoran:.4g} "
                f"（{'有' if pmoran < 0.05 else '无'}显著空间依赖 → "
                f"{'需用空间模型' if pmoran < 0.05 else 'OLS 可能已够'}）\n"
                f"AIC：OLS={diag.get('ols_aic')}, SAR={diag.get('sar_aic')}, SEM={diag.get('sem_aic')}\n"
                f"SAR 空间滞后 ρ = {diag.get('sar_rho')}; SEM 空间误差 λ = {diag.get('sem_lambda')}\n"
                f"按 AIC 优选：{preferred}\n{effect_note}\n\n效应表：\n"
                + coef.to_string(index=False),
                encoding="utf-8",
            )
            files.append("diagnostics.txt")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                cp = coef[~coef["term"].str.contains("Intercept", case=False)]
                if len(cp):
                    fig, ax = plt.subplots(figsize=(5, 3))
                    if is_sar:  # impacts have no SE here -> bar of total effect
                        ax.barh(cp["term"], cp["total"], color="#4C72B0")
                        ax.set_xlabel("total impact (direct + indirect)")
                    else:
                        ax.errorbar(
                            cp["estimate"], range(len(cp)), xerr=1.96 * cp["std_err"], fmt="o"
                        )
                        ax.set_yticks(range(len(cp)))
                        ax.set_yticklabels(cp["term"])
                        ax.set_xlabel("SEM coefficient (95% CI)")
                    ax.axvline(0, color="grey", ls="--")
                    ax.set_title(f"{pref} effects — {outcome}")
                    fig.tight_layout()
                    fig.savefig(d / "coefficients.png", dpi=150)
                    plt.close(fig)
                    files.append("coefficients.png")
            except Exception:
                pass
            for kk in ("sar_rho", "resid_moran_p", "ols_aic", "sar_aic", "sem_aic"):
                if kk in diag:
                    estimates[kk] = round(diag[kk], 4)
            for _, r in coef.iterrows():
                if "Intercept" not in str(r["term"]):
                    estimates[str(r["term"])] = round(float(r[effect_col]), 4)
            summary.append(
                f"{entry.method} 完成（R/spdep）：OLS 残差 Moran p={pmoran:.2g}"
                f"（{'有' if pmoran < 0.05 else '无'}显著空间依赖）；SAR ρ={diag.get('sar_rho'):.3f}；"
                f"AIC OLS {diag.get('ols_aic'):.1f}/SAR {diag.get('sar_aic'):.1f}/"
                f"SEM {diag.get('sem_aic'):.1f} → 优选 {preferred}；{effect_note}"
            )
            code += [
                "library(spdep); library(spatialreg)  # k-NN 权重",
                f'# lagsarlm/errorsarlm({outcome} ~ {" + ".join(predictors)}); 残差 Moran 检验',
            ]
        except Exception as err:
            summary.append(f"空间回归失败：{err}")
        finally:
            try:
                csv.unlink()
            except OSError:
                pass

