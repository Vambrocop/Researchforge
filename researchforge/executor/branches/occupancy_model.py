"""Branch handler for the ecology family — MacKenzie single-season occupancy.

occupancy_model — MacKenzie et al. (2002) single-season site-occupancy model,
constant model ψ(.)p(.): detection-corrected occupancy.

Input is a detection / non-detection matrix: rows = sites, columns = repeat visits
(≥2), entries 0/1 (1 = species detected that visit, 0 = not detected, NaN = visit
not made). The species truly occupies a site with probability ψ; if occupied it is
detected on each visit independently with probability p. The per-site likelihood for
a detection history h_i over its K_i actual visits with d_i detections is

    d_i ≥ 1:  ψ · p^{d_i} · (1 − p)^{K_i − d_i}
    d_i = 0:  ψ · (1 − p)^{K_i} + (1 − ψ)

The d_i = 0 term is the crux: it MIXES two ways to never detect the species —
"occupied but missed on every visit" [ψ·(1−p)^{K_i}] and "genuinely unoccupied"
[(1−ψ)]. Maximise the total log-likelihood over (ψ, p) ∈ (0,1)². We optimise on the
logit scale (logit ψ, logit p) so the search is unconstrained, via
scipy.optimize.minimize; SEs come from the inverse numerical Hessian on the logit
scale, back-transformed to the (0,1) scale by the delta method (95% CIs).

The headline: ψ̂ (detection-corrected) > naive occupancy (fraction of sites with ≥1
detection) because detection is imperfect (p<1) — the naive figure UNDER-counts.

Pure Python (numpy / scipy.optimize); no R. Honest skip when <2 visit columns or
<15 sites; the optimisation is wrapped so a failure degrades to an honest message
rather than crashing. MUTATES ctx (never rebinds). See executor/_branch_api.py and
CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


@register("occupancy_model")
def _branch_occupancy_model(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    # ── Resolve the 0/1 visit columns ──────────────────────────────────────────
    # config["visits"] (list of column names) overrides; otherwise auto-detect
    # every column whose non-missing values are a subset of {0, 1}.
    requested = cfg.get("visits")
    candidates = list(requested) if requested else [c.name for c in fp.columns]

    visit_cols: list[str] = []
    for name in candidates:
        if name not in df.columns:
            continue
        col = pd.to_numeric(df[name], errors="coerce")
        vals = col.dropna().unique()
        if vals.size == 0:
            continue
        if set(np.unique(vals)).issubset({0.0, 1.0}):
            visit_cols.append(name)

    if len(visit_cols) < 2:
        summary.append(
            "占据模型跳过：需要 ≥2 列 0/1 检测/未检测矩阵（每列一次重复访查，"
            "1=检测到、0=未检测、缺失=未访查）。"
        )
        return

    # Build the K×J detection matrix (sites × visits); coerce to numeric, NaN=not visited.
    mat = df[visit_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # Drop sites with NO visits at all (all NaN) — they carry no information.
    has_visit = ~np.all(np.isnan(mat), axis=1)
    mat = mat[has_visit]

    n_sites = int(mat.shape[0])
    if n_sites < 15:
        summary.append(
            f"占据模型跳过：需要 ≥15 个样点（当前有效样点 {n_sites}）以区分占据率 ψ 与检测率 p。"
        )
        return

    # Per-site visit counts K_i and detection counts d_i (ignoring NaN visits).
    K = np.nansum(~np.isnan(mat), axis=1).astype(float)       # visits made per site
    detections = np.where(np.isnan(mat), 0.0, mat)            # NaN -> 0 (not a detection)
    D = np.nansum(detections, axis=1).astype(float)           # detections per site
    detected_any = D >= 1                                     # site had ≥1 detection

    naive_occ = float(np.mean(detected_any))                 # uncorrected occupancy
    mean_visits = float(np.mean(K))

    # ── Negative log-likelihood on the logit scale (unconstrained) ─────────────
    from scipy.special import expit  # numerically-stable logistic 1/(1+e^-x)

    def neg_loglik(theta: np.ndarray) -> float:
        psi = expit(theta[0])
        p = expit(theta[1])
        # clamp away from exact 0/1 for log safety
        psi = min(max(psi, 1e-12), 1.0 - 1e-12)
        p = min(max(p, 1e-12), 1.0 - 1e-12)

        # Per-site detection probability of the observed history GIVEN occupancy:
        #   p^{D} (1-p)^{K-D}
        log_p_obs_given_occ = D * np.log(p) + (K - D) * np.log(1.0 - p)
        p_obs_given_occ = np.exp(log_p_obs_given_occ)

        # Marginal site likelihood:
        #   detected (D>=1): psi * p_obs_given_occ      (must be occupied)
        #   never (D==0):    psi * (1-p)^K + (1-psi)    (occupied-but-missed OR unoccupied)
        like = np.where(
            detected_any,
            psi * p_obs_given_occ,
            psi * p_obs_given_occ + (1.0 - psi),
        )
        like = np.clip(like, 1e-300, None)
        return float(-np.sum(np.log(like)))

    # ── Optimise (logit scale) + Hessian-based SEs + delta-method back-transform ─
    try:
        from scipy.optimize import minimize

        # Sensible starting values from the data (logit of clamped naive figures).
        def _logit(v: float) -> float:
            v = min(max(v, 0.05), 0.95)
            return float(np.log(v / (1.0 - v)))

        # naive p ≈ detections / visits among sites known occupied (fallback global).
        det_among_occ = float(D[detected_any].sum())
        vis_among_occ = float(K[detected_any].sum())
        p0 = det_among_occ / vis_among_occ if vis_among_occ > 0 else 0.4
        psi0 = max(naive_occ, 0.05)
        x0 = np.array([_logit(psi0), _logit(p0)])

        res = minimize(neg_loglik, x0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
        # Polish with BFGS for a smooth Hessian-friendly optimum.
        res = minimize(neg_loglik, res.x, method="BFGS",
                       options={"gtol": 1e-8, "maxiter": 5000})
        # BFGS res.success is unreliable on the flat occupancy likelihood (it can report
        # failure at a perfectly good optimum); treat a small gradient norm as converged.
        gnorm = float(np.linalg.norm(res.jac)) if getattr(res, "jac", None) is not None else float("inf")
        converged = bool(res.success) or gnorm < 1e-3

        logit_psi, logit_p = float(res.x[0]), float(res.x[1])
        psi_hat = float(expit(logit_psi))
        p_hat = float(expit(logit_p))
        loglik = float(-res.fun)

        # Numerical Hessian of the NLL on the logit scale -> inverse = covariance.
        H = _numeric_hessian(neg_loglik, res.x)
        psi_se = p_se = float("nan")
        psi_lo = psi_hi = p_lo = p_hi = float("nan")
        try:
            cov = np.linalg.inv(H)
            var_logit_psi = float(cov[0, 0])
            var_logit_p = float(cov[1, 1])
            if var_logit_psi >= 0 and var_logit_p >= 0:
                # Delta method: d expit(x)/dx = expit(x)(1-expit(x)).
                jac_psi = psi_hat * (1.0 - psi_hat)
                jac_p = p_hat * (1.0 - p_hat)
                psi_se = float(jac_psi * np.sqrt(var_logit_psi))
                p_se = float(jac_p * np.sqrt(var_logit_p))
                # 95% CI built on the logit scale then back-transformed (stays in 0..1).
                psi_lo = float(expit(logit_psi - 1.96 * np.sqrt(var_logit_psi)))
                psi_hi = float(expit(logit_psi + 1.96 * np.sqrt(var_logit_psi)))
                p_lo = float(expit(logit_p - 1.96 * np.sqrt(var_logit_p)))
                p_hi = float(expit(logit_p + 1.96 * np.sqrt(var_logit_p)))
        except np.linalg.LinAlgError:
            pass
    except Exception as err:  # noqa: BLE001 — honest degrade, never crash
        summary.append(f"占据模型失败：优化未能收敛（{err}）。")
        return

    # ── estimates (plain floats; SE / CI in separate keys) ─────────────────────
    def _r(v: float) -> float:
        return round(float(v), 4) if v == v else float("nan")  # NaN-safe round

    estimates["occupancy_psi"] = _r(psi_hat)
    estimates["occupancy_psi_se"] = _r(psi_se)
    estimates["occupancy_psi_ci_low"] = _r(psi_lo)
    estimates["occupancy_psi_ci_high"] = _r(psi_hi)
    estimates["detection_p"] = _r(p_hat)
    estimates["detection_p_se"] = _r(p_se)
    estimates["detection_p_ci_low"] = _r(p_lo)
    estimates["detection_p_ci_high"] = _r(p_hi)
    estimates["naive_occupancy"] = _r(naive_occ)
    estimates["n_sites"] = float(n_sites)
    estimates["mean_visits"] = _r(mean_visits)
    estimates["loglik"] = _r(loglik)
    estimates["converged"] = 1.0 if converged else 0.0

    # ── CSV artifact ────────────────────────────────────────────────────────────
    try:
        pd.DataFrame(
            [
                {
                    "parameter": "occupancy_psi (detection-corrected)",
                    "estimate": _r(psi_hat),
                    "se": _r(psi_se),
                    "ci_low": _r(psi_lo),
                    "ci_high": _r(psi_hi),
                },
                {
                    "parameter": "detection_p",
                    "estimate": _r(p_hat),
                    "se": _r(p_se),
                    "ci_low": _r(p_lo),
                    "ci_high": _r(p_hi),
                },
                {
                    "parameter": "naive_occupancy (uncorrected)",
                    "estimate": _r(naive_occ),
                    "se": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
            ]
        ).to_csv(d / "occupancy_estimates.csv", index=False, encoding="utf-8")
        files.append("occupancy_estimates.csv")
    except Exception:
        pass

    # ── PNG: naive vs detection-corrected occupancy (with ψ CI) ────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["Naive\n(uncorrected)", "Detection-corrected\n(psi-hat)"]
        vals = [naive_occ, psi_hat]
        bars = ax.bar(labels, vals, color=["#bdbdbd", "#2c7fb8"], width=0.55)
        # Error bar on the corrected estimate only (naive has no model CI).
        if psi_lo == psi_lo and psi_hi == psi_hi:  # not NaN
            ax.errorbar(
                1, psi_hat,
                yerr=[[max(psi_hat - psi_lo, 0)], [max(psi_hi - psi_hat, 0)]],
                fmt="none", ecolor="black", capsize=5, lw=1.5,
            )
        ax.set_ylim(0, min(1.0, max(vals) + 0.25))
        ax.set_ylabel("Occupancy probability")
        ax.set_title(f"Occupancy: naive vs detection-corrected (p-hat={p_hat:.2f})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(d / "occupancy_comparison.png", dpi=150)
        plt.close(fig)
        files.append("occupancy_comparison.png")
    except Exception:
        pass

    # ── Chinese summary with ⚠ disclosures ─────────────────────────────────────
    conv_note = "" if converged else "（⚠ 优化未报告收敛，估计需谨慎解读）"
    summary.append(
        f"{entry.method} 完成：{n_sites} 个样点 × 平均 {mean_visits:.1f} 次重复访查，"
        f"检测校正后占据率 ψ̂={psi_hat:.3f}"
        + (f"（95% CI [{psi_lo:.3f}, {psi_hi:.3f}]）" if psi_lo == psi_lo else "")
        + f"，朴素（未校正）占据率={naive_occ:.3f}，"
        f"单次检测率 p̂={p_hat:.3f}"
        + (f"（95% CI [{p_lo:.3f}, {p_hi:.3f}]）" if p_lo == p_lo else "")
        + f"。{conv_note}"
    )
    summary.append(
        "⚠ 假定与偏差披露：① 检测不完美（p<1）使「朴素占据率」系统性低估真实占据——"
        "本模型正是对其的校正，故 ψ̂ 通常高于朴素值；② 假定季节内「闭合」"
        "（调查季内占据状态不随访查改变）；③ 无假阳性（不会把未占据样点误报为检测到）；"
        "④ 各次访查的检测相互独立；⑤ 常数模型 ψ(.)p(.) 假定所有样点同质（ψ、p 为常数）——"
        "含协变量的 ψ(x)/p(x) 模型是其扩展；⑥ 需足够多的重复访查才能把 ψ 与 p 分离开。"
    )

    code += [
        "# MacKenzie single-season occupancy psi(.)p(.) — logit-scale MLE",
        "import numpy as np; from scipy.optimize import minimize; from scipy.special import expit",
        f"visit_cols = {visit_cols!r}",
        "mat = df[visit_cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)",
        "K = np.nansum(~np.isnan(mat), axis=1); D = np.nansum(np.where(np.isnan(mat),0,mat), axis=1)",
        "det = D >= 1",
        "def nll(t):",
        "    psi, p = expit(t[0]), expit(t[1])",
        "    po = np.exp(D*np.log(p) + (K-D)*np.log(1-p))",
        "    like = np.where(det, psi*po, psi*po + (1-psi))",
        "    return -np.sum(np.log(np.clip(like, 1e-300, None)))",
        "res = minimize(nll, [0.0, 0.0], method='BFGS')  # logit-scale; back-transform via expit",
    ]


def _numeric_hessian(f, x, eps: float = 1e-4):
    """Central-difference numerical Hessian of scalar f at x (small 2-D problem)."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # standard second derivative (more accurate than the cross formula on the diagonal)
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                H[i, i] = (f(xp) - 2.0 * fx + f(xm)) / (eps * eps)
            else:
                xi = x.copy(); xi[i] += eps; xi[j] += eps
                fpp = f(xi)
                xi = x.copy(); xi[i] += eps; xi[j] -= eps
                fpm = f(xi)
                xi = x.copy(); xi[i] -= eps; xi[j] += eps
                fmp = f(xi)
                xi = x.copy(); xi[i] -= eps; xi[j] -= eps
                fmm = f(xi)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    return H
