"""Python-backed estimator delegators (semopy / econml / doubleml / rdrobust) —
heavy external-estimator wrappers. R-bridge delegators live in r_backends.py.
Both are re-exported by run.py.
"""

from __future__ import annotations

import os
from researchforge.executor._helpers.core import _sem_latents

def _sem_via_semopy(sub, spec: str) -> dict:
    """Fit a CFA/SEM model spec with pure-Python semopy (portable fallback).
    Extracts standardised loadings for ALL latent factors in the spec (not just a
    single factor named F), so custom multi-factor specs are handled."""
    import semopy

    latents = _sem_latents(spec) or ["F"]
    model = semopy.Model(spec)
    model.fit(sub)
    ins = model.inspect(std_est=True)
    load = ins[(ins["op"] == "~") & (ins["rval"].isin(latents))][
        ["lval", "rval", "Est. Std", "Std. Err", "p-value"]
    ].copy()
    load.columns = ["indicator", "factor", "std_loading", "std_err", "p_value"]
    stats = semopy.calc_stats(model)
    return {
        "loadings": load,
        "fit": {
            "cfi": float(stats["CFI"].iloc[0]),
            "tli": float(stats["TLI"].iloc[0]),
            "rmsea": float(stats["RMSEA"].iloc[0]),
            "chi2": float(stats["chi2"].iloc[0]),
            "dof": float(stats["DoF"].iloc[0]),
            "srmr": float("nan"),
        },
        "summary": str(ins),
        "backend": "semopy (Python)",
    }


def _dml_via_doubleml(df, outcome, treatment, controls, n_folds, discrete, plot_path, seed=0):
    """Double/debiased machine learning (Chernozhukov et al.) via the doubleml
    package: ML-learn the nuisance functions (outcome + treatment/propensity),
    cross-fit, and estimate an orthogonalized average treatment effect. Binary
    treatment -> IRM (ATE, no functional-form on the effect); continuous -> PLR
    (partially-linear). RandomForest learners. Returns a dict (incl. an overlap
    flag, the treatment 0/1 encoding, and the estimand label). Writes an ATE
    point+CI plot. Raises so the caller can degrade honestly."""
    import warnings

    import numpy as np
    import doubleml as dml
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    sub = df[[outcome, treatment, *controls]].dropna().copy()
    # IRM requires the treatment coded exactly {0,1}; normalise any 2-valued
    # encoding ({1,2}, {"Treated","Control"}, …) with a disclosed mapping (Opus catch).
    treat_map = None
    if discrete:
        vals = sorted(sub[treatment].unique().tolist(), key=lambda v: str(v))
        if set(vals) != {0, 1}:
            treat_map = {vals[0]: 0, vals[1]: 1}
            sub[treatment] = sub[treatment].map(treat_map).astype(int)
    # pin the cross-fitting sample split — it is drawn from the global NumPy RNG,
    # which the learners' own random_state does NOT control (Opus catch: ATE was
    # varying run-to-run without this).
    np.random.seed(int(seed))
    data = dml.DoubleMLData(sub, y_col=outcome, d_cols=treatment, x_cols=list(controls))
    reg = RandomForestRegressor(n_estimators=100, random_state=0)
    if discrete:
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        model = dml.DoubleMLIRM(data, ml_g=reg, ml_m=clf, n_folds=n_folds)
        kind = "IRM"
    else:
        model = dml.DoubleMLPLR(
            data, ml_l=reg, ml_m=RandomForestRegressor(n_estimators=100, random_state=1), n_folds=n_folds
        )
        kind = "PLR"
    overlap_warn = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit()
        overlap_warn = any("close to zero or one" in str(w.message) for w in caught)
    ci = model.confint(level=0.95)
    out = {
        "ate": float(model.coef[0]),
        "se": float(model.se[0]),
        "ci_lb": float(ci.iloc[0, 0]),
        "ci_ub": float(ci.iloc[0, 1]),
        "p_value": float(model.pval[0]),
        "model": kind,
        "n": int(sub.shape[0]),
        "overlap_warn": overlap_warn,
        "treat_map": treat_map,
        "estimand": "ATE" if discrete else "PLR_coefficient",
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.errorbar([out["ate"]], [0], xerr=[[out["ate"] - out["ci_lb"]], [out["ci_ub"] - out["ate"]]],
                    fmt="o", color="#4C72B0", capsize=5)
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_yticks([])
        ax.set_xlabel(f"ATE of {treatment} on {outcome} (95% CI)")
        ax.set_title(f"Double ML ({kind}) — ATE = {out['ate']:.3f}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return out


def _causal_forest_via_econml(df, outcome, treatment, modifiers, n_folds, discrete, seed,
                              hist_png, scatter_png, fdr_method="fdr_bh"):
    """Heterogeneous treatment effects (CATE) via econml CausalForestDML — a
    causal-forest DML estimator: residualize Y and T on the covariates (ML), then
    grow a causal forest over the effect-modifiers X to estimate effect(x). Reports
    the overall ATE, the CATE distribution, the share of individuals with a
    significant effect, and which modifiers drive heterogeneity. Binary treatment
    is normalized to {0,1} (disclosed). Returns a dict; writes a CATE histogram and
    a CATE-vs-top-modifier scatter. Raises so the caller can degrade honestly."""
    import numpy as np
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    sub = df[[outcome, treatment, *modifiers]].dropna().copy()
    treat_map = None
    if discrete:
        vals = sorted(sub[treatment].unique().tolist(), key=lambda v: str(v))
        if set(vals) != {0, 1}:
            treat_map = {vals[0]: 0, vals[1]: 1}
            sub[treatment] = sub[treatment].map(treat_map).astype(int)
    np.random.seed(int(seed))
    Y = sub[outcome].astype(float).to_numpy()
    T = sub[treatment].to_numpy()
    X = sub[modifiers].astype(float).to_numpy()
    model_t = (
        RandomForestClassifier(n_estimators=100, random_state=int(seed))
        if discrete
        else RandomForestRegressor(n_estimators=100, random_state=int(seed))
    )
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=int(seed)),
        model_t=model_t, discrete_treatment=discrete, n_estimators=300,
        random_state=int(seed), cv=n_folds,
    )
    est.fit(Y, T, X=X)
    cate = np.asarray(est.effect(X), dtype=float)
    ate = float(est.ate(X))
    a_lb, a_ub = (float(v) for v in est.ate_interval(X, alpha=0.05))
    lb, ub = est.effect_interval(X, alpha=0.05)  # per-row CI
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    frac_sig = float(np.mean((lb > 0) | (ub < 0)))  # uncorrected share with a significant individual effect
    # multiple-comparison correction: with n per-individual tests, the uncorrected share inflates
    # false positives (~5% baseline under the null). Derive per-row SE from the (normal, symmetric)
    # 95% CI — verified to match econml's own stderr to machine precision — get two-sided p-values,
    # and control the false-discovery rate. Default fdr_bh (FDR<=5% under independence/positive
    # dependence); fdr_by is valid under ARBITRARY dependence (more conservative).
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    fdr_method = fdr_method if fdr_method in {"fdr_bh", "fdr_by"} else "fdr_bh"
    se_pt = (ub - lb) / (2.0 * 1.959963984540054)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_pt > 0, cate / se_pt, 0.0)
    pvals = 2.0 * norm.sf(np.abs(z))
    valid = np.isfinite(pvals)
    if valid.sum() > 0:
        rej = np.zeros(pvals.shape, dtype=bool)
        rej[valid] = multipletests(pvals[valid], alpha=0.05, method=fdr_method)[0]
        frac_sig_bh = float(np.mean(rej))
    else:
        frac_sig_bh = float("nan")
    imp = np.asarray(est.feature_importances_, dtype=float)
    order = np.argsort(imp)[::-1]
    drivers = [(modifiers[i], round(float(imp[i]), 3)) for i in order]
    out = {
        "ate": ate, "ate_lb": a_lb, "ate_ub": a_ub,
        "cate_mean": float(cate.mean()), "cate_sd": float(cate.std()),
        "cate_p10": float(np.percentile(cate, 10)), "cate_p90": float(np.percentile(cate, 90)),
        "frac_significant": frac_sig, "frac_significant_bh": frac_sig_bh,
        "fdr_method": fdr_method, "drivers": drivers, "n": int(sub.shape[0]),
        "treat_map": treat_map,
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(cate, bins=30, color="#4C72B0", edgecolor="white")
        ax.axvline(ate, color="#C44E52", ls="--", label=f"ATE={ate:.3f}")
        ax.axvline(0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel(f"individual treatment effect (CATE) of {treatment} on {outcome}")
        ax.set_ylabel("count")
        ax.set_title("CATE distribution (heterogeneous effects)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(hist_png, dpi=150)
        plt.close(fig)
        top = order[0]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(X[:, top], cate, s=10, alpha=0.4, edgecolor="none", color="#55A868")
        ax2.axhline(0, color="grey", ls=":", lw=0.8)
        ax2.set_xlabel(f"{modifiers[top]} (top effect-modifier)")
        ax2.set_ylabel("CATE")
        ax2.set_title(f"CATE vs {modifiers[top]}")
        fig2.tight_layout()
        fig2.savefig(scatter_png, dpi=150)
        plt.close(fig2)
    except Exception:
        pass
    return out


def _rdd_via_rdrobust(df, outcome, running, cutoff, plot_path, fuzzy=None):
    """Regression discontinuity (Calonico–Cattaneo–Titiunik) via rdrobust:
    local-linear estimate of the outcome jump at the cutoff, MSE-optimal bandwidth,
    robust bias-corrected inference. Returns a dict; writes a binned RD plot.
    If ``fuzzy`` (a column of actually-received treatment) is given, estimates the
    *fuzzy* RDD LATE = (outcome jump) / (treatment-probability jump) at the cutoff.
    Raises so the caller can degrade honestly."""
    import numpy as np
    from rdrobust import rdrobust

    cols = [outcome, running] + ([fuzzy] if fuzzy else [])
    sub = df[cols].copy()
    sub[outcome] = sub[outcome].astype(float)
    sub[running] = sub[running].astype(float)
    if fuzzy:
        sub[fuzzy] = sub[fuzzy].astype(float)
    sub = sub.dropna()
    y, x = sub[outcome], sub[running]
    if fuzzy:
        r = rdrobust(y=y, x=x, c=float(cutoff), fuzzy=sub[fuzzy])
    else:
        r = rdrobust(y=y, x=x, c=float(cutoff))
    out = {
        "fuzzy": bool(fuzzy),
        "rd_robust": float(r.coef.loc["Robust", "Coeff"]),
        "rd_conventional": float(r.coef.loc["Conventional", "Coeff"]),
        "ci_lb": float(r.ci.loc["Robust", "CI Lower"]),
        "ci_ub": float(r.ci.loc["Robust", "CI Upper"]),
        "p_robust": float(r.pv.loc["Robust", "P>|z|"]),
        "bw_h": float(r.bws.loc["h", "left"]),
        "n_left": int(r.N_h[0]),
        "n_right": int(r.N_h[1]),
        "bwselect": str(r.bwselect),
    }
    if fuzzy:
        # First stage = jump in treatment probability at the cutoff (weak -> LATE unreliable).
        # This is a STRENGTH DIAGNOSTIC only, run at its own MSE-optimal bandwidth (for T), so it is
        # NOT exactly the LATE denominator (rdrobust uses the y-optimal bandwidth internally); don't
        # reconstruct LATE = reduced_form / first_stage_jump by hand. Conventional point (stable
        # magnitude) + robust p (inference), mirroring how the LATE reports a robust p.
        try:
            rfs = rdrobust(y=sub[fuzzy], x=x, c=float(cutoff))
            out["first_stage_jump"] = float(rfs.coef.loc["Conventional", "Coeff"])
            out["first_stage_p"] = float(rfs.pv.loc["Robust", "P>|z|"])
        except Exception:
            out["first_stage_jump"] = float("nan")
            out["first_stage_p"] = float("nan")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = out["bw_h"]
        xv, yv = x.to_numpy(), y.to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        # binned means within ~3 bandwidths each side of the cutoff
        for lo, hi, col in [(cutoff - 3 * h, cutoff, "#4C72B0"), (cutoff, cutoff + 3 * h, "#C44E52")]:
            sel = (xv >= lo) & (xv < hi) if lo < cutoff else (xv >= lo) & (xv <= hi)
            if sel.sum() > 5:
                edges = np.linspace(lo, hi, 11)
                idx = np.digitize(xv[sel], edges)
                bx = [xv[sel][idx == k].mean() for k in range(1, len(edges)) if (idx == k).any()]
                by = [yv[sel][idx == k].mean() for k in range(1, len(edges)) if (idx == k).any()]
                ax.scatter(bx, by, s=26, c=col)
        ax.axvline(cutoff, color="grey", ls="--", lw=1)
        ax.set_xlabel(f"{running} (cutoff = {cutoff:g})")
        ax.set_ylabel(f"{outcome} (binned mean)")
        ax.set_title(f"Regression discontinuity — jump = {out['rd_robust']:.3f}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return out
