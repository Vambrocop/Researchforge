"""Executor: run the chosen analysis and persist code / tables / figures / report
to outputs/<timestamp>_<analysis>/. Reuses the empirical-analysis-python stack
(statsmodels + matplotlib)."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from pydantic import BaseModel, Field

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.profiler.profile import read_table

_REGRESSION = {"ols_regression", "panel_fixed_effects", "did"}


class RunResult(BaseModel):
    analysis_id: str
    method: str
    output_dir: str
    files: list[str] = Field(default_factory=list)
    report_path: str
    summary: str = ""
    estimates: dict[str, float] = Field(default_factory=dict)


def _run_dir(root: str, entry_id: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    d = Path(root) / f"{ts}_{entry_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_did_treatment(df, fp: DataFingerprint) -> list[str]:
    """The DID treatment is the binary that varies WITHIN units over time (a
    treatment that switches on), not a fixed group flag. Returns [] if none vary."""
    if not (fp.unit_col and fp.time_col):
        return fp.treatment_candidates[:1]
    best = None
    for name in fp.treatment_candidates:
        frac = float((df.groupby(fp.unit_col)[name].nunique() > 1).mean())
        if frac > 0 and (best is None or frac > best[0]):
            best = (frac, name)
    return [best[1]] if best else []


def _regression(df, fp: DataFingerprint, entry: AnalysisEntry):
    import statsmodels.formula.api as smf

    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    if not cont:
        raise ValueError("没有连续型因变量，无法回归")
    y = cont[0]
    exclude = {y, fp.unit_col, fp.time_col}

    fe_terms: list[str] = []
    if entry.id in {"panel_fixed_effects", "did"} and fp.unit_col and fp.time_col:
        fe_terms = [f"C(Q('{fp.unit_col}'))", f"C(Q('{fp.time_col}'))"]

    if entry.id == "did" and fp.treatment_candidates:
        rhs_vars = _pick_did_treatment(df, fp) or fp.treatment_candidates[:1]
    else:
        rhs_vars = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ][:5]

    rhs = [f"Q('{v}')" for v in rhs_vars] + fe_terms
    formula = f"Q('{y}') ~ " + (" + ".join(rhs) if rhs else "1")
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
    return y, rhs_vars, formula, model


def _heatmap(corr, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _coef_plot(model, rhs_vars, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # match formula-style names Q('v') first, else raw names (array-API
        # models like OrderedModel index params by the bare column name).
        names: list[str] = []
        labels: list[str] = []
        for v in rhs_vars:
            qn = f"Q('{v}')"
            if qn in model.params.index:
                names.append(qn)
                labels.append(v)
            elif v in model.params.index:
                names.append(v)
                labels.append(v)
        if not names:
            return
        coefs = model.params[names]
        errs = model.bse[names]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.errorbar(coefs.values, range(len(names)), xerr=1.96 * errs.values, fmt="o")
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("coefficient (95% CI)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


# Per-theme color cycles (theme-specific rc overrides are built in _init_mpl_style).
_THEME_COLORS = {
    "default": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"],
    "nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#7E6148"],
    "aer": ["#000000", "#666666", "#999999", "#333333", "#BBBBBB", "#555555"],
    "dark": ["#4C9BE0", "#FF8C42", "#5CD08A", "#E45757", "#B083E0", "#E0C04C"],
}


def _init_mpl_style(theme: str | None = None) -> None:
    """Apply one clean, publication-friendly look to every figure this run
    produces. Theme is chosen by arg or the RF_THEME env var (default | nature |
    aer | dark). Called once per analysis; best-effort so a missing/old
    matplotlib never breaks an analysis."""
    theme = (theme or os.environ.get("RF_THEME", "default")).strip().lower()
    if theme not in _THEME_COLORS:
        theme = "default"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rc = {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "axes.prop_cycle": plt.cycler(color=_THEME_COLORS[theme]),
        }
        if theme == "nature":  # NPG palette, sans-serif, tighter
            rc.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": 9,
                "axes.linewidth": 0.6,
                "lines.linewidth": 1.3,
            })
        elif theme == "aer":  # economics: serif, grayscale-safe, no grid
            rc.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif"],
                "axes.grid": False,
                "axes.titleweight": "normal",
            })
        elif theme == "dark":  # dashboard dark background
            rc.update({
                "figure.facecolor": "#1e1e1e",
                "savefig.facecolor": "#1e1e1e",
                "axes.facecolor": "#1e1e1e",
                "axes.edgecolor": "#cccccc",
                "axes.labelcolor": "#eeeeee",
                "axes.titlecolor": "#eeeeee",
                "text.color": "#eeeeee",
                "xtick.color": "#cccccc",
                "ytick.color": "#cccccc",
                "grid.color": "#444444",
            })
        plt.rcParams.update(rc)
    except Exception:
        pass


def _quantile_process_plot(qr, predictors, path: Path) -> None:
    """Koenker quantile-process plot: each predictor's coefficient (±95% CI)
    traced across the quantile grid τ=0.1…0.9, so the reader sees how the
    effect shifts down the outcome distribution — the signature quantile-reg
    figure, far more informative than a single median coefficient."""
    try:
        import numpy as np

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fits = {t: qr.fit(q=t) for t in taus}
        names = [v for v in predictors if f"Q('{v}')" in fits[0.5].params.index]
        if not names:
            return
        fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), squeeze=False)
        for ax, v in zip(axes[0], names):
            kn = f"Q('{v}')"
            coef = np.array([fits[t].params[kn] for t in taus])
            se = np.array([fits[t].bse[kn] for t in taus])
            ax.plot(taus, coef, "-o", color="#4C72B0", lw=1.6, ms=4)
            ax.fill_between(taus, coef - 1.96 * se, coef + 1.96 * se, color="#4C72B0", alpha=0.18)
            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.set_title(v)
            ax.set_xlabel("quantile τ")
        axes[0][0].set_ylabel("coefficient (95% CI)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _ordinal_prob_plot(model, df, predictors, levels, path: Path) -> None:
    """Predicted probability of each ordinal level as the first predictor varies
    (others held at their mean) — shows how the whole response distribution
    shifts, the most readable ordered-logit figure."""
    try:
        import numpy as np
        import pandas as pd

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        key = predictors[0]
        grid = np.linspace(float(df[key].min()), float(df[key].max()), 60)
        base = {p: float(df[p].mean()) for p in predictors}
        X = pd.DataFrame([{**base, key: g} for g in grid])[predictors]
        probs = np.asarray(model.predict(X))
        fig, ax = plt.subplots(figsize=(6, 4))
        for j, lvl in enumerate(levels):
            ax.plot(grid, probs[:, j], lw=1.6, label=f"level {lvl}")
        ax.set_xlabel(key)
        ax.set_ylabel("predicted probability")
        ax.set_title(f"predicted level probabilities vs {key}")
        ax.legend(fontsize=8, ncol=min(len(levels), 4), title="ordinal level")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _resid_plot(model, path: Path) -> None:
    """Residuals vs fitted — the basic OLS diagnostic; a funnel flags
    heteroskedasticity, a curve flags missing non-linearity."""
    try:
        import numpy as np

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fitted = np.asarray(model.fittedvalues)
        resid = np.asarray(model.resid)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(fitted, resid, s=14, alpha=0.6, edgecolor="none")
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("fitted values")
        ax.set_ylabel("residuals")
        ax.set_title("residuals vs fitted")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _silhouette_plot(X, labels, path: Path) -> None:
    """Silhouette plot: per-sample silhouette grouped by cluster (cohesion vs
    separation); dashed line is the mean silhouette score."""
    try:
        import numpy as np
        from sklearn.metrics import silhouette_samples, silhouette_score

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = np.asarray(labels)
        uniq = sorted(set(labels.tolist()))
        if len(uniq) < 2:
            return
        sil = silhouette_samples(X, labels)
        avg = float(silhouette_score(X, labels))
        fig, ax = plt.subplots(figsize=(6, 4))
        y_lower = 0
        for k in uniq:
            vals = np.sort(sil[labels == k])
            y_upper = y_lower + len(vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.75)
            ax.text(-0.05, y_lower + len(vals) / 2, str(k), va="center", fontsize=8)
            y_lower = y_upper + 10
        ax.axvline(avg, color="red", ls="--", lw=1, label=f"mean={avg:.2f}")
        ax.set_xlabel("silhouette coefficient")
        ax.set_ylabel("samples grouped by cluster")
        ax.set_title("silhouette plot")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _nca_ceiling(x, y):
    """CE-FDH ceiling for Necessary Condition Analysis (Dul 2016). Returns
    (effect_size_d, sorted_x, cummax_y). The ceiling c(x)=max{yᵢ : xᵢ≤x} is the
    free-disposal-hull upper boundary; d = empty-zone-area / total-scope-area."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    cmax = np.maximum.accumulate(ys)
    scope = (xmax - xmin) * (ymax - ymin)
    if scope <= 0:
        return 0.0, xs, cmax
    empty = float(np.sum((ymax - cmax[:-1]) * np.diff(xs)))  # area above the ceiling
    return empty / scope, xs, cmax


def _nca_plot(sub, outcome, predictors, ceilings, path: Path) -> None:
    """NCA scatter(s) with the CE-FDH ceiling line — the empty upper-left zone
    above the ceiling is the visual evidence of necessity."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = predictors[:4]
        y = sub[outcome].to_numpy(dtype=float)
        fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2), squeeze=False)
        for ax, p in zip(axes[0], names):
            xs, cmax, d = ceilings[p]
            ax.scatter(sub[p], y, s=14, alpha=0.5, edgecolor="none")
            ax.step(xs, cmax, where="post", color="#C44E52", lw=1.5, label=f"ceiling, d={d:.2f}")
            ax.set_xlabel(p)
            ax.set_title(f"{p} (d={d:.2f})")
            ax.legend(fontsize=7)
        axes[0][0].set_ylabel(outcome)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _plotly_corr_heatmap(corr, path: Path) -> None:
    """Interactive correlation heatmap (hover for exact r, zoomable). Best-effort
    so a missing plotly never breaks the run; the static PNG is always produced."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="r"),
                hovertemplate="%{y} – %{x}<br>r = %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Correlation (interactive)",
            width=640,
            height=560,
            template="plotly_white",
        )
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    except Exception:
        pass


def _plotly_scatter(coords, labels, path: Path, title: str, xlab: str, ylab: str) -> None:
    """Interactive 2D scatter colored by group (zoom / pan / hover point index)."""
    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px

        coords = np.asarray(coords)
        y = coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords))
        data = pd.DataFrame(
            {xlab: coords[:, 0], ylab: y, "group": [str(v) for v in labels], "point": range(len(coords))}
        )
        fig = px.scatter(
            data, x=xlab, y=ylab, color="group", hover_data=["point"], title=title, template="plotly_white"
        )
        fig.update_layout(width=660, height=520)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    except Exception:
        pass


def _sem_via_lavaan(csv_path, spec: str) -> dict:
    """Fit a CFA via R's lavaan (gold standard) through the R bridge. Returns a
    backend-agnostic result dict, or raises so the caller falls back to semopy."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rcode = (
        "suppressMessages(library(lavaan))\n"
        f'd <- read.csv("{csv_r}")\n'
        f'fit <- cfa("{spec}", data=d, std.lv=TRUE)\n'
        'std <- standardizedSolution(fit); ld <- std[std$op == "=~", ]\n'
        'cat("##LOADINGS\\n")\n'
        'for (i in seq_len(nrow(ld))) cat(sprintf("%s,%.6f,%.6f,%.6f\\n", '
        "ld$rhs[i], ld$est.std[i], ifelse(is.na(ld$se[i]),0,ld$se[i]), "
        "ifelse(is.na(ld$pvalue[i]),0,ld$pvalue[i])))\n"
        'fm <- fitMeasures(fit, c("chisq","df","cfi","tli","rmsea","srmr","aic","bic"))\n'
        'cat("##FIT\\n"); for (nm in names(fm)) cat(sprintf("%s,%.6f\\n", nm, fm[nm]))\n'
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, lrows, frows = None, [], []
    for line in out.splitlines():
        s = line.strip()
        if s == "##LOADINGS":
            section = "L"
        elif s == "##FIT":
            section = "F"
        elif "," in s:
            (lrows if section == "L" else frows).append(s)
    if not lrows or not frows:
        raise RuntimeError("lavaan produced no parseable output")
    load = pd.DataFrame(
        [r.split(",") for r in lrows], columns=["indicator", "std_loading", "std_err", "p_value"]
    )
    for c in ("std_loading", "std_err", "p_value"):
        load[c] = load[c].astype(float)
    fm = {k: float(v) for k, v in (r.split(",", 1) for r in frows)}
    return {
        "loadings": load,
        "fit": {
            "cfi": fm.get("cfi", float("nan")),
            "tli": fm.get("tli", float("nan")),
            "rmsea": fm.get("rmsea", float("nan")),
            "chi2": fm.get("chisq", float("nan")),
            "dof": fm.get("df", float("nan")),
            "srmr": fm.get("srmr", float("nan")),
        },
        "summary": out,
        "backend": "lavaan (R)",
    }


def _sem_via_semopy(sub, spec: str) -> dict:
    """Fit the same single-factor CFA with pure-Python semopy (portable fallback)."""
    import semopy

    model = semopy.Model(spec)
    model.fit(sub)
    ins = model.inspect(std_est=True)
    load = ins[(ins["op"] == "~") & (ins["rval"] == "F")][
        ["lval", "Est. Std", "Std. Err", "p-value"]
    ].copy()
    load.columns = ["indicator", "std_loading", "std_err", "p_value"]
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


def _fsqca_via_r(csv_path, outcome: str, conditions: list[str]):
    """Run fsQCA with R's QCA package: direct fuzzy calibration (percentile
    anchors) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report an honest message."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        # skewed/discrete columns can tie the 0.1/0.5/0.9 quantiles; calibrate
        # needs strictly increasing anchors, so nudge ties by a tiny epsilon.
        "calib <- function(x) {\n"
        "  thr <- as.numeric(quantile(x, c(0.1,0.5,0.9), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib))\n"
        "names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), '
        "incl.cut=0.8, show.cases=FALSE)\n"
        "sol <- minimize(tt, details=TRUE)\n"
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##IC\\n"); ic <- sol$IC$incl.cov\n'
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclS[i], ic$covS[i], ic$covU[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##IC":
            section = "I"
        elif section == "S" and s:
            sol_str = s
        elif section == "I" and "|" in s:
            rows.append(s.split("|"))
    if not sol_str or not rows:
        raise RuntimeError("QCA 未返回充分配置（可能无解 / 有限多样性）")
    tab = pd.DataFrame(
        rows, columns=["configuration", "consistency", "raw_coverage", "unique_coverage"]
    )
    for c in ("consistency", "raw_coverage", "unique_coverage"):
        # R prints NA (e.g. unique coverage of a single-term solution) -> NaN
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return sol_str, tab


def _spatial_reg_via_r(csv_path, outcome: str, predictors: list[str], lon: str, lat: str):
    """Spatial regression via R spdep/spatialreg: OLS residual Moran test, then
    SAR (lag) and SEM (error) on k-NN weights, AIC model choice. Reports the
    PREFERRED model's variable effects — SEM betas are marginal effects; for SAR
    the betas are NOT marginal, so impacts() direct/indirect/total are reported
    instead. Returns (diagnostics dict, preferred str, effects DataFrame)."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(predictors)
    preds_r = ", ".join(f'"{p}"' for p in predictors)
    rcode = (
        "suppressMessages({library(spdep); library(spatialreg)})\n"
        f'd <- read.csv("{csv_r}")\n'
        f'coords <- as.matrix(d[, c("{lon}","{lat}")])\n'
        "lw <- nb2listw(knn2nb(knearneigh(coords, k=min(6, nrow(d)-1))), style=\"W\")\n"
        f'f <- as.formula("{outcome} ~ {rhs}")\n'
        "ols <- lm(f, data=d); mt <- lm.morantest(ols, lw)\n"
        "sar <- lagsarlm(f, data=d, listw=lw); sem <- errorsarlm(f, data=d, listw=lw)\n"
        'pref <- if (AIC(sar) <= AIC(sem)) "SAR" else "SEM"\n'
        'cat("##DIAG\\n")\n'
        'cat(sprintf("resid_moran_p|%.5g\\n", mt$p.value))\n'
        'cat(sprintf("ols_aic|%.3f\\n", AIC(ols)))\n'
        'cat(sprintf("sar_aic|%.3f\\nsar_rho|%.4f\\n", AIC(sar), sar$rho))\n'
        'cat(sprintf("sem_aic|%.3f\\nsem_lambda|%.4f\\n", AIC(sem), sem$lambda))\n'
        'cat("##PREF\\n"); cat(pref, "\\n")\n'
        f"preds <- c({preds_r})\n"
        'cat("##COEF\\n")\n'
        'if (pref == "SAR") {\n'
        "  imp <- impacts(sar, listw=lw)\n"
        "  for (i in seq_along(preds)) cat(sprintf('%s|%.5f|%.5f|%.5f\\n', "
        "preds[i], imp$direct[i], imp$indirect[i], imp$total[i]))\n"
        "} else {\n"
        "  co <- summary(sem)$Coef\n"
        "  for (nm in preds) cat(sprintf('%s|%.5f|%.5f|%.5g\\n', nm, co[nm,1], co[nm,2], co[nm,4]))\n"
        "}\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, diag, pref, crows = None, {}, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##DIAG":
            section = "D"
        elif s == "##PREF":
            section = "P"
        elif s == "##COEF":
            section = "C"
        elif section == "P" and s:
            pref = s
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            diag[k] = float(v)
        elif "|" in s and section == "C":
            crows.append(s.split("|"))
    if not crows or not pref:
        raise RuntimeError("spatialreg 未返回系数")
    cols = (
        ["term", "direct", "indirect", "total"]
        if pref == "SAR"
        else ["term", "estimate", "std_err", "p_value"]
    )
    coef = pd.DataFrame(crows, columns=cols)
    for c in cols[1:]:
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return diag, pref, coef


def _qca_necessity_via_r(csv_path, outcome: str, conditions: list[str]):
    """QCA necessity analysis via R superSubset on fuzzy-calibrated data. Returns
    a DataFrame [expression, inclN(consistency), RoN, covN(coverage)]. RoN flags
    trivially-necessary (always-high) conditions. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x) {\n"
        "  thr <- as.numeric(quantile(x, c(0.1,0.5,0.9), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'ss <- superSubset(cal, outcome="{outcome}", conditions=c({conds_r}), '
        "incl.cut=0.9, cov.cut=0.5)\n"
        "ic <- ss$incl.cov\n"
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclN[i], ic$RoN[i], ic$covN[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    rows = [s.split("|") for s in out.splitlines() if "|" in s and not s.strip().startswith("#")]
    if not rows:
        raise RuntimeError("superSubset 未返回必要条件（无满足一致性阈值的必要项）")
    tab = pd.DataFrame(rows, columns=["expression", "consistency_inclN", "RoN", "coverage_covN"])
    for c in ("consistency_inclN", "RoN", "coverage_covN"):
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return tab


def _report(entry, fp, summary, files, override) -> str:
    lines = [
        f"# ResearchForge 分析报告：{entry.method}",
        "",
        f"- 数据：`{fp.path}`（{fp.n_rows} 行 × {fp.n_cols} 列）",
        f"- 分析：{entry.method}（{entry.family} / {entry.goal}）",
        "",
    ]
    if override:
        lines += ["> ⚠️ **知情覆盖**：该分析部分前提未满足，结果仅供参考、需谨慎解读。", ""]
    lines += ["## 结果摘要", *[f"- {s}" for s in summary], ""]
    if entry.biases:
        lines += ["## 偏差提醒（需读者判断）", *[f"- {b}" for b in entry.biases], ""]
    lines += ["## 产物文件", *[f"- `{f}`" for f in files]]
    return "\n".join(lines)


def run_analysis(
    fp: DataFingerprint,
    entry: AnalysisEntry,
    output_root: str = "outputs",
    override: bool = False,
) -> RunResult:
    df = read_table(Path(fp.path))
    d = _run_dir(output_root, entry.id)
    _init_mpl_style()
    files: list[str] = []
    summary: list[str] = []
    estimates: dict[str, float] = {}
    code: list[str] = ["import pandas as pd", f"df = pd.read_csv(r'{fp.path}')", ""]

    if entry.id == "descriptive_stats":
        df.describe(include="all").transpose().to_csv(d / "table_describe.csv", encoding="utf-8")
        files.append("table_describe.csv")
        summary.append(f"描述统计完成：{df.shape[0]} 行 × {df.shape[1]} 列")
        high_card = [c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.n_unique > 50]
        if high_card:
            summary.append(f"注意：{len(high_card)} 个高基数列（如 {high_card[0]}）描述统计意义有限。")
        code.append("df.describe(include='all').transpose().to_csv('table_describe.csv')")

    elif entry.id == "correlation":
        num = df.select_dtypes(include="number")
        corr = num.corr()
        corr.to_csv(d / "correlation.csv", encoding="utf-8")
        files.append("correlation.csv")
        _heatmap(corr, d / "correlation_heatmap.png")
        files.append("correlation_heatmap.png")
        _plotly_corr_heatmap(corr, d / "correlation_heatmap.html")
        if (d / "correlation_heatmap.html").exists():
            files.append("correlation_heatmap.html")
        summary.append(f"相关分析完成：{num.shape[1]} 个数值变量")
        code += ["num = df.select_dtypes(include='number')", "num.corr().to_csv('correlation.csv')"]

    elif entry.id in _REGRESSION:
        y, rhs_vars, formula, model = _regression(df, fp, entry)
        (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
        files.append("summary.txt")
        model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
        files.append("coefficients.csv")
        _coef_plot(model, rhs_vars, d / "coefficients.png")
        files.append("coefficients.png")
        _resid_plot(model, d / "residuals_vs_fitted.png")
        if (d / "residuals_vs_fitted.png").exists():
            files.append("residuals_vs_fitted.png")
        for v in rhs_vars:
            kn = f"Q('{v}')"
            if kn in model.params.index:
                estimates[v] = float(model.params[kn])
        key = ""
        if rhs_vars:
            kname = f"Q('{rhs_vars[0]}')"
            if kname in model.params.index:
                key = f"，关键系数 {rhs_vars[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
        n_cont = sum(1 for c in fp.columns if c.kind == "continuous")
        dv_note = f"（数据有 {n_cont} 个连续列，默认取 {y} 为因变量）" if n_cont > 1 else ""
        summary.append(f"{entry.method} 完成：因变量 {y}{key}{dv_note}")
        if not rhs_vars:
            summary.append("⚠️ 无可用解释变量，仅拟合了截距模型，结果无解释意义。")
        if entry.id == "did" and rhs_vars and fp.unit_col:
            if int(df.groupby(fp.unit_col)[rhs_vars[0]].nunique().max()) <= 1:
                summary.append(
                    f"⚠️ 处理变量 {rhs_vars[0]} 在每个单位内不随时间变化，可能不是有效的 DID 处理。"
                )
        code += [
            "import statsmodels.formula.api as smf",
            f'model = smf.ols("{formula}", data=df).fit(cov_type="HC1")',
            "print(model.summary())",
        ]

    elif entry.id == "group_comparison":
        from scipy import stats

        _excl = {fp.unit_col, fp.time_col}
        bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
        cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
        # prefer a binary group; otherwise the lowest-cardinality categorical, so a
        # high-cardinality unit/id column is never picked as the grouping variable.
        cat_cols.sort(key=lambda name: int(df[name].nunique()))
        group_candidates = bin_cols + cat_cols
        cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
        group_col = group_candidates[0] if group_candidates else None
        outcome = cont_cols[0] if cont_cols else None

        if group_col is None or outcome is None:
            summary.append("组间比较失败：未找到分组变量或连续结果变量。")
        else:
            # Per-group means/counts
            group_means = df.groupby(group_col)[outcome].agg(["mean", "count", "std"])
            group_means.to_csv(d / "group_means.csv", encoding="utf-8")
            files.append("group_means.csv")

            # Split outcome by group levels, drop NaN
            levels = df[group_col].dropna().unique().tolist()
            groups = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
            n_groups = len(groups)

            if n_groups == 2:
                stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = "Welch t-test"
            else:
                stat, p = stats.f_oneway(*groups)
                test_name = "one-way ANOVA"

            estimates["statistic"] = float(stat)
            estimates["pvalue"] = float(p)

            # Boxplot
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 4))
                plot_data = [df.loc[df[group_col] == lv, outcome].dropna().values for lv in levels]
                ax.boxplot(plot_data, tick_labels=[str(lv) for lv in levels])
                ax.set_xlabel(group_col)
                ax.set_ylabel(outcome)
                ax.set_title(f"{outcome} by {group_col}")
                fig.tight_layout()
                fig.savefig(d / "boxplot.png", dpi=150)
                plt.close(fig)
                files.append("boxplot.png")
            except Exception:
                pass

            summary.append(
                f"{entry.method} 完成：{outcome} 按 {group_col} 分 {n_groups} 组，"
                f"统计量={stat:.4f}，p={p:.3g}"
            )
            code += [
                "from scipy import stats",
                f"groups = [df.loc[df['{group_col}'] == lv, '{outcome}'].dropna().values",
                f"         for lv in df['{group_col}'].dropna().unique()]",
                "stat, p = stats.ttest_ind(*groups[:2], equal_var=False)  # or f_oneway(*groups)",
                "print(f'statistic={stat:.4f}, p={p:.3g}')",
            ]

    elif entry.id == "random_forest":
        cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
        binary_cols = [c.name for c in fp.columns if c.kind == "binary"]

        # Prefer a continuous outcome (regression). Classify a binary outcome only
        # when there is no continuous column — a lone binary is usually a
        # treatment / flag *feature*, not the prediction target. This prevents
        # silently running the wrong analysis on the common "outcome + indicator" shape.
        if cont_cols:
            outcome, is_clf = cont_cols[0], False
        elif binary_cols:
            outcome, is_clf = binary_cols[0], True
        else:
            outcome, is_clf = None, False

        exclude = {outcome, fp.unit_col, fp.time_col}
        features = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ]

        if outcome is None:
            summary.append("随机森林失败：未找到合适的结果变量（需要连续型或二值列）。")
        elif not features:
            summary.append("随机森林失败：未找到可用的特征列。")
        else:
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.model_selection import train_test_split

                mask = df[features].notna().all(axis=1) & df[outcome].notna()
                X = df.loc[mask, features]
                y = df.loc[mask, outcome]

                if y.nunique() < 2:
                    raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")

                split_kwargs = {"test_size": 0.25, "random_state": 0}
                if is_clf and int(y.value_counts().min()) >= 2:
                    split_kwargs["stratify"] = y
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

                model = (
                    RandomForestClassifier(n_estimators=200, random_state=0)
                    if is_clf
                    else RandomForestRegressor(n_estimators=200, random_state=0)
                )

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                import pandas as pd
                imp_df = pd.DataFrame(
                    {"feature": features, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
                imp_df.to_csv(d / "feature_importances.csv", index=False, encoding="utf-8")
                files.append("feature_importances.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
                    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                    ax.set_xlabel("importance")
                    ax.set_title(f"Feature importances — {outcome}")
                    fig.tight_layout()
                    fig.savefig(d / "feature_importances.png", dpi=150)
                    plt.close(fig)
                    files.append("feature_importances.png")
                except Exception:
                    pass

                estimates["test_score"] = float(score)
                score_label = "accuracy" if is_clf else "R²"
                task_label = "分类" if is_clf else "回归"
                summary.append(
                    f"{entry.method} 完成：{task_label}预测 {outcome}，"
                    f"测试集得分={score:.4f}（{score_label}）"
                )
                code += [
                    "from sklearn.ensemble import "
                    + ("RandomForestClassifier" if is_clf else "RandomForestRegressor"),
                    "from sklearn.model_selection import train_test_split",
                    f"features = {features!r}",
                    f"X = df[features].dropna()",
                    f"y = df.loc[X.index, '{outcome}'].dropna()",
                    f"X = X.loc[y.index]",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)",
                    "model = "
                    + ("RandomForestClassifier" if is_clf else "RandomForestRegressor")
                    + "(n_estimators=200, random_state=0)",
                    "model.fit(X_train, y_train)",
                    f"print('score:', model.score(X_test, y_test))",
                ]
            except Exception as err:
                summary.append(f"随机森林执行失败：{err}")

    elif entry.id == "xgboost":
        cont_cols = [c.name for c in fp.columns if c.kind == "continuous"]
        binary_cols = [c.name for c in fp.columns if c.kind == "binary"]

        # Prefer a continuous outcome (regression). Classify a binary outcome only
        # when there is no continuous column — a lone binary is usually a
        # treatment / flag *feature*, not the prediction target. This prevents
        # silently running the wrong analysis on the common "outcome + indicator" shape.
        if cont_cols:
            outcome, is_clf = cont_cols[0], False
        elif binary_cols:
            outcome, is_clf = binary_cols[0], True
        else:
            outcome, is_clf = None, False

        exclude = {outcome, fp.unit_col, fp.time_col}
        features = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name not in exclude
        ]

        if outcome is None:
            summary.append("XGBoost 失败：未找到合适的结果变量（需要连续型或二值列）。")
        elif not features:
            summary.append("XGBoost 失败：未找到可用的特征列。")
        else:
            try:
                from xgboost import XGBClassifier, XGBRegressor
                from sklearn.model_selection import train_test_split

                mask = df[features].notna().all(axis=1) & df[outcome].notna()
                X = df.loc[mask, features]
                y = df.loc[mask, outcome]

                if y.nunique() < 2:
                    raise ValueError(f"结果变量 {outcome} 取值不足两类，无法建模")

                split_kwargs = {"test_size": 0.25, "random_state": 0}
                if is_clf and int(y.value_counts().min()) >= 2:
                    split_kwargs["stratify"] = y
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

                model = (
                    XGBClassifier(n_estimators=200, random_state=0, verbosity=0)
                    if is_clf
                    else XGBRegressor(n_estimators=200, random_state=0, verbosity=0)
                )

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                import pandas as pd
                imp_df = pd.DataFrame(
                    {"feature": features, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
                imp_df.to_csv(d / "feature_importances.csv", index=False, encoding="utf-8")
                files.append("feature_importances.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
                    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                    ax.set_xlabel("importance")
                    ax.set_title(f"Feature importances — {outcome}")
                    fig.tight_layout()
                    fig.savefig(d / "feature_importances.png", dpi=150)
                    plt.close(fig)
                    files.append("feature_importances.png")
                except Exception:
                    pass

                estimates["test_score"] = float(score)
                score_label = "accuracy" if is_clf else "R²"
                task_label = "分类" if is_clf else "回归"
                summary.append(
                    f"{entry.method} 完成：{task_label}预测 {outcome}，"
                    f"测试集得分={score:.4f}（{score_label}）"
                )
                code += [
                    "from xgboost import "
                    + ("XGBClassifier" if is_clf else "XGBRegressor"),
                    "from sklearn.model_selection import train_test_split",
                    f"features = {features!r}",
                    f"X = df[features].dropna()",
                    f"y = df.loc[X.index, '{outcome}'].dropna()",
                    f"X = X.loc[y.index]",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)",
                    "model = "
                    + ("XGBClassifier" if is_clf else "XGBRegressor")
                    + "(n_estimators=200, random_state=0, verbosity=0)",
                    "model.fit(X_train, y_train)",
                    f"print('score:', model.score(X_test, y_test))",
                ]
            except Exception as err:
                summary.append(f"XGBoost 执行失败：{err}")

    elif entry.id == "kmeans_clustering":
        features = [
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ]
        X = df[features].dropna()  # keep X.index for alignment
        if len(features) < 2 or len(X) < 4:
            summary.append("K-means 跳过：连续特征不足或有效样本太少。")
        else:
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                from sklearn.decomposition import PCA
                import numpy as np
                import pandas as pd

                Xs = StandardScaler().fit_transform(X)
                n = len(Xs)
                k_max = max(2, min(6, n // 10))
                best = None  # (score, k, labels)
                for k in range(2, min(k_max, n - 1) + 1):
                    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
                    if len(set(labels)) < 2:
                        continue
                    score = silhouette_score(Xs, labels)
                    if best is None or score > best[0]:
                        best = (score, k, labels)

                if best is None:
                    summary.append("K-means 未能形成有效聚类（数据可能近常数）。")
                else:
                    score, k, labels = best
                    k = len(set(labels))  # actual cluster count (KMeans may collapse on duplicate points)
                    assign = pd.DataFrame({"row": X.index, "cluster": labels})
                    assign.to_csv(d / "cluster_assignments.csv", index=False, encoding="utf-8")
                    files.append("cluster_assignments.csv")

                    profile_out = X.groupby(labels).mean()
                    size = X.groupby(labels).size()
                    profile_out["size"] = size.values
                    profile_out.to_csv(d / "cluster_profile.csv", encoding="utf-8")
                    files.append("cluster_profile.csv")

                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        n_components = min(2, len(features))
                        pca_coords = PCA(n_components=n_components).fit_transform(Xs)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        if n_components == 2:
                            ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10", s=20)
                        else:
                            ax.scatter(pca_coords[:, 0], [0] * len(pca_coords), c=labels, cmap="tab10", s=20)
                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2" if n_components == 2 else "")
                        ax.set_title(f"K-means (k={k}) — PCA projection")
                        fig.tight_layout()
                        fig.savefig(d / "pca_scatter.png", dpi=150)
                        plt.close(fig)
                        files.append("pca_scatter.png")
                        _plotly_scatter(
                            pca_coords, labels, d / "cluster_scatter.html",
                            f"K-means (k={k}) — interactive", "PC1",
                            "PC2" if n_components == 2 else "",
                        )
                        if (d / "cluster_scatter.html").exists():
                            files.append("cluster_scatter.html")
                    except Exception:
                        pass

                    _silhouette_plot(Xs, labels, d / "silhouette.png")
                    if (d / "silhouette.png").exists():
                        files.append("silhouette.png")

                    estimates["silhouette"] = float(score)
                    estimates["k"] = float(k)
                    summary.append(
                        f"{entry.method} 完成：在 {len(features)} 个连续特征上聚成 {k} 类，silhouette={score:.4f}"
                    )
                    code += [
                        "from sklearn.preprocessing import StandardScaler",
                        "from sklearn.cluster import KMeans",
                        "from sklearn.metrics import silhouette_score",
                        f"features = {features!r}",
                        "X = df[features].dropna()",
                        "Xs = StandardScaler().fit_transform(X)",
                        f"labels = KMeans(n_clusters={k}, random_state=0, n_init=10).fit_predict(Xs)",
                        "print('silhouette:', silhouette_score(Xs, labels))",
                    ]
            except Exception as err:
                summary.append(f"K-means 执行失败：{err}")

    elif entry.id == "pca":
        features = [
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ]
        X = df[features].dropna()
        if len(features) < 2 or len(X) < 3:
            summary.append("PCA 跳过：连续特征不足或样本太少。")
        else:
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                import numpy as np
                import pandas as pd

                Xs = StandardScaler().fit_transform(X)
                n_comp = min(len(features), 10, len(X) - 1)
                pca = PCA(n_components=n_comp).fit(Xs)
                evr = pca.explained_variance_ratio_

                # explained_variance.csv: component (PC1..), explained_variance_ratio, cumulative
                ev_df = pd.DataFrame({
                    "component": [f"PC{i+1}" for i in range(n_comp)],
                    "explained_variance_ratio": evr,
                    "cumulative": np.cumsum(evr),
                })
                ev_df.to_csv(d / "explained_variance.csv", index=False, encoding="utf-8")
                files.append("explained_variance.csv")

                # loadings.csv: rows=features, cols=PC1..n
                load_df = pd.DataFrame(
                    pca.components_.T,
                    index=features,
                    columns=[f"PC{i+1}" for i in range(n_comp)],
                )
                load_df.to_csv(d / "loadings.csv", encoding="utf-8")
                files.append("loadings.csv")

                # scree plot (bar of evr) -> pca_scree.png
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar([f"PC{i+1}" for i in range(n_comp)], evr)
                    ax.set_xlabel("component")
                    ax.set_ylabel("explained variance ratio")
                    ax.set_title("PCA scree plot")
                    fig.tight_layout()
                    fig.savefig(d / "pca_scree.png", dpi=150)
                    plt.close(fig)
                    files.append("pca_scree.png")
                except Exception:
                    pass

                estimates["pc1_explained_ratio"] = float(evr[0])
                estimates["n_components"] = float(n_comp)
                estimates["cum_explained_top2"] = float(np.cumsum(evr)[min(1, n_comp - 1)])
                summary.append(
                    f"{entry.method} 完成：{len(features)} 个连续特征 -> {n_comp} 个主成分，"
                    f"PC1 解释方差={evr[0]:.1%}"
                )
                code += [
                    "from sklearn.preprocessing import StandardScaler",
                    "from sklearn.decomposition import PCA",
                    "import numpy as np",
                    f"features = {features!r}",
                    "X = df[features].dropna()",
                    "Xs = StandardScaler().fit_transform(X)",
                    f"pca = PCA(n_components={n_comp}).fit(Xs)",
                    "print('explained variance ratio:', pca.explained_variance_ratio_)",
                ]
            except Exception as err:
                summary.append(f"PCA 执行失败：{err}")

    elif entry.id == "arima":
        time_col = fp.time_col
        # value_col: forecast the first continuous column. Time columns are
        # datetime/id/count kind (never continuous), so they are never picked here.
        value_col = next((c.name for c in fp.columns if c.kind == "continuous"), None)

        if time_col is None or value_col is None:
            summary.append(
                "ARIMA 失败：未找到时间列或连续值列，请检查数据结构。"
            )
        else:
            try:
                from statsmodels.tsa.arima.model import ARIMA

                sorted_df = df.sort_values(time_col)
                dup = int(sorted_df[time_col].duplicated().sum())
                if dup:
                    sorted_df = sorted_df.drop_duplicates(subset=time_col, keep="first")
                    summary.append(f"注意：{dup} 个重复时间点已去重（保留首次）。")
                y = sorted_df[value_col].astype(float).reset_index(drop=True)
                if y.nunique() < 2 or len(y) < 10:
                    raise ValueError(f"序列有效观测不足或近常数（n={len(y)}），无法拟合 ARIMA")

                model = ARIMA(y, order=(1, 1, 1)).fit()

                (d / "model_summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("model_summary.txt")

                steps = 10
                fc = model.forecast(steps=steps)
                import pandas as _pd
                fc_df = _pd.DataFrame({"step": list(range(1, steps + 1)), "forecast": fc.tolist()})
                fc_df.to_csv(d / "forecast.csv", index=False, encoding="utf-8")
                files.append("forecast.csv")

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(len(y)), y, label="observed")
                    fc_x = list(range(len(y), len(y) + steps))
                    ax.plot(fc_x, fc.tolist(), color="red", linestyle="--", label="forecast")
                    ax.set_xlabel("period index")
                    ax.set_ylabel(value_col)
                    ax.set_title(f"ARIMA(1,1,1) — {value_col}")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(d / "forecast.png", dpi=150)
                    plt.close(fig)
                    files.append("forecast.png")
                except Exception:
                    pass

                estimates["aic"] = float(model.aic)
                summary.append(
                    f"{entry.method} 完成：对 {value_col} 拟合 ARIMA(1,1,1)，"
                    f"AIC={model.aic:.2f}，预测未来 {steps} 期"
                )
                code += [
                    "from statsmodels.tsa.arima.model import ARIMA",
                    f"y = df.sort_values('{time_col}')['{value_col}'].astype(float).reset_index(drop=True)",
                    "model = ARIMA(y, order=(1, 1, 1)).fit()",
                    "print(model.summary())",
                    f"fc = model.forecast(steps={steps})",
                ]
            except Exception as err:
                summary.append(f"ARIMA 拟合失败：{err}")

    elif entry.id == "logistic_regression":
        import statsmodels.formula.api as smf

        # identify outcome (first binary column, excluding unit/time) and predictors
        binary_cols = [
            c.name
            for c in fp.columns
            if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col}
        ]
        outcome = binary_cols[0] if binary_cols else None
        exclude = {outcome, fp.unit_col, fp.time_col}
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count"} and c.name not in exclude
        ][:5]

        if outcome is None:
            summary.append("逻辑回归失败：未找到二值结果变量。")
        else:
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            try:
                model = smf.logit(formula, data=df).fit(disp=False)
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                model.summary2().tables[1].to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                key = ""
                if predictors:
                    kname = f"Q('{predictors[0]}')"
                    if kname in model.params.index:
                        key = f"，关键系数 {predictors[0]} = {model.params[kname]:.4f} (p={model.pvalues[kname]:.3g})"
                amb = (
                    f"（数据有 {len(binary_cols)} 个二值列，已取 {outcome}；若它实为处理/标志变量请改选）"
                    if len(binary_cols) > 1
                    else ""
                )
                summary.append(f"{entry.method} 完成：结果变量 {outcome}{key}{amb}")
                code += [
                    "import statsmodels.formula.api as smf",
                    f'model = smf.logit("{formula}", data=df).fit(disp=False)',
                    "print(model.summary())",
                ]
            except Exception as err:
                summary.append(f"逻辑回归未收敛/失败：{err}")

    elif entry.id == "poisson_regression":
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        import numpy as np

        _excl = {fp.unit_col, fp.time_col}
        count_cols = [
            c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
        ]
        outcome = count_cols[0] if count_cols else None

        if outcome is None:
            summary.append("泊松回归失败：未找到计数型结果变量。")
        else:
            amb = (
                f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
                if len(count_cols) > 1
                else ""
            )
            exclude = {outcome, fp.unit_col, fp.time_col}
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in exclude
            ][:5]
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            recipe = (
                "import statsmodels.formula.api as smf\n"
                "import statsmodels.api as sm\n"
                f'model = smf.glm("{formula}", data=df, family=sm.families.Poisson()).fit()\n'
                "print(model.summary())"
            )
            try:
                model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                tab = model.summary2().tables[1].copy()
                tab["rate_ratio"] = np.exp(model.params.values)
                tab.to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                summary.append(
                    f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}"
                )
                code += [recipe]
            except Exception as err:
                summary.append(f"泊松回归失败：{err}")

    elif entry.id == "negative_binomial_regression":
        import statsmodels.formula.api as smf
        import numpy as np

        _excl = {fp.unit_col, fp.time_col}
        count_cols = [
            c.name for c in fp.columns if c.kind == "count" and c.name not in _excl
        ]
        outcome = count_cols[0] if count_cols else None

        if outcome is None:
            summary.append("负二项回归失败：未找到计数型结果变量。")
        else:
            amb = (
                f"（数据有 {len(count_cols)} 个计数列，已取 {outcome}；若它实为 ID/编码而非计数结果，请改选）"
                if len(count_cols) > 1
                else ""
            )
            exclude = {outcome, fp.unit_col, fp.time_col}
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in exclude
            ][:5]
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            recipe = (
                "import statsmodels.formula.api as smf\n"
                f'model = smf.negativebinomial("{formula}", data=df).fit(disp=False)\n'
                "print(model.summary())"
            )
            try:
                model = smf.negativebinomial(formula, data=df).fit(disp=False)
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                tab = model.summary2().tables[1].copy()
                # model.params includes an 'alpha' (dispersion) row at the end;
                # summary2().tables[1] also includes it — lengths always match,
                # so exp() of all rows is safe (exp(alpha) is a positive scalar,
                # harmless alongside the log-rate coefficients).
                tab["rate_ratio"] = np.exp(model.params.values)
                tab.to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in model.params.index:
                        estimates[v] = float(model.params[kn])
                summary.append(
                    f"{entry.method} 完成：计数结果 {outcome}，{len(predictors)} 个预测变量{amb}"
                )
                code += [recipe]
            except Exception as err:
                summary.append(f"负二项回归失败：{err}")

    elif entry.id == "quantile_regression":
        import statsmodels.formula.api as smf
        import pandas as pd

        _excl = {fp.unit_col, fp.time_col}
        outcome = next(
            (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl),
            None,
        )
        if outcome is None:
            summary.append("分位数回归失败：未找到连续型结果变量。")
        else:
            exclude = {outcome, fp.unit_col, fp.time_col}
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
            ][:5]
            rhs = [f"Q('{v}')" for v in predictors]
            formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
            taus = [0.25, 0.50, 0.75]
            recipe = (
                "import statsmodels.formula.api as smf\n"
                f'qr = smf.quantreg("{formula}", data=df)\n'
                "for tau in (0.25, 0.5, 0.75):\n"
                "    print(tau, qr.fit(q=tau).params)\n"
            )
            try:
                qr = smf.quantreg(formula, data=df)
                fits = {tau: qr.fit(q=tau) for tau in taus}
                med = fits[0.50]
                (d / "summary.txt").write_text(str(med.summary()), encoding="utf-8")
                files.append("summary.txt")
                # coefficients side by side across quantiles — the whole point of
                # quantile regression is seeing how effects differ down the
                # outcome distribution (τ=0.25 lower tail … 0.75 upper tail).
                tab = pd.DataFrame({f"tau={tau}": fits[tau].params for tau in taus})
                tab.to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(med, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                _quantile_process_plot(qr, predictors, d / "quantile_process.png")
                if (d / "quantile_process.png").exists():
                    files.append("quantile_process.png")
                for v in predictors:
                    kn = f"Q('{v}')"
                    if kn in med.params.index:
                        estimates[v] = float(med.params[kn])
                summary.append(
                    f"{entry.method} 完成：结果 {outcome}，{len(predictors)} 个预测变量，"
                    "τ=0.25/0.50/0.75（中位数与尾部效应对比见 coefficients.csv）"
                )
                code += [recipe]
            except Exception as err:
                summary.append(f"分位数回归失败：{err}")

    elif entry.id == "ordered_logit":
        import pandas as pd
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        _excl = {fp.unit_col, fp.time_col}
        # ordinal outcome: a small ordered scale (3–10 levels). Prefer numeric
        # (count) where the level order is unambiguous; fall back to categorical.
        ord_cols = [
            c
            for c in fp.columns
            if c.kind in {"count", "categorical"}
            and 3 <= c.n_unique <= 10
            and c.name not in _excl
        ]
        ord_cols.sort(key=lambda c: 0 if c.kind == "count" else 1)
        outcome = ord_cols[0].name if ord_cols else None

        if outcome is None:
            summary.append("有序 Logit 失败：未找到有序结果变量（3–10 个等级）。")
        else:
            exclude = {outcome, fp.unit_col, fp.time_col}
            predictors = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary", "count"} and c.name not in exclude
            ][:5]
            try:
                if not predictors:
                    raise ValueError("没有可用预测变量")
                yc = pd.Categorical(df[outcome], ordered=True)
                levels = list(yc.categories)
                model = OrderedModel(yc, df[predictors], distr="logit").fit(
                    method="bfgs", disp=False
                )
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                # OrderedResults lacks summary2(); build the table from arrays.
                # Rows include predictor slopes plus threshold cutpoints (e.g. 1/2).
                pd.DataFrame(
                    {
                        "coef": model.params,
                        "std_err": model.bse,
                        "z": model.tvalues,
                        "P>|z|": model.pvalues,
                    }
                ).to_csv(d / "coefficients.csv", encoding="utf-8")
                files.append("coefficients.csv")
                _coef_plot(model, predictors, d / "coefficients.png")
                files.append("coefficients.png")
                _ordinal_prob_plot(model, df, predictors, levels, d / "predicted_probabilities.png")
                if (d / "predicted_probabilities.png").exists():
                    files.append("predicted_probabilities.png")
                for v in predictors:
                    if v in model.params.index:
                        estimates[v] = float(model.params[v])
                is_text = df[outcome].dtype == object or str(df[outcome].dtype) == "string"
                note = f"（等级顺序假定为 {levels}；若不符请重新编码）" if is_text else ""
                summary.append(
                    f"{entry.method} 完成：有序结果 {outcome}（{len(levels)} 级），"
                    f"{len(predictors)} 个预测变量{note}"
                )
                code += [
                    "from statsmodels.miscmodels.ordinal_model import OrderedModel",
                    f"yc = pd.Categorical(df['{outcome}'], ordered=True)",
                    f"model = OrderedModel(yc, df[{predictors!r}], distr='logit')"
                    ".fit(method='bfgs', disp=False)",
                    "print(model.summary())",
                ]
            except Exception as err:
                summary.append(f"有序 Logit 未收敛/失败：{err}")

    elif entry.id == "mixed_effects":
        import statsmodels.formula.api as smf

        # outcome: first continuous column
        outcome = next((c.name for c in fp.columns if c.kind == "continuous"), None)
        if outcome is None:
            summary.append("混合模型失败：未找到连续结果变量。")
        else:
            # group_col: prefer unit_col; else first categorical/binary that is not outcome
            if fp.unit_col:
                group_col = fp.unit_col
            else:
                group_col = next(
                    (
                        c.name
                        for c in fp.columns
                        if c.kind in {"categorical", "binary"} and c.name != outcome
                    ),
                    None,
                )
            if group_col is None:
                summary.append("混合模型失败：未找到分组变量(随机效应)。")
            else:
                predictors = [
                    c.name
                    for c in fp.columns
                    if c.kind in {"continuous", "count", "binary"}
                    and c.name not in {outcome, group_col, fp.unit_col, fp.time_col}
                ][:5]
                rhs = [f"Q('{v}')" for v in predictors]
                # Control for time on panel data — otherwise a staggered treatment is
                # confounded with the time trend (mirrors _regression's FE handling).
                if fp.time_col and fp.time_col != group_col:
                    rhs.append(f"C(Q('{fp.time_col}'))")
                formula = f"Q('{outcome}') ~ " + (" + ".join(rhs) if rhs else "1")
                try:
                    model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit()
                    (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                    files.append("summary.txt")
                    try:
                        import pandas as pd
                        pd.DataFrame(model.summary().tables[1]).to_csv(
                            d / "coefficients.csv", encoding="utf-8"
                        )
                    except Exception:
                        import pandas as pd
                        model.params.to_frame(name="coef").to_csv(
                            d / "coefficients.csv", encoding="utf-8"
                        )
                    files.append("coefficients.csv")
                    for v in predictors:
                        kn = f"Q('{v}')"
                        if kn in model.params.index:
                            estimates[v] = float(model.params[kn])
                    summary.append(
                        f"{entry.method} 完成：结果变量 {outcome}，随机效应分组 {group_col}，"
                        f"固定效应 {len(predictors)} 个"
                    )
                    code += [
                        "import statsmodels.formula.api as smf",
                        f'model = smf.mixedlm("{formula}", data=df, groups=df["{group_col}"]).fit()',
                        "print(model.summary())",
                    ]
                except Exception as err:
                    summary.append(f"混合模型未收敛/失败：{err}")

    elif entry.id == "diversity_indices":
        import numpy as np
        import pandas as pd

        species = [
            c.name
            for c in fp.columns
            if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
        ]
        if len(species) < 2:
            summary.append("多样性指数跳过：未找到 ≥2 个计数列（物种丰度矩阵）。")
        else:
            mat = df[species].fillna(0).clip(lower=0).astype(float)

            def _shannon(counts):
                total = counts.sum()
                if total <= 0:
                    return 0.0
                p = counts[counts > 0] / total
                return float(-(p * np.log(p)).sum())

            def _simpson(counts):
                total = counts.sum()
                if total <= 0:
                    return 0.0
                p = counts / total
                return float(1.0 - (p ** 2).sum())

            div = pd.DataFrame(
                {
                    "shannon": mat.apply(_shannon, axis=1).round(4),
                    "simpson": mat.apply(_simpson, axis=1).round(4),
                    "richness": (mat > 0).sum(axis=1).astype(int),
                    "total_abundance": mat.sum(axis=1),
                }
            )
            div.to_csv(d / "diversity.csv", encoding="utf-8")
            files.append("diversity.csv")
            estimates["mean_shannon"] = float(div["shannon"].mean())
            estimates["mean_richness"] = float(div["richness"].mean())
            estimates["n_species"] = float(len(species))
            summary.append(
                f"{entry.method} 完成：{len(species)} 个物种 × {len(df)} 个样点，"
                f"平均 Shannon={div['shannon'].mean():.3f}，平均丰富度={div['richness'].mean():.2f}"
            )
            code += [
                "import numpy as np",
                f"mat = df[{species!r}].fillna(0)",
                "# 每行(样点): Shannon=-sum(p*ln p), Simpson=1-sum(p^2), richness=present species",
            ]

    elif entry.id == "hierarchical_clustering":
        features = [
            c.name
            for c in fp.columns
            if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
        ]
        X = df[features].dropna()
        if len(features) < 2 or len(X) < 5:
            summary.append("层次聚类跳过：连续特征不足或样本太少。")
        else:
            try:
                import numpy as np
                import pandas as pd
                from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
                from scipy.spatial.distance import pdist
                from sklearn.preprocessing import StandardScaler

                Xs = StandardScaler().fit_transform(X)
                n = len(Xs)
                k = max(2, min(4, n // 5))
                Z = linkage(Xs, method="ward")
                labels = fcluster(Z, t=k, criterion="maxclust")
                pd.DataFrame({"row": X.index, "cluster": labels}).to_csv(
                    d / "cluster_assignments.csv", index=False, encoding="utf-8"
                )
                files.append("cluster_assignments.csv")
                coph, _ = cophenet(Z, pdist(Xs))

                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7, 4))
                    dendrogram(Z, ax=ax, no_labels=(n > 30))
                    ax.set_title(f"Hierarchical clustering (Ward, k={k})")
                    fig.tight_layout()
                    fig.savefig(d / "dendrogram.png", dpi=150)
                    plt.close(fig)
                    files.append("dendrogram.png")
                except Exception:
                    pass

                _silhouette_plot(Xs, labels, d / "silhouette.png")
                if (d / "silhouette.png").exists():
                    files.append("silhouette.png")

                estimates["n_clusters"] = float(len(set(labels)))
                estimates["cophenetic_corr"] = round(float(coph), 4)
                summary.append(
                    f"{entry.method} 完成：{len(features)} 个特征 × {n} 个样本聚成 "
                    f"{len(set(labels))} 类（cophenetic 相关={coph:.3f}）"
                )
                code += [
                    "from scipy.cluster.hierarchy import linkage, fcluster",
                    "Z = linkage(Xs, method='ward')",
                    f"labels = fcluster(Z, t={k}, criterion='maxclust')",
                ]
            except Exception as err:
                summary.append(f"层次聚类失败：{err}")

    elif entry.id == "beta_diversity":
        import numpy as np
        import pandas as pd

        species = [
            c.name
            for c in fp.columns
            if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
        ]
        if len(species) < 2 or len(df) < 2:
            summary.append("Beta 多样性跳过：需要 ≥2 个计数列（物种丰度）与 ≥2 个样点。")
        else:
            try:
                from scipy.spatial.distance import pdist, squareform

                mat = df[species].fillna(0).clip(lower=0).astype(float).values
                dist = squareform(pdist(mat, metric="braycurtis"))
                labels = [f"site{i + 1}" for i in range(len(mat))]
                pd.DataFrame(np.round(dist, 4), index=labels, columns=labels).to_csv(
                    d / "bray_curtis.csv", encoding="utf-8"
                )
                files.append("bray_curtis.csv")

                iu = np.triu_indices(len(mat), k=1)
                mean_bc = float(np.nanmean(dist[iu])) if iu[0].size else 0.0
                estimates["mean_bray_curtis"] = round(mean_bc, 4)
                estimates["n_sites"] = float(len(mat))

                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(dist, cmap="viridis", vmin=0, vmax=1)
                    fig.colorbar(im)
                    ax.set_title("Bray-Curtis dissimilarity")
                    fig.tight_layout()
                    fig.savefig(d / "bray_curtis_heatmap.png", dpi=150)
                    plt.close(fig)
                    files.append("bray_curtis_heatmap.png")
                except Exception:
                    pass

                summary.append(
                    f"{entry.method} 完成：{len(species)} 个物种 × {len(mat)} 个样点，"
                    f"平均 Bray-Curtis 相异度={mean_bc:.3f}"
                )
                code += [
                    "from scipy.spatial.distance import pdist, squareform",
                    f"mat = df[{species!r}].fillna(0).values",
                    "dist = squareform(pdist(mat, metric='braycurtis'))",
                ]
            except Exception as err:
                summary.append(f"Beta 多样性失败：{err}")

    elif entry.id == "nmds":
        species = [
            c.name
            for c in fp.columns
            if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
        ]
        if len(species) < 2 or len(df) < 4:
            summary.append("NMDS 跳过：需要 ≥2 个计数列（物种丰度）与 ≥4 个样点。")
        else:
            try:
                import numpy as np
                import pandas as pd
                from scipy.spatial.distance import pdist, squareform
                from sklearn.manifold import MDS

                mat = df[species].fillna(0).clip(lower=0).astype(float)
                mat = mat[mat.sum(axis=1) > 0]  # Bray-Curtis undefined for empty sites
                if len(mat) < 4:
                    summary.append("NMDS 跳过：有效（非空）样点不足 4 个。")
                else:
                    dist = squareform(pdist(mat.values, metric="braycurtis"))
                    mds = MDS(
                        n_components=2,
                        metric=False,
                        dissimilarity="precomputed",
                        random_state=0,
                        n_init=4,
                        max_iter=300,
                    )
                    coords = mds.fit_transform(dist)
                    labels = [f"site{i + 1}" for i in range(len(mat))]
                    pd.DataFrame(
                        np.round(coords, 4), index=labels, columns=["NMDS1", "NMDS2"]
                    ).to_csv(d / "nmds_coords.csv", encoding="utf-8")
                    files.append("nmds_coords.csv")
                    estimates["stress"] = round(float(mds.stress_), 4)
                    estimates["n_sites"] = float(len(mat))

                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.scatter(coords[:, 0], coords[:, 1], s=25)
                        ax.set_xlabel("NMDS1")
                        ax.set_ylabel("NMDS2")
                        ax.set_title(f"NMDS ordination (stress={mds.stress_:.3f})")
                        fig.tight_layout()
                        fig.savefig(d / "nmds_ordination.png", dpi=150)
                        plt.close(fig)
                        files.append("nmds_ordination.png")
                    except Exception:
                        pass

                    summary.append(
                        f"{entry.method} 完成：{len(species)} 物种 × {len(mat)} 样点 → 2D 排序，"
                        f"stress={mds.stress_:.4f}"
                    )
                    code += [
                        "from sklearn.manifold import MDS",
                        "from scipy.spatial.distance import pdist, squareform",
                        "dist = squareform(pdist(mat.values, metric='braycurtis'))",
                        "coords = MDS(n_components=2, metric=False, dissimilarity='precomputed').fit_transform(dist)",
                    ]
            except Exception as err:
                summary.append(f"NMDS 失败：{err}")

    elif entry.id == "permanova":
        import numpy as np
        import pandas as pd
        from scipy.spatial.distance import pdist, squareform

        species = [
            c.name
            for c in fp.columns
            if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
        ]
        _excl = {fp.unit_col, fp.time_col}
        bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
        cat_cols = [c.name for c in fp.columns if c.kind == "categorical" and c.name not in _excl]
        cat_cols.sort(key=lambda name: int(df[name].nunique()))
        group_col = (bin_cols + cat_cols)[0] if (bin_cols + cat_cols) else None

        if len(species) < 2 or group_col is None or len(df) < 6:
            summary.append(
                "PERMANOVA 跳过：需要 ≥2 个计数列（物种丰度）、一个分组变量，以及 ≥6 个样点。"
            )
        else:
            try:
                sub = df[species + [group_col]].dropna()
                mat = sub[species].clip(lower=0).astype(float).values
                g = sub[group_col].astype(str).values
                keep = mat.sum(axis=1) > 0  # Bray-Curtis undefined for empty rows
                mat, g = mat[keep], g[keep]
                groups = sorted(set(g))
                a = len(groups)
                N = len(g)
                if a < 2 or N < 6:
                    summary.append("PERMANOVA 跳过：有效样本分组后不足两组或总样本 <6。")
                else:
                    D = squareform(pdist(mat, metric="braycurtis"))
                    D2 = D ** 2

                    # ---- pseudo-F (Anderson 2001) ----
                    def _ss_within(D2, labels):
                        ss = 0.0
                        for lev in set(labels):
                            idx = np.where(labels == lev)[0]
                            ng = len(idx)
                            if ng > 1:
                                sub2 = D2[np.ix_(idx, idx)]
                                ss += sub2[np.triu_indices(ng, k=1)].sum() / ng
                        return ss

                    SS_total = D2[np.triu_indices(N, k=1)].sum() / N

                    def _pseudo_F(labels):
                        ssw = _ss_within(D2, labels)
                        ssa = SS_total - ssw
                        # F = (SSA/(a-1)) / (SSW/(N-a)); guard ssw==0
                        denom = ssw / (N - a)
                        return float((ssa / (a - 1)) / denom) if denom > 0 else float("nan")

                    F_obs = _pseudo_F(np.asarray(g))
                    rng = np.random.default_rng(0)
                    n_perm = 999
                    count = 0
                    labs = np.asarray(g)
                    for _ in range(n_perm):
                        perm = rng.permutation(labs)
                        if _pseudo_F(perm) >= F_obs:
                            count += 1
                    p_value = (count + 1) / (n_perm + 1)

                    pd.DataFrame(
                        [{
                            "pseudo_F": round(F_obs, 4),
                            "p_value": round(p_value, 4),
                            "n_groups": a,
                            "N": N,
                            "n_perm": n_perm,
                        }]
                    ).to_csv(d / "permanova_result.csv", index=False, encoding="utf-8")
                    files.append("permanova_result.csv")

                    estimates["pseudo_F"] = round(F_obs, 4)
                    estimates["p_value"] = round(p_value, 4)
                    summary.append(
                        f"{entry.method} 完成：按 {group_col} 分 {a} 组，"
                        f"pseudo-F={F_obs:.3f}，p={p_value:.3f}（{n_perm} 次置换）"
                    )
                    code += [
                        "import numpy as np",
                        "from scipy.spatial.distance import pdist, squareform",
                        f"species = {species!r}",
                        f"sub = df[species + ['{group_col}']].dropna()",
                        "mat = sub[species].clip(lower=0).astype(float).values",
                        f"g = sub['{group_col}'].astype(str).values",
                        "keep = mat.sum(axis=1) > 0",
                        "mat, g = mat[keep], g[keep]",
                        "D = squareform(pdist(mat, metric='braycurtis'))",
                        "D2 = D ** 2",
                        "N = len(g)",
                        "a = len(set(g))",
                        "SS_total = D2[np.triu_indices(N, k=1)].sum() / N",
                        "def ss_within(labels):",
                        "    tot = 0.0",
                        "    for lev in set(labels):",
                        "        idx = np.where(labels == lev)[0]; ng = len(idx)",
                        "        if ng > 1:",
                        "            tot += D2[np.ix_(idx, idx)][np.triu_indices(ng, k=1)].sum() / ng",
                        "    return tot",
                        "def pseudo_F(labels):",
                        "    w = ss_within(labels); return ((SS_total - w)/(a-1)) / (w/(N-a))",
                        "rng = np.random.default_rng(0); g = np.asarray(g)",
                        "F_obs = pseudo_F(g)",
                        "p = (sum(pseudo_F(rng.permutation(g)) >= F_obs for _ in range(999)) + 1) / 1000",
                        "print('pseudo-F =', round(F_obs, 4), 'p =', round(p, 4))",
                    ]
            except Exception as err:
                summary.append(f"PERMANOVA 失败：{err}")

    elif entry.id == "idw_interpolation":
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

    elif entry.id == "nca":
        _excl = {fp.unit_col, fp.time_col}
        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        outcome = cont[0] if cont else None
        if outcome is None or len(cont) < 2:
            summary.append("NCA 失败：需要连续结果变量 + ≥1 个连续条件变量。")
        else:
            import pandas as pd

            predictors = cont[1:6]  # outcome is cont[0]; up to 5 conditions
            sub = df[[outcome] + predictors].dropna()
            if len(sub) < 20:
                summary.append("NCA 失败：有效样本不足（<20）。")
            else:
                y = sub[outcome].to_numpy(dtype=float)

                def _bucket(dv: float) -> str:
                    if dv < 0.1:
                        return "negligible"
                    if dv < 0.3:
                        return "medium"
                    if dv < 0.5:
                        return "large"
                    return "very large"

                ceilings = {}
                rows = []
                for p in predictors:
                    x = sub[p].to_numpy(dtype=float)
                    dv, xs, cmax = _nca_ceiling(x, y)  # `d` is the output dir — don't shadow
                    ceilings[p] = (xs, cmax, dv)
                    rows.append((p, round(dv, 4), _bucket(dv)))
                    estimates[p] = round(dv, 4)

                tab = pd.DataFrame(rows, columns=["condition", "effect_size_d", "necessity"])
                tab.to_csv(d / "nca_effect_sizes.csv", index=False, encoding="utf-8")
                files.append("nca_effect_sizes.csv")
                _nca_plot(sub, outcome, predictors, ceilings, d / "nca_ceiling.png")
                if (d / "nca_ceiling.png").exists():
                    files.append("nca_ceiling.png")

                strong = [r for r in rows if r[1] >= 0.1]
                top = max(rows, key=lambda r: r[1])
                summary.append(
                    f"{entry.method} 完成：结果 {outcome}，{len(predictors)} 个条件；"
                    f"最强必要条件 {top[0]}（d={top[1]}，{top[2]}）；"
                    f"{len(strong)} 个达到有意义阈值 d≥0.1（d=空白区/总域面积，CE-FDH 天花板）"
                )
                code += [
                    "import numpy as np  # NCA (Dul 2016), CE-FDH ceiling",
                    "# c(x)=max{y: x_i<=x}; d = empty_zone_area / scope_area per condition",
                ]

    elif entry.id == "fsqca":
        import re

        from researchforge.executor import rbridge

        _excl = {fp.unit_col, fp.time_col}
        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        outcome = cont[0] if cont else None
        conditions = cont[1:6]  # outcome=cont[0]; up to 5 conditions (truth table 2^k)
        names_safe = outcome is not None and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
        )
        if outcome is None or len(conditions) < 2:
            summary.append("fsQCA 失败：需要 1 个结果变量 + ≥2 个条件变量（均连续）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("QCA")):
            summary.append(
                "fsQCA 需要 R 的 QCA 包（未检测到）。安装：在 R 里 install.packages('QCA')；"
                "或先用 NCA（必要条件分析，纯 Python，无需 R）。"
            )
        elif not names_safe:
            summary.append("fsQCA 失败：列名需为标识符式（字母/数字/. _），R 后端要求。")
        else:
            sub = df[[outcome, *conditions]].dropna()
            csv = d / "_qca_input.csv"
            sub.to_csv(csv, index=False)
            try:
                sol_str, tab = _fsqca_via_r(csv, outcome, conditions)
                tab.to_csv(d / "fsqca_solution.csv", index=False, encoding="utf-8")
                files.append("fsqca_solution.csv")
                (d / "solution.txt").write_text(
                    f"充分性解（complex solution，sufficient configurations） → {outcome}:\n"
                    f"  {sol_str}\n\n"
                    "直接校准(百分位锚点 0.1/0.5/0.9)，incl.cut=0.8；* = 逻辑与（AND）, + = 或（OR）\n"
                    "说明：① 这是 complex 解（不纳入反事实/remainders，最保守）；"
                    "② crossover 锚点取中位数是机械设定，偏态数据会失真，请按理论设锚点；"
                    "③ fsQCA 显示集合关系上的充分性，不等于因果证明。\n\n"
                    + tab.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("solution.txt")
                estimates["n_configurations"] = float(len(tab))
                estimates["min_consistency"] = round(float(tab["consistency"].min()), 4)
                estimates["total_unique_coverage"] = round(float(tab["unique_coverage"].sum()), 4)
                summary.append(
                    f"{entry.method} 完成（R/QCA，complex 解）：充分配置 [{sol_str}] → {outcome}；"
                    f"{len(tab)} 个配置，一致性 {tab['consistency'].min():.3f}–{tab['consistency'].max():.3f}"
                    "（* =AND, + =OR；自动百分位校准+中位 crossover 为机械起点，请按理论设锚点；"
                    "充分性≠因果证明）"
                )
                code += [
                    "library(QCA)  # 直接校准 -> 真值表 -> 布尔最小化",
                    f'# calibrate({[outcome, *conditions]}); truthTable(outcome="{outcome}"); minimize(incl.cut=0.8)',
                ]
            except Exception as err:
                summary.append(f"fsQCA 失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "qca_necessity":
        import re

        from researchforge.executor import rbridge

        _excl = {fp.unit_col, fp.time_col}
        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        outcome = cont[0] if cont else None
        conditions = cont[1:6]
        names_safe = outcome is not None and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
        )
        if outcome is None or len(conditions) < 2:
            summary.append("QCA 必要性分析失败：需要 1 个结果变量 + ≥2 个条件变量（均连续）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("QCA")):
            summary.append(
                "QCA 必要性分析需要 R 的 QCA 包（未检测到）。安装：install.packages('QCA')；"
                "或先用 NCA（必要条件分析，纯 Python，无需 R）。"
            )
        elif not names_safe:
            summary.append("QCA 必要性分析失败：列名需为标识符式（字母/数字/. _）。")
        else:
            sub = df[[outcome, *conditions]].dropna()
            csv = d / "_qca_input.csv"
            sub.to_csv(csv, index=False)
            try:
                tab = _qca_necessity_via_r(csv, outcome, conditions)
                tab = tab.sort_values("consistency_inclN", ascending=False).reset_index(drop=True)
                tab.to_csv(d / "necessity.csv", index=False, encoding="utf-8")
                files.append("necessity.csv")
                (d / "necessity.txt").write_text(
                    f"必要性分析（superSubset） → {outcome}（fuzzy 校准，incl.cut=0.9）:\n"
                    "inclN=必要性一致性；RoN=必要性相关度(越高越非琐碎)；covN=覆盖度；"
                    "~X=非 X，+ =或。\n注意：inclN 高但 RoN 低 = 琐碎必要（条件几乎恒为高）；"
                    "必要性≠因果。\n\n" + tab.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("necessity.txt")
                top = tab.iloc[0]
                estimates["max_inclN"] = round(float(tab["consistency_inclN"].max()), 4)
                estimates["n_necessary_expr"] = float(len(tab))
                summary.append(
                    f"{entry.method} 完成（R/QCA）：最强必要项 [{top['expression']}]"
                    f"（inclN={top['consistency_inclN']:.3f}, RoN={top['RoN']:.3f}, "
                    f"covN={top['coverage_covN']:.3f}）；共 {len(tab)} 项"
                    "（RoN 低=琐碎必要；必要性≠因果证明）"
                )
                code += [
                    "library(QCA)  # 必要性: 模糊校准 -> superSubset(incl.cut=0.9, cov.cut=0.5)",
                    f'# superSubset(cal, outcome="{outcome}", conditions={conditions})',
                ]
            except Exception as err:
                summary.append(f"QCA 必要性分析失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "rarefaction":
        import numpy as np
        from scipy.special import gammaln

        species = [
            c.name
            for c in fp.columns
            if c.kind == "count" and c.name not in {fp.unit_col, fp.time_col}
        ]
        if len(species) < 2:
            summary.append("稀释曲线跳过：需要 ≥2 个计数列（物种/OTU 丰度）。")
        else:
            mat = df[species].fillna(0).clip(lower=0).to_numpy(dtype=float)
            site_N = mat.sum(axis=1)
            keep = site_N > 0
            mat, site_N = mat[keep], site_N[keep]
            n_sites = len(mat)
            if n_sites == 0:
                summary.append("稀释曲线失败：所有样点总丰度为 0。")
            else:

                def hurlbert(counts: np.ndarray, m: int) -> float:
                    # E[S_m] = Σ_i [1 - C(N-N_i, m)/C(N, m)]  (Hurlbert 1971, analytic)
                    counts = counts[counts > 0]
                    total = counts.sum()
                    if m >= total:
                        return float(len(counts))  # full depth -> observed richness
                    log_cnm = gammaln(total + 1) - gammaln(m + 1) - gammaln(total - m + 1)
                    valid = (total - counts) >= m  # else C(N-N_i,m)=0 -> term contributes 1
                    out = float((~valid).sum())
                    cv = counts[valid]
                    if len(cv):
                        lt = gammaln(total - cv + 1) - gammaln(m + 1) - gammaln(total - cv - m + 1) - log_cnm
                        out += float((1.0 - np.exp(lt)).sum())
                    return out

                max_n = int(site_N.max())
                grid = sorted(set(int(round(g)) for g in np.linspace(1, max_n, min(30, max_n))))
                rows_out = []
                richness = []
                for s in range(n_sites):
                    counts = mat[s]
                    n_s = int(site_N[s])
                    richness.append(float((counts > 0).sum()))
                    for m in grid:
                        if m <= n_s:
                            rows_out.append((s, m, round(hurlbert(counts, m), 4)))
                import pandas as pd

                tab = pd.DataFrame(rows_out, columns=["site", "depth", "expected_richness"])
                tab.to_csv(d / "rarefaction.csv", index=False, encoding="utf-8")
                files.append("rarefaction.csv")

                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 4))
                    shown = min(n_sites, 25)  # keep the legend/figure readable
                    for s in range(shown):
                        sub = tab[tab["site"] == s]
                        ax.plot(sub["depth"], sub["expected_richness"], lw=1.2, alpha=0.8)
                    ax.set_xlabel("sampling depth (individuals)")
                    ax.set_ylabel("expected richness E[S]")
                    ax.set_title(f"Rarefaction curves ({shown} of {n_sites} sites)")
                    fig.tight_layout()
                    fig.savefig(d / "rarefaction_curves.png", dpi=150)
                    plt.close(fig)
                    files.append("rarefaction_curves.png")
                except Exception:
                    pass

                estimates["min_depth"] = float(int(site_N.min()))
                estimates["mean_observed_richness"] = round(float(np.mean(richness)), 2)
                estimates["n_sites"] = float(n_sites)
                summary.append(
                    f"{entry.method} 完成：{n_sites} 个样点 × {len(species)} 个物种，"
                    f"平均观测丰度 {np.mean(richness):.1f}，最浅样点深度 {int(site_N.min())}"
                    "（曲线趋平=采样充分；仍上升=需加深采样）"
                )
                code += [
                    "import numpy as np  # Hurlbert (1971) analytic rarefaction",
                    "# E[S_m] = sum_i (1 - comb(N-N_i, m)/comb(N, m)), per site over depth grid",
                ]

    elif entry.id == "moran_i":
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
                k = min(8, n - 1)
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

    elif entry.id == "local_moran":
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
                k = min(8, n - 2)
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

    elif entry.id == "spatial_regression":
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
                diag, pref, coef = _spatial_reg_via_r(csv, outcome, predictors, lon, lat)
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

    elif entry.id == "getis_ord_gi":
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
                k = min(8, n - 2)
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

    elif entry.id == "sem":
        _excl = {fp.unit_col, fp.time_col}
        indicators = [
            c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl
        ]
        if len(indicators) < 3:
            summary.append("SEM 失败：需要 ≥3 个连续指标变量（单因子模型识别要求）。")
        else:
            import re

            import pandas as pd

            from researchforge.executor import rbridge

            inds = indicators[:8]
            sub = df[inds].dropna()
            spec = "F =~ " + " + ".join(inds)
            # prefer lavaan (R, gold standard — also gives SRMR) when available;
            # fall back to pure-Python semopy so the analysis runs anywhere.
            # Only use the R backend with identifier-safe column names: names go
            # into the R model string, so a name with quotes/commas could break
            # parsing or inject R — semopy takes the names as data, no eval.
            names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in inds)
            result = None
            if names_safe and rbridge.r_available() and rbridge.r_package_available("lavaan"):
                csv = d / "_sem_input.csv"
                sub.to_csv(csv, index=False)
                try:
                    result = _sem_via_lavaan(csv, spec)
                except Exception:
                    result = None
                finally:
                    try:
                        csv.unlink()
                    except OSError:
                        pass
            if result is None:
                try:
                    result = _sem_via_semopy(sub, spec)
                except Exception as err:
                    summary.append(f"SEM 拟合失败：{err}")
            if result is not None:
                load = result["loadings"]
                fit = result["fit"]
                (d / "summary.txt").write_text(result["summary"], encoding="utf-8")
                files.append("summary.txt")
                load.to_csv(d / "loadings.csv", index=False, encoding="utf-8")
                files.append("loadings.csv")
                pd.DataFrame([fit]).to_csv(d / "fit_indices.csv", index=False, encoding="utf-8")
                files.append("fit_indices.csv")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 3.2))
                    ax.barh(load["indicator"], load["std_loading"], color="#4C72B0")
                    ax.set_xlabel("standardised loading on F")
                    ax.set_title("SEM single-factor loadings")
                    fig.tight_layout()
                    fig.savefig(d / "loadings.png", dpi=150)
                    plt.close(fig)
                    files.append("loadings.png")
                except Exception:
                    pass
                cfi, tli, rmsea = fit["cfi"], fit["tli"], fit["rmsea"]
                chi2, dof, srmr = fit["chi2"], fit["dof"], fit.get("srmr", float("nan"))
                for kk, vv in (("cfi", cfi), ("tli", tli), ("rmsea", rmsea), ("chi2", chi2), ("dof", dof)):
                    estimates[kk] = round(vv, 4)
                if dof <= 0:
                    # 3 indicators -> just-identified (df=0): CFI/RMSEA perfect by
                    # construction, say nothing about fit (Opus double-review catch).
                    verdict = "恰好识别(df=0)，拟合指数无意义(CFI/RMSEA 必完美)；需 ≥4 指标才能评估拟合"
                elif cfi >= 0.95 and rmsea <= 0.06:
                    verdict = "拟合良好"
                else:
                    verdict = "拟合一般/欠佳"
                srmr_txt = f" SRMR={srmr:.3f}" if srmr == srmr else ""  # NaN-safe
                summary.append(
                    f"{entry.method} 完成（后端：{result['backend']}）：单因子 CFA over "
                    f"{len(inds)} 个指标（df={dof:.0f}）；CFI={cfi:.3f} TLI={tli:.3f} "
                    f"RMSEA={rmsea:.3f}{srmr_txt} → {verdict}"
                    "（此为探索性模板；请按你的理论结构改写 model spec 后重跑）"
                )
                code += [
                    "# SEM single-factor CFA — prefers R/lavaan, falls back to semopy",
                    f'spec = "{spec}"',
                    "# lavaan: cfa(spec, data=df, std.lv=TRUE); semopy: semopy.Model(spec).fit(df)",
                ]

    elif entry.id == "iv_regression":
        summary.append(
            "工具变量回归（2SLS）需要你指定外生工具变量（instrument），引擎无法自动识别。"
            "请在指定工具变量后手动运行；或先用 panel_fixed_effects / did 作为可自动执行的替代。"
        )

    else:
        summary.append(f"{entry.method} 暂未接入执行器（需补依赖/封装），仅生成占位报告。")

    (d / "analysis_code.py").write_text("\n".join(code), encoding="utf-8")
    files.append("analysis_code.py")

    (d / "report.md").write_text(_report(entry, fp, summary, files, override), encoding="utf-8")
    files.append("report.md")

    return RunResult(
        analysis_id=entry.id,
        method=entry.method,
        output_dir=str(d),
        files=files,
        report_path=str(d / "report.md"),
        summary="\n".join(summary),
        estimates=estimates,
    )
