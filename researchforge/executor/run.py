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


def _regression(df, fp: DataFingerprint, entry: AnalysisEntry, cfg: dict | None = None):
    import statsmodels.formula.api as smf

    cfg = cfg or {}
    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    if not cont:
        raise ValueError("没有连续型因变量，无法回归")
    # user override: config["outcome"] picks the dependent variable; else first continuous
    y = cfg["outcome"] if cfg.get("outcome") in cont else cont[0]
    # optional explicit predictor list via config["predictors"]
    forced_rhs = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y]
    exclude = {y, fp.unit_col, fp.time_col}

    fe_terms: list[str] = []
    if entry.id in {"panel_fixed_effects", "did"} and fp.unit_col and fp.time_col:
        fe_terms = [f"C(Q('{fp.unit_col}'))", f"C(Q('{fp.time_col}'))"]

    if entry.id == "did" and fp.treatment_candidates:
        rhs_vars = _pick_did_treatment(df, fp) or fp.treatment_candidates[:1]
    elif forced_rhs:
        rhs_vars = forced_rhs[:8]
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


def _mcda_inputs(df, fp):
    """Shared MCDA setup: numeric criteria (continuous+count, excl id/unit/time),
    an alternative-label column (first id/categorical) or row index. Returns
    (X matrix, criteria names, alternative labels). Raises if < 2 criteria."""
    _excl = {fp.unit_col, fp.time_col}
    crit = [
        c.name for c in fp.columns if c.kind in {"continuous", "count"} and c.name not in _excl
    ]
    if len(crit) < 2:
        raise ValueError("需要 ≥2 个数值型评价指标")
    label_col = next(
        (c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.name not in _excl),
        None,
    )
    cols = crit + ([label_col] if label_col else [])
    sub = df[cols].dropna()
    X = sub[crit].to_numpy(dtype=float)
    labels = (
        sub[label_col].astype(str).tolist()
        if label_col
        else [f"row{i + 1}" for i in range(len(X))]
    )
    return X, crit, labels


def _dea_cross(eval_in, eval_out, ref_in, ref_out, vrs: bool = False):
    """Input-oriented DEA: score each EVAL DMU against the frontier spanned by
    the REF DMUs (envelopment form, λ over ref). θ may exceed 1 for a cross-period
    eval beyond the ref frontier — needed for Malmquist distance functions. CRS
    (vrs=False) or VRS (vrs=True, adds Σλ=1)."""
    import numpy as np
    from scipy.optimize import linprog

    eval_in = np.asarray(eval_in, dtype=float)
    eval_out = np.asarray(eval_out, dtype=float)
    ref_in = np.asarray(ref_in, dtype=float)
    ref_out = np.asarray(ref_out, dtype=float)
    n_ref = ref_in.shape[0]
    eff = np.full(eval_in.shape[0], np.nan)
    for o in range(eval_in.shape[0]):
        c = np.zeros(n_ref + 1)
        c[0] = 1.0  # minimise θ; vars z = [θ, λ_1..λ_n_ref]
        a_ub, b_ub = [], []
        for i in range(ref_in.shape[1]):  # Σ_j λ_j x^ref_ij - θ x^eval_io ≤ 0
            row = np.zeros(n_ref + 1)
            row[0] = -eval_in[o, i]
            row[1:] = ref_in[:, i]
            a_ub.append(row)
            b_ub.append(0.0)
        for r in range(ref_out.shape[1]):  # -Σ_j λ_j y^ref_rj ≤ -y^eval_ro
            row = np.zeros(n_ref + 1)
            row[1:] = -ref_out[:, r]
            a_ub.append(row)
            b_ub.append(-eval_out[o, r])
        a_eq = b_eq = None
        if vrs:
            row = np.zeros(n_ref + 1)
            row[1:] = 1.0
            a_eq, b_eq = [row], [1.0]
        res = linprog(
            c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), A_eq=a_eq, b_eq=b_eq,
            bounds=[(0, None)] * (n_ref + 1), method="highs",
        )
        if res.success:
            eff[o] = res.fun
    return eff


def _dea_efficiency(inputs, outputs, vrs: bool = False):
    """Input-oriented DEA efficiency per DMU vs the same-sample frontier (θ∈(0,1],
    1 = efficient). CCR if vrs=False, BCC if vrs=True."""
    return _dea_cross(inputs, outputs, inputs, outputs, vrs=vrs)


def _mcda_rank_plot(res, score_col: str, title: str, path: Path) -> None:
    """Shared horizontal bar chart of the top-20 ranked alternatives for MCDA."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = res.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.32)))
        ax.barh(top["alternative"].astype(str), top[score_col], color="#4C72B0")
        ax.set_xlabel(score_col)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _minmax01(X, cost=None):
    """Min-max normalise each column to [0,1]. Benefit direction by default
    ((x-lo)/rng); if `cost` is a boolean mask, those columns use the beneficial
    transform (hi-x)/rng so lower=better maps to higher score (textbook MCDA
    direction handling — keeps all downstream methods benefit-oriented). Constant
    columns -> 0.5 (they get ~zero entropy weight downstream)."""
    import numpy as np

    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = hi - lo
    safe = np.where(rng == 0, 1.0, rng)
    benefit = np.where(rng == 0, 0.5, (X - lo) / safe)
    if cost is None:
        return benefit
    cost = np.asarray(cost, dtype=bool)
    flipped = np.where(rng == 0, 0.5, (hi - X) / safe)
    return np.where(cost, flipped, benefit)


def _cost_mask(crit, cfg):
    """Build a benefit/cost mask aligned with `crit` from cfg['cost_criteria']
    (list of criterion names that are cost-type, lower=better). Returns
    (mask_or_None, recognized_names). None mask -> all benefit (fast path)."""
    import numpy as np

    names = (cfg or {}).get("cost_criteria") or []
    recognized = [c for c in names if c in crit]
    if not recognized:
        return None, []
    return np.array([c in recognized for c in crit], dtype=bool), recognized


def _mcda_direction_note(cost_names) -> str:
    """Disclosure line for MCDA: which criteria were treated as cost-type."""
    if cost_names:
        return f"成本型指标（越小越好，已反向）：{cost_names}；其余按效益型处理。"
    return (
        "⚠ 假定所有指标为效益型（越大越好）；若有成本型指标（越小越好），"
        "用 config={\"cost_criteria\": [\"列名\", ...]} 指定以反向。"
    )


def _io_names(crit, cfg):
    """Resolve (input_names, output_names) for efficiency methods. cfg may specify
    'inputs' and 'outputs' (lists of criterion names, intersected with `crit`);
    otherwise the engine default is first column = output, the rest = inputs."""
    cfg = cfg or {}
    want_in = [c for c in (cfg.get("inputs") or []) if c in crit]
    want_out = [c for c in (cfg.get("outputs") or []) if c in crit]
    if want_in and want_out:
        return want_in, want_out
    return crit[1:], [crit[0]]


def _dea_io(X, crit, cfg):
    """Split the MCDA matrix into (inputs, outputs) for DEA using `_io_names`.
    Returns (inputs_array, outputs_array, input_names, output_names)."""
    in_names, out_names = _io_names(crit, cfg)
    in_idx = [crit.index(c) for c in in_names]
    out_idx = [crit.index(c) for c in out_names]
    return X[:, in_idx], X[:, out_idx], in_names, out_names


def _knn_k(cfg, upper, default=8):
    """Number of k-nearest-neighbour spatial weights. cfg['knn_k'] overrides the
    default, clamped to [1, upper] (upper = n-1 or n-2 per estimator's stability
    constraint). Non-int values fall back to the default."""
    k = (cfg or {}).get("knn_k", default)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = default
    return max(1, min(k, upper))


def _qca_anchors(cfg, default=(0.1, 0.5, 0.9)):
    """Fuzzy-calibration percentile anchors (exclusion, crossover, inclusion).
    cfg['anchors'] overrides; must be 3 strictly increasing values in (0,1)."""
    a = (cfg or {}).get("anchors")
    try:
        a = tuple(float(x) for x in a)
        if len(a) == 3 and 0.0 < a[0] < a[1] < a[2] < 1.0:
            return a
    except (TypeError, ValueError):
        pass
    return default


def _qca_incl_cut(cfg, default):
    """Raw-consistency cut-off for QCA truth-table / superSubset. cfg['incl_cut']
    overrides; must be in (0,1]."""
    v = (cfg or {}).get("incl_cut")
    try:
        v = float(v)
        if 0.0 < v <= 1.0:
            return v
    except (TypeError, ValueError):
        pass
    return default


def _gmm_lags(cfg, default=(2, 4)):
    """GMM instrument lag range (lo, hi) for difference-GMM. cfg['gmm_lags']
    overrides; must satisfy 1 <= lo <= hi (lo>=2 in differences is standard, but
    we allow lo>=1 for predetermined-style instruments)."""
    v = (cfg or {}).get("gmm_lags")
    try:
        lo, hi = int(v[0]), int(v[1])
        if 1 <= lo <= hi:
            return lo, hi
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    return default


def _entropy_weights(Z):
    """Objective entropy weights from a [0,1] benefit matrix Z (m alts × k crit).
    Higher dispersion -> higher weight. Equal weights if degenerate."""
    import numpy as np

    m = Z.shape[0]
    if m < 2:
        return np.ones(Z.shape[1]) / Z.shape[1]
    col_sum = np.where(Z.sum(axis=0) == 0, 1.0, Z.sum(axis=0))
    P = Z / col_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        plnp = np.where(P > 0, P * np.log(P), 0.0)
    e = -plnp.sum(axis=0) / np.log(m)
    diff = 1.0 - e
    return diff / diff.sum() if diff.sum() > 0 else np.ones(Z.shape[1]) / Z.shape[1]


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
        'for (i in seq_len(nrow(ld))) cat(sprintf("%s,%s,%.6f,%.6f,%.6f\\n", '
        "ld$rhs[i], ld$lhs[i], ld$est.std[i], ifelse(is.na(ld$se[i]),0,ld$se[i]), "
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
        [r.split(",") for r in lrows],
        columns=["indicator", "factor", "std_loading", "std_err", "p_value"],
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


def _sem_latents(spec: str) -> list[str]:
    """Latent-variable names = the LHS of every `=~` measurement line in a
    lavaan/semopy model spec. Used to pick out measurement loadings generically."""
    import re

    return [m.group(1) for m in re.finditer(r"([A-Za-z_]\w*)\s*=~", spec)]


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


def _csqca_via_r(csv_path, outcome: str, conditions: list[str], incl_cut=0.8):
    """Crisp-set QCA via R QCA: dichotomise (binary kept; continuous cut at the
    median) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report honestly."""
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
        "  nn <- x[!is.na(x)]; u <- unique(nn)\n"
        "  if (length(u) <= 2 && all(u %in% c(0,1))) return(as.integer(x))\n"
        "  as.integer(x > median(nn))\n"  # crisp threshold at the median
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), incl.cut={float(incl_cut)}, show.cases=FALSE)\n'
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
        raise RuntimeError("QCA 未返回充分配置（无解 / 有限多样性）")
    tab = pd.DataFrame(rows, columns=["configuration", "consistency", "raw_coverage", "unique_coverage"])
    for c in ("consistency", "raw_coverage", "unique_coverage"):
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return sol_str, tab


def _fsqca_via_r(csv_path, outcome: str, conditions: list[str], anchors=(0.1, 0.5, 0.9), incl_cut=0.8):
    """Run fsQCA with R's QCA package: direct fuzzy calibration (percentile
    anchors) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report an honest message."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        # skewed/discrete columns can tie the anchor quantiles; calibrate
        # needs strictly increasing anchors, so nudge ties by a tiny epsilon.
        "calib <- function(x) {\n"
        f"  thr <- as.numeric(quantile(x, c({a0},{a1},{a2}), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib))\n"
        "names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), '
        f"incl.cut={float(incl_cut)}, show.cases=FALSE)\n"
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


def _kriging_via_r(csv_path, lon: str, lat: str, value: str):
    """Ordinary kriging via R gstat: auto-fit a variogram (Sph/Exp/Gau), LOO
    cross-validation RMSE, and predict a 40x40 grid with kriging variance.
    Returns (meta dict, grid DataFrame[lon,lat,pred,var]). Raises on failure."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rcode = (
        "suppressMessages({library(gstat); library(sp)})\n"
        f'd <- read.csv("{csv_r}")\n'
        f"coordinates(d) <- ~{lon}+{lat}\n"
        f"v <- variogram({value} ~ 1, d)\n"
        # explicit initial psill/range/nugget so fit.variogram doesn't silently
        # return a poor/zero-nugget fit from bare defaults (Opus catch).
        f'vy <- var(d@data[["{value}"]], na.rm=TRUE)\n'
        # Spherical/Exponential only — the Gaussian model gives near-singular
        # kriging matrices and unstable (blown-up) predictions (Opus catch via test).
        'vm <- fit.variogram(v, vgm(psill=0.9*vy, model=c("Sph","Exp"), '
        "range=max(v$dist)/2, nugget=0.1*vy))\n"
        f"cv <- krige.cv({value} ~ 1, d, model=vm, nfold=nrow(d@data), verbose=FALSE)\n"
        "rmse <- sqrt(mean(cv$residual^2, na.rm=TRUE))\n"
        "bb <- d@bbox\n"
        f"g <- expand.grid({lon}=seq(bb[1,1],bb[1,2],length.out=40), "
        f"{lat}=seq(bb[2,1],bb[2,2],length.out=40)); coordinates(g) <- ~{lon}+{lat}\n"
        f"k <- krige({value} ~ 1, d, g, model=vm, debug.level=0)\n"
        'cat("##META\\n")\n'
        "n <- nrow(vm)\n"
        'cat(sprintf("model|%s\\nrange|%.5f\\nloo_rmse|%.5f\\n", as.character(vm$model[n]), vm$range[n], rmse))\n'
        'cat("##GRID\\n"); gd <- as.data.frame(k)\n'
        f'for (i in seq_len(nrow(gd))) cat(sprintf("%.5f|%.5f|%.5f|%.5f\\n", gd${lon}[i], gd${lat}[i], gd$var1.pred[i], gd$var1.var[i]))\n'
    )
    out = rbridge.run_r(rcode, timeout=240)
    section, meta, rows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##META":
            section = "M"
        elif s == "##GRID":
            section = "G"
        elif section == "M" and "|" in s:
            kk, vv = s.split("|", 1)
            meta[kk] = vv if kk == "model" else float(vv)
        elif section == "G" and "|" in s:
            rows.append(s.split("|"))
    if not rows or "loo_rmse" not in meta:
        raise RuntimeError("gstat kriging 未返回结果（变异函数拟合可能失败）")
    grid = pd.DataFrame(rows, columns=[lon, lat, "prediction", "kriging_variance"])
    for c in (lon, lat, "prediction", "kriging_variance"):
        grid[c] = pd.to_numeric(grid[c], errors="coerce")
    return meta, grid


def _spatial_reg_via_r(csv_path, outcome: str, predictors: list[str], lon: str, lat: str, k: int = 6):
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
        f"lw <- nb2listw(knn2nb(knearneigh(coords, k=min({int(k)}, nrow(d)-1))), style=\"W\")\n"
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


def _dynamic_gmm_via_r(
    csv_path, unit: str, time: str, y: str, predictors: list[str],
    endogenous=None, gmm_lags=(2, 4),
):
    """Arellano-Bond difference-GMM dynamic panel via R plm::pgmm. Returns
    (coef DataFrame [term, estimate, std_err, p_value], diagnostics dict with
    sargan_stat/p and ar1_p/ar2_p). Raises so the caller can degrade honestly.

    By default every covariate is treated as strictly exogenous and only the
    lagged dependent is GMM-instrumented (lags `gmm_lags`). If `endogenous` names
    covariates, those join the GMM-instrument block (instrumented with their own
    lags `gmm_lags`, the Arellano-Bond treatment for endogenous regressors) and a
    third "normal instruments" formula part lists the remaining exogenous
    covariates as their own instruments."""
    import pandas as pd

    from researchforge.executor import rbridge

    endo = [p for p in (endogenous or []) if p in predictors]
    exog = [p for p in predictors if p not in endo]
    lo, hi = int(gmm_lags[0]), int(gmm_lags[1])
    # The lagged dependent must use lag>=2 in the differenced equation (y_{t-1}
    # correlates with Δε_t, so lag 1 is an INVALID instrument; Opus catch). Floor
    # its block at 2 even if the user lowers `lo` for predetermined covariates.
    y_lo = max(2, lo)
    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join([f"lag({y}, 1)", *predictors])
    # cap GMM instruments at lags lo:hi (not 2:99) to curb instrument proliferation,
    # which otherwise inflates/weakens the Sargan test (Roodman; Opus catch).
    gmm_inst = " + ".join([f"lag({y}, {y_lo}:{hi})", *(f"lag({p}, {lo}:{hi})" for p in endo)])
    if endo:
        # 3-part pgmm formula is REQUIRED here: pgmm auto-instruments omitted
        # covariates as their own (normal) instruments, which would wrongly treat an
        # endogenous regressor as exogenous. Listing only `exog` in part 3 (verified:
        # 2-part≡3-part for purely-exogenous models) confines self-instrumentation to
        # the exogenous set; if all covariates are endogenous, no part 3 is needed.
        formula = f"{y} ~ {rhs} | {gmm_inst}" + (f" | {' + '.join(exog)}" if exog else "")
    else:
        formula = f"{y} ~ {rhs} | {gmm_inst}"  # default: pgmm self-instruments all exog
    rcode = (
        "suppressMessages(library(plm))\n"
        f'd <- read.csv("{csv_r}")\n'
        f'pd <- pdata.frame(d, index=c("{unit}","{time}"))\n'
        f'm <- pgmm({formula}, data=pd, effect="individual", model="twosteps")\n'
        "s <- summary(m, robust=TRUE); ct <- s$coefficients\n"
        'cat("##COEF\\n")\n'
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,4]))\n'
        "a1 <- mtest(m, order=1); a2 <- mtest(m, order=2)\n"
        'cat("##DIAG\\n")\n'
        'cat(sprintf("sargan_stat|%.6f\\nsargan_p|%.6f\\n", s$sargan$statistic, s$sargan$p.value))\n'
        'cat(sprintf("ar1_p|%.6f\\nar2_p|%.6f\\n", a1$p.value, a2$p.value))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, crows, diag = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##DIAG":
            section = "D"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))  # term may contain no |; split last 3 fields
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            diag[k] = float(v)
    if not crows:
        raise RuntimeError("pgmm 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return coef, diag


def _diff_abundance_aldex2_via_r(csv_path, taxa: list[str], group: str):
    """Differential abundance via R ALDEx2 (compositional gold standard): CLR with
    Monte-Carlo Dirichlet sampling of the counts, Welch t per taxon over the MC
    instances, BH-FDR. ALDEx2 expects features (taxa) as ROWS, samples as COLUMNS.
    Returns a DataFrame [taxon, effect, diff_btw, p_value, q_value]. Raises so the
    caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    taxa_r = ", ".join(f'"{t}"' for t in taxa)
    rcode = (
        "suppressMessages(library(ALDEx2))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"taxa <- c({taxa_r})\n"
        "counts <- t(as.matrix(d[, taxa]))\n"  # features x samples
        f'conds <- as.character(d[["{group}"]])\n'
        # 128 MC Dirichlet instances (ALDEx2 default); Welch t with effect sizes
        'x <- aldex(round(counts), conds, mc.samples=128, test="t", effect=TRUE, denom="all", verbose=FALSE)\n'
        'cat("##R\\n")\n'
        'for (i in seq_len(nrow(x))) cat(sprintf("%s|%.6f|%.6f|%.6g|%.6g\\n", '
        "rownames(x)[i], x$effect[i], x$diff.btw[i], x$we.ep[i], x$we.eBH[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=300)
    rows = []
    section = None
    for line in out.splitlines():
        s = line.strip()
        if s == "##R":
            section = "R"
        elif section == "R" and "|" in s:
            rows.append(s.split("|"))
    if not rows:
        raise RuntimeError("ALDEx2 未返回结果")
    res = pd.DataFrame(rows, columns=["taxon", "effect", "diff_btw", "p_value", "q_value"])
    for c in ("effect", "diff_btw", "p_value", "q_value"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


def _meta_via_r(csv_path, *, measure, roles, study_col, method, forest_png, funnel_png):
    """Random/fixed-effects meta-analysis via R metafor. `measure` is "GEN"
    (pre-computed effect sizes yi + vi/sei) or an escalc measure ("SMD"/"MD"/
    "OR"/"RR"/"RD") computed from raw study columns. Column names are passed as
    d[["name"]] vectors (no R symbol eval → injection-safe). Writes forest + funnel
    PNGs. Returns (meta dict, per-study DataFrame). Raises so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    fp_r = str(forest_png).replace("\\", "/")
    fn_r = str(funnel_png).replace("\\", "/")
    slab = f'd[["{study_col}"]]' if study_col else "NULL"
    if measure == "GEN":
        vi = f'd[["{roles["vi"]}"]]' if roles.get("vi") else f'(d[["{roles["sei"]}"]])^2'
        esc = f'yi <- d[["{roles["yi"]}"]]; vi <- {vi}\n'
    elif measure in ("SMD", "MD"):
        esc = (
            f'es <- escalc(measure="{measure}", m1i=d[["{roles["m1"]}"]], '
            f'sd1i=d[["{roles["sd1"]}"]], n1i=d[["{roles["n1"]}"]], '
            f'm2i=d[["{roles["m2"]}"]], sd2i=d[["{roles["sd2"]}"]], n2i=d[["{roles["n2"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    else:  # OR / RR / RD from a 2x2 table
        esc = (
            f'es <- escalc(measure="{measure}", ai=d[["{roles["ai"]}"]], '
            f'bi=d[["{roles["bi"]}"]], ci=d[["{roles["ci"]}"]], di=d[["{roles["di"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    rcode = (
        "suppressMessages(library(metafor))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        + esc
        + f'm <- rma(yi, vi, method="{method}", slab={slab})\n'
        'cat("##M\\n")\n'
        'cat(sprintf("estimate|%.6f\\nci_lb|%.6f\\nci_ub|%.6f\\npval|%.6g\\n", '
        "m$b[1], m$ci.lb, m$ci.ub, m$pval))\n"
        'cat(sprintf("Q|%.6f\\nQp|%.6g\\nI2|%.4f\\ntau2|%.6f\\nk|%d\\n", '
        "m$QE, m$QEp, m$I2, m$tau2, m$k))\n"
        # Egger's regression test for funnel asymmetry (needs k>=3); guard failures
        "eg <- tryCatch(regtest(m), error=function(e) NULL)\n"
        'if (!is.null(eg)) cat(sprintf("egger_z|%.4f\\negger_p|%.6g\\n", eg$zval, eg$pval))\n'
        'cat("##S\\n")\n'
        "sl <- if (is.null(m$slab)) seq_along(yi) else as.character(m$slab)\n"
        'for (i in seq_along(yi)) cat(sprintf("%s|%.6f|%.6f\\n", sl[i], yi[i], vi[i]))\n'
        f'png("{fp_r}", width=920, height=max(380, 70+34*length(yi)), res=120); '
        "forest(m); dev.off()\n"
        f'png("{fn_r}", width=720, height=620, res=120); funnel(m); dev.off()\n'
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, meta, srows = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##M":
            section = "M"
        elif s == "##S":
            section = "S"
        elif "|" in s and section == "M":
            k, v = s.split("|", 1)
            meta[k] = float(v)
        elif "|" in s and section == "S":
            srows.append(s.rsplit("|", 2))
    if "estimate" not in meta or not srows:
        raise RuntimeError("metafor 未返回结果（检查效应量/方差列）")
    study = pd.DataFrame(srows, columns=["study", "yi", "vi"])
    for c in ("yi", "vi"):
        study[c] = pd.to_numeric(study[c], errors="coerce")
    return meta, study


def _panel_qca_via_r(csv_path, outcome, conditions, unit, anchors, incl_cut):
    """Panel/clustered fsQCA via R SetMethods::cluster (Garcia-Castro & Ariño):
    fuzzy-calibrate, derive the pooled fsQCA solution, then decompose each
    solution term's consistency into POOLED vs BETWEEN-unit vs WITHIN-unit, with
    the between→pooled (dBP) and within→pooled (dWP) distances. A large distance
    means the configuration does NOT hold uniformly across units / over time
    (clustered heterogeneity). Returns (solution_str, terms_df). Raises on no
    result. Column names go through an identifier guard upstream."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages({library(SetMethods); library(QCA)})\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x){ thr<-as.numeric(quantile(x,c("
        f"{a0},{a1},{a2}),na.rm=TRUE)); rng<-diff(range(x,na.rm=TRUE)); "
        "eps<-if(rng>0) rng*1e-6 else 1e-6; if(thr[2]<=thr[1]) thr[2]<-thr[1]+eps; "
        'if(thr[3]<=thr[2]) thr[3]<-thr[2]+eps; calibrate(x,type="fuzzy",thresholds=thr) }\n'
        "cal <- as.data.frame(lapply(d[,cols], calib)); names(cal) <- cols\n"
        f'cal[["UNIT"]] <- as.character(d[["{unit}"]])\n'
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), incl.cut={float(incl_cut)})\n'
        "sol <- minimize(tt, details=TRUE)\n"
        f'cl <- cluster(data=cal, results=sol, outcome="{outcome}", unit_id="UNIT", cluster_id="UNIT")\n'
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##TERMS\\n")\n'
        "for (t in names(cl$output)) { o <- cl$output[[t]]; "
        'cat(sprintf("%s|%.4f|%.4f|%.4f|%.4f\\n", t, o$POCOS, o$Coverages$pooled, o$dBP, o$dWP)) }\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##TERMS":
            section = "T"
        elif section == "S" and s:
            sol_str = s
        elif "|" in s and section == "T":
            rows.append(s.rsplit("|", 4))
    if not rows:
        raise RuntimeError("SetMethods cluster 未返回分解（可能无解/单解项/有限多样性）")
    terms = pd.DataFrame(
        rows, columns=["term", "pooled_consistency", "pooled_coverage", "dist_between", "dist_within"]
    )
    for c in ("pooled_consistency", "pooled_coverage", "dist_between", "dist_within"):
        terms[c] = pd.to_numeric(terms[c], errors="coerce")
    return sol_str, terms


def _cna_via_r(csv_path, factors, outcome, con, cov, anchors, fuzzy):
    """Coincidence Analysis via R `cna` — finds configurational causal structures
    (Boolean solution formulas) and, unlike QCA, can recover structures with
    MULTIPLE outcomes (it does not require pre-designating one). Crisp (0/1) data
    is used directly (type="cs"); continuous factors are fuzzy-calibrated by
    percentile anchors (type="fs"). Returns (asf_df, n_csf). asf columns:
    outcome, condition, consistency, coverage, complexity. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    facs_r = ", ".join(f'"{c}"' for c in factors)
    out_arg = f', outcome=c("{outcome}")' if outcome else ""
    typ = "fs" if fuzzy else "cs"
    calib = (
        # fuzzy-calibrate non-binary columns to [0,1]; leave 0/1 columns as-is
        "calib <- function(x){ u<-unique(x[!is.na(x)]); "
        "if(length(u)<=2 && all(u %in% c(0,1))) return(x); "
        f"thr<-as.numeric(quantile(x,c({a0},{a1},{a2}),na.rm=TRUE)); "
        "rng<-diff(range(x,na.rm=TRUE)); eps<-if(rng>0) rng*1e-6 else 1e-6; "
        "if(thr[2]<=thr[1]) thr[2]<-thr[1]+eps; if(thr[3]<=thr[2]) thr[3]<-thr[2]+eps; "
        'calibrate(x,type="fuzzy",thresholds=thr) }\n'
        "d <- as.data.frame(lapply(d[,facs], calib)); names(d)<-facs\n"
        if fuzzy
        else "d <- d[, facs]\n"
    )
    rcode = (
        "suppressMessages(library(cna))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"facs <- c({facs_r})\n"
        + calib
        + f'x <- cna(d, type="{typ}", con={float(con)}, cov={float(cov)}{out_arg})\n'
        "a <- asf(x)\n"
        'cat("##ASF\\n")\n'
        "if (nrow(a)) for (i in seq_len(nrow(a))) cat(sprintf('%s|%s|%.4f|%.4f|%d\\n', "
        "a$outcome[i], a$condition[i], a$con[i], a$cov[i], a$complexity[i]))\n"
        'cat("##CSF\\n"); cat(sprintf("n|%d\\n", nrow(csf(x))))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, rows, n_csf = None, [], 0
    for line in out.splitlines():
        s = line.strip()
        if s == "##ASF":
            section = "A"
        elif s == "##CSF":
            section = "C"
        elif "|" in s and section == "A":
            rows.append(s.rsplit("|", 4))
        elif s.startswith("n|") and section == "C":
            n_csf = int(s.split("|", 1)[1])
    if not rows:
        raise RuntimeError("cna 未返回解（可能 con/cov 阈值过高或无配置性结构）")
    asf = pd.DataFrame(rows, columns=["outcome", "condition", "consistency", "coverage", "complexity"])
    for c in ("consistency", "coverage", "complexity"):
        asf[c] = pd.to_numeric(asf[c], errors="coerce")
    return asf, n_csf


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


def _rdd_via_rdrobust(df, outcome, running, cutoff, plot_path):
    """Sharp regression discontinuity (Calonico–Cattaneo–Titiunik) via rdrobust:
    local-linear estimate of the outcome jump at the cutoff, MSE-optimal bandwidth,
    robust bias-corrected inference. Returns a dict; writes a binned RD plot.
    Raises so the caller can degrade honestly."""
    import numpy as np
    from rdrobust import rdrobust

    sub = df[[outcome, running]].copy()
    sub[outcome] = sub[outcome].astype(float)
    sub[running] = sub[running].astype(float)
    sub = sub.dropna()
    y, x = sub[outcome], sub[running]
    r = rdrobust(y=y, x=x, c=float(cutoff))
    out = {
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


def _synthetic_control(df, unit, time, outcome, treated, treat_time, predictors, gaps_png, exclude=None):
    """Synthetic control (Abadie): build a weighted combination of donor (control)
    units that tracks the treated unit's PRE-treatment outcome path, then read the
    post-treatment gap (treated − synthetic) as the treatment effect. `exclude` =
    units to drop from the donor pool besides `treated` (e.g. OTHER ever-treated
    units, whose contamination would bias the counterfactual). Returns
    (weights Series, att dict, pre_rmspe, n_donors, post_periods). Writes a gaps
    plot. Raises so the caller can degrade honestly."""
    import numpy as np
    from pysyncon import Dataprep, Synth

    drop = {treated} | set(exclude or set())
    times = sorted(t for t in df[time].dropna().unique())
    pre = [t for t in times if t < treat_time]
    post = [t for t in times if t >= treat_time]
    controls = [u for u in df[unit].dropna().unique() if u not in drop]
    if len(pre) < 2 or not post or len(controls) < 2:
        raise RuntimeError("合成控制需要 ≥2 个干预前期、≥1 个干预后期、≥2 个对照单位")
    preds = [p for p in predictors if p != outcome] or [outcome]
    dp = Dataprep(
        foo=df,
        predictors=preds,
        predictors_op="mean",
        dependent=outcome,
        unit_variable=unit,
        time_variable=time,
        treatment_identifier=treated,
        controls_identifier=controls,
        time_predictors_prior=pre,
        time_optimize_ssr=pre,
    )
    synth = Synth()
    synth.fit(dp)
    weights = synth.weights().sort_values(ascending=False)
    att = synth.att(time_period=post)
    pre_rmspe = float(np.sqrt(synth.mspe()))  # pre-treatment root mean squared prediction error
    try:
        import warnings

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with warnings.catch_warnings():  # pysyncon calls plt.show() internally (Agg warns)
            warnings.simplefilter("ignore")
            synth.gaps_plot(time_period=times, treatment_time=treat_time)
        fig = plt.gcf()
        fig.set_size_inches(7, 4.5)
        ax = plt.gca()
        ax.set_title(f"Synthetic control gap — treated unit {treated}")
        ax.set_xlabel(str(time))
        ax.set_ylabel(f"{outcome}: treated − synthetic")
        fig.tight_layout()
        fig.savefig(gaps_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return weights, att, pre_rmspe, len(controls), post


def _glmm_via_r(csv_path, outcome: str, predictors: list[str], group: str, family: str):
    """Generalized linear mixed model via R lme4::glmer — a fixed-effect part plus
    a random intercept (1|group), for a binary (family="binomial", logit) or count
    (family="poisson", log) outcome. Returns (fixed_df[term,estimate,se,z,p],
    re dict{group_var, group_sd, icc, n_groups, n_obs, aic, overdispersion}).
    Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(predictors) or "1"
    fam = "binomial" if family == "binomial" else "poisson"
    rcode = (
        "suppressMessages(library(lme4))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'm <- glmer({outcome} ~ {rhs} + (1|{group}), data=d, family={fam})\n'
        "s <- summary(m); ct <- s$coefficients\n"
        'cat("##FX\\n")\n'
        'for (i in seq_len(nrow(ct))) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", '
        "rownames(ct)[i], ct[i,1], ct[i,2], ct[i,4]))\n"
        "vc <- as.data.frame(VarCorr(m)); gv <- vc$vcov[1]\n"
        # latent-scale ICC: binomial logit residual var = pi^2/3; poisson left as NA
        f'icc <- if ("{fam}" == "binomial") gv/(gv + (pi^2)/3) else NA\n'
        "od <- sum(residuals(m, type='pearson')^2) / df.residual(m)\n"
        'cat("##RE\\n")\n'
        'cat(sprintf("group_var|%.6f\\ngroup_sd|%.6f\\n", gv, sqrt(gv)))\n'
        'if (!is.na(icc)) cat(sprintf("icc|%.6f\\n", icc))\n'
        'cat(sprintf("n_groups|%d\\nn_obs|%d\\naic|%.4f\\noverdispersion|%.4f\\n", '
        "as.integer(ngrps(m)[1]), as.integer(nobs(m)), AIC(m), od))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, fx, re_d = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##FX":
            section = "FX"
        elif s == "##RE":
            section = "RE"
        elif "|" in s and section == "FX":
            fx.append(s.rsplit("|", 3))
        elif "|" in s and section == "RE":
            k, v = s.split("|", 1)
            re_d[k] = float(v)
    if not fx or "group_var" not in re_d:
        raise RuntimeError("lme4 glmer 未返回结果（可能不收敛）")
    fixed = pd.DataFrame(fx, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        fixed[c] = pd.to_numeric(fixed[c], errors="coerce")
    return fixed, re_d


def _gam_via_r(csv_path, outcome: str, smooth_terms: list[str], linear_terms: list[str], png_path):
    """Generalized additive model via R mgcv (gold standard): smooth (penalised
    spline) terms s(x) for continuous predictors, parametric terms for the rest,
    fit by REML. Returns (smooth_df[term,edf,F,p], param_df[term,est,se,p],
    fit{dev_expl,r_sq,n}). Writes the partial-effect smooth panels to png_path.
    Raises so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    png_r = str(png_path).replace("\\", "/")
    rhs = " + ".join([f"s({t})" for t in smooth_terms] + list(linear_terms)) or "1"
    n_panels = max(1, len(smooth_terms))
    rcode = (
        "suppressMessages(library(mgcv))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f'm <- gam({outcome} ~ {rhs}, data=d, method="REML")\n'
        "s <- summary(m)\n"
        'cat("##S\\n")\n'
        "if (!is.null(s$s.table)) for (i in seq_len(nrow(s$s.table))) "
        'cat(sprintf("%s|%.4f|%.4f|%.6g\\n", rownames(s$s.table)[i], '
        's$s.table[i,"edf"], s$s.table[i,"F"], s$s.table[i,"p-value"]))\n'
        'cat("##P\\n")\n'
        "if (!is.null(s$p.table)) for (i in seq_len(nrow(s$p.table))) "
        'cat(sprintf("%s|%.6f|%.6f|%.6g\\n", rownames(s$p.table)[i], '
        "s$p.table[i,1], s$p.table[i,2], s$p.table[i,4]))\n"
        'cat("##F\\n")\n'
        'cat(sprintf("dev_expl|%.4f\\nr_sq|%.4f\\nn|%d\\n", s$dev.expl, s$r.sq, s$n))\n'
        # worst-case concurvity (nonlinear analogue of collinearity); deterministic,
        # no refit. >~0.8 means a smooth is well-approximated by the others (Opus).
        'cw <- tryCatch(max(concurvity(m, full=TRUE)["worst", ], na.rm=TRUE), error=function(e) NA)\n'
        'if (!is.na(cw)) cat(sprintf("concurvity|%.4f\\n", cw))\n'
        f'png("{png_r}", width=900, height=max(350, 320*ceiling({n_panels}/2)), res=120)\n'
        f"par(mfrow=c(ceiling({n_panels}/2), min(2,{n_panels})))\n"
        "plot(m, shade=TRUE, seWithMean=TRUE, residuals=FALSE); dev.off()\n"
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, srows, prows, fit = None, [], [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##S":
            section = "S"
        elif s == "##P":
            section = "P"
        elif s == "##F":
            section = "F"
        elif "|" in s and section == "S":
            srows.append(s.rsplit("|", 3))
        elif "|" in s and section == "P":
            prows.append(s.rsplit("|", 3))
        elif "|" in s and section == "F":
            k, v = s.split("|", 1)
            fit[k] = float(v)
    if "dev_expl" not in fit:
        raise RuntimeError("mgcv gam 未返回结果")
    smooth_df = pd.DataFrame(srows, columns=["term", "edf", "F", "p_value"]) if srows else pd.DataFrame(
        columns=["term", "edf", "F", "p_value"]
    )
    param_df = pd.DataFrame(prows, columns=["term", "estimate", "std_err", "p_value"]) if prows else pd.DataFrame(
        columns=["term", "estimate", "std_err", "p_value"]
    )
    for col in ("edf", "F", "p_value"):
        if len(smooth_df):
            smooth_df[col] = pd.to_numeric(smooth_df[col], errors="coerce")
    for col in ("estimate", "std_err", "p_value"):
        if len(param_df):
            param_df[col] = pd.to_numeric(param_df[col], errors="coerce")
    return smooth_df, param_df, fit


def _meta_regression_via_r(csv_path, *, measure, roles, moderators, study, method, bubble_png, bubble_mod=None):
    """Meta-regression via R metafor — rma(yi, vi, mods = ~ moderators): explains
    between-study heterogeneity with study-level moderators. `measure` is "GEN"
    (pre-computed yi + vi/sei) or an escalc measure ("SMD"/"MD"/"OR"/"RR"/"RD").
    Reports the moderator coefficient table, the omnibus moderator test (QM),
    residual heterogeneity (I2/tau2), and R2 (% heterogeneity explained). Writes a
    bubble plot for the first moderator. Returns (coef_df, stats dict). Column
    names go through an identifier guard upstream. Raises so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    bp_r = str(bubble_png).replace("\\", "/")
    if measure == "GEN":
        vi = f'd[["{roles["vi"]}"]]' if roles.get("vi") else f'(d[["{roles["sei"]}"]])^2'
        esc = f'yi <- d[["{roles["yi"]}"]]; vi <- {vi}\n'
    elif measure in ("SMD", "MD"):
        esc = (
            f'es <- escalc(measure="{measure}", m1i=d[["{roles["m1"]}"]], '
            f'sd1i=d[["{roles["sd1"]}"]], n1i=d[["{roles["n1"]}"]], '
            f'm2i=d[["{roles["m2"]}"]], sd2i=d[["{roles["sd2"]}"]], n2i=d[["{roles["n2"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    else:  # OR / RR / RD
        esc = (
            f'es <- escalc(measure="{measure}", ai=d[["{roles["ai"]}"]], '
            f'bi=d[["{roles["bi"]}"]], ci=d[["{roles["ci"]}"]], di=d[["{roles["di"]}"]])\n'
            "yi <- es$yi; vi <- es$vi\n"
        )
    mods_r = "~ " + " + ".join(moderators)
    rcode = (
        "suppressMessages(library(metafor))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        + esc
        + f"m <- rma(yi, vi, mods = {mods_r}, data=d, method=\"{method}\")\n"
        'cat("##C\\n")\n'
        "for (i in seq_len(nrow(m$beta))) cat(sprintf('%s|%.6f|%.6f|%.6g|%.6f|%.6f\\n', "
        "rownames(m$beta)[i], m$beta[i], m$se[i], m$pval[i], m$ci.lb[i], m$ci.ub[i]))\n"
        'cat("##S\\n")\n'
        'cat(sprintf("QM|%.6f\\nQMp|%.6g\\nR2|%.4f\\nI2_resid|%.4f\\ntau2_resid|%.6f\\nk|%d\\n", '
        "m$QM, m$QMp, ifelse(is.null(m$R2),NA,m$R2), m$I2, m$tau2, m$k))\n"
        # also report the surviving moderator terms so the caller can detect drops
        'cat("##T\\n"); for (t in rownames(m$beta)) cat(t, "\\n")\n'
        # bubble plot for a NUMERIC moderator only (regplot needs continuous); guard failures
        + (
            f'tryCatch({{png("{bp_r}", width=760, height=600, res=120); '
            f'regplot(m, mod="{bubble_mod}", shade=TRUE); dev.off()}}, error=function(e) NULL)\n'
            if bubble_mod else ""
        )
    )
    out = rbridge.run_r(rcode, timeout=120)
    section, crows, stats, surviving = None, [], {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##C":
            section = "C"
        elif s == "##S":
            section = "S"
        elif s == "##T":
            section = "T"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 5))
        elif "|" in s and section == "S":
            k, v = s.split("|", 1)
            try:
                stats[k] = float(v)  # NA (constant/collinear moderator) -> NaN, not a crash
            except ValueError:
                stats[k] = float("nan")
        elif section == "T" and s:
            surviving.append(s)
    if not crows or "QM" not in stats:
        raise RuntimeError("metafor meta-regression 未返回结果（检查效应量/调节变量列）")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "se", "p_value", "ci_lb", "ci_ub"])
    for c in ("estimate", "se", "p_value", "ci_lb", "ci_ub"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    stats["_surviving"] = [t for t in surviving if t != "intrcpt"]
    return coef, stats


def _sfa_via_r(csv_path, output: str, inputs: list[str]):
    """Stochastic Frontier Analysis via R frontier: Cobb-Douglas production
    frontier log(y) ~ Σ log(x), ML with composed error v−u. Returns
    (coef dict incl. sigmaSq & gamma, per-row technical-efficiency array)."""
    import numpy as np

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(f"log({c})" for c in inputs)
    rcode = (
        "suppressMessages(library(frontier))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"m <- sfa(log({output}) ~ {rhs}, data=d)\n"
        "co <- coef(m)\n"
        # one-sided LR test of γ=0 (no inefficiency) vs OLS; boundary mixed χ² ⇒ ½ weight
        f"ols <- lm(log({output}) ~ {rhs}, data=d)\n"
        "lr <- max(0, 2*(as.numeric(logLik(m)) - as.numeric(logLik(ols))))\n"
        "pval <- 0.5 * pchisq(lr, df=1, lower.tail=FALSE)\n"
        'cat("##COEF\\n"); for (nm in names(co)) cat(sprintf("%s|%.6f\\n", nm, co[nm]))\n'
        'cat(sprintf("lr_stat|%.6f\\nlr_pvalue|%.6f\\n", lr, pval))\n'
        "eff <- efficiencies(m, asInData=TRUE)\n"
        'cat("##EFF\\n"); for (i in seq_along(eff)) cat(sprintf("%.6f\\n", eff[i]))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, coef, te = None, {}, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##EFF":
            section = "E"
        elif section == "C" and "|" in s:
            k, v = s.split("|", 1)
            coef[k] = float(v)
        elif section == "E" and s and "|" not in s:
            try:
                te.append(float(s))
            except ValueError:
                pass
    if not coef or not te:
        raise RuntimeError("frontier sfa 未返回结果")
    return coef, np.array(te)


def _qca_necessity_via_r(csv_path, outcome: str, conditions: list[str], anchors=(0.1, 0.5, 0.9), incl_cut=0.9):
    """QCA necessity analysis via R superSubset on fuzzy-calibrated data. Returns
    a DataFrame [expression, inclN(consistency), RoN, covN(coverage)]. RoN flags
    trivially-necessary (always-high) conditions. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x) {\n"
        f"  thr <- as.numeric(quantile(x, c({a0},{a1},{a2}), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'ss <- superSubset(cal, outcome="{outcome}", conditions=c({conds_r}), '
        f"incl.cut={float(incl_cut)}, cov.cut=0.5)\n"
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


def _usda_texture(sand: float, silt: float, clay: float) -> str:
    """USDA soil texture class from sand/silt/clay percentages (sum ~100).
    Canonical decision sequence — order matters; verified against reference points."""
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (
        clay < 7 and silt < 50 and silt + 2 * clay >= 30
    ):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    if clay >= 40 and sand <= 45 and silt < 40:
        return "clay"
    return "unclassified"


def _varimax(phi, q: int = 30, tol: float = 1e-6):
    """Kaiser varimax rotation of a loading matrix (items x factors). Returns the
    rotated loadings; identity (no rotation) for a single factor."""
    import numpy as np

    p, k = phi.shape
    if k < 2:
        return phi
    rot = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        lam = phi @ rot
        u, s, vt = np.linalg.svd(
            phi.T @ (lam**3 - (1.0 / p) * lam @ np.diag(np.diag(lam.T @ lam)))
        )
        rot = u @ vt
        d = float(np.sum(s))
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return phi @ rot


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
    config: dict | None = None,
) -> RunResult:
    df = read_table(Path(fp.path))
    # user-supplied overrides for the engine's substantive defaults (column roles,
    # anchors, etc.) — each branch reads cfg.get(<key>) and falls back to its auto
    # default. See docs/loop-decisions.md for the configurable keys per analysis.
    cfg = config or {}
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
        y, rhs_vars, formula, model = _regression(df, fp, entry, cfg)
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

    elif entry.id == "efa":
        import numpy as np
        import pandas as pd

        _excl = {fp.unit_col, fp.time_col}
        items = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        if len(items) < 3:
            summary.append("EFA 失败：需要 ≥3 个连续变量（题项）。")
        else:
            try:
                from sklearn.decomposition import FactorAnalysis
                from sklearn.preprocessing import StandardScaler

                sub = df[items].dropna()
                z = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
                # sampling adequacy (factor_analyzer's KMO/Bartlett work despite its
                # FactorAnalyzer.fit sklearn-compat bug; extraction via sklearn instead).
                kmo = bartlett_p = float("nan")
                try:
                    from factor_analyzer.factor_analyzer import (
                        calculate_bartlett_sphericity,
                        calculate_kmo,
                    )

                    _, bartlett_p = calculate_bartlett_sphericity(sub)
                    _, kmo = calculate_kmo(sub)
                except Exception:
                    pass
                ev = np.sort(np.linalg.eigvalsh(np.corrcoef(z, rowvar=False)))[::-1]
                n_factors = max(1, int((ev > 1).sum()))  # Kaiser criterion
                fa = FactorAnalysis(n_components=n_factors, random_state=0).fit(z)
                load = _varimax(fa.components_.T)  # (items, factors), varimax-rotated
                ssl = (load**2).sum(axis=0)  # SS loadings per factor
                prop_var = ssl / len(items)
                load_df = pd.DataFrame(
                    np.round(load, 4),
                    index=items,
                    columns=[f"F{i + 1}" for i in range(n_factors)],
                )
                load_df.to_csv(d / "loadings.csv", encoding="utf-8")
                files.append("loadings.csv")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(range(1, len(ev) + 1), ev, "-o", color="#4C72B0")
                    ax.axhline(1.0, color="#C44E52", ls="--", lw=0.8, label="Kaiser (eigenvalue=1)")
                    ax.set_xlabel("factor")
                    ax.set_ylabel("eigenvalue")
                    ax.set_title(f"Scree plot (retained {n_factors} factors)")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    fig.savefig(d / "scree.png", dpi=150)
                    plt.close(fig)
                    files.append("scree.png")
                except Exception:
                    pass
                estimates["n_factors"] = float(n_factors)
                estimates["kmo"] = round(float(kmo), 4) if kmo == kmo else -1.0
                estimates["cumulative_variance"] = round(float(prop_var.sum()), 4)
                kmo_txt = (
                    f"KMO={kmo:.3f}（{'适合因子分析' if kmo >= 0.6 else '⚠ <0.6 数据不太适合'}）"
                    if kmo == kmo
                    else "KMO 不可用"
                )
                bart_txt = f"，Bartlett p={bartlett_p:.2g}" if bartlett_p == bartlett_p else ""
                summary.append(
                    f"{entry.method} 完成：{len(items)} 个变量 → Kaiser 准则保留 {n_factors} 个因子"
                    f"（累计解释方差 {prop_var.sum():.1%}）；{kmo_txt}{bart_txt}；varimax 旋转载荷见 loadings.csv。"
                    "⚠ 因子数(特征值>1)是启发式,碎石/平行分析可能不同；EFA 为探索性,确证用 SEM/CFA；"
                    "提取用 sklearn FactorAnalysis(factor_analyzer 提取器与本机 sklearn 不兼容)。"
                )
                code += [
                    "from sklearn.decomposition import FactorAnalysis  # EFA",
                    "# 标准化 -> 相关阵特征值定 Kaiser n -> FactorAnalysis -> 手写 varimax 旋转",
                ]
            except Exception as err:
                summary.append(f"EFA 失败：{err}")

    elif entry.id == "multinomial_logit":
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        _excl = {fp.unit_col, fp.time_col}
        out_cands = [
            c
            for c in fp.columns
            if c.kind in {"count", "categorical"} and 3 <= c.n_unique <= 10 and c.name not in _excl
        ]
        out_cands.sort(key=lambda c: 0 if c.kind == "categorical" else 1)  # prefer nominal
        outcome = out_cands[0].name if out_cands else None
        predictors = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in _excl | {outcome}
        ][:6]
        if outcome is None or not predictors:
            summary.append("多项 logit 失败：需要 3–10 类名义结果变量 + ≥1 个连续/二值预测变量。")
        else:
            try:
                sub = df[[outcome, *predictors]].dropna()
                codes, cats = pd.factorize(sub[outcome])
                X = sm.add_constant(sub[predictors].astype(float))
                model = sm.MNLogit(codes, X).fit(disp=False)
                (d / "summary.txt").write_text(str(model.summary()), encoding="utf-8")
                files.append("summary.txt")
                params, pvals = model.params, model.pvalues
                rrr = np.exp(params)
                rows = []
                for ci in params.columns:  # ci = 0..K-2 -> class cats[ci+1] vs baseline cats[0]
                    cls = str(cats[ci + 1])
                    for term in params.index:
                        rows.append(
                            (
                                cls,
                                str(term),
                                round(float(params.loc[term, ci]), 4),
                                round(float(rrr.loc[term, ci]), 4),
                                round(float(pvals.loc[term, ci]), 4),
                            )
                        )
                pd.DataFrame(
                    rows, columns=["class_vs_baseline", "term", "coef", "RRR", "p_value"]
                ).to_csv(d / "coefficients.csv", index=False, encoding="utf-8")
                files.append("coefficients.csv")
                pred = np.asarray(model.predict(X))
                acc = float((pred.argmax(axis=1) == codes).mean())
                estimates["accuracy"] = round(acc, 4)
                estimates["n_classes"] = float(len(cats))
                estimates["pseudo_r2"] = round(float(model.prsquared), 4)
                summary.append(
                    f"{entry.method} 完成：名义结果 {outcome}（{len(cats)} 类，基准={cats[0]}），"
                    f"{len(predictors)} 个预测变量；类内准确率={acc:.1%}，"
                    f"McFadden pseudo-R²={model.prsquared:.3f}；相对风险比(RRR)见 coefficients.csv。"
                    "⚠ 假定结果无序（名义）——若类别有序请用 ordered_logit；并假定 IIA（无关方案独立性）。"
                )
                code += [
                    "import statsmodels.api as sm  # 多项 logit",
                    f"# codes,_=pd.factorize(df['{outcome}']); sm.MNLogit(codes, sm.add_constant(X)).fit()",
                ]
            except Exception as err:
                summary.append(f"多项 logit 失败：{err}")

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

    elif entry.id == "topsis":
        import numpy as np

        try:
            X, crit, labels = _mcda_inputs(df, fp)
        except ValueError as err:
            summary.append(f"TOPSIS 失败：{err}")
        else:
            import pandas as pd

            cost_mask, cost_names = _cost_mask(crit, cfg)
            Z = _minmax01(X, cost_mask)  # benefit-normalised to [0,1] (cost cols flipped)
            w = _entropy_weights(Z)
            V = Z * w
            a_best, a_worst = V.max(axis=0), V.min(axis=0)
            dp = np.sqrt(((V - a_best) ** 2).sum(axis=1))
            dn = np.sqrt(((V - a_worst) ** 2).sum(axis=1))
            score = dn / (dp + dn + 1e-12)
            res = pd.DataFrame({"alternative": labels, "score": np.round(score, 4)})
            res["rank"] = res["score"].rank(ascending=False, method="min").astype(int)
            res = res.sort_values("rank").reset_index(drop=True)
            res.to_csv(d / "topsis_scores.csv", index=False, encoding="utf-8")
            files.append("topsis_scores.csv")
            pd.DataFrame({"criterion": crit, "entropy_weight": np.round(w, 4)}).to_csv(
                d / "weights.csv", index=False, encoding="utf-8"
            )
            files.append("weights.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                top = res.head(20).iloc[::-1]
                fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.32)))
                ax.barh(top["alternative"].astype(str), top["score"], color="#4C72B0")
                ax.set_xlabel("TOPSIS closeness score")
                ax.set_title(f"Entropy-weighted TOPSIS ranking (top {len(top)})")
                fig.tight_layout()
                fig.savefig(d / "topsis_ranking.png", dpi=150)
                plt.close(fig)
                files.append("topsis_ranking.png")
            except Exception:
                pass
            best = labels[int(np.argmax(score))]
            estimates["top_score"] = round(float(score.max()), 4)
            estimates["n_alternatives"] = float(len(labels))
            estimates["n_criteria"] = float(len(crit))
            summary.append(
                f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
                f"最优方案 [{best}]（贴近度 {score.max():.3f}）；熵权见 weights.csv。"
                + _mcda_direction_note(cost_names)
            )
            code += [
                "import numpy as np  # 熵权-TOPSIS",
                "# Z=min-max[0,1]; w=熵权; V=Z*w; 距理想最优/最劣 -> 贴近度 dn/(dp+dn)",
            ]

    elif entry.id == "dea":
        import numpy as np

        try:
            X, crit, labels = _mcda_inputs(df, fp)
        except ValueError as err:
            summary.append(f"DEA 失败：{err}")
        else:
            # engine default: first numeric column = output, the rest = inputs;
            # config={"inputs":[...],"outputs":[...]} overrides the i/o roles.
            inputs, outputs, in_names, out_names = _dea_io(X, crit, cfg)
            if inputs.shape[1] < 1:
                summary.append("DEA 失败：需要 ≥1 个投入 + 1 个产出（≥2 个数值列）。")
            elif (inputs <= 0).any() or (outputs <= 0).any():
                summary.append(
                    "DEA 失败：投入/产出需为正值（DEA 假定正数据）。请确保投入产出列均为正，"
                    "或移除含 0/负值的列。"
                )
            else:
                import pandas as pd

                ccr = _dea_efficiency(inputs, outputs, vrs=False)
                bcc = _dea_efficiency(inputs, outputs, vrs=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = np.where(bcc > 0, ccr / bcc, np.nan)
                res = pd.DataFrame(
                    {
                        "DMU": labels,
                        "CCR_efficiency": np.round(ccr, 4),
                        "BCC_efficiency": np.round(bcc, 4),
                        "scale_efficiency": np.round(scale, 4),
                    }
                )
                res["rank"] = res["CCR_efficiency"].rank(ascending=False, method="min").astype(int)
                res = res.sort_values("rank").reset_index(drop=True)
                res.to_csv(d / "dea_efficiency.csv", index=False, encoding="utf-8")
                files.append("dea_efficiency.csv")
                rplot = res.rename(columns={"DMU": "alternative"})
                _mcda_rank_plot(
                    rplot, "CCR_efficiency", "DEA CCR efficiency (top 20)", d / "dea_efficiency.png"
                )
                if (d / "dea_efficiency.png").exists():
                    files.append("dea_efficiency.png")
                n_eff = int(np.sum(np.isclose(ccr, 1.0, atol=1e-4)))
                estimates["n_ccr_efficient"] = float(n_eff)
                estimates["mean_ccr_efficiency"] = round(float(np.nanmean(ccr)), 4)
                estimates["n_dmu"] = float(len(labels))
                _io_note = (
                    "（按 config 指定）"
                    if (cfg.get("inputs") and cfg.get("outputs"))
                    else "⚠ 默认首列为产出、其余为投入——可用 config={\"inputs\":[...],\"outputs\":[...]} 指定。"
                )
                summary.append(
                    f"{entry.method} 完成：{len(labels)} 个 DMU，产出 {out_names}，"
                    f"投入 {in_names}; CCR 技术有效 {n_eff} 个（θ=1），平均效率 "
                    f"{np.nanmean(ccr):.3f}；规模效率=CCR/BCC。" + _io_note
                )
                code += [
                    "from scipy.optimize import linprog  # 投入导向 DEA(CCR+BCC)",
                    f'# 产出={out_names}, 投入={in_names}; min θ s.t. Σλx≤θx_o, Σλy≥y_o, λ≥0',
                ]

    elif entry.id == "sfa":
        import re

        import numpy as np

        from researchforge.executor import rbridge

        _excl = {fp.unit_col, fp.time_col}
        crit = [
            c.name for c in fp.columns if c.kind in {"continuous", "count"} and c.name not in _excl
        ]
        names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in crit)
        if len(crit) < 2:
            summary.append("SFA 失败：需要 ≥1 投入 + 1 产出（≥2 个数值列）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("frontier")):
            summary.append(
                "SFA 需要 R 的 frontier 包（未检测到）。安装：install.packages('frontier')；"
                "或用 DEA（确定性前沿，纯 Python，无需 R）。"
            )
        elif not names_safe:
            summary.append("SFA 失败：列名需为标识符式（字母/数字/. _）。")
        else:
            # config={"inputs":[...],"outputs":[...]} overrides i/o roles; SFA is
            # single-output (Cobb-Douglas), so only the first output is used.
            in_names, out_names = _io_names(crit, cfg)
            output_col, in_cols = out_names[0], in_names
            sfa_multi_out = (
                "；⚠ SFA 为单产出模型，仅用首个产出 " + output_col if len(out_names) > 1 else ""
            )
            label_col = next(
                (c.name for c in fp.columns if c.kind in {"id", "categorical"} and c.name not in _excl),
                None,
            )
            sub = df[crit + ([label_col] if label_col else [])].dropna()
            if (sub[crit].to_numpy(dtype=float) <= 0).any():
                summary.append("SFA 失败：投入/产出需为正值（Cobb-Douglas 取对数）。")
            else:
                labels = (
                    sub[label_col].astype(str).tolist()
                    if label_col
                    else [f"row{i + 1}" for i in range(len(sub))]
                )
                csv = d / "_sfa_input.csv"
                sub[crit].to_csv(csv, index=False)
                try:
                    import pandas as pd

                    coef, te = _sfa_via_r(csv, output_col, in_cols)
                    elastic = {
                        k: v
                        for k, v in coef.items()
                        if k not in ("sigmaSq", "gamma", "lr_stat", "lr_pvalue")
                    }
                    pd.DataFrame(
                        {"term": list(elastic), "elasticity": [round(v, 4) for v in elastic.values()]}
                    ).to_csv(d / "frontier_coefficients.csv", index=False, encoding="utf-8")
                    files.append("frontier_coefficients.csv")
                    teres = pd.DataFrame({"unit": labels, "technical_efficiency": np.round(te, 4)})
                    teres = teres.sort_values("technical_efficiency", ascending=False).reset_index(
                        drop=True
                    )
                    teres.to_csv(d / "technical_efficiency.csv", index=False, encoding="utf-8")
                    files.append("technical_efficiency.csv")
                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(te, bins=min(20, max(5, len(te) // 4)), color="#4C72B0", edgecolor="white")
                        ax.axvline(float(np.mean(te)), color="#C44E52", ls="--", label=f"mean={np.mean(te):.3f}")
                        ax.set_xlabel("technical efficiency")
                        ax.set_ylabel("count")
                        ax.set_title("SFA technical-efficiency distribution")
                        ax.legend(fontsize=8)
                        fig.tight_layout()
                        fig.savefig(d / "efficiency_distribution.png", dpi=150)
                        plt.close(fig)
                        files.append("efficiency_distribution.png")
                    except Exception:
                        pass
                    mean_te = float(np.mean(te))
                    gamma = coef.get("gamma", float("nan"))
                    lr_p = coef.get("lr_pvalue", float("nan"))
                    estimates["mean_technical_efficiency"] = round(mean_te, 4)
                    estimates["gamma"] = round(gamma, 4)
                    estimates["lr_inefficiency_pvalue"] = round(lr_p, 4)
                    estimates["n_dmu"] = float(len(labels))
                    for k, v in elastic.items():
                        if "Intercept" not in k:
                            estimates[k] = round(v, 4)
                    # if the one-sided LR test can't reject γ=0, the model is ~OLS and
                    # the technical-efficiency scores are not trustworthy (Opus catch).
                    ineff_sig = lr_p < 0.05
                    te_warn = (
                        ""
                        if ineff_sig
                        else "；⚠ 低效 LR 检验不显著（无统计学上的无效率），模型接近 OLS，技术效率值不可靠"
                    )
                    summary.append(
                        f"{entry.method} 完成（R/frontier）：Cobb-Douglas 前沿 [{output_col}] ~ {in_cols}；"
                        f"平均技术效率 {mean_te:.3f}（最优=1）；γ={gamma:.3f}"
                        "（=σ_u²/(σ_u²+σ_v²) 比值，越近 1 越说明偏离前沿主要是低效而非噪声）；"
                        f"低效存在性 LR 检验 p={lr_p:.3g}（{'显著存在低效' if ineff_sig else '不显著'}）"
                        f"{te_warn}{sfa_multi_out}。弹性见 frontier_coefficients.csv。"
                        + (
                            "（投入产出按 config 指定）"
                            if (cfg.get("inputs") and cfg.get("outputs"))
                            else "⚠ 默认首列为产出、其余投入（可用 config inputs/outputs 指定）"
                        )
                        + "；假定 Cobb-Douglas + 半正态低效。"
                    )
                    code += [
                        "library(frontier)  # 随机前沿(Cobb-Douglas, ML)",
                        f"# sfa(log({output_col}) ~ {' + '.join(f'log({c})' for c in in_cols)}); efficiencies()",
                    ]
                except Exception as err:
                    summary.append(f"SFA 拟合失败：{err}")
                finally:
                    try:
                        csv.unlink()
                    except OSError:
                        pass

    elif entry.id == "malmquist":
        import numpy as np

        _excl = {fp.unit_col, fp.time_col}
        crit = [
            c.name for c in fp.columns if c.kind in {"continuous", "count"} and c.name not in _excl
        ]
        if not (fp.unit_col and fp.time_col):
            summary.append("Malmquist 失败：需要面板数据（单位列 + 时间列）。")
        elif len(crit) < 2:
            summary.append("Malmquist 失败：需要 ≥1 投入 + 1 产出（≥2 个数值列）。")
        else:
            import pandas as pd

            # config={"inputs":[...],"outputs":[...]} overrides i/o; Malmquist DEA
            # supports multiple outputs. out_col kept as a single label for the title.
            in_names, out_names = _io_names(crit, cfg)
            in_cols, out_cols = in_names, out_names
            out_col = ", ".join(out_cols)
            periods = sorted(df[fp.time_col].dropna().unique())
            if len(periods) < 2:
                summary.append("Malmquist 失败：需要 ≥2 个时间期。")
            else:
                # default: first vs last period; config={"periods":[start,end]} picks
                # a specific base/end pair (both must exist in the data).
                t0, t1 = periods[0], periods[-1]
                want = cfg.get("periods")
                if isinstance(want, (list, tuple)) and len(want) == 2:
                    pset = set(periods)
                    if want[0] in pset and want[1] in pset:
                        t0, t1 = want[0], want[1]
                d0 = df[df[fp.time_col] == t0].drop_duplicates(fp.unit_col).set_index(fp.unit_col)
                d1 = df[df[fp.time_col] == t1].drop_duplicates(fp.unit_col).set_index(fp.unit_col)
                common = [u for u in d0.index if u in d1.index]
                d0, d1 = d0.loc[common], d1.loc[common]
                xi0, yo0 = d0[in_cols].to_numpy(float), d0[out_cols].to_numpy(float)
                xi1, yo1 = d1[in_cols].to_numpy(float), d1[out_cols].to_numpy(float)
                if len(common) < 3:
                    summary.append("Malmquist 失败：两期共同单位不足（<3）。")
                elif (xi0 <= 0).any() or (yo0 <= 0).any() or (xi1 <= 0).any() or (yo1 <= 0).any():
                    summary.append("Malmquist 失败：投入/产出需为正值。")
                else:
                    # CRS distance functions (4 cross-period DEA scores per DMU)
                    e_tt = _dea_cross(xi0, yo0, xi0, yo0)  # t  obs vs t  frontier
                    e_11 = _dea_cross(xi1, yo1, xi1, yo1)  # t1 obs vs t1 frontier
                    e_t_1 = _dea_cross(xi1, yo1, xi0, yo0)  # t1 obs vs t  frontier
                    e_1_t = _dea_cross(xi0, yo0, xi1, yo1)  # t  obs vs t1 frontier
                    ec = e_11 / e_tt  # efficiency change (catch-up)
                    tc = np.sqrt((e_t_1 / e_11) * (e_tt / e_1_t))  # technical change (frontier shift)
                    m = ec * tc  # Malmquist TFP change (>1 = growth)
                    res = pd.DataFrame(
                        {
                            str(fp.unit_col): common,
                            "malmquist_tfp": np.round(m, 4),
                            "efficiency_change": np.round(ec, 4),
                            "technical_change": np.round(tc, 4),
                        }
                    )
                    res.to_csv(d / "malmquist.csv", index=False, encoding="utf-8")
                    files.append("malmquist.csv")

                    def _gmean(a):
                        a = a[np.isfinite(a) & (a > 0)]
                        return float(np.exp(np.mean(np.log(a)))) if len(a) else float("nan")

                    gm_m, gm_ec, gm_tc = _gmean(m), _gmean(ec), _gmean(tc)
                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(6, max(3, len(common) * 0.3)))
                        ax.barh([str(u) for u in common][::-1], m[::-1], color="#55A868")
                        ax.axvline(1.0, color="grey", ls="--", lw=0.8)
                        ax.set_xlabel("Malmquist TFP index (>1 = growth)")
                        ax.set_title(f"Malmquist productivity change {t0}→{t1}")
                        fig.tight_layout()
                        fig.savefig(d / "malmquist.png", dpi=150)
                        plt.close(fig)
                        files.append("malmquist.png")
                    except Exception:
                        pass
                    estimates["mean_malmquist_tfp"] = round(gm_m, 4)
                    estimates["mean_efficiency_change"] = round(gm_ec, 4)
                    estimates["mean_technical_change"] = round(gm_tc, 4)
                    estimates["n_dmu"] = float(len(common))
                    verdict = "TFP 上升" if gm_m > 1.01 else ("TFP 下降" if gm_m < 0.99 else "TFP 基本不变")
                    summary.append(
                        f"{entry.method} 完成：{len(common)} 个单位 {t0}→{t1}；产出 {out_cols}，"
                        f"投入 {in_cols}；总体 Malmquist TFP={gm_m:.3f}（{verdict}）"
                        f"= 效率变化 {gm_ec:.3f} × 技术变化 {gm_tc:.3f}（>1 为增长）。"
                        + (
                            "（投入产出/期间按 config 指定）"
                            if (cfg.get("inputs") and cfg.get("outputs")) or cfg.get("periods")
                            else "⚠ 默认首数值列为产出、其余投入、首末两期；CRS 距离函数（可配 inputs/outputs/periods）"
                        )
                    )
                    code += [
                        "from scipy.optimize import linprog  # Malmquist(Färe1994), CRS 距离函数",
                        "# M = (E11/Ett)·sqrt((E[t1|t]/E11)·(Ett/E[t|t1])); 分解 EC×TC",
                    ]

    elif entry.id == "critic":
        import numpy as np

        try:
            X, crit, labels = _mcda_inputs(df, fp)
        except ValueError as err:
            summary.append(f"CRITIC 失败：{err}")
        else:
            import pandas as pd

            cost_mask, cost_names = _cost_mask(crit, cfg)
            Z = _minmax01(X, cost_mask)  # benefit-normalised [0,1] (cost cols flipped)
            sigma = Z.std(axis=0, ddof=1)  # contrast intensity per criterion
            # clip to [-1,1]: float noise can push r just past 1, making (1-r)<0 and
            # flipping weights negative when criteria are (near-)perfectly correlated.
            corr = np.clip(np.nan_to_num(np.corrcoef(Z, rowvar=False), nan=0.0), -1.0, 1.0)
            if corr.ndim == 0:
                corr = np.array([[1.0]])
            conflict = (1.0 - corr).sum(axis=1)  # conflict = Σ_k (1 - r_jk) ≥ 0
            info = sigma * conflict  # CRITIC information content C_j
            w = info / info.sum() if info.sum() > 0 else np.ones(len(crit)) / len(crit)
            composite = Z @ w
            res = pd.DataFrame({"alternative": labels, "critic_score": np.round(composite, 4)})
            res["rank"] = res["critic_score"].rank(ascending=False, method="min").astype(int)
            res = res.sort_values("rank").reset_index(drop=True)
            res.to_csv(d / "critic_scores.csv", index=False, encoding="utf-8")
            files.append("critic_scores.csv")
            pd.DataFrame({"criterion": crit, "critic_weight": np.round(w, 4)}).to_csv(
                d / "weights.csv", index=False, encoding="utf-8"
            )
            files.append("weights.csv")
            _mcda_rank_plot(res, "critic_score", "CRITIC-weighted ranking (top 20)", d / "critic_ranking.png")
            if (d / "critic_ranking.png").exists():
                files.append("critic_ranking.png")
            best = labels[int(np.argmax(composite))]
            estimates["top_score"] = round(float(composite.max()), 4)
            estimates["n_alternatives"] = float(len(labels))
            estimates["n_criteria"] = float(len(crit))
            summary.append(
                f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
                f"最优 [{best}]（CRITIC 加权得分 {composite.max():.3f}）；"
                "CRITIC 权重=对比度(标准差)×冲突性(1-相关) 客观赋权,见 weights.csv。"
                + _mcda_direction_note(cost_names)
            )
            code += [
                "import numpy as np  # CRITIC 客观赋权",
                "# w_j ∝ σ_j · Σ_k(1-r_jk); 综合得分 = Σ_j w_j · min-max(x_ij)",
            ]

    elif entry.id == "membership_function":
        import numpy as np

        try:
            X, crit, labels = _mcda_inputs(df, fp)
        except ValueError as err:
            summary.append(f"隶属函数法失败：{err}")
        else:
            import pandas as pd

            cost_mask, cost_names = _cost_mask(crit, cfg)
            M = _minmax01(X, cost_mask)  # membership degrees in [0,1] (cost cols flipped)
            composite = M.mean(axis=1)  # classic equal-weight average membership
            res = pd.DataFrame({"alternative": labels, "membership_score": np.round(composite, 4)})
            res["rank"] = res["membership_score"].rank(ascending=False, method="min").astype(int)
            res = res.sort_values("rank").reset_index(drop=True)
            res.to_csv(d / "membership_scores.csv", index=False, encoding="utf-8")
            files.append("membership_scores.csv")
            memb = pd.DataFrame(np.round(M, 4), columns=crit)
            memb.insert(0, "alternative", labels)
            memb.to_csv(d / "membership_matrix.csv", index=False, encoding="utf-8")
            files.append("membership_matrix.csv")
            _mcda_rank_plot(
                res, "membership_score", "Membership-function ranking (top 20)",
                d / "membership_ranking.png",
            )
            if (d / "membership_ranking.png").exists():
                files.append("membership_ranking.png")
            best = labels[int(np.argmax(composite))]
            estimates["top_score"] = round(float(composite.max()), 4)
            estimates["n_alternatives"] = float(len(labels))
            estimates["n_criteria"] = float(len(crit))
            summary.append(
                f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
                f"最优 [{best}]（隶属度均值 {composite.max():.3f}，等权）。"
                + _mcda_direction_note(cost_names)
            )
            code += [
                "import numpy as np  # 隶属函数法(等权)",
                "# M = min-max[0,1] 隶属度; 综合得分 = 各指标隶属度的等权平均",
            ]

    elif entry.id == "grey_relational":
        import numpy as np

        try:
            X, crit, labels = _mcda_inputs(df, fp)
        except ValueError as err:
            summary.append(f"灰色关联分析失败：{err}")
        else:
            import pandas as pd

            cost_mask, cost_names = _cost_mask(crit, cfg)
            M = _minmax01(X, cost_mask)
            delta = np.abs(1.0 - M)  # distance to the ideal (benefit -> ideal = 1)
            dmin, dmax, rho = delta.min(), delta.max(), 0.5
            xi = (dmin + rho * dmax) / (delta + rho * dmax + 1e-12)  # grey relational coef
            grade = xi.mean(axis=1)  # grey relational grade (equal weight)
            res = pd.DataFrame({"alternative": labels, "relational_grade": np.round(grade, 4)})
            res["rank"] = res["relational_grade"].rank(ascending=False, method="min").astype(int)
            res = res.sort_values("rank").reset_index(drop=True)
            res.to_csv(d / "grey_relational.csv", index=False, encoding="utf-8")
            files.append("grey_relational.csv")
            _mcda_rank_plot(
                res, "relational_grade", "Grey relational ranking (top 20)",
                d / "grey_ranking.png",
            )
            if (d / "grey_ranking.png").exists():
                files.append("grey_ranking.png")
            best = labels[int(np.argmax(grade))]
            estimates["top_grade"] = round(float(grade.max()), 4)
            estimates["n_alternatives"] = float(len(labels))
            estimates["n_criteria"] = float(len(crit))
            summary.append(
                f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
                f"最优 [{best}]（关联度 {grade.max():.3f}，ρ=0.5，参考序列=各指标理想值）。"
                + _mcda_direction_note(cost_names)
            )
            code += [
                "import numpy as np  # 灰色关联分析(GRA)",
                "# Δ=|1-min-max|; ξ=(Δmin+0.5Δmax)/(Δ+0.5Δmax); 关联度=ξ 行均值",
            ]

    elif entry.id == "soil_texture":
        import numpy as np
        import pandas as pd

        def _find(kw):
            # name-locked to the texture fraction; accept any numeric kind
            # (whole-number distinct % columns can profile as "id").
            return next(
                (
                    c.name
                    for c in fp.columns
                    if kw in c.name.lower() and c.kind in {"continuous", "count", "id"}
                ),
                None,
            )

        sand_c, silt_c, clay_c = _find("sand"), _find("silt"), _find("clay")
        if not (sand_c and silt_c and clay_c):
            summary.append("土壤质地分类失败：需要 sand/silt/clay（砂/粉/黏粒）百分比列。")
        else:
            raw = df[[sand_c, silt_c, clay_c]].dropna().astype(float)
            raw = raw[raw.sum(axis=1) > 0]
            norm = raw.div(raw.sum(axis=1), axis=0) * 100.0  # renormalise rows to sum 100
            classes = [
                _usda_texture(float(r[sand_c]), float(r[silt_c]), float(r[clay_c]))
                for _, r in norm.iterrows()
            ]
            res = norm.round(2)
            res["usda_texture"] = classes
            res.to_csv(d / "soil_texture.csv", index=False, encoding="utf-8")
            files.append("soil_texture.csv")
            dist = pd.Series(classes).value_counts()
            dist.rename_axis("texture_class").reset_index(name="count").to_csv(
                d / "texture_distribution.csv", index=False, encoding="utf-8"
            )
            files.append("texture_distribution.csv")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                cats = list(dist.index)
                cmap = plt.get_cmap("tab20")
                cidx = {c: cmap(i % 20) for i, c in enumerate(cats)}
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                # soil texture triangle (clay apex at top)
                cl = norm[clay_c].to_numpy()
                si = norm[silt_c].to_numpy()
                x = si + 0.5 * cl
                y = cl * (np.sqrt(3) / 2)
                tri = np.array([[0, 0], [100, 0], [50, 100 * np.sqrt(3) / 2], [0, 0]])
                axes[0].plot(tri[:, 0], tri[:, 1], color="#444", lw=1)
                for c in cats:
                    m = np.array([k == c for k in classes])
                    axes[0].scatter(x[m], y[m], s=22, color=cidx[c], label=c, edgecolor="#333", linewidth=0.3)
                axes[0].text(0, -4, "sand", ha="center")
                axes[0].text(100, -4, "silt", ha="center")
                axes[0].text(50, 100 * np.sqrt(3) / 2 + 3, "clay", ha="center")
                axes[0].set_title("USDA soil texture triangle")
                axes[0].axis("off")
                axes[0].legend(fontsize=6, loc="upper right", ncol=2)
                axes[1].barh([str(c) for c in cats][::-1], dist.values[::-1], color="#55A868")
                axes[1].set_xlabel("count")
                axes[1].set_title("Texture class distribution")
                fig.tight_layout()
                fig.savefig(d / "soil_texture.png", dpi=150)
                plt.close(fig)
                files.append("soil_texture.png")
            except Exception:
                pass
            dominant = str(dist.index[0])
            estimates["n_samples"] = float(len(norm))
            estimates["n_classes"] = float(len(dist))
            estimates["dominant_class_pct"] = round(100.0 * float(dist.iloc[0]) / len(norm), 2)
            summary.append(
                f"{entry.method} 完成：{len(norm)} 个土样按 USDA 质地三角分入 {len(dist)} 类；"
                f"最多为「{dominant}」（{100.0 * dist.iloc[0] / len(norm):.0f}%）；"
                f"质地三角图见 soil_texture.png。（各行已归一化至砂+粉+黏=100%）"
            )
            code += [
                "# USDA 质地三角分类: 按 sand/silt/clay% 的标准边界规则判类",
                "# silt+1.5*clay<15 -> sand; ... clay>=40&silt<40&sand<=45 -> clay 等 12 类",
            ]

    elif entry.id == "kriging":
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
            import pandas as pd

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
                _anch = _qca_anchors(cfg)
                _ic = _qca_incl_cut(cfg, 0.8)
                sol_str, tab = _fsqca_via_r(csv, outcome, conditions, anchors=_anch, incl_cut=_ic)
                tab.to_csv(d / "fsqca_solution.csv", index=False, encoding="utf-8")
                files.append("fsqca_solution.csv")
                (d / "solution.txt").write_text(
                    f"充分性解（complex solution，sufficient configurations） → {outcome}:\n"
                    f"  {sol_str}\n\n"
                    f"直接校准(百分位锚点 {_anch[0]}/{_anch[1]}/{_anch[2]})，incl.cut={_ic}；"
                    "* = 逻辑与（AND）, + = 或（OR）\n"
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
                _anch_note = (
                    "（锚点/incl.cut 按 config 指定）"
                    if (cfg.get("anchors") or cfg.get("incl_cut"))
                    else f"（锚点 {_anch[0]}/{_anch[1]}/{_anch[2]}、incl.cut={_ic} 为机械起点，"
                    "可用 config anchors/incl_cut 按理论设定）"
                )
                summary.append(
                    f"{entry.method} 完成（R/QCA，complex 解）：充分配置 [{sol_str}] → {outcome}；"
                    f"{len(tab)} 个配置，一致性 {tab['consistency'].min():.3f}–{tab['consistency'].max():.3f}"
                    "（* =AND, + =OR；充分性≠因果证明）" + _anch_note
                )
                code += [
                    "library(QCA)  # 直接校准 -> 真值表 -> 布尔最小化",
                    f'# calibrate({[outcome, *conditions]}); truthTable(outcome="{outcome}"); minimize(incl.cut={_ic})',
                ]
            except Exception as err:
                summary.append(f"fsQCA 失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "csqca":
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
            summary.append("csQCA 失败：需要 1 个结果变量 + ≥2 个条件变量（均连续，按中位数二分）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("QCA")):
            summary.append(
                "csQCA 需要 R 的 QCA 包（未检测到）。安装：install.packages('QCA')；"
                "或先用 NCA（必要条件分析，纯 Python，无需 R）。"
            )
        elif not names_safe:
            summary.append("csQCA 失败：列名需为标识符式（字母/数字/. _）。")
        else:
            sub = df[[outcome, *conditions]].dropna()
            csv = d / "_qca_input.csv"
            sub.to_csv(csv, index=False)
            try:
                _ic = _qca_incl_cut(cfg, 0.8)
                sol_str, tab = _csqca_via_r(csv, outcome, conditions, incl_cut=_ic)
                tab.to_csv(d / "csqca_solution.csv", index=False, encoding="utf-8")
                files.append("csqca_solution.csv")
                (d / "solution.txt").write_text(
                    f"清晰集 QCA 充分性解（complex solution） → {outcome}:\n  {sol_str}\n\n"
                    f"清晰校准(二值列直接用,连续列按中位数二分),incl.cut={_ic}；* =AND, + =OR\n"
                    "说明：① complex 解(不纳入反事实)；② 中位数二分丢失信息——连续条件通常 fsQCA 更优,"
                    "且偏态/离散数据的中位数二分可能极不均衡(近恒值)；③ 充分性≠因果。\n\n"
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
                    "（* =AND, + =OR；连续条件按中位数二分=信息损失,连续数据建议改用 fsQCA；充分性≠因果）"
                    + ("（incl.cut 按 config 指定）" if cfg.get("incl_cut") else f"（incl.cut={_ic} 可配）")
                )
                code += [
                    "library(QCA)  # 清晰集 QCA: 中位数二分 -> 真值表 -> 布尔最小化",
                    f'# truthTable(outcome="{outcome}", conditions={conditions}); minimize(incl.cut={_ic})',
                ]
            except Exception as err:
                summary.append(f"csQCA 失败：{err}")
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
                _anch = _qca_anchors(cfg)
                _ic = _qca_incl_cut(cfg, 0.9)
                tab = _qca_necessity_via_r(csv, outcome, conditions, anchors=_anch, incl_cut=_ic)
                tab = tab.sort_values("consistency_inclN", ascending=False).reset_index(drop=True)
                tab.to_csv(d / "necessity.csv", index=False, encoding="utf-8")
                files.append("necessity.csv")
                (d / "necessity.txt").write_text(
                    f"必要性分析（superSubset） → {outcome}（fuzzy 校准 {_anch[0]}/{_anch[1]}/{_anch[2]}，incl.cut={_ic}）:\n"
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
                    + ("（锚点/incl.cut 按 config 指定）" if (cfg.get("anchors") or cfg.get("incl_cut")) else f"（incl.cut={_ic} 可配 anchors/incl_cut）")
                )
                code += [
                    f"library(QCA)  # 必要性: 模糊校准 -> superSubset(incl.cut={_ic}, cov.cut=0.5)",
                    f'# superSubset(cal, outcome="{outcome}", conditions={conditions})',
                ]
            except Exception as err:
                summary.append(f"QCA 必要性分析失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "panel_qca":
        import re

        from researchforge.executor import rbridge

        unit, time = fp.unit_col, fp.time_col
        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {unit, time}]
        outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
        forced = [c for c in (cfg.get("predictors") or cfg.get("conditions") or []) if c in cont and c != outcome]
        conditions = forced[:5] if forced else [c for c in cont if c != outcome][:5]
        anchors = _qca_anchors(cfg)
        incl_cut = _qca_incl_cut(cfg, 0.8)
        names_safe = outcome is not None and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, *conditions]
        )
        if not (unit and time):
            summary.append("面板 QCA 失败：需要面板数据（单位列 + 时间列）。")
        elif outcome is None or len(conditions) < 2:
            summary.append("面板 QCA 失败：需要 1 个结果 + ≥2 个条件（均连续，将模糊校准）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("SetMethods") and rbridge.r_package_available("QCA")):
            summary.append("面板 QCA 需要 R 的 SetMethods + QCA 包（未检测到）。安装：install.packages(c('QCA','SetMethods'))；或用 fsqca（截面）。")
        elif not names_safe:
            summary.append("面板 QCA 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
        else:
            import pandas as pd

            sub = df[[outcome, *conditions, unit]].dropna()
            csv = d / "_pqca_input.csv"
            sub.to_csv(csv, index=False)
            try:
                sol_str, terms = _panel_qca_via_r(csv, outcome, conditions, unit, anchors, incl_cut)
                terms = terms.sort_values("pooled_consistency", ascending=False).reset_index(drop=True)
                terms.to_csv(d / "panel_qca_terms.csv", index=False, encoding="utf-8")
                files.append("panel_qca_terms.csv")
                # large between/within distance => the configuration is NOT uniform
                # across units / over time (clustered heterogeneity)
                het = terms[(terms["dist_between"] > 0.2) | (terms["dist_within"] > 0.2)]["term"].tolist()
                estimates["n_terms"] = float(len(terms))
                estimates["max_pooled_consistency"] = round(float(terms["pooled_consistency"].max()), 4)
                estimates["max_dist_between"] = round(float(terms["dist_between"].max()), 4)
                estimates["max_dist_within"] = round(float(terms["dist_within"].max()), 4)
                het_txt = (
                    f"⚠ 跨单位/时间不稳定项：{het}（between/within→pooled 距离>0.2，配置在子总体间不一致）"
                    if het else "各项 between/within 距离均小（配置在单位/时间间较稳定）"
                )
                (d / "panel_qca.txt").write_text(
                    f"面板/聚类 fsQCA（SetMethods cluster，分位锚点 {anchors[0]}/{anchors[1]}/{anchors[2]}，incl.cut={incl_cut}）\n"
                    f"汇总(pooled)充分性解 → {outcome}:  {sol_str}\n"
                    f"按单位 {unit} 聚类，分解每个解项的 一致性：汇总(POCOS) vs 组间(between) vs 组内(within)；"
                    "dBP/dWP=组间/组内到汇总的距离(越大越不一致)。\n"
                    f"{het_txt}\n\n" + terms.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("panel_qca.txt")
                summary.append(
                    f"{entry.method} 完成（R/SetMethods）：汇总解 [{sol_str}] → {outcome}（按 {unit} 聚类）；"
                    f"{len(terms)} 个解项，最高汇总一致性 {terms['pooled_consistency'].max():.3f}，"
                    f"最大组间距离 {terms['dist_between'].max():.3f}、组内 {terms['dist_within'].max():.3f}。{het_txt}。"
                    "⚠ 配置性充分≠因果；距离大说明 pooled 解掩盖了子总体差异。"
                )
                code += [
                    "library(SetMethods)  # 面板/聚类 fsQCA",
                    f"# minimize(...) -> cluster(results=sol, unit_id, cluster_id='{unit}'); POCOS/dBP/dWP",
                ]
            except Exception as err:
                summary.append(f"面板 QCA 失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "cna":
        import re

        from researchforge.executor import rbridge

        # CNA's factors ARE binary/continuous conditions — do NOT drop binary
        # treatment_candidates (they're exactly the configurational factors we need).
        _excl = {fp.unit_col, fp.time_col}
        binc = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
        contc = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        factors = (binc + contc)[:8]
        fuzzy = bool(contc)  # any continuous factor -> fuzzy-calibrate; else crisp 0/1
        outcome = cfg["outcome"] if cfg.get("outcome") in factors else None
        con = _qca_incl_cut({"incl_cut": cfg.get("con")} if cfg.get("con") else {}, 0.8)
        cov = _qca_incl_cut({"incl_cut": cfg.get("cov")} if cfg.get("cov") else {}, 0.8)
        anchors = _qca_anchors(cfg)
        names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in factors)
        if len(factors) < 3:
            summary.append("CNA 失败：需要 ≥3 个因子（二值或连续条件/结果列）。")
        elif not (rbridge.r_available() and rbridge.r_package_available("cna")):
            summary.append("CNA 需要 R 的 cna 包（未检测到）。安装：install.packages('cna')；或用 fsqca（单结果）。")
        elif not names_safe:
            summary.append("CNA 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
        else:
            import pandas as pd

            sub = df[factors].dropna()
            csv = d / "_cna_input.csv"
            sub.to_csv(csv, index=False)
            try:
                asf, n_csf = _cna_via_r(csv, factors, outcome, con, cov, anchors, fuzzy)
                asf = asf.sort_values(["consistency", "coverage"], ascending=False).reset_index(drop=True)
                asf.to_csv(d / "cna_solutions.csv", index=False, encoding="utf-8")
                files.append("cna_solutions.csv")
                cal_txt = (
                    f"模糊校准(分位锚点 {anchors[0]}/{anchors[1]}/{anchors[2]})"
                    if fuzzy else "清晰集(0/1 直接用)"
                )
                outs = sorted(asf["outcome"].unique().tolist())
                top = asf.iloc[0]
                estimates["n_solutions"] = float(len(asf))
                estimates["n_outcomes"] = float(len(outs))
                estimates["max_consistency"] = round(float(asf["consistency"].max()), 4)
                estimates["n_complex_structures"] = float(n_csf)
                (d / "cna_solutions.txt").write_text(
                    f"巧合分析 CNA（R/cna，{cal_txt}，con≥{con}，cov≥{cov}）\n"
                    f"发现 {len(asf)} 个原子解(asf)，涉及结果 {outs}；复杂结构(csf) {n_csf} 个。\n"
                    "记号：* =与(AND)，+ =或(OR)，<-> 左侧为右侧结果的(配置性)原因；"
                    "con=一致性(充分性)，cov=覆盖率(必要性)。\n"
                    "CNA 不预设单一结果，可揭示多结果因果链；与 QCA 互补。\n\n"
                    + asf.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("cna_solutions.txt")
                summary.append(
                    f"{entry.method} 完成（R/cna，{cal_txt}）：{len(asf)} 个原子解、"
                    f"结果变量 {outs}、复杂结构 {n_csf} 个；最强解 [{top['condition']}]"
                    f"（con={top['consistency']:.3f}, cov={top['coverage']:.3f}）。"
                    "⚠ 配置性因果≠净效应；解依赖 con/cov 阈值与校准锚点；有限多样性下慎读。"
                )
                code += [
                    "library(cna)  # 巧合分析(多结果配置性因果)",
                    f"# cna(d, type='{'fs' if fuzzy else 'cs'}', con={con}, cov={cov}); asf()/csf()",
                ]
            except Exception as err:
                summary.append(f"CNA 失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "double_ml":
        import importlib.util

        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
        # treatment: config, else a binary column (treatment candidate)
        treatment = cfg.get("treatment")
        if treatment is None:
            treatment = next(
                (c.name for c in fp.columns if c.kind == "binary" and c.name not in {fp.unit_col, fp.time_col}),
                None,
            )
        outcome = cfg["outcome"] if cfg.get("outcome") in cont else next((c for c in cont if c != treatment), None)
        forced_ctrl = [c for c in (cfg.get("controls") or cfg.get("predictors") or []) if c in df.columns and c not in {outcome, treatment}]
        if forced_ctrl:
            controls = forced_ctrl[:20]
        else:
            controls = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "count", "binary"}
                and c.name not in {outcome, treatment, fp.unit_col, fp.time_col}
            ][:15]
        try:
            n_folds = max(2, int(cfg.get("n_folds", 5)))
        except (TypeError, ValueError):
            n_folds = 5
        if importlib.util.find_spec("doubleml") is None:
            summary.append("双重机器学习需要 doubleml 包（未检测到）。安装：pip install doubleml；或用 ols_regression（线性控制）。")
        elif treatment is None:
            summary.append(
                "双重机器学习失败：需要一个处理变量。用 config={\"treatment\":\"<列>\"} 指定"
                "（二值→ATE/IRM，连续→偏线性/PLR）。"
            )
        elif outcome is None:
            summary.append("双重机器学习失败：需要一个连续结果变量（≠ treatment）。")
        elif not controls:
            summary.append("双重机器学习失败：需要 ≥1 个混杂/控制变量（DML 靠它们去混杂）。")
        else:
            try:
                import pandas as pd

                disc = df[treatment].dropna().nunique() == 2
                try:
                    seed = int(cfg.get("seed", 0))
                except (TypeError, ValueError):
                    seed = 0
                res = _dml_via_doubleml(df, outcome, treatment, controls, n_folds, disc, d / "dml_ate.png", seed=seed)
                if (d / "dml_ate.png").exists():
                    files.append("dml_ate.png")
                ate, lb, ub = res["ate"], res["ci_lb"], res["ci_ub"]
                p, kind = res["p_value"], res["model"]
                label = res["estimand"]  # "ATE" (IRM) or "PLR_coefficient" (continuous treatment)
                pd.DataFrame(
                    {"quantity": [label], "estimate": [round(ate, 4)], "se": [round(res["se"], 4)],
                     "ci_lower": [round(lb, 4)], "ci_upper": [round(ub, 4)], "p_value": [round(p, 4)]}
                ).to_csv(d / "dml_estimate.csv", index=False, encoding="utf-8")
                files.append("dml_estimate.csv")
                estimates["ate"] = round(ate, 4)
                estimates["ci_lb"] = round(lb, 4)
                estimates["ci_ub"] = round(ub, 4)
                estimates["p_value"] = round(p, 4)
                estimates["n_controls"] = float(len(controls))
                sig = "显著" if p < 0.05 else "不显著"
                overlap_txt = (
                    "；⚠ 倾向得分接近 0/1（重叠/正值性可疑，ATE 不稳，检查处理组与对照的协变量重叠）"
                    if res["overlap_warn"] else ""
                )
                mlabel = "IRM（二值处理 ATE）" if kind == "IRM" else "PLR（偏线性）"
                est_word = "平均处理效应 ATE" if kind == "IRM" else "PLR 偏线性系数（假定处理的边际效应恒定）"
                enc_txt = ""
                if res.get("treat_map"):
                    k0 = [k for k, v in res["treat_map"].items() if v == 0][0]
                    k1 = [k for k, v in res["treat_map"].items() if v == 1][0]
                    enc_txt = f"；处理编码 {k0}→0、{k1}→1（ATE 为 1 相对 0 的效应）"
                (d / "dml_summary.txt").write_text(
                    f"双重/去偏机器学习 DML（doubleml {kind}，RandomForest 学习器，{n_folds} 折交叉拟合，seed={seed}）\n"
                    f"处理 {treatment} → 结果 {outcome}，控制 {len(controls)} 个协变量{enc_txt}\n"
                    f"{est_word} = {ate:.4f}（SE={res['se']:.4f}），95% CI [{lb:.4f}, {ub:.4f}]，p={p:.4g}\n"
                    "DML 用 ML 学习干扰项(结果模型+处理/倾向模型)再正交化+交叉拟合，去偏后做 √n 推断。\n"
                    "假定：无未观测混杂(条件可忽略)、重叠(正值性)；因果解释依赖这些假定，非自动成立。\n"
                    "注：交叉拟合样本切分已按 seed 固定（可复现）；可用 config seed 改。\n",
                    encoding="utf-8",
                )
                files.append("dml_summary.txt")
                summary.append(
                    f"{entry.method} 完成（doubleml {mlabel}，RF×{n_folds}折，seed={seed}）：处理 {treatment} → {outcome}，"
                    f"控制 {len(controls)} 协变量；{est_word}={ate:.4f}（95% CI [{lb:.4f}, {ub:.4f}]，{sig}，p={p:.3g}）。"
                    + enc_txt + "⚠ 因果解释依赖无未观测混杂 + 重叠假定；≠净相关。" + overlap_txt
                )
                code += [
                    "import doubleml as dml  # 双重/去偏机器学习",
                    f"# DoubleML{kind}(DoubleMLData(y={outcome!r}, d={treatment!r}, x=controls), RF, n_folds={n_folds}).fit()",
                ]
            except Exception as err:
                summary.append(f"双重机器学习拟合失败：{err}")

    elif entry.id == "rdd":
        import importlib.util

        import pandas as pd

        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
        running = cfg.get("running") or cfg.get("running_var")
        outcome = cfg["outcome"] if cfg.get("outcome") in cont else next((c for c in cont if c != running), None)
        try:
            cutoff = float(cfg.get("cutoff", 0.0))
        except (TypeError, ValueError):
            cutoff = None
        if importlib.util.find_spec("rdrobust") is None:
            summary.append("RDD 需要 rdrobust 包（未检测到）。安装：pip install rdrobust；或用 did（面板双重差分）。")
        elif not running or running not in df.columns:
            summary.append(
                "RDD 失败：需用 config 指定 running（驱动/分配变量），例 "
                "config={\"running\":\"score\",\"cutoff\":60}。RDD 是设计驱动的，引擎无法自动猜驱动变量。"
            )
        elif outcome is None:
            summary.append("RDD 失败：需要一个连续结果变量（≠ running）。可用 config['outcome'] 指定。")
        elif cutoff is None:
            summary.append("RDD 失败：cutoff 需为数值。")
        else:
            rv = pd.to_numeric(df[running], errors="coerce").dropna()
            if rv.empty or not (rv.min() < cutoff < rv.max()):
                summary.append(
                    f"RDD 失败：cutoff={cutoff:g} 不在 {running} 取值范围内"
                    f"（{rv.min():.3g}–{rv.max():.3g}）；用 config['cutoff'] 指定正确断点。"
                )
            else:
                try:
                    res = _rdd_via_rdrobust(df, outcome, running, cutoff, d / "rdd_plot.png")
                    if (d / "rdd_plot.png").exists():
                        files.append("rdd_plot.png")
                    rd, lb, ub = res["rd_robust"], res["ci_lb"], res["ci_ub"]
                    p, h = res["p_robust"], res["bw_h"]
                    nl, nr = res["n_left"], res["n_right"]
                    pd.DataFrame(
                        {
                            "estimator": ["Conventional", "Robust (bias-corrected)"],
                            "rd_effect": [round(res["rd_conventional"], 4), round(rd, 4)],
                            "ci_lower": ["", round(lb, 4)],
                            "ci_upper": ["", round(ub, 4)],
                            "p_value": ["", round(p, 4)],
                        }
                    ).to_csv(d / "rdd_estimate.csv", index=False, encoding="utf-8")
                    files.append("rdd_estimate.csv")
                    estimates["rd_effect"] = round(rd, 4)
                    estimates["ci_lb"] = round(lb, 4)
                    estimates["ci_ub"] = round(ub, 4)
                    estimates["p_value"] = round(p, 4)
                    estimates["bandwidth"] = round(h, 4)
                    estimates["n_effective"] = float(nl + nr)
                    sig = "显著" if p < 0.05 else "不显著"
                    (d / "rdd_summary.txt").write_text(
                        f"断点回归 RDD（rdrobust，sharp，{res['bwselect']} 最优带宽）\n"
                        f"驱动变量 {running} @ 断点 {cutoff:g}，结果 {outcome}\n"
                        f"断点处跳跃（局部处理效应 LATE）= {rd:.4f}，稳健 95% CI [{lb:.4f}, {ub:.4f}]，p={p:.4g}\n"
                        f"MSE 最优带宽 h={h:.4f}；带内有效样本 左 {nl} / 右 {nr}\n"
                        "注：RDD 只识别断点局部的处理效应（外推性有限）；这是 sharp 设计，"
                        "假定断点处处理概率从 0 跳到 1（完全合规）——若实际为模糊断点（take-up<100%）"
                        "本估计有偏，需 fuzzy RDD（未启用）；并依赖潜在结果均值在断点连续、"
                        "无其他政策/混杂同址跳变、个体无法精确操纵驱动变量（应做 McCrary 密度检验）。\n",
                        encoding="utf-8",
                    )
                    files.append("rdd_summary.txt")
                    summary.append(
                        f"{entry.method} 完成（rdrobust，sharp）：{running}@{cutoff:g} 处 {outcome} "
                        f"跳跃 = {rd:.4f}（稳健 95% CI [{lb:.4f}, {ub:.4f}]，{sig}，p={p:.3g}）；"
                        f"最优带宽 h={h:.3f}，带内有效样本 {nl}+{nr}。"
                        "⚠ 仅识别断点局部效应（外推有限）；sharp 设计（假定断点处处理概率 0→1、"
                        "完全合规，模糊断点需 fuzzy RDD，未启用）；假定潜在结果均值在断点连续、"
                        "无其他政策/混杂同址跳变、驱动变量不可精确操纵（建议 McCrary 密度检验）。"
                    )
                    code += [
                        "from rdrobust import rdrobust  # 断点回归(局部线性, 稳健偏差校正)",
                        f"# rdrobust(y={outcome!r}, x={running!r}, c={cutoff!r}); MSE 最优带宽 + robust CI",
                    ]
                except Exception as err:
                    summary.append(f"RDD 拟合失败：{err}")

    elif entry.id == "synthetic_control":
        unit, time = fp.unit_col, fp.time_col
        _excl = {unit, time, *fp.treatment_candidates}
        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        outcome = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
        # treated unit + treatment time: config, else derive from a treatment 0/1 column
        treated = cfg.get("treated_unit")
        treat_time = cfg.get("treatment_time")
        ever_treated: set = set()  # ALL units ever treated → excluded from donor pool
        if unit and time and fp.treatment_candidates:
            tcol = next((c for c in fp.treatment_candidates if c in df.columns), None)
            if tcol is not None:
                trows = df[df[tcol] == 1]
                if len(trows):
                    ever_treated = set(trows[unit].dropna().unique())
                    if treated is None:
                        treated = trows[unit].dropna().unique()[0]
                    if treat_time is None:
                        treat_time = df[(df[unit] == treated) & (df[tcol] == 1)][time].min()
        # coerce a JSON-supplied treatment_time to the time column's dtype
        if treat_time is not None and time and time in df.columns:
            try:
                treat_time = type(df[time].dropna().iloc[0])(treat_time)
            except Exception:
                pass
        forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in {outcome, unit, time}]
        preds = forced or [c for c in cont if c != outcome][:5]
        if not (unit and time):
            summary.append("合成控制失败：需要面板数据（单位列 + 时间列）。")
        elif outcome is None:
            summary.append("合成控制失败：需要一个连续结果变量。")
        elif treated is None or treat_time is None:
            summary.append(
                "合成控制失败：需指定干预单位与干预时点。"
                "用 config={\"treated_unit\": \"<单位>\", \"treatment_time\": <时点>}，"
                "或在数据中提供处理指示列（0/1）。"
            )
        else:
            try:
                import pysyncon  # noqa: F401
            except ImportError:
                summary.append("合成控制需要 pysyncon 包（未检测到）。安装：pip install pysyncon；或用 did。")
            else:
                try:
                    import pandas as pd

                    # exclude OTHER ever-treated units from donors (contamination bias).
                    other_treated = {u for u in ever_treated if u != treated}
                    weights, att, pre_rmspe, n_don, post = _synthetic_control(
                        df, unit, time, outcome, treated, treat_time, preds, d / "gaps.png",
                        exclude=other_treated,
                    )
                    if len(ever_treated) > 1:
                        contam_note = (
                            f"；⚠ 检测到 {len(ever_treated)} 个被处理单位（疑交错采纳），"
                            f"仅对 {treated} 建模、其余已剔出供体池；交错处理建议改用 did"
                        )
                    elif not ever_treated:
                        contam_note = "；⚠ 由 config 指定处理单位、未据处理列校验供体，假定供体未受干预"
                    else:
                        contam_note = ""
                    wdf = weights.reset_index()
                    wdf.columns = ["donor", "weight"]
                    wdf = wdf[wdf["weight"] > 1e-4].reset_index(drop=True)
                    wdf.to_csv(d / "donor_weights.csv", index=False, encoding="utf-8")
                    files.append("donor_weights.csv")
                    if (d / "gaps.png").exists():
                        files.append("gaps.png")
                    att_val = float(att.get("att", float("nan")))
                    att_se = float(att.get("se", float("nan")))
                    estimates["att"] = round(att_val, 4)
                    estimates["att_se"] = round(att_se, 4)
                    estimates["pre_rmspe"] = round(pre_rmspe, 4)
                    estimates["n_donors_used"] = float(len(wdf))
                    top = wdf.head(5)
                    top_txt = ", ".join(f"{r.donor}={r.weight:.2f}" for r in top.itertuples())
                    (d / "synth_summary.txt").write_text(
                        f"合成控制法（Abadie）：干预单位 {treated}，干预时点 {treat_time}，"
                        f"结果变量 {outcome}\n"
                        f"ATT（干预后平均处理效应）= {att_val:.4f}（SE≈{att_se:.4f}）\n"
                        f"干预前拟合 RMSPE = {pre_rmspe:.4f}（越小=合成体越贴合真实前期路径）\n"
                        f"合成体权重（前 5）：{top_txt}\n"
                        f"对照供体池 {n_don} 个，实际赋权 {len(wdf)} 个。\n"
                        "注：ATT 可信度取决于干预前拟合好坏（RMSPE 小）；推断需做安慰剂检验。\n\n"
                        + wdf.to_string(index=False),
                        encoding="utf-8",
                    )
                    files.append("synth_summary.txt")
                    summary.append(
                        f"{entry.method} 完成（pysyncon）：干预单位 {treated} @ {treat_time}，"
                        f"结果 {outcome}；ATT={att_val:.4f}（SE≈{att_se:.4f}）；"
                        f"干预前 RMSPE={pre_rmspe:.4f}；合成体由 {len(wdf)} 个供体加权（{top_txt}）。"
                        "⚠ ATT 可信度依赖干预前拟合；正式推断需安慰剂检验（in-space/in-time placebo）。"
                        + contam_note
                    )
                    code += [
                        "from pysyncon import Dataprep, Synth  # 合成控制法",
                        f"# 干预单位={treated}, 干预时点={treat_time}; 干预前拟合权重 -> 后期 gap=ATT",
                    ]
                except Exception as err:
                    summary.append(f"合成控制拟合失败：{err}")

    elif entry.id == "glmm":
        import re

        from researchforge.executor import rbridge

        _excl = {fp.unit_col, fp.time_col}
        binary = [c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl]
        counts = [c.name for c in fp.columns if c.kind == "count" and c.name not in _excl]
        # outcome + family: config outcome (kind decides family), else binary->binomial,
        # else count->poisson. GLMM is the non-Gaussian complement of mixed_effects.
        cfg_out = cfg.get("outcome")
        if cfg_out in binary:
            outcome, family = cfg_out, "binomial"
        elif cfg_out in counts:
            outcome, family = cfg_out, "poisson"
        elif binary:
            outcome, family = binary[0], "binomial"
        elif counts:
            outcome, family = counts[0], "poisson"
        else:
            outcome, family = None, None
        # grouping for the random intercept: config group, else panel unit, else a
        # categorical/id column with real clustering (2..n-1 distinct values).
        group = cfg.get("group") or fp.unit_col
        if not group:
            group = next(
                (
                    c.name for c in fp.columns
                    if c.kind in {"categorical", "id"} and c.name not in _excl
                    # >=5 groups for a stable variance component (Opus); config can override
                    and 5 <= c.n_unique < fp.n_rows
                ),
                None,
            )
        forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c not in {outcome, group}]
        if forced:
            preds = forced[:6]
        else:
            preds = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "count", "binary"}
                and c.name not in {outcome, group, fp.unit_col, fp.time_col}
            ][:5]
        names_safe = outcome and group and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [outcome, group, *preds]
        )
        if outcome is None:
            summary.append(
                "GLMM 失败：需要二值或计数型结果变量（高斯/连续结果用 mixed_effects 线性混合模型）。"
            )
        elif group is None:
            summary.append("GLMM 失败：需要一个分组变量做随机截距（面板单位列或重复出现的类别列）。")
        elif not preds:
            summary.append("GLMM 失败：需要 ≥1 个固定效应预测变量。")
        elif not (rbridge.r_available() and rbridge.r_package_available("lme4")):
            summary.append("GLMM 需要 R 的 lme4 包（未检测到）。安装：install.packages('lme4')；二值结果可先用 logistic_regression。")
        elif not names_safe:
            summary.append("GLMM 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
        else:
            import math

            import pandas as pd

            sub = df[[outcome, group, *preds]].dropna()
            csv = d / "_glmm_input.csv"
            sub.to_csv(csv, index=False)
            try:
                fixed, re_d = _glmm_via_r(csv, outcome, preds, group, family)
                link = "logit（二值, 系数 exp→OR）" if family == "binomial" else "log（计数, 系数 exp→IRR）"
                fixed["exp_coef"] = fixed["estimate"].map(lambda b: round(math.exp(b), 4))
                fixed.to_csv(d / "fixed_effects.csv", index=False, encoding="utf-8")
                files.append("fixed_effects.csv")
                gv, gsd = re_d["group_var"], re_d["group_sd"]
                ng, nobs = int(re_d["n_groups"]), int(re_d["n_obs"])
                aic, od = re_d["aic"], re_d.get("overdispersion", float("nan"))
                estimates["group_intercept_var"] = round(gv, 4)
                estimates["n_groups"] = float(ng)
                estimates["aic"] = round(aic, 2)
                for _, r in fixed.iterrows():
                    if "Intercept" not in str(r["term"]):
                        estimates[str(r["term"])] = round(float(r["estimate"]), 4)
                icc_txt = ""
                if "icc" in re_d:
                    estimates["icc"] = round(re_d["icc"], 4)
                    icc_txt = (
                        f"，组内相关 ICC={re_d['icc']:.3f}（潜变量 logit 尺度，"
                        f"{re_d['icc']:.0%} 方差在组间；观测 0/1 尺度通常更低）"
                    )
                od_txt = ""
                if family == "poisson" and od == od:  # NaN-safe
                    estimates["overdispersion"] = round(od, 3)
                    # ratio & residual df are approximate for a mixed model (Bolker GLMM-FAQ)
                    od_txt = (
                        f"；⚠ 过离散≈{od:.2f}>1.5（近似），泊松假定可能被违反，考虑负二项 GLMM"
                        if od > 1.5 else f"；过离散≈{od:.2f}（近似，≈1 可接受）"
                    )
                sig = [str(r["term"]) for _, r in fixed.iterrows()
                       if "Intercept" not in str(r["term"]) and float(r["p_value"]) < 0.05]
                (d / "glmm_summary.txt").write_text(
                    f"广义线性混合模型 GLMM（lme4 glmer，{family}，{link}）\n"
                    f"{outcome} ~ {' + '.join(preds)} + (1|{group})\n"
                    f"随机截距方差={gv:.4f}（SD={gsd:.4f}），分组数={ng}，n={nobs}，AIC={aic:.1f}{icc_txt}{od_txt}\n"
                    f"显著固定效应（p<0.05）：{sig}\n"
                    "exp(系数)=OR(二值)/IRR(计数)；随机截距吸收组间基线差异。\n\n"
                    + fixed.to_string(index=False),
                    encoding="utf-8",
                )
                files.append("glmm_summary.txt")
                summary.append(
                    f"{entry.method} 完成（R/lme4，{family}）：{outcome} ~ {len(preds)} 个固定效应 "
                    f"+ (1|{group})；随机截距方差={gv:.3f}（{ng} 组，n={nobs}）{icc_txt}；"
                    f"显著项 {sig}{od_txt}。⚠ exp(系数)=OR/IRR；随机截距假定组效应~正态。"
                )
                code += [
                    "library(lme4)  # 广义线性混合模型 GLMM",
                    f"# glmer({outcome} ~ {' + '.join(preds)} + (1|{group}), family={family})",
                ]
            except Exception as err:
                summary.append(f"GLMM 拟合失败（可能不收敛）：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "gam":
        import re

        from researchforge.executor import rbridge

        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}]
        y = cfg["outcome"] if cfg.get("outcome") in cont else (cont[0] if cont else None)
        forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y]
        if forced:
            preds = forced[:8]
        else:
            preds = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "count", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
            ][:6]
        # smooth continuous predictors with enough distinct values (mgcv s() needs
        # ~>=10 unique points); binary / low-cardinality enter as parametric terms.
        smooth = [p for p in preds if p in cont and df[p].dropna().nunique() >= 10]
        linear = [p for p in preds if p not in smooth]
        names_safe = y is not None and all(
            re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in [y, *preds]
        )
        if y is None or not preds:
            summary.append("GAM 失败：需要 1 个连续结果变量 + ≥1 个预测变量。")
        elif not smooth:
            summary.append(
                "GAM 跳过：没有可平滑的连续预测变量（需 ≥10 个不同取值）。"
                "用 ols_regression（线性）或确认预测变量为连续型。"
            )
        elif not (rbridge.r_available() and rbridge.r_package_available("mgcv")):
            summary.append("GAM 需要 R 的 mgcv 包（未检测到）。安装：install.packages('mgcv')；或用 ols_regression。")
        elif not names_safe:
            summary.append("GAM 失败：列名需为标识符式（字母/数字/. _），R 公式要求。")
        else:
            import pandas as pd

            sub = df[[y, *preds]].dropna()
            csv = d / "_gam_input.csv"
            sub.to_csv(csv, index=False)
            try:
                smooth_df, param_df, fit = _gam_via_r(csv, y, smooth, linear, d / "gam_smooths.png")
                if len(smooth_df):
                    smooth_df.to_csv(d / "smooth_terms.csv", index=False, encoding="utf-8")
                    files.append("smooth_terms.csv")
                if len(param_df):
                    param_df.to_csv(d / "parametric_terms.csv", index=False, encoding="utf-8")
                    files.append("parametric_terms.csv")
                if (d / "gam_smooths.png").exists():
                    files.append("gam_smooths.png")
                dev, r2, n = fit["dev_expl"], fit["r_sq"], int(fit["n"])
                estimates["deviance_explained"] = round(dev, 4)
                estimates["adj_r_squared"] = round(r2, 4)
                estimates["n"] = float(n)
                # edf>~1 means the term bends away from a straight line (nonlinear)
                nonlin = []
                for _, r in smooth_df.iterrows():
                    estimates[f"edf_{r['term']}"] = round(float(r["edf"]), 3)
                    if float(r["edf"]) > 1.5 and float(r["p_value"]) < 0.05:
                        nonlin.append(str(r["term"]))
                nl_txt = (
                    f"显著非线性项：{nonlin}（edf>1.5 且 p<0.05，提示曲线关系）"
                    if nonlin
                    else "各平滑项 edf≈1 或不显著（近线性，可考虑普通回归）"
                )
                cc_txt = ""
                if "concurvity" in fit:
                    estimates["worst_concurvity"] = round(fit["concurvity"], 3)
                    if fit["concurvity"] > 0.8:
                        cc_txt = (
                            f"；⚠ 最差 concurvity={fit['concurvity']:.2f}>0.8（平滑项间强非线性共线），"
                            "单项偏效应/显著性不稳"
                        )
                sig_s = [str(r["term"]) for _, r in smooth_df.iterrows() if float(r["p_value"]) < 0.05]
                (d / "gam_summary.txt").write_text(
                    f"广义可加模型 GAM（mgcv，REML）：{y} ~ "
                    + " + ".join([f"s({t})" for t in smooth] + linear) + "\n"
                    f"偏差解释 {dev:.1%}，调整 R² {r2:.3f}，n={n}\n"
                    f"显著平滑项（p<0.05）：{sig_s}\n{nl_txt}\n"
                    "edf=有效自由度（1=直线，越大越弯）；平滑项 p 检验该项整体是否≠0（非"
                    "「是否非线性」的正式检验）——edf>1.5 仅为非线性的描述性标记。\n"
                    "默认高斯族+identity link，假定结果连续无界、近似同方差正态。\n\n"
                    "平滑项：\n" + (smooth_df.to_string(index=False) if len(smooth_df) else "（无）")
                    + "\n\n参数项：\n" + (param_df.to_string(index=False) if len(param_df) else "（无）"),
                    encoding="utf-8",
                )
                files.append("gam_summary.txt")
                summary.append(
                    f"{entry.method} 完成（R/mgcv，REML）：{y} ~ {len(smooth)} 个平滑项 + "
                    f"{len(linear)} 个线性项；偏差解释 {dev:.1%}，调整 R²={r2:.3f}（n={n}）；{nl_txt}{cc_txt}。"
                    "⚠ 高斯族(连续无界结果)；平滑项默认薄板样条 k=10，边界/稀疏区外推不可靠；偏效应图 gam_smooths.png。"
                )
                _rhs_preview = " + ".join([f"s({t})" for t in smooth] + linear)
                code += [
                    "library(mgcv)  # 广义可加模型 GAM",
                    f"# gam({y} ~ {_rhs_preview}, method='REML'); summary()/plot(shade=TRUE)",
                ]
            except Exception as err:
                summary.append(f"GAM 拟合失败：{err}")
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

    elif entry.id == "meta_regression":
        import re

        import pandas as pd

        from researchforge.executor import rbridge

        low = {c.name.lower(): c.name for c in fp.columns}

        def _pick(*names):
            return next((low[n] for n in names if n in low), None)

        measure = str(cfg.get("measure") or "").upper()
        method = str(cfg.get("method") or "REML").upper()
        study = cfg.get("study") or _pick("study", "study_id", "studyid", "label", "author", "trial", "name", "id", "source")
        yi = cfg.get("effect") or _pick("yi", "effect", "es", "effect_size", "smd", "logor", "d", "g", "lnrr")
        vi = cfg.get("variance") or _pick("vi", "var", "variance", "v", "samp_var")
        sei = cfg.get("se") or _pick("sei", "se", "std_err", "stderr", "se_effect")
        m1, sd1, n1 = _pick("m1", "m1i", "mean1", "mean_t", "mt"), _pick("sd1", "sd1i", "sd_t", "sdt"), _pick("n1", "n1i", "nt", "n_t")
        m2, sd2, n2 = _pick("m2", "m2i", "mean2", "mean_c", "mc"), _pick("sd2", "sd2i", "sd_c", "sdc"), _pick("n2", "n2i", "nc", "n_c")
        ai, bi, ci_, di = _pick("ai", "events1", "a"), _pick("bi", "b"), _pick("ci", "events2", "c"), _pick("di", "d")
        roles, es_cols = {}, []
        if measure in ("", "GEN") and yi and (vi or sei):
            measure, roles, es_cols = "GEN", {"yi": yi, "vi": vi, "sei": sei}, [yi, vi or sei]
        elif measure in ("", "SMD", "MD") and all([m1, sd1, n1, m2, sd2, n2]):
            measure = measure if measure in ("SMD", "MD") else "SMD"
            roles, es_cols = {"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2}, [m1, sd1, n1, m2, sd2, n2]
        elif measure in ("", "OR", "RR", "RD") and all([ai, bi, ci_, di]):
            measure = measure if measure in ("OR", "RR", "RD") else "OR"
            roles, es_cols = {"ai": ai, "bi": bi, "ci": ci_, "di": di}, [ai, bi, ci_, di]
        # moderators: config, else numeric/categorical columns not used as effect-size/study.
        # exclude ALL resolved role columns (consumed or not) so a leftover precision column
        # (e.g. an unused sei when vi was picked) can't sneak in as a moderator (Opus catch).
        used = {c for c in (yi, vi, sei, m1, sd1, n1, m2, sd2, n2, ai, bi, ci_, di, study) if c}
        forced_mods = [c for c in (cfg.get("moderators") or cfg.get("predictors") or []) if c in df.columns and c not in used]
        if forced_mods:
            moderators = forced_mods[:5]
        else:
            moderators = [
                c.name for c in fp.columns
                if c.kind in {"continuous", "count", "binary", "categorical"} and c.name not in used
                and c.name not in {fp.unit_col, fp.time_col}
            ][:4]
        # drop constant moderators — metafor silently drops them and would mislabel the
        # omnibus as a moderator test (Opus catch); filter in Python first.
        const_mods = [m for m in moderators if df[m].dropna().nunique() <= 1]
        moderators = [m for m in moderators if m not in const_mods]
        _num_kind = {c.name: c.kind for c in fp.columns}
        bubble_mod = next((m for m in moderators if _num_kind.get(m) in {"continuous", "count"}), None)
        cols_all = [c for c in [*es_cols, study, *moderators] if c]
        names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in moderators) if moderators else False
        if not roles:
            summary.append(
                "Meta 回归失败：未识别到效应量数据（需 yi+vi/sei，或两组 m/sd/n，或 2×2 ai/bi/ci/di）。详见 docs/meta-analysis.md。"
            )
        elif not moderators:
            summary.append("Meta 回归失败：需要 ≥1 个研究层调节变量（解释异质性的协变量）。用 config['moderators'] 指定。")
        elif not (rbridge.r_available() and rbridge.r_package_available("metafor")):
            summary.append("Meta 回归需要 R 的 metafor 包（未检测到）。安装：install.packages('metafor')。")
        elif not names_safe:
            summary.append("Meta 回归失败：调节变量列名需为标识符式（字母/数字/. _），R 公式要求。")
        else:
            sub = df[cols_all].dropna()
            if len(sub) < len(moderators) + 3:
                summary.append(f"Meta 回归失败：有效研究数 {len(sub)} 太少（需 > 调节变量数 {len(moderators)}+2）。")
            else:
                csv = d / "_metareg_input.csv"
                sub.to_csv(csv, index=False)
                try:
                    coef, st = _meta_regression_via_r(
                        csv, measure=measure, roles=roles, moderators=moderators,
                        study=study, method=method, bubble_png=d / "bubble.png", bubble_mod=bubble_mod,
                    )
                    surviving = st.get("_surviving", [])
                    if not surviving:
                        raise RuntimeError("所有调节变量被 metafor 剔除（常量/完全共线），无可解释的调节项")
                    coef.to_csv(d / "meta_regression_coef.csv", index=False, encoding="utf-8")
                    files.append("meta_regression_coef.csv")
                    if (d / "bubble.png").exists():
                        files.append("bubble.png")
                    qm, qmp = st["QM"], st["QMp"]
                    r2 = float("nan") if method == "FE" else st.get("R2", float("nan"))
                    i2r, k = st["I2_resid"], int(st["k"])
                    # disclose moderators dropped as constant (Python) or collinear (metafor)
                    drop_note = f"；⚠ 常量调节变量已剔除：{const_mods}" if const_mods else ""
                    if len(surviving) < len(moderators):
                        drop_note += "；⚠ 部分调节变量因共线被 metafor 剔除（见系数表实际项）"
                    estimates["QM_pvalue"] = round(qmp, 4)
                    estimates["R2_percent"] = round(r2, 2) if r2 == r2 else float("nan")
                    estimates["I2_residual"] = round(i2r, 2)
                    estimates["k_studies"] = float(k)
                    for _, r in coef.iterrows():
                        if r["term"] != "intrcpt":
                            estimates[f"beta_{r['term']}"] = round(float(r["estimate"]), 4)
                    sig_mods = [r["term"] for _, r in coef.iterrows() if r["term"] != "intrcpt" and float(r["p_value"]) < 0.05]
                    r2_txt = f"，调节变量解释了 {r2:.0f}% 的研究间异质性" if r2 == r2 else ""
                    (d / "meta_regression.txt").write_text(
                        f"Meta 回归（metafor rma，{method}，k={k} 研究，measure={measure}）\n"
                        f"调节变量：{moderators}\n"
                        f"omnibus 检验 QM={qm:.3f}，p={qmp:.4g}"
                        f"（{'调节变量整体显著' if qmp < 0.05 else '调节变量整体不显著'}）\n"
                        f"残差异质性 I²={i2r:.1f}%{r2_txt}\n"
                        f"显著调节变量（p<0.05）：{sig_mods}\n"
                        "注：meta 回归是观察性的（研究层混杂、生态谬误风险）；"
                        "调节变量少、研究数少时易过拟合/假阳性（建议每 10 项研究 ≤1 个调节变量）。\n\n"
                        + coef.to_string(index=False),
                        encoding="utf-8",
                    )
                    files.append("meta_regression.txt")
                    summary.append(
                        f"{entry.method} 完成（R/metafor，{method}）：合并 {k} 项研究、调节变量 {moderators}；"
                        f"omnibus p={qmp:.3g}（{'整体显著' if qmp < 0.05 else '整体不显著'}）"
                        f"{r2_txt}；残差 I²={i2r:.1f}%；显著项 {sig_mods}。"
                        "⚠ 研究层观察性关联（非个体因果，慎防生态谬误）；研究少易过拟合。" + drop_note
                    )
                    code += [
                        "library(metafor)  # meta 回归（调节变量解释异质性）",
                        f"# rma(yi, vi, mods = ~ {' + '.join(moderators)}, method='{method}'); QM 检验 + R²",
                    ]
                except Exception as err:
                    summary.append(f"Meta 回归失败：{err}")
                finally:
                    try:
                        csv.unlink()
                    except OSError:
                        pass

    elif entry.id == "meta_analysis":
        import pandas as pd

        from researchforge.executor import rbridge

        low = {c.name.lower(): c.name for c in fp.columns}

        def _pick(*names):
            return next((low[n] for n in names if n in low), None)

        # role resolution: config overrides, else detect by column name. Three
        # input formats — pre-computed effect sizes, raw two-group means, 2x2 counts.
        measure = str(cfg.get("measure") or "").upper()
        method = str(cfg.get("method") or "REML").upper()  # REML/DL random, FE fixed
        study_col = cfg.get("study") or _pick(
            "study", "study_id", "studyid", "label", "author", "trial", "name", "id", "source"
        )
        yi = cfg.get("effect") or _pick("yi", "effect", "es", "effect_size", "smd", "logor", "d", "g", "lnrr")
        vi = cfg.get("variance") or _pick("vi", "var", "variance", "v", "samp_var")
        sei = cfg.get("se") or _pick("sei", "se", "std_err", "stderr", "se_effect")
        m1, sd1, n1 = _pick("m1", "m1i", "mean1", "mean_t", "mt"), _pick("sd1", "sd1i", "sd_t", "sdt"), _pick("n1", "n1i", "nt", "n_t")
        m2, sd2, n2 = _pick("m2", "m2i", "mean2", "mean_c", "mc"), _pick("sd2", "sd2i", "sd_c", "sdc"), _pick("n2", "n2i", "nc", "n_c")
        ai, bi, ci_, di = _pick("ai", "events1", "a"), _pick("bi", "b"), _pick("ci", "events2", "c"), _pick("di", "d")

        roles, used = {}, []
        if measure in ("", "GEN") and yi and (vi or sei):
            measure = "GEN"
            roles = {"yi": yi, "vi": vi, "sei": sei}
            used = [yi, vi or sei]
        elif measure in ("", "SMD", "MD") and all([m1, sd1, n1, m2, sd2, n2]):
            measure = measure if measure in ("SMD", "MD") else "SMD"
            roles = {"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2}
            used = [m1, sd1, n1, m2, sd2, n2]
        elif measure in ("", "OR", "RR", "RD") and all([ai, bi, ci_, di]):
            measure = measure if measure in ("OR", "RR", "RD") else "OR"
            roles = {"ai": ai, "bi": bi, "ci": ci_, "di": di}
            used = [ai, bi, ci_, di]

        cols_all = [c for c in [*used, study_col] if c]
        names_safe = all(('"' not in c and "]" not in c and "\\" not in c) for c in cols_all)
        if not roles:
            summary.append(
                "Meta 分析失败：未识别到效应量数据。需以下任一格式（列名可用 config 指定）："
                "① 预算效应量 yi + 方差 vi（或标准误 sei）；"
                "② 两组原始均值 m1,sd1,n1,m2,sd2,n2（→标准化均差 SMD）；"
                "③ 2×2 计数 ai,bi,ci,di（→比值比 OR）。详见 docs/meta-analysis.md。"
            )
        elif not (rbridge.r_available() and rbridge.r_package_available("metafor")):
            summary.append(
                "Meta 分析需要 R 的 metafor 包（未检测到）。安装：install.packages('metafor')。"
            )
        elif not names_safe:
            summary.append("Meta 分析失败：相关列名含特殊字符（\" ] \\），请重命名。")
        else:
            sub = df[cols_all].dropna()
            if len(sub) < 2:
                summary.append("Meta 分析失败：有效研究数 <2，无法合并。")
            else:
                csv = d / "_meta_input.csv"
                sub.to_csv(csv, index=False)
                try:
                    meta, study = _meta_via_r(
                        csv, measure=measure, roles=roles, study_col=study_col,
                        method=method, forest_png=d / "forest.png", funnel_png=d / "funnel.png",
                    )
                    study.to_csv(d / "study_effects.csv", index=False, encoding="utf-8")
                    files.append("study_effects.csv")
                    for png in ("forest.png", "funnel.png"):
                        if (d / png).exists():
                            files.append(png)
                    import math

                    est, lb, ub = meta["estimate"], meta["ci_lb"], meta["ci_ub"]
                    i2, tau2, k = meta["I2"], meta["tau2"], int(meta["k"])
                    qp, pval = meta["Qp"], meta["pval"]
                    # OR/RR are pooled on the log scale -> exp() the WHOLE interval for
                    # display (point AND CI), not just the point estimate.
                    log_scale = measure in ("OR", "RR")
                    if log_scale:
                        de, dlb, dub = math.exp(est), math.exp(lb), math.exp(ub)
                        shown_est = f"{de:.3f}（{measure}, 由 log 尺度还原）"
                        ci_str = f"[{dlb:.3f}, {dub:.3f}]"
                        estimates["pooled_effect"] = round(de, 4)
                        estimates["ci_lb"] = round(dlb, 4)
                        estimates["ci_ub"] = round(dub, 4)
                        estimates["pooled_log_effect"] = round(est, 4)
                    else:
                        shown_est = f"{est:.3f}"
                        ci_str = f"[{lb:.3f}, {ub:.3f}]"
                        estimates["pooled_effect"] = round(est, 4)
                        estimates["ci_lb"] = round(lb, 4)
                        estimates["ci_ub"] = round(ub, 4)
                    estimates["I2_percent"] = round(i2, 2)
                    estimates["tau2"] = round(tau2, 4)
                    estimates["k_studies"] = float(k)
                    het = "高" if i2 >= 75 else ("中" if i2 >= 50 else ("低" if i2 >= 25 else "极低"))
                    is_fe = method == "FE"
                    eg_txt = ""
                    if "egger_p" in meta:
                        estimates["egger_p"] = round(meta["egger_p"], 4)
                        if k < 10:
                            # Egger / funnel-asymmetry tests are underpowered at k<10
                            # (Cochrane) — report but flag, don't over-reassure (Opus catch).
                            eg_txt = f"；Egger 检验 p={meta['egger_p']:.3g}（⚠ k<10，偏倚检验功效不足、不可靠）"
                        else:
                            eg_txt = (
                                f"；Egger 检验 p={meta['egger_p']:.3g}"
                                f"（{'⚠ 漏斗图不对称、可能有发表偏倚' if meta['egger_p'] < 0.05 else '未见明显不对称'}）"
                            )
                    # under fixed-effect, tau2/I2 are 0 by assumption (not estimated) —
                    # the Q test is the meaningful heterogeneity signal (Opus catch).
                    het_txt = (
                        f"（固定效应假定同质，I²/τ² 不估计；看 Q 检验 p={qp:.3g}）"
                        if is_fe
                        else f"异质性 I²={i2:.1f}%（{het}）、τ²={tau2:.4f}"
                    )
                    mlabel = {"FE": "固定效应", "DL": "随机效应(DL)"}.get(method, "随机效应(REML)")
                    (d / "meta_summary.txt").write_text(
                        f"Meta 分析（metafor，{mlabel}，measure={measure}，k={k} 研究）\n"
                        f"合并效应 = {shown_est}，95% CI {ci_str}，p={pval:.4g}\n"
                        f"异质性：I²={i2:.1f}%（{het}），τ²={tau2:.4f}，Q 检验 p={qp:.4g}\n"
                        f"{'森林图 forest.png、漏斗图 funnel.png' }\n"
                        "注：I²>50% 提示研究间异质性较大，合并需谨慎、宜探究调节变量；"
                        "随机效应不假定各研究共享同一真效应。\n",
                        encoding="utf-8",
                    )
                    files.append("meta_summary.txt")
                    sig = "显著" if pval < 0.05 else "不显著"
                    summary.append(
                        f"{entry.method} 完成（R/metafor，{mlabel}）：合并 {k} 项研究，"
                        f"合并效应={shown_est}，95% CI {ci_str}（{sig}，p={pval:.3g}）；"
                        f"{het_txt}{eg_txt}。"
                        "⚠ I²>50% 宜查调节变量；合并不能修正原始研究的偏倚。"
                    )
                    code += [
                        "library(metafor)  # 随机效应 meta 分析",
                        f"# escalc(measure='{measure}', ...) → rma(yi, vi, method='{method}'); forest()/funnel()/regtest()",
                    ]
                except Exception as err:
                    summary.append(f"Meta 分析失败：{err}")
                finally:
                    try:
                        csv.unlink()
                    except OSError:
                        pass

    elif entry.id == "differential_abundance":
        import numpy as np
        import pandas as pd
        from scipy.stats import mannwhitneyu
        from statsmodels.stats.multitest import multipletests

        _excl = {fp.unit_col, fp.time_col}
        taxa = [c.name for c in fp.columns if c.kind == "count" and c.name not in _excl]
        group_col = next(
            (
                c.name
                for c in fp.columns
                if c.kind in {"binary", "categorical"}
                and c.name not in _excl
                and df[c.name].dropna().nunique() == 2
            ),
            None,
        )
        if len(taxa) < 2 or group_col is None:
            summary.append("差异丰度失败：需要 ≥2 个计数列（物种/OTU）+ 一个 2 水平分组变量。")
        else:
            sub = df[[*taxa, group_col]].dropna()
            grps = list(pd.Series(sub[group_col].astype(str)).unique())
            if len(grps) != 2:
                summary.append("差异丰度失败：分组变量需恰好 2 组。")
            else:
                # config={"da_method": ...}: "aldex2" (R 金标准, MC-CLR + Welch) /
                # "clr_mw" (默认, CLR+Mann-Whitney) / "clr_welch" (CLR+Welch t)。
                # "ancombc" 桥待接 → 诚实降级。请求 R 法而包/桥不可用也诚实降级。
                from researchforge.executor import rbridge

                da_method = str(cfg.get("da_method") or "clr_mw").lower()
                _degrade_note = ""
                use_aldex2 = False
                ar = None
                if da_method == "aldex2":
                    if rbridge.r_available() and rbridge.r_package_available("ALDEx2"):
                        _csv = d / "_da_input.csv"
                        sub[[*taxa, group_col]].to_csv(_csv, index=False)
                        try:
                            ar = _diff_abundance_aldex2_via_r(_csv, taxa, group_col)
                            use_aldex2 = True
                        except Exception as err:
                            _degrade_note = f"；⚠ ALDEx2 运行失败（{err}），已降级 CLR+Mann-Whitney"
                        finally:
                            try:
                                _csv.unlink()
                            except OSError:
                                pass
                    else:
                        _degrade_note = (
                            "；⚠ 请求 ALDEx2 但未检测到（装：BiocManager::install('ALDEx2')），"
                            "已用 CLR+Mann-Whitney 保底"
                        )
                elif da_method in {"ancombc", "ancom-bc"}:
                    _degrade_note = (
                        "；⚠ ANCOM-BC 专用桥尚未接（API 需 TreeSummarizedExperiment，待接，"
                        "见 loop-decisions），已用 CLR+Mann-Whitney 保底；如需 ALDEx2 请 da_method=aldex2"
                    )
                if not use_aldex2 and da_method not in {"clr_mw", "clr_welch"}:
                    da_method = "clr_mw"  # unknown / degraded → default

                if use_aldex2:
                    method_label = "ALDEx2 (R, MC-CLR + Welch)"
                    effect_col = f"median_CLR_diff_{grps[1]}_vs_{grps[0]}"
                    x_label = f"median CLR difference ({grps[1]} vs {grps[0]})"
                    taxa_out = ar["taxon"].tolist()
                    effect_vals = ar["diff_btw"].to_numpy(dtype=float)
                    pvals = ar["p_value"].to_numpy(dtype=float)
                    qvals = ar["q_value"].to_numpy(dtype=float)
                else:
                    use_welch = da_method == "clr_welch"
                    method_label = "CLR+Welch t" if use_welch else "CLR+Mann-Whitney"
                    effect_col = f"log2FC_{grps[1]}_vs_{grps[0]}"
                    x_label = f"log2 fold-change ({grps[1]} vs {grps[0]})"
                    taxa_out = taxa
                    mat = sub[taxa].clip(lower=0).to_numpy(dtype=float)
                    rel = mat / mat.sum(axis=1, keepdims=True).clip(min=1e-12)
                    logm = np.log(mat + 0.5)  # CLR (compositional-aware), pseudocount 0.5
                    clr = logm - logm.mean(axis=1, keepdims=True)
                    g = sub[group_col].astype(str).to_numpy()
                    ma, mb = g == grps[0], g == grps[1]
                    pvals, l2fc = [], []
                    if use_welch:
                        from scipy.stats import ttest_ind
                    for j in range(len(taxa)):
                        try:
                            if use_welch:
                                _, p = ttest_ind(clr[ma, j], clr[mb, j], equal_var=False)
                            else:
                                _, p = mannwhitneyu(clr[ma, j], clr[mb, j], alternative="two-sided")
                            if not np.isfinite(p):
                                p = 1.0
                        except ValueError:
                            p = 1.0
                        pvals.append(p)
                        l2fc.append(np.log2((rel[mb, j].mean() + 1e-9) / (rel[ma, j].mean() + 1e-9)))
                    pvals = np.array(pvals)
                    qvals = multipletests(pvals, method="fdr_bh")[1]
                    effect_vals = np.array(l2fc)

                res = pd.DataFrame(
                    {
                        "taxon": taxa_out,
                        effect_col: np.round(effect_vals, 4),
                        "p_value": np.round(pvals, 4),
                        "q_value": np.round(qvals, 4),
                    }
                )
                res["significant"] = res["q_value"] < 0.05
                res = res.sort_values("q_value").reset_index(drop=True)
                res.to_csv(d / "differential_abundance.csv", index=False, encoding="utf-8")
                files.append("differential_abundance.csv")
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fc = np.asarray(effect_vals, dtype=float)
                    nlq = -np.log10(np.clip(qvals, 1e-300, 1.0))
                    sig = np.asarray(qvals) < 0.05
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    ax.scatter(fc[~sig], nlq[~sig], s=18, c="#999999", label="ns")
                    ax.scatter(fc[sig], nlq[sig], s=24, c="#C44E52", label="q<0.05")
                    ax.axhline(-np.log10(0.05), color="grey", ls="--", lw=0.8)
                    ax.axvline(0, color="grey", ls="--", lw=0.6)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("-log10(q)")
                    ax.set_title(f"Differential abundance ({method_label})")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    fig.savefig(d / "volcano.png", dpi=150)
                    plt.close(fig)
                    files.append("volcano.png")
                except Exception:
                    pass
                n_sig = int(res["significant"].sum())
                estimates["n_significant"] = float(n_sig)
                estimates["n_taxa"] = float(len(taxa_out))
                _caveat = (
                    "ALDEx2 用 Monte-Carlo Dirichlet 采样 + CLR，组成性严谨（金标准）。"
                    if use_aldex2
                    else "⚠ 组成性数据：相对丰度受总和约束，本法用 CLR 缓解但非金标准；"
                    "CLR 各物种共享每样本分母、非独立，BH-FDR 的独立性假定略被违反；"
                    "严格分析可 da_method=aldex2（R ALDEx2）。"
                )
                summary.append(
                    f"{entry.method} 完成：{len(taxa_out)} 个物种 × {len(sub)} 样本，比较 "
                    f"{grps[0]} vs {grps[1]}；{n_sig} 个物种丰度差异显著（q<0.05，{method_label}+BH-FDR）。"
                    + _caveat + _degrade_note
                )
                code += [
                    f"# 差异丰度 ({method_label}); config da_method: aldex2 / clr_mw / clr_welch",
                    "# ALDEx2: aldex(counts, conds, test='t', effect=TRUE); 纯Py: CLR + 检验 + BH-FDR",
                ]

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

    elif entry.id == "sem":
        import re

        _excl = {fp.unit_col, fp.time_col}
        indicators = [
            c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl
        ]
        # config={"model_spec": "<lavaan/semopy syntax>"} lets the user supply their
        # theoretical structure (multi-factor CFA / paths) instead of the auto
        # single-factor template. Columns are taken from those named in the spec.
        user_spec = cfg.get("model_spec") or cfg.get("sem_spec")
        if user_spec:
            used = [
                c for c in df.columns
                if re.search(rf"(?<![\w.]){re.escape(str(c))}(?![\w.])", user_spec)
            ]
            spec = user_spec
        else:
            used = indicators[:8]
            spec = "F =~ " + " + ".join(used)
        if not user_spec and len(indicators) < 3:
            summary.append("SEM 失败：需要 ≥3 个连续指标变量（单因子模型识别要求）。")
        elif user_spec and len(used) < 2:
            summary.append("SEM 失败：config model_spec 中未匹配到 ≥2 个数据列名。")
        else:
            import pandas as pd

            from researchforge.executor import rbridge

            inds = used
            sub = df[inds].dropna()
            # prefer lavaan (R, gold standard — also gives SRMR) when available;
            # fall back to pure-Python semopy so the analysis runs anywhere.
            # Only use the R backend with identifier-safe column names: names go
            # into the R model string, so a name with quotes/commas could break
            # parsing or inject R — semopy takes the names as data, no eval.
            names_safe = all(re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c)) for c in inds)
            # the spec is interpolated into cfa("...") as an R string literal; a stray
            # double-quote/backslash would break out, so a custom spec with those is
            # routed to semopy instead (takes the spec as a Python string, no R eval).
            spec_safe = '"' not in spec and "\\" not in spec
            result = None
            if names_safe and spec_safe and rbridge.r_available() and rbridge.r_package_available("lavaan"):
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
                    _ylab = (
                        load["indicator"].astype(str) + " ← " + load["factor"].astype(str)
                        if "factor" in load.columns and load["factor"].nunique() > 1
                        else load["indicator"].astype(str)
                    )
                    ax.barh(_ylab, load["std_loading"], color="#4C72B0")
                    ax.set_xlabel("standardised loading")
                    ax.set_title("SEM measurement loadings")
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
                _n_factors = len(set(_sem_latents(spec))) or 1
                _model_desc = (
                    f"自定义模型（{_n_factors} 因子，按 config model_spec）"
                    if user_spec
                    else "单因子 CFA"
                )
                _tail = (
                    "" if user_spec
                    else "（此为探索性模板；可用 config={\"model_spec\": \"lavaan语法\"} 按理论结构改写后重跑）"
                )
                summary.append(
                    f"{entry.method} 完成（后端：{result['backend']}）：{_model_desc} over "
                    f"{len(inds)} 个指标（df={dof:.0f}）；CFI={cfi:.3f} TLI={tli:.3f} "
                    f"RMSEA={rmsea:.3f}{srmr_txt} → {verdict}" + _tail
                )
                code += [
                    "# SEM single-factor CFA — prefers R/lavaan, falls back to semopy",
                    f'spec = "{spec}"',
                    "# lavaan: cfa(spec, data=df, std.lv=TRUE); semopy: semopy.Model(spec).fit(df)",
                ]

    elif entry.id == "mediation":
        import numpy as np
        import pandas as pd

        _excl = {fp.unit_col, fp.time_col}
        cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
        y_col = cont[0] if cont else None
        cand = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "binary"} and c.name not in _excl | {y_col}
        ]
        if y_col is None or len(cand) < 2:
            summary.append("中介分析失败：需要连续结果变量 Y + ≥2 个变量（自变量 X、中介 M）。")
        else:
            x_col, m_col = cand[0], cand[1]  # default by column order; X→M→Y assumption
            try:
                import statsmodels.api as sm
                from statsmodels.stats.mediation import Mediation

                sub = df[[y_col, x_col, m_col]].dropna().rename(
                    columns={y_col: "_y", x_col: "_x", m_col: "_m"}
                )
                om = sm.OLS.from_formula("_y ~ _x + _m", sub)  # outcome: Y ~ X + M
                mm = sm.OLS.from_formula("_m ~ _x", sub)  # mediator: M ~ X
                med = Mediation(om, mm, "_x", "_m").fit(n_rep=1000)
                s = med.summary()
                s.to_csv(d / "mediation_summary.csv", encoding="utf-8")
                files.append("mediation_summary.csv")

                def _row(label):
                    return s.loc[label] if label in s.index else None

                acme = _row("ACME (average)")
                ade = _row("ADE (average)")
                tot = _row("Total effect")
                pm = _row("Prop. mediated (average)")
                indirect = float(acme["Estimate"])
                direct = float(ade["Estimate"])
                total = float(tot["Estimate"])
                prop = float(pm["Estimate"]) if pm is not None else float("nan")
                acme_p = float(acme["P-value"])
                estimates["indirect_effect_ACME"] = round(indirect, 4)
                estimates["direct_effect_ADE"] = round(direct, 4)
                estimates["total_effect"] = round(total, 4)
                estimates["prop_mediated"] = round(prop, 4)
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    labels = ["indirect (ACME)", "direct (ADE)", "total"]
                    est = [indirect, direct, total]
                    lo = [float(acme["Lower CI bound"]), float(ade["Lower CI bound"]), float(tot["Lower CI bound"])]
                    hi = [float(acme["Upper CI bound"]), float(ade["Upper CI bound"]), float(tot["Upper CI bound"])]
                    err = [[e - l for e, l in zip(est, lo)], [h - e for e, h in zip(est, hi)]]
                    fig, ax = plt.subplots(figsize=(5.5, 3.2))
                    ax.errorbar(est, range(3), xerr=err, fmt="o", capsize=4)
                    ax.axvline(0, color="grey", ls="--")
                    ax.set_yticks(range(3))
                    ax.set_yticklabels(labels)
                    ax.set_xlabel("effect (95% CI)")
                    ax.set_title(f"Mediation {x_col} → {m_col} → {y_col}")
                    fig.tight_layout()
                    fig.savefig(d / "mediation_effects.png", dpi=150)
                    plt.close(fig)
                    files.append("mediation_effects.png")
                except Exception:
                    pass
                verdict = "存在显著中介" if acme_p < 0.05 else "中介效应不显著"
                # prop. mediated is meaningless under suppression (opposite signs) or
                # near-zero total effect — flag rather than print a misleading % (Opus catch).
                suppression = abs(total) < 0.05 or (direct * indirect < 0)
                prop_txt = "不稳定（抑制效应/总效应近零，比例无意义）" if suppression else f"{prop:.1%}"
                summary.append(
                    f"{entry.method} 完成：路径 {x_col} → {m_col} → {y_col}；"
                    f"间接效应 ACME={indirect:.4f}（p={acme_p:.3g}，{verdict}），"
                    f"直接效应 ADE={direct:.4f}，总效应={total:.4f}，中介比例={prop_txt}（Monte-Carlo CI）。"
                    "⚠ X/M/Y 按列序默认（首连续=Y，其后=X、M），且 **X↔M 不对称**——交换二者是不同模型、"
                    "列序只是选了一个假设而非事实，请核对你的理论路径；中介推断需 no-unmeasured-confounding 假定（非纯相关即因果）。"
                )
                code += [
                    "from statsmodels.stats.mediation import Mediation",
                    f"# OLS('{y_col}~{x_col}+{m_col}') + OLS('{m_col}~{x_col}'); Mediation(...).fit(n_rep=1000)",
                ]
            except Exception as err:
                summary.append(f"中介分析失败：{err}")

    elif entry.id == "survival_analysis":
        import numpy as np
        import pandas as pd

        # NB: do NOT exclude fp.time_col — in survival data the duration IS the
        # time column (profiler may flag a "time"/"days" column as time_col).
        _excl = {fp.unit_col}

        def _named(kws, kinds, extra_excl=()):
            ex = _excl | set(extra_excl)
            return next(
                (
                    c.name
                    for c in fp.columns
                    if any(k in c.name.lower() for k in kws) and c.kind in kinds and c.name not in ex
                ),
                None,
            )

        ev_kws = ["event", "status", "death", "dead", "fail", "censor", "relapse"]
        dur_kws = ["duration", "time", "days", "month", "year", "tenure", "surviv", "followup", "week", "age_at"]
        event_col = _named(ev_kws, {"binary"}) or next(
            (c.name for c in fp.columns if c.kind == "binary" and c.name not in _excl), None
        )
        dur_col = _named(dur_kws, {"continuous", "count"}, extra_excl=(event_col,) if event_col else ())
        if dur_col is None and event_col is not None:
            dur_col = next(
                (
                    c.name
                    for c in fp.columns
                    if c.kind in {"continuous", "count"}
                    and c.name not in _excl | {event_col}
                    and (df[c.name].dropna() > 0).all()
                ),
                None,
            )
        if event_col is None or dur_col is None:
            summary.append(
                "生存分析失败：需要一个事件列（二值 0/1，如 event/status/death）"
                "+ 一个时长列（正数，如 time/duration/days）。"
            )
        else:
            try:
                from lifelines import CoxPHFitter, KaplanMeierFitter
                from lifelines.statistics import logrank_test, proportional_hazard_test

                group_col = next(
                    (
                        c.name
                        for c in fp.columns
                        if c.kind in {"binary", "categorical"}
                        and c.name not in _excl | {event_col}
                        and df[c.name].dropna().nunique() == 2
                    ),
                    None,
                )
                covars = [
                    c.name
                    for c in fp.columns
                    if c.kind in {"continuous", "binary"} and c.name not in _excl | {dur_col, event_col}
                ][:5]
                keep = [dur_col, event_col] + ([group_col] if group_col else []) + covars
                sub = df[list(dict.fromkeys(keep))].dropna()
                sub = sub[sub[dur_col].astype(float) > 0]
                dur = sub[dur_col].astype(float)
                ev = sub[event_col].astype(int)
                # event must be 0/1 (1=event, 0=censored) for lifelines event_observed.
                # (profiler's "binary" is already exactly {0,1}; defensive guard + the
                # disclosure below also flags the un-detectable reversed-coding risk.)
                if set(ev.dropna().unique()) - {0, 1}:
                    raise ValueError(f"事件列 {event_col} 非 0/1 编码（应 1=事件,0=删失）")
                n_ev, n_cens = int(ev.sum()), int((ev == 0).sum())

                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                kmf = KaplanMeierFitter()
                fig, ax = plt.subplots(figsize=(6, 4.5))
                logrank_p = None
                if group_col:
                    for gval, gdf in sub.groupby(group_col):
                        kmf.fit(gdf[dur_col].astype(float), gdf[event_col].astype(int), label=f"{group_col}={gval}")
                        kmf.plot_survival_function(ax=ax)
                    levels = list(sub.groupby(group_col).groups)
                    g0 = sub[sub[group_col] == levels[0]]
                    g1 = sub[sub[group_col] == levels[1]]
                    lr = logrank_test(
                        g0[dur_col].astype(float), g1[dur_col].astype(float),
                        g0[event_col].astype(int), g1[event_col].astype(int),
                    )
                    logrank_p = float(lr.p_value)
                else:
                    kmf.fit(dur, ev)
                    kmf.plot_survival_function(ax=ax)
                ax.set_xlabel(f"{dur_col} (time)")
                ax.set_ylabel("survival probability")
                ax.set_title("Kaplan-Meier survival")
                fig.tight_layout()
                fig.savefig(d / "km_curve.png", dpi=150)
                plt.close(fig)
                files.append("km_curve.png")

                kmf_all = KaplanMeierFitter().fit(dur, ev)
                median = float(kmf_all.median_survival_time_)
                estimates["median_survival"] = round(median, 4) if np.isfinite(median) else -1.0
                estimates["n_events"] = float(n_ev)
                estimates["n_censored"] = float(n_cens)

                cox_msg = ""
                if covars and n_ev >= 2 * len(covars) + 2:
                    try:
                        cph = CoxPHFitter()
                        cph.fit(sub[[dur_col, event_col, *covars]], dur_col, event_col)
                        cs = cph.summary
                        ctab = cs[["coef", "exp(coef)", "se(coef)", "p"]].copy()
                        ctab.columns = ["coef", "hazard_ratio", "std_err", "p_value"]
                        ctab.round(4).to_csv(d / "cox_hazard_ratios.csv", encoding="utf-8")
                        files.append("cox_hazard_ratios.csv")
                        for cv in covars:
                            if cv in cs.index:
                                estimates[f"HR_{cv}"] = round(float(cs.loc[cv, "exp(coef)"]), 4)
                        try:
                            ph = proportional_hazard_test(cph, sub[[dur_col, event_col, *covars]])
                            ph_p = float(ph.summary["p"].min())
                            cox_msg = (
                                f"；Cox 风险比见 cox_hazard_ratios.csv（{len(covars)} 协变量）；"
                                f"比例风险检验 min-p={ph_p:.3g}"
                                f"（{'满足 PH 假定' if ph_p > 0.05 else '⚠ PH 假定可能不成立'}）"
                            )
                        except Exception:
                            cox_msg = f"；Cox 风险比见 cox_hazard_ratios.csv（{len(covars)} 协变量）"
                    except Exception as cerr:
                        cox_msg = f"；Cox 拟合跳过（{str(cerr)[:60]}）"

                med_txt = f"{median:.3g}" if np.isfinite(median) else "未达（>50% 存活到末期）"
                grp_txt = (
                    f"；按 {group_col} 分组 KM + log-rank p={logrank_p:.3g}"
                    f"（{'组间生存有显著差异' if logrank_p is not None and logrank_p < 0.05 else '组间无显著差异'}）"
                    if group_col
                    else ""
                )
                summary.append(
                    f"{entry.method} 完成：{len(sub)} 例（{n_ev} 事件 / {n_cens} 删失）；"
                    f"事件列={event_col}，时长列={dur_col}；中位生存={med_txt}{grp_txt}{cox_msg}。"
                    "⚠ 事件/时长列按列名自动识别，请核对；事件列须 0/1（1=事件,0=删失），"
                    "若反向编码 HR 会反转；删失假定为随机非信息性。"
                )
                code += [
                    "from lifelines import KaplanMeierFitter, CoxPHFitter  # 生存分析",
                    f"# KM.fit(df['{dur_col}'], df['{event_col}']); CoxPHFitter().fit(df, '{dur_col}', '{event_col}')",
                ]
            except Exception as err:
                summary.append(f"生存分析失败：{err}")

    elif entry.id == "random_effects":
        if not (fp.unit_col and fp.time_col):
            summary.append("随机效应模型失败：需要面板数据（单位列 + 时间列）。")
        else:
            import numpy as np
            import pandas as pd
            from scipy.stats import chi2

            y = next(
                (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
                None,
            )
            preds = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
            ][:6]
            if y is None or not preds:
                summary.append("随机效应模型失败：需要连续结果变量 + ≥1 个预测变量。")
            else:
                try:
                    from linearmodels.panel import PanelOLS, RandomEffects

                    dd = (
                        df[[fp.unit_col, fp.time_col, y, *preds]]
                        .dropna()
                        .drop_duplicates([fp.unit_col, fp.time_col])
                        .set_index([fp.unit_col, fp.time_col])
                    )
                    fe = PanelOLS(dd[y], dd[preds], entity_effects=True).fit(
                        cov_type="clustered", cluster_entity=True
                    )
                    re = RandomEffects(dd[y], dd[preds].assign(const=1.0)).fit(
                        cov_type="clustered", cluster_entity=True
                    )
                    # classic Hausman needs the EFFICIENT (unadjusted) covariances so
                    # V_FE - V_RE is PSD; clustered/robust covs break the χ² basis (Opus
                    # catch). Use unadjusted fits for the test; keep clustered for SEs.
                    fe_u = PanelOLS(dd[y], dd[preds], entity_effects=True).fit()
                    re_u = RandomEffects(dd[y], dd[preds].assign(const=1.0)).fit()
                    common = [p for p in preds if p in fe.params.index and p in re.params.index]
                    diff = (fe_u.params[common] - re_u.params[common]).values
                    vdiff = (fe_u.cov.loc[common, common] - re_u.cov.loc[common, common]).values
                    h_stat = max(0.0, float(diff @ np.linalg.pinv(vdiff) @ diff))
                    h_p = float(chi2.sf(h_stat, len(common)))
                    use_fe = h_p < 0.05
                    rec = "FE（固定效应）" if use_fe else "RE（随机效应）"
                    tab = pd.DataFrame(
                        {
                            "term": common,
                            "FE_coef": [round(float(fe.params[c]), 4) for c in common],
                            "FE_p": [round(float(fe.pvalues[c]), 4) for c in common],
                            "RE_coef": [round(float(re.params[c]), 4) for c in common],
                            "RE_p": [round(float(re.pvalues[c]), 4) for c in common],
                        }
                    )
                    tab.to_csv(d / "fe_re_coefficients.csv", index=False, encoding="utf-8")
                    files.append("fe_re_coefficients.csv")
                    chosen = fe if use_fe else re
                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        se = chosen.std_errors[common]
                        co = chosen.params[common]
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.errorbar(co.values, range(len(common)), xerr=1.96 * se.values, fmt="o")
                        ax.axvline(0, color="grey", ls="--")
                        ax.set_yticks(range(len(common)))
                        ax.set_yticklabels(common)
                        ax.set_xlabel(f"{rec[:2]} coefficient (95% CI)")
                        fig.tight_layout()
                        fig.savefig(d / "coefficients.png", dpi=150)
                        plt.close(fig)
                        files.append("coefficients.png")
                    except Exception:
                        pass
                    estimates["hausman_stat"] = round(h_stat, 4)
                    estimates["hausman_p"] = round(h_p, 4)
                    for c in common:
                        estimates[c] = round(float(chosen.params[c]), 4)
                    summary.append(
                        f"{entry.method} 完成：面板 {dd.index.get_level_values(0).nunique()} 单位 × "
                        f"{dd.index.get_level_values(1).nunique()} 期；结果 {y}，{len(common)} 个预测变量。"
                        f"Hausman H={h_stat:.3f}, p={h_p:.3g} → 推荐 {rec}"
                        f"（p<0.05 表示随机效应与回归元相关、RE 不一致，应用 FE）。系数对比见 fe_re_coefficients.csv（聚类稳健 SE）。"
                    )
                    code += [
                        "from linearmodels.panel import PanelOLS, RandomEffects  # 面板 RE + Hausman",
                        "# FE=PanelOLS(entity_effects); RE=RandomEffects; H=(b_fe-b_re)'pinv(Vfe-Vre)(b_fe-b_re)",
                    ]
                except Exception as err:
                    summary.append(f"随机效应模型失败：{err}")

    elif entry.id == "dynamic_panel_gmm":
        import re

        from researchforge.executor import rbridge

        if not (fp.unit_col and fp.time_col):
            summary.append("动态面板 GMM 失败：需要面板数据（单位列 + 时间列）。")
        else:
            y = next(
                (c.name for c in fp.columns if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}),
                None,
            )
            preds = [
                c.name
                for c in fp.columns
                if c.kind in {"continuous", "binary"} and c.name not in {y, fp.unit_col, fp.time_col}
            ][:5]
            names_safe = y is not None and all(
                re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(c))
                for c in [y, *preds, fp.unit_col, fp.time_col]
            )
            n_periods = int(df[fp.time_col].nunique())
            if y is None or not preds:
                summary.append("动态面板 GMM 失败：需要连续结果变量 + ≥1 个预测变量。")
            elif n_periods < 3:
                summary.append("动态面板 GMM 失败：需要 ≥3 个时间期（差分 GMM 与 AR 检验要求）。")
            elif not (rbridge.r_available() and rbridge.r_package_available("plm")):
                summary.append(
                    "动态面板 GMM 需要 R 的 plm 包（未检测到）。安装：install.packages('plm')；"
                    "或用 panel_fixed_effects / random_effects。"
                )
            elif not names_safe:
                summary.append("动态面板 GMM 失败：列名需为标识符式（字母/数字/. _）。")
            else:
                sub = df[[fp.unit_col, fp.time_col, y, *preds]].dropna()
                csv = d / "_gmm_input.csv"
                sub.to_csv(csv, index=False)
                try:
                    import numpy as np

                    _endo = [p for p in (cfg.get("endogenous") or []) if p in preds]
                    _lo, _hi = _gmm_lags(cfg)
                    coef, diag = _dynamic_gmm_via_r(
                        csv, fp.unit_col, fp.time_col, y, preds,
                        endogenous=_endo, gmm_lags=(_lo, _hi),
                    )
                    coef["term"] = [
                        f"lag_{y}" if str(t).startswith("lag(") else str(t) for t in coef["term"]
                    ]
                    coef.to_csv(d / "gmm_coefficients.csv", index=False, encoding="utf-8")
                    files.append("gmm_coefficients.csv")
                    sargan_p = diag.get("sargan_p", float("nan"))
                    ar1_p = diag.get("ar1_p", float("nan"))
                    ar2_p = diag.get("ar2_p", float("nan"))
                    sargan_ok = sargan_p > 0.05
                    ar2_ok = ar2_p > 0.05
                    _endo_note = (
                        f"内生变量（用 lag {_lo}:{_hi} 工具）：{_endo}"
                        if _endo
                        else "全部协变量设为严格外生（可用 config endogenous 标出内生变量）"
                    )
                    if _lo < 2:
                        _endo_note += (
                            "；⚠ gmm_lags 起始<2 仅对前定(predetermined)变量有效，"
                            "滞后被解释变量已强制 lag≥2（差分方程中 lag1 为无效工具）"
                        )
                    (d / "diagnostics.txt").write_text(
                        "动态面板 GMM (Arellano-Bond 差分 GMM, twosteps, Windmeijer 稳健 SE)\n"
                        f"工具集限 lag {_lo}-{_hi}（抑制工具过度增殖；多工具会抬高/弱化 Sargan）。\n"
                        f"{_endo_note}。\n"
                        f"Sargan 过度识别检验 p = {sargan_p:.4g}（注：Sargan 非稳健,非 Hansen J）"
                        f"（{'工具有效（不拒）' if sargan_ok else '被拒 → 工具集可疑'}）\n"
                        f"AR(1) p = {ar1_p:.4g}（差分后通常显著，正常）\n"
                        f"AR(2) p = {ar2_p:.4g} "
                        f"（{'无二阶自相关 → GMM 一致' if ar2_ok else '⚠ 有二阶自相关 → GMM 不一致'}）\n\n"
                        + coef.to_string(index=False),
                        encoding="utf-8",
                    )
                    files.append("diagnostics.txt")
                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.errorbar(
                            coef["estimate"], range(len(coef)),
                            xerr=1.96 * coef["std_err"], fmt="o",
                        )
                        ax.axvline(0, color="grey", ls="--")
                        ax.set_yticks(range(len(coef)))
                        ax.set_yticklabels(coef["term"])
                        ax.set_xlabel("GMM coefficient (95% CI)")
                        ax.set_title("Dynamic panel GMM (Arellano-Bond)")
                        fig.tight_layout()
                        fig.savefig(d / "coefficients.png", dpi=150)
                        plt.close(fig)
                        files.append("coefficients.png")
                    except Exception:
                        pass
                    lag_rows = coef[coef["term"].str.startswith("lag_")]
                    persistence = float(lag_rows.iloc[0]["estimate"]) if len(lag_rows) else float("nan")
                    estimates["persistence_lag_coef"] = round(persistence, 4)
                    estimates["sargan_p"] = round(sargan_p, 4)
                    estimates["ar2_p"] = round(ar2_p, 4)
                    for _, r in coef.iterrows():
                        if not str(r["term"]).startswith("lag_"):
                            estimates[str(r["term"])] = round(float(r["estimate"]), 4)
                    valid = sargan_ok and ar2_ok
                    summary.append(
                        f"{entry.method} 完成（R/plm，{n_periods} 期）：滞后被解释变量系数（持续性）"
                        f"={persistence:.3f}；Sargan p={sargan_p:.3g}"
                        f"（{'工具有效' if sargan_ok else '工具可疑'}），AR(2) p={ar2_p:.3g}"
                        f"（{'无二阶自相关' if ar2_ok else '⚠有二阶自相关、GMM不一致'}）"
                        f"{'' if valid else ' —— ⚠ 诊断未全通过,结果存疑'}。系数见 gmm_coefficients.csv。"
                        + (f"（内生变量 {_endo}、工具滞后 {_lo}:{_hi} 按 config 指定）" if (_endo or cfg.get("gmm_lags")) else "")
                    )
                    code += [
                        "library(plm)  # Arellano-Bond 差分 GMM",
                        f"# pgmm({y} ~ lag({y},1) + ... | lag({y},{_lo}:{_hi})[+ lag(endo)], model='twosteps')",
                    ]
                except Exception as err:
                    summary.append(f"动态面板 GMM 失败：{err}")
                finally:
                    try:
                        csv.unlink()
                    except OSError:
                        pass

    elif entry.id == "var_granger":
        import numpy as np
        import pandas as pd

        _excl = {fp.unit_col, fp.time_col}
        series = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl][:6]
        if len(series) < 2:
            summary.append("VAR/Granger 失败：需要 ≥2 个连续时间序列变量。")
        else:
            try:
                from statsmodels.tsa.api import VAR

                d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
                data = d2[series].dropna().reset_index(drop=True)
                n = len(data)
                if n < 20:
                    summary.append("VAR/Granger 失败：观测不足（<20），无法稳健拟合 VAR。")
                else:
                    maxlags = max(1, min(8, n // (len(series) + 1) - 1))
                    res = VAR(data).fit(maxlags=maxlags, ic="aic")
                    if res.k_ar < 1:
                        res = VAR(data).fit(1)  # AIC picked 0 lags -> force lag 1 for Granger
                    pmat = pd.DataFrame(np.nan, index=series, columns=series)  # rows=causing -> cols=caused
                    for causing in series:
                        for caused in series:
                            if causing != caused:
                                try:
                                    pmat.loc[causing, caused] = float(
                                        res.test_causality(caused, [causing]).pvalue
                                    )
                                except Exception:
                                    pass
                    pmat.round(4).to_csv(d / "granger_pvalues.csv", encoding="utf-8")
                    files.append("granger_pvalues.csv")
                    links = [
                        f"{r}→{c}"
                        for r in series
                        for c in series
                        if r != c and pd.notna(pmat.loc[r, c]) and pmat.loc[r, c] < 0.05
                    ]
                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        mat = -np.log10(pmat.to_numpy(dtype=float).clip(1e-300, 1))
                        np.fill_diagonal(mat, np.nan)
                        fig, ax = plt.subplots(figsize=(5.5, 4.5))
                        im = ax.imshow(mat, cmap="Reds")
                        ax.set_xticks(range(len(series)))
                        ax.set_xticklabels(series, rotation=45, ha="right")
                        ax.set_yticks(range(len(series)))
                        ax.set_yticklabels(series)
                        ax.set_xlabel("caused →")
                        ax.set_ylabel("causing →")
                        ax.set_title("Granger causality  -log10(p)")
                        fig.colorbar(im, label="-log10(p)")
                        fig.tight_layout()
                        fig.savefig(d / "granger_heatmap.png", dpi=150)
                        plt.close(fig)
                        files.append("granger_heatmap.png")
                    except Exception:
                        pass
                    try:
                        fig = res.irf(10).plot()
                        fig.savefig(d / "irf.png", dpi=120)
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                        files.append("irf.png")
                    except Exception:
                        pass
                    # active stationarity check (ADF) — non-stationary series give
                    # spurious Granger causality; flag loudly, not just in prose (Opus catch).
                    n_nonstat = 0
                    try:
                        from statsmodels.tsa.stattools import adfuller

                        for s in series:
                            if adfuller(data[s].to_numpy(dtype=float), autolag="AIC")[1] > 0.05:
                                n_nonstat += 1
                    except Exception:
                        n_nonstat = -1
                    estimates["selected_lag"] = float(res.k_ar)
                    estimates["n_series"] = float(len(series))
                    estimates["n_causal_links"] = float(len(links))
                    estimates["n_nonstationary"] = float(n_nonstat)
                    stat_warn = (
                        f"；⚠ ADF 检验：{n_nonstat}/{len(series)} 个序列非平稳，Granger 结果恐为伪因果——请先差分/平稳化再解读"
                        if n_nonstat > 0
                        else ""
                    )
                    time_warn = "" if fp.time_col else "；⚠ 无时间列，按行序当作时间序列处理（请确认行序即时序）"
                    summary.append(
                        f"{entry.method} 完成：{len(series)} 个序列 × {n} 期，VAR 阶数={res.k_ar}（AIC 选）；"
                        f"Granger 因果 p 值矩阵见 granger_pvalues.csv；显著(p<0.05)有向因果："
                        f"{('、'.join(links) if links else '无')}{stat_warn}{time_warn}。"
                        f"按{'时间列 ' + str(fp.time_col) if fp.time_col else '行序'}排序；"
                        "Granger 因果是「预测性」非结构因果。"
                    )
                    code += [
                        "from statsmodels.tsa.api import VAR  # VAR + Granger 因果",
                        "# VAR(data).fit(ic='aic'); res.test_causality(caused, [causing]).pvalue; res.irf().plot()",
                    ]
            except Exception as err:
                summary.append(f"VAR/Granger 失败：{err}")

    elif entry.id == "pls_sem":
        summary.append(
            "PLS-SEM（偏最小二乘结构方程）需要你指定**测量模型**（哪些指标→哪个潜变量）与结构路径；"
            "引擎无法自动推断（随意分组会产出无意义结果，故不自动跑）。请指定测量模型后用 plspm / SmartPLS 运行；"
            "或先用 **SEM**（CB-SEM，自动单因子 CFA，经 lavaan/semopy）/ **EFA**（探索因子结构）作可自动执行的替代。"
        )

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
