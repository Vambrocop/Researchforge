"""Executor helpers (compute, plotting, core) — moved out of the run.py monolith.
Re-exported by run.py and imported by branches/*.py. No dependency on run.py.
"""

from __future__ import annotations

import datetime
import os
import re
from functools import lru_cache
from pathlib import Path

from researchforge.catalog.schema import AnalysisEntry
from researchforge.profiler.fingerprint import DataFingerprint


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


# CJK font candidates, in preference order: cross-platform Noto/Source Han first,
# then the fonts each OS ships (Windows: YaHei/SimHei; macOS: PingFang/Hiragino;
# Linux: WenQuanYi). Detect-first — we do NOT bundle a font (a Noto Sans CJK ttf is
# ~16 MB); we use whatever CJK font is already installed and degrade to English if none.
_CJK_CANDIDATES = (
    "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "Source Han Sans",
    "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft JhengHei",
    "SimSun", "Arial Unicode MS",
)


@lru_cache(maxsize=1)
def _detect_cjk_font() -> str | None:
    """First available CJK-capable font name on this machine, or None. Cached — the
    font-manager scan is not free and the answer is stable per process. Best-effort:
    never raises (returns None if matplotlib/font-manager is unavailable)."""
    try:
        from matplotlib import font_manager as fm

        available = {f.name for f in fm.fontManager.ttflist}
        for name in _CJK_CANDIDATES:
            if name in available:
                return name
    except Exception:
        pass
    return None


# ── figure-label localization (English → Chinese) ────────────────────────────
# A vocabulary glossary applied to figure TITLES / AXIS LABELS / LEGEND entries only
# (never tick labels or data text), longest phrase first so multi-word terms win over
# their component words, word-boundary + case-insensitive. Coverage of the common
# figure vocabulary; unknown terms are left in English (graceful, extensible). Applied
# only when a CJK font is present (so headless/CI without CJK fonts keeps English and
# never renders tofu) and RF_FIG_LANG != "en".
_FIG_GLOSSARY = {
    # multi-word phrases (matched first)
    "posterior predictive check": "后验预测检验", "observed vs replicated": "观测 vs 复制",
    "survival probability": "生存概率", "technical efficiency": "技术效率",
    "number of nodes": "节点数", "number of customers": "客户数", "number of": "数量：",
    "time order (rank)": "时间次序（秩）", "time step": "时间步", "time index": "时间索引",
    "period index": "期数", "period t": "期 t", "posterior density": "后验密度",
    "fitted values": "拟合值", "partial residuals": "偏残差", "partial effect": "偏效应",
    "credible band": "可信带", "credible interval": "可信区间",
    "confidence interval": "置信区间", "standardized value": "标准化值",
    "control chart": "控制图", "epidemic curve": "流行曲线",
    "block-connection matrix": "块连接矩阵", "held-out auc": "留出 AUC",
    "odds ratio": "比值比", "hazard ratio": "风险比", "risk ratio": "风险比",
    "standard deviation": "标准差", "subgroup mean": "子组均值", "subgroup index": "子组序号",
    # single words / short terms
    "counts": "计数", "count": "计数", "frequency": "频数", "density": "密度",
    "probability": "概率", "values": "值", "value": "值", "observed": "观测",
    "predicted": "预测", "expected": "期望", "actual": "实际", "residuals": "残差",
    "residual": "残差", "fitted": "拟合", "coefficients": "系数", "coefficient": "系数",
    "estimates": "估计", "estimate": "估计", "importance": "重要性", "groups": "组",
    "group": "组", "lag": "滞后", "median": "中位数", "mean": "均值", "variance": "方差",
    "distribution": "分布", "samples": "样本", "sample": "样本", "scores": "得分",
    "score": "得分", "rank": "秩", "threshold": "阈值", "weights": "权重", "weight": "权重",
    "age": "年龄", "trend": "趋势", "proportion": "比例", "cumulative": "累积",
    "effects": "效应", "effect": "效应", "survival": "生存", "hazard": "风险率",
    "nodes": "节点", "node": "节点", "edges": "边", "edge": "边",
    "communities": "社团", "community": "社团", "clusters": "簇", "cluster": "簇",
    "components": "成分", "component": "成分", "factors": "因子", "factor": "因子",
    "loadings": "载荷", "loading": "载荷", "quantiles": "分位数", "quantile": "分位数",
    "percentile": "百分位", "iterations": "迭代", "iteration": "迭代", "levels": "水平",
    "level": "水平", "categories": "类别", "category": "类别", "classes": "类",
    "outcome": "结果", "treatment": "处理", "control": "对照", "exposure": "暴露",
    "correlation": "相关", "covariance": "协方差", "regression": "回归",
    "predictions": "预测", "prediction": "预测", "errors": "误差", "error": "误差",
    "accuracy": "准确率", "loss": "损失", "income": "收入", "wealth": "财富",
    "price": "价格", "cost": "成本", "revenue": "营收", "demand": "需求",
    "duration": "时长", "survivors": "存活数", "infected": "感染", "susceptible": "易感",
    "recovered": "康复", "time": "时间", "period": "期", "index": "索引",
    "frequency (hz)": "频率(Hz)", "risk": "风险", "units sold": "销量", "target": "目标",
}
# merge the auto-generated long-tail glossary (387 figure labels translated in one
# batch via Gemini + spot-reviewed); hand-curated entries above win on any key clash.
try:
    from researchforge.executor._helpers.fig_glossary import EXTRA as _FIG_EXTRA

    _FIG_GLOSSARY = {**_FIG_EXTRA, **_FIG_GLOSSARY}
except Exception:
    pass
# Pre-compiled longest-first. The word boundary EXCLUDES digits and underscore too, so
# we never translate a fragment of an identifier (e.g. "level" inside "level_k", "age"
# inside "age2", "score" inside "x_score") — only standalone English words. Pre-compiling
# also avoids re's 512-pattern cache thrashing with this many entries.
_FIG_GLOSSARY_ORDERED = [
    (re.compile(rf"(?<![A-Za-z0-9_]){re.escape(_en)}(?![A-Za-z0-9_])", re.IGNORECASE), _zh)
    for _en, _zh in sorted(_FIG_GLOSSARY.items(), key=lambda kv: -len(kv[0]))
]


def _translate_label(text: str) -> str:
    """Translate an English figure label to Chinese via the glossary (word-boundary,
    case-insensitive, longest phrase first). Unknown terms are left untouched. A column
    name that exactly equals an English glossary word can still be translated — an
    inherent limit of label-layer translation (the savefig hook can't know a token was
    a column name); see CLAUDE.md."""
    if not text or not text.strip():
        return text
    out = text
    for pat, zh in _FIG_GLOSSARY_ORDERED:
        out = pat.sub(lambda m, _zh=zh: _zh, out)  # callable repl: zh treated literally
    return out


def _localize_figure(fig) -> None:
    """Translate a figure's title / axis labels / legend entries in place (best-effort;
    never raises). Tick labels and data text are deliberately left alone. Idempotent
    per-figure (a sentinel) so a re-saved figure is never double-translated (which could
    corrupt a label whose Chinese still contained an English glossary token)."""
    try:
        if getattr(fig, "_rf_localized", False):
            return
        for ax in fig.axes:
            for art in (ax.title, ax.xaxis.label, ax.yaxis.label):
                t = art.get_text()
                if t:
                    art.set_text(_translate_label(t))
            leg = ax.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    s = txt.get_text()
                    if s:
                        txt.set_text(_translate_label(s))
        sup = getattr(fig, "_suptitle", None)
        if sup is not None and sup.get_text():
            sup.set_text(_translate_label(sup.get_text()))
        fig._rf_localized = True
    except Exception:
        pass


def _install_savefig_localizer() -> None:
    """Monkeypatch Figure.savefig (once, in-process) to localize labels to Chinese just
    before saving — the single chokepoint every branch's figure flows through, so no
    per-branch edits are needed. Idempotent; guarded by a sentinel attribute."""
    try:
        import matplotlib.figure as _mfig

        if getattr(_mfig.Figure.savefig, "_rf_localized", False):
            return
        _orig = _mfig.Figure.savefig

        def savefig(self, *args, **kwargs):
            _localize_figure(self)
            return _orig(self, *args, **kwargs)

        savefig._rf_localized = True
        _mfig.Figure.savefig = savefig
    except Exception:
        pass


def _init_mpl_style(theme: str | None = None) -> None:
    """Apply one clean, publication-friendly look to every figure this run
    produces. Theme is chosen by arg or the RF_THEME env var (default | nature |
    aer | dark). Called once per analysis; best-effort so a missing/old
    matplotlib never breaks an analysis.

    Also enables CJK rendering: if a Chinese/Japanese/Korean font is installed it is
    prepended to the font fallback chain (with ``axes.unicode_minus=False`` so the
    minus sign still renders under CJK fonts), so figure labels may be Chinese without
    becoming tofu boxes. When no CJK font is found it degrades silently to the Latin
    fonts — labels written in English keep working unchanged. This is the single
    unified entry every run flows through (run_analysis calls it before any branch
    plots), so the font policy applies engine-wide without per-branch changes."""
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
        # CJK rendering: prepend an installed CJK font to the fallback chain (and turn
        # off the Unicode minus glyph, which many CJK fonts lack). Latin text still uses
        # the existing Latin fonts; Chinese labels now render instead of tofu boxes.
        rc["axes.unicode_minus"] = False
        cjk = _detect_cjk_font()
        if cjk:
            sans = rc.get("font.sans-serif") or ["DejaVu Sans", "Arial", "Helvetica"]
            serif = rc.get("font.serif") or ["DejaVu Serif", "Times New Roman"]
            rc["font.sans-serif"] = [cjk] + [f for f in sans if f != cjk]
            rc["font.serif"] = [cjk] + [f for f in serif if f != cjk]
            rc.setdefault("font.family", "serif" if theme == "aer" else "sans-serif")
        plt.rcParams.update(rc)
        # Localize figure labels to Chinese (only when a CJK font is present, so we never
        # render tofu, and unless RF_FIG_LANG=en forces English for publication/CI).
        if cjk and os.environ.get("RF_FIG_LANG", "zh").strip().lower() != "en":
            _install_savefig_localizer()
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


def _sem_latents(spec: str) -> list[str]:
    """Latent-variable names = the LHS of every `=~` measurement line in a
    lavaan/semopy model spec. Used to pick out measurement loadings generically."""
    import re

    return [m.group(1) for m in re.finditer(r"([A-Za-z_]\w*)\s*=~", spec)]


def _conformal_prediction(df, outcome, predictors, alpha, seed, plot_path):
    """Split (inductive) conformal prediction (Vovk; Lei et al.): distribution-free
    prediction intervals with a finite-sample marginal coverage guarantee >= 1-alpha,
    for ANY base regressor. Splits data into train / calibration / test; fits a
    RandomForest on train; the conformity threshold q = the ceil((n_cal+1)(1-alpha))-th
    smallest absolute calibration residual; interval = yhat +/- q. Returns a metrics
    dict (target vs empirical coverage, mean width, q). Writes a coverage plot."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    sub = df[[outcome, *predictors]].dropna()
    rng = np.random.default_rng(int(seed))
    n = len(sub)
    idx = rng.permutation(n)
    n_tr, n_cal = int(0.5 * n), int(0.25 * n)
    tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]
    X = sub[predictors].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    model = RandomForestRegressor(n_estimators=200, random_state=int(seed))
    model.fit(X[tr], y[tr])
    cal_scores = np.abs(y[cal] - model.predict(X[cal]))  # conformity scores
    n_c = len(cal_scores)
    raw_k = int(np.ceil((n_c + 1) * (1 - alpha)))  # exact conformal rank (finite-sample valid)
    # raw_k > n_cal means the formal threshold is +inf (cal set too small for a 1-alpha
    # guarantee); cap to the max residual as an approximation and flag it (Opus catch).
    cal_too_small = raw_k > n_c
    k = min(n_c, raw_k)
    q = float(np.sort(cal_scores)[k - 1])  # k-th smallest -> threshold
    yhat_te = model.predict(X[te])
    covered = np.abs(y[te] - yhat_te) <= q
    emp_cov = float(np.mean(covered))
    ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
    r2_te = float(1 - np.sum((y[te] - yhat_te) ** 2) / ss_tot) if ss_tot > 1e-9 else float("nan")
    out = {
        "target_coverage": round(1 - alpha, 3),
        "empirical_coverage": round(emp_cov, 3),
        "mean_interval_width": round(2 * q, 4),
        "conformal_q": round(q, 4),
        "test_r2": round(r2_te, 4) if r2_te == r2_te else float("nan"),
        "n_test": int(len(te)),
        "n_calibration": int(n_c),
        "cal_too_small": bool(cal_too_small),
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(yhat_te)
        xs = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.fill_between(xs, (yhat_te - q)[order], (yhat_te + q)[order], color="#4C72B0", alpha=0.25,
                        label=f"{(1 - alpha):.0%} prediction interval")
        ax.plot(xs, yhat_te[order], color="#4C72B0", lw=1, label="prediction")
        ax.scatter(xs, y[te][order], s=12, c=np.where(covered[order], "#55A868", "#C44E52"),
                   label="actual (green=covered)")
        ax.set_xlabel("test points (sorted by prediction)")
        ax.set_ylabel(outcome)
        ax.set_title(f"Conformal prediction — empirical coverage {emp_cov:.1%} (target {(1 - alpha):.0%})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return out


def _network_via_nx(df, source, target, weight, directed, plot_path):
    """Graph / network analysis via networkx: graph-level metrics, node centralities
    (degree/betweenness/closeness/eigenvector), and Louvain community detection.
    Deterministic (community uses a fixed seed). Returns (metrics dict, node
    centrality DataFrame). Writes a spring-layout plot coloured by community.
    Raises so the caller can degrade honestly."""
    import networkx as nx
    import pandas as pd

    cols = [source, target] + ([weight] if weight else [])
    sub = df[cols].dropna()
    create = nx.DiGraph if directed else nx.Graph
    G = nx.from_pandas_edgelist(
        sub, source, target, edge_attr=(weight if weight else None), create_using=create()
    )
    if G.number_of_nodes() < 3:
        raise RuntimeError("有效节点 <3，无法做网络分析")
    n, m = G.number_of_nodes(), G.number_of_edges()
    UG = G.to_undirected() if directed else G
    # components (weak for directed) + largest component for distance metrics
    comps = list(nx.weakly_connected_components(G) if directed else nx.connected_components(G))
    largest = max(comps, key=len)
    Glc = UG.subgraph(largest)
    metrics = {
        "n_nodes": n, "n_edges": m,
        "density": round(nx.density(G), 4),
        "avg_degree": round((2 * m / n) if not directed else (m / n), 3),
        "avg_clustering": round(nx.average_clustering(UG), 4),
        "n_components": len(comps),
        "largest_component_frac": round(len(largest) / n, 3),
    }
    if len(largest) <= 1500:  # distance metrics are O(N*E); cap to stay fast
        metrics["diameter_largest"] = int(nx.diameter(Glc)) if len(largest) > 1 else 0
        metrics["avg_path_len_largest"] = round(nx.average_shortest_path_length(Glc), 3) if len(largest) > 1 else 0.0
    try:
        metrics["degree_assortativity"] = round(nx.degree_assortativity_coefficient(G), 4)
    except Exception:
        metrics["degree_assortativity"] = float("nan")

    w = weight if weight else None
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight=w, seed=0) if n > 2 else {k: 0.0 for k in G}
    clo = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight=w)
    except Exception:
        eig = {k: float("nan") for k in G}
    cent = pd.DataFrame({
        "node": list(G.nodes()),
        "degree_centrality": [round(deg[x], 4) for x in G.nodes()],
        "betweenness": [round(bet[x], 4) for x in G.nodes()],
        "closeness": [round(clo[x], 4) for x in G.nodes()],
        "eigenvector": [round(eig[x], 4) if eig[x] == eig[x] else float("nan") for x in G.nodes()],
    }).sort_values("degree_centrality", ascending=False).reset_index(drop=True)

    # Louvain communities on the undirected graph (seeded -> reproducible)
    comm = nx.community.louvain_communities(UG, weight=w, seed=0)
    node2comm = {x: i for i, c in enumerate(comm) for x in c}
    metrics["n_communities"] = len(comm)
    metrics["modularity"] = round(nx.community.modularity(UG, comm, weight=w), 4)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Gp = UG if n <= 400 else UG.subgraph(largest)  # cap plotted graph size
        pos = nx.spring_layout(Gp, seed=0, k=None)
        colors = [node2comm.get(x, 0) for x in Gp.nodes()]
        sizes = [30 + 600 * deg.get(x, 0) for x in Gp.nodes()]
        fig, ax = plt.subplots(figsize=(7, 6))
        nx.draw_networkx_edges(Gp, pos, alpha=0.25, ax=ax)
        nx.draw_networkx_nodes(Gp, pos, node_color=colors, node_size=sizes, cmap="tab20", ax=ax)
        ax.set_title(f"Network ({Gp.number_of_nodes()} nodes, {metrics['n_communities']} communities)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return metrics, cent


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


def _report(entry, fp, summary, files, override, estimates=None) -> str:
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
    # Presentation-only "report intelligence" — an analyst-style narrative inserted
    # right after 结果摘要. Purely additive; defensively wrapped so a narrative
    # failure can never break report generation (falls back to no narrative).
    try:
        from researchforge.executor._helpers.report_narrative import build_narrative

        narrative = build_narrative(entry, fp, summary, override, estimates)
        if narrative:
            lines += [*narrative]
    except Exception:
        pass
    if entry.biases:
        lines += ["## 偏差提醒（需读者判断）", *[f"- {b}" for b in entry.biases], ""]
    lines += ["## 产物文件", *[f"- `{f}`" for f in files]]
    return "\n".join(lines)
