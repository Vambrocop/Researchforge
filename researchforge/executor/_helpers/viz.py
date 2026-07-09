"""Executor helpers — FIGURE + REPORT side, split out of _helpers/core.py (which was
approaching the 1500-line module guard; see CLAUDE.md「防巨石复发的扩展约定」).

Lives here: the matplotlib style/localization chokepoint (`_init_mpl_style` and its
CJK-font-detection / English→Chinese figure-label-translation machinery), pure
figure/plot helpers (take already-computed data/model and draw a PNG — `_heatmap`,
`_coef_plot`, `_resid_plot`, `_quantile_process_plot`, `_ordinal_prob_plot`,
`_silhouette_plot`, `_nca_plot`, `_plotly_*`, `_mcda_rank_plot`), and the report-side
helper `_report` (assembles report.md, including the report_narrative hook).

COMPUTE/RESOLVER helpers (regression, resolve_outcome, DEA/MCDA math, QCA/GMM config
readers, `_varimax`, `_usda_texture`, …) stay in `_helpers/core.py`. Three routines
that FIT a model/estimator and only incidentally emit a diagnostic plot as a side
effect of a larger returned result — `_conformal_prediction`, `_network_via_nx`,
`_synthetic_control` — were judged primarily-compute and were kept in core.py rather
than moved here (see the module docstring there).

No dependency on run.py or core.py (avoids any import cycle); core.py re-exports the
names below at its end so every existing `from researchforge.executor.run import _xxx`
/ `from researchforge.executor._helpers.core import _xxx` import keeps working.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path


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


def _figure_language_is_zh() -> bool:
    """Re-read the localization policy fresh every call (never cached), so a long-lived
    process — e.g. the FastAPI server, where `_init_mpl_style` only *installs* the
    savefig patch once — still honors a later ``RF_FIG_LANG=en`` or a machine with no
    CJK font: True only when RF_FIG_LANG isn't "en" AND a CJK font is available
    (``_detect_cjk_font`` is itself lru_cached, so this check is effectively free)."""
    if os.environ.get("RF_FIG_LANG", "").strip().lower() == "en":
        return False
    return bool(_detect_cjk_font())


def _install_savefig_localizer() -> None:
    """Monkeypatch Figure.savefig (once, in-process) to localize labels to Chinese just
    before saving — the single chokepoint every branch's figure flows through, so no
    per-branch edits are needed. Idempotent; guarded by a sentinel attribute.

    The patch itself is installed once (this function is a no-op on repeat calls), but
    the *decision* of whether to translate is deferred to each call of the wrapped
    ``savefig`` (see ``_figure_language_is_zh``) rather than frozen at install time —
    otherwise the first analysis run in a long-lived process would lock in the
    language/font state for every run after it."""
    try:
        import matplotlib.figure as _mfig

        if getattr(_mfig.Figure.savefig, "_rf_localized", False):
            return
        _orig = _mfig.Figure.savefig

        def savefig(self, *args, **kwargs):
            if _figure_language_is_zh():
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
        # Install the savefig localization chokepoint unconditionally (idempotent — a
        # no-op after the first call). Whether a given save actually translates is
        # decided per-call inside the patched savefig (via _figure_language_is_zh), not
        # here: gating the *install* on the current cjk/RF_FIG_LANG state would freeze
        # that state for the rest of the process (e.g. a long-lived FastAPI server),
        # ignoring a later RF_FIG_LANG=en. See CLAUDE.md P2-4.
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
