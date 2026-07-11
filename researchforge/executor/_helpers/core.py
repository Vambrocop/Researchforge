"""Executor helpers — COMPUTE / RESOLVER side, moved out of the run.py monolith.

Lives here: role/column resolution (`resolve_outcome`, `_pick_did_treatment`),
regression setup (`_regression`), DEA/MCDA/QCA/GMM math and config readers,
`_varimax`, `_usda_texture`, `_run_dir`, and a few compute-heavy routines that only
incidentally emit a diagnostic plot as a side effect of a larger returned result
(`_conformal_prediction`, `_network_via_nx`, `_synthetic_control` — kept here rather
than in viz.py because most of each function is model fitting/estimation, not
plotting; see CLAUDE.md「防巨石复发的扩展约定」for the split rationale).

FIGURE/REPORT helpers (matplotlib style + CJK/localization chokepoint, pure
figure-drawing helpers, `_report`) moved to `_helpers/viz.py` when this module
neared the 1500-line guard; re-exported at the end of this file so every existing
`from researchforge.executor.run import _xxx` / `from researchforge.executor._helpers
.core import _xxx` import keeps working unchanged.

Re-exported by run.py and imported by branches/*.py. No dependency on run.py.
"""

from __future__ import annotations

import datetime
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


def resolve_outcome(fp: DataFingerprint, cfg: dict | None, candidates: list[str]) -> str:
    """Pick the dependent variable for a modeling run — the single place role detection
    becomes BINDING (closes the selection→execution loop). Priority:

      1. an explicit ``config["outcome"]`` (user intent always wins),
      2. a HIGH-confidence detected outcome (an unambiguous DV name like target/y/outcome)
         when it is one of the candidate columns — so ``[x1, x2, target]`` models ``target``,
         not the first column,
      3. otherwise the first candidate that is NOT treatment-named (treat/arm/exposed/dose…) —
         a treatment indicator is almost never the dependent variable, so ``[treated, died]``
         resolves to ``died`` even when no outcome name fires. Name-signal only (no positional
         treatment guess), so unnamed column conventions are never flipped. A candidate that
         is itself the role-detected ``likely_outcome`` (ANY confidence) is never skipped:
         compound measurement names like ``body_condition_score`` / ``group_size`` carry a
         treatment word as a segment but ARE the outcome — the outcome signal vetoes the skip
         (medium still doesn't jump the order; the veto only protects the convention).
      4. otherwise the first candidate (the long-standing "first column" default).

    MEDIUM/LOW confidence hints deliberately do NOT bind: a domain word (price/sales/score)
    is just as often a predictor, so binding it could model the wrong column. ``candidates``
    is the eligible dependent-variable columns in dataframe order (must be non-empty)."""
    from researchforge.profiler.roles import is_treatment_named

    cfg = cfg or {}
    if cfg.get("outcome") in candidates:
        return cfg["outcome"]
    lo = getattr(fp, "likely_outcome", None)
    if getattr(fp, "likely_outcome_confidence", "") == "high" and lo in candidates:
        return lo
    non_treat = [c for c in candidates if not is_treatment_named(c) or c == lo]
    return non_treat[0] if non_treat else candidates[0]


def resolve_predictors(
    fp, cfg, outcome, kinds=("continuous", "count", "binary"), cap=5, df=None, forced_cap=None
):
    """Predictor columns for a modeling run — the shared convention next to resolve_outcome:
    an explicit config["predictors"] (filtered to existing columns != outcome) wins; else the
    numeric/binary columns in dataframe order, excluding outcome/unit/time. `kinds` and `cap`
    are the caller's convention (regression 5, bayesian 20, ...). When df is given, config
    names are validated against df.columns (wider than fp for coerced/derived cols).
    ``forced_cap`` lets the explicit-config path keep its own cap (regression forces up to 8
    but auto-selects only 5) — it defaults to ``cap`` when not given."""
    cfg = cfg or {}
    cols = df.columns if df is not None else [c.name for c in fp.columns]
    forced = [c for c in (cfg.get("predictors") or []) if c in cols and c != outcome]
    if forced:
        return forced[: forced_cap if forced_cap is not None else cap]
    exclude = {outcome, fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind in set(kinds) and c.name not in exclude][:cap]


def _regression(df, fp: DataFingerprint, entry: AnalysisEntry, cfg: dict | None = None):
    import statsmodels.formula.api as smf

    cfg = cfg or {}
    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    if not cont:
        raise ValueError("没有连续型因变量，无法回归")
    # dependent variable: config override > high-confidence detected outcome > first continuous
    y = resolve_outcome(fp, cfg, cont)

    fe_terms: list[str] = []
    if entry.id in {"panel_fixed_effects", "did"} and fp.unit_col and fp.time_col:
        fe_terms = [f"C(Q('{fp.unit_col}'))", f"C(Q('{fp.time_col}'))"]

    if entry.id == "did" and fp.treatment_candidates:
        rhs_vars = _pick_did_treatment(df, fp) or fp.treatment_candidates[:1]
    else:
        # optional explicit predictor list via config["predictors"] (cap 8) else
        # auto continuous/count/binary columns in dataframe order (cap 5)
        rhs_vars = resolve_predictors(
            fp, cfg, y, kinds=("continuous", "count", "binary"), cap=5, forced_cap=8, df=df
        )

    rhs = [f"Q('{v}')" for v in rhs_vars] + fe_terms
    formula = f"Q('{y}') ~ " + (" + ".join(rhs) if rhs else "1")

    # Panel data: default HC1 (heteroskedasticity-robust only) ignores within-unit serial
    # correlation and understates SE -> spuriously tiny p-values (dogfood P3: 1e-187-class
    # on firm x year panels). Cluster by unit_col instead — same statsmodels idiom already
    # used for the other unit/time formula fits in this codebase (event_study.py /
    # staggered_did.py / did_advanced.py: cov_type="cluster", cov_kwds={"groups": ...});
    # econometrics.py's linearmodels RandomEffects/PanelOLS use the linearmodels-equivalent
    # cov_type="clustered", cluster_entity=True. Applies to ols_regression too — pooled OLS
    # fit on panel-shaped data still has within-unit correlated errors even without FE terms.
    if fp.is_panel and fp.unit_col and fp.unit_col in df.columns:
        formula_cols = [y, *rhs_vars]
        if fe_terms:
            formula_cols += [fp.unit_col, fp.time_col]
        fit_cols = list(dict.fromkeys([*formula_cols, fp.unit_col]))
        fit_df = df[fit_cols].dropna()
        model = smf.ols(formula, data=df.loc[fit_df.index]).fit(
            cov_type="cluster", cov_kwds={"groups": fit_df[fp.unit_col]}
        )
    else:
        model = smf.ols(formula, data=df).fit(cov_type="HC1")
    return y, rhs_vars, formula, model


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


# ── viz.py re-exports (FIGURE/REPORT side) ────────────────────────────────────
# Kept importable from `researchforge.executor._helpers.core` (and, via run.py's own
# re-export block, from `researchforge.executor.run`) for backward compatibility —
# every existing import path keeps working unchanged. See `_helpers/viz.py` for the
# actual implementations. Explicit names (not *) so this stays lint-friendly; each is
# individually unused HERE (only re-exported), hence the per-line noqa.
from researchforge.executor._helpers.viz import (  # noqa: F401
    _CJK_CANDIDATES,
    _FIG_GLOSSARY,
    _FIG_GLOSSARY_ORDERED,
    _THEME_COLORS,
    _coef_plot,
    _detect_cjk_font,
    _figure_language_is_zh,
    _heatmap,
    _init_mpl_style,
    _install_savefig_localizer,
    _localize_figure,
    _mcda_rank_plot,
    _nca_plot,
    _ordinal_prob_plot,
    _plotly_corr_heatmap,
    _plotly_scatter,
    _quantile_process_plot,
    _report,
    _resid_plot,
    _silhouette_plot,
    _translate_label,
)
