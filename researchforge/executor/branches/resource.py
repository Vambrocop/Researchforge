"""Branch handlers for the RESOURCE / SUSTAINABILITY family.

This module owns ``composite_index`` — a composite-indicator (composite index)
builder, the standard OECD/JRC methodology for combining many sub-indicators into a
single score. It is the core of building an Energy-Water-Food (EWF) nexus
security / pressure index, a sustainability index, a competitiveness ranking, etc.

(NOTE: ``input_output`` lives in a SEPARATE module ``resource_io.py`` under the same
family and is owned by another agent — this file does not touch it.)

PIPELINE (OECD/JRC "Handbook on Constructing Composite Indicators", 2008):

  1. DIRECTION — each sub-indicator is either a *benefit* (higher = better) or a
     *cost* (higher = worse). config ``cost_indicators`` lists the cost ones; cost
     indicators are inverted during normalization so that, on the common [0,1]
     scale, a higher normalized value always means "better". Default: all benefit
     (disclosed).

  2. NORMALIZE — config ``normalization`` in {minmax, zscore} (default minmax):
       * minmax (benefit): x' = (x - min) / (max - min)             -> [0,1]
       * minmax (cost)   : x' = (max - x) / (max - min) = 1 - above  -> [0,1]
       * zscore (benefit): x' = (x - mean) / sd
       * zscore (cost)   : x' = -(x - mean) / sd = (mean - x) / sd
     A zero-range column (max==min, or sd==0) carries no information; it is mapped
     to a constant (0.5 for minmax, 0.0 for zscore) and disclosed.

  3. WEIGHT — config ``weighting`` in {equal, entropy} (default equal):
       * equal  : w_j = 1 / m for all m indicators.
       * entropy: Shannon-entropy objective weights. On the min-max-to-[0,1]
         normalized matrix Z (n units x m indicators) compute the column shares
         p_ij = Z_ij / Σ_i Z_ij (a degenerate all-zero column -> uniform shares),
         the entropy e_j = -k Σ_i p_ij ln(p_ij) with k = 1/ln(n) and the
         convention 0·ln 0 = 0 (an epsilon guards log(0)), the degree of
         diversification d_j = 1 - e_j, and w_j = d_j / Σ_j d_j. More dispersed
         indicators (more diversification, lower entropy) get more weight. (This is
         the standard CRITIC/entropy-weight MCDA method, re-implemented locally —
         per the spec we do NOT import another branch's helper.)

  4. AGGREGATE — config ``aggregation`` in {linear, geometric} (default linear):
       * linear (weighted arithmetic mean): C_i = Σ_j w_j · Z_ij — fully
         COMPENSATORY (a high pillar can offset a low pillar).
       * geometric (weighted geometric mean): C_i = Π_j Z_ij^{w_j} — partially
         compensatory: it PENALIZES imbalance across pillars (a near-zero pillar
         drags the whole score down), a key nexus / sustainability property.
         Geometric needs strictly positive inputs, so the normalized matrix is
         shifted into (0, 1] first (Z -> ε + (1-ε)·Z) and this is disclosed.

OUTPUT: composite score per unit, the ranking (1 = best), and the weight vector.
Products: a ranked-scores CSV (unit, composite_score, rank, + normalized pillar
columns), a horizontal bar chart of unit scores (English labels). estimates:
``n_units``, ``n_indicators``, ``weight__<indicator>`` per indicator, plus
``top_score`` / ``min_score`` (all plain floats).

Honest degrade (Chinese "composite_index 跳过：…") when there are fewer than two
numeric indicators or no usable rows — never crash / fabricate.

Pure Python (numpy / pandas / matplotlib Agg, English plot labels).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers (local to this family — no cross-branch imports)
# --------------------------------------------------------------------------- #
def _numeric_indicator_cols(ctx: Ctx) -> list[str]:
    """Column names usable as sub-indicators.

    Accepts continuous, count AND id kinds (an integer indicator with all-distinct
    values is misclassified ``id`` by the profiler — CLAUDE.md「id 陷阱」; a small
    integer stream profiles as ``count``). The unit-label column (fp.unit_col) and
    the time column (fp.time_col) are excluded.
    """
    fp = ctx.fp
    excl = {fp.unit_col, fp.time_col}
    return [
        c.name
        for c in fp.columns
        if c.name not in excl and c.kind in ("continuous", "count", "id")
    ]


def _save_fig(d, fname, files, build) -> None:
    """best-effort matplotlib figure (Agg). build(plt) draws on the current figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        build(plt)
        plt.tight_layout()
        plt.savefig(d / fname, dpi=150)
        plt.close("all")
        files.append(fname)
    except Exception:
        pass


def _normalize(matrix, cost_mask, method: str):
    """Normalize an (n units x m indicators) float matrix to a common scale where a
    higher value = better, inverting the cost-flagged columns.

    Returns (Z, zero_range_cols) where Z has the same shape and zero_range_cols is
    the list of column indices that carried no spread (constant -> mapped to a flat
    value: 0.5 for minmax, 0.0 for zscore).
    """
    import numpy as np

    X = np.asarray(matrix, dtype=float)
    n, m = X.shape
    Z = np.zeros_like(X)
    zero_cols: list[int] = []
    for j in range(m):
        col = X[:, j]
        is_cost = bool(cost_mask[j])
        if method == "zscore":
            mu = float(np.nanmean(col))
            sd = float(np.nanstd(col, ddof=0))
            if sd == 0.0 or not np.isfinite(sd):
                Z[:, j] = 0.0
                zero_cols.append(j)
                continue
            z = (col - mu) / sd
            Z[:, j] = -z if is_cost else z
        else:  # minmax (default)
            cmin = float(np.nanmin(col))
            cmax = float(np.nanmax(col))
            rng = cmax - cmin
            if rng == 0.0 or not np.isfinite(rng):
                Z[:, j] = 0.5
                zero_cols.append(j)
                continue
            z = (col - cmin) / rng  # benefit -> [0,1]
            Z[:, j] = (1.0 - z) if is_cost else z
    return Z, zero_cols


def _entropy_weights_local(matrix, cost_mask):
    """Shannon-entropy objective weights from an (n x m) raw matrix.

    The entropy method needs a non-negative, direction-corrected matrix, so we first
    min-max normalize to [0,1] (cost columns flipped) — the standard
    min-max-to-[0,1]-then-p_ij approach (STOP-AND-REPORT default). Then per column:
      p_ij = Z_ij / Σ_i Z_ij         (all-zero column -> uniform p = 1/n)
      e_j  = -k Σ_i p_ij ln p_ij,  k = 1/ln(n),  with 0·ln0 = 0 (epsilon guards log0)
      d_j  = 1 - e_j                 (degree of diversification)
      w_j  = d_j / Σ_j d_j           (all-degenerate -> equal weights)

    Returns a length-m numpy array summing to 1.
    """
    import numpy as np

    Z, _ = _normalize(matrix, cost_mask, "minmax")  # -> [0,1], higher = better
    Z = np.clip(Z, 0.0, None)
    n, m = Z.shape
    eps = 1e-12
    if n <= 1:  # entropy needs >=2 units; fall back to equal weights
        return np.full(m, 1.0 / m)
    k = 1.0 / np.log(n)
    d = np.zeros(m)
    for j in range(m):
        col_sum = float(Z[:, j].sum())
        if col_sum <= 0.0:
            p = np.full(n, 1.0 / n)  # degenerate column -> uniform -> max entropy
        else:
            p = Z[:, j] / col_sum
        e = -k * float(np.sum(p * np.log(p + eps)))
        d[j] = 1.0 - e
    d = np.clip(d, 0.0, None)
    total = float(d.sum())
    if total <= 0.0:  # every column degenerate -> equal weights
        return np.full(m, 1.0 / m)
    return d / total


def _aggregate(Z, weights, method: str):
    """Aggregate the normalized matrix Z (n x m, higher=better) with a weight vector
    into a per-unit composite score.

      linear   : C_i = Σ_j w_j Z_ij                (compensatory weighted sum)
      geometric: C_i = Π_j Z_ij^{w_j}             (penalizes imbalance)

    For geometric, Z is shifted into (0, 1] first (Z -> ε + (1-ε)Z) so zeros/negatives
    don't blow up the product / logs. Returns a length-n numpy array.
    """
    import numpy as np

    Zf = np.asarray(Z, dtype=float)
    w = np.asarray(weights, dtype=float)
    if method == "geometric":
        eps = 1e-6
        zmin = float(np.nanmin(Zf))
        zmax = float(np.nanmax(Zf))
        rng = zmax - zmin
        if rng > 0:
            base = (Zf - zmin) / rng  # -> [0,1] (handles zscore's negatives too)
        else:
            base = np.zeros_like(Zf)
        shifted = eps + (1.0 - eps) * base  # -> (0,1]
        return np.exp(np.sum(w * np.log(shifted), axis=1))
    return Zf @ w  # linear weighted sum


# ===========================================================================
# composite_index — OECD/JRC composite-indicator builder
#   Refs: OECD/JRC "Handbook on Constructing Composite Indicators" (2008);
#         Saisana & Tarantola, "State-of-the-art report on composite indicators".
# ===========================================================================
@register("composite_index")
def _branch_composite_index(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    # ---- unit (entity) label column -----------------------------------------
    # Resolve the label column FIRST so it is excluded from the indicator set.
    # The profiler only sets fp.unit_col for panel data (needs a time col), so for a
    # flat cross-section it is usually None — fall back to the first non-numeric col.
    def _coerces_numeric(col: str) -> bool:
        return pd.to_numeric(df[col], errors="coerce").notna().any()

    unit_col = cfg.get("unit") or fp.unit_col
    if unit_col not in df.columns:
        # first column that does NOT look numeric makes a natural label
        unit_col = next((c for c in df.columns if not _coerces_numeric(c)), None)

    # ---- choose indicator columns -------------------------------------------
    # exclude the unit label + time col; keep only genuinely-numeric columns (a text
    # id column profiles as kind "id" but coerces to all-NaN — drop it, don't 0-fill).
    excl = {unit_col, fp.unit_col, fp.time_col}
    auto_inds = [c for c in _numeric_indicator_cols(ctx)
                 if c not in excl and _coerces_numeric(c)]
    user_inds = cfg.get("indicators")
    if user_inds:
        inds = [c for c in user_inds if c in df.columns and c != unit_col
                and _coerces_numeric(c)]
        ind_note = f"（指标列取自 config indicators：{', '.join(inds)}）"
    else:
        inds = auto_inds
        ind_note = "（指标列 = 全部数值列，已排除单位/时间列）"
    if len(inds) < 2:
        summary.append(
            "composite_index 跳过：可用数值指标列不足 2 个"
            f"（找到 {len(inds)} 个）。综合指数至少需要 2 个子指标。"
        )
        return

    # ---- build the indicator matrix, drop all-NaN rows ----------------------
    M = df[inds].apply(pd.to_numeric, errors="coerce")
    keep = ~M.isna().all(axis=1)
    M = M[keep]
    if unit_col is not None and unit_col in df.columns:
        units = df.loc[keep, unit_col].astype(str).to_numpy()
    else:
        units = np.array([f"unit_{i+1}" for i in range(len(M))])
    # impute remaining holes with the column mean so a partial row still scores
    M = M.fillna(M.mean(numeric_only=True))
    M = M.fillna(0.0)  # a fully-empty column (all-NaN) -> 0
    X = M.to_numpy(dtype=float)
    n_units = X.shape[0]
    if n_units < 1:
        summary.append("composite_index 跳过：去除全空行后没有可用的单位（行）。")
        return

    # ---- direction / config knobs -------------------------------------------
    cost_list = cfg.get("cost_indicators") or []
    cost_set = {c for c in cost_list if c in inds}
    cost_mask = np.array([ind in cost_set for ind in inds], dtype=bool)

    norm = str(cfg.get("normalization", "minmax")).lower()
    if norm not in ("minmax", "zscore"):
        norm = "minmax"
    weighting = str(cfg.get("weighting", "equal")).lower()
    if weighting not in ("equal", "entropy"):
        weighting = "equal"
    aggregation = str(cfg.get("aggregation", "linear")).lower()
    if aggregation not in ("linear", "geometric"):
        aggregation = "linear"

    try:
        # ---- normalize ------------------------------------------------------
        Z, zero_cols = _normalize(X, cost_mask, norm)

        # ---- weights --------------------------------------------------------
        m = len(inds)
        if weighting == "entropy":
            w = _entropy_weights_local(X, cost_mask)
        else:
            w = np.full(m, 1.0 / m)

        # ---- aggregate ------------------------------------------------------
        scores = _aggregate(Z, w, aggregation)
        scores = np.asarray(scores, dtype=float)

        # ---- rank (1 = best) ------------------------------------------------
        order = np.argsort(-scores, kind="mergesort")  # stable, descending
        ranks = np.empty(n_units, dtype=int)
        ranks[order] = np.arange(1, n_units + 1)

        # ---- results table --------------------------------------------------
        norm_cols = {f"norm__{inds[j]}": np.round(Z[:, j], 6) for j in range(m)}
        out = pd.DataFrame(
            {"unit": units, "composite_score": np.round(scores, 6), "rank": ranks}
        )
        for k, v in norm_cols.items():
            out[k] = v
        out = out.sort_values("rank").reset_index(drop=True)
        out.to_csv(d / "composite_scores.csv", index=False)
        files.append("composite_scores.csv")

        wtable = pd.DataFrame(
            {
                "indicator": inds,
                "direction": ["cost" if cost_mask[j] else "benefit" for j in range(m)],
                "weight": np.round(w, 6),
            }
        )
        wtable.to_csv(d / "indicator_weights.csv", index=False)
        files.append("indicator_weights.csv")

        # ---- estimates (plain floats only) ----------------------------------
        estimates["n_units"] = float(n_units)
        estimates["n_indicators"] = float(m)
        estimates["top_score"] = float(np.nanmax(scores))
        estimates["min_score"] = float(np.nanmin(scores))
        for j in range(m):
            estimates[f"weight__{inds[j]}"] = float(w[j])

        # ---- horizontal bar chart (English labels) --------------------------
        def _build(plt):
            top = out.head(min(30, len(out)))  # avoid an unreadably tall axis
            labels = top["unit"].astype(str).tolist()
            vals = top["composite_score"].to_numpy()
            ypos = np.arange(len(labels))[::-1]  # best at the top
            plt.figure(figsize=(8, max(3, 0.35 * len(labels) + 1)))
            plt.barh(ypos, vals, color="#3b7dd8")
            plt.yticks(ypos, labels)
            plt.xlabel("Composite score")
            agg_lbl = "geometric" if aggregation == "geometric" else "linear"
            plt.title(
                f"Composite index ranking (norm={norm}, "
                f"weights={weighting}, agg={agg_lbl})"
            )
            plt.grid(axis="x", alpha=0.3)

        _save_fig(d, "composite_index.png", files, _build)

        # ---- code string (reproducible) -------------------------------------
        code.append(
            "# composite_index — OECD/JRC composite-indicator builder\n"
            f"indicators = {inds!r}\n"
            f"cost_indicators = {sorted(cost_set)!r}\n"
            f"normalization, weighting, aggregation = "
            f"{norm!r}, {weighting!r}, {aggregation!r}\n"
            "import numpy as np\n"
            "X = df[indicators].apply(pd.to_numeric, errors='coerce')\n"
            "X = X.fillna(X.mean()).fillna(0.0).to_numpy(dtype=float)\n"
            "cost_mask = np.array([c in set(cost_indicators) for c in indicators])\n"
            "# 1) normalize (cost columns inverted) -> higher = better\n"
            "Z = np.zeros_like(X)\n"
            "for j in range(X.shape[1]):\n"
            "    col = X[:, j]\n"
            "    if normalization == 'zscore':\n"
            "        sd = col.std(ddof=0); z = (col-col.mean())/sd if sd else col*0\n"
            "        Z[:, j] = -z if cost_mask[j] else z\n"
            "    else:\n"
            "        rng = col.max()-col.min()\n"
            "        z = (col-col.min())/rng if rng else col*0+0.5\n"
            "        Z[:, j] = (1-z) if cost_mask[j] else z\n"
            "# 2) weights (equal or Shannon entropy)\n"
            "if weighting == 'entropy':\n"
            "    Zp = np.clip(Z, 0, None); n = Zp.shape[0]; k = 1/np.log(n)\n"
            "    d = np.empty(Zp.shape[1])\n"
            "    for j in range(Zp.shape[1]):\n"
            "        s = Zp[:, j].sum(); p = Zp[:, j]/s if s>0 else np.full(n,1/n)\n"
            "        d[j] = 1 - (-k*np.sum(p*np.log(p+1e-12)))\n"
            "    w = np.clip(d,0,None); w = w/w.sum() if w.sum()>0 else np.full(len(w),1/len(w))\n"
            "else:\n"
            "    w = np.full(Z.shape[1], 1/Z.shape[1])\n"
            "# 3) aggregate (linear sum or weighted geometric mean)\n"
            "if aggregation == 'geometric':\n"
            "    rng = Z.max()-Z.min(); base = (Z-Z.min())/rng if rng else Z*0\n"
            "    shifted = 1e-6 + (1-1e-6)*base\n"
            "    scores = np.exp((w*np.log(shifted)).sum(axis=1))\n"
            "else:\n"
            "    scores = Z @ w\n"
            "ranking = (-scores).argsort().argsort() + 1\n"
        )

        # ---- summary (Chinese, with ⚠ disclosures) --------------------------
        winner = out.iloc[0]["unit"]
        norm_name = {"minmax": "极差标准化(min-max→[0,1])", "zscore": "Z 分数标准化"}[norm]
        w_name = {"equal": "等权", "entropy": "熵权(客观赋权)"}[weighting]
        agg_name = {"linear": "线性加权和(完全补偿)", "geometric": "加权几何平均(惩罚不均衡)"}[
            aggregation
        ]
        top_w = sorted(zip(inds, w), key=lambda t: -t[1])[:3]
        w_str = "，".join(f"{nm}={wt:.3f}" for nm, wt in top_w)
        cost_note = (
            f"成本型(越大越差)指标：{', '.join(sorted(cost_set))}（已在标准化中反向）"
            if cost_set
            else "⚠ 未指定 cost_indicators，默认全部指标为效益型(越大越好)；"
            "若某指标越大越差，请用 config cost_indicators 标注，否则排名会错。"
        )
        lines = [
            f"综合指数(composite_index)：{n_units} 个单位 × {m} 个子指标 {ind_note}。",
            f"方法：{norm_name} → {w_name} → {agg_name}。",
            f"第 1 名：{winner}（综合得分 {float(scores[order[0]]):.4f}）。"
            f"权重前三：{w_str}（完整权重见 indicator_weights.csv）。",
            cost_note,
            "⚠ 标准化方式、赋权方式、聚合方式都是“价值判断”，会直接改变排名——"
            "综合指数的可信度不超过这些选择的可辩护性。",
            "⚠ 线性聚合允许指标间相互补偿(高分抵消低分)；几何聚合惩罚不均衡"
            "(某一支柱接近 0 会拖垮总分，是 EWF nexus/可持续性的关键性质)。"
            "建议做敏感性检验(切换标准化/赋权/聚合，看排名稳不稳)。",
        ]
        if zero_cols:
            zc = ", ".join(inds[j] for j in zero_cols)
            flat = "0.5" if norm == "minmax" else "0"
            lines.append(
                f"⚠ 零方差(常数)指标 [{zc}] 无区分度，已映射为常数 {flat}，对排名无贡献。"
            )
        if weighting == "entropy":
            lines.append(
                "ℹ 熵权法在 min-max→[0,1] 矩阵上按列离散度赋权"
                "(离散度越大→熵越小→权重越大)；log(0) 用 ε=1e-12 守护。"
                "⚠ 因先做了逐列 min-max（各列拉伸填满 [0,1]），反映的是各指标在共同 [0,1] 尺度上的"
                "分布形状、而非原始量纲的离散度，故权重差异被压缩(弱于直接对原始数据的熵权)。"
            )
        if aggregation == "geometric":
            lines.append(
                "ℹ 几何聚合前已把标准化值平移进 (0,1]（Z→ε+(1−ε)·Z，ε=1e-6）以避免 0/负数。"
            )
        summary.append(" ".join(lines))
    except Exception as exc:  # never crash the run
        summary.append(f"composite_index 跳过：计算时出错（{type(exc).__name__}: {exc}）。")
        return
