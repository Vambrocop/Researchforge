"""Branch handlers for the RESOURCE-ECONOMICS bridge (cost <-> physical units).

This module lives in the "resource" family (shared via auto-discovery + a separate
yaml file) but speaks the language of DECISION ECONOMICS — it bridges money (cost)
and physical/resource units (effect / abatement). It is the cost-side piece of an
Energy-Water-Food (EWF) nexus toolkit: once you have physical flows / savings from
the other resource methods (input_output, MFA, composite_index), these two methods
tell you WHICH option / WHICH portfolio is worth paying for.

  - cost_effectiveness_analysis  (CEA / ICER) — compares decision ALTERNATIVES
    (rows = options) on cost vs an effect, builds the cost-effectiveness frontier
    (drops dominated + extended-dominated options) and the incremental
    cost-effectiveness ratios (ICERs); optional willingness-to-pay (WTP) picks the
    optimum and reports net monetary benefit (NMB).

  - marginal_abatement_cost  (MACC) — ranks a PORTFOLIO of interventions
    (rows = measures) by cost per unit of resource / emission abated, building the
    classic marginal-abatement-cost curve (cumulative abatement on x, MAC on y) and
    flagging negative-cost ("no-regret") measures; optional carbon price reports the
    abatement achievable at or below that price.

INPUT MODEL: each row is a decision option / measure. A cost column (config ``cost``,
else a column named cost/price/expense) and an effect/abatement column (config keys
below, else named effect/outcome/benefit/... or abatement/saving/...). An optional
label column (config ``option``/``measure``, else the first non-numeric column).

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法> 跳过:…" appended to summary + return (never
    crash / fabricate). Need a cost col + an effect/abatement col + >=2 usable rows.
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    float ``estimates`` dict (plain floats only; nan for N/A), Chinese ``summary``
    with ⚠ disclosures, reproducible ``code`` string.
  * The profiler may classify an integer cost/effect column as ``count`` or ``id``
    (all-distinct integers) — numeric-column resolution accepts continuous/count/id.

EXTENDED-DOMINANCE ALGORITHM (STOP-AND-REPORT note, CEA):
  Extended (weak) dominance is ITERATIVE — removing one option can change another's
  ICER. We implement the STANDARD iterative algorithm used in health-economics
  textbooks (e.g. Drummond et al., "Methods for the Economic Evaluation of Health
  Care Programmes"):
    1. Drop STRONGLY dominated options (an option with >= cost AND <= effect than
       some other option, at least one strict — i.e. another option is at least as
       cheap and at least as effective).
    2. On the survivors sorted by effect ascending, compute consecutive ICERs.
       If any ICER is LOWER than the ICER of the immediately preceding segment
       (i.e. ICERs are not monotonically non-decreasing), the option at the start
       of the offending segment is EXTENDED-dominated (dominated by a linear
       combination of its neighbours) — remove it and RECOMPUTE ICERs.
    3. Repeat step 2 until all consecutive ICERs are non-decreasing.
  The frontier is then the set of survivors with strictly increasing ICERs.

Pure Python (numpy / pandas / matplotlib Agg). No heavy deps.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers (local to this module — no cross-branch imports)
# --------------------------------------------------------------------------- #
def _numeric_cols(ctx: Ctx) -> list[str]:
    """Column names usable as a cost / effect / abatement series.

    Accepts continuous, count AND id kinds: an integer money/effect column with all-
    distinct values is misclassified ``id`` by the profiler (CLAUDE.md「id 陷阱」), and
    a small integer stream profiles as ``count`` — both are legitimate numbers. The
    unit-label and time columns (fp.unit_col / fp.time_col) are excluded.
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


def _coerces_numeric(df, col: str) -> bool:
    import pandas as pd

    return pd.to_numeric(df[col], errors="coerce").notna().any()


def _pick_named_col(df, names: list[str], candidates: list[str]):
    """First column in ``candidates`` whose lower-cased name contains one of ``names``
    (and which coerces to a number). Returns the column name or None."""
    for cand in candidates:
        low = str(cand).lower()
        if any(tok in low for tok in names) and _coerces_numeric(df, cand):
            return cand
    return None


def _label_col(ctx: Ctx, cfg_key: str, used: set):
    """Resolve the option/measure label column: config ``cfg_key`` else fp.unit_col
    else the first non-numeric column not already used as cost/effect. May be None."""
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    lc = cfg.get(cfg_key)
    if lc in df.columns:
        return lc
    if fp.unit_col and fp.unit_col in df.columns and fp.unit_col not in used:
        return fp.unit_col
    return next(
        (c for c in df.columns if c not in used and not _coerces_numeric(df, c)),
        None,
    )


def _resolve_cost_effect(ctx: Ctx, effect_cfg_keys: list[str], effect_names: list[str]):
    """Resolve (cost_col, effect_col) for an options/measures table.

    cost  : config ``cost`` else a column named cost/price/expense.
    effect: config key in ``effect_cfg_keys`` (first present) else a column whose name
            matches one of ``effect_names``, else the first numeric column that is not
            the cost column.
    Returns (cost_col, effect_col) or (None, None) when either is unresolvable.
    """
    df, cfg = ctx.df, ctx.cfg
    nums = _numeric_cols(ctx)
    nums = [c for c in nums if _coerces_numeric(df, c)]

    # ---- cost ----
    cost_col = cfg.get("cost")
    if cost_col not in df.columns:
        cost_col = _pick_named_col(df, ["cost", "price", "expense"], nums)
    if cost_col not in df.columns:
        # fall back to the first numeric column
        cost_col = nums[0] if nums else None
    if cost_col is None:
        return None, None

    # ---- effect / abatement ----
    eff_col = None
    for k in effect_cfg_keys:
        v = cfg.get(k)
        if v in df.columns:
            eff_col = v
            break
    if eff_col not in df.columns:
        cands = [c for c in nums if c != cost_col]
        eff_col = _pick_named_col(df, effect_names, cands)
    if eff_col not in df.columns:
        eff_col = next((c for c in nums if c != cost_col), None)
    if eff_col is None or eff_col == cost_col:
        return None, None
    return cost_col, eff_col


# ===========================================================================
# cost_effectiveness_analysis — CEA / ICER on decision alternatives
#   Refs: Drummond et al., "Methods for the Economic Evaluation of Health Care
#         Programmes"; Neumann et al., "Cost-Effectiveness in Health and Medicine".
# ===========================================================================
def _cea_frontier(costs, effects):
    """Build the cost-effectiveness frontier from option (cost, effect) pairs.

    Implements the standard iterative dominance algorithm (see module docstring NOTE).
    ``costs`` / ``effects`` are length-n float arrays; option index i is positional.

    Returns a dict per ORIGINAL index i:
        status[i] in {"frontier", "dominated", "extended_dominated"}
        icer[i]   = incremental cost-effectiveness ratio vs the previous frontier
                    option (nan for the first frontier option / non-frontier options).
    The frontier options (status=="frontier") are sorted by effect ascending and have
    strictly increasing ICERs.
    """
    import numpy as np

    n = len(costs)
    status = ["frontier"] * n  # provisional; refined below
    icer = [float("nan")] * n

    idx = list(range(n))

    # ---- 1) strong dominance --------------------------------------------------
    # i is strongly dominated if some j != i has cost_j <= cost_i AND eff_j >= eff_i
    # with at least one strict inequality (j is at least as cheap AND as effective).
    strongly = set()
    for i in idx:
        for j in idx:
            if j == i:
                continue
            if (
                costs[j] <= costs[i] + 1e-12
                and effects[j] >= effects[i] - 1e-12
                and (costs[j] < costs[i] - 1e-12 or effects[j] > effects[i] + 1e-12)
            ):
                strongly.add(i)
                break
    for i in strongly:
        status[i] = "dominated"

    # survivors of strong dominance, sorted by effect asc (tie -> cost asc)
    survivors = [i for i in idx if i not in strongly]
    survivors.sort(key=lambda i: (effects[i], costs[i]))

    # ---- 2) iterative extended (weak) dominance ------------------------------
    extended = set()
    while True:
        cur = [i for i in survivors if i not in extended]
        # consecutive ICERs along the (effect-ascending) survivor chain
        icers = []
        for k in range(1, len(cur)):
            de = effects[cur[k]] - effects[cur[k - 1]]
            dc = costs[cur[k]] - costs[cur[k - 1]]
            icers.append(dc / de if abs(de) > 1e-12 else float("inf"))
        # find the first ICER that DECREASES vs the previous one -> the option at the
        # START of that segment (cur[k]) is extended-dominated by a linear combination.
        removed = False
        for k in range(1, len(icers)):
            if icers[k] < icers[k - 1] - 1e-12:
                extended.add(cur[k])  # cur[k] starts segment k (cur[k-1]->cur[k])
                removed = True
                break
        if not removed:
            break
    for i in extended:
        status[i] = "extended_dominated"

    # ---- 3) final frontier + incremental ICERs -------------------------------
    frontier = [i for i in survivors if i not in extended]
    frontier.sort(key=lambda i: (effects[i], costs[i]))
    for pos, i in enumerate(frontier):
        if pos == 0:
            icer[i] = float("nan")  # baseline (least effective) frontier option
        else:
            prev = frontier[pos - 1]
            de = effects[i] - effects[prev]
            dc = costs[i] - costs[prev]
            icer[i] = dc / de if abs(de) > 1e-12 else float("inf")
    return {"status": status, "icer": icer, "frontier": frontier}


@register("cost_effectiveness_analysis")
def _branch_cost_effectiveness_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cost_col, eff_col = _resolve_cost_effect(
        ctx,
        effect_cfg_keys=["effect"],
        effect_names=[
            "effect", "outcome", "benefit", "saving", "qaly", "co2", "water",
        ],
    )
    if cost_col is None or eff_col is None:
        summary.append(
            "cost_effectiveness_analysis 跳过：需要一个成本列(config cost 或名为 "
            "cost/price/expense 的列)和一个效果列(config effect 或名为 "
            "effect/outcome/benefit/saving/qaly/co2/water 的列)，未能同时识别到。"
        )
        return

    used = {cost_col, eff_col}
    opt_col = _label_col(ctx, "option", used)

    # ---- assemble the options table, drop unusable rows ---------------------
    cost = pd.to_numeric(df[cost_col], errors="coerce")
    eff = pd.to_numeric(df[eff_col], errors="coerce")
    if opt_col is not None and opt_col in df.columns:
        labels = df[opt_col].astype(str)
    else:
        labels = pd.Series([f"option_{i+1}" for i in range(len(df))], index=df.index)

    tbl = pd.DataFrame({"option": labels, "cost": cost, "effect": eff})
    # an effect of 0 makes ACER undefined and breaks the frontier slope — drop it.
    tbl = tbl[tbl["cost"].notna() & tbl["effect"].notna() & (tbl["effect"] != 0.0)]
    tbl = tbl.reset_index(drop=True)
    if len(tbl) < 2:
        summary.append(
            "cost_effectiveness_analysis 跳过：去除缺失/效果为 0 的行后，可比较的"
            f"备选方案不足 2 个（仅 {len(tbl)} 个）。CEA 需要至少 2 个备选方案。"
        )
        return

    try:
        costs = tbl["cost"].to_numpy(dtype=float)
        effects = tbl["effect"].to_numpy(dtype=float)
        n = len(tbl)

        # ---- ACER = cost / effect (average cost-effectiveness ratio) --------
        acer = costs / effects

        # ---- frontier + ICERs ----------------------------------------------
        fr = _cea_frontier(costs, effects)
        status = fr["status"]
        icer = fr["icer"]
        frontier = fr["frontier"]
        n_dominated = sum(1 for s in status if s != "frontier")

        # ---- optional WTP: optimum + net monetary benefit ------------------
        wtp = cfg.get("wtp")
        nmb = [float("nan")] * n
        best_opt_idx = None
        if wtp is not None:
            try:
                wtp_v = float(wtp)
                nmb = [wtp_v * effects[i] - costs[i] for i in range(n)]
                # optimal = MOST effective frontier option whose incremental ICER<=wtp
                # (walk the frontier from least to most effective; accept each step
                #  while its ICER<=wtp; the last accepted is the optimum).
                chosen = None
                for pos, i in enumerate(frontier):
                    if pos == 0:
                        chosen = i  # baseline always available
                    else:
                        if icer[i] <= wtp_v + 1e-9:
                            chosen = i
                        else:
                            break
                best_opt_idx = chosen
            except (TypeError, ValueError):
                wtp = None

        # ---- results table --------------------------------------------------
        out = pd.DataFrame(
            {
                "option": tbl["option"].to_numpy(),
                "cost": np.round(costs, 6),
                "effect": np.round(effects, 6),
                "acer": np.round(acer, 6),
                "status": status,
                "icer": [round(v, 6) if np.isfinite(v) else v for v in icer],
            }
        )
        if wtp is not None:
            out["nmb"] = [round(v, 6) if np.isfinite(v) else v for v in nmb]
        out.to_csv(d / "cost_effectiveness.csv", index=False)
        files.append("cost_effectiveness.csv")

        # ---- estimates (plain floats only; nan for N/A) --------------------
        estimates["n_options"] = float(n)
        estimates["n_dominated"] = float(n_dominated)
        estimates["best_acer"] = float(np.nanmin(acer)) if n else float("nan")
        for i in range(n):
            key = str(tbl["option"].iloc[i])
            if np.isfinite(icer[i]):
                estimates[f"icer__{key}"] = float(icer[i])
            if wtp is not None and np.isfinite(nmb[i]):
                estimates[f"nmb__{key}"] = float(nmb[i])
        if best_opt_idx is not None:
            estimates["optimal_effect"] = float(effects[best_opt_idx])
            estimates["optimal_cost"] = float(costs[best_opt_idx])

        # ---- cost-effectiveness plane (English labels) ---------------------
        def _build(plt):
            plt.figure(figsize=(7.5, 5.5))
            colors = {
                "frontier": "#2c7fb8",
                "dominated": "#bdbdbd",
                "extended_dominated": "#f03b20",
            }
            for st in ("dominated", "extended_dominated", "frontier"):
                xs = [effects[i] for i in range(n) if status[i] == st]
                ys = [costs[i] for i in range(n) if status[i] == st]
                if xs:
                    plt.scatter(xs, ys, c=colors[st], label=st.replace("_", " "),
                                s=60, zorder=3)
            # frontier line (effect-ascending)
            fx = [effects[i] for i in frontier]
            fy = [costs[i] for i in frontier]
            plt.plot(fx, fy, "-", color="#2c7fb8", lw=1.5, zorder=2,
                     label="CE frontier")
            for i in range(n):
                plt.annotate(str(tbl["option"].iloc[i]),
                             (effects[i], costs[i]),
                             textcoords="offset points", xytext=(5, 4), fontsize=8)
            plt.xlabel(f"Effect ({eff_col})")
            plt.ylabel(f"Cost ({cost_col})")
            plt.title("Cost-effectiveness plane (frontier & dominance)")
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)

        _save_fig(d, "cost_effectiveness_plane.png", files, _build)

        # ---- code string (reproducible) ------------------------------------
        code.append(
            "# cost_effectiveness_analysis — ACER / ICER frontier (iterative dominance)\n"
            f"cost_col, effect_col = {cost_col!r}, {eff_col!r}\n"
            "import numpy as np\n"
            "t = df[[cost_col, effect_col]].apply(pd.to_numeric, errors='coerce').dropna()\n"
            "t = t[t[effect_col] != 0]\n"
            "costs = t[cost_col].to_numpy(float); effects = t[effect_col].to_numpy(float)\n"
            "acer = costs / effects                       # average cost-effectiveness ratio\n"
            "idx = list(range(len(costs)))\n"
            "# 1) strong dominance: j at least as cheap AND as effective as i\n"
            "dom = {i for i in idx for j in idx if j!=i and costs[j]<=costs[i]\n"
            "       and effects[j]>=effects[i] and (costs[j]<costs[i] or effects[j]>effects[i])}\n"
            "surv = sorted([i for i in idx if i not in dom], key=lambda i:(effects[i],costs[i]))\n"
            "# 2) iterative extended dominance: drop options where ICER decreases, recompute\n"
            "ext = set()\n"
            "while True:\n"
            "    cur = [i for i in surv if i not in ext]\n"
            "    ic = [ (costs[cur[k]]-costs[cur[k-1]])/(effects[cur[k]]-effects[cur[k-1]])\n"
            "           for k in range(1,len(cur)) ]\n"
            "    bad = next((k for k in range(1,len(ic)) if ic[k] < ic[k-1]), None)\n"
            "    if bad is None: break\n"
            "    ext.add(cur[bad])\n"
            "# 3) frontier ICERs (consecutive survivors, effect-ascending)\n"
            "front = [i for i in surv if i not in ext]\n"
            "icer = {front[k]: (costs[front[k]]-costs[front[k-1]])/(effects[front[k]]-effects[front[k-1]])\n"
            "        for k in range(1,len(front))}\n"
            + (f"wtp = {float(cfg.get('wtp'))}\n"
               "nmb = wtp*effects - costs                 # net monetary benefit\n"
               if wtp is not None else "")
        )

        # ---- summary (Chinese, with ⚠ disclosures) -------------------------
        best_acer_idx = int(np.nanargmin(acer))
        best_acer_opt = str(tbl["option"].iloc[best_acer_idx])
        front_names = [str(tbl["option"].iloc[i]) for i in frontier]
        ext_names = [str(tbl["option"].iloc[i]) for i in range(n)
                     if status[i] == "extended_dominated"]
        dom_names = [str(tbl["option"].iloc[i]) for i in range(n)
                     if status[i] == "dominated"]
        lines = [
            f"成本-效果分析(CEA)：{n} 个备选方案，成本列={cost_col}，效果列={eff_col}。",
            f"平均成本-效果比(ACER=成本/效果)最低的方案：{best_acer_opt}"
            f"（ACER={acer[best_acer_idx]:.4f}）。",
            f"成本-效果前沿(按效果升序、ICER 递增)：{' → '.join(front_names)}。",
        ]
        # report each frontier ICER with its comparator
        for pos, i in enumerate(frontier):
            if pos == 0:
                continue
            prev = frontier[pos - 1]
            lines.append(
                f"  • {str(tbl['option'].iloc[i])} vs "
                f"{str(tbl['option'].iloc[prev])}（比较对象）："
                f"ICER=Δ成本/Δ效果="
                f"{(costs[i]-costs[prev]):.4g}/{(effects[i]-effects[prev]):.4g}"
                f"={icer[i]:.4f}。"
            )
        if dom_names:
            lines.append(f"强占优(被剔除)方案：{', '.join(dom_names)}"
                         "（另有方案不更贵且不更差）。")
        if ext_names:
            lines.append(f"扩展占优(弱占优、被剔除)方案：{', '.join(ext_names)}"
                         "（被相邻方案的线性组合占优——其 ICER 高于更有效方案）。")
        if wtp is not None and best_opt_idx is not None:
            wtp_v = float(cfg.get("wtp"))
            lines.append(
                f"给定支付意愿 WTP={wtp_v:g}/单位效果：最优方案="
                f"{str(tbl['option'].iloc[best_opt_idx])}"
                f"（前沿上 ICER≤WTP 的最有效方案；其净货币效益 "
                f"NMB=WTP·效果−成本={nmb[best_opt_idx]:.4f}）。"
            )
        lines.append(
            "⚠ ICER 的符号/象限决定其含义：更贵更有效的方案，其 ICER 必须对照 WTP 阈值"
            "判断是否值得；更便宜更有效的方案直接“占优”(dominates)对手，无需 ICER。"
            "本报告对每个前沿 ICER 都标注了它的比较对象(comparator)。"
        )
        lines.append(
            "⚠ ACER(平均比)只反映单方案的性价比，不能用于在方案间做边际取舍；"
            "决策应看前沿上的增量 ICER 与 WTP 的对比。"
        )
        import numpy as _np
        if _np.any(effects <= 0):
            n_neg = int(_np.count_nonzero(effects <= 0))
            lines.append(
                f"⚠ 有 {n_neg} 个方案的效果值 ≤0：ACER=成本/效果 与 ICER 在此象限**难以解释**"
                "（负效果意味着方案使结果变差，性价比为负或方向反转）——"
                "请核对效果列方向/单位，必要时剔除或重新编码后再读比率。"
            )
        summary.append(" ".join(lines))
    except Exception as exc:  # never crash the run
        summary.append(
            f"cost_effectiveness_analysis 跳过：计算时出错"
            f"（{type(exc).__name__}: {exc}）。"
        )
        return


# ===========================================================================
# marginal_abatement_cost — MACC on a portfolio of interventions
#   Refs: McKinsey "Global GHG Abatement Cost Curve"; Kesicki & Ekins (2012),
#         "Marginal abatement cost curves: a call for caution".
# ===========================================================================
@register("marginal_abatement_cost")
def _branch_marginal_abatement_cost(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cost_col, ab_col = _resolve_cost_effect(
        ctx,
        effect_cfg_keys=["abatement"],
        effect_names=["abatement", "saving", "reduction", "co2", "avoided"],
    )
    if cost_col is None or ab_col is None:
        summary.append(
            "marginal_abatement_cost 跳过：需要一个成本列(config cost 或名为 "
            "cost/price/expense 的列)和一个减排/节约列(config abatement 或名为 "
            "abatement/saving/reduction/co2/avoided 的列)，未能同时识别到。"
        )
        return

    used = {cost_col, ab_col}
    meas_col = _label_col(ctx, "measure", used)

    cost = pd.to_numeric(df[cost_col], errors="coerce")
    ab = pd.to_numeric(df[ab_col], errors="coerce")
    if meas_col is not None and meas_col in df.columns:
        labels = df[meas_col].astype(str)
    else:
        labels = pd.Series([f"measure_{i+1}" for i in range(len(df))], index=df.index)

    tbl = pd.DataFrame({"measure": labels, "cost": cost, "abatement": ab})
    n_raw = len(tbl)
    tbl = tbl[tbl["cost"].notna() & tbl["abatement"].notna()]
    # GUARD: drop measures with abatement <= 0 — a per-unit cost is undefined there.
    n_before_pos = len(tbl)
    tbl = tbl[tbl["abatement"] > 0.0].reset_index(drop=True)
    n_dropped_ab = n_before_pos - len(tbl)
    if len(tbl) < 2:
        summary.append(
            "marginal_abatement_cost 跳过：去除缺失及减排量≤0 的行后，可排序的"
            f"措施不足 2 个（仅 {len(tbl)} 个）。MACC 需要至少 2 项可减排措施。"
        )
        return

    try:
        costs = tbl["cost"].to_numpy(dtype=float)
        ab_arr = tbl["abatement"].to_numpy(dtype=float)
        n = len(tbl)

        # ---- MAC = cost / abatement (cost may be negative = no-regret) ------
        mac = costs / ab_arr

        # ---- sort ascending by MAC -> cumulative abatement -----------------
        order = np.argsort(mac, kind="mergesort")  # stable
        out = pd.DataFrame(
            {
                "measure": tbl["measure"].to_numpy()[order],
                "cost": np.round(costs[order], 6),
                "abatement": np.round(ab_arr[order], 6),
                "mac": np.round(mac[order], 6),
            }
        )
        out["cumulative_abatement"] = np.round(np.cumsum(ab_arr[order]), 6)
        out["no_regret"] = costs[order] < 0.0  # negative-cost = net-saving
        out.to_csv(d / "macc_curve.csv", index=False)
        files.append("macc_curve.csv")

        n_no_regret = int(np.sum(costs < 0.0))
        total_ab = float(np.sum(ab_arr))
        total_cost = float(np.sum(costs))

        # ---- optional carbon price threshold -------------------------------
        price = cfg.get("carbon_price")
        if price is None:
            price = cfg.get("price")
        ab_below = float("nan")
        cost_below = float("nan")
        if price is not None:
            try:
                price_v = float(price)
                mask = mac <= price_v + 1e-12
                ab_below = float(np.sum(ab_arr[mask]))
                cost_below = float(np.sum(costs[mask]))
            except (TypeError, ValueError):
                price = None

        # ---- estimates (plain floats only; nan for N/A) --------------------
        estimates["n_measures"] = float(n)
        estimates["n_no_regret"] = float(n_no_regret)
        estimates["total_abatement"] = total_ab
        estimates["total_cost"] = total_cost
        if price is not None:
            estimates["abatement_below_price"] = ab_below
            estimates["cost_below_price"] = cost_below
        for i in range(n):
            key = str(tbl["measure"].iloc[i])
            estimates[f"mac__{key}"] = float(mac[i])

        # ---- MACC step-bar chart (width=abatement, height=MAC) -------------
        def _build(plt):
            plt.figure(figsize=(9, 5.5))
            widths = ab_arr[order]
            heights = mac[order]
            left = np.concatenate([[0.0], np.cumsum(widths)[:-1]])
            colors = ["#2ca25f" if c < 0 else "#43a2ca" for c in costs[order]]
            plt.bar(left, heights, width=widths, align="edge",
                    color=colors, edgecolor="white", linewidth=0.6)
            for li, w, h, nm in zip(left, widths, heights,
                                    tbl["measure"].to_numpy()[order]):
                plt.text(li + w / 2.0, h, str(nm), rotation=90,
                         va="bottom" if h >= 0 else "top", ha="center", fontsize=7)
            plt.axhline(0.0, color="#444", lw=0.8)
            if price is not None:
                plt.axhline(float(price), color="#d95f0e", lw=1.0, ls="--",
                            label=f"price = {float(price):g}")
                plt.legend(fontsize=8)
            plt.xlabel(f"Cumulative abatement ({ab_col})")
            plt.ylabel(f"Marginal abatement cost (MAC, {cost_col}/unit)")
            plt.title("Marginal abatement cost curve (MACC)")
            plt.grid(axis="y", alpha=0.3)

        _save_fig(d, "macc_curve.png", files, _build)

        # ---- code string (reproducible) ------------------------------------
        code.append(
            "# marginal_abatement_cost — MAC ranking + cumulative-abatement step curve\n"
            f"cost_col, ab_col = {cost_col!r}, {ab_col!r}\n"
            "import numpy as np\n"
            "t = df[[cost_col, ab_col]].apply(pd.to_numeric, errors='coerce').dropna()\n"
            "t = t[t[ab_col] > 0]                          # drop abatement<=0 (no per-unit cost)\n"
            "costs = t[cost_col].to_numpy(float); ab = t[ab_col].to_numpy(float)\n"
            "mac = costs / ab                              # marginal abatement cost\n"
            "order = np.argsort(mac)                       # sort ascending by MAC\n"
            "cumulative = np.cumsum(ab[order])\n"
            "no_regret = costs[order] < 0                  # negative-cost measures\n"
            + (f"price = {float(price)}\n"
               "abatement_below_price = ab[mac <= price].sum()\n"
               "cost_below_price = costs[mac <= price].sum()\n"
               if price is not None else "")
        )

        # ---- summary (Chinese, with ⚠ disclosures) -------------------------
        cheapest_idx = int(np.argmin(mac))
        cheapest = str(tbl["measure"].iloc[cheapest_idx])
        ranked = out["measure"].astype(str).tolist()
        lines = [
            f"边际减排成本曲线(MACC)：{n} 项措施，成本列={cost_col}，"
            f"减排/节约列={ab_col}。",
            f"按 MAC=成本/减排量 升序排列：{' < '.join(ranked)}（最便宜=最先采用）。",
            f"最低 MAC 措施：{cheapest}（MAC={mac[cheapest_idx]:.4f}）。",
            f"总可减排量={total_ab:.4g}，全部采用的总成本={total_cost:.4g}。",
        ]
        if n_no_regret > 0:
            nr_names = [str(tbl["measure"].iloc[i]) for i in range(n) if costs[i] < 0.0]
            lines.append(
                f"“无悔”(负成本、净节约)措施 {n_no_regret} 项：{', '.join(nr_names)}"
                "（MAC<0，即使不计减排收益也省钱，应优先采用）。"
            )
        if n_dropped_ab > 0:
            lines.append(
                f"⚠ 已剔除 {n_dropped_ab} 项减排量≤0 的措施（无法计算单位减排成本）。"
            )
        if price is not None:
            price_v = float(price)
            lines.append(
                f"给定碳价/阈值={price_v:g}/单位：MAC≤该价的措施合计可减排 "
                f"{ab_below:.4g}，对应总成本 {cost_below:.4g}。"
            )
        lines.append(
            "⚠ MACC 忽略措施之间的相互作用与采用顺序（一项措施的减排量/成本可能依赖"
            "另一项是否已采用），是静态分析；且对成本/减排量的边界(系统边界、基准线、"
            "贴现、寿命)假设非常敏感——同样的措施换一套假设，排序可能改变。"
        )
        summary.append(" ".join(lines))
    except Exception as exc:  # never crash the run
        summary.append(
            f"marginal_abatement_cost 跳过：计算时出错"
            f"（{type(exc).__name__}: {exc}）。"
        )
        return
