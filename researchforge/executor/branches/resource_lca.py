"""Branch handler for FOOTPRINT ANALYSIS (LCA-style) — resource family / sustainability.

One method — ``footprint_analysis`` — the LIFE-CYCLE INVENTORY / footprint piece of
an Energy-Water-Food (EWF) nexus & sustainability toolkit. It pairs with
``life_cycle_cost`` (techno_economic) to complete the life-cycle pair: footprint =
the PHYSICAL/environmental burden of an activity; LCC = the monetary cost of owning
it.

(SEPARATE agents own ``resource.py`` / ``resource_io.py`` / ``resource_mfa.py`` /
``resource_econ.py`` and their yamls; all coexist in the "resource" family via
catalog auto-discovery — this module only owns ``footprint_analysis`` and never
touches the others.)

WHAT IT DOES
------------
A footprint is the classic LCA-style inventory aggregation:

    footprint_a = quantity_a  x  factor_a            (per activity a)
    TOTAL       = sum_a footprint_a
    share_a     = footprint_a / TOTAL                (contribution / hotspot analysis)
    intensity   = TOTAL / sum(functional-unit column)  (per functional unit, optional)

i.e. each activity's physical quantity (energy used, km driven, kg of material, ...)
is multiplied by an EMISSION / INTENSITY FACTOR (e.g. kg CO2e per kWh, L water per
kg) and the burdens are summed. Contribution (hotspot) analysis ranks activities by
their share of the total so the dominant contributors are visible.

FOOTPRINTS REQUIRE FACTORS - the system NEVER fabricates them. If no factor source
is resolvable we degrade honestly (Chinese "跳过") and tell the user how to supply them.

FACTOR RESOLUTION (priority order - see ``_resolve_footprint``)
--------------------------------------------------------------
1. config ``factors`` = a dict {activity_column_name: factor}. The PRIMARY, least
   ambiguous form (STOP-AND-REPORT: dict form is primary). Each named column is an
   activity quantity column; its column-sum is multiplied by the scalar factor. This
   is the wide-table form (one column per activity, one factor each).
2. LONG-format table: config ``activity`` (label column), ``quantity`` (numeric
   amount column) and ``factor`` (numeric per-row factor column) - one row per
   activity carrying its own quantity and factor. footprint per row = quantity x
   factor; activities are the ``activity`` labels.
3. config ``factor_column`` naming a per-row factor column, multiplied against the
   config ``quantity`` column (per-row footprint = quantity x factor_column);
   activities are the rows (labelled by the unit/label column if available).

If none resolve -> honest degrade.

PRODUCTS
--------
  * a contribution bar chart PNG (footprint by activity, sorted desc, ENGLISH labels)
  * a CSV: activity, quantity, factor, footprint, share_pct
  * estimates: total_footprint, intensity_per_unit (nan if no functional unit),
    n_activities, top_contributor_share, footprint__<activity> per activity.
  * Chinese summary with warning disclosures (footprint quality = factor quality;
    factors are user-supplied; system boundary & allocation choices matter; ONE
    impact category per run - rerun with water/energy factors for those).

Pure Python (numpy / pandas / matplotlib Agg, ENGLISH plot labels). No R, no heavy
deps. Refs: ISO 14040/14044 (LCA principles); GHG Protocol (activity x emission
factor); water/material footprint methodology (Hoekstra; Wiedmann & Minx).
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _to_float(v):
    """Best-effort scalar float (None / non-numeric -> None; NaN rejected)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # reject NaN


def _safe_label(name: str) -> str:
    """A short, plot-safe activity label. The CSV keeps the original; only the PNG
    x-tick is truncated for legibility."""
    s = str(name)
    return s if len(s) <= 24 else (s[:21] + "...")


def _resolve_footprint(ctx: Ctx):
    """Resolve per-activity footprints from one of the three factor sources.

    Returns (rows, source_note, None) on success where ``rows`` is a list of dicts
    {"activity": str, "quantity": float, "factor": float, "footprint": float}, or
    (None, None, err_msg) on honest failure (no resolvable factors).
    """
    import pandas as pd

    df, cfg, fp = ctx.df, ctx.cfg, ctx.fp

    def _num_series(col):
        return pd.to_numeric(df[col], errors="coerce")

    # ---- (1) config factors = {activity_column: factor} (PRIMARY, wide form) ---- #
    factors = cfg.get("factors")
    if isinstance(factors, dict) and factors:
        rows = []
        used = []
        for col, fac in factors.items():
            if col not in df.columns:
                continue
            facf = _to_float(fac)
            if facf is None:
                continue
            qty = float(_num_series(col).fillna(0.0).sum())
            rows.append({
                "activity": str(col),
                "quantity": qty,
                "factor": facf,
                "footprint": qty * facf,
            })
            used.append(col)
        if rows:
            note = (
                f"因子来源=config factors（宽表：每列一个活动，列总量×因子）；"
                f"用到的活动列：{', '.join(used)}。"
            )
            return rows, note, None
        return None, None, (
            "footprint 需要排放/强度因子：config factors 字典里没有匹配到任何数据列"
            "（键应是活动列名、值是该列的因子）。"
        )

    # ---- (2) long-format: activity / quantity / factor columns ---- #
    act_col = cfg.get("activity")
    qty_col = cfg.get("quantity")
    fac_col = cfg.get("factor")
    if act_col in df.columns and qty_col in df.columns and fac_col in df.columns:
        q = _num_series(qty_col)
        f = _num_series(fac_col)
        rows = []
        for i in range(len(df)):
            qi = _to_float(q.iloc[i])
            fi = _to_float(f.iloc[i])
            if qi is None or fi is None:
                continue
            rows.append({
                "activity": str(df[act_col].iloc[i]),
                "quantity": qi,
                "factor": fi,
                "footprint": qi * fi,
            })
        if rows:
            note = (
                f"因子来源=长表（每行一个活动）：活动列={act_col}、数量列={qty_col}、"
                f"因子列={fac_col}；逐行 footprint=数量×因子。"
            )
            return rows, note, None
        return None, None, (
            f"footprint 长表（{act_col}/{qty_col}/{fac_col}）没有任何有效的"
            "数量×因子行（数量或因子全为缺失/非数值）。"
        )

    # ---- (3) factor_column x quantity column (per-row) ---- #
    fac_col2 = cfg.get("factor_column")
    qty_col2 = cfg.get("quantity")
    if fac_col2 in df.columns and qty_col2 in df.columns:
        q = _num_series(qty_col2)
        f = _num_series(fac_col2)
        label_col = (
            fp.unit_col if (fp.unit_col and fp.unit_col in df.columns)
            else (fp.time_col if (fp.time_col and fp.time_col in df.columns) else None)
        )
        rows = []
        for i in range(len(df)):
            qi = _to_float(q.iloc[i])
            fi = _to_float(f.iloc[i])
            if qi is None or fi is None:
                continue
            lab = str(df[label_col].iloc[i]) if label_col else f"row_{i}"
            rows.append({
                "activity": lab,
                "quantity": qi,
                "factor": fi,
                "footprint": qi * fi,
            })
        if rows:
            note = (
                f"因子来源=逐行因子列：因子列={fac_col2}、数量列={qty_col2}"
                f"（逐行 footprint=数量×因子"
                f"{('，行标签取 ' + label_col) if label_col else ''}）。"
            )
            return rows, note, None
        return None, None, (
            f"footprint 逐行（因子列 {fac_col2} × 数量列 {qty_col2}）没有任何有效行。"
        )

    return None, None, (
        "footprint 需要排放/强度因子：用 config factors={列:因子}（宽表），"
        "或提供 activity/quantity/factor 三列（长表），"
        "或 factor_column + quantity 两列（逐行因子）。系统不会凭空生成因子。"
    )


def _save_fig(d, fname, files, build):
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


# Identifier guard for estimate keys derived from activity names (keep dict keys tame).
def _est_key(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return s if s else "x"


# ===========================================================================
# footprint_analysis - LCA-style carbon / water / material footprint
#   footprint = sum (quantity x factor); contribution (hotspot) analysis;
#   intensity per functional unit (optional).
# ===========================================================================
@register("footprint_analysis")
def _branch_footprint_analysis(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    rows, src_note, err = _resolve_footprint(ctx)
    if err is not None:
        summary.append(f"足迹分析(footprint) 跳过：{err}")
        return
    try:
        df, cfg, fp = ctx.df, ctx.cfg, ctx.fp

        activities = [r["activity"] for r in rows]
        quantities = np.array([r["quantity"] for r in rows], dtype=float)
        rfactors = np.array([r["factor"] for r in rows], dtype=float)
        foots = np.array([r["footprint"] for r in rows], dtype=float)

        total = float(foots.sum())
        n_act = int(len(rows))

        if not np.isfinite(total) or total == 0.0:
            # all footprints zero (or non-finite) -> shares undefined; honest degrade
            summary.append(
                f"足迹分析(footprint) 跳过：总足迹={total:.6g}（为零或非有限），"
                "贡献占比无定义（请确认数量与因子非全零）。"
            )
            return

        shares = foots / total  # contribution share per activity (fraction)

        # ---- intensity per functional unit (optional) ---- #
        fu_col = cfg.get("functional_unit")
        fu_total = None
        intensity = float("nan")
        if fu_col in df.columns:
            fu_sum = float(pd.to_numeric(df[fu_col], errors="coerce").fillna(0.0).sum())
            if fu_sum != 0.0 and np.isfinite(fu_sum):
                fu_total = fu_sum
                intensity = total / fu_sum
            # zero/invalid denominator -> leave intensity nan, report absolute only

        # ---- hotspots (sorted desc by footprint) ---- #
        order = np.argsort(-foots)
        sorted_acts = [activities[i] for i in order]
        sorted_foots = foots[order]
        sorted_shares = shares[order]
        sorted_qty = quantities[order]
        sorted_fac = rfactors[order]
        top_share = float(sorted_shares[0]) if n_act > 0 else float("nan")

        # ---- CSV (activity, quantity, factor, footprint, share_pct), sorted desc ---- #
        tbl = pd.DataFrame({
            "activity": sorted_acts,
            "quantity": np.round(sorted_qty, 8),
            "factor": np.round(sorted_fac, 10),
            "footprint": np.round(sorted_foots, 8),
            "share_pct": np.round(sorted_shares * 100.0, 6),
        })
        tbl.to_csv(d / "footprint_contribution.csv", index=False, encoding="utf-8")
        files.append("footprint_contribution.csv")

        # ---- estimates ---- #
        estimates.update({
            "total_footprint": round(total, 8),
            "intensity_per_unit": round(intensity, 10) if intensity == intensity else float("nan"),
            "n_activities": float(n_act),
            "top_contributor_share": round(top_share, 8) if top_share == top_share else float("nan"),
        })
        # per-activity footprint (plain floats). Guard against duplicate sanitised keys.
        seen: dict[str, int] = {}
        for act, fv in zip(activities, foots):
            base = _est_key(act)
            if base in seen:
                seen[base] += 1
                key = f"footprint__{base}_{seen[base]}"
            else:
                seen[base] = 0
                key = f"footprint__{base}"
            estimates[key] = round(float(fv), 8)

        # ---- contribution bar chart (sorted desc, top 20 for legibility) ---- #
        def _plot(plt):
            k = min(20, n_act)
            labs = [_safe_label(a) for a in sorted_acts[:k]]
            vals = sorted_foots[:k]
            fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * k + 2.0), 4.4))
            ax.bar(np.arange(k), vals, color="#55A868", edgecolor="white")
            ax.set_xticks(np.arange(k))
            ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("footprint")
            ax.set_xlabel("activity (sorted by contribution)")
            top_lab = _safe_label(sorted_acts[0])
            ax.set_title(
                f"Footprint contribution (total={total:.4g}, "
                f"top='{top_lab}' {top_share:.1%})"
            )

        _save_fig(d, "footprint_contribution.png", files, _plot)

        # ---- Chinese summary ---- #
        hotspot_txt = "、".join(
            f"{sorted_acts[i]}({sorted_shares[i]:.1%})" for i in range(min(3, n_act))
        )
        intensity_txt = (
            f"功能单位强度={intensity:.6g}（总足迹/{fu_col} 总量 {fu_total:.6g}）"
            if (fu_total is not None and intensity == intensity)
            else ("功能单位强度=不适用（未提供 config functional_unit 或其总量为零，"
                  "仅报绝对足迹）")
        )
        summary.append(
            f"{ctx.entry.method} 完成：{src_note} 共 {n_act} 项活动；"
            f"总足迹={total:.6g}。{intensity_txt}。"
            f"贡献(热点)排序前几位：{hotspot_txt}；最大贡献占比={top_share:.1%}。"
            "明细见 footprint_contribution.csv、贡献条形图见 footprint_contribution.png。"
            " ⚠ 足迹质量=因子质量：因子由用户提供，系统不会凭空生成或校核其数值；"
            "请确保数量与因子的单位口径一致（如 kWh × kgCO2e/kWh）。"
            " ⚠ 结果取决于系统边界与分摊(allocation)选择（计入哪些活动、共生产品如何分摊）——这是 LCA 的关键主观假定。"
            " ⚠ 每次运行只算一个影响类别(single impact category)：本次用的是所给因子对应的类别；"
            "水/能源等其它足迹请换用对应因子重跑。贡献分析只指出热点占比，不代表可削减性。"
        )
        code += [
            "import numpy as np",
            "footprint = quantity * factor          # 每活动足迹 = 数量 × 因子",
            "total = footprint.sum()                # 总足迹",
            "share = footprint / total             # 贡献(热点)占比",
            "intensity = total / functional_unit_total   # 功能单位强度(可选)",
        ]
    except Exception as exc:
        summary.append(f"足迹分析(footprint) 计算失败：{exc}")
