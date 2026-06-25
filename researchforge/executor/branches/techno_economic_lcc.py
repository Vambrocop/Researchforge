"""Branch handler for LIFE-CYCLE COST (LCC / total cost of ownership) - techno_economic
family. The life-cycle-COST piece of the life-cycle pair: it complements
``footprint_analysis`` (resource, physical burden) and ``npv_irr`` (net benefit) by
comparing alternatives on the discounted COST of owning an asset over its life.

(SEPARATE agents own ``techno_economic.py`` and its yaml - npv_irr / cost_benefit /
breakeven_analysis / sensitivity_tornado / monte_carlo_cashflow / lcoe; all coexist
in the "techno_economic" family via catalog auto-discovery - this module only owns
``life_cycle_cost`` and never touches the others.)

WHAT IT DOES
------------
Total discounted cost of ownership over an asset's life:

    LCC = capex  +  sum_{t=1..N} opex_t / (1+r)^t  -  salvage / (1+r)^N

i.e. the up-front capital cost plus the present value of recurring operating /
maintenance costs minus the present value of any end-of-life salvage (residual)
value. From it we also report:

  * Equivalent Annual Cost  EAC = LCC x CRF
        CRF (capital recovery factor) = r(1+r)^N / ((1+r)^N - 1)
        guard: r == 0  ->  CRF = 1/N   (limit as r -> 0)
    EAC is the constant per-period cost whose present value over N periods equals the
    LCC - it lets alternatives with DIFFERENT lifetimes be compared on a like basis.
  * Cost-phase breakdown: capex %, opex-PV %, disposal/salvage % (salvage is a
    negative cost contribution; the breakdown is computed over absolute magnitudes so
    the bars are readable, and signs are disclosed).
  * Discounted total (LCC) vs undiscounted total (capex + sum opex - salvage).

INPUT RESOLUTION (see body; STOP-AND-REPORT: prefer an explicit ``opex`` column)
-------------------------------------------------------------------------------
CAPEX (up-front capital cost), in priority:
  1. config ``capex`` scalar.
  2. a column named capex / capital / investment (its first non-null value).
  3. the first period's cost (cost[0]) when opex comes from a ``cost`` cashflow.

OPEX (recurring per-period operating / maintenance cost), in priority:
  1. config ``opex`` column = per-period opex values (PREFERRED, least ambiguous).
  2. config ``cost`` column = a cashflow where row 0 = capex and rows 1..N = opex.
  3. config scalar ``annual_opex`` repeated for N periods.

OTHER config:
  * ``rate``    - discount rate per period (default 0.08), guarded > -1.
  * ``periods`` / ``life`` - number of operating periods N (else inferred from rows).
  * ``salvage`` - end-of-life residual value (default 0; received at period N).

Honest degrade ("跳过") when neither a capex nor any opex source is resolvable -
never crash / fabricate.

PRODUCTS
--------
  * a cost-breakdown bar/pie PNG (capex / opex PV / disposal-salvage; ENGLISH labels)
  * a CSV: phase, present_value, share_pct
  * estimates: lcc, eac, capex, opex_pv, salvage_pv, rate, n_periods (plain floats)
  * Chinese summary with warning disclosures (LCC compares COST not value - only fair
    when alternatives deliver the same output/service; sensitive to discount rate &
    assumed life; reuse npv_irr / cost_benefit when benefits differ).

Pure Python (numpy / pandas / matplotlib Agg, ENGLISH plot labels). No R, no heavy
deps. Refs: ISO 15686-5 (LCC for buildings); Fuller & Petersen NIST 135 (LCC
manual); Park "Contemporary Engineering Economics" (EAC / capital recovery factor).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _scalar(cfg, key):
    """A finite float from config[key], else None."""
    v = cfg.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _rate(ctx, default: float = 0.08) -> float:
    """Discount rate from config ``rate`` (per period), default 0.08, guarded > -1."""
    r = _scalar(ctx.cfg, "rate")
    if r is None:
        return default
    return r if r > -1.0 else default


def _crf(rate: float, n: int) -> float:
    """Capital recovery factor CRF = r(1+r)^N / ((1+r)^N - 1); r==0 -> 1/N."""
    if n <= 0:
        return float("nan")
    if rate == 0.0:
        return 1.0 / n
    g = (1.0 + rate) ** n
    denom = g - 1.0
    if denom == 0.0:
        return 1.0 / n
    return rate * g / denom


def _opex_column(df, name):
    """Per-period opex values from a column (NaN -> 0), as a float list."""
    import pandas as pd

    s = pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return [float(x) for x in s.to_numpy()]


def _capex_column(df):
    """First non-null value of a column named capex/capital/investment, else None."""
    import pandas as pd

    for cand in ("capex", "capital", "investment"):
        for col in df.columns:
            if str(col).strip().lower() == cand:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s):
                    return float(s.iloc[0])
    return None


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


# ===========================================================================
# life_cycle_cost - LCC / total cost of ownership + EAC + cost-phase breakdown
# ===========================================================================
@register("life_cycle_cost")
def _branch_life_cycle_cost(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    df, cfg = ctx.df, ctx.cfg
    try:
        rate = _rate(ctx)

        # ---- resolve OPEX (per-period operating/maintenance) ---- #
        opex = None              # list of per-period opex (length = N operating periods)
        capex = _scalar(cfg, "capex")   # may be overridden below
        opex_note = ""

        opex_col = cfg.get("opex")
        cost_col = cfg.get("cost")
        annual = _scalar(cfg, "annual_opex")

        if opex_col in df.columns:
            opex = _opex_column(df, opex_col)
            opex_note = f"opex 来源=列 {opex_col}（每期运维成本）"
        elif cost_col in df.columns:
            # cashflow column: row 0 = capex, rows 1..N = opex
            series = _opex_column(df, cost_col)
            if len(series) >= 1 and capex is None:
                capex = series[0]
            opex = series[1:] if len(series) > 1 else []
            opex_note = (
                f"opex 来源=cost 现金流列 {cost_col}（行 0=capex、行 1..N=opex）"
            )
        elif annual is not None:
            # scalar annual opex repeated for N periods (N from periods/life/rows)
            n_cfg = _scalar(cfg, "periods")
            if n_cfg is None:
                n_cfg = _scalar(cfg, "life")
            n = int(n_cfg) if (n_cfg is not None and n_cfg > 0) else max(1, len(df))
            opex = [annual] * n
            opex_note = f"opex 来源=标量 annual_opex={annual:g}，重复 {n} 期"

        # ---- resolve CAPEX (if still unknown) ---- #
        if capex is None:
            capex = _capex_column(df)
            if capex is not None:
                opex_note = (opex_note + "；" if opex_note else "") + "capex 来源=capex/capital/investment 列"

        # honest degrade: need at least one of capex / opex
        has_opex = opex is not None and len(opex) > 0
        if capex is None and not has_opex:
            summary.append(
                "全生命周期成本(LCC) 跳过：需要资本性支出与/或每期运维成本——"
                "用 config capex（标量）+ config opex（每期成本列）"
                "，或 config cost（行0=capex、行1..N=opex 的现金流列）"
                "，或 config annual_opex（标量）。"
            )
            return

        if capex is None:
            capex = 0.0
            opex_note = (opex_note + "；" if opex_note else "") + "未提供 capex，按 0 处理"
        opex = opex or []

        # ---- number of operating periods N ---- #
        n_cfg = _scalar(cfg, "periods")
        if n_cfg is None:
            n_cfg = _scalar(cfg, "life")
        if n_cfg is not None and n_cfg > 0:
            n = int(n_cfg)
            # pad / truncate opex to N periods
            if len(opex) < n:
                opex = opex + [opex[-1] if opex else 0.0] * (n - len(opex))
            elif len(opex) > n:
                opex = opex[:n]
        else:
            n = len(opex)
        if n <= 0:
            n = 1
            if not opex:
                opex = [0.0]

        opex_arr = np.array(opex[:n], dtype=float)
        salvage = _scalar(cfg, "salvage") or 0.0

        # ---- present values: opex over t=1..N, salvage at t=N ---- #
        t = np.arange(1, n + 1)
        disc = 1.0 / (1.0 + rate) ** t
        opex_pv = float(np.sum(opex_arr * disc))
        salvage_pv = float(salvage / (1.0 + rate) ** n)

        lcc = capex + opex_pv - salvage_pv
        crf = _crf(rate, n)
        eac = lcc * crf

        undisc_total = float(capex + opex_arr.sum() - salvage)

        # ---- cost-phase breakdown (over absolute magnitudes so bars are readable) -- #
        comp_capex = float(capex)
        comp_opex = float(opex_pv)
        comp_salvage = float(salvage_pv)  # a cost REDUCTION (disposal credit)
        denom = abs(comp_capex) + abs(comp_opex) + abs(comp_salvage)

        def _share(x):
            return (abs(x) / denom * 100.0) if denom > 0 else float("nan")

        import pandas as pd
        phases = pd.DataFrame({
            "phase": ["capex", "opex_pv", "salvage_pv (credit)"],
            "present_value": [round(comp_capex, 6), round(comp_opex, 6), round(-comp_salvage, 6)],
            "share_pct": [round(_share(comp_capex), 4), round(_share(comp_opex), 4),
                          round(_share(comp_salvage), 4)],
        })
        phases.to_csv(d / "lcc_breakdown.csv", index=False, encoding="utf-8")
        files.append("lcc_breakdown.csv")

        # ---- estimates (plain floats) ---- #
        estimates.update({
            "lcc": round(lcc, 6),
            "eac": round(eac, 6),
            "capex": round(comp_capex, 6),
            "opex_pv": round(comp_opex, 6),
            "salvage_pv": round(comp_salvage, 6),
            "rate": round(rate, 6),
            "n_periods": float(n),
        })

        # ---- cost-breakdown chart (bar of PV magnitudes) ---- #
        def _plot(plt):
            labels = ["capex", "opex PV", "salvage credit"]
            vals = [comp_capex, comp_opex, -comp_salvage]
            colors = ["#4C72B0", "#C44E52", "#55A868"]
            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            ax.bar(np.arange(3), vals, color=colors, edgecolor="white")
            ax.axhline(0.0, color="#333333", lw=0.8)
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("present value (cost)")
            ax.set_title(
                f"Life-cycle cost breakdown (LCC={lcc:.4g}, EAC={eac:.4g}, r={rate:.1%})"
            )

        _save_fig(d, "lcc_breakdown.png", files, _plot)

        # ---- Chinese summary ---- #
        salv_txt = (f"减去期末残值现值 {salvage_pv:.6g}（salvage={salvage:g}）"
                    if salvage != 0.0 else "无期末残值")
        summary.append(
            f"{ctx.entry.method} 完成：{opex_note}；折现率 r={rate:.2%}，运营 {n} 期。"
            f"资本支出 capex={comp_capex:.6g}、运维现值 opex_PV={comp_opex:.6g}、{salv_txt}。"
            f"全生命周期成本 LCC=capex+ΣopexPV−salvagePV={lcc:.6g}；"
            f"等额年成本 EAC=LCC×CRF={eac:.6g}（CRF={crf:.6g}{'（r=0→1/N）' if rate == 0.0 else ''}）。"
            f"折现总成本(LCC)={lcc:.6g} vs 不折现总成本={undisc_total:.6g}。"
            "成本阶段构成见 lcc_breakdown.csv 与图。"
            " ⚠ LCC 比较的是「成本」而非「价值」：只有当各备选方案产出/服务水平相同时直接比 LCC 才公平；"
            "产出不同的方案请改用 NPV/IRR（net benefit）或 cost_benefit（效益成本比）。"
            " ⚠ LCC/EAC 对折现率 r 与假定寿命 N 高度敏感（小改动可能改变方案排序，建议做敏感性分析）；"
            "EAC 用于比较寿命不同的方案（化为等额年成本）。残值为成本抵减项（disposal credit），"
            "占比图按现值绝对值计算以便阅读、符号已在 CSV 标明。"
        )
        code += [
            "import numpy as np",
            "t = np.arange(1, N+1); disc = 1/(1+r)**t",
            "opex_pv = np.sum(opex * disc)                 # 运维现值",
            "salvage_pv = salvage / (1+r)**N               # 残值现值",
            "lcc = capex + opex_pv - salvage_pv            # 全生命周期成本",
            "crf = r*(1+r)**N / ((1+r)**N - 1) if r else 1/N   # 资本回收系数",
            "eac = lcc * crf                               # 等额年成本",
        ]
    except Exception as exc:
        summary.append(f"全生命周期成本(LCC) 计算失败：{exc}")
