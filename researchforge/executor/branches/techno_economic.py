"""Branch handlers for the TECHNO-ECONOMIC ANALYSIS family (engineering economics
/ project appraisal). High demand for energy / engineering / policy / ag-econ work.

Six coherent project-appraisal methods, all built on a per-period CASH-FLOW series:

  - npv_irr             — Net Present Value + Internal Rate of Return + payback periods.
  - cost_benefit        — discounted Benefit-Cost Ratio (BCR) + net present benefit.
  - breakeven_analysis  — break-even quantity / price, contribution margin, safety margin.
  - sensitivity_tornado — one-at-a-time ±% sensitivity of NPV, ranked (tornado chart).
  - monte_carlo_cashflow— Monte-Carlo risk on NPV (P(loss), VaR percentile, histogram).
  - lcoe                — Levelized Cost Of Energy / output (Σ disc. cost / Σ disc. output).

INPUT MODEL (see ``_resolve_cashflow``):
  * a single numeric column of per-period NET cash flows (config ``cashflow``), OR
  * separate ``cost`` & ``revenue`` columns -> net = revenue - cost, OR
  * sensible defaults (first suitable numeric column / first two numeric columns).
  Period 0 is the FIRST row (an up-front capital outlay is typically a negative CF_0).
  An optional ``period`` column just orders the rows; values are still period indices
  0,1,2,… for discounting (we DISCLOSE this — we do NOT use raw period values as t).

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "跳过:…" appended to summary + return (never crash/fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    float ``estimates`` dict, Chinese ``summary`` with ⚠ assumption/bias disclosures.
  * The profiler may classify an integer cash-flow column as ``count`` or even ``id``
    (all-distinct integers) — so numeric-column resolution accepts continuous/count/id.

Pure Python (numpy / pandas; scipy.optimize.brentq used opportunistically for IRR with
a hand-rolled bisection fallback). NO heavy deps — finance math is hand-rolled; if
``numpy_financial`` is present we do NOT require it.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _numeric_cols(ctx: Ctx):
    """Names of columns usable as a cash-flow / cost / revenue / output series.

    Accepts continuous, count AND id kinds: an integer cash-flow column with all-
    distinct values is misclassified as ``id`` by the profiler (CLAUDE.md「id 陷阱」),
    and a small integer stream profiles as ``count`` — both are legitimate numbers.
    The time/period column (fp.time_col) is excluded from auto-pick.
    """
    fp = ctx.fp
    excl = {fp.unit_col, fp.time_col}
    out = []
    for c in fp.columns:
        if c.name in excl:
            continue
        if c.kind in ("continuous", "count", "id"):
            out.append(c.name)
    return out


def _order_df(ctx):
    """Row order for the cash-flow series: by config ``period`` col else fp.time_col
    else natural row order. Returns (ordered_df, period_note)."""
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    pcol = cfg.get("period")
    if pcol not in df.columns:
        pcol = fp.time_col if (fp.time_col and fp.time_col in df.columns) else None
    if pcol is not None:
        try:
            return df.sort_values(pcol), (
                f"（按 {pcol} 列排序；折现期 t 仍取行序号 0,1,2,…，"
                "未把 {pcol} 的原始数值当作 t）".replace("{pcol}", str(pcol))
            )
        except Exception:
            return df, ""
    return df, ""


def _resolve_cashflow(ctx: Ctx, min_n: int = 2):
    """Resolve the per-period NET cash-flow series.

    Resolution order:
      1. config ``cashflow`` -> that column (net cash flows directly).
      2. config ``cost`` & ``revenue`` (or ``benefit``) -> net = revenue - cost.
      3. auto: a single numeric column (the first) as the net series.

    Returns (cf_array, label, note, None) on success or (None, None, None, msg) on
    honest failure. ``cf_array`` is a 1-D float numpy array, period 0 = first row.
    """
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    d2, pnote = _order_df(ctx)
    nums = _numeric_cols(ctx)

    cf_name = cfg.get("cashflow")
    cost_name = cfg.get("cost")
    rev_name = cfg.get("revenue") or cfg.get("benefit")

    # coerce (never raise) — a non-numeric column becomes NaN and is caught by the
    # finite/min-period guard below, degrading honestly instead of crashing.
    def _num(col):
        return pd.to_numeric(d2[col], errors="coerce").to_numpy(dtype=float)

    label = ""
    note = pnote
    if cf_name in df.columns:
        cf = _num(cf_name)
        label = f"净现金流列 {cf_name}"
    elif cost_name in df.columns and rev_name in df.columns:
        cost = _num(cost_name)
        rev = _num(rev_name)
        cf = rev - cost
        label = f"净现金流 = {rev_name} − {cost_name}"
    elif len(nums) >= 1:
        # auto: first numeric column is the net cash-flow stream.
        auto = nums[0]
        cf = _num(auto)
        label = f"净现金流列 {auto}（自动选第一个数值列）"
        note = (note + " " if note else "") + (
            f"（⚠ 未指定 config['cashflow']，已自动把数值列 {auto} 当作每期净现金流；"
            "config['cashflow'] 或 config['cost']+config['revenue'] 可显式指定）"
        )
    else:
        return None, None, None, "需要一个数值现金流序列：config['cashflow'] 或 config['cost']+config['revenue']。"

    cf = cf[np.isfinite(cf)]
    if cf.size < min_n:
        return None, None, None, f"现金流有效期数过少（{cf.size}<{min_n}）。"
    return cf.astype(float), label, note, None


def _npv(rate: float, cf) -> float:
    """NPV = Σ_t CF_t / (1+rate)^t, t = 0,1,2,… (period 0 is the first element)."""
    import numpy as np

    cf = np.asarray(cf, dtype=float)
    t = np.arange(cf.size)
    return float(np.sum(cf / (1.0 + rate) ** t))


def _irr(cf, lo: float = -0.9999, hi: float = 10.0):
    """Internal Rate of Return: the rate r solving NPV(r)=0.

    Strategy: scan a grid of rates over [lo, hi], find a SIGN CHANGE of NPV(r), then
    refine with scipy.optimize.brentq if available else a hand-rolled bisection.
    Returns (irr, status) where status in {"ok", "no_sign_change", "multiple_roots",
    "no_invest"}. We DISCLOSE multiple sign changes (non-conventional cash flows can
    have several IRRs — the classic multiple-IRR pathology).
    """
    import numpy as np

    cf = np.asarray(cf, dtype=float)
    # IRR is only meaningful when there is at least one outflow AND one inflow.
    if not (np.any(cf < 0) and np.any(cf > 0)):
        return float("nan"), "no_invest"

    grid = np.linspace(lo, hi, 600)
    vals = np.array([_npv(r, cf) for r in grid])
    finite = np.isfinite(vals)
    grid, vals = grid[finite], vals[finite]
    # bracket indices where NPV changes sign between consecutive grid points
    sign = np.sign(vals)
    brackets = []
    for i in range(len(vals) - 1):
        if sign[i] == 0.0:
            brackets.append((grid[i], grid[i]))
        elif sign[i] * sign[i + 1] < 0:
            brackets.append((grid[i], grid[i + 1]))
    if not brackets:
        return float("nan"), "no_sign_change"

    def _root(a, b):
        if a == b:
            return a
        try:
            from scipy.optimize import brentq

            return float(brentq(lambda r: _npv(r, cf), a, b, maxiter=200))
        except Exception:
            # hand-rolled bisection (no scipy / brentq failure)
            fa = _npv(a, cf)
            for _ in range(200):
                m = 0.5 * (a + b)
                fm = _npv(m, cf)
                if abs(fm) < 1e-10 or (b - a) < 1e-12:
                    return m
                if fa * fm < 0:
                    b = m
                else:
                    a, fa = m, fm
            return 0.5 * (a + b)

    roots = sorted({round(_root(a, b), 10) for (a, b) in brackets})
    if len(roots) > 1:
        # multiple IRRs — report the smallest economically-sensible one but flag it.
        return float(roots[0]), "multiple_roots"
    return float(roots[0]), "ok"


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


def _rate(ctx, default: float = 0.1) -> float:
    """Discount rate from config ``rate`` (per period), default 0.1, guarded > -1."""
    try:
        r = float(ctx.cfg.get("rate", default))
    except (TypeError, ValueError):
        r = default
    return r if r > -1.0 else default


# ===========================================================================
# 1) npv_irr — Net Present Value + Internal Rate of Return + payback
#    Refs: Brealey-Myers-Allen "Principles of Corporate Finance"; Park
#          "Contemporary Engineering Economics".
# ===========================================================================
@register("npv_irr")
def _branch_npv_irr(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cf, label, note, err = _resolve_cashflow(ctx, min_n=2)
    if err is not None:
        summary.append(f"NPV/IRR 跳过：{err}")
        return
    try:
        rate = _rate(ctx)
        n = cf.size
        t = np.arange(n)
        disc = 1.0 / (1.0 + rate) ** t
        pv = cf * disc
        npv = float(pv.sum())

        irr, irr_status = _irr(cf)

        # cumulative (undiscounted) and discounted cash-flow for payback periods
        cum = np.cumsum(cf)
        cum_disc = np.cumsum(pv)

        def _payback(cumulative):
            """First period where cumulative >= 0, linearly interpolated; None if never."""
            idx = np.where(cumulative >= 0)[0]
            if idx.size == 0:
                return None
            k = int(idx[0])
            if k == 0:
                return 0.0
            prev = cumulative[k - 1]
            here = cumulative[k]
            if here == prev:
                return float(k)
            return float(k - 1 + (0.0 - prev) / (here - prev))

        simple_pb = _payback(cum)
        disc_pb = _payback(cum_disc)

        decision = "接受 (NPV>0)" if npv > 0 else ("临界 (NPV=0)" if npv == 0 else "拒绝 (NPV<0)")

        tbl = pd.DataFrame({
            "period": t,
            "cash_flow": np.round(cf, 6),
            "discount_factor": np.round(disc, 6),
            "pv": np.round(pv, 6),
            "cum_cash_flow": np.round(cum, 6),
            "cum_discounted": np.round(cum_disc, 6),
        })
        tbl.to_csv(d / "npv_schedule.csv", index=False, encoding="utf-8")
        files.append("npv_schedule.csv")

        estimates.update({
            "npv": round(npv, 6),
            "irr": round(irr, 6) if irr == irr else float("nan"),
            "discount_rate": round(rate, 6),
            "simple_payback": round(simple_pb, 4) if simple_pb is not None else float("nan"),
            "discounted_payback": round(disc_pb, 4) if disc_pb is not None else float("nan"),
            "n_periods": float(n),
            "total_undiscounted": round(float(cf.sum()), 6),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.bar(t, cf, color="#bbbbbb", edgecolor="white", label="cash flow")
            ax.plot(t, cum_disc, color="#4C72B0", lw=1.6, marker="o", ms=3,
                    label="cumulative discounted CF")
            ax.axhline(0.0, color="#333333", lw=0.8)
            if disc_pb is not None:
                ax.axvline(disc_pb, color="#C44E52", ls="--", lw=1.2,
                           label=f"disc. payback = {disc_pb:.2f}")
            ax.set_xlabel("period t")
            ax.set_ylabel("cash flow / cumulative PV")
            ax.set_title(f"NPV schedule (r={rate:.1%}, NPV={npv:.4g})")
            ax.legend(fontsize=8)

        _save_fig(d, "npv_schedule.png", files, _plot)

        irr_txt = {
            "ok": f"IRR={irr:.4%}",
            "multiple_roots": f"IRR≈{irr:.4%}（⚠ 检测到多个符号变化→可能存在多个 IRR，非常规现金流，请以 NPV 为决策依据）",
            "no_sign_change": "IRR=无实根（NPV(r) 在搜索区间内不变号——现金流可能全正或全负）",
            "no_invest": "IRR=不适用（现金流缺少「先投入后回收」的正负结构）",
        }[irr_status]
        pb_txt = (f"{simple_pb:.2f} 期" if simple_pb is not None else "永不回收")
        dpb_txt = (f"{disc_pb:.2f} 期" if disc_pb is not None else "永不回收")
        summary.append(
            f"{ctx.entry.method} 完成：{label}（{n} 期，折现率 r={rate:.2%}）{(' ' + note) if note else ''}；"
            f"NPV={npv:.6g} → 决策：{decision}；{irr_txt}；"
            f"简单回收期={pb_txt}、折现回收期={dpb_txt}。明细见 npv_schedule.csv 与图。"
            " ⚠ 期 0 为第一行（资本性支出通常为负的 CF_0）；折现期 t=0,1,2,… 取行序号，"
            "未把 period 列原始数值当 t。 ⚠ NPV 对折现率 r 高度敏感（见敏感性/蒙特卡洛分析）；"
            "IRR 对非常规现金流可能多解或无解，且隐含「中间现金流按 IRR 再投资」假定（高估），"
            "决策应以 NPV 为准。回收期忽略回收期后的现金流、且简单回收期不计时间价值。"
        )
        code += [
            "import numpy as np",
            "t = np.arange(cf.size)",
            "npv = np.sum(cf / (1+r)**t)                 # 净现值",
            "# IRR: 解 NPV(r)=0，网格找符号变化 + brentq 求根",
            "from scipy.optimize import brentq",
            "irr = brentq(lambda x: np.sum(cf/(1+x)**t), a, b)",
            "cum_disc = np.cumsum(cf/(1+r)**t)           # 折现累计→折现回收期",
        ]
    except Exception as exc:
        summary.append(f"NPV/IRR 计算失败：{exc}")


# ===========================================================================
# 2) cost_benefit — discounted Benefit-Cost Ratio (BCR) + net present benefit
#    Refs: Boardman et al. "Cost-Benefit Analysis: Concepts and Practice".
# ===========================================================================
@register("cost_benefit")
def _branch_cost_benefit(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    d2, pnote = _order_df(ctx)
    nums = _numeric_cols(ctx)

    cost_name = cfg.get("cost")
    ben_name = cfg.get("benefit") or cfg.get("revenue")
    # auto: need a cost stream AND a benefit stream — take the first two numeric cols.
    auto_note = ""
    if cost_name not in df.columns or ben_name not in df.columns:
        if len(nums) >= 2:
            cost_name, ben_name = nums[0], nums[1]
            auto_note = (
                f"（⚠ 未指定 config['cost']/['benefit']，自动取 {cost_name}=成本、{ben_name}=效益；"
                "可用 config['cost']/config['benefit'] 显式指定）"
            )
        else:
            summary.append(
                "成本-效益分析 跳过：需要成本与效益两条数值流——"
                "config['cost'] + config['benefit']（或两个数值列）。"
            )
            return
    try:
        rate = _rate(ctx)
        cost = d2[cost_name].astype(float).to_numpy()
        ben = d2[ben_name].astype(float).to_numpy()
        m = np.isfinite(cost) & np.isfinite(ben)
        cost, ben = cost[m], ben[m]
        if cost.size < 1:
            summary.append("成本-效益分析 跳过：成本/效益序列无有效观测。")
            return
        n = cost.size
        t = np.arange(n)
        disc = 1.0 / (1.0 + rate) ** t
        pv_cost = float(np.sum(cost * disc))
        pv_ben = float(np.sum(ben * disc))

        if pv_cost <= 0:
            summary.append(
                f"成本-效益分析 跳过：折现成本现值 PV(cost)={pv_cost:.6g} 非正，BCR 无定义"
                "（请确认成本列为正向支出）。"
            )
            return

        bcr = pv_ben / pv_cost
        npb = pv_ben - pv_cost  # net present benefit
        decision = "接受 (BCR>1)" if bcr > 1 else ("临界 (BCR=1)" if bcr == 1 else "拒绝 (BCR<1)")

        pd.DataFrame({
            "period": t,
            "cost": np.round(cost, 6),
            "benefit": np.round(ben, 6),
            "discount_factor": np.round(disc, 6),
            "pv_cost": np.round(cost * disc, 6),
            "pv_benefit": np.round(ben * disc, 6),
        }).to_csv(d / "cost_benefit.csv", index=False, encoding="utf-8")
        files.append("cost_benefit.csv")

        estimates.update({
            "bcr": round(bcr, 6),
            "pv_benefit": round(pv_ben, 6),
            "pv_cost": round(pv_cost, 6),
            "net_present_benefit": round(npb, 6),
            "discount_rate": round(rate, 6),
            "n_periods": float(n),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.bar(t - 0.18, cost * disc, width=0.36, color="#C44E52", label="PV cost")
            ax.bar(t + 0.18, ben * disc, width=0.36, color="#55A868", label="PV benefit")
            ax.set_xlabel("period t")
            ax.set_ylabel("present value")
            ax.set_title(f"Discounted cost vs benefit (BCR={bcr:.3f}, r={rate:.1%})")
            ax.legend(fontsize=8)

        _save_fig(d, "cost_benefit.png", files, _plot)

        summary.append(
            f"{ctx.entry.method} 完成：成本={cost_name}、效益={ben_name}（{n} 期，r={rate:.2%}）"
            f"{(' ' + pnote) if pnote else ''}{auto_note}；"
            f"PV(效益)={pv_ben:.6g}、PV(成本)={pv_cost:.6g}；"
            f"贴现效益成本比 BCR={bcr:.4f} → 决策：{decision}；净现值效益 NPB={npb:.6g}。"
            "（明细见 cost_benefit.csv 与图）"
            " ⚠ BCR=PV(效益)/PV(成本)，>1 接受；对效益/成本的口径与折现率敏感。"
            " ⚠ BCR 受分子分母如何归类（净化某项 vs 计入成本/负效益）影响、不宜跨方案直接比较规模；"
            "净现值效益(NPB=PV效益−PV成本)与 NPV 同序，规模决策更稳健。未含影子价格/外部性校正。"
        )
        code += [
            "import numpy as np",
            "t = np.arange(cost.size); disc = 1/(1+r)**t",
            "pv_cost = np.sum(cost*disc); pv_ben = np.sum(ben*disc)",
            "bcr = pv_ben / pv_cost          # 贴现效益成本比, >1 接受",
            "npb = pv_ben - pv_cost          # 净现值效益",
        ]
    except Exception as exc:
        summary.append(f"成本-效益分析 计算失败：{exc}")


# ===========================================================================
# 3) breakeven_analysis — break-even quantity/price + contribution margin
#    Refs: Horngren "Cost Accounting"; CVP (cost-volume-profit) analysis.
# ===========================================================================
@register("breakeven_analysis")
def _branch_breakeven_analysis(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cfg = ctx.cfg

    def _scalar(key):
        v = cfg.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fixed = _scalar("fixed_cost")
    var = _scalar("var_cost")
    price = _scalar("price")
    # optional current/expected sales volume for margin-of-safety
    current_units = _scalar("units")

    # If not given as scalars, try to INFER from same-named numeric columns (means).
    df = ctx.df
    note_parts = []
    for name, val in (("fixed_cost", fixed), ("var_cost", var), ("price", price)):
        if val is None and name in df.columns:
            try:
                inferred = float(df[name].astype(float).dropna().mean())
                note_parts.append(f"{name}={inferred:.4g}(取列均值)")
                if name == "fixed_cost":
                    fixed = inferred
                elif name == "var_cost":
                    var = inferred
                else:
                    price = inferred
            except Exception:
                pass

    missing = [k for k, v in (("fixed_cost", fixed), ("var_cost", var), ("price", price)) if v is None]
    if missing:
        summary.append(
            "盈亏平衡分析 跳过：需要固定成本/单位变动成本/单价三个标量——"
            f"缺 {missing}。用 config['fixed_cost']/['var_cost']/['price'] 提供"
            "（或在数据中提供同名列）。"
        )
        return
    try:
        cm = price - var  # contribution margin per unit
        if cm <= 0:
            summary.append(
                f"盈亏平衡分析 跳过：单位贡献毛益 = 单价({price:.4g}) − 单位变动成本({var:.4g}) = "
                f"{cm:.4g} ≤ 0，无盈亏平衡点（每多卖一件亏损更大）。"
            )
            return
        cm_ratio = cm / price if price != 0 else float("nan")
        be_units = fixed / cm  # break-even quantity (units)
        be_revenue = be_units * price  # break-even sales revenue
        # break-even PRICE at the current volume (if a volume is supplied): price s.t.
        # profit=0 -> price = var + fixed/units.
        be_price = (var + fixed / current_units) if (current_units and current_units > 0) else float("nan")

        # margin of safety vs current/expected units (if supplied)
        if current_units and current_units > 0:
            mos_units = current_units - be_units
            mos_ratio = mos_units / current_units if current_units != 0 else float("nan")
            profit_at_current = cm * current_units - fixed
        else:
            mos_units = float("nan")
            mos_ratio = float("nan")
            profit_at_current = float("nan")

        estimates.update({
            "breakeven_units": round(be_units, 6),
            "breakeven_revenue": round(be_revenue, 6),
            "contribution_margin": round(cm, 6),
            "contribution_margin_ratio": round(cm_ratio, 6) if cm_ratio == cm_ratio else float("nan"),
            "breakeven_price": round(be_price, 6) if be_price == be_price else float("nan"),
            "margin_of_safety_units": round(mos_units, 6) if mos_units == mos_units else float("nan"),
            "margin_of_safety_ratio": round(mos_ratio, 6) if mos_ratio == mos_ratio else float("nan"),
            "profit_at_current_units": round(profit_at_current, 6) if profit_at_current == profit_at_current else float("nan"),
            "fixed_cost": round(fixed, 6), "var_cost": round(var, 6), "price": round(price, 6),
        })

        # CVP table across a unit grid bracketing the break-even point
        hi = be_units * 2 if be_units > 0 else 10.0
        if current_units and current_units > be_units:
            hi = max(hi, current_units * 1.1)
        grid = np.linspace(0, hi, 25)
        pd.DataFrame({
            "units": np.round(grid, 4),
            "total_cost": np.round(fixed + var * grid, 4),
            "total_revenue": np.round(price * grid, 4),
            "profit": np.round(cm * grid - fixed, 4),
        }).to_csv(d / "breakeven_cvp.csv", index=False, encoding="utf-8")
        files.append("breakeven_cvp.csv")

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.plot(grid, fixed + var * grid, color="#C44E52", lw=1.6, label="total cost")
            ax.plot(grid, price * grid, color="#55A868", lw=1.6, label="total revenue")
            ax.axvline(be_units, color="#4C72B0", ls="--", lw=1.2,
                       label=f"break-even = {be_units:.1f} units")
            ax.axhline(be_revenue, color="#999999", ls=":", lw=0.9)
            ax.set_xlabel("units sold")
            ax.set_ylabel("cost / revenue")
            ax.set_title("Cost-Volume-Profit (break-even) analysis")
            ax.legend(fontsize=8)

        _save_fig(d, "breakeven.png", files, _plot)

        infer_note = ("（" + "、".join(note_parts) + "）") if note_parts else ""
        extra = ""
        if current_units and current_units > 0:
            extra = (f" 在当前销量 {current_units:g} 单位：安全边际={mos_units:.4g} 单位"
                     f"（{mos_ratio:.1%}）、利润={profit_at_current:.4g}；保本单价={be_price:.4g}。")
        summary.append(
            f"{ctx.entry.method} 完成{infer_note}：固定成本={fixed:.4g}、单位变动成本={var:.4g}、单价={price:.4g}；"
            f"单位贡献毛益 CM={cm:.4g}（贡献毛益率={cm_ratio:.1%}）；"
            f"盈亏平衡销量={be_units:.4g} 单位、保本销售额={be_revenue:.4g}。{extra}"
            "（CVP 表见 breakeven_cvp.csv 与图）"
            " ⚠ 线性 CVP 模型假定：单价与单位变动成本恒定（无规模/折扣效应）、单一产品（或固定产品组合）、"
            "成本可清晰二分为固定/变动、产销一致。现实中这些常被违反；安全边际衡量销量下滑到保本点的缓冲。"
        )
        code += [
            "cm = price - var_cost                 # 单位贡献毛益",
            "breakeven_units = fixed_cost / cm     # 盈亏平衡销量",
            "breakeven_revenue = breakeven_units * price",
            "margin_of_safety = current_units - breakeven_units",
        ]
    except Exception as exc:
        summary.append(f"盈亏平衡分析 计算失败：{exc}")


# ===========================================================================
# 4) sensitivity_tornado — one-at-a-time ±% sensitivity of NPV (tornado chart)
#    Deterministic. Refs: standard project-appraisal sensitivity analysis.
# ===========================================================================
@register("sensitivity_tornado")
def _branch_sensitivity_tornado(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cf, label, note, err = _resolve_cashflow(ctx, min_n=2)
    if err is not None:
        summary.append(f"敏感性(龙卷风图) 跳过：{err}")
        return
    try:
        rate = _rate(ctx)
        try:
            pct = float(ctx.cfg.get("sensitivity_pct", 20.0))
        except (TypeError, ValueError):
            pct = 20.0
        pct = abs(pct)
        frac = pct / 100.0
        base_npv = _npv(rate, cf)
        n = cf.size

        # Parameters perturbed one-at-a-time: the discount rate, and each non-trivial
        # cash-flow period (period 0 = initial investment; later periods = operating CF).
        rows = []

        def _swing(name, low_npv, high_npv):
            rows.append({
                "parameter": name,
                "low_npv": round(float(low_npv), 6),
                "high_npv": round(float(high_npv), 6),
                "swing": round(abs(float(high_npv) - float(low_npv)), 6),
            })

        # discount rate ±pct (relative)
        _swing("discount_rate",
               _npv(rate * (1 - frac), cf), _npv(rate * (1 + frac), cf))

        # each cash-flow period ±pct (skip exact zeros — perturbing 0 does nothing)
        for i in range(n):
            if cf[i] == 0:
                continue
            cf_lo = cf.copy(); cf_lo[i] = cf[i] * (1 - frac)
            cf_hi = cf.copy(); cf_hi[i] = cf[i] * (1 + frac)
            _swing(f"cashflow_t{i}", _npv(rate, cf_lo), _npv(rate, cf_hi))

        tbl = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
        tbl.to_csv(d / "sensitivity.csv", index=False, encoding="utf-8")
        files.append("sensitivity.csv")

        estimates.update({
            "base_npv": round(base_npv, 6),
            "sensitivity_pct": round(pct, 4),
            "discount_rate": round(rate, 6),
            "n_parameters": float(len(rows)),
        })
        if not tbl.empty:
            estimates["top_driver_swing"] = float(tbl.iloc[0]["swing"])

        # tornado chart: horizontal bars from low_npv..high_npv, sorted by swing desc
        def _plot(plt):
            top = tbl.head(12).iloc[::-1]  # largest swing on top
            ys = np.arange(len(top))
            lows = top["low_npv"].to_numpy()
            highs = top["high_npv"].to_numpy()
            fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(top) + 1.2)))
            for y, lo, hi in zip(ys, lows, highs):
                left, right = (lo, hi) if lo <= hi else (hi, lo)
                ax.barh(y, right - left, left=left, color="#4C72B0",
                        edgecolor="white", height=0.6)
            ax.axvline(base_npv, color="#C44E52", ls="--", lw=1.3,
                       label=f"base NPV = {base_npv:.4g}")
            ax.set_yticks(ys)
            ax.set_yticklabels(top["parameter"].tolist(), fontsize=8)
            ax.set_xlabel("NPV")
            ax.set_title(f"Tornado: NPV sensitivity to +/-{pct:g}% (one-at-a-time)")
            ax.legend(fontsize=8)

        _save_fig(d, "tornado.png", files, _plot)

        drivers = "、".join(
            f"{r['parameter']}(摆幅 {r['swing']:.4g})" for _, r in tbl.head(3).iterrows()
        )
        summary.append(
            f"{ctx.entry.method} 完成：{label}（{n} 期，r={rate:.2%}，单因素 ±{pct:g}%）"
            f"{(' ' + note) if note else ''}；基准 NPV={base_npv:.6g}。"
            f"按 NPV 摆幅排序，前几大驱动：{drivers}。完整排序见 sensitivity.csv、龙卷风图见 tornado.png。"
            " ⚠ 这是「单因素一次一变(OAT)」局部敏感性：每次只动一个参数、其余不变，"
            "因此忽略参数间的交互与相关（联合不确定性见 monte_carlo_cashflow 蒙特卡洛分析）；"
            "摆幅大小取决于所设 ±% 幅度，不同参数用同一相对幅度未必同等合理。"
        )
        code += [
            "base = npv(r, cf)",
            "# 每个参数 ±pct%，记录 NPV 摆幅(swing)，按摆幅排序画龙卷风图",
            "swing = abs(npv_high - npv_low)",
        ]
    except Exception as exc:
        summary.append(f"敏感性(龙卷风图) 计算失败：{exc}")


# ===========================================================================
# 5) monte_carlo_cashflow — Monte-Carlo risk on NPV (P(loss), VaR, histogram)
#    INFERENCE-BEARING (distributional simulation). Refs: Savvides (1994)
#    "Risk analysis in investment appraisal"; standard project risk simulation.
# ===========================================================================
@register("monte_carlo_cashflow")
def _branch_monte_carlo_cashflow(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cf, label, note, err = _resolve_cashflow(ctx, min_n=2)
    if err is not None:
        summary.append(f"蒙特卡洛现金流 跳过：{err}")
        return
    try:
        cfg = ctx.cfg
        rate = _rate(ctx)
        n = cf.size
        t = np.arange(n)

        try:
            n_sim = int(cfg.get("n_sim", 10000))
        except (TypeError, ValueError):
            n_sim = 10000
        n_sim = max(100, min(n_sim, 500000))  # guardrail
        try:
            seed = int(cfg.get("seed", 12345))
        except (TypeError, ValueError):
            seed = 12345
        # coefficient of variation of each per-period cash flow (sd = cv * |CF_t|)
        try:
            cv = float(cfg.get("cv", 0.15))
        except (TypeError, ValueError):
            cv = 0.15
        cv = abs(cv)
        dist = str(cfg.get("dist", "normal")).lower()
        if dist not in ("normal", "triangular"):
            dist = "normal"
        try:
            var_pct = float(cfg.get("var_pct", 5.0))
        except (TypeError, ValueError):
            var_pct = 5.0
        var_pct = min(max(var_pct, 0.01), 50.0)

        rng = np.random.default_rng(seed)
        disc = 1.0 / (1.0 + rate) ** t
        base_npv = float(np.sum(cf * disc))

        # DISTRIBUTIONAL MODEL: each per-period cash flow CF_t is treated as an
        # independent random variable centred at the point estimate with scale
        # sd_t = cv * |CF_t|. normal: CF_t ~ N(CF_t, sd_t). triangular: symmetric
        # triangular on [CF_t - w, CF_t + w] with half-width w = sqrt(6)*sd_t (so its
        # sd matches cv*|CF_t|, since a symmetric triangular of half-width w has
        # sd = w/sqrt(6)). Periods with CF_t==0 stay fixed (sd 0). Draws are
        # independent ACROSS periods (no autocorrelation/common shocks modelled).
        sd = cv * np.abs(cf)
        sims = np.empty((n_sim, n), dtype=float)
        for i in range(n):
            if sd[i] <= 0:
                sims[:, i] = cf[i]
            elif dist == "normal":
                sims[:, i] = rng.normal(cf[i], sd[i], size=n_sim)
            else:  # symmetric triangular with matching sd
                w = np.sqrt(6.0) * sd[i]
                sims[:, i] = rng.triangular(cf[i] - w, cf[i], cf[i] + w, size=n_sim)

        npvs = sims @ disc  # (n_sim,) vector of simulated NPVs
        mean_npv = float(npvs.mean())
        sd_npv = float(npvs.std(ddof=1))
        p_loss = float(np.mean(npvs < 0.0))  # probability of a loss (NPV<0)
        # NPV value-at-risk: the lower var_pct percentile of the NPV distribution
        # (a low/negative number = the loss not exceeded with (1-var_pct%) confidence).
        npv_var = float(np.percentile(npvs, var_pct))
        p05 = float(np.percentile(npvs, 5))
        p50 = float(np.percentile(npvs, 50))
        p95 = float(np.percentile(npvs, 95))

        pd.DataFrame({
            "statistic": ["base_npv", "mean", "sd", "p_loss", f"p{var_pct:g}_VaR",
                          "p5", "p50", "p95", "n_sim", "cv", "dist"],
            "value": [round(base_npv, 6), round(mean_npv, 6), round(sd_npv, 6),
                      round(p_loss, 6), round(npv_var, 6), round(p05, 6),
                      round(p50, 6), round(p95, 6), float(n_sim), round(cv, 6), dist],
        }).to_csv(d / "monte_carlo_npv.csv", index=False, encoding="utf-8")
        files.append("monte_carlo_npv.csv")

        estimates.update({
            "base_npv": round(base_npv, 6),
            "mc_mean_npv": round(mean_npv, 6),
            "mc_sd_npv": round(sd_npv, 6),
            "prob_loss": round(p_loss, 6),
            "npv_var": round(npv_var, 6),
            "npv_p5": round(p05, 6),
            "npv_p50": round(p50, 6),
            "npv_p95": round(p95, 6),
            "n_sim": float(n_sim),
            "cv": round(cv, 6),
            "discount_rate": round(rate, 6),
            "seed": float(seed),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.hist(npvs, bins=60, color="#bbbbbb", edgecolor="white", alpha=0.9)
            ax.axvline(0.0, color="#333333", lw=1.0, label="NPV = 0")
            ax.axvline(mean_npv, color="#4C72B0", ls="--", lw=1.4,
                       label=f"mean = {mean_npv:.4g}")
            ax.axvline(npv_var, color="#C44E52", ls="--", lw=1.4,
                       label=f"P{var_pct:g} VaR = {npv_var:.4g}")
            ax.set_xlabel("simulated NPV")
            ax.set_ylabel("frequency")
            ax.set_title(f"Monte-Carlo NPV ({n_sim} sims, P(loss)={p_loss:.1%})")
            ax.legend(fontsize=8)

        _save_fig(d, "monte_carlo_npv.png", files, _plot)

        risk_txt = ("低" if p_loss < 0.1 else ("中" if p_loss < 0.4 else "高"))
        summary.append(
            f"{ctx.entry.method} 完成：{label}（{n} 期，r={rate:.2%}，"
            f"{n_sim} 次模拟，seed={seed}，分布={dist}，cv={cv:.2f}）"
            f"{(' ' + note) if note else ''}；确定性基准 NPV={base_npv:.6g}。"
            f"模拟 NPV 均值={mean_npv:.6g}、标准差={sd_npv:.6g}；亏损概率 P(NPV<0)={p_loss:.1%}（风险{risk_txt}）；"
            f"P{var_pct:g} 风险价值 VaR={npv_var:.6g}（即有 {100 - var_pct:g}% 把握 NPV 不低于此值）；"
            f"P5/P50/P95={p05:.4g}/{p50:.4g}/{p95:.4g}。分布见 monte_carlo_npv.png、汇总见 monte_carlo_npv.csv。"
            f" ⚠ 分布假定（关键且主观）：每期现金流 CF_t 独立服从以点估计为中心、标准差 σ_t=cv·|CF_t| 的"
            f"{'正态' if dist == 'normal' else '对称三角'}分布（cv 由 config 给定，默认 0.15）；"
            "各期相互独立（未建模自相关或共同冲击，会低估尾部联合风险）。"
            " ⚠ 正态分布无界、可能抽出负的本应为正的收入项；cv 假定全期相同。"
            " ⚠ **初始投资 CF_0 也按同一 cv 视为不确定**（且 t=0 不折现、方差权重最大）——"
            "若资本支出已锁定，应将其设为确定值，否则当前设定会相对放大 NPV 标准差与亏损概率。"
            "结果对 cv/分布/折现率高度敏感，应做情景对比，勿当点预测。模拟有抽样误差（固定 seed 可复现）。"
        )
        code += [
            "import numpy as np; rng = np.random.default_rng(seed)",
            "disc = 1/(1+r)**np.arange(cf.size); sd = cv*np.abs(cf)",
            "sims = rng.normal(cf, sd, size=(n_sim, cf.size))  # 每期独立",
            "npvs = sims @ disc                                # 模拟 NPV 分布",
            "p_loss = np.mean(npvs < 0)                        # 亏损概率",
            "npv_var = np.percentile(npvs, var_pct)            # NPV 风险价值",
        ]
    except Exception as exc:
        summary.append(f"蒙特卡洛现金流 计算失败：{exc}")


# ===========================================================================
# 6) lcoe — Levelized Cost Of Energy / output
#    Refs: NREL Simple LCOE; IEA/IRENA levelized-cost methodology.
#    LCOE = Σ_t cost_t/(1+r)^t  /  Σ_t output_t/(1+r)^t
# ===========================================================================
@register("lcoe")
def _branch_lcoe(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    d2, pnote = _order_df(ctx)
    nums = _numeric_cols(ctx)

    cost_name = cfg.get("cost")
    out_name = cfg.get("output") or cfg.get("energy") or cfg.get("production")
    auto_note = ""
    if cost_name not in df.columns or out_name not in df.columns:
        if len(nums) >= 2:
            cost_name, out_name = nums[0], nums[1]
            auto_note = (
                f"（⚠ 未指定 config['cost']/['output']，自动取 {cost_name}=成本、{out_name}=产出；"
                "可用 config['cost']/config['output'] 显式指定）"
            )
        else:
            summary.append(
                "平准化成本(LCOE) 跳过：需要每期成本与产出两条数值流——"
                "config['cost'] + config['output']（或两个数值列）。"
            )
            return
    try:
        rate = _rate(ctx)
        cost = d2[cost_name].astype(float).to_numpy()
        out = d2[out_name].astype(float).to_numpy()
        m = np.isfinite(cost) & np.isfinite(out)
        cost, out = cost[m], out[m]
        if cost.size < 1:
            summary.append("平准化成本(LCOE) 跳过：成本/产出序列无有效观测。")
            return
        n = cost.size
        t = np.arange(n)
        disc = 1.0 / (1.0 + rate) ** t
        pv_cost = float(np.sum(cost * disc))
        pv_out = float(np.sum(out * disc))

        if pv_out <= 0:
            summary.append(
                f"平准化成本(LCOE) 跳过：折现产出现值 Σ output_t/(1+r)^t = {pv_out:.6g} 非正，"
                "LCOE 无定义（请确认产出列为正、单位一致）。"
            )
            return

        lcoe = pv_cost / pv_out  # cost per unit of output (levelized)
        undisc_cost = float(cost.sum())
        undisc_out = float(out.sum())
        simple_avg_cost = (undisc_cost / undisc_out) if undisc_out > 0 else float("nan")

        pd.DataFrame({
            "period": t,
            "cost": np.round(cost, 6),
            "output": np.round(out, 6),
            "discount_factor": np.round(disc, 6),
            "pv_cost": np.round(cost * disc, 6),
            "pv_output": np.round(out * disc, 6),
        }).to_csv(d / "lcoe.csv", index=False, encoding="utf-8")
        files.append("lcoe.csv")

        estimates.update({
            "lcoe": round(lcoe, 6),
            "pv_cost": round(pv_cost, 6),
            "pv_output": round(pv_out, 6),
            "simple_avg_cost": round(simple_avg_cost, 6) if simple_avg_cost == simple_avg_cost else float("nan"),
            "discount_rate": round(rate, 6),
            "n_periods": float(n),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.bar(t, cost * disc, color="#C44E52", alpha=0.8, label="PV cost")
            ax2 = ax.twinx()
            ax2.plot(t, out * disc, color="#55A868", lw=1.6, marker="o", ms=3,
                     label="PV output")
            ax.axhline(0.0, color="#333333", lw=0.6)
            ax.set_xlabel("period t")
            ax.set_ylabel("PV cost")
            ax2.set_ylabel("PV output")
            ax.set_title(f"Levelized cost = {lcoe:.4g} per unit (r={rate:.1%})")
            ax.legend(fontsize=8, loc="upper left")
            ax2.legend(fontsize=8, loc="upper right")

        _save_fig(d, "lcoe.png", files, _plot)

        summary.append(
            f"{ctx.entry.method} 完成：成本={cost_name}、产出={out_name}（{n} 期，r={rate:.2%}）"
            f"{(' ' + pnote) if pnote else ''}{auto_note}；"
            f"折现总成本 Σcost_t/(1+r)^t={pv_cost:.6g}、折现总产出 Σoutput_t/(1+r)^t={pv_out:.6g}；"
            f"平准化单位成本 LCOE={lcoe:.6g}（每单位产出的成本，与不折现简单均成本 {simple_avg_cost:.6g} 对比）。"
            "（明细见 lcoe.csv 与图）"
            " ⚠ LCOE=折现总成本/折现总产出，对折现率与项目寿命极敏感；"
            "产出也被折现（能源/产量项的标准做法，使「现在的一单位产出」比「未来的一单位」更值钱）。"
            " ⚠ LCOE 不含收入/电价、不反映产出的时间价值差异（峰谷电价、可调度性），不宜单独用于技术选型，"
            "应结合 NPV/补贴/系统价值综合判断；成本与产出的单位口径须一致。"
        )
        code += [
            "import numpy as np",
            "t = np.arange(cost.size); disc = 1/(1+r)**t",
            "lcoe = np.sum(cost*disc) / np.sum(output*disc)  # 平准化单位成本",
        ]
    except Exception as exc:
        summary.append(f"平准化成本(LCOE) 计算失败：{exc}")
