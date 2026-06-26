"""Branch handlers for the OPERATIONS-RESEARCH family (classic OR / management-
science decision models). High demand for supply-chain, logistics, services and
capacity-planning work.

Three textbook OR models, each driven by a handful of SCALAR parameters that can
be supplied via ``config`` OR read from same-named columns / the first data row
(so the analysis still runs standalone on tiny data):

  - eoq_inventory — Economic Order Quantity (deterministic lot-sizing): Q* =
        sqrt(2·D·S/H), number of orders, cycle time, total annual cost, reorder
        point (+ optional safety stock from a service level & demand variability).
  - queue_mmc    — M/M/c queue steady-state: utilisation ρ, P0, Erlang-C wait
        probability, Lq/Wq/L/W. Honest degrade when ρ = λ/(c·μ) ≥ 1 (unstable).
  - newsvendor   — single-period stochastic inventory: critical ratio
        CR = Cu/(Cu+Co), optimal order quantity Q* = μ_d + z·σ_d with z = Φ⁻¹(CR)
        (normal demand) OR the empirical CR-quantile when a demand column is given.

Conventions (CLAUDE.md「引擎约定」):
  * config keys override; else auto-detect from same-named numeric columns / the
    first data row; defaults still let the method run on its own.
  * Honest degrade -> Chinese "<方法> 跳过：<原因>" appended to summary + return
    (never crash / never fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort
    try/except), float-only ``estimates`` dict (NaN for N/A — never tuples/strings),
    Chinese ``summary`` with ⚠ assumption disclosures. Append reproducible ``code``.

Pure Python (numpy / pandas; scipy.stats.norm used for the inverse-normal quantile
with a hand-rolled rational-approximation fallback so newsvendor works without
scipy). No heavy / R dependencies.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _scalar(cfg: dict, df, key: str):
    """Resolve a positive-or-any scalar parameter for ``key``.

    Resolution order (first that yields a finite number wins):
      1. ``config[key]`` (a scalar the user passed directly), then
      2. a same-named numeric COLUMN — use its first finite value (the parameters
         of an OR model are constants, so the first row is the natural source;
         a single-row "parameters" CSV is the typical layout).

    Returns a float or ``None`` (key absent / non-numeric everywhere). Never raises.
    """
    import pandas as pd

    if cfg.get(key) is not None:
        try:
            return float(cfg.get(key))
        except (TypeError, ValueError):
            pass
    if df is not None and key in getattr(df, "columns", []):
        try:
            col = pd.to_numeric(df[key], errors="coerce").dropna()
            if len(col) > 0:
                return float(col.iloc[0])
        except Exception:
            pass
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


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard-normal CDF Φ⁻¹(p) for p in (0,1).

    Prefers ``scipy.stats.norm.ppf``; falls back to Acklam's rational approximation
    (|error| < 1.15e-9) so newsvendor's z-value works even without scipy. Returns
    -inf / +inf for p<=0 / p>=1 and NaN for non-finite input.
    """
    import math

    if not (p == p) or p <= 0.0:
        return float("-inf") if p <= 0.0 else float("nan")
    if p >= 1.0:
        return float("inf")
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:
        pass
    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# ===========================================================================
# 1) eoq_inventory — Economic Order Quantity (Wilson lot-size model)
#    Refs: Harris (1913); Wilson; Nahmias "Production and Operations Analysis";
#          Silver, Pyke & Peterson "Inventory Management and Production Planning".
#    EOQ Q* = sqrt(2·D·S / H);  TC = (D/Q)·S + (Q/2)·H (+ purchase D·c);
#    ROP = d_L = D·(L/year) (+ safety stock z·σ_L if a service level & σ given).
# ===========================================================================
@register("eoq_inventory")
def _branch_eoq_inventory(ctx: Ctx) -> None:
    df, cfg = ctx.df, ctx.cfg
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    D = _scalar(cfg, df, "demand")
    S = _scalar(cfg, df, "order_cost")
    H = _scalar(cfg, df, "holding_cost")
    price = _scalar(cfg, df, "unit_price")          # optional, for purchase cost
    lead_time = _scalar(cfg, df, "lead_time")       # optional, in YEARS (default 0)
    demand_sd = _scalar(cfg, df, "demand_sd")       # optional, σ of demand over lead time horizon
    service_level = _scalar(cfg, df, "service_level")  # optional, e.g. 0.95 -> safety stock

    missing = [k for k, v in (("demand", D), ("order_cost", S), ("holding_cost", H)) if v is None]
    if missing:
        summary.append(
            "经济订货批量(EOQ) 跳过：需要年需求 D、单次订货/准备成本 S、单位年持有成本 H 三个参数——"
            f"缺 {missing}。用 config['demand']/['order_cost']/['holding_cost'] 提供"
            "（或在数据中提供同名列，取首行）。"
        )
        return
    if not (D > 0 and S > 0 and H > 0):
        summary.append(
            f"经济订货批量(EOQ) 跳过：D({D:.4g})、S({S:.4g})、H({H:.4g}) 必须均为正数。"
        )
        return
    try:
        eoq = float(np.sqrt(2.0 * D * S / H))
        n_orders = D / eoq                              # orders per year
        cycle_time_years = eoq / D                      # time between orders (years)
        cycle_time_days = cycle_time_years * 365.0
        ordering_cost = (D / eoq) * S
        holding_cost = (eoq / 2.0) * H
        purchase_cost = (D * price) if (price is not None and price > 0) else 0.0
        total_cost = ordering_cost + holding_cost + purchase_cost

        # reorder point: demand over the lead time (lead_time in YEARS).
        L = lead_time if (lead_time is not None and lead_time >= 0) else 0.0
        d_L = D * L                                      # expected demand during lead time
        # safety stock z·σ_L if BOTH a service level and a demand σ are supplied.
        safety_stock = 0.0
        z = float("nan")
        if (service_level is not None and 0.0 < service_level < 1.0
                and demand_sd is not None and demand_sd > 0):
            z = _inv_norm_cdf(service_level)
            if np.isfinite(z):
                safety_stock = z * demand_sd
        rop = d_L + safety_stock

        # EOQ total-cost curve around Q* (for the chart)
        qs = np.linspace(max(eoq * 0.2, 1e-9), eoq * 2.5, 60)
        oc = (D / qs) * S
        hc = (qs / 2.0) * H
        tc = oc + hc + purchase_cost
        pd.DataFrame({
            "order_quantity": np.round(qs, 6),
            "ordering_cost": np.round(oc, 6),
            "holding_cost": np.round(hc, 6),
            "total_cost": np.round(tc, 6),
        }).to_csv(d / "eoq_cost_curve.csv", index=False, encoding="utf-8")
        files.append("eoq_cost_curve.csv")

        estimates.update({
            "eoq": round(eoq, 6),
            "n_orders": round(float(n_orders), 6),
            "total_cost": round(float(total_cost), 6),
            "ordering_cost": round(float(ordering_cost), 6),
            "holding_cost": round(float(holding_cost), 6),
            "purchase_cost": round(float(purchase_cost), 6),
            "reorder_point": round(float(rop), 6),
            "safety_stock": round(float(safety_stock), 6),
            "cycle_time_days": round(float(cycle_time_days), 6),
            "z_value": round(float(z), 6) if z == z else float("nan"),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.plot(qs, oc, color="#C44E52", lw=1.5, label="ordering cost D/Q*S")
            ax.plot(qs, hc, color="#55A868", lw=1.5, label="holding cost Q/2*H")
            ax.plot(qs, tc - purchase_cost, color="#4C72B0", lw=2.0,
                    label="total (ordering+holding)")
            ax.axvline(eoq, color="#333333", ls="--", lw=1.2,
                       label=f"EOQ = {eoq:.1f}")
            ax.set_xlabel("order quantity Q")
            ax.set_ylabel("annual cost")
            ax.set_title(f"Economic Order Quantity (EOQ = {eoq:.2f})")
            ax.legend(fontsize=8)

        _save_fig(d, "eoq_cost_curve.png", files, _plot)

        ss_txt = ""
        if safety_stock > 0:
            ss_txt = (f"；服务水平={service_level:.0%}（z={z:.3f}）下安全库存="
                      f"{safety_stock:.4g}（σ_L={demand_sd:.4g}）")
        elif service_level is not None or demand_sd is not None:
            ss_txt = "；未计安全库存（需同时给 service_level∈(0,1) 与 demand_sd>0）"
        rop_txt = (f"再订货点 ROP={rop:.4g}（提前期需求 d_L={d_L:.4g}"
                   f"{('+安全库存' if safety_stock > 0 else '')}）"
                   if L > 0 or safety_stock > 0 else
                   f"再订货点 ROP={rop:.4g}（未给提前期 lead_time，默认提前期需求为 0）")
        purch_txt = (f"，采购成本 D·c={purchase_cost:.6g}" if purchase_cost > 0 else "")
        summary.append(
            f"{ctx.entry.method} 完成：年需求 D={D:.6g}、订货成本 S={S:.6g}、单位年持有成本 H={H:.6g}。"
            f"经济订货批量 EOQ Q*=√(2·D·S/H)={eoq:.4f}；每年订货次数={n_orders:.4f}、"
            f"订货周期={cycle_time_days:.2f} 天；年总成本（订货+持有{('+采购' if purchase_cost > 0 else '')}）="
            f"{total_cost:.6g}（订货={ordering_cost:.6g}、持有={holding_cost:.6g}{purch_txt}）。"
            f"{rop_txt}{ss_txt}。成本曲线见 eoq_cost_curve.csv 与图。"
            " ⚠ EOQ(Wilson) 假定：需求恒定且确定、不允许缺货、订货成本与持有成本固定、瞬时补货、"
            "无数量折扣、单一产品。现实违反这些假定时（随机需求/折扣/缺货成本），应改用"
            "(Q,r)/报童(newsvendor)/带折扣的 EOQ 等模型。"
            " ⚠ 提前期 lead_time 以「年」为单位与 D 同口径；安全库存按正态需求、σ 为提前期内需求标准差。"
        )
        code += [
            "import numpy as np",
            "eoq = np.sqrt(2*D*S/H)                       # 经济订货批量 Q*",
            "n_orders = D/eoq; cycle_days = eoq/D*365",
            "total_cost = (D/eoq)*S + (eoq/2)*H + D*price # 订货+持有+采购",
            "rop = D*lead_time_years + z*demand_sd        # 再订货点(+安全库存)",
        ]
    except Exception as exc:
        summary.append(f"经济订货批量(EOQ) 计算失败：{exc}")


# ===========================================================================
# 2) queue_mmc — M/M/c queue steady-state (Erlang-C)
#    Refs: Erlang (1917); Gross & Harris "Fundamentals of Queueing Theory";
#          Hillier & Lieberman "Introduction to Operations Research".
#    a = λ/μ (offered load, erlangs); ρ = a/c = λ/(c·μ) (must be < 1).
#    P0 = [ Σ_{n=0}^{c-1} a^n/n!  +  a^c/(c!·(1-ρ)) ]^{-1}
#    P_wait (Erlang-C) = [ a^c/(c!·(1-ρ)) ] · P0
#    Lq = P_wait · ρ/(1-ρ);  Wq = Lq/λ;  L = Lq + a;  W = Wq + 1/μ.
# ===========================================================================
@register("queue_mmc")
def _branch_queue_mmc(ctx: Ctx) -> None:
    df, cfg = ctx.df, ctx.cfg
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import math

    import numpy as np
    import pandas as pd

    lam = _scalar(cfg, df, "lambda")
    if lam is None:
        lam = _scalar(cfg, df, "arrival_rate")
    mu = _scalar(cfg, df, "mu")
    if mu is None:
        mu = _scalar(cfg, df, "service_rate")
    c_raw = _scalar(cfg, df, "servers")
    if c_raw is None:
        c_raw = _scalar(cfg, df, "c")

    missing = [k for k, v in (("lambda", lam), ("mu", mu)) if v is None]
    if missing:
        summary.append(
            "M/M/c 排队 跳过：需要到达率 λ 与单服务台服务率 μ——"
            f"缺 {missing}。用 config['lambda']/['mu']（与可选 config['servers']，默认 1）提供"
            "（或在数据中提供同名列）。"
        )
        return
    c = int(round(c_raw)) if (c_raw is not None and c_raw >= 1) else 1
    if not (lam > 0 and mu > 0):
        summary.append(
            f"M/M/c 排队 跳过：到达率 λ({lam:.4g}) 与服务率 μ({mu:.4g}) 必须为正数。"
        )
        return
    try:
        a = lam / mu                  # offered load (erlangs)
        rho = a / c                   # server utilisation = λ/(c·μ)
        if rho >= 1.0:
            summary.append(
                f"M/M/c 排队 跳过：系统不稳定——利用率 ρ = λ/(c·μ) = "
                f"{lam:.4g}/({c}·{mu:.4g}) = {rho:.4f} ≥ 1，队列无界、无稳态。"
                f"请增加服务台数 c（当前 {c}）、提高服务率 μ 或降低到达率 λ。"
            )
            return

        # P0 via the standard sum; use logs-free direct sum (c is small).
        sum_terms = sum(a ** n / math.factorial(n) for n in range(c))
        last = a ** c / (math.factorial(c) * (1.0 - rho))
        p0 = 1.0 / (sum_terms + last)
        prob_wait = last * p0          # Erlang-C: P(an arrival must wait)
        lq = prob_wait * rho / (1.0 - rho)   # mean number waiting in queue
        wq = lq / lam                  # mean waiting time in queue
        w = wq + 1.0 / mu              # mean time in system
        ell = lq + a                   # mean number in system (L = Lq + λ/μ)

        # n-distribution table (P_n) for a few states, for the CSV/plot.
        nmax = max(c + 10, 15)
        pn = []
        for n in range(nmax + 1):
            if n < c:
                val = (a ** n / math.factorial(n)) * p0
            else:
                val = (a ** n / (math.factorial(c) * c ** (n - c))) * p0
            pn.append(val)
        pn = np.asarray(pn, dtype=float)
        pd.DataFrame({"n_in_system": np.arange(nmax + 1),
                      "prob_pn": np.round(pn, 8)}).to_csv(
            d / "queue_state_dist.csv", index=False, encoding="utf-8")
        files.append("queue_state_dist.csv")

        estimates.update({
            "rho": round(float(rho), 6),
            "p0": round(float(p0), 6),
            "lq": round(float(lq), 6),
            "wq": round(float(wq), 6),
            "l": round(float(ell), 6),
            "w": round(float(w), 6),
            "prob_wait": round(float(prob_wait), 6),
            "servers": float(c),
            "offered_load": round(float(a), 6),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ns = np.arange(nmax + 1)
            ax.bar(ns, pn, color="#4C72B0", edgecolor="white")
            ax.axvline(c - 0.5, color="#C44E52", ls="--", lw=1.2,
                       label=f"c = {c} servers")
            ax.set_xlabel("number in system n")
            ax.set_ylabel("P(n)")
            ax.set_title(f"M/M/{c} steady state (rho={rho:.2f}, Lq={lq:.3f})")
            ax.legend(fontsize=8)

        _save_fig(d, "queue_state_dist.png", files, _plot)

        summary.append(
            f"{ctx.entry.method} 完成：到达率 λ={lam:.6g}、服务率 μ={mu:.6g}（每台）、服务台 c={c}。"
            f"利用率 ρ=λ/(c·μ)={rho:.4f}（<1 稳定）；空系统概率 P0={p0:.4f}；"
            f"需等待概率（Erlang-C）={prob_wait:.4f}；队列平均长度 Lq={lq:.4f}、"
            f"平均排队时间 Wq={wq:.4f}；系统平均人数 L={ell:.4f}、平均逗留时间 W={w:.4f}。"
            "状态分布见 queue_state_dist.csv 与图。"
            " ⚠ M/M/c 假定：到达服从泊松过程（指数到达间隔）、服务时间服从指数分布、"
            "先到先服务(FCFS)、队列容量与顾客源无限、c 个同质并行服务台。"
            "现实中到达/服务常非指数（变异系数≠1）、有放弃/优先级/有限容量，"
            "届时应改用 M/G/c、G/G/c 近似或仿真。Wq、Lq 满足 Little 定律 L=λ·W。"
        )
        code += [
            "import math",
            "a = lam/mu; rho = a/c                         # 负荷与利用率",
            "P0 = 1/(sum(a**n/math.factorial(n) for n in range(c))",
            "        + a**c/(math.factorial(c)*(1-rho)))   # 空系统概率",
            "Pw = a**c/(math.factorial(c)*(1-rho))*P0      # Erlang-C 等待概率",
            "Lq = Pw*rho/(1-rho); Wq = Lq/lam              # 队长与排队时间",
            "L  = Lq + a;          W  = Wq + 1/mu          # Little 定律",
        ]
    except Exception as exc:
        summary.append(f"M/M/c 排队 计算失败：{exc}")


# ===========================================================================
# 3) newsvendor — single-period stochastic inventory (critical-fractile)
#    Refs: Arrow, Harris & Marschak (1951); Porteus "Foundations of Stochastic
#          Inventory Theory"; Cachon & Terwiesch "Matching Supply with Demand".
#    Critical ratio CR = Cu/(Cu+Co).
#    Normal demand: Q* = μ_d + z·σ_d, z = Φ⁻¹(CR).
#    Empirical demand (a demand column): Q* = the CR-quantile of the data.
# ===========================================================================
@register("newsvendor")
def _branch_newsvendor(ctx: Ctx) -> None:
    df, cfg = ctx.df, ctx.cfg
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    # --- underage / overage costs ------------------------------------------ #
    cu = _scalar(cfg, df, "underage_cost")
    co = _scalar(cfg, df, "overage_cost")
    price = _scalar(cfg, df, "price")
    cost = _scalar(cfg, df, "cost")
    salvage = _scalar(cfg, df, "salvage")
    if cu is None and price is not None and cost is not None:
        cu = price - cost                       # underage = lost margin per unit
    if co is None and cost is not None:
        co = cost - (salvage if salvage is not None else 0.0)  # overage = cost - salvage

    if cu is None or co is None:
        summary.append(
            "报童模型(newsvendor) 跳过：需要缺货成本 Cu 与过量成本 Co——"
            "用 config['underage_cost']/['overage_cost']，"
            "或 config['price']+['cost'](+['salvage']) 推导（Cu=price−cost、Co=cost−salvage）。"
        )
        return
    if not (cu > 0 and co > 0):
        summary.append(
            f"报童模型(newsvendor) 跳过：缺货成本 Cu({cu:.4g}) 与过量成本 Co({co:.4g}) 必须为正数"
            "（否则临界比无意义）。"
        )
        return

    cr = cu / (cu + co)                          # critical ratio (critical fractile)

    # --- demand model: empirical column if given, else normal(μ_d, σ_d) ---- #
    demand_col = cfg.get("demand_col")
    demand_vals = None
    if demand_col is not None and demand_col in df.columns:
        demand_vals = pd.to_numeric(df[demand_col], errors="coerce").dropna().to_numpy()
        if demand_vals.size < 2:
            demand_vals = None  # too few -> fall back to scalar μ/σ if available

    mu_d = _scalar(cfg, df, "demand_mean")
    sigma_d = _scalar(cfg, df, "demand_sd")

    try:
        if demand_vals is not None and demand_vals.size >= 2:
            # EMPIRICAL: optimal Q* is the CR-quantile of the demand sample.
            mode = "empirical"
            q_star = float(np.quantile(demand_vals, cr))
            mu_emp = float(np.mean(demand_vals))
            sd_emp = float(np.std(demand_vals, ddof=1))
            z = float("nan")
            # empirical expected profit (per-period) at Q* over the sample, assuming
            # selling price = Cu + cost? we cannot know revenue, so report expected
            # underage+overage MISMATCH COST (the quantity the model minimises):
            #   cost(Q) = Cu·E[(D-Q)+] + Co·E[(Q-D)+].
            short = np.maximum(demand_vals - q_star, 0.0)
            over = np.maximum(q_star - demand_vals, 0.0)
            exp_cost = float(cu * short.mean() + co * over.mean())
            fill_rate = float(np.mean(np.minimum(demand_vals, q_star)) /
                              max(mu_emp, 1e-12))
            mu_used, sd_used = mu_emp, sd_emp
            exp_profit = float("nan")  # no revenue info -> profit N/A
            note = (f"经验分布：用需求列 {demand_col}（n={demand_vals.size}）的 "
                    f"CR={cr:.4f} 分位数作为 Q*；均值={mu_emp:.4g}、标准差={sd_emp:.4g}")
        else:
            # NORMAL: Q* = μ_d + z·σ_d, z = Φ⁻¹(CR).
            if mu_d is None or sigma_d is None:
                summary.append(
                    "报童模型(newsvendor) 跳过：正态需求需要均值 μ_d 与标准差 σ_d——"
                    "用 config['demand_mean']/['demand_sd']，或提供 config['demand_col'] 用经验分布。"
                )
                return
            if not (sigma_d >= 0):
                summary.append(f"报童模型(newsvendor) 跳过：需求标准差 σ_d({sigma_d:.4g}) 必须非负。")
                return
            mode = "normal"
            z = _inv_norm_cdf(cr)
            q_star = mu_d + z * sigma_d
            mu_used, sd_used = mu_d, sigma_d
            # expected mismatch cost for normal demand:
            #   E[(D-Q)+] = σ·[φ(z) - z·(1-Φ(z))]   (standard normal loss function)
            #   E[(Q-D)+] = E[(D-Q)+] + (Q-μ)
            try:
                from scipy.stats import norm
                phi = float(norm.pdf(z))
                Phi = float(norm.cdf(z))
            except Exception:
                import math
                phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            exp_short = sigma_d * (phi - z * (1.0 - Phi))   # E[(D-Q)+]
            exp_over = exp_short + (q_star - mu_d)          # E[(Q-D)+]
            exp_cost = float(cu * exp_short + co * exp_over)
            fill_rate = float(1.0 - exp_short / mu_d) if mu_d > 0 else float("nan")
            # expected profit only if a selling price is available:
            #   profit = price·E[sales] - cost·Q + salvage·E[(Q-D)+], E[sales]=μ-E[(D-Q)+]
            if price is not None and cost is not None:
                exp_sales = mu_d - exp_short
                sv = salvage if salvage is not None else 0.0
                exp_profit = float(price * exp_sales - cost * q_star + sv * exp_over)
            else:
                exp_profit = float("nan")
            note = (f"正态需求：μ_d={mu_d:.4g}、σ_d={sigma_d:.4g}；"
                    f"Q*=μ_d+z·σ_d，z=Φ⁻¹(CR)={z:.4f}")

        # newsvendor CR-quantile illustration table
        crs = np.linspace(0.01, 0.99, 50)
        if mode == "empirical":
            qq = np.quantile(demand_vals, crs)
        else:
            qq = mu_used + np.array([_inv_norm_cdf(p) for p in crs]) * sd_used
        pd.DataFrame({"critical_ratio": np.round(crs, 6),
                      "optimal_q": np.round(qq, 6)}).to_csv(
            d / "newsvendor_curve.csv", index=False, encoding="utf-8")
        files.append("newsvendor_curve.csv")

        estimates.update({
            "critical_ratio": round(float(cr), 6),
            "q_star": round(float(q_star), 6),
            "z_value": round(float(z), 6) if z == z else float("nan"),
            "expected_profit": round(float(exp_profit), 6) if exp_profit == exp_profit else float("nan"),
            "expected_cost": round(float(exp_cost), 6) if exp_cost == exp_cost else float("nan"),
            "fill_rate": round(float(fill_rate), 6) if fill_rate == fill_rate else float("nan"),
            "underage_cost": round(float(cu), 6),
            "overage_cost": round(float(co), 6),
            "demand_mean": round(float(mu_used), 6),
            "demand_sd": round(float(sd_used), 6),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(crs, qq, color="#4C72B0", lw=1.8, label="optimal Q vs CR")
            ax.axvline(cr, color="#C44E52", ls="--", lw=1.3,
                       label=f"CR = {cr:.3f}")
            ax.axhline(q_star, color="#55A868", ls=":", lw=1.2,
                       label=f"Q* = {q_star:.2f}")
            ax.set_xlabel("critical ratio CR = Cu/(Cu+Co)")
            ax.set_ylabel("optimal order quantity Q*")
            ax.set_title(f"Newsvendor critical-fractile ({mode} demand)")
            ax.legend(fontsize=8)

        _save_fig(d, "newsvendor_curve.png", files, _plot)

        profit_txt = (f"；期望利润={exp_profit:.6g}" if exp_profit == exp_profit
                      else "；期望利润 N/A（未提供 price/cost，无法算收入）")
        fill_txt = (f"、满足率≈{fill_rate:.1%}" if fill_rate == fill_rate else "")
        summary.append(
            f"{ctx.entry.method} 完成：缺货成本 Cu={cu:.6g}、过量成本 Co={co:.6g}。"
            f"临界比 CR=Cu/(Cu+Co)={cr:.4f}（即最优 Q* 使 P(需求≤Q*)=CR）。{note}。"
            f"最优订货量 Q*={q_star:.4f}{profit_txt}；期望失配成本(Cu·缺货+Co·过量)={exp_cost:.6g}{fill_txt}。"
            "Q*–CR 曲线见 newsvendor_curve.csv 与图。"
            " ⚠ 报童模型为单期（一次性订货、期末残值/损失，不可补货、无库存结转）。"
            + ("正态需求假定：可能给出负的 Q*（当 CR<0.5 且 σ 大），需截断到 ≥0；"
               "若需求偏态/有界，正态近似失真，请用经验分布(demand_col)或对数正态等。"
               if mode == "normal" else
               "经验分布直接取样本 CR 分位数，受样本量与代表性限制（小样本分位数不稳）；"
               "未做平滑/外推。")
            + " ⚠ Cu/Co 的口径决定结果：Cu 含缺货的边际损失（利润+商誉），Co 含过量的边际损失"
            "（成本−残值+持有/处置）；漏算商誉/惩罚会低估 Q*。"
        )
        code += [
            "cr = cu/(cu+co)                              # 临界比(critical fractile)",
            "# 正态需求:",
            "from scipy.stats import norm",
            "z = norm.ppf(cr); q_star = mu_d + z*sigma_d  # 最优订货量",
            "# 经验需求(给定 demand 列):",
            "q_star = np.quantile(demand, cr)             # CR 分位数",
        ]
    except Exception as exc:
        summary.append(f"报童模型(newsvendor) 计算失败：{exc}")
