"""Branch handlers for the timeseries family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome


def _periodogram_period(x, n):
    """Dominant seasonal period via the periodogram, or None if no SIGNIFICANT periodicity.
    Linearly detrends first (a trend's low-frequency power otherwise dominates), requires >=3
    cycles (period <= n/3), and applies Fisher's g-test (alpha=0.05) so pure noise/trend -> None."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    idx = np.arange(n)
    c = np.polyfit(idx, x, 1)        # remove linear trend
    x = x - (c[0] * idx + c[1])
    if np.std(x) == 0:
        return None
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    mask = freqs >= 3.0 / n          # candidate seasonal freqs (period <= n/3)
    if not mask.any():
        return None
    pm = power[mask]
    m = len(pm)
    if m < 2 or pm.sum() <= 0:
        return None
    g = float(pm.max() / pm.sum())                    # Fisher's g statistic
    g_crit = 1.0 - (0.05 / m) ** (1.0 / (m - 1))      # alpha=0.05 critical value
    if g <= g_crit:                                   # no significant periodicity
        return None
    freq = freqs[mask][int(np.argmax(pm))]
    if freq <= 0:
        return None
    per = int(round(1.0 / freq))
    return per if 2 <= per <= n // 3 else None


# ── auto-order selection (Hyndman-Khandakar style) ───────────────────────────────────
_GRID_FIT_BUDGET = 48   # hard cap on candidate fits so a long grid can't stall a run
# Wall-clock ceiling for the whole order search. A seasonal state-space model carries `sp`
# lags of state, so one fit costs 0.2s at sp=0 and 219s at sp=52 (measured, n=2225) — a
# fit-COUNT budget alone cannot bound the work. The search spends up to this long, keeps
# the best candidate it reached, and discloses that it stopped early.
_SEARCH_TIME_BUDGET_S = 20.0
# Projected wall-clock SECONDS for one candidate fit. Two corrections from cold review B:
#
# 1. The state dimension is statsmodels' own k_states (sarimax.py:425 / 453-457, and
#    simple_differencing defaults to False, so the differencing DOES ride in the state):
#        k_states = max(p + sp*P, q + sp*Q + 1) + sp*D + d
#    verified against `SARIMAX(...).k_states` on 12 order combinations, 12/12. The previous
#    model `max(p, q+1) + sp*(P+Q)` was wrong twice — it dropped `sp*D + d`, and it ADDED the
#    AR and MA sides where statsmodels takes their max. (The 266x gap I had used to argue that
#    D does not count was optimiser ITERATIONS, nfev 15 vs 212 — the dimensions are 54 vs 106.)
#
# 2. The budget is time, not a dimensionless score. Measured fit time tracks
#    t ≈ c · n · k_states**4 (exponent 4.09 / 4.06 across sp=12/24/52); the old n·dim² made one
#    threshold mean 8s at sp=12, 31s at sp=24 and 151s at sp=52 — 19x spread, and looser for
#    exactly the big periods that motivated the gate. c is calibrated on this machine
#    (n=600, sp=52, k_states=107 -> 51.1s measured); projections are labelled as estimates.
_KALMAN_SEC_PER_UNIT = 6.5e-10


def _sarimax_k_states(order, sorder) -> int:
    """statsmodels' state dimension for a SARIMAX with simple_differencing=False.

    Mirrors sarimax.py:425 (`_k_order`) and 453-457 (`k_states += sp*seasonal_diff + diff`).
    Module-level and covered by a test that compares it with `SARIMAX(...).k_states` itself,
    so the formula cannot drift from the library without something going red."""
    p, d, q = order
    P, D, Q, s = sorder
    return max(p + int(s) * P, q + int(s) * Q + 1) + int(s) * D + d


def _fit_seconds(n, order, sorder) -> float:
    """Projected wall-clock seconds for ONE SARIMAX fit — see _KALMAN_SEC_PER_UNIT."""
    return _KALMAN_SEC_PER_UNIT * float(n) * float(_sarimax_k_states(order, sorder)) ** 4



def _ndiffs_adf(y, max_d: int = 2) -> int:
    """Non-seasonal differencing order `d` chosen by the ADF unit-root TEST, never by AIC.

    The log-likelihood (hence AIC) is computed on the DIFFERENCED sample, so models with a
    different `d` are fitted to different effective data and their AICs are not comparable —
    ranking `d` by AIC is a classic error. Hyndman-Khandakar therefore fixes `d` by test first
    and ranks only (p,q) by information criterion. Returns 0 when the level series is already
    stationary; degrades to the current level on any test failure."""
    import numpy as np
    from statsmodels.tsa.stattools import adfuller

    # adfuller raises on NaN; without this a couple of gaps would silently abort the loop at
    # k=0 and return d=0 (under-differencing) while the summary still credits "ADF 检验"
    # (inference-review SHOULD-FIX). SARIMAX itself handles NaN natively via the Kalman filter,
    # so only the TEST needs the clean copy.
    z = np.asarray(y, dtype=float)
    z = z[np.isfinite(z)]
    for k in range(max_d + 1):
        if len(z) < 8 or float(np.std(z)) == 0.0:
            return k
        try:
            if float(adfuller(z, autolag="AIC")[1]) <= 0.05:
                return k                       # stationary at this differencing level
        except Exception:
            return k
        z = np.diff(z)
    return max_d


def _fit_sarimax(y, order, seasonal_order):
    """One SARIMAX fit (ARIMA is the seasonal_order=(0,0,0,0) special case, so the whole search
    uses ONE estimator — keeping every candidate's likelihood on the same footing).

    Two settings are load-bearing for the ORDER SEARCH and must not be relaxed:

    * ``enforce_stationarity/invertibility=True`` — with them OFF, statsmodels cannot use the
      stationary initialization and falls back to an approximate-diffuse one whose
      ``loglikelihood_burn`` grows with the state dimension (hence with p,q,P,Q). The
      log-likelihood is then summed over a DIFFERENT effective sample per candidate, so AIC/AICc
      are not comparable and the search stampedes toward the largest order (inference-review
      MUST-FIX: on a random walk the truth (0,1,0) was picked 0/30 times, mean p+q=3.7). With
      them ON the burn is constant ``d + D*sp`` across the whole grid, which is what makes the
      "fixed d/D ⇒ comparable" claim actually true.
    * ``trend='c'`` when nothing is differenced — SARIMAX defaults to NO constant, so an
      undifferenced series would be fitted as a MEAN-ZERO process (a series around 500 forecasts
      0.0). statsmodels' ARIMA wrapper defaults to a constant at d==0; that difference only
      surfaced once auto-`d` made d=0 routine.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    trend = "c" if (order[1] == 0 and seasonal_order[1] == 0) else None
    return SARIMAX(y, order=order, seasonal_order=seasonal_order, trend=trend,
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)


def _aicc(res) -> float:
    """Small-sample-corrected AIC. Plain AIC under-penalises parameters at the sample sizes
    typical of a seasonal series (a few dozen points), which biases the search toward
    over-parameterised models; AICc is the standard correction for ARIMA order selection.

    Uses statsmodels' own ``res.aicc``, which divides by the EFFECTIVE sample
    (``nobs - loglikelihood_burn``) — the sample the likelihood was actually computed on.
    Computing it from ``res.nobs`` (the full series length) understates the small-sample
    penalty, i.e. biases toward over-parameterisation (inference-review SHOULD-FIX)."""
    import numpy as np

    try:
        val = float(res.aicc)
    except Exception:
        return float("inf")
    return val if np.isfinite(val) else float("inf")


def _auto_order(y, sp, cfg):
    """Choose (order, seasonal_order) by a bounded AICc grid with d/D FIXED FIRST.

    Returns (order, seasonal_order, fitted_result, info). `info` records what was actually
    searched so the summary can disclose it honestly (a bounded grid is NOT a full
    Hyndman-Khandakar stepwise search). Never raises: returns fitted_result=None when every
    candidate failed, leaving the caller to fall back."""
    import numpy as np

    cfg = cfg or {}
    seasonal = bool(sp)

    def _int_cfg(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(cfg[key])))
        except (KeyError, TypeError, ValueError):
            return default

    d = _int_cfg("d", -1, 0, 2)
    d_source = "config"
    if d < 0:
        d, d_source = _ndiffs_adf(y), "ADF 检验"
    D = 1 if seasonal else 0
    # seasonal grid is deliberately narrower: each seasonal fit is far costlier and D=1 already
    # removes most seasonal non-stationarity.
    max_p = _int_cfg("max_p", 2 if seasonal else 3, 0, 5)
    max_q = _int_cfg("max_q", 2 if seasonal else 3, 0, 5)
    max_P = _int_cfg("max_P", 1, 0, 2) if seasonal else 0
    max_Q = _int_cfg("max_Q", 1, 0, 2) if seasonal else 0

    cands = [
        ((p, d, q), (P, D, Q, sp) if seasonal else (0, 0, 0, 0))
        for p in range(max_p + 1) for q in range(max_q + 1)
        for P in range(max_P + 1) for Q in range(max_Q + 1)
    ]
    # parsimonious-first, so a truncated budget still covers the simple models
    cands.sort(key=lambda c: (c[0][0] + c[0][2] + c[1][0] + c[1][2]))
    import time as _time

    _budget = float(cfg.get("search_seconds") or _SEARCH_TIME_BUDGET_S)
    _n = int(len(y))
    # One fit may spend the whole search budget, never more. The previous gate let a candidate
    # projected under a dimensionless threshold run for 51s inside a 20s search budget
    # (measured: n=600, sp=52, (0,1,1)(0,1,1,52)) — a per-fit cap in the same unit as the
    # search budget is what makes the two coherent.
    _fit_budget_s = float(cfg.get("fit_seconds") or _budget)

    skipped_costly = 0
    skipped_min_s = None
    best, best_ic, n_fits = None, np.inf, 0
    timed_out = False
    _t0 = _time.perf_counter()
    for order, sorder in cands[:_GRID_FIT_BUDGET]:
        # Candidates are parsimony-sorted, so stopping early keeps the simple models that were
        # already fitted rather than abandoning the search with nothing.
        if best is not None and _time.perf_counter() - _t0 > _budget:
            timed_out = True
            break
        _proj = _fit_seconds(_n, order, sorder)
        if _proj > _fit_budget_s:
            skipped_costly += 1
            skipped_min_s = _proj if skipped_min_s is None else min(skipped_min_s, _proj)
            continue
        try:
            res = _fit_sarimax(y, order, sorder)
        except Exception:
            continue
        n_fits += 1
        ic = _aicc(res)
        if ic < best_ic:
            best, best_ic = (order, sorder, res), ic
    _elapsed = _time.perf_counter() - _t0
    # A winner sitting ON the grid edge means the search was cut short of the true optimum —
    # report it rather than presenting a boundary pick as "the" selected order.
    at_edge = False
    if best is not None:
        (bp, _, bq), (bP, _, bQ, _) = best[0], best[1]
        # Only the NON-seasonal edges are reported. At the default max_P=max_Q=1 the textbook
        # airline model (0,1,1)(0,1,1)[s] sits on the seasonal edge by construction, so flagging
        # it would cry wolf on the most common correct answer; seasonal orders above 1 are rare.
        at_edge = bool((max_p > 0 and bp == max_p) or (max_q > 0 and bq == max_q)
                       or (seasonal and ((max_P > 1 and bP == max_P)
                                         or (max_Q > 1 and bQ == max_Q))))
    info = {
        "d": d, "D": D, "d_source": d_source, "n_fits": n_fits,
        "grid": f"p≤{max_p}, q≤{max_q}" + (f", P≤{max_P}, Q≤{max_Q}" if seasonal else ""),
        "aicc": None if best is None else float(best_ic),
        "truncated": len(cands) > _GRID_FIT_BUDGET or timed_out,
        "timed_out": timed_out,
        "skipped_costly": skipped_costly,
        "skipped_min_s": None if skipped_min_s is None else round(float(skipped_min_s), 1),
        "fit_budget_s": round(float(_fit_budget_s), 1),
        "elapsed_s": round(float(_elapsed), 1),
        "n_candidates": len(cands),
        "at_edge": at_edge,
    }
    if best is None:
        # The fallback must respect the cost gate too: skipping 36 costly candidates saved
        # nothing when the caller then fitted (1,d,1)(1,D,1,sp) — the very model that costs
        # 219s. Fall back to the cheapest admissible shape instead.
        _fb_order, _fb_sorder = (1, d, 1), ((1, D, 1, sp) if seasonal else (0, 0, 0, 0))
        if seasonal and _fit_seconds(_n, _fb_order, _fb_sorder) > _fit_budget_s:
            _fb_order, _fb_sorder = (0, d, 1), (0, D, 0, sp)
            info["fallback_cheap"] = True
        return _fb_order, _fb_sorder, None, info
    return best[0], best[1], best[2], info


@register("arima")
def _branch_arima(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    time_col = fp.time_col
    # value_col: config override (the family convention — `column`/`value`, same keys every
    # other TS entry declares), else the first continuous column. Time columns are
    # datetime/id/count kind (never continuous), so they are never picked here.
    #
    # arima was the ONE branch in this module without the override (its three siblings below
    # all have it) and its catalog entry declared no params at all — so on a multi-series
    # frame the engine forecast whichever series came first and the user could not ask for
    # another one.
    _excl = {fp.unit_col, fp.time_col}
    _cfg_col = cfg.get("column") or cfg.get("value")
    value_col = _cfg_col if _cfg_col in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)

    if time_col is None or value_col is None:
        summary.append(
            "ARIMA 失败：未找到时间列或连续值列，请检查数据结构。"
        )
    else:
        try:
            import numpy as np

            sorted_df = df.sort_values(time_col)
            dup = int(sorted_df[time_col].duplicated().sum())
            if dup:
                sorted_df = sorted_df.drop_duplicates(subset=time_col, keep="first")
                summary.append(f"注意：{dup} 个重复时间点已去重（保留首次）。")
            y = sorted_df[value_col].astype(float).reset_index(drop=True)
            if y.nunique() < 2 or len(y) < 10:
                raise ValueError(f"序列有效观测不足或近常数（n={len(y)}），无法拟合 ARIMA")

            # Seasonal period: the calendar-aware, strength-confirmed detector
            # (forecasting._detect_period; lazy import breaks the timeseries↔forecasting cycle).
            from researchforge.executor.branches.forecasting import _detect_period

            n = len(y)
            sp = _detect_period(ctx, y.to_numpy())
            if cfg.get("seasonal") in {"none", "no", "off"}:
                sp = None
            degrade_note = ""
            # Seasonal DIFFERENCING (D=1) consumes `sp` observations on top of the two cycles
            # Holt-Winters needs, so require ≥3 full cycles: (n - sp) >= 2*sp ⇔ n >= 3*sp.
            # Below it the seasonal AR/MA at lag `sp` is not identifiable — SARIMAX still returns
            # converged=True with a boundary (llf≈0) AIC that would read as a spuriously
            # excellent model (inference-review MUST-FIX).
            if sp and n < 3 * sp:
                degrade_note = (
                    f" ⚠ 已检出季节周期={sp}，但样本不足以稳健估计季节差分模型"
                    f"（季节差分需 ≥3 个完整周期，n={n}<{3 * sp}），已改用非季节模型。"
                )
                sp = None

            order, sorder, model, oinfo = _auto_order(y.to_numpy(), sp, cfg)
            if model is None:
                # Cold review B/A1: this used to OVERWRITE _auto_order's return with
                # (1,d,1)(1,D,1,sp) — the very 219s model the cost gate had just skipped, so
                # the hang protection was void on exactly this path. _auto_order already
                # returns a budget-respecting order; fit THAT.
                model = _fit_sarimax(y.to_numpy(), order, sorder)
                _why = ("全部候选的投影计算成本都超出预算" if oinfo.get("skipped_costly")
                        and not oinfo["n_fits"] else "网格全部拟合失败")
                degrade_note += (
                    f" ⚠ 自动定阶未能拟合任何候选（{_why}），已回退 "
                    f"{order}{sorder[:3]} —— **该阶数未经 AICc 比较**，不是任何候选集里的最优。"
                )
            seasonal_used = int(sorder[3]) if sp else 0
            if not bool(getattr(model, "mle_retvals", {}).get("converged", True)):
                degrade_note += (
                    " ⚠ 参数优化未完全收敛（近确定性/强季节数据常见）：预测仍反映拟合结构，"
                    "但 AIC 与标准误可能不可靠。"
                )
            model_label = (
                f"SARIMA{order}{sorder[:3]}[{seasonal_used}]".replace(" ", "")
                if seasonal_used else f"ARIMA{order}".replace(" ", "")
            )

            (d / "model_summary.txt").write_text(str(model.summary()), encoding="utf-8")
            files.append("model_summary.txt")

            # ── forecast WITH a prediction interval ───────────────────────────────────
            # A point forecast without an interval is not reportable. The state-space
            # get_forecast() gives the MODEL-CONSISTENT interval directly — it widens correctly
            # with the fitted AR/MA + differencing structure — so there is no reason to omit it.
            steps = 10
            try:
                level = float(cfg.get("ci", 0.95))
            except (TypeError, ValueError):
                level = 0.95
            level = level if 0.5 < level < 1.0 else 0.95
            alpha = 1.0 - level
            fcres = model.get_forecast(steps=steps)
            fc = np.asarray(fcres.predicted_mean, dtype=float)
            ci_arr = np.asarray(fcres.conf_int(alpha=alpha), dtype=float)
            lower, upper = ci_arr[:, 0], ci_arr[:, 1]

            import pandas as _pd
            _pd.DataFrame({
                "step": list(range(1, steps + 1)),
                "forecast": fc,
                "lower": lower,
                "upper": upper,
            }).to_csv(d / "forecast.csv", index=False, encoding="utf-8")
            files.append("forecast.csv")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(len(y)), y, label="observed")
                fc_x = list(range(len(y), len(y) + steps))
                ax.plot(fc_x, fc, color="red", linestyle="--", label="forecast")
                ax.fill_between(fc_x, lower, upper, color="red", alpha=0.15,
                                label=f"{level:.0%} prediction interval")
                ax.set_xlabel("period index")
                ax.set_ylabel(value_col)
                ax.set_title(f"{model_label} — {value_col}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(d / "forecast.png", dpi=150)
                plt.close(fig)
                files.append("forecast.png")
            except Exception:
                pass

            estimates["aic"] = float(model.aic)
            if oinfo["aicc"] is not None:
                estimates["aicc"] = float(oinfo["aicc"])
            estimates["seasonal_periods"] = float(seasonal_used)
            estimates["p"], estimates["d"], estimates["q"] = (
                float(order[0]), float(order[1]), float(order[2]))
            if seasonal_used:
                estimates["P"], estimates["D"], estimates["Q"] = (
                    float(sorder[0]), float(sorder[1]), float(sorder[2]))
            estimates["forecast_next"] = float(fc[0])
            estimates["pi_lower_next"] = float(lower[0])
            estimates["pi_upper_next"] = float(upper[0])

            order_zh = (
                f"阶数由自动定阶选出（差分 d={oinfo['d']} 来自{oinfo['d_source']}，随后在 "
                f"{oinfo['grid']} 的网格上按 AICc 排序，实拟合 {oinfo['n_fits']} 个候选"
                + (f"；网格超预算已截断，按简约优先取前 {_GRID_FIT_BUDGET} 个"
                   if oinfo["truncated"] and not oinfo.get("timed_out") else "")
                # 墙钟预算：季节状态空间模型带 sp 阶状态，单次拟合的成本随周期爆炸
                # （实测 n=2225：sp=0 时 0.2s，sp=52 时 219s）。只报「拟合了几个候选」会让
                # 用户以为搜索完整；必须说清是时间到了，以及怎么要更多。
                # A4: the old text asserted "单次拟合可达数百秒" regardless of sp and n — it
                # printed that on an sp=12 / n=72 run whose seasonal fits take 0.4s. Print the
                # projection that was actually computed. A1: "可承受候选中的最优" is only true
                # when something was in fact fitted.
                + (f"；⚠ 有 {oinfo.get('skipped_costly')} 个候选的投影单次拟合耗时超过 "
                   f"{oinfo.get('fit_budget_s')}s 的预算被跳过"
                   + (f"（最便宜的一个估计约 {oinfo.get('skipped_min_s')}s，按本机基准）"
                      if oinfo.get("skipped_min_s") is not None else "")
                   + ("——当前阶数是**可承受候选中的最优**" if oinfo["n_fits"]
                      else "——**一个候选都没能拟合**，当前阶数是未经比较的回退阶数")
                   + "，被跳过的候选完全可能才是 AICc 最优的"
                   '（实测过 11637 vs 16412 的差距）；若要搜索它们：'
                   'config={"search_seconds":<秒>} 或 {"fit_seconds":<秒>}'
                   if oinfo.get("skipped_costly") else "")
                + (f"；⚠ 搜索在 {oinfo.get('elapsed_s')}s 处达到时间预算而提前停止"
                   f"（季节周期 {sp} 下单次拟合很贵，{oinfo.get('n_candidates')} 个候选只试了 "
                   f"{oinfo['n_fits']} 个），当前阶数是**已试候选中的最优**、未必是全局最优——"
                   'config={"search_seconds":<秒>} 可放宽，或 config seasonal="none" 关季节'
                   if oinfo.get("timed_out") else "")
                + "）。"
                + ("⚠ 选中阶数落在网格边界，真优可能在更高阶——可 config max_p/max_q(/max_P/max_Q) 放宽后重跑。"
                   if oinfo.get("at_edge") else "")
                + "⚠ d 由单位根检验固定、不参与 AICc 比较——不同差分阶数的似然基于不同有效样本，"
                "AIC/AICc 跨 d 不可比；固定 d/D 后的 (p,q[,P,Q]) 比较才有效。"
                "⚠ 这是有界网格、非完整 Hyndman-Khandakar 逐步搜索，可 config max_p/max_q/"
                "max_P/max_Q/d 调整。"
            )
            season_zh = (
                f"季节周期={seasonal_used}（按日期频率+季节强度自动判定，可 config seasonal_periods "
                "覆盖 / seasonal=none 关闭）。⚠ 此 SARIMA 的 AIC 因含季节差分，不可与非季节 ARIMA "
                "或其他方法的 AIC 直接比较。"
                if seasonal_used else "未检出可靠季节（如为季节数据可 config seasonal_periods 指定）。"
            )
            summary.append(
                f"{entry.method} 完成：对 {value_col} 拟合 {model_label}，"
                f"AIC={model.aic:.2f}"
                + (f"、AICc={oinfo['aicc']:.2f}" if oinfo["aicc"] is not None else "")
                + f"；预测未来 {steps} 期，含 {level:.0%} 预测区间"
                f"（下一期 {fc[0]:.4g}，区间 [{lower[0]:.4g}, {upper[0]:.4g}]，见 forecast.csv/png）。"
                f"{order_zh}{season_zh}"
                " ⚠ 预测区间为模型一致的状态空间区间，但条件于所选阶数与所估参数——既未计入"
                "定阶本身的不确定性，也未计入参数估计误差"
                "（自动选阶后区间偏窄是已知现象）。"
                + degrade_note
            )
            code += [
                "from statsmodels.tsa.statespace.sarimax import SARIMAX",
                f"y = df.sort_values('{time_col}')['{value_col}'].astype(float).reset_index(drop=True)",
                f"model = SARIMAX(y, order={order}, seasonal_order={sorder},"
                " enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)",
                "print(model.summary())",
                f"fc = model.get_forecast(steps={steps})",
                f"mean, ci = fc.predicted_mean, fc.conf_int(alpha={alpha:.3g})  # 点预测 + 预测区间",
            ]
        except Exception as err:
            summary.append(f"ARIMA 拟合失败：{err}")



@register("var_granger")
def _branch_var_granger(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
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
                forced_lag1 = False
                if res.k_ar < 1:
                    res = VAR(data).fit(1)  # AIC picked 0 lags -> force lag 1 for Granger
                    forced_lag1 = True
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
                lag_note = "（AIC 选 0，已强制为 1 阶）" if forced_lag1 else "（AIC 选）"
                summary.append(
                    f"{entry.method} 完成：{len(series)} 个序列 × {n} 期，VAR 阶数={res.k_ar}{lag_note}；"
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


@register("cointegration_vecm")
def _branch_cointegration_vecm(ctx: Ctx) -> None:
    # Cointegration (Engle-Granger + Johansen) and, if cointegrated, a VECM:
    # long-run equilibrium relation among I(1) series + short-run adjustment speeds.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    _excl = {fp.unit_col, fp.time_col}
    forced = [c for c in (cfg.get("series") or cfg.get("predictors") or []) if c in df.columns and c not in _excl]
    series = (forced if len(forced) >= 2 else
              [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl])[:6]
    if len(series) < 2:
        summary.append("协整/VECM 失败：需要 ≥2 个连续时间序列变量（config['series'] 可指定）。")
        return
    try:
        from statsmodels.tsa.stattools import adfuller, coint
        from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank, select_order

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        data = d2[series].dropna().reset_index(drop=True).astype(float)
        n = len(data)
        if n < 25:
            summary.append("协整/VECM 失败：观测不足（<25），无法稳健做 Johansen/VECM。")
            return
        # I(1) precondition: levels should be non-stationary, first differences stationary
        lvl_nonstat = sum(adfuller(data[s].to_numpy(), autolag="AIC")[1] > 0.05 for s in series)
        diff_stat = sum(adfuller(data[s].diff().dropna().to_numpy(), autolag="AIC")[1] <= 0.05 for s in series)
        # lag order (in differences) by AIC; fall back to 1
        try:
            kmax = max(1, min(8, n // (len(series) + 1) - 1))
            k = max(1, int(select_order(data, maxlags=kmax, deterministic="ci").aic))
        except Exception:
            k = 1
        joh = coint_johansen(data, det_order=0, k_ar_diff=k)
        trace, cv95 = joh.lr1, joh.cvt[:, 1]
        # Johansen trace is SEQUENTIAL (test rank<=0, rank<=1, …; STOP at the first non-rejection).
        # Summing exceedances over-counts (~2.5% of cases): the trace stats AND their critical values
        # both shrink across steps and can re-cross. Use the canonical sequential routine.
        r = int(select_coint_rank(data, det_order=0, k_ar_diff=k, signif=0.05).rank)
        eg_p = float(coint(data[series[0]], data[series[1]])[1])  # Engle-Granger (first pair)

        estimates.update({
            "n_coint_relations": float(r), "johansen_trace_r0": float(trace[0]),
            "johansen_cv95_r0": float(cv95[0]), "eg_pvalue_pair": round(eg_p, 4),
            "levels_nonstationary": float(lvl_nonstat), "diffs_stationary": float(diff_stat),
            "k_ar_diff": float(k), "n_obs": float(n),
        })
        # NOTE: reject_pointwise is an element-wise trace>cv95 flag per row; it is NOT the
        # authoritative rank (sequential re-crossing can make it disagree with select_coint_rank).
        # The authoritative rank is `r` above (from the sequential select_coint_rank routine).
        pd.DataFrame({"r_le": list(range(len(trace))), "trace_stat": np.round(trace, 3),
                      "crit_95": np.round(cv95, 3), "reject_pointwise_(coint>r)_not_authoritative": trace > cv95}
                     ).to_csv(d / "johansen_trace.csv", index=False, encoding="utf-8")
        files.append("johansen_trace.csv")

        longrun = ""
        full_rank = r >= len(series)  # r == #vars -> levels stationary (I(0)), not a cointegrated I(1) system
        if 1 <= r < len(series):
            # deterministic="co": matches the det_order=0 Johansen rank test above. statsmodels'
            # coint_johansen only tabulates nc/co/lo critical values (no "ci" table exists), so fitting
            # the VECM with the restricted-constant "ci" would test and fit under DIFFERENT deterministic
            # assumptions (co's lenient 95% critical values vs ci's stricter ones) -> spurious cointegration
            # in the gap between them. "co" keeps the constant unrestricted and OUTSIDE the cointegrating
            # relation, consistent with the rank test that was actually used to pick r.
            vecm = VECM(data, k_ar_diff=k, coint_rank=r, deterministic="co").fit()
            beta = np.asarray(vecm.beta)[:, 0].astype(float)
            alpha = np.asarray(vecm.alpha)[:, 0].astype(float)
            beta_n = beta / beta[0] if abs(beta[0]) > 1e-12 else beta
            terms = " ".join(f"{'+' if b >= 0 else '-'}{abs(b):.3f}·{s}" for b, s in zip(beta_n, series))
            longrun = f"长期均衡关系（标准化 {series[0]}=1）：{terms} ≈ 0；"
            estimates["adjustment_speed_eq1"] = round(float(alpha[0]), 4)
            # deterministic="co": the constant is unrestricted and lives OUTSIDE the cointegrating
            # relation (each equation's own intercept), so no constant term is added into the ECT here
            # (a nonzero mean of the equilibrium error shows up via the axhline in the plot instead).
            ect = data.to_numpy() @ beta
            try:
                ect_p = float(adfuller(ect, autolag="AIC")[1])
                estimates["ect_adf_pvalue"] = round(ect_p, 4)
            except Exception:
                pass
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 3.6))
                ax.plot(ect, color="#4C72B0")
                ax.axhline(float(np.mean(ect)), color="grey", ls="--", lw=1)
                ax.set_title("Cointegrating residual (ECT) — mean-reverting if cointegrated")
                ax.set_xlabel("period index")
                ax.set_ylabel("equilibrium error")
                fig.tight_layout()
                fig.savefig(d / "cointegration_ect.png", dpi=150)
                plt.close(fig)
                files.append("cointegration_ect.png")
            except Exception:
                pass
            (d / "vecm_summary.txt").write_text(str(vecm.summary()), encoding="utf-8")
            files.append("vecm_summary.txt")

        i1_note = ("" if (lvl_nonstat >= 1 and diff_stat >= 1) else
                   "；⚠ I(1) 前提存疑（levels 应非平稳、差分应平稳）——协整解读需谨慎")
        if full_rank:
            verdict = (f"协整秩 r={r} = 序列数 → levels 近似平稳(I(0))，不是 I(1) 协整系统；"
                       "协整/VECM 不适用，宜直接对 levels 建模(VAR)")
        elif r >= 1:
            verdict = (f"检出 {r} 个协整关系（Johansen trace 序贯检验，95%）；{longrun}"
                       f"调整速度 α₁={estimates.get('adjustment_speed_eq1')}（负=向均衡回拉）；"
                       f"ECT 回均值（ADF p={estimates.get('ect_adf_pvalue','—')}；注：基于估计的协整向量，p 偏乐观）")
        else:
            verdict = (f"未检出协整关系（Johansen trace r=0，trace={trace[0]:.2f} vs CV95={cv95[0]:.2f}；"
                       f"Engle-Granger 首对 p={eg_p:.3g}）——序列各自漂移、无长期均衡，宜对差分建模(VAR/ARIMA)")
        summary.append(
            f"{entry.method} 完成：{len(series)} 个序列 × {n} 期（diff 阶数 k={k}）。{verdict}。"
            f" ⚠ 协整要求各序列 I(1)（已查：{lvl_nonstat}/{len(series)} levels 非平稳、"
            f"{diff_stat}/{len(series)} 差分平稳{i1_note}）；Johansen 对滞后阶/确定性项设定敏感；"
            "秩检验与 VECM 均采用『无约束常数(co)』设定（statsmodels 的 Johansen 秩检验仅提供 "
            "nc/co/lo 临界值表，无约束常数外置，允许水平序列有非零均值）；"
            "johansen_trace.csv 的逐行 reject 列为逐点比较，非权威结果——权威协整秩以序贯 "
            "select_coint_rank（即上文 r）为准；长期关系是统计均衡、非结构因果。"
        )
        code += [
            "from statsmodels.tsa.vector_ar.vecm import select_coint_rank, VECM  # 协整 + VECM",
            f"# r=select_coint_rank(data, det_order=0, k_ar_diff={k}, signif=0.05).rank (序贯); "
            f"VECM(data, k_ar_diff={k}, coint_rank=r, deterministic='co').fit()",
        ]
    except Exception as err:
        summary.append(f"协整/VECM 失败：{err}")


@register("garch")
def _branch_garch(ctx: Ctx) -> None:
    # GARCH(1,1) conditional-volatility model: captures volatility clustering in a series.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("value") if cfg.get("value") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if importlib.util.find_spec("arch") is None:
        summary.append("GARCH 需要 arch 包（未检测到）。安装：pip install arch。")
        return
    if value is None:
        summary.append("GARCH 失败：需要一个连续序列（收益/波动序列）。config['value'] 可指定。")
        return
    try:
        import pandas as pd
        from arch import arch_model
        from statsmodels.stats.diagnostic import het_arch

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 50 or y.nunique() < 5:
            summary.append("GARCH 失败：观测不足（<50）或近常数序列。")
            return
        # arch fits best when data are scaled ~[1, 1000]; rescale OUT-of-band series (tiny or huge)
        # to a single multiplicative scale (target std ~10) and restore volatility after — divide-back
        # stays exact for any scale. In-band series are left as-is.
        s = float(y.std())
        scale = 1.0 if 0.1 <= s <= 1000.0 else 10.0 / s
        ys = y * scale
        try:
            arch_lm_p = float(het_arch(ys - ys.mean(), nlags=min(10, n // 5))[1])
        except Exception:
            arch_lm_p = float("nan")
        res = arch_model(ys, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
        conv_note = "；⚠ GARCH 优化器未收敛，系数不可靠" if getattr(res, "convergence_flag", 0) else ""
        a, b = float(res.params.get("alpha[1]", 0.0)), float(res.params.get("beta[1]", 0.0))
        omega = float(res.params.get("omega", 0.0))
        persistence = a + b
        cond_vol = np.asarray(res.conditional_volatility, dtype=float) / scale  # back to original scale
        uncond = float(np.sqrt(omega / (1 - persistence)) / scale) if persistence < 1 else float("nan")
        estimates.update({
            "alpha1": round(a, 4), "beta1": round(b, 4), "persistence": round(persistence, 4),
            "omega": round(omega, 6), "arch_lm_pvalue": round(arch_lm_p, 4) if arch_lm_p == arch_lm_p else float("nan"),
            "uncond_volatility": round(uncond, 6) if uncond == uncond else float("nan"),
            "aic": round(float(res.aic), 2), "n_obs": float(n),
        })
        pd.DataFrame({"period": range(n), "cond_volatility": np.round(cond_vol, 6)}).to_csv(
            d / "garch_volatility.csv", index=False, encoding="utf-8")
        files.append("garch_volatility.csv")
        (d / "garch_summary.txt").write_text(str(res.summary()), encoding="utf-8")
        files.append("garch_summary.txt")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.plot(cond_vol, color="#C44E52", label="conditional volatility σ_t")
            ax.plot(np.abs(y - y.mean()).to_numpy(), color="#bbbbbb", lw=0.6, alpha=0.7, label="|series - mean|")
            ax.set_xlabel("period index")
            ax.set_ylabel(f"volatility of {value}")
            ax.set_title("GARCH(1,1) conditional volatility")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "garch_volatility.png", dpi=150)
            plt.close(fig)
            files.append("garch_volatility.png")
        except Exception:
            pass
        no_arch = arch_lm_p == arch_lm_p and arch_lm_p > 0.05
        arch_note = "；⚠ ARCH-LM 不显著(p>0.05)：无明显波动聚集，GARCH 或非必要" if no_arch else ""
        pers_note = "；⚠ α+β≥1：波动近单位根(IGARCH)，无条件方差不存在" if persistence >= 1 else ""
        omega_note = f"（在 ×{scale:g} 缩放序列的尺度上）" if scale != 1 else ""
        scale_note = f"（拟合时已×{scale:g}，条件/无条件波动率已还原原尺度）" if scale != 1 else ""
        summary.append(
            f"{entry.method} 完成：{value} GARCH(1,1){scale_note}；"
            f"ω={omega:.4g}{omega_note}、α₁={a:.3f}、β₁={b:.3f}；波动持续性 α+β={persistence:.3f}（越近 1 越持久）；"
            f"ARCH-LM p={arch_lm_p:.3g}（检波动聚集）；AIC={res.aic:.1f}。{conv_note}{arch_note}{pers_note}"
            " ⚠ GARCH 建模条件异方差（波动聚集），假定均值方程已设定、序列(弱)平稳；α+β<1 才有有限无条件方差；"
            "正态新息默认（厚尾可换 t 分布）；ω 是缩放序列上的方差截距（α/β 无量纲、波动率已还原，ω 未还原）。"
        )
        code += [
            "from arch import arch_model  # GARCH 条件波动率",
            "# arch_model(y, mean='Constant', vol='GARCH', p=1, q=1).fit(); 持续性=α₁+β₁",
        ]
    except Exception as err:
        summary.append(f"GARCH 拟合失败：{err}")


@register("structural_breaks")
def _branch_structural_breaks(ctx: Ctx) -> None:
    # Multiple structural-break (change-point) detection in a series' MEAN level via ruptures PELT
    # (Bai-Perron-style), with a ~2xBIC penalty (penalty_mult, default 2.0) auto-selecting the breaks.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("value") if cfg.get("value") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if importlib.util.find_spec("ruptures") is None:
        summary.append("结构突变检测需要 ruptures 包（未检测到）。安装：pip install ruptures。")
        return
    if value is None:
        summary.append("结构突变检测失败：需要一个连续序列。config['value'] 可指定。")
        return
    try:
        import pandas as pd
        import ruptures as rpt

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 30 or y.nunique() < 5:
            summary.append("结构突变检测失败：观测不足（<30）或近常数序列。")
            return
        sig = y.to_numpy()
        # noise variance from first differences (immune to mean shifts -> not inflated by the breaks)
        sigma2 = float(np.var(np.diff(sig)) / 2.0) if n > 2 else float(np.var(sig))
        try:
            mult = float(cfg.get("penalty_mult", 2.0))
        except (TypeError, ValueError):
            mult = 2.0
        pen = mult * np.log(n) * max(sigma2, 1e-12)
        min_size = max(5, n // 20)
        nb = cfg.get("n_breaks")
        if isinstance(nb, int) and nb >= 1:
            bkps = rpt.Dynp(model="l2", min_size=min_size).fit(sig).predict(n_bkps=nb)
            sel = f"固定 {nb} 个断点 (Dynp)"
        else:
            bkps = rpt.Pelt(model="l2", min_size=min_size).fit(sig).predict(pen=pen)
            sel = f"PELT 自动选 (~2×BIC 惩罚 pen={pen:.3g}，penalty_mult 默认 2.0，越大越少断点)"
        breaks = [int(b) for b in bkps if b < n]  # segment boundaries (drop the trailing n)
        bounds = [0] + breaks + [n]
        seg = [{"start": bounds[i], "end": bounds[i + 1], "n": bounds[i + 1] - bounds[i],
                "mean": round(float(sig[bounds[i]:bounds[i + 1]].mean()), 4),
                "sd": round(float(sig[bounds[i]:bounds[i + 1]].std()), 4)}
               for i in range(len(bounds) - 1)]
        time_vals = None
        if fp.time_col and fp.time_col in d2.columns:
            tv = d2[fp.time_col].reset_index(drop=True)
            time_vals = [tv.iloc[b] for b in breaks if b < len(tv)]
        # trend confound: l2 detects MEAN shifts; a strong linear trend gets approximated by steps
        idx = np.arange(n)
        trend_r = float(abs(np.corrcoef(idx, sig)[0, 1])) if np.std(sig) > 0 else 0.0
        estimates.update({"n_breaks": float(len(breaks)), "n_obs": float(n),
                          "penalty": round(float(pen), 4), "trend_abs_corr": round(trend_r, 3)})
        pd.DataFrame(seg).to_csv(d / "segments.csv", index=False, encoding="utf-8")
        files.append("segments.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.plot(sig, color="#bbbbbb", lw=0.8)
            for s in seg:
                ax.hlines(s["mean"], s["start"], s["end"], color="#4C72B0", lw=2)
            for b in breaks:
                ax.axvline(b, color="#C44E52", ls="--", lw=1)
            ax.set_xlabel("period index")
            ax.set_ylabel(value)
            ax.set_title(f"Structural breaks — {len(breaks)} change point(s) in mean")
            fig.tight_layout()
            fig.savefig(d / "structural_breaks.png", dpi=150)
            plt.close(fig)
            files.append("structural_breaks.png")
        except Exception:
            pass
        shift_txt = ""
        if len(seg) >= 2:
            shifts = [abs(seg[i + 1]["mean"] - seg[i]["mean"]) for i in range(len(seg) - 1)]
            j = int(np.argmax(shifts))
            bt = (f"≈{fp.time_col}={time_vals[j]}" if time_vals and j < len(time_vals) else f"index {breaks[j]}")
            shift_txt = f"最大均值跳变在断点 #{j + 1}（{bt}）：{seg[j]['mean']}→{seg[j + 1]['mean']}；"
        trend_note = ("；⚠ 序列有强线性趋势（|r|=%.2f）——均值突变检测可能在用台阶逼近趋势，"
                      "建议先去趋势/差分再检测" % trend_r) if trend_r > 0.7 else ""
        loc_txt = "、".join(str(b) for b in breaks) if breaks else "无"
        summary.append(
            f"{entry.method} 完成：{value}（n={n}）检出 {len(breaks)} 个结构突变点（{sel}）；"
            f"断点位置(index)：{loc_txt}；{shift_txt}段均值见 segments.csv 与图。{trend_note}"
            " ⚠ 检测的是均值水平突变（非斜率/方差突变）；惩罚越大断点越少"
            "（config penalty_mult 调，或 n_breaks 固定个数）；突变点是数据驱动的探索性结果，"
            "需结合事件/政策时点佐证、非因果。"
        )
        code += [
            "import ruptures as rpt  # 结构突变(变点)检测",
            f"# rpt.Pelt(model='l2', min_size={min_size}).fit(y).predict(pen={pen:.3g})  # 段均值/断点",
        ]
    except Exception as err:
        summary.append(f"结构突变检测失败：{err}")


@register("stl_decomposition")
def _branch_stl_decomposition(ctx: Ctx) -> None:
    # STL (Seasonal-Trend decomposition via Loess): split a series into trend + seasonal + residual,
    # with Hyndman seasonal/trend strength measures. Descriptive (not a forecast/test).
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    _excl = {fp.unit_col, fp.time_col}
    value = cfg.get("value") if cfg.get("value") in df.columns else next(
        (c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl), None)
    if value is None:
        summary.append("STL 分解失败：需要一个连续序列。config['value'] 可指定。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.seasonal import STL

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        y = d2[value].astype(float).dropna().reset_index(drop=True)
        n = len(y)
        if n < 20 or y.nunique() < 5:
            summary.append("STL 分解失败：观测不足（<20）或近常数序列。")
            return
        period, auto = None, False
        try:
            cp = int(cfg["period"]) if cfg.get("period") is not None else None
            if cp and 2 <= cp <= n // 2:
                period = cp
        except (TypeError, ValueError):
            period = None
        if period is None:
            period, auto = _periodogram_period(y.to_numpy(), n), True
        if period is None:
            summary.append("STL 分解失败：未检出明显季节周期，请用 config['period'] 指定"
                           "（如月度=12、季度=4、周=7）。")
            return
        res = STL(y.to_numpy(), period=period, robust=True).fit()
        tr, se, rs = np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid)
        Fs = max(0.0, 1 - np.var(rs) / np.var(se + rs)) if np.var(se + rs) > 0 else 0.0
        Ft = max(0.0, 1 - np.var(rs) / np.var(tr + rs)) if np.var(tr + rs) > 0 else 0.0
        estimates.update({"period": float(period), "seasonal_strength": round(float(Fs), 3),
                          "trend_strength": round(float(Ft), 3), "n_obs": float(n)})
        pd.DataFrame({"index": range(n), "observed": np.round(y.to_numpy(), 4),
                      "trend": np.round(tr, 4), "seasonal": np.round(se, 4), "resid": np.round(rs, 4)}
                     ).to_csv(d / "stl_components.csv", index=False, encoding="utf-8")
        files.append("stl_components.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
            for ax, dat, lab, col in zip(
                axes, [y.to_numpy(), tr, se, rs],
                ["observed", "trend", "seasonal", "resid"],
                ["#333333", "#4C72B0", "#55A868", "#bbbbbb"],
            ):
                ax.plot(dat, color=col, lw=1)
                ax.set_ylabel(lab, fontsize=9)
            axes[-1].set_xlabel("period index")
            axes[0].set_title(f"STL decomposition — {value} (period={period})")
            fig.tight_layout()
            fig.savefig(d / "stl_decomposition.png", dpi=150)
            plt.close(fig)
            files.append("stl_decomposition.png")
        except Exception:
            pass
        seas_word = "强" if Fs >= 0.6 else ("中等" if Fs >= 0.3 else "弱")
        trend_word = "强" if Ft >= 0.6 else ("中等" if Ft >= 0.3 else "弱")
        weak_note = "；⚠ 季节强度弱(Fs<0.3)：该周期下季节性不明显，确认 period 是否合适" if Fs < 0.3 else ""
        src = (f"周期图自动检出={period}（建议人工确认）" if auto else f"config 指定={period}")
        summary.append(
            f"{entry.method} 完成：{value}（n={n}）STL 分解（{src}）；"
            f"季节强度 Fs={Fs:.3f}（{seas_word}）、趋势强度 Ft={Ft:.3f}（{trend_word}）；"
            f"分量见 stl_components.csv 与四联图。{weak_note}"
            " ⚠ STL 是描述性分解（趋势+季节+余项），非预测/检验；周期需正确"
            "（自动检出基于周期图主峰，可 config['period'] 覆盖）；robust=True 降异常值影响。"
        )
        code += [
            "from statsmodels.tsa.seasonal import STL  # STL 季节-趋势分解",
            f"# STL(y, period={period}, robust=True).fit(); 季节强度=1-Var(resid)/Var(seasonal+resid)",
        ]
    except Exception as err:
        summary.append(f"STL 分解失败：{err}")


@register("ardl_bounds")
def _branch_ardl_bounds(ctx: Ctx) -> None:
    # ARDL bounds test (Pesaran-Shin-Smith) for a long-run relationship valid under a mix of
    # I(0)/I(1) regressors, plus the error-correction speed and long-run coefficients.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    _excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in _excl]
    # H4: bind the DETECTED dependent variable (config > high-conf role > first
    # non-treatment candidate) rather than raw cont[0].
    outcome = resolve_outcome(fp, cfg, cont) if cont else None
    forced = [c for c in (cfg.get("predictors") or cfg.get("regressors") or [])
              if c in df.columns and c != outcome and c not in _excl]
    regs = (forced if forced else [c for c in cont if c != outcome])[:5]
    if outcome is None or not regs:
        summary.append("ARDL 边界检验失败：需要 1 个连续结果 + ≥1 个连续回归变量"
                       "（config outcome/predictors 可指定）。")
        return
    try:
        import pandas as pd
        from statsmodels.tsa.ardl import UECM, ardl_select_order

        d2 = df.sort_values(fp.time_col) if (fp.time_col and fp.time_col in df.columns) else df
        data = d2[[outcome, *regs]].dropna().reset_index(drop=True).astype(float)
        n = len(data)
        if n < 30:
            summary.append("ARDL 边界检验失败：观测不足（<30）。")
            return
        maxlag = max(1, min(4, n // 20))
        sel = ardl_select_order(data[outcome], maxlag=maxlag, exog=data[regs],
                                maxorder=maxlag, ic="aic", trend="c")
        fellback = False
        try:
            ur = UECM.from_ardl(sel.model).fit()
            used_order = sel.model.ardl_order
        except Exception:
            # AIC can drop the exog (0 lags) when there is no relationship -> from_ardl fails;
            # fall back to a forced order-1 ARDL so the bounds test still has the level (x.L1) term.
            fellback = True
            p = max(1, int(sel.model.ardl_order[0]) if (sel.model.ardl_order and sel.model.ardl_order[0]) else 1)
            ur = UECM(data[outcome], lags=p, exog=data[regs], order=1, trend="c").fit()
            used_order = (p,) + (1,) * len(regs)
        # I(2) screen (mirrors cointegration_vecm): ADF on first differences; if a differenced series
        # is still non-stationary it may be I(2), which invalidates the bounds test.
        i2_flag = 0
        try:
            from statsmodels.tsa.stattools import adfuller

            for c_ in [outcome, *regs]:
                if adfuller(data[c_].diff().dropna().to_numpy(), autolag="AIC")[1] > 0.05:
                    i2_flag += 1
        except Exception:
            i2_flag = -1
        bt = ur.bounds_test(case=3)
        F = float(bt.stat)
        lo95 = float(bt.crit_vals.loc[95.0, "lower"])
        up95 = float(bt.crit_vals.loc[95.0, "upper"])
        if F > up95:
            concl = "存在长期(协整)关系（F>I(1)上界）"
        elif F < lo95:
            concl = "无长期关系（F<I(0)下界）"
        else:
            concl = "不确定（F 落在 I(0)/I(1) 界之间）"
        ec = float(ur.params.get(f"{outcome}.L1", float("nan")))  # error-correction speed
        lr = {}
        for r_ in regs:
            key = f"{r_}.L1"
            if key in ur.params.index and ec == ec and abs(ec) > 1e-9:
                lr[r_] = round(-float(ur.params[key]) / ec, 4)
        estimates.update({
            "bounds_F": round(F, 3), "crit_lower_95": round(lo95, 3), "crit_upper_95": round(up95, 3),
            "speed_of_adjustment": round(ec, 4) if ec == ec else float("nan"),
            "ardl_p": float(used_order[0]) if used_order else 1.0,
            "maybe_i2": float(i2_flag), "n_obs": float(n),
        })
        for r_, v in lr.items():
            estimates[f"longrun_{r_}"] = v
        (d / "ardl_uecm_summary.txt").write_text(str(ur.summary()), encoding="utf-8")
        files.append("ardl_uecm_summary.txt")
        pd.DataFrame([{"regressor": r_, "longrun_coef": v} for r_, v in lr.items()]).to_csv(
            d / "ardl_longrun.csv", index=False, encoding="utf-8")
        files.append("ardl_longrun.csv")
        lr_txt = "；".join(f"{r_}={v}" for r_, v in lr.items()) or "—"
        ec_note = ("（负且回拉=支持长期关系）" if (ec == ec and ec < 0)
                   else "（⚠ EC 项非负，长期关系存疑）")
        fb_note = "；注：AIC 删除外生项，已强制 order-1 ARDL 以做边界检验" if fellback else ""
        i2_note = f"；⚠ {i2_flag} 个序列差分后仍非平稳(疑似 I(2))，边界检验或失效" if i2_flag > 0 else ""
        summary.append(
            f"{entry.method} 完成：{outcome} ~ {len(regs)} 个回归变量（ARDL{used_order}，"
            f"trend=c，n={n}）；边界检验 F={F:.3f}（95% 界 [{lo95:.2f}, {up95:.2f}]）→ {concl}；"
            f"误差修正速度 EC={ec:.3f}{ec_note}；长期系数 {lr_txt}。{fb_note}{i2_note}"
            " ⚠ ARDL 边界检验适用 I(0)/I(1) 混合（任一变量 I(2) 则失效）；case=3（不受限常数）；"
            "对滞后阶/确定性项设定敏感；长期关系是统计均衡、非结构因果。"
        )
        code += [
            "from statsmodels.tsa.ardl import ardl_select_order, UECM  # ARDL 边界检验 + ECM",
            "# UECM.from_ardl(ardl_select_order(y, exog=X, ic='aic', trend='c').model).fit().bounds_test(case=3)",
        ]
    except Exception as err:
        summary.append(f"ARDL 边界检验失败：{err}")

