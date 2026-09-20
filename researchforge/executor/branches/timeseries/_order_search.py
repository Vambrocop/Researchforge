"""SARIMA automatic order selection — the machinery behind @register("arima").

Lives beside its only caller (the layout convention: when a family is promoted to a
package, a method's helpers move in with the method). The cost model and its budget are
the part with the most measurement behind them — see _sarimax_k_states.
"""

from __future__ import annotations

# numpy / time are imported INSIDE the functions that need them, matching the rest of
# the branches package: module load must stay cheap (branches/__init__ imports them all).

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
