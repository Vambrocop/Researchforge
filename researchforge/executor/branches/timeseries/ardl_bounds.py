"""ARDL bounds test for a long-run relationship.

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome


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
