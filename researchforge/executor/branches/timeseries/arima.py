"""ARIMA / SARIMA with automatic order selection and forecast intervals.

One analysis, one module (see branches/__init__.py — walk_packages auto-registers it).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.timeseries._order_search import (
    _GRID_FIT_BUDGET,
    _auto_order,
    _fit_sarimax,
)


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
            # Cold review B/A5.3: this block used to emit
            # `enforce_stationarity=False, enforce_invertibility=False` and no `trend`,
            # i.e. the exported "reproduce this" code reproduced the TWO BUGS the previous
            # cold review had just fixed in _fit_sarimax (an incomparable likelihood across
            # the grid, and a mean-zero fit that forecast 0.0 for a series around 500).
            # The exported settings must BE the fitted settings — a test now runs this code
            # and compares its forecast with the branch's own.
            _ex_trend = "c" if (order[1] == 0 and sorder[1] == 0) else None
            code += [
                "from statsmodels.tsa.statespace.sarimax import SARIMAX",
                f"y = df.sort_values('{time_col}')['{value_col}'].astype(float).reset_index(drop=True)",
                "# enforce_stationarity/invertibility=True 与 d==0 时的常数项都是 load-bearing:",
                "# 关掉前者会让不同阶数的对数似然算在不同有效样本上(AICc 失效);",
                "# 缺后者会把未差分序列当成零均值过程(均值 500 的序列预测成 0)。",
                f"model = SARIMAX(y, order={order}, seasonal_order={sorder},"
                f" trend={_ex_trend!r},",
                "                enforce_stationarity=True,"
                " enforce_invertibility=True).fit(disp=False)",
                "print(model.summary())",
                f"fc = model.get_forecast(steps={steps})",
                f"mean, ci = fc.predicted_mean, fc.conf_int(alpha={alpha:.3g})  # 点预测 + 预测区间",
            ]
        except Exception as err:
            summary.append(f"ARIMA 拟合失败：{err}")
