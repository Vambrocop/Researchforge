"""Branch handlers for IV/dynamic-panel completion (family econometrics).

Two estimators that finish the dynamic / instrumental-variable panel toolkit,
both delegating to R's gold-standard ``plm`` package with graceful degrade (R is
optional — if R or plm is missing we fall back to an honest Chinese skip pointing
at a runnable alternative, never crash, never fabricate):

* ``system_gmm``      — Blundell-Bond SYSTEM GMM dynamic panel: the level+difference
                        extension of the Arellano-Bond difference GMM
                        (``dynamic_panel_gmm``). Adds level-equation moment
                        conditions (levels instrumented by lagged differences),
                        which is far better behaved when the series is PERSISTENT
                        and the difference-GMM instruments are weak.
* ``hausman_taylor``  — Hausman-Taylor (1981) panel IV: estimates coefficients on
                        TIME-INVARIANT regressors (which fixed effects sweep away)
                        while letting some regressors correlate with the unit effect,
                        using the within-means of the exogenous time-varying
                        regressors as instruments.

Engine conventions (see CLAUDE.md「引擎约定」): ``@register("<id>") def _branch_<id>(ctx)``;
unpack ctx, MUTATE summary/estimates/files/code. estimates are plain floats (CIs/SEs
as separate scalar keys). Column names are identifier-guarded before entering an R
formula string (anti-injection / parse safety); the temp CSV is written to the output
dir and removed in ``finally``. R scripts are audited, fixed strings — NEVER fetched at
runtime. matplotlib labels are English (CJK would tofu-box); summaries are Chinese with
⚠ disclosures.
"""

from __future__ import annotations

import re

from researchforge.executor._branch_api import Ctx, register
# Reuse the canonical GMM-lag-window parser (shared with dynamic_panel_gmm) rather than
# duplicate it — keeps the (lo, hi) default + validation in one place.
from researchforge.executor.run import _gmm_lags

_IDENT = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")


def _ident_ok(*cols) -> bool:
    return all(re.fullmatch(_IDENT, str(c)) for c in cols if c is not None)


def _resolve_unit_time(ctx: Ctx):
    """(unit, time) honoring config overrides (the documented contract) and falling
    back to the profiler's detected panel columns. Either may be None."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    unit = cfg.get("unit") if cfg.get("unit") in df.columns else fp.unit_col
    time = cfg.get("time") if cfg.get("time") in df.columns else fp.time_col
    return unit, time


def _resolve_panel_roles(ctx: Ctx, *, max_preds: int, unit, time):
    """Resolve (y, predictors) for a panel model by the econometrics convention:
    y = first continuous (config outcome); predictors = remaining continuous/binary
    (config predictors), capped. Returns (y, preds) — either may be None/[]."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    excl = {unit, time}
    forced_y = cfg.get("outcome")
    if forced_y in df.columns:
        y = forced_y
    else:
        y = next((c.name for c in fp.columns
                  if c.kind == "continuous" and c.name not in excl), None)
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y]
    if forced:
        preds = forced[:max_preds]
    else:
        preds = [c.name for c in fp.columns
                 if c.kind in {"continuous", "binary"} and c.name not in {y, *excl}][:max_preds]
    return y, preds


# ─────────────────────────────────────────────────────────────────────────────
# 1) system_gmm — Blundell-Bond system GMM (R plm::pgmm, transformation="ld")
# ─────────────────────────────────────────────────────────────────────────────
def _system_gmm_via_r(csv_path, unit, time, y, predictors, endogenous, gmm_lags):
    """Blundell-Bond system GMM via R plm::pgmm(transformation="ld"). Returns
    (coef DataFrame[term,estimate,std_err,p_value], diag dict). Raises on failure
    so the caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    endo = [p for p in (endogenous or []) if p in predictors]
    exog = [p for p in predictors if p not in endo]
    lo, hi = int(gmm_lags[0]), int(gmm_lags[1])
    y_lo = max(2, lo)  # lagged dependent: lag>=2 is the only valid instrument in differences
    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join([f"lag({y}, 1)", *predictors])
    gmm_inst = " + ".join([f"lag({y}, {y_lo}:{hi})", *(f"lag({p}, {lo}:{hi})" for p in endo)])
    if endo:
        formula = f"{y} ~ {rhs} | {gmm_inst}" + (f" | {' + '.join(exog)}" if exog else "")
    else:
        formula = f"{y} ~ {rhs} | {gmm_inst}"
    rcode = (
        "suppressMessages(library(plm))\n"
        f'd <- read.csv("{csv_r}")\n'
        f'pd <- pdata.frame(d, index=c("{unit}","{time}"))\n'
        # transformation="ld" => SYSTEM GMM (levels + differences); "twosteps" + robust SE
        f'm <- pgmm({formula}, data=pd, effect="individual", model="twosteps", transformation="ld")\n'
        "s <- summary(m, robust=TRUE); ct <- s$coefficients\n"
        'cat("##COEF\\n")\n'
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,4]))\n'
        "a1 <- mtest(m, order=1); a2 <- mtest(m, order=2)\n"
        'cat("##DIAG\\n")\n'
        'cat(sprintf("sargan_stat|%.6f\\nsargan_p|%.6f\\n", s$sargan$statistic, s$sargan$p.value))\n'
        'cat(sprintf("ar1_p|%.6f\\nar2_p|%.6f\\n", a1$p.value, a2$p.value))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, crows, diag = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##DIAG":
            section = "D"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            try:
                diag[k] = float(v)
            except ValueError:
                pass
    if not crows:
        raise RuntimeError("pgmm(system) 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return coef, diag


@register("system_gmm")
def _branch_system_gmm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    from researchforge.executor import rbridge

    unit, time = _resolve_unit_time(ctx)
    if not (unit and time):
        summary.append("系统 GMM 失败：需要面板数据（单位列 + 时间列）。")
        return
    y, preds = _resolve_panel_roles(ctx, max_preds=5, unit=unit, time=time)
    if y is None or not preds:
        summary.append("系统 GMM 失败：需要连续结果变量 + ≥1 个预测变量。")
        return
    n_periods = int(df[time].nunique())
    if n_periods < 3:
        summary.append("系统 GMM 失败：需要 ≥3 个时间期（差分方程 + AR 检验要求）。")
        return
    if not (rbridge.r_available() and rbridge.r_package_available("plm")):
        summary.append(
            "系统 GMM 需要 R 的 plm 包（未检测到）。安装：install.packages('plm')；"
            "或用 dynamic_panel_gmm（差分 GMM）/ panel_fixed_effects / random_effects。"
        )
        return
    if not _ident_ok(y, *preds, unit, time):
        summary.append("系统 GMM 失败：列名需为标识符式（字母/数字/. _）。")
        return

    endo = [p for p in (cfg.get("endogenous") or []) if p in preds]
    lo, hi = _gmm_lags(cfg)
    sub = df[[unit, time, y, *preds]].dropna()
    csv = d / "_sysgmm_input.csv"
    sub.to_csv(csv, index=False)
    try:
        coef, diag = _system_gmm_via_r(csv, unit, time, y, preds,
                                       endogenous=endo, gmm_lags=(lo, hi))
        coef["term"] = [f"lag_{y}" if str(t).startswith("lag(") else str(t) for t in coef["term"]]
        coef.to_csv(d / "system_gmm_coefficients.csv", index=False, encoding="utf-8")
        files.append("system_gmm_coefficients.csv")

        sargan_p = diag.get("sargan_p", float("nan"))
        ar1_p = diag.get("ar1_p", float("nan"))
        ar2_p = diag.get("ar2_p", float("nan"))
        sargan_ok = sargan_p > 0.05
        ar2_ok = ar2_p > 0.05
        lag_rows = coef[coef["term"].str.startswith("lag_")]
        persistence = float(lag_rows.iloc[0]["estimate"]) if len(lag_rows) else float("nan")

        estimates["persistence_lag_coef"] = round(persistence, 4)
        estimates["sargan_p"] = round(sargan_p, 4)
        estimates["ar1_p"] = round(ar1_p, 4)
        estimates["ar2_p"] = round(ar2_p, 4)
        for _, r in coef.iterrows():
            if not str(r["term"]).startswith("lag_"):
                estimates[str(r["term"])] = round(float(r["estimate"]), 4)

        _endo_note = (f"内生变量（用 lag {lo}:{hi} 工具）：{endo}" if endo
                      else "全部协变量设为严格外生（可用 config endogenous 标出内生变量）")
        if lo < 2:
            _endo_note += (
                "；⚠ gmm_lags 起始<2 仅对前定(predetermined)变量有效，"
                "滞后被解释变量已强制 lag≥2（差分方程中 lag1 为无效工具），但协变量块仍用所设 lo"
            )
        (d / "system_gmm_diagnostics.txt").write_text(
            "Blundell-Bond 系统 GMM (twosteps, transformation='ld', Windmeijer 稳健 SE)\n"
            f"工具集限 lag {lo}-{hi}（抑制工具过度增殖；过多工具会弱化 Sargan/Hansen）。\n"
            f"{_endo_note}。滞后被解释变量在差分方程中强制 lag≥2。\n"
            f"Sargan 过度识别 p = {sargan_p:.4g}（注：s$sargan 为非稳健 Sargan,非 Hansen J）"
            f"（{'工具有效（不拒）' if sargan_ok else '被拒 → 工具集可疑'}）\n"
            f"AR(1) p = {ar1_p:.4g}（差分后通常显著，正常）\n"
            f"AR(2) p = {ar2_p:.4g}"
            f"（{'无二阶自相关 → GMM 一致' if ar2_ok else '⚠ 有二阶自相关 → GMM 不一致'}）\n\n"
            + coef.to_string(index=False),
            encoding="utf-8",
        )
        files.append("system_gmm_diagnostics.txt")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.errorbar(coef["estimate"], range(len(coef)),
                        xerr=1.96 * coef["std_err"], fmt="o")
            ax.axvline(0, color="grey", ls="--")
            ax.set_yticks(range(len(coef)))
            ax.set_yticklabels(coef["term"])
            ax.set_xlabel("System-GMM coefficient (95% CI)")
            ax.set_title("Blundell-Bond system GMM")
            fig.tight_layout()
            fig.savefig(d / "system_gmm_coefficients.png", dpi=150)
            plt.close(fig)
            files.append("system_gmm_coefficients.png")
        except Exception:
            pass

        valid = sargan_ok and ar2_ok
        summary.append(
            f"{entry.method} 完成（R/plm，系统 GMM，{n_periods} 期）：滞后被解释变量系数（持续性）"
            f"={persistence:.3f}；Sargan p={sargan_p:.3g}"
            f"（{'工具有效' if sargan_ok else '工具可疑'}），AR(2) p={ar2_p:.3g}"
            f"（{'无二阶自相关' if ar2_ok else '⚠有二阶自相关、GMM不一致'}）"
            f"{'' if valid else ' —— ⚠ 诊断未全通过,结果存疑'}。系数见 system_gmm_coefficients.csv。"
            " ⚠ 系统 GMM 在差分矩条件外追加**水平方程矩条件**(以滞后差分作工具)，"
            "比差分 GMM 在**高持续性**序列上更有效、偏误更小；代价是依赖额外的**初始条件/均值平稳**假定，"
            "且工具更多——务必看 Sargan/Hansen 是否被过多工具弱化(Roodman)。可与 dynamic_panel_gmm(差分GMM)对照。"
            + (f"（内生变量 {endo}、工具滞后 {lo}:{hi} 按 config）" if (endo or cfg.get("gmm_lags")) else "")
        )
        code += [
            "library(plm)  # Blundell-Bond 系统 GMM",
            f"# pgmm({y} ~ lag({y},1)+... | lag({y},{lo}:{hi})[+lag(endo)], model='twosteps', transformation='ld')",
        ]
    except Exception as err:
        summary.append(f"系统 GMM 失败：{err}")
    finally:
        try:
            csv.unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 2) hausman_taylor — Hausman-Taylor panel IV (R plm model="ht")
# ─────────────────────────────────────────────────────────────────────────────
def _hausman_taylor_via_r(csv_path, unit, time, y, regressors, instruments):
    """Hausman-Taylor (1981) via R plm(model="ht", inst.method="baltagi"). The
    2-part formula ``y ~ <regressors> | <instruments>`` lists as instruments the
    regressors assumed exogenous w.r.t. the unit effect (endogenous ones excluded;
    plm uses the within-means of the exogenous time-varying regressors to identify
    the time-invariant coefficients). Returns coef DataFrame[term,estimate,std_err,
    p_value]. Raises on failure."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    rhs = " + ".join(regressors)
    inst = " + ".join(instruments) if instruments else rhs
    formula = f"{y} ~ {rhs} | {inst}"
    rcode = (
        "suppressMessages(library(plm))\n"
        f'd <- read.csv("{csv_r}")\n'
        f'pd <- pdata.frame(d, index=c("{unit}","{time}"))\n'
        f'm <- plm({formula}, data=pd, model="ht", inst.method="baltagi")\n'
        "s <- summary(m); ct <- s$coefficients\n"
        'cat("##COEF\\n")\n'
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,4]))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, crows = None, []
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))
    if not crows:
        raise RuntimeError("plm(ht) 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return coef


def _time_invariant(df, unit, cols):
    """Names among ``cols`` that are constant within EVERY unit (time-invariant)."""
    import numpy as np

    ti = []
    g = df.groupby(unit)
    for c in cols:
        # max within-unit std ~ 0 => the column never changes within a unit
        stds = g[c].std(ddof=0)
        if float(np.nanmax(stds.to_numpy(dtype=float))) < 1e-9:
            ti.append(c)
    return ti


@register("hausman_taylor")
def _branch_hausman_taylor(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import pandas as pd

    from researchforge.executor import rbridge

    unit, time = _resolve_unit_time(ctx)
    if not (unit and time):
        summary.append("Hausman-Taylor 失败：需要面板数据（单位列 + 时间列）。")
        return
    y, preds = _resolve_panel_roles(ctx, max_preds=8, unit=unit, time=time)
    if y is None or not preds:
        summary.append("Hausman-Taylor 失败：需要连续结果变量 + ≥1 个预测变量。")
        return

    sub = df[[unit, time, y, *preds]].dropna()
    # numeric-coerce predictors for the time-invariance test
    for c in preds:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    ti = _time_invariant(sub, unit, preds)
    tv = [c for c in preds if c not in ti]
    if not ti:
        summary.append(
            "Hausman-Taylor 跳过：未发现**时不变**回归元（各列在每个单位内都随时间变化）。"
            "HT 的价值在于估计时不变变量的系数——无时不变变量时请用 random_effects / panel_fixed_effects。"
        )
        return
    if not (rbridge.r_available() and rbridge.r_package_available("plm")):
        summary.append(
            "Hausman-Taylor 需要 R 的 plm 包（未检测到）。安装：install.packages('plm')；"
            "或用 random_effects / mundlak（相关随机效应）。"
        )
        return
    if not _ident_ok(y, *preds, unit, time):
        summary.append("Hausman-Taylor 失败：列名需为标识符式（字母/数字/. _）。")
        return

    # endogenous = regressors correlated with the unit effect (config); the rest are
    # exogenous and serve as instruments (their within-means identify TI coefficients).
    endo = [c for c in (cfg.get("endogenous") or []) if c in preds]
    instruments = [c for c in preds if c not in endo]

    # Order condition: HT identifies the time-invariant coefficients using the within-
    # means of the EXOGENOUS time-varying regressors as instruments, so it needs at
    # least as many exogenous TV regressors as endogenous TI regressors. If the user
    # over-flags endogeneity, the model is under-identified — skip with an honest stat
    # message rather than let plm error out raw.
    exog_tv = [c for c in tv if c not in endo]
    endo_ti = [c for c in ti if c in endo]
    if len(exog_tv) < len(endo_ti):
        summary.append(
            "Hausman-Taylor 跳过：模型欠识别——外生时变回归元（作工具，"
            f"{len(exog_tv)} 个）少于内生时不变回归元（{len(endo_ti)} 个），"
            "无法识别时不变系数。请少标几个 endogenous，或改用 random_effects / fixed_effects。"
        )
        return

    csv = d / "_ht_input.csv"
    sub.to_csv(csv, index=False)
    try:
        coef = _hausman_taylor_via_r(csv, unit, time, y, preds, instruments)
        # tag time-invariant terms (HT's payoff) in the table
        coef["time_invariant"] = [str(t) in set(ti) for t in coef["term"]]
        coef.to_csv(d / "hausman_taylor_coefficients.csv", index=False, encoding="utf-8")
        files.append("hausman_taylor_coefficients.csv")

        for _, r in coef.iterrows():
            term = str(r["term"])
            if term in ("(Intercept)", "(intercept)"):
                continue
            estimates[term] = round(float(r["estimate"]), 4)
            estimates[f"{term}_se"] = round(float(r["std_err"]), 4)
            estimates[f"{term}_p"] = round(float(r["p_value"]), 4)
        estimates["n_time_invariant"] = float(len(ti))
        estimates["n_time_varying"] = float(len(tv))
        estimates["n_endogenous"] = float(len(endo))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot = coef[coef["term"].str.lower() != "(intercept)"].reset_index(drop=True)
            colors = ["#C44E52" if ti_flag else "#4C72B0" for ti_flag in plot["time_invariant"]]
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            ax.errorbar(plot["estimate"], range(len(plot)),
                        xerr=1.96 * plot["std_err"], fmt="o", ecolor="grey", linestyle="none")
            for i, col in enumerate(colors):
                ax.plot(plot["estimate"].iloc[i], i, "o", color=col)
            ax.axvline(0, color="grey", ls="--")
            ax.set_yticks(range(len(plot)))
            ax.set_yticklabels(plot["term"])
            ax.set_xlabel("HT coefficient (95% CI; red = time-invariant)")
            ax.set_title("Hausman-Taylor estimator")
            fig.tight_layout()
            fig.savefig(d / "hausman_taylor_coefficients.png", dpi=150)
            plt.close(fig)
            files.append("hausman_taylor_coefficients.png")
        except Exception:
            pass

        ti_txt = "、".join(ti)
        endo_txt = ("、".join(endo) if endo else "（无——默认全部外生，此时 HT≈随机效应）")
        summary.append(
            f"{entry.method} 完成（R/plm，inst.method='baltagi'）：结果={y}；"
            f"时不变回归元={ti_txt}（{len(ti)} 个，**HT 的核心价值**——固定效应会把它们消去、HT 仍能估计），"
            f"时变回归元 {len(tv)} 个；标记为内生（与单位效应相关）的变量：{endo_txt}。"
            "系数（含时不变项）见 hausman_taylor_coefficients.csv。"
            " ⚠ HT 用**外生时变回归元的组内均值**作工具来识别时不变系数——其有效性取决于这些工具与单位效应无关；"
            "若把某个真正内生的时变变量误标为外生，会使时不变系数有偏。"
            " ⚠ 未声明任何 endogenous 时，HT 退化为随机效应(RE)；HT 的增量价值来自把部分时变变量标为内生(config endogenous)，"
            "从而对时不变变量做 IV 识别。可与 hausman_test / mundlak 搭配判断 FE-vs-RE 与内生性。"
        )
        code += [
            "library(plm)  # Hausman-Taylor",
            f"# plm({y} ~ {' + '.join(preds)} | {' + '.join(instruments)}, model='ht', inst.method='baltagi')",
        ]
    except Exception as err:
        summary.append(f"Hausman-Taylor 失败：{err}")
    finally:
        try:
            csv.unlink()
        except OSError:
            pass
