"""Branch handlers for the ACTUARIAL / DEMOGRAPHY family.

Three workhorse methods from actuarial science and demography:

  - life_table        — period life table from age-specific mortality (q_x / l_x /
                        L_x / T_x / e_x, life expectancy e_0). Demographic, deterministic.
  - chain_ladder      — claims-reserving on a run-off triangle: volume-weighted
                        age-to-age development factors, projected ultimates, IBNR
                        reserve (+ optional Mack standard error). Real estimation.
  - loss_distribution — severity fit (lognormal / gamma / pareto / weibull) by MLE,
                        best by AIC, with VaR and TVaR/CVaR risk measures. Real MLE.

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法> 跳过:<原因>" appended to summary + return
    (never crash/fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    float ``estimates`` dict (plain floats; nan for N/A; never tuples/strings),
    Chinese ``summary`` with ⚠ assumption / bias disclosures.
  * The profiler may classify an integer AGE column or ORIGIN-period column of distinct
    integers as ``id`` (the「id 陷阱」) or as ``count`` — so age / origin resolution
    accepts continuous / count / id kinds.

Pure Python (numpy / pandas / scipy.stats / scipy.integrate; matplotlib optional).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _numeric_cols(ctx: Ctx):
    """Column names usable as numeric series (continuous / count / id).

    Accepts ``id`` and ``count`` kinds because an integer age column or an integer
    origin-period column with all-distinct values is misclassified as ``id`` by the
    profiler (CLAUDE.md「id 陷阱」), and small-integer streams profile as ``count``.
    """
    out = []
    for c in ctx.fp.columns:
        if c.kind in ("continuous", "count", "id"):
            out.append(c.name)
    return out


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
# 1) life_table — period life table from age-specific mortality
#    Refs: Preston, Heuveline & Guillot "Demography: Measuring and Modeling
#    Population Processes"; standard period life-table construction.
#
#    Given age-specific mortality m_x (or deaths D_x + exposure E_x -> m_x = D_x/E_x)
#    on age intervals of width n_x (from the age column spacing, default 1):
#      q_x = n*m_x / (1 + (n - a_x)*m_x)      probability of death in [x, x+n)
#      p_x = 1 - q_x                          probability of survival
#      l_0 = radix (default 100000)           survivors at exact age x
#      l_{x+1} = l_x * p_x
#      d_x = l_x - l_{x+1} = l_x * q_x        deaths in the interval
#      L_x = n*l_{x+1} + a_x*d_x              person-years lived in [x, x+n)
#      T_x = sum_{y>=x} L_y                   person-years lived above age x
#      e_x = T_x / l_x                        life expectancy at exact age x
#    Last (open) interval is closed with q_x=1, L_x = l_x / m_x (if m_x>0).
#    a_x = 0.5 (mid-period) by default; DISCLOSED.
# ===========================================================================
@register("life_table")
def _branch_life_table(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    nums = _numeric_cols(ctx)

    # ---- resolve the age column -------------------------------------------
    age_name = cfg.get("age")
    if age_name not in df.columns:
        # auto: a column literally named like an age, else the first numeric column.
        cand = [c for c in df.columns if str(c).strip().lower() in
                ("age", "age_group", "agegroup", "x", "age_x")]
        age_name = cand[0] if cand else (nums[0] if nums else None)
    if age_name is None or age_name not in df.columns:
        summary.append(
            "生命表 跳过：需要一个年龄列（config['age']）以及死亡率 m_x/q_x 或 死亡数+暴露。"
        )
        return

    try:
        age = pd.to_numeric(df[age_name], errors="coerce").to_numpy(dtype=float)
    except Exception:
        summary.append("生命表 跳过：年龄列无法解析为数值。")
        return

    # ---- resolve the mortality input: q_x, or m_x, or deaths/exposure -----
    qx_name = cfg.get("qx")
    rate_name = cfg.get("rate")          # m_x (central death rate)
    deaths_name = cfg.get("deaths")
    expo_name = cfg.get("exposure")

    def _num(name):
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    mx = None
    qx_in = None
    src_note = ""
    if qx_name in df.columns:
        qx_in = _num(qx_name)
        src_note = f"由 q_x 列 {qx_name} 给定"
    elif rate_name in df.columns:
        mx = _num(rate_name)
        src_note = f"由 m_x 列 {rate_name} 给定"
    elif deaths_name in df.columns and expo_name in df.columns:
        deaths = _num(deaths_name)
        expo = _num(expo_name)
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = np.where(expo > 0, deaths / expo, np.nan)
        src_note = f"m_x = {deaths_name}/{expo_name}（死亡数/暴露）"
    else:
        # auto: look for conventionally-named columns
        lc = {str(c).strip().lower(): c for c in df.columns}
        if "qx" in lc or "q_x" in lc:
            col = lc.get("qx") or lc.get("q_x")
            qx_in = _num(col)
            src_note = f"由 q_x 列 {col} 自动识别"
        elif "mx" in lc or "m_x" in lc:
            col = lc.get("mx") or lc.get("m_x")
            mx = _num(col)
            src_note = f"由 m_x 列 {col} 自动识别"
        elif ("deaths" in lc or "dx" in lc) and ("exposure" in lc or "pop" in lc or "population" in lc):
            dcol = lc.get("deaths") or lc.get("dx")
            ecol = lc.get("exposure") or lc.get("pop") or lc.get("population")
            deaths = _num(dcol)
            expo = _num(ecol)
            with np.errstate(divide="ignore", invalid="ignore"):
                mx = np.where(expo > 0, deaths / expo, np.nan)
            src_note = f"m_x = {dcol}/{ecol}（自动识别）"
        else:
            summary.append(
                "生命表 跳过：未找到死亡率输入——请用 config['qx'] 或 config['rate']（m_x）"
                "或 config['deaths']+config['exposure']（→ m_x=死亡/暴露）指定。"
            )
            return

    # ---- order by age, drop rows missing age or mortality -----------------
    order = np.argsort(age, kind="stable")
    age = age[order]
    if mx is not None:
        mx = mx[order]
        valid = np.isfinite(age) & np.isfinite(mx)
    else:
        qx_in = qx_in[order]
        valid = np.isfinite(age) & np.isfinite(qx_in)
    age = age[valid]
    if mx is not None:
        mx = mx[valid]
    else:
        qx_in = qx_in[valid]
    n_ages = int(age.size)
    if n_ages < 2:
        summary.append(f"生命表 跳过：有效年龄组过少（{n_ages}<2）。")
        return

    try:
        # ---- interval widths n_x from age spacing (last interval reuses prior) --
        nwid = np.diff(age)
        nwid = np.append(nwid, nwid[-1] if nwid.size else 1.0)
        nwid = np.where(np.isfinite(nwid) & (nwid > 0), nwid, 1.0)

        # ---- a_x (mid-period default 0.5; config overridable scalar) ----------
        try:
            ax_default = float(cfg.get("a_x", 0.5))
        except (TypeError, ValueError):
            ax_default = 0.5
        ax = np.full(n_ages, ax_default, dtype=float)
        # a_x is expressed per-interval as a fraction; person-years use a_x*n locally.
        ax_years = ax * nwid

        try:
            radix = float(cfg.get("radix", 100000.0))
        except (TypeError, ValueError):
            radix = 100000.0
        if not (radix > 0):
            radix = 100000.0

        # ---- q_x from m_x (or use supplied q_x directly) ----------------------
        if mx is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                qx = (nwid * mx) / (1.0 + (nwid - ax_years) * mx)
            qx = np.clip(qx, 0.0, 1.0)
            mx_eff = mx.copy()
        else:
            qx = np.clip(qx_in.astype(float), 0.0, 1.0)
            # back out an approximate m_x for the open-interval closure / e_x check
            with np.errstate(divide="ignore", invalid="ignore"):
                mx_eff = qx / (nwid - qx * (nwid - ax_years))
            mx_eff = np.where(np.isfinite(mx_eff), mx_eff, np.nan)

        # close the last (open) interval: everyone alive must eventually die
        qx[-1] = 1.0
        px = 1.0 - qx

        # ---- l_x (radix l_0), d_x ------------------------------------------
        lx = np.empty(n_ages, dtype=float)
        lx[0] = radix
        for i in range(1, n_ages):
            lx[i] = lx[i - 1] * px[i - 1]
        dx = lx * qx  # = l_x - l_{x+1}

        # ---- L_x person-years lived in [x, x+n) ----------------------------
        lx_next = lx - dx  # = l_{x+1}
        Lx = nwid * lx_next + ax_years * dx
        # open last interval: L_x = l_x / m_x if m_x>0, else fall back to l_x*a_x*n
        if np.isfinite(mx_eff[-1]) and mx_eff[-1] > 0:
            Lx[-1] = lx[-1] / mx_eff[-1]
        else:
            Lx[-1] = lx[-1] * max(ax_years[-1], nwid[-1])

        # ---- T_x (reverse cumulative person-years) and e_x -----------------
        Tx = np.cumsum(Lx[::-1])[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ex = np.where(lx > 0, Tx / lx, np.nan)

        tbl = pd.DataFrame({
            "age": np.round(age, 6),
            "n": np.round(nwid, 6),
            "m_x": np.round(mx_eff, 8),
            "q_x": np.round(qx, 8),
            "p_x": np.round(px, 8),
            "l_x": np.round(lx, 4),
            "d_x": np.round(dx, 4),
            "L_x": np.round(Lx, 4),
            "T_x": np.round(Tx, 4),
            "e_x": np.round(ex, 4),
        })
        tbl.to_csv(d / "life_table.csv", index=False, encoding="utf-8")
        files.append("life_table.csv")

        e0 = float(ex[0])
        e_min = float(ex[0])  # ex at the minimum age (= first row after sort)
        total_py = float(Tx[0])

        estimates.update({
            "e0": round(e0, 4),
            "e_at_min_age": round(e_min, 4),
            "radix": round(radix, 4),
            "n_ages": float(n_ages),
            "total_person_years": round(total_py, 4),
            "min_age": round(float(age[0]), 4),
            "max_age": round(float(age[-1]), 4),
        })

        def _plot(plt):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
            ax1.plot(age, lx, color="#4C72B0", lw=1.8, marker="o", ms=3)
            ax1.set_xlabel("age x")
            ax1.set_ylabel("survivors l_x")
            ax1.set_title(f"Survival curve (radix l_0={radix:.0f})")
            ax2.plot(age, ex, color="#C44E52", lw=1.8, marker="s", ms=3)
            ax2.set_xlabel("age x")
            ax2.set_ylabel("life expectancy e_x")
            ax2.set_title(f"Life expectancy (e_0 = {e0:.2f} yrs)")

        _save_fig(d, "life_table.png", files, _plot)

        summary.append(
            f"{ctx.entry.method} 完成：年龄列={age_name}，{src_note}（{n_ages} 个年龄组，"
            f"基数 radix={radix:.0f}）；出生时预期寿命 e_0={e0:.2f} 岁。"
            f"全表（q_x/p_x/l_x/d_x/L_x/T_x/e_x）见 life_table.csv，生存曲线与 e_x 见图。"
            " ⚠ 这是「时期(period)生命表」——用当期各年龄死亡率合成的假想队列，"
            "不是真实出生队列(cohort)的经历，受死亡率随时间变化影响。"
            " ⚠ 区间内死亡分布假定 a_x=0.5（年中、均匀），婴儿/老年组实际并非如此（可用 config['a_x'] 调整）。"
            " ⚠ 末段为开区间，按 q_x=1、L_x=l_x/m_x 封闭，最高龄组的 e_x 对此封闭法较敏感。"
        )
        code += [
            "import numpy as np",
            "n = np.diff(age); n = np.append(n, n[-1])     # 区间宽度",
            "a_x = 0.5                                      # 年中假定",
            "q_x = n*m_x / (1 + (n - a_x*n)*m_x)            # 死亡概率",
            "q_x[-1] = 1.0                                  # 末开区间封闭",
            "p_x = 1 - q_x",
            "l_x = radix * np.cumprod(np.r_[1, p_x[:-1]])   # 生存人数",
            "d_x = l_x * q_x",
            "L_x = n*(l_x - d_x) + a_x*n*d_x                # 人年；末段 = l_x/m_x",
            "T_x = np.cumsum(L_x[::-1])[::-1]",
            "e_x = T_x / l_x                                # 预期寿命；e_0 = e_x[0]",
        ]
    except Exception as exc:
        summary.append(f"生命表 计算失败：{exc}")


# ===========================================================================
# 2) chain_ladder — claims-reserving on a run-off triangle (real estimation)
#    Refs: Mack (1993) "Distribution-free calculation of the standard error of
#    chain ladder reserve estimates", ASTIN Bulletin; England & Verrall (2002).
#
#    On a cumulative run-off triangle C_{i,j} (origin i x development j):
#      f_j = sum_i C_{i,j+1} / sum_i C_{i,j}    volume-weighted age-to-age factor
#            (over origins i present in BOTH columns j and j+1)
#      CDF_j = prod_{k>=j} f_k                   cumulative development factor
#      project: C_{i,j+1} = C_{i,j} * f_j        fill the lower triangle
#      ultimate_i = latest_i * prod(remaining f) projected ultimate per origin
#      IBNR_i = ultimate_i - latest_i            reserve per origin
#      total reserve = sum_i IBNR_i
#    Mack standard error (NICE bonus) computed when the triangle is regular.
# ===========================================================================
@register("chain_ladder")
def _branch_chain_ladder(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg

    tri, origins, devs, layout_note, err = _resolve_triangle(ctx)
    if err is not None:
        summary.append(f"链梯法(准备金) 跳过：{err}")
        return

    try:
        n_origin, n_dev = tri.shape
        # require at least 2 development columns to estimate a factor
        if n_dev < 2:
            summary.append("链梯法(准备金) 跳过：发展期不足（至少需要 2 个发展期列）。")
            return

        # ---- volume-weighted age-to-age development factors f_j -------------
        f = np.full(n_dev - 1, np.nan, dtype=float)
        n_pairs = np.zeros(n_dev - 1, dtype=int)
        for j in range(n_dev - 1):
            cj = tri[:, j]
            cj1 = tri[:, j + 1]
            m = np.isfinite(cj) & np.isfinite(cj1)
            denom = np.nansum(cj[m])
            numer = np.nansum(cj1[m])
            n_pairs[j] = int(np.count_nonzero(m))
            if denom > 0:
                f[j] = numer / denom

        # any factor that could not be estimated (no overlapping pairs / zero
        # exposure) is treated as 1.0 (no further development) and disclosed.
        unest = ~np.isfinite(f)
        n_unest = int(np.count_nonzero(unest))
        f_filled = np.where(np.isfinite(f), f, 1.0)

        # cumulative development factors CDF_j = prod_{k>=j} f_k (length n_dev;
        # CDF at the last column = 1.0). CDF_j multiplies a value at dev j to ultimate.
        cdf = np.ones(n_dev, dtype=float)
        for j in range(n_dev - 2, -1, -1):
            cdf[j] = cdf[j + 1] * f_filled[j]

        # ---- project lower triangle + per-origin ultimate / latest / IBNR --
        full = tri.copy()
        latest = np.full(n_origin, np.nan, dtype=float)
        latest_dev = np.full(n_origin, -1, dtype=int)
        ultimate = np.full(n_origin, np.nan, dtype=float)
        for i in range(n_origin):
            row = tri[i, :]
            known = np.where(np.isfinite(row))[0]
            if known.size == 0:
                continue
            k = int(known.max())          # latest observed development period
            latest[i] = row[k]
            latest_dev[i] = k
            # fill forward through the lower triangle with f
            val = row[k]
            for j in range(k, n_dev - 1):
                val = val * f_filled[j]
                full[i, j + 1] = val
            ultimate[i] = val if k < n_dev - 1 else row[k]

        ibnr = ultimate - latest
        total_reserve = float(np.nansum(ibnr))
        total_ultimate = float(np.nansum(ultimate))
        total_latest = float(np.nansum(latest))

        # ---- optional Mack standard error (NICE bonus) ---------------------
        mack_total_se = _mack_total_se(tri, f_filled, latest_dev, ultimate)

        # ---- per-origin reserve table --------------------------------------
        per = pd.DataFrame({
            "origin": origins,
            "latest_dev": latest_dev,
            "latest_paid": np.round(latest, 4),
            "ultimate": np.round(ultimate, 4),
            "ibnr_reserve": np.round(ibnr, 4),
        })
        per.to_csv(d / "chain_ladder_reserve.csv", index=False, encoding="utf-8")
        files.append("chain_ladder_reserve.csv")

        # completed (projected) triangle
        full_df = pd.DataFrame(np.round(full, 4),
                               index=[str(o) for o in origins],
                               columns=[f"dev_{dv}" for dv in devs])
        full_df.to_csv(d / "chain_ladder_triangle.csv", encoding="utf-8")
        files.append("chain_ladder_triangle.csv")

        # development factors table
        fac_df = pd.DataFrame({
            "from_dev": [str(devs[j]) for j in range(n_dev - 1)],
            "to_dev": [str(devs[j + 1]) for j in range(n_dev - 1)],
            "age_to_age_factor": np.round(f_filled, 6),
            "cumulative_factor": np.round(cdf[:-1], 6),
            "n_pairs": n_pairs,
            "estimated": ~unest,
        })
        fac_df.to_csv(d / "chain_ladder_factors.csv", index=False, encoding="utf-8")
        files.append("chain_ladder_factors.csv")

        estimates.update({
            "total_reserve": round(total_reserve, 4),
            "total_ultimate": round(total_ultimate, 4),
            "total_latest_paid": round(total_latest, 4),
            "n_origins": float(n_origin),
            "n_dev_periods": float(n_dev),
            "oldest_factor": round(float(f_filled[0]), 6),
            "latest_factor": round(float(f_filled[-1]), 6),
            "mack_total_se": round(mack_total_se, 4) if mack_total_se == mack_total_se else float("nan"),
        })
        cv = (mack_total_se / total_reserve) if (mack_total_se == mack_total_se and total_reserve > 0) else float("nan")
        if cv == cv:
            estimates["mack_cv"] = round(cv, 6)

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            xs = np.arange(n_dev - 1)
            ax.bar(xs, f_filled, color="#4C72B0", edgecolor="white")
            ax.axhline(1.0, color="#333333", lw=0.8, ls="--")
            for x, v in zip(xs, f_filled):
                ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{devs[j]}->{devs[j+1]}" for j in range(n_dev - 1)],
                               rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("development period transition")
            ax.set_ylabel("age-to-age factor f_j")
            ax.set_title(f"Chain-ladder development factors (reserve={total_reserve:.4g})")

        _save_fig(d, "chain_ladder_factors.png", files, _plot)

        mack_txt = ""
        if mack_total_se == mack_total_se:
            mack_txt = (f" Mack 标准误 SE(总准备金)={mack_total_se:.4g}"
                        f"（变异系数 CV={cv:.1%}）。")
        unest_txt = ""
        if n_unest > 0:
            unest_txt = (f" ⚠ 有 {n_unest} 个发展因子因无重叠数据/零暴露无法估计，已置为 1.0"
                         "（视为不再发展），相应准备金可能被低估。")
        summary.append(
            f"{ctx.entry.method} 完成（{layout_note}）：{n_origin} 个事故年(origin) × {n_dev} 个发展期；"
            f"按体量加权的发展因子 f_j（最早 {f_filled[0]:.4f} → 最新 {f_filled[-1]:.4f}）见 chain_ladder_factors.csv 与图；"
            f"投影下三角后，已决赔款合计={total_latest:.4g}、预计最终赔款(ultimate)合计={total_ultimate:.4g}，"
            f"**未决赔款准备金(IBNR)合计={total_reserve:.4g}**（逐 origin 见 chain_ladder_reserve.csv）。{mack_txt}{unest_txt}"
            " ⚠ 链梯法假定各事故年的发展模式稳定、未来比例与历史一致，且无日历年(通胀/法规/理赔速度变化)效应；"
            "这些被违反时（如赔付加速、近年大案）准备金会有偏。"
            + ("" if mack_total_se == mack_total_se else
               " ⚠ 基础链梯法只给点估计、不含置信区间（本次未给出 Mack 标准误，可能因三角不规则）。")
        )
        code += [
            "import numpy as np",
            "# 体量加权年龄-年龄发展因子 f_j = sum_i C[i,j+1] / sum_i C[i,j]",
            "f = np.array([np.nansum(C[:, j+1][m]) / np.nansum(C[:, j][m])",
            "              for j in range(C.shape[1]-1)])   # m = 两列都有的行",
            "# 投影下三角：C[i,j+1] = C[i,j] * f[j]；ultimate = latest * prod(剩余 f)",
            "ibnr = ultimate - latest                       # 逐 origin 准备金",
            "total_reserve = np.nansum(ibnr)                # IBNR 合计",
        ]
    except Exception as exc:
        summary.append(f"链梯法(准备金) 计算失败：{exc}")


def _resolve_triangle(ctx: Ctx):
    """Resolve a cumulative run-off triangle as a (n_origin x n_dev) float array.

    Returns (tri, origins, devs, layout_note, None) on success, or
    (None, None, None, None, err_msg) on honest failure.

    Layout detection (see STOP-AND-REPORT in the build brief):
      * LONG form, preferred when config gives the roles: config['origin'],
        config['dev'], config['claims'] -> pivot to a triangle (cumulative claims
        per origin x dev). Used whenever all three are present as columns.
      * WIDE form otherwise: a square-ish numeric block whose lower-left is NaN is
        treated as the triangle directly. An ORIGIN-LABEL column is peeled off when
        (a) a leading non-numeric column exists, OR (b) a column is named like an
        origin/accident/year/cohort identifier, OR (c) the first column is fully
        observed (no NaN) while a later column has NaNs (the run-off staircase lives
        in the remaining columns) — common when origin years (2018, 2019, ...) are
        numeric and would otherwise be mistaken for a development period. The
        remaining numeric columns are the development periods 0, 1, 2, ...
    """
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg

    origin_name = cfg.get("origin")
    dev_name = cfg.get("dev")
    claims_name = cfg.get("claims")

    # names that mark a column as the ORIGIN-period label rather than claims data
    _origin_like = ("origin", "accident", "acc_year", "acc", "ay", "year",
                    "cohort", "period", "origin_period", "underwriting", "uw")

    # ---- LONG form: explicit origin / dev / claims roles -------------------
    if (origin_name in df.columns and dev_name in df.columns and claims_name in df.columns):
        try:
            sub = df[[origin_name, dev_name, claims_name]].copy()
            sub[claims_name] = pd.to_numeric(sub[claims_name], errors="coerce")
            piv = sub.pivot_table(index=origin_name, columns=dev_name,
                                  values=claims_name, aggfunc="sum")
            piv = piv.sort_index(axis=0).sort_index(axis=1)
            origins = list(piv.index)
            devs = list(piv.columns)
            tri = piv.to_numpy(dtype=float)
            if tri.shape[0] < 2 or tri.shape[1] < 2:
                return None, None, None, None, "长表数据透视后三角过小（origin 或 dev 不足 2）。"
            note = f"长表：origin={origin_name}, dev={dev_name}, claims={claims_name}"
            return tri, origins, devs, note, None
        except Exception as exc:
            return None, None, None, None, f"长表透视失败：{exc}"

    # ---- WIDE form: a numeric block with a NaN lower triangle ---------------
    num_block = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    numeric_mask = num_block.notna().any(axis=0)
    numeric_cols = [c for c in df.columns if numeric_mask.get(c, False)]

    # Identify the ORIGIN-LABEL column (peeled off; NOT a development period):
    origin_label_col = None
    #  (a) the first leading non-numeric column, if any
    for c in df.columns:
        if c not in numeric_cols:
            origin_label_col = c
            break
    #  (b) a column whose NAME marks it as an origin identifier (origin/accident/
    #      year/cohort/...). We deliberately do NOT guess a label from a label-less
    #      pure-numeric block: in a run-off triangle the first development column is
    #      also fully observed, so a "staircase" heuristic would wrongly drop dev0.
    #      For an unnamed numeric origin column, the user supplies long-form roles.
    if origin_label_col is None:
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc in _origin_like or any(lc.startswith(tok + "_") for tok in _origin_like):
                origin_label_col = c
                break

    # the development columns = numeric columns minus the origin label
    dev_cols = [c for c in numeric_cols if c != origin_label_col]
    if len(dev_cols) < 2:
        return None, None, None, None, (
            "未识别到发展三角：请用宽表（数值矩阵、下三角为 NaN）"
            "或长表 config['origin']+config['dev']+config['claims']。"
        )

    tri = df[dev_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).to_numpy(dtype=float)
    if tri.shape[0] < 2:
        return None, None, None, None, "宽表三角行数不足（至少需要 2 个 origin）。"

    # sanity: a run-off triangle has NaNs in the lower-left (later origins are
    # observed for fewer development periods). Warn-but-proceed if it looks square.
    n_origin, n_dev = tri.shape
    lower_left_nan = False
    for i in range(n_origin):
        # rows further down should have trailing NaNs
        if i > 0 and np.isnan(tri[i, -1]) and not np.isnan(tri[0, -1]):
            lower_left_nan = True
            break
    shape_note = "下三角为 NaN" if lower_left_nan else "矩形/满矩阵（按已有值估计）"

    if origin_label_col is not None:
        origins = list(df[origin_label_col].astype(str))
    else:
        origins = [str(i) for i in range(n_origin)]
    devs = list(range(n_dev))
    note = f"宽表：{n_origin}×{n_dev} 数值块（{shape_note}）"
    return tri, origins, devs, note, None


def _mack_total_se(tri, f, latest_dev, ultimate):
    """Mack (1993) standard error of the TOTAL chain-ladder reserve.

    Returns nan when the triangle is too irregular to estimate the volatility
    parameters sigma_j (the engine then discloses a point estimate only).

    sigma_j^2 = (1/(I-j-1)) * sum_i C_{i,j} * (C_{i,j+1}/C_{i,j} - f_j)^2
    (last sigma extrapolated by Mack's min rule). MSE of each origin's ultimate +
    the cross-origin covariance term give the total SE. This is the standard Mack
    formula; see ASTIN Bulletin 23(2).
    """
    import numpy as np

    try:
        n_origin, n_dev = tri.shape
        J = n_dev - 1
        sigma2 = np.full(J, np.nan, dtype=float)
        for j in range(J):
            cj = tri[:, j]
            cj1 = tri[:, j + 1]
            m = np.isfinite(cj) & np.isfinite(cj1) & (cj > 0)
            cnt = int(np.count_nonzero(m))
            if cnt >= 2:
                ratio = cj1[m] / cj[m]
                sigma2[j] = np.sum(cj[m] * (ratio - f[j]) ** 2) / (cnt - 1)
            elif cnt == 1:
                sigma2[j] = np.nan  # filled below by Mack's tail rule
        # Mack tail rule for the last (or any unestimable) sigma:
        # sigma_J^2 = min(sigma_{J-1}^2, sigma_{J-2}^2, sigma_{J-1}^4/sigma_{J-2}^2)
        finite_idx = [j for j in range(J) if np.isfinite(sigma2[j])]
        if len(finite_idx) == 0:
            return float("nan")
        last_known = finite_idx[-1]
        for j in range(J):
            if not np.isfinite(sigma2[j]):
                if j == J - 1 and len(finite_idx) >= 2:
                    a = sigma2[finite_idx[-1]]
                    b = sigma2[finite_idx[-2]]
                    sigma2[j] = min(a, b, (a * a / b) if b > 0 else a)
                else:
                    sigma2[j] = sigma2[last_known]

        # column sums S_j = sum of C_{i,j} over origins observed at dev j (used in
        # the parameter-uncertainty term)
        col_sum = np.array([np.nansum(tri[:, j][np.isfinite(tri[:, j])]) for j in range(n_dev)])

        # per-origin MSE of the ultimate (process + estimation), Mack eq.
        mse = np.zeros(n_origin, dtype=float)
        for i in range(n_origin):
            k = int(latest_dev[i])
            if k < 0 or k >= J or not np.isfinite(ultimate[i]):
                continue
            u2 = ultimate[i] ** 2
            term = 0.0
            for j in range(k, J):
                if f[j] <= 0 or not np.isfinite(sigma2[j]):
                    continue
                Sj = col_sum[j]                     # column volume S_j
                cij = _full_value(tri, ultimate, f, i, j)  # projected C_{i,j}
                if not (cij and cij > 0):
                    continue
                # Mack: (sigma_j^2 / f_j^2) * (1/C_{i,j} [process] + 1/S_j [estimation])
                proc = 1.0 / cij
                estm = (1.0 / Sj) if Sj > 0 else 0.0
                term += (sigma2[j] / (f[j] ** 2)) * (proc + estm)
            mse[i] = u2 * term

        total_var_proc_est = float(np.nansum(mse))

        # cross-origin covariance term (parameter uncertainty shared via S_j)
        cov = 0.0
        for j in range(J):
            if f[j] <= 0 or not np.isfinite(sigma2[j]) or col_sum[j] <= 0:
                continue
            affected = [i for i in range(n_origin)
                        if 0 <= latest_dev[i] <= j < J and np.isfinite(ultimate[i])]
            if len(affected) < 2:
                continue
            usum = np.sum([ultimate[i] for i in affected])
            usq = np.sum([ultimate[i] ** 2 for i in affected])
            cross = usum * usum - usq
            cov += cross * (sigma2[j] / (f[j] ** 2)) / col_sum[j]

        total_var = total_var_proc_est + cov
        if total_var <= 0 or not np.isfinite(total_var):
            return float("nan")
        return float(np.sqrt(total_var))
    except Exception:
        return float("nan")


def _full_value(tri, ultimate, f, i, j):
    """Projected cumulative value C_{i,j} for origin i at development j (observed if
    known, else back-projected from the ultimate). Used by the Mack process-variance
    term. Returns nan if not determinable."""
    import numpy as np

    v = tri[i, j]
    if np.isfinite(v):
        return float(v)
    # back out from ultimate by dividing off the forward factors j..J-1
    J = tri.shape[1] - 1
    if not np.isfinite(ultimate[i]):
        return float("nan")
    val = ultimate[i]
    for k in range(J - 1, j - 1, -1):
        if f[k] and f[k] > 0:
            val = val / f[k]
        else:
            return float("nan")
    return float(val)


# ===========================================================================
# 3) loss_distribution — severity fit (MLE) + risk measures VaR / TVaR
#    Refs: Klugman, Panjer & Willmot "Loss Models: From Data to Decisions";
#    Artzner et al. (1999) coherent risk measures (TVaR / Expected Shortfall).
#
#    Fit candidate severity distributions by maximum likelihood (scipy .fit):
#      lognormal, gamma, pareto, weibull_min. Pick the best by AIC = 2k - 2*loglik.
#    Risk measures at level alpha:
#      VaR_alpha = F^{-1}(alpha)                            (the alpha-quantile)
#      TVaR_alpha = E[X | X > VaR_alpha]
#                 = (1/(1-alpha)) * integral_alpha^1 F^{-1}(u) du   (tail mean / CVaR)
# ===========================================================================
@register("loss_distribution")
def _branch_loss_distribution(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg

    losses, loss_name, err = _resolve_losses(ctx)
    if err is not None:
        summary.append(f"损失分布(严重度) 跳过：{err}")
        return

    try:
        from scipy import stats
    except Exception:
        summary.append("损失分布(严重度) 跳过：缺少 scipy（MLE 拟合依赖 scipy.stats）。")
        return

    try:
        # alpha levels for risk measures (95% and 99% by default; config['alpha']
        # may give a scalar or a list to add/override the primary level).
        alphas = [0.95, 0.99]
        a_cfg = cfg.get("alpha")
        if a_cfg is not None:
            try:
                if isinstance(a_cfg, (list, tuple)):
                    extra = [float(x) for x in a_cfg]
                else:
                    extra = [float(a_cfg)]
                for x in extra:
                    if 0.0 < x < 1.0 and x not in alphas:
                        alphas.append(x)
            except (TypeError, ValueError):
                pass
        alphas = sorted(set(alphas))

        n = int(losses.size)
        mean_loss = float(np.mean(losses))
        median_loss = float(np.median(losses))

        # candidate distributions: name -> (scipy dist, n_params incl loc/scale we fit)
        candidates = {
            "lognormal": stats.lognorm,
            "gamma": stats.gamma,
            "pareto": stats.pareto,
            "weibull": stats.weibull_min,
        }

        fits = {}  # name -> dict(params, loglik, k, aic, ok)
        for name, dist in candidates.items():
            try:
                # fit with loc fixed at 0 for these positive-support severities
                # (floc=0 is the standard actuarial convention for loss data).
                params = dist.fit(losses, floc=0.0)
                ll = float(np.sum(dist.logpdf(losses, *params)))
                if not np.isfinite(ll):
                    continue
                # free parameters: shape(s) + scale (loc is fixed -> not counted)
                k = len([p for p in params]) - 1  # subtract the fixed loc
                aic = 2.0 * k - 2.0 * ll
                bic = k * np.log(n) - 2.0 * ll
                fits[name] = {
                    "params": params, "loglik": ll, "k": k,
                    "aic": float(aic), "bic": float(bic), "dist": dist,
                }
            except Exception:
                continue

        if not fits:
            summary.append(
                "损失分布(严重度) 跳过：所有候选分布(lognormal/gamma/pareto/weibull)的 MLE 拟合均失败"
                "（数据可能不适合这些正支撑严重度分布）。"
            )
            return

        # pick best by AIC
        best_name = min(fits, key=lambda nm: fits[nm]["aic"])
        best = fits[best_name]
        best_dist = best["dist"]
        best_params = best["params"]

        def _var(dist, params, a):
            return float(dist.ppf(a, *params))

        def _tvar(dist, params, a):
            # TVaR_a = (1/(1-a)) * integral_a^1 ppf(u) du, by numerical quadrature
            from scipy import integrate
            if a >= 1.0:
                return float("nan")
            try:
                val, _ = integrate.quad(lambda u: dist.ppf(u, *params), a, 1.0,
                                        limit=200)
                return float(val / (1.0 - a))
            except Exception:
                # fallback: average ppf on a fine grid in (a,1)
                us = np.linspace(a, 1.0, 2000, endpoint=False)[1:]
                return float(np.mean(dist.ppf(us, *params)))

        # risk measures at every alpha for the BEST distribution
        var_at = {a: _var(best_dist, best_params, a) for a in alphas}
        tvar_at = {a: _tvar(best_dist, best_params, a) for a in alphas}

        # comparison table of all fitted distributions
        rows = []
        for name in sorted(fits, key=lambda nm: fits[nm]["aic"]):
            fdat = fits[name]
            rows.append({
                "distribution": name,
                "n_params": fdat["k"],
                "loglik": round(fdat["loglik"], 4),
                "aic": round(fdat["aic"], 4),
                "bic": round(fdat["bic"], 4),
                "delta_aic": round(fdat["aic"] - best["aic"], 4),
                "params": ", ".join(f"{p:.6g}" for p in fdat["params"]),
            })
        fits_df = pd.DataFrame(rows)
        fits_df.to_csv(d / "loss_distribution_fits.csv", index=False, encoding="utf-8")
        files.append("loss_distribution_fits.csv")

        # risk-measure table for the best distribution
        risk_rows = []
        for a in alphas:
            risk_rows.append({
                "alpha": a,
                "VaR": round(var_at[a], 6),
                "TVaR_CVaR": round(tvar_at[a], 6),
            })
        pd.DataFrame(risk_rows).to_csv(d / "loss_distribution_risk.csv",
                                       index=False, encoding="utf-8")
        files.append("loss_distribution_risk.csv")

        # fitted shape parameter (first shape param; for lognorm = sigma, gamma = a,
        # pareto = b, weibull = c) — a single representative number for estimates.
        fitted_shape = float(best_params[0]) if len(best_params) >= 1 else float("nan")

        est_update = {
            "best_aic": round(best["aic"], 4),
            "best_bic": round(best["bic"], 4),
            "best_loglik": round(best["loglik"], 4),
            "mean_loss": round(mean_loss, 6),
            "median_loss": round(median_loss, 6),
            "fitted_shape": round(fitted_shape, 6) if fitted_shape == fitted_shape else float("nan"),
            "n_losses": float(n),
            "n_dists_fit": float(len(fits)),
        }
        # required named VaR/TVaR at 95 and 99 (nan if that level wasn't requested)
        for lvl, key in ((0.95, "95"), (0.99, "99")):
            est_update[f"var_{key}"] = round(var_at[lvl], 6) if lvl in var_at else float("nan")
            est_update[f"tvar_{key}"] = round(tvar_at[lvl], 6) if lvl in tvar_at else float("nan")
        estimates.update(est_update)

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.hist(losses, bins=min(50, max(10, n // 5)), density=True,
                    color="#bbbbbb", edgecolor="white", alpha=0.85, label="loss data")
            xs = np.linspace(max(1e-9, losses.min()), losses.max() * 1.05, 400)
            ax.plot(xs, best_dist.pdf(xs, *best_params), color="#4C72B0", lw=2.0,
                    label=f"fitted {best_name} (AIC={best['aic']:.1f})")
            if 0.95 in var_at:
                ax.axvline(var_at[0.95], color="#DD8452", ls="--", lw=1.3,
                           label=f"VaR 95% = {var_at[0.95]:.4g}")
            if 0.99 in var_at:
                ax.axvline(var_at[0.99], color="#C44E52", ls="--", lw=1.3,
                           label=f"VaR 99% = {var_at[0.99]:.4g}")
            ax.set_xlabel("loss amount")
            ax.set_ylabel("density")
            ax.set_title(f"Loss severity: best fit = {best_name}")
            ax.legend(fontsize=8)

        _save_fig(d, "loss_distribution.png", files, _plot)

        rank_txt = "、".join(
            f"{r['distribution']}(AIC={r['aic']:.1f})" for r in rows[:4]
        )
        v95 = var_at.get(0.95, float("nan"))
        v99 = var_at.get(0.99, float("nan"))
        t95 = tvar_at.get(0.95, float("nan"))
        t99 = tvar_at.get(0.99, float("nan"))
        summary.append(
            f"{ctx.entry.method} 完成：损失列={loss_name}（n={n}，均值={mean_loss:.4g}、中位数={median_loss:.4g}）；"
            f"按 AIC 在 lognormal/gamma/pareto/weibull 中选优——**最优={best_name}**（AIC={best['aic']:.2f}），"
            f"候选排序：{rank_txt}（全表见 loss_distribution_fits.csv）。"
            f"风险度量（基于最优分布）：VaR_95%={v95:.4g}、VaR_99%={v99:.4g}；"
            f"TVaR/CVaR_95%={t95:.4g}、TVaR/CVaR_99%={t99:.4g}（即超过 VaR 的尾部条件均值，见 loss_distribution_risk.csv）。"
            " ⚠ VaR/TVaR 完全取决于所拟合的尾部——**模型风险**：换一个分布尾部行为差异巨大，"
            "本引擎用 AIC 选优但 AIC 主要看整体拟合、对极尾约束有限。"
            " ⚠ 重尾(如 pareto)数据需谨慎：均值/方差可能不存在，外推到 99%+ 分位很不稳定，样本量小尤甚。"
            " ⚠ 仅适用于正值损失（loc 固定为 0）；若数据含免赔额/限额(截断/删失)，应改用截断/删失似然，本次未处理。"
            " ⚠ TVaR(尾部条件均值)比 VaR 更稳健且为一致性风险度量(可加)，建议优先看 TVaR。"
        )
        code += [
            "from scipy import stats, integrate",
            "import numpy as np",
            "cands = {'lognormal': stats.lognorm, 'gamma': stats.gamma,",
            "         'pareto': stats.pareto, 'weibull': stats.weibull_min}",
            "fits = {}",
            "for nm, dist in cands.items():",
            "    p = dist.fit(losses, floc=0.0)            # MLE, loc 固定为 0",
            "    ll = np.sum(dist.logpdf(losses, *p))",
            "    k = len(p) - 1                            # 自由参数(扣除 loc)",
            "    fits[nm] = (p, dist, 2*k - 2*ll)          # AIC = 2k - 2ll",
            "best = min(fits, key=lambda nm: fits[nm][2]) # 按 AIC 选优",
            "p, dist, _ = fits[best]",
            "VaR = dist.ppf(alpha, *p)                     # 分位数",
            "TVaR = integrate.quad(lambda u: dist.ppf(u,*p), alpha,1)[0]/(1-alpha)",
        ]
    except Exception as exc:
        summary.append(f"损失分布(严重度) 计算失败：{exc}")


def _resolve_losses(ctx: Ctx):
    """Resolve the loss / claim-amount series as a 1-D float numpy array of POSITIVE
    values. Returns (losses, name, None) on success or (None, None, err) on failure.

    Resolution order:
      1. config['loss'] / config['amount'] -> that column.
      2. auto: the first numeric column whose values are predominantly positive.
    """
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    name = cfg.get("loss") or cfg.get("amount")

    chosen = None
    if name in df.columns:
        chosen = name
    else:
        for c in _numeric_cols(ctx):
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 5 and np.mean(vals > 0) >= 0.9:
                chosen = c
                break
        if chosen is None and _numeric_cols(ctx):
            chosen = _numeric_cols(ctx)[0]

    if chosen is None:
        return None, None, "需要一个损失/赔款金额数值列（config['loss'] 或 config['amount']）。"

    losses = pd.to_numeric(df[chosen], errors="coerce").to_numpy(dtype=float)
    losses = losses[np.isfinite(losses)]
    losses = losses[losses > 0]  # severity models require positive losses
    if losses.size < 5:
        return None, None, (
            f"列 {chosen} 的正值损失过少（{losses.size}<5）——严重度拟合需要足够的正值观测。"
        )
    return losses.astype(float), chosen, None
