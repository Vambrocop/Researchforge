"""Branch handlers for the STATISTICAL PROCESS CONTROL / quality-engineering
family (``family="spc"``, ``domain="quality"``).

A coherent quality trio — **monitor → assess capability → assess the gauge**:

  - control_chart      — Shewhart control chart (X-bar & R if subgroups, else I-MR)
                         with out-of-control detection (basic 3σ rule + Western
                         Electric runs rules).
  - process_capability — Cp / Cpk (within-σ) and Pp / Ppk (overall-σ) vs spec
                         limits, plus the normal-model %-out-of-spec (ppm).
  - gage_rr            — Gauge R&R Measurement-System-Analysis by the ANOVA method
                         (AIAG): variance components, %Contribution, %StudyVar, ndc.

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "跳过:…" appended to summary + return (never crash/fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    a flat float ``estimates`` dict (NaN for an index that does not apply), Chinese
    ``summary`` with ⚠ assumption/bias disclosures.
  * Reproducible analysis code appended to ``code``.

Profiler traps handled: a measurement column profiles as ``continuous``; a subgroup /
part id may profile as ``id`` (all-distinct integers) or ``count``; an operator label
as ``categorical``. Column resolution is tolerant (accepts continuous/count/id for the
measurement, and categorical/binary/count/id for grouping roles).

Pure Python (numpy / pandas / matplotlib; statsmodels OLS for the Gage R&R two-way
ANOVA — already a project dependency). NO R.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
# Shewhart X-bar & R control-chart constants by subgroup size n (n = 2..10).
# Standard AIAG / ASTM table: A2 (X-bar limits), D3 / D4 (R-chart limits),
# d2 (mean range -> sigma: sigma_hat = R_bar / d2).
_XBAR_R_CONST = {
    # n:  (A2,    D3,    D4,    d2)
    2:  (1.880, 0.000, 3.267, 1.128),
    3:  (1.023, 0.000, 2.574, 1.693),
    4:  (0.729, 0.000, 2.282, 2.059),
    5:  (0.577, 0.000, 2.114, 2.326),
    6:  (0.483, 0.000, 2.004, 2.534),
    7:  (0.419, 0.076, 1.924, 2.704),
    8:  (0.373, 0.136, 1.864, 2.847),
    9:  (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}
# d2 for the moving range of 2 consecutive individuals (I-MR chart): n=2 -> 1.128,
# so the individuals control limits are x_bar ± 3 * (MR_bar / 1.128) = x_bar ± 2.66*MR_bar.
_D2_MR2 = 1.128


def _measurement_col(ctx: Ctx):
    """Resolve the numeric measurement column.

    Priority: config ``measurement`` -> a continuous/count/id column. The
    time/unit columns are excluded from the auto-pick. Returns the column name or
    None if no usable numeric column exists."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    m = cfg.get("measurement")
    if m in df.columns:
        return m
    excl = {fp.unit_col, fp.time_col}
    for c in fp.columns:
        if c.name in excl:
            continue
        if c.kind in ("continuous", "count", "id"):
            return c.name
    return None


def _group_col(ctx: Ctx, measurement: str, key: str, max_card_frac: float = 0.6):
    """Resolve a grouping/role column (subgroup / part / operator).

    config[key] overrides with ANY column (a part id may profile as id/count, an
    operator as categorical). Auto-default: the first categorical/binary/count/id
    column that is not the measurement and not the unit/time column and whose
    cardinality is not too high (avoids picking a pure row-id as a subgroup)."""
    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df
    g = cfg.get(key)
    if g in df.columns and g != measurement:
        return g
    excl = {fp.unit_col, fp.time_col, measurement}
    n = max(len(df), 1)
    for c in fp.columns:
        if c.name in excl:
            continue
        if c.kind in ("categorical", "binary", "count", "id"):
            if c.n_unique <= 1:
                continue
            if c.n_unique / n > max_card_frac:
                continue  # too granular to be a meaningful subgroup
            return c.name
    return None


def _num(series):
    """Coerce a pandas Series to a float numpy array (never raises on dirty data)."""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


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


def _western_electric(vals, cl, sigma):
    """Western Electric runs rules on a 1-D array of points around a center line.

    Returns a list[str] same length as ``vals``; each entry is "" (in control) or a
    comma-joined list of the rules that fired at that point. Rules implemented:
      R1: a single point beyond 3σ (the basic out-of-control rule).
      R2: 2 of 3 consecutive points beyond 2σ on the SAME side.
      R3: 8 consecutive points on the SAME side of the center line.
    Zones use the chart's own sigma (the spread that produced the 3σ limits)."""
    import numpy as np

    vals = np.asarray(vals, dtype=float)
    n = vals.size
    flags = [[] for _ in range(n)]
    if not np.isfinite(sigma) or sigma <= 0:
        # fall back to basic rule via the explicit control limits only
        return ["" for _ in range(n)]

    z = (vals - cl) / sigma  # signed distance in sigma units
    # R1: beyond 3 sigma
    for i in range(n):
        if abs(z[i]) > 3.0:
            flags[i].append("R1>3σ")
    # R2: 2 of 3 consecutive beyond 2 sigma on the same side
    for i in range(2, n):
        win = z[i - 2:i + 1]
        for sign in (1.0, -1.0):
            beyond = np.sum(sign * win > 2.0)
            if beyond >= 2 and sign * z[i] > 2.0:
                flags[i].append("R2:2of3>2σ")
                break
    # R3: 8 consecutive on the same side of the center line
    for i in range(7, n):
        win = z[i - 7:i + 1]
        if np.all(win > 0) or np.all(win < 0):
            flags[i].append("R3:8sameSide")
    return [",".join(f) for f in flags]


# ===========================================================================
# 1) control_chart — Shewhart control chart (X-bar & R or I-MR)
#    Refs: Montgomery "Introduction to Statistical Quality Control"; ASTM/AIAG.
# ===========================================================================
@register("control_chart")
def _branch_control_chart(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cfg = ctx.cfg
    meas = _measurement_col(ctx)
    if meas is None:
        summary.append("控制图 跳过：找不到数值测量列（用 config['measurement'] 指定）。")
        return
    try:
        df = ctx.df
        sub_col = _group_col(ctx, meas, "subgroup")

        # Decide chart type: X-bar & R if a subgroup column yields constant subgroup
        # size n in 2..10 (Shewhart constants table); else fall back to I-MR.
        use_xbar = False
        n_sub = None
        if sub_col is not None:
            grp = df.groupby(sub_col, sort=False)
            sizes = grp[meas].apply(lambda s: int(np.isfinite(_num(s)).sum()))
            uniq = sorted(set(int(x) for x in sizes.tolist()))
            # need equal subgroup sizes (Shewhart X-bar&R) in the tabulated range, >=2 groups
            if len(uniq) == 1 and 2 <= uniq[0] <= 10 and len(sizes) >= 2:
                use_xbar = True
                n_sub = uniq[0]

        if use_xbar:
            # ---- X-bar & R chart -------------------------------------------------
            A2, D3, D4, d2 = _XBAR_R_CONST[n_sub]
            order = list(grp.groups.keys())
            means, ranges, labels = [], [], []
            for key in order:
                vals = _num(grp.get_group(key)[meas])
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                means.append(float(vals.mean()))
                ranges.append(float(vals.max() - vals.min()))
                labels.append(str(key))
            means = np.asarray(means, dtype=float)
            ranges = np.asarray(ranges, dtype=float)
            x_dbar = float(means.mean())     # grand mean (center line, X-bar chart)
            r_bar = float(ranges.mean())     # mean range (center line, R chart)
            sigma_hat = r_bar / d2 if d2 > 0 else float("nan")

            xbar_ucl = x_dbar + A2 * r_bar
            xbar_lcl = x_dbar - A2 * r_bar
            r_ucl = D4 * r_bar
            r_lcl = D3 * r_bar

            # out-of-control: X-bar point beyond its limits OR R point beyond its limits
            x_out = (means > xbar_ucl) | (means < xbar_lcl)
            r_out = (ranges > r_ucl) | (ranges < r_lcl)
            # Western Electric runs rules on the X-bar series (zone sigma = A2*r_bar/3,
            # so that ±3 zone-sigma == the X-bar control limits).
            zone_sigma = (A2 * r_bar) / 3.0 if A2 > 0 else float("nan")
            we = _western_electric(means, x_dbar, zone_sigma) if np.isfinite(zone_sigma) else ["" for _ in means]
            out_flag = x_out | r_out | np.array([bool(s) for s in we])
            n_out = int(out_flag.sum())

            tbl = pd.DataFrame({
                "subgroup": labels,
                "xbar": np.round(means, 6),
                "range": np.round(ranges, 6),
                "xbar_out": x_out,
                "range_out": r_out,
                "we_rules": we,
                "out_of_control": out_flag,
            })
            tbl.to_csv(d / "control_chart_points.csv", index=False, encoding="utf-8")
            files.append("control_chart_points.csv")

            estimates.update({
                "center_line": round(x_dbar, 6),
                "ucl": round(float(xbar_ucl), 6),
                "lcl": round(float(xbar_lcl), 6),
                "r_bar": round(r_bar, 6),
                "r_ucl": round(float(r_ucl), 6),
                "r_lcl": round(float(r_lcl), 6),
                "sigma_hat": round(sigma_hat, 6) if sigma_hat == sigma_hat else float("nan"),
                "subgroup_size": float(n_sub),
                "n_subgroups": float(len(means)),
                "n_out_of_control": float(n_out),
            })

            def _plot(plt):
                x = np.arange(len(means))
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
                ax1.plot(x, means, color="#4C72B0", lw=1.3, marker="o", ms=4, label="subgroup mean")
                ax1.axhline(x_dbar, color="#333333", lw=1.0, label=f"CL={x_dbar:.4g}")
                ax1.axhline(xbar_ucl, color="#C44E52", ls="--", lw=1.1, label=f"UCL={xbar_ucl:.4g}")
                ax1.axhline(xbar_lcl, color="#C44E52", ls="--", lw=1.1, label=f"LCL={xbar_lcl:.4g}")
                if out_flag.any():
                    ax1.scatter(x[out_flag], means[out_flag], color="red", zorder=5, s=45,
                                label="out of control")
                ax1.set_ylabel("subgroup mean")
                ax1.set_title(f"X-bar chart (n={n_sub})")
                ax1.legend(fontsize=7, loc="best")
                ax2.plot(x, ranges, color="#55A868", lw=1.3, marker="o", ms=4, label="range")
                ax2.axhline(r_bar, color="#333333", lw=1.0, label=f"CL={r_bar:.4g}")
                ax2.axhline(r_ucl, color="#C44E52", ls="--", lw=1.1, label=f"UCL={r_ucl:.4g}")
                ax2.axhline(r_lcl, color="#C44E52", ls="--", lw=1.1, label=f"LCL={r_lcl:.4g}")
                if r_out.any():
                    ax2.scatter(x[r_out], ranges[r_out], color="red", zorder=5, s=45)
                ax2.set_xlabel("subgroup index")
                ax2.set_ylabel("range")
                ax2.set_title("R chart")
                ax2.legend(fontsize=7, loc="best")

            _save_fig(d, "control_chart.png", files, _plot)

            status = "受控（无失控点）" if n_out == 0 else f"检出 {n_out} 个失控点"
            summary.append(
                f"{ctx.entry.method} 完成：X-bar & R 控制图（分组列={sub_col}，子组大小 n={n_sub}，"
                f"{len(means)} 个子组）；中心线 X̿={x_dbar:.6g}，X-bar 控制限 [{xbar_lcl:.6g}, {xbar_ucl:.6g}]"
                f"（A2={A2}）；R 图中心线 R̄={r_bar:.6g}、上限={r_ucl:.6g}（D3={D3}, D4={D4}）；"
                f"组内标准差估计 σ̂=R̄/d2={sigma_hat:.6g}（d2={d2}）。{status}（明细见 control_chart_points.csv 与图，"
                "失控点标红，附 Western Electric 判异规则 R1/R2/R3）。"
                " ⚠ 控制限是过程「自身声音」（基于过程变异估计），**不是规格限**——受控不等于合格，"
                "合格与否见过程能力分析(process_capability)。 ⚠ 控制限假定建立时过程稳定；"
                "若用含失控点的数据估限，限会被污染（应剔除可归因异常后重算）。X-bar 图对正态偏离较稳健（中心极限），"
                "但 R 图与失控规则对非正态/自相关数据敏感。"
            )
            code += [
                "import numpy as np",
                "g = df.groupby(subgroup)[measurement]",
                "means = g.mean(); ranges = g.apply(lambda s: s.max()-s.min())",
                "A2, D3, D4, d2 = 0.577, 0, 2.114, 2.326   # 例: n=5 的 Shewhart 常数",
                "x_dbar = means.mean(); r_bar = ranges.mean()",
                "xbar_ucl, xbar_lcl = x_dbar + A2*r_bar, x_dbar - A2*r_bar",
                "r_ucl, r_lcl = D4*r_bar, D3*r_bar",
                "sigma_hat = r_bar / d2                       # 组内标准差估计",
            ]
        else:
            # ---- I-MR chart (individuals + moving range) -------------------------
            vals = _num(df[meas])
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                summary.append("控制图 跳过：有效测量点过少（I-MR 至少需要 2 个点）。")
                return
            mr = np.abs(np.diff(vals))          # moving range of 2 consecutive points
            x_bar = float(vals.mean())
            mr_bar = float(mr.mean())
            sigma_hat = mr_bar / _D2_MR2        # d2(2)=1.128
            # individuals control limits = x_bar ± 3*sigma_hat = x_bar ± 2.66*MR_bar
            ind_ucl = x_bar + 3.0 * sigma_hat
            ind_lcl = x_bar - 3.0 * sigma_hat
            # MR chart limits: D4(n=2)=3.267, D3(n=2)=0 -> MR_ucl = 3.267*MR_bar
            mr_ucl = _XBAR_R_CONST[2][2] * mr_bar
            mr_lcl = _XBAR_R_CONST[2][1] * mr_bar

            ind_out = (vals > ind_ucl) | (vals < ind_lcl)
            we = _western_electric(vals, x_bar, sigma_hat) if (sigma_hat == sigma_hat and sigma_hat > 0) else ["" for _ in vals]
            # MR out-of-control aligns to the SECOND of each consecutive pair
            mr_out_full = np.zeros(vals.size, dtype=bool)
            mr_out_full[1:] = (mr > mr_ucl) | (mr < mr_lcl)
            out_flag = ind_out | mr_out_full | np.array([bool(s) for s in we])
            n_out = int(out_flag.sum())

            mr_col = np.concatenate([[np.nan], mr])
            tbl = pd.DataFrame({
                "index": np.arange(vals.size),
                "value": np.round(vals, 6),
                "moving_range": np.round(mr_col, 6),
                "individual_out": ind_out,
                "mr_out": mr_out_full,
                "we_rules": we,
                "out_of_control": out_flag,
            })
            tbl.to_csv(d / "control_chart_points.csv", index=False, encoding="utf-8")
            files.append("control_chart_points.csv")

            estimates.update({
                "center_line": round(x_bar, 6),
                "ucl": round(float(ind_ucl), 6),
                "lcl": round(float(ind_lcl), 6),
                "mr_bar": round(mr_bar, 6),
                "mr_ucl": round(float(mr_ucl), 6),
                "mr_lcl": round(float(mr_lcl), 6),
                "sigma_hat": round(float(sigma_hat), 6),
                "subgroup_size": 1.0,
                "n_subgroups": float(vals.size),
                "n_out_of_control": float(n_out),
            })

            def _plot(plt):
                x = np.arange(vals.size)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
                ax1.plot(x, vals, color="#4C72B0", lw=1.2, marker="o", ms=4, label="individual")
                ax1.axhline(x_bar, color="#333333", lw=1.0, label=f"CL={x_bar:.4g}")
                ax1.axhline(ind_ucl, color="#C44E52", ls="--", lw=1.1, label=f"UCL={ind_ucl:.4g}")
                ax1.axhline(ind_lcl, color="#C44E52", ls="--", lw=1.1, label=f"LCL={ind_lcl:.4g}")
                if out_flag.any():
                    ax1.scatter(x[out_flag], vals[out_flag], color="red", zorder=5, s=45,
                                label="out of control")
                ax1.set_ylabel("individual value")
                ax1.set_title("Individuals (I) chart")
                ax1.legend(fontsize=7, loc="best")
                ax2.plot(x[1:], mr, color="#55A868", lw=1.2, marker="o", ms=4, label="moving range")
                ax2.axhline(mr_bar, color="#333333", lw=1.0, label=f"CL={mr_bar:.4g}")
                ax2.axhline(mr_ucl, color="#C44E52", ls="--", lw=1.1, label=f"UCL={mr_ucl:.4g}")
                ax2.set_xlabel("observation index")
                ax2.set_ylabel("moving range")
                ax2.set_title("Moving-range (MR) chart")
                ax2.legend(fontsize=7, loc="best")

            _save_fig(d, "control_chart.png", files, _plot)

            status = "受控（无失控点）" if n_out == 0 else f"检出 {n_out} 个失控点"
            summary.append(
                f"{ctx.entry.method} 完成：I-MR 单值-移动极差控制图（测量列={meas}，{vals.size} 个观测，"
                f"未提供等大小子组列故用个体图）；中心线 x̄={x_bar:.6g}，控制限 = x̄ ± 2.66·MR̄ = "
                f"[{ind_lcl:.6g}, {ind_ucl:.6g}]；移动极差 MR̄={mr_bar:.6g}、MR 上限={mr_ucl:.6g}；"
                f"σ̂=MR̄/d2={sigma_hat:.6g}（d2(2)=1.128）。{status}（明细见 control_chart_points.csv 与图，失控点标红，"
                "附 Western Electric 规则 R1/R2/R3）。"
                " ⚠ 控制限是过程自身变异、**不是规格限**（合格性见 process_capability）。"
                " ⚠ I-MR 假定单值近似独立同分布；自相关或趋势会使移动极差低估σ、控制限过窄而误报。"
                "若有合理的、大小相等的子组，X-bar & R 图更灵敏（请用 config['subgroup'] 指定分组列）。"
                "控制限假定建立时过程稳定。"
            )
            code += [
                "import numpy as np",
                "vals = pd.to_numeric(df[measurement], errors='coerce').dropna().to_numpy()",
                "mr = np.abs(np.diff(vals))                  # 移动极差 (n=2)",
                "x_bar, mr_bar = vals.mean(), mr.mean()",
                "sigma_hat = mr_bar / 1.128                  # d2(2)=1.128",
                "ucl, lcl = x_bar + 2.66*mr_bar, x_bar - 2.66*mr_bar  # = x̄ ± 3σ̂",
            ]
    except Exception as exc:
        summary.append(f"控制图 计算失败：{exc}")


# ===========================================================================
# 2) process_capability — Cp / Cpk / Pp / Ppk + %-out-of-spec (ppm)
#    Refs: Montgomery SQC; AIAG SPC manual; ISO 22514.
# ===========================================================================
@register("process_capability")
def _branch_process_capability(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cfg = ctx.cfg
    meas = _measurement_col(ctx)
    if meas is None:
        summary.append("过程能力分析 跳过：找不到数值测量列（用 config['measurement'] 指定）。")
        return

    def _spec(key):
        v = cfg.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    lsl = _spec("lsl")
    usl = _spec("usl")
    target = _spec("target")
    if lsl is None and usl is None:
        summary.append("过程能力分析跳过：过程能力分析需要规格限 LSL/USL（用 --config 提供）。")
        return
    if lsl is not None and usl is not None and lsl >= usl:
        summary.append(f"过程能力分析 跳过：规格限不合理 LSL({lsl:.6g}) >= USL({usl:.6g})。")
        return
    try:
        df = ctx.df
        vals = _num(df[meas])
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            summary.append("过程能力分析 跳过：有效测量点过少（至少需要 2 个）。")
            return
        x_bar = float(vals.mean())
        sigma_overall = float(vals.std(ddof=1))  # long-term / overall SD -> Pp, Ppk
        if not np.isfinite(sigma_overall) or sigma_overall <= 0:
            summary.append("过程能力分析 跳过：测量值无变异（σ=0），能力指数无定义。")
            return

        # within-subgroup sigma (R-bar/d2) if equal-size subgroups are available; else
        # fall back to the overall SD and SAY the within/overall sigmas coincide.
        sub_col = _group_col(ctx, meas, "subgroup")
        sigma_within = sigma_overall
        within_note = "（无子组列：组内σ取整体样本标准差，二者等同）"
        if sub_col is not None:
            grp = df.groupby(sub_col, sort=False)
            sizes = grp[meas].apply(lambda s: int(np.isfinite(_num(s)).sum()))
            uniq = sorted(set(int(x) for x in sizes.tolist()))
            if len(uniq) == 1 and 2 <= uniq[0] <= 10 and len(sizes) >= 2:
                n_sub = uniq[0]
                d2 = _XBAR_R_CONST[n_sub][3]
                ranges = []
                for key in grp.groups:
                    gv = _num(grp.get_group(key)[meas])
                    gv = gv[np.isfinite(gv)]
                    if gv.size:
                        ranges.append(float(gv.max() - gv.min()))
                if ranges and d2 > 0:
                    sigma_within = float(np.mean(ranges)) / d2
                    within_note = f"（组内σ=R̄/d2，分组列={sub_col}，n={n_sub}，d2={d2}）"

        def _cp(sig):
            return (usl - lsl) / (6.0 * sig) if (lsl is not None and usl is not None) else float("nan")

        def _cpk(sig):
            sides = []
            if usl is not None:
                sides.append((usl - x_bar) / (3.0 * sig))
            if lsl is not None:
                sides.append((x_bar - lsl) / (3.0 * sig))
            return float(min(sides)) if sides else float("nan")

        cp = _cp(sigma_within)
        cpk = _cpk(sigma_within)
        pp = _cp(sigma_overall)
        ppk = _cpk(sigma_overall)

        # one-sided indices (named, for clarity in the table)
        cpu = (usl - x_bar) / (3.0 * sigma_within) if usl is not None else float("nan")
        cpl = (x_bar - lsl) / (3.0 * sigma_within) if lsl is not None else float("nan")

        # estimated % out of spec under the normal model (using overall sigma -> ppm).
        try:
            from scipy.stats import norm

            p_above = (1.0 - norm.cdf((usl - x_bar) / sigma_overall)) if usl is not None else 0.0
            p_below = norm.cdf((lsl - x_bar) / sigma_overall) if lsl is not None else 0.0
        except Exception:
            # erf-based normal CDF fallback (no scipy)
            import math

            def _ncdf(z):
                return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

            p_above = (1.0 - _ncdf((usl - x_bar) / sigma_overall)) if usl is not None else 0.0
            p_below = _ncdf((lsl - x_bar) / sigma_overall) if lsl is not None else 0.0
        frac_out = float(p_above + p_below)
        ppm_out = frac_out * 1.0e6

        pd.DataFrame({
            "index": ["Cp", "Cpk", "Pp", "Ppk", "Cpu", "Cpl", "x_bar",
                      "sigma_within", "sigma_overall", "ppm_out"],
            "value": [round(cp, 6), round(cpk, 6), round(pp, 6), round(ppk, 6),
                      round(cpu, 6), round(cpl, 6), round(x_bar, 6),
                      round(sigma_within, 6), round(sigma_overall, 6), round(ppm_out, 4)],
        }).to_csv(d / "process_capability.csv", index=False, encoding="utf-8")
        files.append("process_capability.csv")

        estimates.update({
            "cp": round(cp, 6) if cp == cp else float("nan"),
            "cpk": round(cpk, 6) if cpk == cpk else float("nan"),
            "pp": round(pp, 6) if pp == pp else float("nan"),
            "ppk": round(ppk, 6) if ppk == ppk else float("nan"),
            "cpu": round(cpu, 6) if cpu == cpu else float("nan"),
            "cpl": round(cpl, 6) if cpl == cpl else float("nan"),
            "ppm_out": round(ppm_out, 4),
            "sigma": round(sigma_within, 6),
            "sigma_overall": round(sigma_overall, 6),
            "mean": round(x_bar, 6),
            "n": float(vals.size),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8.2, 4.6))
            ax.hist(vals, bins=min(30, max(8, vals.size // 3)), density=True,
                    color="#bbbbbb", edgecolor="white", alpha=0.85, label="data")
            xs = np.linspace(vals.min(), vals.max(), 200)
            pdf = (1.0 / (sigma_overall * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((xs - x_bar) / sigma_overall) ** 2)
            ax.plot(xs, pdf, color="#4C72B0", lw=1.6, label="fitted normal")
            if lsl is not None:
                ax.axvline(lsl, color="#C44E52", ls="--", lw=1.4, label=f"LSL={lsl:.4g}")
            if usl is not None:
                ax.axvline(usl, color="#C44E52", ls="--", lw=1.4, label=f"USL={usl:.4g}")
            if target is not None:
                ax.axvline(target, color="#55A868", ls=":", lw=1.4, label=f"Target={target:.4g}")
            ax.axvline(x_bar, color="#333333", lw=1.0, label=f"mean={x_bar:.4g}")
            ax.set_xlabel("measurement")
            ax.set_ylabel("density")
            ax.set_title(f"Process capability (Cpk={cpk:.3f}, Ppk={ppk:.3f})")
            ax.legend(fontsize=7, loc="best")

        _save_fig(d, "process_capability.png", files, _plot)

        spec_txt = (f"[LSL={lsl:.6g}, USL={usl:.6g}]" if (lsl is not None and usl is not None)
                    else (f"单侧 USL={usl:.6g}" if usl is not None else f"单侧 LSL={lsl:.6g}"))
        verdict = ("能力充足 (Cpk≥1.33)" if cpk >= 1.33 else
                   ("勉强可接受 (1.0≤Cpk<1.33)" if cpk >= 1.0 else "能力不足 (Cpk<1.0)"))
        cp_txt = f"Cp={cp:.4g}、" if cp == cp else ""
        pp_txt = f"Pp={pp:.4g}、" if pp == pp else ""
        summary.append(
            f"{ctx.entry.method} 完成：测量列={meas}（{vals.size} 个观测，规格 {spec_txt}）；"
            f"x̄={x_bar:.6g}，组内σ={sigma_within:.6g}{within_note}、整体σ={sigma_overall:.6g}。"
            f"短期能力 {cp_txt}Cpk={cpk:.4g}（{verdict}）；长期表现 {pp_txt}Ppk={ppk:.4g}；"
            f"正态模型估计超规格比例≈{frac_out:.4%}（{ppm_out:.1f} ppm）。明细见 process_capability.csv 与图。"
            " ⚠ Cp/Cpk 用组内（短期）σ衡量「过程潜力」，Pp/Ppk 用整体（长期）样本σ衡量「实际表现」；"
            "Cpk 远大于 Ppk 说明子组间漂移大（过程不稳）。 ⚠ 全部指数与 ppm 估计假定数据近似正态且过程稳定；"
            "对非正态数据应先做变换或用分布专用能力指数（如基于分位数的 Cp）。规格限为工程/客户要求，"
            "与控制限是两回事（控制图判过程是否稳定，能力指数判是否满足规格）。"
        )
        code += [
            "import numpy as np",
            "x_bar = vals.mean(); sigma = vals.std(ddof=1)   # 整体σ -> Pp/Ppk",
            "cp  = (usl - lsl) / (6*sigma_within)            # 组内σ -> Cp/Cpk",
            "cpk = min((usl - x_bar)/(3*sigma_within), (x_bar - lsl)/(3*sigma_within))",
            "pp  = (usl - lsl) / (6*sigma)",
            "ppk = min((usl - x_bar)/(3*sigma), (x_bar - lsl)/(3*sigma))",
            "from scipy.stats import norm",
            "ppm = 1e6*((1-norm.cdf((usl-x_bar)/sigma)) + norm.cdf((lsl-x_bar)/sigma))",
        ]
    except Exception as exc:
        summary.append(f"过程能力分析 计算失败：{exc}")


# ===========================================================================
# 3) gage_rr — Gauge R&R (Measurement System Analysis) by the ANOVA method (AIAG)
#    Refs: AIAG MSA Reference Manual (4th ed.); Montgomery SQC Ch.8;
#          Burdick, Borror & Montgomery "Design & Analysis of Gauge R&R Studies".
# ===========================================================================
@register("gage_rr")
def _branch_gage_rr(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cfg = ctx.cfg
    meas = _measurement_col(ctx)
    if meas is None:
        summary.append("量具重复性与再现性(Gage R&R) 跳过：找不到数值测量列（用 config['measurement'] 指定）。")
        return
    try:
        df = ctx.df
        part = _group_col(ctx, meas, "part", max_card_frac=1.0)
        operator = _group_col(ctx, meas, "operator", max_card_frac=1.0)
        # part and operator must be distinct columns
        if part is not None and part == operator:
            operator = None
        if part is None or operator is None:
            summary.append(
                "量具重复性与再现性(Gage R&R) 跳过：需要零件/样本列与操作员/评价者列——"
                "用 config['part'] 与 config['operator'] 指定（经典 零件×操作员×重复 交叉设计）。"
            )
            return

        sub = df[[meas, part, operator]].copy()
        sub[meas] = _num(sub[meas])
        sub = sub.dropna(subset=[meas])
        sub[part] = sub[part].astype(str)
        sub[operator] = sub[operator].astype(str)

        p_levels = sub[part].nunique()
        o_levels = sub[operator].nunique()
        if p_levels < 2 or o_levels < 2:
            summary.append(
                f"量具重复性与再现性(Gage R&R) 跳过：需 ≥2 个零件且 ≥2 个操作员"
                f"（当前零件={p_levels}、操作员={o_levels}）。"
            )
            return

        # balanced crossed design check: every part×operator cell has the same count r>=2.
        cell = sub.groupby([part, operator], observed=True)[meas].size()
        cell_counts = sorted(set(int(x) for x in cell.tolist()))
        full_cells = p_levels * o_levels
        balanced = (len(cell_counts) == 1 and cell.size == full_cells)
        if not balanced:
            summary.append(
                "量具重复性与再现性(Gage R&R) 跳过：非清晰平衡的交叉设计——"
                f"应每个 零件×操作员 组合都有相同的重复次数（共 {full_cells} 个组合，"
                f"实测 {cell.size} 个、重复数取值 {cell_counts}）；请提供平衡数据或检查列指定。"
            )
            return
        r = cell_counts[0]  # replicates per cell
        if r < 2:
            summary.append(
                "量具重复性与再现性(Gage R&R) 跳过：每个 零件×操作员 组合只有 1 次测量，"
                "无法分离重复性（误差）方差——需要 ≥2 次重复。"
            )
            return

        n_p, n_o = p_levels, o_levels

        # ---- Two-way ANOVA with interaction via statsmodels OLS --------------
        # NOTE (STOP-AND-REPORT, design choice): AIAG ANOVA method. We fit the full
        # crossed model measurement ~ part + operator + part:operator, then derive
        # variance components from the Expected Mean Squares (EMS) for a RANDOM-effects
        # balanced two-way crossed design (parts and operators both random):
        #   E[MS_error]       = σ²_rep
        #   E[MS_interaction] = σ²_rep + r·σ²_PO
        #   E[MS_operator]    = σ²_rep + r·σ²_PO + p·r·σ²_O
        #   E[MS_part]        = σ²_rep + r·σ²_PO + o·r·σ²_P
        # solving (method-of-moments):
        #   σ²_rep = MS_error
        #   σ²_PO  = (MS_PO - MS_error) / r
        #   σ²_O   = (MS_O  - MS_PO)    / (p·r)
        #   σ²_P   = (MS_P  - MS_PO)    / (o·r)
        # INTERACTION POOLING (defensible convention): if the part:operator term is not
        # significant (we use the common AIAG threshold p > 0.25), pool it into the
        # error and refit the reduced model measurement ~ part + operator, with EMS:
        #   σ²_rep = MS_error ; σ²_O = (MS_O - MS_error)/(p·r) ; σ²_P = (MS_P - MS_error)/(o·r)
        # and σ²_PO := 0. (AIAG MSA manual; Montgomery SQC.)
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # safe, formula-identifier-friendly column names (CLAUDE.md formula guard)
        sub2 = sub.rename(columns={meas: "y", part: "PART", operator: "OPER"})
        sub2["PART"] = sub2["PART"].astype("category")
        sub2["OPER"] = sub2["OPER"].astype("category")

        full = ols("y ~ C(PART) + C(OPER) + C(PART):C(OPER)", data=sub2).fit()
        aov_full = sm.stats.anova_lm(full, typ=2)

        def _row(aov, key):
            ms = float(aov.loc[key, "sum_sq"]) / float(aov.loc[key, "df"])
            return ms, float(aov.loc[key, "df"])

        ms_part, df_part = _row(aov_full, "C(PART)")
        ms_oper, df_oper = _row(aov_full, "C(OPER)")
        ms_int, df_int = _row(aov_full, "C(PART):C(OPER)")
        ms_err, df_err = _row(aov_full, "Residual")
        # interaction p-value (F = MS_int / MS_err)
        try:
            p_int = float(aov_full.loc["C(PART):C(OPER)", "PR(>F)"])
        except Exception:
            p_int = 1.0

        pooled = p_int > 0.25 or not np.isfinite(p_int)
        neg_truncated = False
        if not pooled:
            # full model EMS solution
            var_rep = ms_err
            var_po = (ms_int - ms_err) / r
            var_oper = (ms_oper - ms_int) / (n_p * r)
            var_part = (ms_part - ms_int) / (n_o * r)
            model_note = f"全模型（含交互；交互 p={p_int:.3g}≤0.25 保留）"
        else:
            # pool interaction into error, refit reduced model y ~ part + operator
            reduced = ols("y ~ C(PART) + C(OPER)", data=sub2).fit()
            aov_red = sm.stats.anova_lm(reduced, typ=2)
            ms_part, df_part = _row(aov_red, "C(PART)")
            ms_oper, df_oper = _row(aov_red, "C(OPER)")
            ms_err, df_err = _row(aov_red, "Residual")
            var_rep = ms_err
            var_po = 0.0
            var_oper = (ms_oper - ms_err) / (n_p * r)
            var_part = (ms_part - ms_err) / (n_o * r)
            model_note = (f"简化模型（交互 p={p_int:.3g}>0.25 不显著，已并入误差/重复性）"
                          if np.isfinite(p_int) else "简化模型（交互项不可估，已并入误差）")

        # ⚠ negative variance estimates -> truncate to 0 (disclose).
        for nm, v in (("rep", var_rep), ("po", var_po), ("oper", var_oper), ("part", var_part)):
            if v < 0:
                neg_truncated = True
        var_rep = max(var_rep, 0.0)
        var_po = max(var_po, 0.0)
        var_oper = max(var_oper, 0.0)
        var_part = max(var_part, 0.0)

        # variance partition
        var_repeat = var_rep                       # repeatability (equipment / EV)
        var_reprod = var_oper + var_po             # reproducibility (operator + interaction, AV)
        var_gagerr = var_repeat + var_reprod       # total Gage R&R
        var_total = var_gagerr + var_part          # total variation

        if var_total <= 0:
            summary.append("量具重复性与再现性(Gage R&R) 跳过：总方差估计为 0，无法计算百分比。")
            return

        # %Contribution = variance component / total variance (× 100)
        pct_contrib_grr = 100.0 * var_gagerr / var_total
        pct_contrib_repeat = 100.0 * var_repeat / var_total
        pct_contrib_reprod = 100.0 * var_reprod / var_total
        pct_contrib_part = 100.0 * var_part / var_total

        # %StudyVariation = 6σ_component / 6σ_total = σ_component / σ_total (× 100)
        sd_total = float(np.sqrt(var_total))
        pct_sv_grr = 100.0 * np.sqrt(var_gagerr) / sd_total
        pct_sv_repeat = 100.0 * np.sqrt(var_repeat) / sd_total
        pct_sv_reprod = 100.0 * np.sqrt(var_reprod) / sd_total
        pct_sv_part = 100.0 * np.sqrt(var_part) / sd_total

        # number of distinct categories ndc = 1.41 * (σ_part / σ_gagerr)
        sd_gagerr = float(np.sqrt(var_gagerr))
        sd_part = float(np.sqrt(var_part))
        ndc = 1.41 * (sd_part / sd_gagerr) if sd_gagerr > 0 else float("inf")
        ndc_report = float(np.floor(ndc)) if np.isfinite(ndc) else float("inf")

        comp_tbl = pd.DataFrame({
            "source": ["Total Gage R&R", "  Repeatability (EV)", "  Reproducibility (AV)",
                       "    Operator", "    Part*Operator", "Part-to-Part (PV)", "Total Variation"],
            "variance": [round(var_gagerr, 8), round(var_repeat, 8), round(var_reprod, 8),
                         round(var_oper, 8), round(var_po, 8), round(var_part, 8), round(var_total, 8)],
            "pct_contribution": [round(pct_contrib_grr, 4), round(pct_contrib_repeat, 4),
                                 round(pct_contrib_reprod, 4), round(100.0 * var_oper / var_total, 4),
                                 round(100.0 * var_po / var_total, 4), round(pct_contrib_part, 4), 100.0],
            "pct_study_var": [round(pct_sv_grr, 4), round(pct_sv_repeat, 4), round(pct_sv_reprod, 4),
                              round(100.0 * np.sqrt(var_oper) / sd_total, 4),
                              round(100.0 * np.sqrt(var_po) / sd_total, 4),
                              round(pct_sv_part, 4), 100.0],
        })
        comp_tbl.to_csv(d / "gage_rr_components.csv", index=False, encoding="utf-8")
        files.append("gage_rr_components.csv")

        estimates.update({
            "pct_gagerr_studyvar": round(pct_sv_grr, 4),
            "pct_gagerr_contribution": round(pct_contrib_grr, 4),
            "pct_repeatability": round(pct_sv_repeat, 4),
            "pct_reproducibility": round(pct_sv_reprod, 4),
            "pct_part": round(pct_sv_part, 4),
            "ndc": float(ndc_report) if np.isfinite(ndc_report) else float("inf"),
            "var_gagerr": round(var_gagerr, 8),
            "var_part": round(var_part, 8),
            "var_repeatability": round(var_repeat, 8),
            "var_reproducibility": round(var_reprod, 8),
            "n_parts": float(n_p),
            "n_operators": float(n_o),
            "n_replicates": float(r),
        })

        def _plot(plt):
            cats = ["Gage R&R", "Repeat", "Reprod", "Part-to-Part"]
            contrib = [pct_contrib_grr, pct_contrib_repeat, pct_contrib_reprod, pct_contrib_part]
            studyvar = [pct_sv_grr, pct_sv_repeat, pct_sv_reprod, pct_sv_part]
            x = np.arange(len(cats))
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.bar(x - 0.2, contrib, width=0.4, color="#C44E52", label="% Contribution")
            ax.bar(x + 0.2, studyvar, width=0.4, color="#4C72B0", label="% Study Var")
            ax.axhline(10, color="#55A868", ls=":", lw=1.0)
            ax.axhline(30, color="#dd8452", ls=":", lw=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(cats)
            ax.set_ylabel("percent")
            ax.set_title(f"Gage R&R variance components (%GRR(SV)={pct_sv_grr:.1f}%, ndc={ndc_report:.0f})")
            ax.legend(fontsize=8)

        _save_fig(d, "gage_rr_components.png", files, _plot)

        if pct_sv_grr < 10:
            verdict = "测量系统可接受（%GRR<10%）"
        elif pct_sv_grr <= 30:
            verdict = "测量系统勉强可接受（10%≤%GRR≤30%，视应用而定）"
        else:
            verdict = "测量系统不可接受（%GRR>30%）"
        ndc_txt = (f"ndc={ndc_report:.0f}（{'≥5 可区分' if ndc_report >= 5 else '<5 区分力不足'}）"
                   if np.isfinite(ndc_report) else "ndc=∞（量具误差≈0）")
        neg_note = (" ⚠ 出现负的方差分量估计，已截断为 0（小样本/分量接近 0 时常见，提示该来源方差实际很小）。"
                    if neg_truncated else "")
        summary.append(
            f"{ctx.entry.method} 完成（ANOVA 法 / AIAG，{model_note}）：测量={meas}、零件={part}（{n_p} 个）、"
            f"操作员={operator}（{n_o} 个）、每组合重复={r} 次。方差分量：重复性(EV)={var_repeat:.6g}、"
            f"再现性(AV)={var_reprod:.6g}、零件间(PV)={var_part:.6g}、总={var_total:.6g}。"
            f"总 Gage R&R：%贡献={pct_contrib_grr:.2f}%（方差占比）、%研究变差={pct_sv_grr:.2f}%（标准差占比，6σ比）；"
            f"其中重复性 %SV={pct_sv_repeat:.2f}%、再现性 %SV={pct_sv_reprod:.2f}%、零件 %SV={pct_sv_part:.2f}%；"
            f"{ndc_txt}。判定：{verdict}。明细见 gage_rr_components.csv 与图。{neg_note}"
            " ⚠ 验收准则（AIAG）：%研究变差 <10% 优、10–30% 视情况可接受、>30% 不可接受；ndc≥5 为佳。"
            " ⚠ %Contribution（方差占比）与 %StudyVar（标准差/6σ占比）口径不同，二者都报告；前者用于和总方差直接相加，"
            "后者更贴近「过程变差被量具占用了多少」。 ⚠ 方差分量由随机效应 EMS 用矩法求解，假定平衡交叉设计、"
            "误差近似正态同方差；交互项按 p>0.25 并入误差（AIAG 常用约定）。负方差分量截断为 0。"
            "结论依赖零件应覆盖实际过程变差范围——若零件人为挑选则 PV 与 %GRR、ndc 会失真。"
        )
        code += [
            "import statsmodels.api as sm",
            "from statsmodels.formula.api import ols",
            "# 双因子交叉 ANOVA（AIAG 法）",
            "m = ols('y ~ C(PART) + C(OPER) + C(PART):C(OPER)', data=df).fit()",
            "aov = sm.stats.anova_lm(m, typ=2)",
            "MS_part, MS_oper, MS_int, MS_err = (aov['sum_sq']/aov['df']).values",
            "var_rep = MS_err                          # 重复性",
            "var_po  = max((MS_int - MS_err)/r, 0)     # 交互",
            "var_o   = max((MS_oper - MS_int)/(p*r), 0)# 操作员(再现性)",
            "var_p   = max((MS_part - MS_int)/(o*r), 0)# 零件",
            "var_grr = var_rep + var_o + var_po",
            "ndc = 1.41 * (var_p**0.5 / var_grr**0.5)  # 可区分类别数",
        ]
    except Exception as exc:
        summary.append(f"量具重复性与再现性(Gage R&R) 计算失败：{exc}")
