"""Branch handlers for the mediation_extra family — PROCESS-flavoured mediation /
moderation extensions on continuous columns, all OLS via statsmodels (no R).
Complements conditional_process (moderated_mediation / johnson_neyman) and
causal/mediation (single-mediator).

  * serial_mediation     — PROCESS model 6: two mediators in SEQUENCE,
                           X -> M1 -> M2 -> Y. Decomposes the total indirect effect
                           into X→M1→Y (a1·b1), X→M2→Y (a2·b2) and the serial
                           X→M1→M2→Y (a1·d21·b2), each with a bootstrap percentile CI.
  * parallel_mediation   — PROCESS model 4 with k PARALLEL mediators: specific
                           indirect a_i·b_i per mediator + total indirect, plus
                           pairwise contrasts between specific indirects — all with
                           bootstrap CIs.
  * moderated_moderation — PROCESS model 3: a THREE-WAY interaction X×W×Z on Y. The
                           X→Y effect is moderated by W, and that moderation itself
                           depends on Z. Reports the 3-way term + the conditional
                           effect of X (analytic SE) over the W×Z grid (mean ± SD).

Roles resolve from the continuous columns (config overridable), degrade honestly
(too few continuous cols / too few rows / constant column / missing import ->
append a Chinese "<method>跳过：<reason>" and RETURN — never crash), write CSV + PNG
(matplotlib Agg, ENGLISH plot labels), fill float `estimates`, append a Chinese
`summary` ending with ⚠ disclosures, and MUTATE ctx. See _branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_SEED = 20240607
_N_BOOT = 2000


def _resolve_continuous(ctx: Ctx, min_cont: int, label: str):
    import importlib.util

    if importlib.util.find_spec("statsmodels") is None:
        return None, f"{label}跳过：需要 statsmodels 包（未检测到）。pip install statsmodels。"
    fp = ctx.fp
    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]
    if len(cont) < min_cont:
        return None, f"{label}跳过：需要 ≥{min_cont} 个连续列，当前仅 {len(cont)} 个。"
    return cont, None


def _pick(cfg_val, cont, used):
    """Use cfg_val if a valid unused continuous col, else first unused col.
    Returns (name, auto) where auto=True means the role was NOT satisfied by an
    explicit config value (i.e. picked automatically by column order) — callers
    that disclose auto-assignment to the user need this flag."""
    if cfg_val and cfg_val in cont and cfg_val not in used:
        return cfg_val, False
    for c in cont:
        if c not in used:
            return c, True
    return None, True


def _constant_guard(sub, cols, label, summary):
    """Honest degrade for a constant (zero-variance) column: bootstrap/jackknife
    acceleration and OLS design matrices go degenerate on a constant regressor
    (singular fit or a silently-meaningless a/b path), so catch it explicitly
    before fitting rather than surfacing a raw exception or NaNs. Returns True
    (and appends a skip message) if any of `cols` is constant/near-constant.

    Checks exact-value distinctness (nunique<=1) first — pandas' two-pass std()
    formula can report a tiny NONZERO value for an exactly-constant float column
    (catastrophic cancellation), so a bare `std() == 0` check is not reliable —
    then falls back to a magnitude-scaled std threshold for near-constant cols."""
    for c in cols:
        col = sub[c]
        if col.nunique(dropna=True) <= 1:
            summary.append(f"{label}跳过：列 {c} 为常数/无方差。")
            return True
        sd = col.std(ddof=1)
        scale = float(col.abs().max())
        if sd != sd or sd <= 1e-9 * max(scale, 1.0):  # sd!=sd catches NaN
            summary.append(f"{label}跳过：列 {c} 为常数/无方差。")
            return True
    return False


def _ci(arr, lo=2.5, hi=97.5):
    import numpy as np

    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 10:
        return float("nan"), float("nan")
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def _bca_ci(boot, obs, jack, alpha=0.05):
    """Bias-corrected & accelerated (BCa) bootstrap CI (Efron 1987).

    boot : array of bootstrap replicates of the statistic.
    obs  : the observed (full-sample) statistic.
    jack : array of jackknife (leave-one-out) replicates of the SAME statistic.

    z0 = Φ⁻¹( #{boot < obs} / B )  (bias-correction; tie-aware mean rank, clamped
         away from 0/1 so the inverse-normal is finite).
    a  = Σ(θ̄_(·) − θ̂_(i))³ / (6·[Σ(θ̄_(·) − θ̂_(i))²]^1.5)  (acceleration from jackknife).
    α1 = Φ( z0 + (z0+z_lo)/(1−a(z0+z_lo)) ), α2 = Φ( z0 + (z0+z_hi)/(1−a(z0+z_hi)) )
    with z_lo=Φ⁻¹(α/2), z_hi=Φ⁻¹(1−α/2); CI = those percentiles of the bootstrap dist.

    Degrades to the plain percentile interval if the jackknife/bootstrap is
    degenerate (constant acceleration denominator or <3 distinct bootstrap values).
    """
    import numpy as np
    from scipy import stats as _sps

    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    if b.size < 10 or not np.isfinite(obs):
        return float("nan"), float("nan")

    B = b.size
    # tie-aware proportion-below (matches scipy's percentile-of-score convention)
    prop = (float(np.sum(b < obs)) + 0.5 * float(np.sum(b == obs))) / B
    prop = min(max(prop, 1.0 / (2 * B)), 1.0 - 1.0 / (2 * B))
    z0 = float(_sps.norm.ppf(prop))

    j = np.asarray(jack, dtype=float)
    j = j[np.isfinite(j)]
    sumsq = float(np.sum((j.mean() - j) ** 2)) if j.size >= 2 else 0.0
    if sumsq > 0 and j.size >= 2:
        num = float(np.sum((j.mean() - j) ** 3))
        a = num / (6.0 * sumsq ** 1.5)
    else:
        a = 0.0

    # degenerate -> just the percentile interval (still BCa-shaped fallback)
    if int(np.unique(b).size) < 3:
        return float(np.percentile(b, 100 * (alpha / 2))), float(np.percentile(b, 100 * (1 - alpha / 2)))

    z_lo = float(_sps.norm.ppf(alpha / 2.0))
    z_hi = float(_sps.norm.ppf(1.0 - alpha / 2.0))

    def _pct(zq):
        denom = 1.0 - a * (z0 + zq)
        if denom == 0:
            denom = 1e-12
        return float(_sps.norm.cdf(z0 + (z0 + zq) / denom))

    a1, a2 = _pct(z_lo), _pct(z_hi)
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)
    lo = float(np.percentile(b, 100 * a1))
    hi = float(np.percentile(b, 100 * a2))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _ols(y, X):
    """OLS of y on X (X already has a constant). Returns the fitted results."""
    import statsmodels.api as sm

    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------------
# 1. serial_mediation — PROCESS model 6 (X -> M1 -> M2 -> Y)
# ---------------------------------------------------------------------------
@register("serial_mediation")
def _branch_serial_mediation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont, prob = _resolve_continuous(ctx, 4, "序列中介分析")
    if prob:
        summary.append(prob)
        return
    used: list[str] = []
    y, y_auto = _pick(cfg.get("y"), cont, used); used.append(y)
    x, x_auto = _pick(cfg.get("x"), cont, used); used.append(x)
    m1, m1_auto = _pick(cfg.get("m1"), cont, used); used.append(m1)
    m2, m2_auto = _pick(cfg.get("m2"), cont, used); used.append(m2)
    if None in (y, x, m1, m2) or len({y, x, m1, m2}) < 4:
        summary.append("序列中介分析跳过：需 4 个不同连续列 X/M1/M2/Y。"
                        'config={"x":..,"m1":..,"m2":..,"y":..} 指定。')
        return

    sub = df[[y, x, m1, m2]].dropna()
    try:
        sub = sub.astype(float)
    except (TypeError, ValueError):
        summary.append("序列中介分析跳过：所选列存在非数值。")
        return
    n = len(sub)
    if n < 30:
        summary.append(f"序列中介分析跳过：有效行 {n} < 30，bootstrap 不可靠。")
        return
    if _constant_guard(sub, (y, x, m1, m2), "序列中介分析", summary):
        return

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        def _paths(data):
            xv = data[x].to_numpy(float)
            m1v = data[m1].to_numpy(float)
            m2v = data[m2].to_numpy(float)
            yv = data[y].to_numpy(float)
            # M1 ~ X
            r1 = _ols(m1v, sm.add_constant(xv))
            a1 = float(r1.params[1])
            # M2 ~ X + M1
            r2 = _ols(m2v, sm.add_constant(np.column_stack([xv, m1v])))
            a2, d21 = float(r2.params[1]), float(r2.params[2])
            # Y ~ X + M1 + M2
            r3 = _ols(yv, sm.add_constant(np.column_stack([xv, m1v, m2v])))
            cprime, b1, b2 = float(r3.params[1]), float(r3.params[2]), float(r3.params[3])
            return a1, a2, d21, b1, b2, cprime

        a1, a2, d21, b1, b2, cprime = _paths(sub)
        ind_m1 = a1 * b1
        ind_m2 = a2 * b2
        ind_serial = a1 * d21 * b2
        total_ind = ind_m1 + ind_m2 + ind_serial
        total_eff = cprime + total_ind

        rng = np.random.default_rng(_SEED)
        idx = np.arange(n)
        bm1 = np.empty(_N_BOOT); bm2 = np.empty(_N_BOOT)
        bser = np.empty(_N_BOOT); btot = np.empty(_N_BOOT)
        for i in range(_N_BOOT):
            bs = sub.iloc[rng.choice(idx, n, replace=True)]
            try:
                qa1, qa2, qd21, qb1, qb2, _ = _paths(bs)
            except Exception:
                bm1[i] = bm2[i] = bser[i] = btot[i] = np.nan
                continue
            bm1[i] = qa1 * qb1
            bm2[i] = qa2 * qb2
            bser[i] = qa1 * qd21 * qb2
            btot[i] = bm1[i] + bm2[i] + bser[i]

        # jackknife (leave-one-out) of the SAME four indirect statistics, for BCa
        # acceleration. n refits — fine for the test sizes (see CLAUDE.md note).
        jm1 = np.full(n, np.nan); jm2 = np.full(n, np.nan)
        jser = np.full(n, np.nan); jtot = np.full(n, np.nan)
        for i in range(n):
            loo = sub.drop(sub.index[i])
            try:
                ja1, ja2, jd21, jb1, jb2, _ = _paths(loo)
            except Exception:
                continue
            jm1[i] = ja1 * jb1
            jm2[i] = ja2 * jb2
            jser[i] = ja1 * jd21 * jb2
            jtot[i] = jm1[i] + jm2[i] + jser[i]

        ci_m1 = _bca_ci(bm1, ind_m1, jm1)
        ci_m2 = _bca_ci(bm2, ind_m2, jm2)
        ci_ser = _bca_ci(bser, ind_serial, jser)
        ci_tot = _bca_ci(btot, total_ind, jtot)

        def _sig(ci):
            return "显著" if (np.isfinite(ci[0]) and (ci[0] > 0 or ci[1] < 0)) else "不显著"

        estimates.update({
            "a1": round(a1, 6), "a2": round(a2, 6), "d21": round(d21, 6),
            "b1": round(b1, 6), "b2": round(b2, 6), "direct_effect": round(cprime, 6),
            "indirect_via_m1": round(ind_m1, 6),
            "indirect_via_m1_lo": round(ci_m1[0], 6), "indirect_via_m1_hi": round(ci_m1[1], 6),
            "indirect_via_m2": round(ind_m2, 6),
            "indirect_via_m2_lo": round(ci_m2[0], 6), "indirect_via_m2_hi": round(ci_m2[1], 6),
            "indirect_serial": round(ind_serial, 6),
            "indirect_serial_lo": round(ci_ser[0], 6), "indirect_serial_hi": round(ci_ser[1], 6),
            "total_indirect": round(total_ind, 6),
            "total_indirect_lo": round(ci_tot[0], 6), "total_indirect_hi": round(ci_tot[1], 6),
            "total_effect": round(total_eff, 6), "n": float(n),
        })

        tab = pd.DataFrame([
            {"path": f"{x}->{m1}->{y} (a1·b1)", "effect": ind_m1, "ci_lo": ci_m1[0], "ci_hi": ci_m1[1]},
            {"path": f"{x}->{m2}->{y} (a2·b2)", "effect": ind_m2, "ci_lo": ci_m2[0], "ci_hi": ci_m2[1]},
            {"path": f"{x}->{m1}->{m2}->{y} (a1·d21·b2)", "effect": ind_serial,
             "ci_lo": ci_ser[0], "ci_hi": ci_ser[1]},
            {"path": "total indirect", "effect": total_ind, "ci_lo": ci_tot[0], "ci_hi": ci_tot[1]},
            {"path": f"direct {x}->{y} (c')", "effect": cprime, "ci_lo": float("nan"), "ci_hi": float("nan")},
        ])
        tab.to_csv(d / "serial_mediation_effects.csv", index=False, encoding="utf-8")
        files.append("serial_mediation_effects.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = ["via M1", "via M2", "serial", "total ind."]
            pts = [ind_m1, ind_m2, ind_serial, total_ind]
            los = [ci_m1[0], ci_m2[0], ci_ser[0], ci_tot[0]]
            his = [ci_m1[1], ci_m2[1], ci_ser[1], ci_tot[1]]
            err = [[p - lo for p, lo in zip(pts, los)], [hi - p for p, hi in zip(pts, his)]]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(4), pts, color="#4C72B0")
            ax.errorbar(range(4), pts, yerr=np.abs(err), fmt="none", ecolor="black", capsize=4)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(range(4)); ax.set_xticklabels(labels)
            ax.set_ylabel("indirect effect (95% boot CI)")
            ax.set_title("Serial mediation (model 6)")
            fig.tight_layout()
            fig.savefig(d / "serial_mediation.png", dpi=150)
            plt.close(fig)
            files.append("serial_mediation.png")
        except Exception:
            pass

        code += [
            "import numpy as np, statsmodels.api as sm",
            "# M1~X; M2~X+M1; Y~X+M1+M2 ; serial indirect = a1*d21*b2",
            "# CI = bootstrap BCa (z0 from #{boot<obs}/B; accel a from jackknife LOO refits)",
        ]
        auto_note = ""
        if any([y_auto, x_auto, m1_auto, m2_auto]):
            auto_note = (
                f"⚠ 角色按列序自动指派（首连续列=Y，其后依次为 X、M1、M2）：Y={y}、X={x}、"
                f"M1={m1}、M2={m2}——**顺序很重要**（换一组指派就是另一模型），"
                'config={"x":..,"m1":..,"m2":..,"y":..} 可核对你的理论路径。 '
            )
        summary.append(
            f"{entry.method}（PROCESS model 6）：X={x} → M1={m1} → M2={m2} → Y={y}（n={n}）。"
            f"特定间接：经 M1 (a1·b1)={ind_m1:.4f}（{_sig(ci_m1)}，CI[{ci_m1[0]:.4f},{ci_m1[1]:.4f}]）；"
            f"经 M2 (a2·b2)={ind_m2:.4f}（{_sig(ci_m2)}）；**序列 (a1·d21·b2)={ind_serial:.4f}**"
            f"（{_sig(ci_ser)}，CI[{ci_ser[0]:.4f},{ci_ser[1]:.4f}]）。总间接={total_ind:.4f}"
            f"（{_sig(ci_tot)}），直接 c'={cprime:.4f}。bootstrap BCa 95% CI（B={_N_BOOT}、seed={_SEED}、"
            "jackknife 加速度校正）。" + auto_note +
            " ⚠ 中介=**相关性分解非因果证明**，需 X 时序先于 M、M 先于 Y 且无未测混杂（强假定）；"
            "序列方向 M1→M2 由你设定，反向需重设 config；CI 为偏差校正+加速度(BCa)的 bootstrap 95% CI，"
            "不含 0 即显著（优于朴素百分位法）。"
        )
    except Exception as e:
        summary.append(f"序列中介分析失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 2. parallel_mediation — PROCESS model 4 with k parallel mediators
# ---------------------------------------------------------------------------
@register("parallel_mediation")
def _branch_parallel_mediation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont, prob = _resolve_continuous(ctx, 3, "并行中介分析")
    if prob:
        summary.append(prob)
        return
    used: list[str] = []
    y, y_auto = _pick(cfg.get("y"), cont, used); used.append(y)
    x, x_auto = _pick(cfg.get("x"), cont, used); used.append(x)
    cfg_meds = cfg.get("mediators")
    meds_auto = not isinstance(cfg_meds, (list, tuple))
    if isinstance(cfg_meds, (list, tuple)):
        meds = [m for m in cfg_meds if m in cont and m not in used]
    else:
        meds = [c for c in cont if c not in used]
    if y is None or x is None or len(meds) < 1:
        summary.append("并行中介分析跳过：需 Y + X + ≥1 个中介（连续列）。"
                       'config={"x":..,"y":..,"mediators":[..]} 指定。')
        return

    sub = df[[y, x] + meds].dropna()
    try:
        sub = sub.astype(float)
    except (TypeError, ValueError):
        summary.append("并行中介分析跳过：所选列存在非数值。")
        return
    n, k = len(sub), len(meds)
    if n < 30:
        summary.append(f"并行中介分析跳过：有效行 {n} < 30，bootstrap 不可靠。")
        return
    if _constant_guard(sub, [y, x] + meds, "并行中介分析", summary):
        return

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        def _paths(data):
            xv = data[x].to_numpy(float)
            mvs = [data[m].to_numpy(float) for m in meds]
            a = [float(_ols(mv, sm.add_constant(xv)).params[1]) for mv in mvs]
            yv = data[y].to_numpy(float)
            ry = _ols(yv, sm.add_constant(np.column_stack([xv] + mvs)))
            cprime = float(ry.params[1])
            b = [float(ry.params[2 + j]) for j in range(k)]
            return np.array(a), np.array(b), cprime

        a, b, cprime = _paths(sub)
        spec = a * b
        total_ind = float(spec.sum())
        total_eff = cprime + total_ind

        rng = np.random.default_rng(_SEED)
        idx = np.arange(n)
        bspec = np.full((_N_BOOT, k), np.nan)
        btot = np.full(_N_BOOT, np.nan)
        for i in range(_N_BOOT):
            bs = sub.iloc[rng.choice(idx, n, replace=True)]
            try:
                qa, qb, _ = _paths(bs)
            except Exception:
                continue
            bspec[i] = qa * qb
            btot[i] = float((qa * qb).sum())

        # jackknife (leave-one-out) of the specific indirects + total, for BCa
        # acceleration. n refits — fine for the test sizes (see CLAUDE.md note).
        jspec = np.full((n, k), np.nan)
        jtot = np.full(n, np.nan)
        for i in range(n):
            loo = sub.drop(sub.index[i])
            try:
                ja, jb, _ = _paths(loo)
            except Exception:
                continue
            jspec[i] = ja * jb
            jtot[i] = float((ja * jb).sum())

        def _sig(lo, hi):
            return "显著" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else "不显著"

        rows = []
        for j, m in enumerate(meds):
            lo, hi = _bca_ci(bspec[:, j], float(spec[j]), jspec[:, j])
            estimates.update({
                f"a_{m}": round(float(a[j]), 6), f"b_{m}": round(float(b[j]), 6),
                f"indirect_{m}": round(float(spec[j]), 6),
                f"indirect_{m}_lo": round(lo, 6), f"indirect_{m}_hi": round(hi, 6),
            })
            rows.append({"mediator": m, "a": a[j], "b": b[j], "indirect": spec[j],
                         "ci_lo": lo, "ci_hi": hi, "sig": _sig(lo, hi)})
        ci_tot = _bca_ci(btot, total_ind, jtot)
        estimates.update({
            "total_indirect": round(total_ind, 6),
            "total_indirect_lo": round(ci_tot[0], 6), "total_indirect_hi": round(ci_tot[1], 6),
            "direct_effect": round(cprime, 6), "total_effect": round(total_eff, 6),
            "n": float(n), "n_mediators": float(k),
        })
        pd.DataFrame(rows).to_csv(d / "parallel_mediation_effects.csv", index=False, encoding="utf-8")
        files.append("parallel_mediation_effects.csv")

        # pairwise contrasts between specific indirects (which mediator transmits more)
        contrasts = []
        for j1 in range(k):
            for j2 in range(j1 + 1, k):
                diff = float(spec[j1] - spec[j2])
                clo, chi = _bca_ci(bspec[:, j1] - bspec[:, j2], diff,
                                   jspec[:, j1] - jspec[:, j2])
                contrasts.append({"contrast": f"{meds[j1]} - {meds[j2]}", "diff": diff,
                                  "ci_lo": clo, "ci_hi": chi, "sig": _sig(clo, chi)})
        if contrasts:
            pd.DataFrame(contrasts).to_csv(d / "parallel_mediation_contrasts.csv",
                                           index=False, encoding="utf-8")
            files.append("parallel_mediation_contrasts.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            los = [r["ci_lo"] for r in rows]; his = [r["ci_hi"] for r in rows]
            pts = [r["indirect"] for r in rows]
            err = [[p - lo for p, lo in zip(pts, los)], [hi - p for p, hi in zip(pts, his)]]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(k), pts, color="#55A868")
            ax.errorbar(range(k), pts, yerr=np.abs(err), fmt="none", ecolor="black", capsize=4)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(range(k)); ax.set_xticklabels(meds, rotation=30, ha="right")
            ax.set_ylabel("specific indirect (95% boot CI)")
            ax.set_title("Parallel mediation (model 4)")
            fig.tight_layout()
            fig.savefig(d / "parallel_mediation.png", dpi=150)
            plt.close(fig)
            files.append("parallel_mediation.png")
        except Exception:
            pass

        code += [
            "import numpy as np, statsmodels.api as sm",
            "# each Mi~X (a_i); Y~X+M1+..+Mk (b_i, c'); specific indirect_i = a_i*b_i",
            "# CI = bootstrap BCa (z0 from #{boot<obs}/B; accel a from jackknife LOO refits)",
        ]
        sig_meds = [r["mediator"] for r in rows if r["sig"] == "显著"]
        auto_note = ""
        if any([y_auto, x_auto, meds_auto]):
            auto_note = (
                f"⚠ 角色按列序自动指派（首连续列=Y，其后为 X，其余连续列为并行中介）：Y={y}、X={x}、"
                f"中介={'、'.join(meds)}——**顺序很重要**，"
                'config={"x":..,"y":..,"mediators":[..]} 可核对你的理论路径。 '
            )
        summary.append(
            f"{entry.method}（PROCESS model 4，{k} 个并行中介）：X={x} → Y={y}，中介={'、'.join(meds)}（n={n}）。"
            f"总间接={total_ind:.4f}（{_sig(ci_tot[0], ci_tot[1])}，CI[{ci_tot[0]:.4f},{ci_tot[1]:.4f}]），"
            f"直接 c'={cprime:.4f}。显著的特定中介：{('、'.join(sig_meds)) if sig_meds else '无'}。"
            f"成对对比见 contrasts.csv。bootstrap BCa 95% CI（B={_N_BOOT}、seed={_SEED}、jackknife 加速度校正）。"
            + auto_note +
            " ⚠ 中介=相关分解非因果；并行中介**相互控制**（每个 b_i 已偏其余中介），"
            "故并行结果与各自单中介模型可不同；需无未测混杂、X 先于 M 先于 Y；"
            "CI 为偏差校正+加速度(BCa)的 bootstrap 95% CI，不含 0 即显著。"
        )
    except Exception as e:
        summary.append(f"并行中介分析失败：{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 3. moderated_moderation — PROCESS model 3 (three-way X×W×Z on Y)
# ---------------------------------------------------------------------------
@register("moderated_moderation")
def _branch_moderated_moderation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    cont, prob = _resolve_continuous(ctx, 4, "调节的调节分析")
    if prob:
        summary.append(prob)
        return
    used: list[str] = []
    y, _ = _pick(cfg.get("y"), cont, used); used.append(y)
    x, _ = _pick(cfg.get("x"), cont, used); used.append(x)
    w, _ = _pick(cfg.get("w"), cont, used); used.append(w)
    z, _ = _pick(cfg.get("z"), cont, used); used.append(z)
    if None in (y, x, w, z) or len({y, x, w, z}) < 4:
        summary.append("调节的调节分析跳过：需 4 个不同连续列 X/W/Z/Y。"
                       'config={"x":..,"w":..,"z":..,"y":..} 指定。')
        return

    sub = df[[y, x, w, z]].dropna()
    try:
        sub = sub.astype(float)
    except (TypeError, ValueError):
        summary.append("调节的调节分析跳过：所选列存在非数值。")
        return
    n = len(sub)
    if n < 20:
        summary.append(f"调节的调节分析跳过：有效行 {n} < 20，自由度不足。")
        return

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        # mean-center X, W, Z so lower-order terms are interpretable at the means
        xc = sub[x].to_numpy(float) - sub[x].mean()
        wc = sub[w].to_numpy(float) - sub[w].mean()
        zc = sub[z].to_numpy(float) - sub[z].mean()
        yv = sub[y].to_numpy(float)
        # design: const, X, W, Z, XW, XZ, WZ, XWZ  (fixed column order)
        cols = ["X", "W", "Z", "XW", "XZ", "WZ", "XWZ"]
        D = np.column_stack([xc, wc, zc, xc * wc, xc * zc, wc * zc, xc * wc * zc])
        Xd = sm.add_constant(D)
        res = _ols(yv, Xd)
        params = np.asarray(res.params, dtype=float)   # [const, X, W, Z, XW, XZ, WZ, XWZ]
        cov = np.asarray(res.cov_params(), dtype=float)
        # index map within params/cov (0=const)
        iX, iXW, iXZ, iXWZ = 1, 4, 5, 7
        b_xwz = float(params[iXWZ])
        p_xwz = float(res.pvalues[iXWZ])

        w_sd, z_sd = float(sub[w].std()), float(sub[z].std())
        df_resid = float(res.df_resid)
        from scipy import stats

        # conditional effect of X: theta = bX + bXW*W + bXZ*Z + bXWZ*W*Z (W,Z centered)
        def _theta(wv, zv):
            L = np.zeros(len(params))
            L[iX] = 1.0; L[iXW] = wv; L[iXZ] = zv; L[iXWZ] = wv * zv
            est = float(L @ params)
            se = float(np.sqrt(L @ cov @ L))
            t = est / se if se > 0 else float("nan")
            p = float(2 * stats.t.sf(abs(t), df_resid)) if np.isfinite(t) else float("nan")
            return est, se, p

        grid_rows = []
        for wl, wv in [("W-1SD", -w_sd), ("W mean", 0.0), ("W+1SD", w_sd)]:
            for zl, zv in [("Z-1SD", -z_sd), ("Z mean", 0.0), ("Z+1SD", z_sd)]:
                est, se, p = _theta(wv, zv)
                grid_rows.append({"W": wl, "Z": zl, "X_effect": est, "se": se, "p": p,
                                  "sig": "显著" if (np.isfinite(p) and p < 0.05) else "不显著"})

        estimates.update({
            "b_three_way_XWZ": round(b_xwz, 6), "p_three_way_XWZ": round(p_xwz, 6),
            "r_squared": round(float(res.rsquared), 6),
            "b_XW": round(float(params[iXW]), 6), "b_XZ": round(float(params[iXZ]), 6),
            "n": float(n),
        })
        # corner conditional slopes into estimates for quick access
        for r in grid_rows:
            if r["W"] in ("W-1SD", "W+1SD") and r["Z"] in ("Z-1SD", "Z+1SD"):
                key = f"X_effect_{r['W'].replace(' ', '')}_{r['Z'].replace(' ', '')}"
                estimates[key] = round(float(r["X_effect"]), 6)

        pd.DataFrame(grid_rows).to_csv(d / "conditional_x_effects.csv", index=False, encoding="utf-8")
        files.append("conditional_x_effects.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            wvals = np.array([-w_sd, 0.0, w_sd])
            fig, ax = plt.subplots(figsize=(6, 4))
            for zl, zv, c in [("Z-1SD", -z_sd, "C0"), ("Z mean", 0.0, "C1"), ("Z+1SD", z_sd, "C2")]:
                th = [_theta(wv, zv)[0] for wv in wvals]
                ax.plot(["W-1SD", "W mean", "W+1SD"], th, marker="o", label=zl, color=c)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel(f"conditional effect of {x} on {y}")
            ax.set_title("Moderated moderation (model 3): X effect by W, Z")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "moderated_moderation.png", dpi=150)
            plt.close(fig)
            files.append("moderated_moderation.png")
        except Exception:
            pass

        code += [
            "import numpy as np, statsmodels.api as sm",
            "# Y ~ X + W + Z + XW + XZ + WZ + XWZ (X,W,Z mean-centered);",
            "# conditional X effect theta = bX + bXW*W + bXZ*Z + bXWZ*W*Z, SE via L'cov L",
        ]
        sig = "显著" if p_xwz < 0.05 else "不显著"
        summary.append(
            f"{entry.method}（PROCESS model 3）：Y={y}，X={x}，调节变量 W={w}、Z={z}（n={n}，均中心化）。"
            f"**三阶交互 X×W×Z 系数={b_xwz:.4f}, p={p_xwz:.4g}（{sig}）**，R²={res.rsquared:.3f}。"
            f"X 对 Y 的条件效应随 (W,Z) 变化见 conditional_x_effects.csv（mean±SD 网格，含解析 SE/p）。"
            " ⚠ 显著的三阶交互表示「W 对 X→Y 的调节作用本身又被 Z 调节」；W/Z 已中心化故低阶项"
            "为均值处效应。条件效应 SE 由 L'·cov·L 解析得出（非 bootstrap）；交互模型对异常值敏感、"
            "需足够样本支撑高阶项；这是关联性调节非因果。"
        )
    except Exception as e:
        summary.append(f"调节的调节分析失败：{type(e).__name__}: {e}")
