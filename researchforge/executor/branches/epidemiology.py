"""Branch handlers for the EPIDEMIOLOGY family — a clinical-biostatistics triangle:

  - diagnostic_test_eval   discrimination (ROC / AUC / sensitivity-specificity / LR)
  - epi_risk_measures      association (RR / OR / RD / NNT / attributable risk)
  - calibration_assessment calibration (Brier / Hosmer-Lemeshow / calibration curve)

Pure Python (numpy / scipy / pandas / matplotlib, Agg + English plot labels). Every
handler honestly degrades (Chinese "<方法>跳过：<原因>" + return) when its preconditions
are not met, and never fabricates numbers. See executor/_branch_api.py for Ctx/register.
"""

from __future__ import annotations

import math

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome
from researchforge.profiler.roles import is_treatment_named


def _resolve_exposure(fp, bin_cols):
    """Exposure ≈ TREATMENT (the reverse of an outcome): prefer a treatment-named binary
    (exposed/treated/arm…), else the first binary that is NOT the role-detected outcome,
    else the first binary. Mirrors resolve_outcome but with inverted preference — an epi
    exposure is the intervention, not the disease."""
    if not bin_cols:
        return None
    lo = getattr(fp, "likely_outcome", None)
    treat = [c for c in bin_cols if is_treatment_named(c)]
    if treat:
        return treat[0]
    non_out = [c for c in bin_cols if c != lo]
    return non_out[0] if non_out else bin_cols[0]


# --------------------------------------------------------------------------- #
# shared small helpers (module-local; epidemiology proportions/CIs)
# --------------------------------------------------------------------------- #
def _binary_cols(ctx: Ctx) -> list[str]:
    """Column names that are usable as a binary variable: profiled kind == 'binary',
    OR any numeric column with exactly 2 distinct non-null values (the profiler can
    label a 0/1 column 'count' or 'id'). Excludes panel unit/time columns."""
    import pandas as pd

    df, fp = ctx.df, ctx.fp
    excl = {fp.unit_col, fp.time_col}
    out: list[str] = []
    for c in fp.columns:
        if c.name in excl:
            continue
        if c.kind == "binary":
            out.append(c.name)
            continue
        # tolerant fallback: numeric column with exactly 2 distinct non-null values
        s = pd.to_numeric(df[c.name], errors="coerce").dropna()
        if len(s) and s.nunique() == 2:
            out.append(c.name)
    return out


def _continuous_cols(ctx: Ctx) -> list[str]:
    """Continuous-score candidate columns (kind == 'continuous'), excluding unit/time."""
    fp = ctx.fp
    excl = {fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]


def _as_binary01(s) -> "tuple":
    """Coerce a 2-level column to a 0/1 numpy array (higher level -> 1, the 'positive'
    / case / exposed level). Returns (arr01, pos_label, neg_label). NaN-dropped upstream."""
    import pandas as pd

    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().all():  # numeric: positive = the larger value
        vals = sorted(pd.unique(sn.dropna()))
        pos, neg = vals[-1], vals[0]
        return (sn.to_numpy() == pos).astype(float), pos, neg
    # non-numeric labels: sort lexicographically, treat the LAST as positive
    vals = sorted(pd.unique(s.dropna()).tolist(), key=str)
    pos, neg = vals[-1], vals[0]
    return (s.to_numpy() == pos).astype(float), pos, neg


def _fmt_level(v) -> str:
    """Human-readable rendering of a resolved 0/1 level for disclosure text (e.g. a
    numpy scalar 1.0 -> '1'; a string label is passed through unchanged). Guards
    against a silent positive/negative inversion on string-labeled columns being
    invisible to the user — see _as_binary01."""
    try:
        fv = float(v)
        if fv == int(fv):
            return str(int(fv))
    except (TypeError, ValueError, OverflowError):
        pass
    return str(v)


def _wilson_ci(k: float, n: float, z: float = 1.959963984540054) -> "tuple":
    """Wilson score 95% CI for a binomial proportion k/n (better than Wald for small n
    / extreme proportions). Returns (phat, lo, hi); NaNs if n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return (phat, max(0.0, centre - half), min(1.0, centre + half))


def _z_for(ci: float) -> float:
    """Two-sided normal quantile for a CI level (default 0.95 -> 1.96)."""
    from scipy import stats

    lvl = ci if (ci and 0 < ci < 1) else 0.95
    return float(stats.norm.ppf(0.5 + lvl / 2.0))


# =========================================================================== #
# 1) diagnostic_test_eval — diagnostic accuracy & ROC
# =========================================================================== #
@register("diagnostic_test_eval")
def _branch_diagnostic_test_eval(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    bin_cols = _binary_cols(ctx)
    cont_cols = _continuous_cols(ctx)
    ci_level = cfg.get("ci")
    try:
        ci_level = float(ci_level) if ci_level is not None else 0.95
    except (TypeError, ValueError):
        ci_level = 0.95
    z = _z_for(ci_level)

    # --- resolve truth (a binary gold-standard column) ---
    # truth is OUTCOME-semantics (the disease / gold standard) → the shared resolver picks a
    # high-confidence detected outcome and skips a treatment-named binary (an 'exposed' column
    # is the test-condition, not the truth), instead of grabbing whichever binary comes first.
    truth = cfg.get("truth") or cfg.get("outcome")
    if truth not in bin_cols:
        truth = resolve_outcome(fp, {}, bin_cols) if bin_cols else None
    if truth is None:
        summary.append("诊断试验评价跳过：需要一个二值金标准/疾病列（真值）。config 用 truth 指定。")
        return

    # --- resolve test: prefer a continuous score; else a second binary column ---
    # An explicitly-configured test/score column is honoured regardless of the
    # profiler's kind label (the 'id'/'count' trap): numeric with >2 distinct values
    # -> treated as a continuous score; with exactly 2 -> binary.
    test = cfg.get("test") or cfg.get("score")
    test_is_cont = False
    if test is not None and test in df.columns and test != truth:
        if test in cont_cols:
            test_is_cont = True
        elif test in bin_cols:
            test_is_cont = False
        else:
            s = pd.to_numeric(df[test], errors="coerce").dropna()
            if len(s) and s.nunique() > 2:
                test_is_cont = True  # numeric multi-valued -> score
            elif len(s) and s.nunique() == 2:
                test_is_cont = False
            else:
                test = None  # configured but unusable -> fall through to auto-detect
    else:
        test = None
    if test is None:
        cont_left = [c for c in cont_cols if c != truth]
        if cont_left:
            test, test_is_cont = cont_left[0], True
        else:
            bin_left = [c for c in bin_cols if c != truth]
            test, test_is_cont = (bin_left[0] if bin_left else None), False
    if test is None:
        summary.append("诊断试验评价跳过：需要一个检验列（连续评分/概率，或第二个二值结果）。config 用 test 指定。")
        return

    sub = df[[truth, test]].dropna()
    if sub.shape[0] < 4:
        summary.append("诊断试验评价跳过：成对删除缺失后样本不足 4 行。")
        return
    y, pos_y, neg_y = _as_binary01(sub[truth])
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        summary.append(f"诊断试验评价跳过：金标准 {truth} 只有一个类别（病例 {n_pos}，非病例 {n_neg}）。")
        return

    prevalence = n_pos / len(y)

    def _metrics_from_2x2(tp, fp_, fn, tn):
        """sens/spec/ppv/npv/lr+/lr-/accuracy with Wilson CIs for each proportion."""
        sens = _wilson_ci(tp, tp + fn, z)
        spec = _wilson_ci(tn, tn + fp_, z)
        ppv = _wilson_ci(tp, tp + fp_, z)
        npv = _wilson_ci(tn, tn + fn, z)
        acc = _wilson_ci(tp + tn, tp + fp_ + fn + tn, z)
        se_, sp_ = sens[0], spec[0]
        lr_plus = se_ / (1 - sp_) if (1 - sp_) > 0 else float("inf")
        lr_minus = (1 - se_) / sp_ if sp_ > 0 else float("inf")
        return sens, spec, ppv, npv, acc, lr_plus, lr_minus

    if test_is_cont:
        scores = pd.to_numeric(sub[test], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(scores)
        scores, yv = scores[ok], y[ok]
        n_pos = int(yv.sum())
        n_neg = int(len(yv) - n_pos)
        if n_pos == 0 or n_neg == 0:
            summary.append("诊断试验评价跳过：连续评分有效行不足以构成两类。")
            return

        # --- empirical ROC over all candidate thresholds + AUC (rank/Mann-Whitney) ---
        order = np.argsort(-scores, kind="mergesort")  # descending score
        s_sorted = scores[order]
        y_sorted = yv[order]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        # keep one ROC point per distinct score (right edge of each tie block)
        distinct = np.r_[np.diff(s_sorted) != 0, True]
        tpr = np.r_[0.0, tps[distinct] / n_pos]
        fpr = np.r_[0.0, fps[distinct] / n_neg]
        thr = np.r_[np.inf, s_sorted[distinct]]
        auc = float(getattr(np, "trapezoid", np.trapz)(tpr, fpr))
        # orient so AUC >= 0.5 (a test that runs the "wrong way" still gets a fair read)
        flipped = False
        if auc < 0.5:
            scores = -scores
            auc = 1.0 - auc
            flipped = True
            order = np.argsort(-scores, kind="mergesort")
            s_sorted = scores[order]
            y_sorted = yv[order]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            distinct = np.r_[np.diff(s_sorted) != 0, True]
            tpr = np.r_[0.0, tps[distinct] / n_pos]
            fpr = np.r_[0.0, fps[distinct] / n_neg]
            thr = np.r_[np.inf, s_sorted[distinct]]

        # --- AUC 95% CI: DeLong (preferred) with Hanley-McNeil honest fallback ---
        auc_lo, auc_hi, ci_method = _auc_ci_delong(scores, yv, auc, z)

        # --- Youden's J optimal cutoff: max (sens + spec - 1) = max (tpr - fpr) ---
        j = tpr - fpr
        # ignore the (0,0) endpoint at index 0 (threshold = +inf, classifies all as neg)
        j_idx = int(np.argmax(j[1:])) + 1
        cutoff = float(thr[j_idx])
        # apply cutoff on the (possibly flipped) score: predict positive if score >= cutoff
        pred_pos = scores >= cutoff
        tp = float(((pred_pos) & (yv == 1)).sum())
        fp_ = float(((pred_pos) & (yv == 0)).sum())
        fn = float(((~pred_pos) & (yv == 1)).sum())
        tn = float(((~pred_pos) & (yv == 0)).sum())
        sens, spec, ppv, npv, acc, lr_plus, lr_minus = _metrics_from_2x2(tp, fp_, fn, tn)
        # report the cutoff on the ORIGINAL score scale if we flipped
        cutoff_report = -cutoff if flipped else cutoff

        roc_df = pd.DataFrame({"threshold": thr, "fpr_1_minus_spec": fpr, "tpr_sensitivity": tpr})
        roc_df.to_csv(d / "roc_curve.csv", index=False, encoding="utf-8")
        files.append("roc_curve.csv")
        _roc_plot(fpr, tpr, fpr[j_idx], tpr[j_idx], auc, d / "roc_curve.png")
        if (d / "roc_curve.png").exists():
            files.append("roc_curve.png")

        estimates["auc"] = round(auc, 6)
        estimates["auc_ci_low"] = round(auc_lo, 6)
        estimates["auc_ci_high"] = round(auc_hi, 6)
        estimates["youden_cutoff"] = round(cutoff_report, 6)
        flip_note = (
            f"（注：评分与疾病负相关，已翻转方向使 AUC≥0.5；切点 {cutoff_report:.4g} 为原始尺度，"
            "判定规则反向：原始评分 ≤ 切点 判为阳性）" if flipped else ""
        )
        ci_note = "（DeLong 法）" if ci_method == "delong" else "（⚠ Hanley-McNeil 法，DeLong 退化时的诚实回退）"
        if abs(auc_hi - auc_lo) < 1e-9:  # perfect separation → variance degenerates to 0
            ci_note = ci_note.rstrip("）") + "；⚠ 完美区分，方差退化，CI 收敛为点估计，非真实抽样不确定性）"
    else:
        # --- BINARY test: build the 2x2 directly, skip ROC/AUC ---
        t, pos_t, neg_t = _as_binary01(sub[test])
        if len(t) != len(y):  # defensive; both dropna'd from same sub
            t = t[: len(y)]
        tp = float(((t == 1) & (y == 1)).sum())
        fp_ = float(((t == 1) & (y == 0)).sum())
        fn = float(((t == 0) & (y == 1)).sum())
        tn = float(((t == 0) & (y == 0)).sum())
        sens, spec, ppv, npv, acc, lr_plus, lr_minus = _metrics_from_2x2(tp, fp_, fn, tn)
        cutoff_report = float("nan")
        flip_note = ""
        ci_note = ""

    # --- common: metrics CSV + estimates + Chinese summary ---
    def _row(name, tup):
        return {"metric": name, "estimate": round(tup[0], 4),
                "ci_low": round(tup[1], 4), "ci_high": round(tup[2], 4)}

    metrics_rows = [
        _row("sensitivity", sens), _row("specificity", spec),
        _row("ppv", ppv), _row("npv", npv), _row("accuracy", acc),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(d / "diagnostic_metrics.csv", index=False, encoding="utf-8")
    files.append("diagnostic_metrics.csv")

    estimates["sens"] = round(sens[0], 4)
    estimates["spec"] = round(spec[0], 4)
    estimates["ppv"] = round(ppv[0], 4)
    estimates["npv"] = round(npv[0], 4)
    estimates["accuracy"] = round(acc[0], 4)
    estimates["lr_plus"] = round(lr_plus, 4) if math.isfinite(lr_plus) else float("inf")
    estimates["lr_minus"] = round(lr_minus, 4) if math.isfinite(lr_minus) else float("inf")
    estimates["prevalence"] = round(prevalence, 4)

    cutoff_txt = (f"，Youden 最优切点={cutoff_report:.4g}" if test_is_cont and math.isfinite(cutoff_report) else "")
    auc_txt = (f"AUC={estimates['auc']:.3f}（{int(ci_level*100)}% CI {estimates['auc_ci_low']:.3f}–"
               f"{estimates['auc_ci_high']:.3f}）{ci_note}{cutoff_txt}；" if test_is_cont else "")
    cell_warn = ""
    if (sens[1] != sens[1] or (sens[2] - sens[1]) > 0.4 or (spec[2] - spec[1]) > 0.4):
        cell_warn = "；⚠ 部分小格 CI 很宽（样本/某类过少，估计不稳）"
    # disclose the RESOLVED positive/disease level so a string-labeled column
    # (e.g. "case"/"control") can't silently invert sens<->spec on the reader —
    # _as_binary01 maps the lexicographically-LAST label to positive=1.
    pos_note = f"（金标准阳性/病例取 {truth}={_fmt_level(pos_y)} 一档"
    if not test_is_cont:
        pos_note += f"；检验阳性取 {test}={_fmt_level(pos_t)} 一档"
    pos_note += "；如反了请用 config 指定）"
    summary.append(
        f"{entry.method} 完成：金标准 {truth}（病例 {n_pos}/非病例 {n_neg}，患病率 {prevalence:.1%}），"
        f"检验 {test}（{'连续评分' if test_is_cont else '二值'}）{pos_note}。{auc_txt}"
        f"敏感度={sens[0]:.1%}，特异度={spec[0]:.1%}，PPV={ppv[0]:.1%}，NPV={npv[0]:.1%}，"
        f"LR+={estimates['lr_plus']:.2f}，LR−={estimates['lr_minus']:.2f}，准确率={acc[0]:.1%}。{flip_note}"
        "⚠ PPV/NPV 随患病率变化（此处按样本患病率，非人群真实患病率）；LR+/LR− 与患病率无关，"
        "更适合跨人群迁移；各比例为 Wilson 95% CI（小格 CI 偏宽）。" + cell_warn
    )
    code += [
        "import numpy as np, pandas as pd  # 诊断试验评价",
        "# 连续评分: 经验 ROC + AUC(秩) + DeLong CI; Youden J 取最优切点 -> 2x2",
        "# 二值检验: 直接 2x2 -> sens/spec/PPV/NPV/LR±/accuracy (Wilson CI)",
    ]


def _auc_ci_delong(scores, y, auc, z):
    """95% CI for AUC.

    DeLong (1988) variance from per-subject midrank placement values (the structural
    components of the AUC), which is exact for ties. If that computation is degenerate
    (one class size <2, or zero variance), fall back to the Hanley-McNeil (1982)
    normal-approximation SE and SAY SO via the returned method tag.

    NOTE (STOP-AND-REPORT (a)): DeLong with tied scores is handled by using MIDRANKS
    (average ranks within tie blocks) in the placement values — the standard tie
    treatment in pROC/Sun-Xu; this is the defensible convention. Returns (lo, hi, tag)
    where tag in {"delong", "hanley"}.
    """
    import numpy as np

    pos = scores[y == 1]
    neg = scores[y == 0]
    m, n = len(pos), len(neg)
    try:
        if m < 2 or n < 2:
            raise ValueError("class too small for DeLong variance")

        def _midrank(x):
            order = np.argsort(x, kind="mergesort")
            xs = x[order]
            r = np.empty(len(x), dtype=float)
            i = 0
            while i < len(xs):
                j = i
                while j < len(xs) and xs[j] == xs[i]:
                    j += 1
                r[i:j] = 0.5 * (i + j - 1) + 1  # 1-based average rank over the tie block
                i = j
            out = np.empty(len(x), dtype=float)
            out[order] = r
            return out

        both = np.concatenate([pos, neg])
        r_all = _midrank(both)
        r_pos = _midrank(pos)
        r_neg = _midrank(neg)
        # DeLong structural components (Sun & Xu 2014 fast algorithm)
        v01 = (r_all[:m] - r_pos) / n              # placement of each positive among negatives
        v10 = 1.0 - (r_all[m:] - r_neg) / m        # placement of each negative among positives
        s01 = np.var(v01, ddof=1) if m > 1 else 0.0
        s10 = np.var(v10, ddof=1) if n > 1 else 0.0
        var_auc = s01 / m + s10 / n
        if not np.isfinite(var_auc):
            raise ValueError("non-finite DeLong variance")
        # zero variance is genuine at a perfect/degenerate AUC (no overlap between
        # classes) — the honest interval is a point at the estimate, not a fallback.
        if var_auc <= 0:
            return (auc, auc, "delong")
        se = math.sqrt(var_auc)
        return (max(0.0, auc - z * se), min(1.0, auc + z * se), "delong")
    except Exception:
        # Hanley-McNeil normal-approximation SE (honest fallback)
        q1 = auc / (2.0 - auc)
        q2 = 2.0 * auc * auc / (1.0 + auc)
        var = (auc * (1 - auc) + (m - 1) * (q1 - auc * auc) + (n - 1) * (q2 - auc * auc)) / (m * n) \
            if m and n else float("nan")
        se = math.sqrt(var) if (var == var and var > 0) else float("nan")
        if se != se:  # perfect AUC (var=0) or undefined -> degenerate point interval
            return (auc, auc, "hanley")
        return (max(0.0, auc - z * se), min(1.0, auc + z * se), "hanley")


def _roc_plot(fpr, tpr, j_fpr, j_tpr, auc, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, "-", color="#4C72B0", lw=1.8, label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=0.9, label="chance")
        ax.plot([j_fpr], [j_tpr], "o", color="#C44E52", ms=8,
                label=f"Youden point (sens={j_tpr:.2f}, spec={1 - j_fpr:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("1 - Specificity (FPR)")
        ax.set_ylabel("Sensitivity (TPR)")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


# =========================================================================== #
# 2) epi_risk_measures — 2x2 association measures
# =========================================================================== #
@register("epi_risk_measures")
def _branch_epi_risk_measures(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import pandas as pd

    bin_cols = _binary_cols(ctx)
    ci_level = cfg.get("ci")
    try:
        ci_level = float(ci_level) if ci_level is not None else 0.95
    except (TypeError, ValueError):
        ci_level = 0.95
    z = _z_for(ci_level)

    # exposure ≈ TREATMENT (prefer a treatment-named binary); outcome ≈ DISEASE (the shared
    # outcome resolver among the remaining binaries). Reverse rule: without it, column order
    # alone decided which binary was exposure vs outcome — {disease, exposed} vs {exposed,
    # disease} gave opposite (and half the time wrong) assignments.
    exposure = cfg.get("exposure")
    if exposure not in bin_cols:
        exposure = _resolve_exposure(fp, bin_cols)
    outcome = cfg.get("outcome")
    if outcome not in bin_cols or outcome == exposure:
        rest = [c for c in bin_cols if c != exposure]
        outcome = resolve_outcome(fp, {}, rest) if rest else None
    if exposure is None or outcome is None or exposure == outcome:
        summary.append("流行病学风险测度跳过：需要 1 个二值暴露列 + 1 个不同的二值结局列。config 用 exposure/outcome 指定。")
        return

    sub = df[[exposure, outcome]].dropna()
    if sub.shape[0] < 4:
        summary.append("流行病学风险测度跳过：成对删除缺失后样本不足 4 行。")
        return
    e, pos_e, neg_e = _as_binary01(sub[exposure])  # exposed = 1
    o, pos_o, neg_o = _as_binary01(sub[outcome])   # case = 1
    if len(set(e)) < 2 or len(set(o)) < 2:
        summary.append("流行病学风险测度跳过：暴露或结局只有一个类别（无法构成 2×2）。")
        return

    # 2x2: a=exposed&case, b=exposed&noncase, c=unexposed&case, d=unexposed&noncase
    a = float(((e == 1) & (o == 1)).sum())
    b = float(((e == 1) & (o == 0)).sum())
    c = float(((e == 0) & (o == 1)).sum())
    dd = float(((e == 0) & (o == 0)).sum())

    any_zero = (a == 0 or b == 0 or c == 0 or dd == 0)
    # Haldane-Anscombe 0.5 continuity correction when any cell is zero (for OR/RR log-CI)
    ca, cb, cc, cd = (a, b, c, dd)
    if any_zero:
        ca, cb, cc, cd = a + 0.5, b + 0.5, c + 0.5, dd + 0.5

    n_exp = ca + cb
    n_unexp = cc + cd
    risk_exp = ca / n_exp
    risk_unexp = cc / n_unexp

    # --- Risk Ratio with log CI ---
    rr = risk_exp / risk_unexp if risk_unexp > 0 else float("inf")
    if math.isfinite(rr) and rr > 0:
        se_log_rr = math.sqrt((1 / ca - 1 / n_exp) + (1 / cc - 1 / n_unexp))
        rr_lo = rr * math.exp(-z * se_log_rr)
        rr_hi = rr * math.exp(z * se_log_rr)
    else:
        rr_lo = rr_hi = float("nan")

    # --- Odds Ratio with Woolf log CI ---
    orr = (ca * cd) / (cb * cc) if (cb * cc) > 0 else float("inf")
    if math.isfinite(orr) and orr > 0:
        se_log_or = math.sqrt(1 / ca + 1 / cb + 1 / cc + 1 / cd)
        or_lo = orr * math.exp(-z * se_log_or)
        or_hi = orr * math.exp(z * se_log_or)
    else:
        or_lo = or_hi = float("nan")

    # --- Risk Difference (Wald CI, computed on the RAW cells for an honest absolute scale) ---
    n_exp_raw = a + b
    n_unexp_raw = c + dd
    r1 = a / n_exp_raw if n_exp_raw > 0 else float("nan")
    r0 = c / n_unexp_raw if n_unexp_raw > 0 else float("nan")
    rd = r1 - r0
    se_rd = math.sqrt(r1 * (1 - r1) / n_exp_raw + r0 * (1 - r0) / n_unexp_raw) \
        if (n_exp_raw > 0 and n_unexp_raw > 0) else float("nan")
    rd_lo = rd - z * se_rd if se_rd == se_rd else float("nan")
    rd_hi = rd + z * se_rd if se_rd == se_rd else float("nan")

    # --- NNT/NNH = 1/|RD| with harm/benefit interpretation ---
    nnt = 1.0 / abs(rd) if (rd == rd and rd != 0) else float("inf")
    if rd > 0:
        nnt_label = "NNH（暴露增加风险，每暴露这么多人多 1 例）"
    elif rd < 0:
        nnt_label = "NNT（暴露降低风险，每暴露这么多人少 1 例）"
    else:
        nnt_label = "NNT/NNH 不适用（RD≈0）"

    # --- Attributable risk % in the exposed = (RR-1)/RR ---
    ar_pct = (rr - 1.0) / rr * 100.0 if (math.isfinite(rr) and rr > 0) else float("nan")
    # --- Population attributable risk % (Levin): exposure prevalence from the data ---
    pe = n_exp_raw / (n_exp_raw + n_unexp_raw) if (n_exp_raw + n_unexp_raw) > 0 else float("nan")
    par_pct = (pe * (rr - 1.0)) / (pe * (rr - 1.0) + 1.0) * 100.0 \
        if (pe == pe and math.isfinite(rr) and rr > 0) else float("nan")

    # --- 2x2 table CSV ---
    table = pd.DataFrame(
        [[int(a), int(b), int(a + b)], [int(c), int(dd), int(c + dd)],
         [int(a + c), int(b + dd), int(a + b + c + dd)]],
        index=[f"exposed({exposure}=1)", f"unexposed({exposure}=0)", "total"],
        columns=[f"case({outcome}=1)", f"noncase({outcome}=0)", "total"],
    )
    table.to_csv(d / "two_by_two.csv", encoding="utf-8")
    files.append("two_by_two.csv")

    _forest_plot(
        [("RR", rr, rr_lo, rr_hi, 1.0), ("OR", orr, or_lo, or_hi, 1.0),
         ("RD", rd, rd_lo, rd_hi, 0.0)],
        d / "risk_forest.png",
    )
    if (d / "risk_forest.png").exists():
        files.append("risk_forest.png")

    estimates["rr"] = round(rr, 4) if math.isfinite(rr) else float("inf")
    estimates["rr_ci_low"] = round(rr_lo, 4) if rr_lo == rr_lo else float("nan")
    estimates["rr_ci_high"] = round(rr_hi, 4) if rr_hi == rr_hi else float("nan")
    estimates["or"] = round(orr, 4) if math.isfinite(orr) else float("inf")
    estimates["or_ci_low"] = round(or_lo, 4) if or_lo == or_lo else float("nan")
    estimates["or_ci_high"] = round(or_hi, 4) if or_hi == or_hi else float("nan")
    estimates["rd"] = round(rd, 4) if rd == rd else float("nan")
    estimates["rd_ci_low"] = round(rd_lo, 4) if rd_lo == rd_lo else float("nan")
    estimates["rd_ci_high"] = round(rd_hi, 4) if rd_hi == rd_hi else float("nan")
    estimates["nnt"] = round(nnt, 2) if math.isfinite(nnt) else float("inf")
    estimates["ar_pct"] = round(ar_pct, 2) if ar_pct == ar_pct else float("nan")
    if par_pct == par_pct:
        estimates["par_pct"] = round(par_pct, 2)

    cc_note = ("；⚠ 存在 0 单元格，OR/RR 已用 Haldane-Anscombe 0.5 连续性校正（RD 仍用原始格）"
               if any_zero else "")
    rare_note = ("（结局罕见，OR≈RR）" if (r1 < 0.1 and r0 < 0.1 and r1 == r1 and r0 == r0) else "")
    lvl = int(ci_level * 100)
    # disclose the RESOLVED positive levels so a string-labeled column (e.g.
    # "case"/"control") can't silently invert RR/OR/RD — _as_binary01 maps the
    # lexicographically-LAST label to positive=1 (correct for 0/1 numeric coding,
    # but not self-evident for string labels).
    pos_note = (f"（阳性/病例取 {outcome}={_fmt_level(pos_o)} 一档、"
                f"暴露取 {exposure}={_fmt_level(pos_e)} 一档；如反了请用 config 指定）")
    summary.append(
        f"{entry.method} 完成：暴露 {exposure}、结局 {outcome}{pos_note}，"
        f"2×2 = [a={int(a)}, b={int(b)}, c={int(c)}, d={int(dd)}]。"
        f"RR={rr:.3f}（{lvl}% CI {rr_lo:.3f}–{rr_hi:.3f}）；OR={orr:.3f}{rare_note}（{lvl}% CI {or_lo:.3f}–{or_hi:.3f}）；"
        f"RD={rd:+.3f}（{lvl}% CI {rd_lo:+.3f}–{rd_hi:+.3f}）；{nnt_label} ≈ {nnt:.1f}；"
        f"暴露人群归因比 AR%={ar_pct:.1f}%"
        + (f"，人群归因比 PAR%={par_pct:.1f}%（暴露率 {pe:.1%}）" if par_pct == par_pct else "")
        + f"。{cc_note}"
        "⚠ OR 仅在结局罕见时≈RR（病例对照研究常报 OR）；以上为关联非因果（无混杂调整）。"
    )
    code += [
        "import numpy as np  # 流行病学 2x2 风险测度",
        "# a,b,c,d = 暴露×结局四格; RR/OR(log CI, Woolf), RD(Wald CI), NNT=1/|RD|",
        "# AR%=(RR-1)/RR; PAR%=pe(RR-1)/(pe(RR-1)+1); 任一格为0用 Haldane-Anscombe 0.5 校正",
    ]


def _forest_plot(rows, path):
    """rows = [(label, est, lo, hi, null_ref), ...] -> horizontal CI forest plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [r[0] for r in rows]
        ys = list(range(len(rows)))[::-1]
        fig, ax = plt.subplots(figsize=(6, 0.9 * len(rows) + 1.5))
        for (_label, est, lo, hi, _ref), yy in zip(rows, ys):
            if est == est and np.isfinite(est):
                if lo == lo and hi == hi:
                    ax.plot([lo, hi], [yy, yy], "-", color="#4C72B0", lw=1.6)
                ax.plot([est], [yy], "s", color="#4C72B0", ms=8)
        # one reference line per distinct null (RR/OR null=1, RD null=0) — draw both
        ax.axvline(1.0, color="#C44E52", ls="--", lw=0.7, label="null (RR/OR=1)")
        ax.axvline(0.0, color="#55A868", ls=":", lw=0.7, label="null (RD=0)")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_xlabel("effect estimate with 95% CI")
        ax.set_title("Association measures (forest)")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


# =========================================================================== #
# 3) calibration_assessment — calibration of predicted probabilities
# =========================================================================== #
@register("calibration_assessment")
def _branch_calibration_assessment(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    bin_cols = _binary_cols(ctx)
    cont_cols = _continuous_cols(ctx)

    # number of calibration bins / HL groups (config; default 10 deciles)
    try:
        n_bins = int(cfg.get("bins") or cfg.get("groups") or 10)
    except (TypeError, ValueError):
        n_bins = 10
    n_bins = max(3, min(n_bins, 20))

    # --- resolve the predicted-probability column: a column whose values lie in [0,1].
    # An explicitly-configured prob/predicted column is tried first regardless of the
    # profiler's kind label (the 'count'/'id' trap); otherwise scan continuous columns.
    prob_col = cfg.get("prob") or cfg.get("predicted")
    if prob_col is not None and prob_col in df.columns:
        candidates = [prob_col] + [c for c in cont_cols if c != prob_col]
    else:
        candidates = cont_cols

    def _in01(name):
        s = pd.to_numeric(df[name], errors="coerce").dropna()
        return len(s) > 0 and float(s.min()) >= 0.0 and float(s.max()) <= 1.0

    prob = next((c for c in candidates if c is not None and c in df.columns and _in01(c)), None)
    if prob is None:
        summary.append("校准评估需要预测概率列(0-1)；未检测到。请提供取值在 [0,1] 的连续列，或 config 用 prob 指定。")
        return

    # --- resolve the binary outcome (the observed event paired with the predicted prob) ---
    # OUTCOME-semantics → shared resolver among the non-prob binaries (high-confidence name,
    # treatment-name skip) rather than first-binary.
    outcome = cfg.get("outcome")
    if outcome not in bin_cols or outcome == prob:
        rest = [c for c in bin_cols if c != prob]
        outcome = resolve_outcome(fp, {}, rest) if rest else None
    if outcome is None:
        summary.append("校准评估跳过：需要一个二值结局列（与预测概率配对）。config 用 outcome 指定。")
        return

    sub = df[[prob, outcome]].dropna()
    if sub.shape[0] < max(2 * n_bins, 10):
        summary.append(f"校准评估跳过：成对删除缺失后样本不足（需 ≥{max(2 * n_bins, 10)} 行）。")
        return
    p = pd.to_numeric(sub[prob], errors="coerce").to_numpy(dtype=float)
    y, pos_y, neg_y = _as_binary01(sub[outcome])
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if len(set(y)) < 2:
        summary.append("校准评估跳过：结局只有一个类别，无法评估校准。")
        return
    N = len(y)

    # --- Brier score + Murphy 3-component decomposition (reliability/resolution/uncertainty) ---
    brier = float(np.mean((p - y) ** 2))
    base_rate = float(np.mean(y))
    uncertainty = base_rate * (1 - base_rate)
    # bin for the decomposition (same K bins as the curve)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    rows = []
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        pbar = float(np.mean(p[mask]))   # mean predicted in bin
        obar = float(np.mean(y[mask]))   # observed frequency in bin
        reliability += nk * (pbar - obar) ** 2
        resolution += nk * (obar - base_rate) ** 2
        lo, hi = _wilson_ci(obar * nk, nk)[1:]
        rows.append({"bin": k + 1, "bin_low": round(edges[k], 4), "bin_high": round(edges[k + 1], 4),
                     "n": nk, "mean_pred": round(pbar, 4), "obs_freq": round(obar, 4),
                     "obs_ci_low": round(lo, 4), "obs_ci_high": round(hi, 4)})
    reliability /= N
    resolution /= N
    # Murphy: Brier = reliability - resolution + uncertainty (identity holds approximately
    # with binning; we report the decomposition AND the direct Brier separately)

    # --- Hosmer-Lemeshow (group by deciles of predicted prob; chi-square, df = g-2) ---
    # NOTE (STOP-AND-REPORT (b)): HL groups by QUANTILES of predicted probability
    # (deciles) — the canonical Hosmer-Lemeshow C-statistic — not by equal-width bins,
    # and df = g - 2 (standard for the C-statistic). Groups with no cases/non-cases or
    # degenerate expected counts are dropped, reducing df accordingly.
    hl_stat, hl_p, hl_df = _hosmer_lemeshow(p, y, n_bins)

    # --- calibration slope & intercept via logistic recalibration: y ~ logit(p) ---
    cal_slope, cal_intercept = _calibration_slope_intercept(p, y)

    per_bin = pd.DataFrame(rows)
    per_bin.to_csv(d / "calibration_bins.csv", index=False, encoding="utf-8")
    files.append("calibration_bins.csv")
    _calibration_plot(per_bin, brier, cal_slope, cal_intercept, d / "calibration_plot.png")
    if (d / "calibration_plot.png").exists():
        files.append("calibration_plot.png")

    estimates["brier"] = round(brier, 4)
    estimates["brier_reliability"] = round(reliability, 4)
    estimates["brier_resolution"] = round(resolution, 4)
    estimates["brier_uncertainty"] = round(uncertainty, 4)
    estimates["hl_stat"] = round(hl_stat, 4) if hl_stat == hl_stat else float("nan")
    estimates["hl_p"] = round(hl_p, 4) if hl_p == hl_p else float("nan")
    estimates["hl_df"] = float(hl_df)
    estimates["cal_slope"] = round(cal_slope, 4) if cal_slope == cal_slope else float("nan")
    estimates["cal_intercept"] = round(cal_intercept, 4) if cal_intercept == cal_intercept else float("nan")
    estimates["n"] = float(N)

    slope_txt = ""
    if cal_slope == cal_slope:
        if cal_slope < 0.85:
            slope_txt = f"，校准斜率={cal_slope:.2f}<1（⚠ 预测过于自信/可能过拟合，极端概率需收缩）"
        elif cal_slope > 1.15:
            slope_txt = f"，校准斜率={cal_slope:.2f}>1（预测过于保守/欠自信）"
        else:
            slope_txt = f"，校准斜率={cal_slope:.2f}（≈1，斜率校准良好）"
    hl_txt = (f"Hosmer-Lemeshow χ²={hl_stat:.2f}（df={hl_df}，p={hl_p:.3g}"
              f"，{'未见显著失准' if (hl_p == hl_p and hl_p > 0.05) else '提示失准'}）"
              if hl_stat == hl_stat else "Hosmer-Lemeshow 不可用（分组退化）")
    # disclose the RESOLVED event/positive level — _as_binary01 maps the
    # lexicographically-LAST label to positive=1, which is not self-evident for a
    # string-labeled outcome and would otherwise silently invert what "预测概率" means.
    pos_note = f"（事件/阳性取 {outcome}={_fmt_level(pos_y)} 一档；如反了请用 config 指定）"
    summary.append(
        f"{entry.method} 完成：预测概率 {prob}、结局 {outcome}{pos_note}（n={N}，事件率 {base_rate:.1%}）。"
        f"Brier={brier:.4f}（可靠性 {reliability:.4f} − 分辨度 {resolution:.4f} + 不确定度 {uncertainty:.4f}，越小越好）；"
        f"{hl_txt}；校准截距={cal_intercept:.3f}{slope_txt}。"
        "⚠ HL 检验对分组方式/样本量敏感（大样本易显著、小样本检验力低）；斜率<1 表示过拟合/过自信，"
        "建议结合校准曲线一并判读。"
    )
    code += [
        "import numpy as np  # 校准评估",
        "# Brier=mean((p-y)^2) + Murphy 三分解(可靠性/分辨度/不确定度)",
        "# Hosmer-Lemeshow: 按预测概率十分位分组, chi2, df=g-2",
        "# 校准斜率/截距: 逻辑重校准 logit(y) ~ logit(p)",
    ]


def _hosmer_lemeshow(p, y, g):
    """Hosmer-Lemeshow C-statistic: group by g quantiles of predicted probability,
    chi-square over observed vs expected cases (and non-cases) per group, df = g_eff - 2.
    Returns (stat, p_value, df). NaN/0 if degenerate."""
    import numpy as np
    from scipy import stats

    try:
        # quantile cut into g groups (deciles by default). Use rank to break ties stably.
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=float)
        ranks[order] = np.arange(len(p))
        grp = np.minimum((ranks * g // len(p)).astype(int), g - 1)
        stat = 0.0
        used = 0
        for k in range(g):
            mask = grp == k
            nk = int(mask.sum())
            if nk == 0:
                continue
            obs_case = float(y[mask].sum())
            exp_case = float(p[mask].sum())
            obs_non = nk - obs_case
            exp_non = nk - exp_case
            if exp_case <= 0 or exp_non <= 0:
                continue
            stat += (obs_case - exp_case) ** 2 / exp_case + (obs_non - exp_non) ** 2 / exp_non
            used += 1
        df = used - 2
        if df < 1:
            return (float("nan"), float("nan"), max(df, 0))
        pval = float(stats.chi2.sf(stat, df))
        return (float(stat), pval, df)
    except Exception:
        return (float("nan"), float("nan"), 0)


def _calibration_slope_intercept(p, y):
    """Logistic recalibration: fit y ~ a + b*logit(p). b = calibration slope,
    a = calibration intercept (calibration-in-the-large after the slope). Clips p away
    from 0/1 to keep the logit finite. Returns (slope, intercept), NaN on failure."""
    import numpy as np

    try:
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        logit_p = np.log(pc / (1 - pc))
        import statsmodels.api as sm

        X = sm.add_constant(logit_p)
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        intercept = float(model.params[0])
        slope = float(model.params[1])
        return (slope, intercept)
    except Exception:
        return (float("nan"), float("nan"))


def _calibration_plot(per_bin, brier, slope, intercept, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if per_bin is None or per_bin.empty:
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=0.9, label="perfect calibration")
        mp = per_bin["mean_pred"].to_numpy()
        of = per_bin["obs_freq"].to_numpy()
        lo = of - per_bin["obs_ci_low"].to_numpy()
        hi = per_bin["obs_ci_high"].to_numpy() - of
        ax.errorbar(mp, of, yerr=[lo, hi], fmt="o-", color="#4C72B0", lw=1.4, ms=5,
                    capsize=3, label="observed (95% Wilson CI)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ttl = f"Calibration (Brier={brier:.3f}"
        if slope == slope:
            ttl += f", slope={slope:.2f}, intercept={intercept:.2f}"
        ttl += ")"
        ax.set_title(ttl)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
