"""Branch handler for capture-recapture closed-population abundance estimation.

A NEW family-module sibling of ecology.py (which is near the size guardrail — do not
touch it). Estimates the size N of a CLOSED population (constant over the study: no
births/deaths/immigration/emigration, no mark loss, equal catchability) from a
capture-history matrix (rows = individuals captured at least once, columns = sampling
occasions, entries 0/1).

Estimators (chosen by #occasions K):
  - K = 2: Chapman (1951) bias-corrected Lincoln-Petersen + the M0 constant-p MLE.
  - K >= 3: Schnabel (1938) + the M0 constant-p MLE.

References (formulas verified against): Krebs, *Ecological Methodology* (2nd ed.,
Lincoln-Petersen / Chapman / Schnabel); Williams, Nichols & Conroy, *Analysis and
Management of Animal Populations* (closed-population M0). Pure numpy/scipy, no R.
"""

from __future__ import annotations

import math

from researchforge.executor._branch_api import Ctx, register


def _resolve_occasion_cols(df, fp, cfg):
    """Return the list of 0/1 occasion column names.

    Explicit config['occasions'] wins (filtered to existing columns); otherwise every
    numeric column whose non-null values are a subset of {0, 1}. Occasion columns that
    profile as `binary` need both 0 and 1 present, but an all-1 column (everyone caught
    that occasion) is still a valid occasion — so we test the value set directly rather
    than trusting profiler kinds.
    """
    import pandas as pd

    requested = cfg.get("occasions")
    if requested:
        cols = [c for c in requested if c in df.columns]
    else:
        excl = {fp.unit_col, fp.time_col}
        cols = []
        for c in df.columns:
            if c in excl:
                continue
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            vals = set(s.unique().tolist())
            if vals <= {0, 1}:
                cols.append(c)
    return cols


def _m0_mle(n_observed: int, total_captures: int, K: int):
    """M0 (constant-capture-probability) closed-population estimator.

    With K occasions, capture prob p constant across individuals/occasions, observed
    n distinct individuals and T total captures, the MLE solves jointly:
        p_hat = T / (K * N_hat)
        n     = N_hat * (1 - (1 - p_hat)^K)
    Solve by fixed-point iteration on N (Otis et al. 1978; Williams-Nichols-Conroy).
    SE from the observed-information / asymptotic variance of N_hat. Returns
    (N_hat, se, p_hat) as plain floats, or (nan, nan, nan) if not estimable
    (e.g. every individual caught every occasion -> p_hat -> 1, N_hat -> n).
    """
    n = float(n_observed)
    T = float(total_captures)
    if K < 1 or n <= 0 or T <= 0 or T >= K * n:
        # T == K*n means everyone caught every time: p_hat -> 1, no information on
        # the unseen; N_hat collapses to n (return n with no SE, honest).
        if T >= K * n and n > 0:
            return n, float("nan"), 1.0
        return float("nan"), float("nan"), float("nan")

    N = max(n + 1.0, T / K)  # start above the observed count
    for _ in range(500):
        p = T / (K * N)
        p = min(max(p, 1e-9), 1 - 1e-12)
        q_pow = (1.0 - p) ** K
        denom = 1.0 - q_pow
        if denom <= 0:
            break
        N_new = n / denom
        if not math.isfinite(N_new):
            break
        if abs(N_new - N) < 1e-8:
            N = N_new
            break
        N = N_new
    if not math.isfinite(N) or N < n:
        return float("nan"), float("nan"), float("nan")
    p_hat = T / (K * N)
    p_hat = min(max(p_hat, 1e-9), 1 - 1e-12)

    # Asymptotic variance of N_hat for M0 (Otis et al. 1978 / Williams-Nichols-Conroy),
    # from the Fisher information of the M0 profile log-likelihood:
    #   var(N) = N / ( (1-p)^(-K) - 1 - K*p/(1-p) )
    q = 1.0 - p_hat
    try:
        info_denom = q ** (-K) - 1.0 - (K * p_hat) / q
        var_N = N / info_denom if info_denom > 0 else float("nan")
        se = math.sqrt(var_N) if math.isfinite(var_N) and var_N > 0 else float("nan")
    except (OverflowError, ValueError, ZeroDivisionError):
        se = float("nan")
    return N, se, p_hat


@register("capture_recapture")
def _branch_capture_recapture(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    occ_cols = _resolve_occasion_cols(df, fp, cfg)
    if len(occ_cols) < 2:
        summary.append(
            "标记重捕跳过：需要 ≥2 个 0/1 捕获场次列（行=个体，列=场次，1=该场次被捕）。"
            f"（自动检出 occasions={occ_cols}；可用 config occasions 指定列名。）"
        )
        return

    # numeric-coerce, keep only rows captured at least once (the standard input is the
    # capture-history of individuals seen >= once).
    mat_df = df[occ_cols].apply(pd.to_numeric, errors="coerce")
    mat = mat_df.to_numpy(dtype=float)
    mat = np.where(np.isnan(mat), 0.0, mat)
    mat = (mat > 0).astype(int)  # binarise defensively
    caught_any = mat.sum(axis=1) > 0
    mat = mat[caught_any]

    n_observed = int(mat.shape[0])
    K = int(mat.shape[1])
    if n_observed < 8:
        summary.append(
            f"标记重捕跳过：捕获过的个体仅 {n_observed} 个（<8），样本过少，估计不稳。"
        )
        return

    total_captures = int(mat.sum())

    # ---- per-occasion bookkeeping: C_t (caught), newly-marked, R_t (recaptures) ----
    # M_t = distinct individuals marked BEFORE occasion t (cumulative). At occasion t an
    # individual is a "recapture" if it was caught in any earlier occasion.
    seen_before = np.zeros(n_observed, dtype=bool)
    rows_per_occ = []
    sum_CM = 0.0  # Schnabel numerator   Sum C_t * M_t
    sum_R = 0.0   # Schnabel denominator Sum R_t
    for t in range(K):
        col = mat[:, t].astype(bool)
        C_t = int(col.sum())
        M_t = int(seen_before.sum())              # marked in population before t
        R_t = int((col & seen_before).sum())      # already-marked, recaptured at t
        new_marks = C_t - R_t                      # newly marked this occasion
        rows_per_occ.append(
            {
                "occasion": t + 1,
                "occasion_col": occ_cols[t],
                "caught_C": C_t,
                "marked_before_M": M_t,
                "newly_marked": new_marks,
                "recaptures_R": R_t,
            }
        )
        sum_CM += C_t * M_t
        sum_R += R_t
        seen_before = seen_before | col

    per_occ = pd.DataFrame(rows_per_occ)
    per_occ.to_csv(d / "capture_recapture_occasions.csv", index=False, encoding="utf-8")
    files.append("capture_recapture_occasions.csv")

    estimates["n_observed"] = float(n_observed)
    estimates["n_occasions"] = float(K)
    estimates["total_captures"] = float(total_captures)

    # ---- estimators ----
    est_labels = []      # for the bar plot: (name, N, ci_low, ci_high)
    primary_txt = ""     # headline estimator sentence

    # M0 MLE (any K)
    m0_N, m0_se, p_hat = _m0_mle(n_observed, total_captures, K)
    if math.isfinite(m0_N):
        estimates["m0_mle_N"] = round(float(m0_N), 2)
        if math.isfinite(m0_se):
            estimates["m0_mle_N_se"] = round(float(m0_se), 2)
        m0_ci = None
        if math.isfinite(m0_se):
            lo = max(float(n_observed), m0_N - 1.96 * m0_se)
            hi = m0_N + 1.96 * m0_se
            m0_ci = (lo, hi)
        est_labels.append(("M0 MLE", m0_N, m0_ci[0] if m0_ci else m0_N,
                           m0_ci[1] if m0_ci else m0_N))
    if math.isfinite(p_hat):
        estimates["capture_prob"] = round(float(p_hat), 4)

    if K == 2:
        # ---- Chapman (1951) bias-corrected Lincoln-Petersen ----
        n1 = int(mat[:, 0].sum())
        n2 = int(mat[:, 1].sum())
        m2 = int((mat[:, 0].astype(bool) & mat[:, 1].astype(bool)).sum())
        N_hat = ((n1 + 1) * (n2 + 1) / (m2 + 1)) - 1.0
        var_N = (
            (n1 + 1) * (n2 + 1) * (n1 - m2) * (n2 - m2)
            / (((m2 + 1) ** 2) * (m2 + 2))
        )
        se = math.sqrt(var_N) if var_N > 0 else 0.0
        ci_low = max(float(n_observed), N_hat - 1.96 * se)
        ci_high = N_hat + 1.96 * se
        estimates["petersen_chapman_N"] = round(float(N_hat), 2)
        estimates["petersen_chapman_N_se"] = round(float(se), 2)
        estimates["petersen_chapman_N_ci_low"] = round(float(ci_low), 2)
        estimates["petersen_chapman_N_ci_high"] = round(float(ci_high), 2)
        est_labels.append(("Chapman", N_hat, ci_low, ci_high))
        primary_txt = (
            f"Chapman 偏差校正 Lincoln-Petersen N̂={N_hat:.1f}"
            f"（95% CI [{ci_low:.1f}, {ci_high:.1f}]，SE={se:.1f}；"
            f"n1={n1}, n2={n2}, m2={m2}）"
        )
        code += [
            "# Chapman (1951) bias-corrected Lincoln-Petersen (K=2 occasions)",
            "n1, n2 = mat[:,0].sum(), mat[:,1].sum()",
            "m2 = (mat[:,0].astype(bool) & mat[:,1].astype(bool)).sum()",
            "N = ((n1+1)*(n2+1)/(m2+1)) - 1",
            "var = (n1+1)*(n2+1)*(n1-m2)*(n2-m2) / ((m2+1)**2 * (m2+2))",
            "se = var ** 0.5  # 95% CI: N +/- 1.96*se",
        ]
    else:
        # ---- Schnabel (1938) for K >= 3 ----
        if sum_R > 0:
            N_hat = sum_CM / sum_R
            # Poisson/normal approx CI on Sum R_t: var(N) ~ (Sum C_t M_t)^2 / (Sum R_t)^3
            # (Schnabel via Poisson approx on the recapture total, Krebs).
            var_N = (sum_CM ** 2) / (sum_R ** 3)
            se = math.sqrt(var_N) if var_N > 0 else 0.0
            ci_low = max(float(n_observed), N_hat - 1.96 * se)
            ci_high = N_hat + 1.96 * se
            estimates["schnabel_N"] = round(float(N_hat), 2)
            estimates["schnabel_N_se"] = round(float(se), 2)
            estimates["schnabel_N_ci_low"] = round(float(ci_low), 2)
            estimates["schnabel_N_ci_high"] = round(float(ci_high), 2)
            est_labels.append(("Schnabel", N_hat, ci_low, ci_high))
            primary_txt = (
                f"Schnabel N̂={N_hat:.1f}（95% CI [{ci_low:.1f}, {ci_high:.1f}]，"
                f"SE={se:.1f}；ΣC·M={sum_CM:.0f}, ΣR={sum_R:.0f}）"
            )
        else:
            primary_txt = "Schnabel 不可估：全程零重捕（ΣR=0），无重捕信息估计 N"
        code += [
            "# Schnabel (1938), K>=3 occasions; M_t = distinct marked before occasion t",
            "# N_hat = sum(C_t * M_t) / sum(R_t)   (R_t = recaptures at t)",
            "# var(N) ~ (sum C_t M_t)^2 / (sum R_t)^3  (Poisson/normal approx CI)",
        ]

    if not primary_txt and math.isfinite(m0_N):
        primary_txt = f"M0 (恒定捕获率) MLE N̂={m0_N:.1f}"

    # ---- plot: N_hat estimates with 95% CIs ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if est_labels:
            names = [e[0] for e in est_labels]
            vals = [e[1] for e in est_labels]
            los = [e[1] - e[2] for e in est_labels]
            his = [e[3] - e[1] for e in est_labels]
            fig, ax = plt.subplots(figsize=(5.5, 4))
            xpos = np.arange(len(names))
            ax.bar(xpos, vals, color="#4C72B0", alpha=0.85,
                   yerr=[los, his], capsize=5, ecolor="#333333")
            ax.axhline(n_observed, color="#C44E52", ls="--", lw=1.0,
                       label=f"observed n = {n_observed}")
            ax.set_xticks(xpos)
            ax.set_xticklabels(names)
            ax.set_ylabel("estimated population size N")
            ax.set_title(f"Closed-population N estimates (K={K} occasions)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "capture_recapture_estimates.png", dpi=150)
            plt.close(fig)
            files.append("capture_recapture_estimates.png")
    except Exception:
        pass

    # ---- Chinese summary with disclosures ----
    primary_name = "Chapman" if K == 2 else "Schnabel"
    summary.append(
        f"{entry.method} 完成：观测到 {n_observed} 个不同个体（总捕获 {total_captures} 次）"
        f"，{K} 个捕获场次。主估计 {primary_txt}。"
        + (f" M0 恒定捕获率 MLE N̂={m0_N:.1f}" if math.isfinite(m0_N) else "")
        + (f"，捕获率 p̂={p_hat:.3f}。" if math.isfinite(p_hat) else "。")
        + " ⚠ 闭合种群假定：研究期内种群恒定（无出生/死亡/迁入/迁出）、标记不丢失、"
        "所有个体捕获概率相等——个体异质性（部分个体更易被捕）会使 N̂ 系统性**偏低**。"
        f" ⚠ {primary_name} 为本场次数下的首选；Chapman 是 Lincoln-Petersen 的小样本"
        "偏差校正式（直接用 n1·n2/m2 在小样本/低重捕时偏高）。"
        " ⚠ M0 假定捕获率恒定（无时间/行为/个体差异），与 Mh/Mb/Mt 模型相比是最简模型。"
    )
    code += [
        "import numpy as np  # capture-history matrix: rows=individuals, cols=occasions (0/1)",
        f"occ_cols = {occ_cols!r}",
        "mat = (df[occ_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy() > 0).astype(int)",
        "mat = mat[mat.sum(axis=1) > 0]  # individuals caught >= once",
        "# M0 MLE: solve n = N*(1-(1-p)^K), p = total_captures/(K*N)  (Otis et al. 1978)",
    ]
