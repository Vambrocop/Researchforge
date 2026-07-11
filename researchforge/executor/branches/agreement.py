"""Branch handlers for the agreement / inter-rater-reliability & method-comparison family.

Pure-Python (numpy / scipy / pandas; sklearn only for a confusion matrix) — no R bridge.
Three estimators:

  * ``cohens_kappa`` — Cohen's kappa (Cohen 1960) for TWO raters on categorical
    ratings, with linear / quadratic weighted kappa (Cohen 1968) for ordinal
    scales, the asymptotic (Fleiss-Cohen) SE + 95% CI, the Landis-Koch (1977)
    interpretation band, and the kappa-paradox (prevalence/bias) flag.
  * ``fleiss_kappa`` — Fleiss' kappa (Fleiss 1971) for N raters assigning subjects
    to categories, from a subjects x categories tally, with per-category kappa,
    the asymptotic SE / z-test of kappa = 0, and the interpretation band.
  * ``bland_altman`` — Bland-Altman (1986) method-comparison agreement for TWO
    continuous measurements: bias, 95% limits of agreement (LoA) with their CIs,
    and a proportional-bias regression (difference on mean).

Engine conventions (see CLAUDE.md): handlers MUTATE summary/estimates/files/code
(never rebind); column roles default by kind and are overridable via config
(rater1/rater2/raters, method1/method2, weights); products are CSV + best-effort
English-labelled Agg PNG; estimates hold floats; summary is Chinese ending with
⚠ disclosures; on failure append a Chinese "<method> 跳过/失败：<reason>" line + return.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# ---------------------------------------------------------------------------
# Shared helpers (local to this family)
# ---------------------------------------------------------------------------

def _landis_koch(k: float) -> str:
    """Landis & Koch (1977) strength-of-agreement bands (Chinese labels)."""
    if k != k:  # NaN
        return "不可用"
    if k < 0.0:
        return "差于随机(<0,⚠)"
    if k < 0.20:
        return "微弱(slight, 0–0.20)"
    if k < 0.40:
        return "尚可(fair, 0.21–0.40)"
    if k < 0.60:
        return "中等(moderate, 0.41–0.60)"
    if k < 0.80:
        return "显著(substantial, 0.61–0.80)"
    return "几乎完美(almost perfect, 0.81–1)"


def _categorical_cols(ctx: Ctx) -> list[str]:
    """Rater columns: config rater1/rater2 (or raters) else the categorical-ish
    columns (categorical / binary / small-integer count), excluding unit/time."""
    df, fp = ctx.df, ctx.fp
    excl = {fp.unit_col, fp.time_col}
    return [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary", "count"} and c.name not in excl
    ]


def _looks_like_questionnaire(cols: list[str], fp) -> bool:
    """Fleiss' kappa's "subjects x raters" wide layout is structurally IDENTICAL
    to a "respondents x items" questionnaire matrix (same shape: rows = one
    thing, columns = several small-integer codes) -- fitting kappa on the
    latter still returns a number, but the number has no rater-agreement
    meaning. This is a real identifiability limit (same ambiguity Cronbach's
    alpha item-resolution faces), not a bug we can resolve automatically --
    so we only flag it, never block or silently reroute.

    Two simple, cheap signals (either is enough to suspect a questionnaire):
      (a) most column names match a common item/question naming pattern
          (item1, q3, question_2, ...);
      (b) most columns are profiler-flagged ``ordinal_like`` (bounded 1..k
          Likert-style rating scales) -- the classic "k rating-scale items"
          questionnaire shape.
    """
    import re

    if not cols:
        return False
    item_pat = re.compile(r"^(item|q|question)[\s_-]*\d+$", re.IGNORECASE)
    n_item_named = sum(1 for c in cols if item_pat.match(str(c).strip()))
    kind_by_name = {c.name: c for c in fp.columns}
    n_ordinal = sum(1 for c in cols if getattr(kind_by_name.get(c), "ordinal_like", False))
    half = len(cols) / 2.0
    return n_item_named > half or n_ordinal > half


# ===========================================================================
# 1. Cohen's kappa  (two raters)
# ===========================================================================

@register("cohens_kappa")
def _branch_cohens_kappa(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    cands = _categorical_cols(ctx)
    r1 = cfg.get("rater1") if cfg.get("rater1") in df.columns else None
    r2 = cfg.get("rater2") if cfg.get("rater2") in df.columns else None
    if r1 is None or r2 is None:
        rest = [c for c in cands if c not in {r1, r2}]
        if r1 is None:
            r1 = rest.pop(0) if rest else None
        if r2 is None:
            r2 = rest.pop(0) if rest else None
    if r1 is None or r2 is None or r1 == r2:
        summary.append("Cohen's κ 跳过：需要 2 个类别/序数评分列（两位评分者）。config 用 rater1/rater2 指定。")
        return
    try:
        sub = df[[r1, r2]].dropna()
        if sub.shape[0] < 2:
            summary.append("Cohen's κ 跳过：成对删除缺失后样本不足 2 行。")
            return
        # Unified ordered category set across both raters (so the confusion
        # matrix is square and weighted kappa uses a consistent ordering).
        cats = pd.unique(pd.concat([sub[r1], sub[r2]], ignore_index=True))
        # sort: numeric ascending if all numeric, else lexicographic — gives a
        # defensible ordinal axis for weighted kappa.
        try:
            cats = sorted(cats, key=lambda v: float(v))
        except (TypeError, ValueError):
            cats = sorted(cats, key=lambda v: str(v))
        idx = {c: i for i, c in enumerate(cats)}
        q = len(cats)
        N = sub.shape[0]

        # observed q x q confusion (counts), via sklearn when available else numpy
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(
                sub[r1].map(idx).to_numpy(), sub[r2].map(idx).to_numpy(), labels=list(range(q))
            ).astype(float)
        except Exception:
            cm = np.zeros((q, q), dtype=float)
            for a, b in zip(sub[r1].map(idx), sub[r2].map(idx)):
                cm[int(a), int(b)] += 1.0

        P = cm / N  # joint proportions
        row_marg = P.sum(axis=1)  # rater1 marginals
        col_marg = P.sum(axis=0)  # rater2 marginals

        # --- unweighted kappa (Cohen 1960) ---
        p_o = float(np.trace(P))
        p_e = float(row_marg @ col_marg)
        kappa = (p_o - p_e) / (1.0 - p_e) if p_e != 1.0 else float("nan")

        # --- asymptotic SE of unweighted kappa (Fleiss, Cohen & Everitt 1969) ---
        # se = sqrt( (A + B - C) / (N (1-pe)^2) )
        #   A = sum_i P_ii [1 - (row_i + col_i)(1 - kappa)]^2
        #   B = (1-kappa)^2 sum_{i!=j} P_ij (col_i + row_j)^2
        #   C = [kappa - pe(1-kappa)]^2
        se = float("nan")
        ci_low = ci_high = float("nan")
        if p_e != 1.0 and kappa == kappa:
            A = sum(
                P[i, i] * (1.0 - (row_marg[i] + col_marg[i]) * (1.0 - kappa)) ** 2
                for i in range(q)
            )
            B = (1.0 - kappa) ** 2 * sum(
                P[i, j] * (col_marg[i] + row_marg[j]) ** 2
                for i in range(q)
                for j in range(q)
                if i != j
            )
            C = (kappa - p_e * (1.0 - kappa)) ** 2
            var = (A + B - C) / (N * (1.0 - p_e) ** 2)
            if var > 0:
                se = float(np.sqrt(var))
                ci_low = kappa - 1.96 * se
                ci_high = kappa + 1.96 * se

        # --- weighted kappa (Cohen 1968): linear & quadratic disagreement weights ---
        # standard disagreement-weight penalty matrices:
        #   linear:    w_ij = |i-j| / (q-1)
        #   quadratic: w_ij = (i-j)^2 / (q-1)^2
        # kappa_w = 1 - (sum w_ij P_ij) / (sum w_ij Pe_ij)
        kw_linear = kw_quad = float("nan")
        if q >= 2:
            ii, jj = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
            Pe = np.outer(row_marg, col_marg)  # expected joint under independence
            w_lin = np.abs(ii - jj) / (q - 1)
            w_quad = (ii - jj) ** 2 / (q - 1) ** 2
            denom_lin = float((w_lin * Pe).sum())
            denom_quad = float((w_quad * Pe).sum())
            kw_linear = 1.0 - float((w_lin * P).sum()) / denom_lin if denom_lin > 0 else float("nan")
            kw_quad = 1.0 - float((w_quad * P).sum()) / denom_quad if denom_quad > 0 else float("nan")

        # which weighting the summary headlines (config weights: none/linear/quadratic)
        wmode = str(cfg.get("weights", "none")).lower()
        weighted_headline = {"linear": kw_linear, "quadratic": kw_quad}.get(wmode)

        # --- kappa paradox / prevalence + bias indices (Byrt, Bishop & Carlin 1993) ---
        # 2x2 only: PI = (a - d)/N (prevalence index), BI = (b - c)/N (bias index).
        # High p_o but low kappa with large |PI| is the classic kappa paradox.
        prevalence_index = bias_index = float("nan")
        paradox = False
        if q == 2:
            a, b = cm[0, 0], cm[0, 1]
            c, dd = cm[1, 0], cm[1, 1]
            prevalence_index = float((a - dd) / N)
            bias_index = float((b - c) / N)
            if p_o >= 0.80 and kappa == kappa and kappa < 0.40 and abs(prevalence_index) >= 0.30:
                paradox = True

        # confusion matrix CSV
        cm_df = pd.DataFrame(
            cm.astype(int),
            index=[f"{r1}={c}" for c in cats],
            columns=[f"{r2}={c}" for c in cats],
        )
        cm_df.to_csv(d / "confusion_matrix.csv", encoding="utf-8")
        files.append("confusion_matrix.csv")

        # estimates table CSV
        est_rows = [
            {"metric": "cohens_kappa", "value": round(kappa, 4) if kappa == kappa else None},
            {"metric": "kappa_linear_weighted", "value": round(kw_linear, 4) if kw_linear == kw_linear else None},
            {"metric": "kappa_quadratic_weighted", "value": round(kw_quad, 4) if kw_quad == kw_quad else None},
            {"metric": "observed_agreement", "value": round(p_o, 4)},
            {"metric": "expected_agreement", "value": round(p_e, 4)},
            {"metric": "se", "value": round(se, 4) if se == se else None},
            {"metric": "ci_low", "value": round(ci_low, 4) if ci_low == ci_low else None},
            {"metric": "ci_high", "value": round(ci_high, 4) if ci_high == ci_high else None},
            {"metric": "n_pairs", "value": float(N)},
            {"metric": "n_categories", "value": float(q)},
        ]
        pd.DataFrame(est_rows).to_csv(d / "kappa_estimates.csv", index=False, encoding="utf-8")
        files.append("kappa_estimates.csv")

        # heatmap of confusion matrix
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(4, q * 0.8), max(3.5, q * 0.7)))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(q)); ax.set_xticklabels([str(c) for c in cats], rotation=30, ha="right")
            ax.set_yticks(range(q)); ax.set_yticklabels([str(c) for c in cats])
            ax.set_xlabel(f"rater 2 ({r2})")
            ax.set_ylabel(f"rater 1 ({r1})")
            for i in range(q):
                for j in range(q):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
            ax.set_title(f"Cohen's kappa = {kappa:.3f}  (observed agreement = {p_o:.1%})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(d / "confusion_heatmap.png", dpi=150)
            plt.close(fig)
            files.append("confusion_heatmap.png")
        except Exception:
            pass

        estimates["cohens_kappa"] = round(float(kappa), 4) if kappa == kappa else float("nan")
        estimates["observed_agreement"] = round(p_o, 4)
        estimates["expected_agreement"] = round(p_e, 4)
        if kw_linear == kw_linear:
            estimates["kappa_linear_weighted"] = round(float(kw_linear), 4)
        if kw_quad == kw_quad:
            estimates["kappa_quadratic_weighted"] = round(float(kw_quad), 4)
        if se == se:
            estimates["se"] = round(se, 4)
            estimates["ci_low"] = round(ci_low, 4)
            estimates["ci_high"] = round(ci_high, 4)
        estimates["n_pairs"] = float(N)
        estimates["n_categories"] = float(q)
        if prevalence_index == prevalence_index:
            estimates["prevalence_index"] = round(prevalence_index, 4)
            estimates["bias_index"] = round(bias_index, 4)

        ci_txt = f"，95% CI=[{ci_low:.3f}, {ci_high:.3f}]" if se == se else ""
        w_txt = ""
        if weighted_headline is not None and weighted_headline == weighted_headline:
            w_txt = f"；加权 κ（{wmode}）={weighted_headline:.3f}"
        elif kw_quad == kw_quad:
            w_txt = f"；线性加权 κ={kw_linear:.3f}，二次加权 κ={kw_quad:.3f}（序数量表可用）"
        msg = (
            f"{entry.method} 完成：评分者 {r1} vs {r2}（{N} 对，{q} 类）。"
            f"κ={kappa:.3f}（{_landis_koch(kappa)}）{ci_txt}；观测一致率={p_o:.1%}，期望(随机)一致率={p_e:.1%}{w_txt}。"
        )
        if paradox:
            msg += (
                f" ⚠ κ 悖论：观测一致率高（{p_o:.0%}）但 κ 偏低，因类别患病率极不均衡"
                f"（prevalence index={prevalence_index:+.2f}）——κ 对边际患病率/基率敏感，此时高一致≠高 κ。"
            )
        msg += (
            " ⚠ κ 把一致率做了随机校正；它对边际患病率/基率敏感（患病率极偏时即「κ 悖论」）；"
            "序数量表请看加权 κ；仅适用 2 位评分者（≥3 位用 fleiss_kappa）。"
        )
        summary.append(msg)
        code += [
            "import numpy as np  # Cohen (1960) kappa + Cohen (1968) weighted kappa",
            f"sub = df[['{r1}', '{r2}']].dropna()",
            "# P = q x q joint-proportion confusion; po=trace(P); pe=row_marg @ col_marg",
            "kappa = (po - pe) / (1 - pe)",
            "# weighted: w_quad=(i-j)^2/(q-1)^2; kappa_w = 1 - sum(w*P)/sum(w*Pe)",
        ]
    except Exception as err:
        summary.append(f"Cohen's κ 失败：{err}")


# ===========================================================================
# 2. Fleiss' kappa  (>= 3 raters)
# ===========================================================================

@register("fleiss_kappa")
def _branch_fleiss_kappa(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    # Two accepted layouts:
    #  (a) wide subjects x raters matrix of categorical codes (config `raters`,
    #      else all categorical/binary/count columns) -> build the count matrix;
    #  (b) an already-tallied subjects x categories COUNT matrix (config
    #      `count_matrix: true`) where each cell is "# raters in that category".
    cands = _categorical_cols(ctx)
    raters = cfg.get("raters")
    if raters:
        raters = [c for c in raters if c in df.columns]
    else:
        raters = cands

    count_mode = bool(cfg.get("count_matrix"))
    questionnaire_like = False
    try:
        if count_mode:
            # rows = subjects, cols = categories; cell = count of raters
            cols = raters or [c.name for c in fp.columns if c.kind in {"count", "continuous"}]
            mat = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
            if mat.shape[0] < 2 or mat.shape[1] < 2:
                summary.append("Fleiss' κ 跳过：计数矩阵需 ≥2 个被试行、≥2 个类别列。")
                return
            counts = mat
            cat_labels = list(cols)
        else:
            if len(raters) < 3:
                summary.append(
                    "Fleiss' κ 跳过：需要 ≥3 个评分者列（宽表 被试×评分者）。"
                    "config 用 raters=[...] 指定；已计数表用 count_matrix=true（≥3 个评分者）。"
                )
                return
            questionnaire_like = _looks_like_questionnaire(raters, fp)
            sub = df[raters].dropna()
            if sub.shape[0] < 2:
                summary.append("Fleiss' κ 跳过：成行删除缺失后被试不足 2 行。")
                return
            # build subjects x categories count matrix
            cats = pd.unique(sub.to_numpy().ravel())
            try:
                cats = sorted(cats, key=lambda v: float(v))
            except (TypeError, ValueError):
                cats = sorted(cats, key=lambda v: str(v))
            cat_idx = {c: j for j, c in enumerate(cats)}
            counts = np.zeros((sub.shape[0], len(cats)), dtype=float)
            arr = sub.to_numpy()
            for i in range(arr.shape[0]):
                for v in arr[i]:
                    counts[i, cat_idx[v]] += 1.0
            cat_labels = [str(c) for c in cats]

        n_subjects, q = counts.shape
        # n = ratings per subject; Fleiss assumes equal n. Use the actual MODE (most common
        # row sum), not the median — for a tie like [3,3,4,4] the median (3.5→4) is arbitrary
        # and drops half the subjects; the mode retains the largest equal-n subset.
        row_n = counts.sum(axis=1)
        _vals, _cnts = np.unique(row_n, return_counts=True)
        n_raters = int(_vals[int(np.argmax(_cnts))])
        unequal = bool(np.any(row_n != row_n[0]))
        if not np.all(row_n > 1):
            summary.append("Fleiss' κ 跳过：每个被试需 ≥2 个评分（P_i 需要 n>1）。")
            return
        # If unequal, drop subjects whose total != modal n (Fleiss requires equal n).
        if unequal:
            keep = row_n == n_raters
            counts = counts[keep]
            n_subjects = counts.shape[0]
            if n_subjects < 2:
                summary.append("Fleiss' κ 跳过：评分数不一致，按众数对齐后被试不足 2 行。")
                return
        n = n_raters

        # --- Fleiss (1971) ---
        # P_j = category proportions = sum_i n_ij / (N n)
        N = n_subjects
        p_j = counts.sum(axis=0) / (N * n)
        # P_i (per-subject agreement) = (sum_j n_ij^2 - n) / (n (n-1))
        P_i = (np.square(counts).sum(axis=1) - n) / (n * (n - 1))
        P_bar = float(P_i.mean())
        P_e = float(np.square(p_j).sum())
        kappa = (P_bar - P_e) / (1.0 - P_e) if P_e != 1.0 else float("nan")

        # --- per-category kappa (Fleiss 1971) ---
        # kappa_j = 1 - (sum_i n_ij (n - n_ij)) / (N n (n-1) p_j (1 - p_j))
        kappa_j = np.full(q, np.nan)
        for j in range(q):
            denom = N * n * (n - 1) * p_j[j] * (1.0 - p_j[j])
            if denom > 0:
                num = float((counts[:, j] * (n - counts[:, j])).sum())
                kappa_j[j] = 1.0 - num / denom

        # --- asymptotic SE + z-test of overall kappa = 0 (Fleiss, Levin & Paik 2003) ---
        #   SE = sqrt(2) / ( s1 sqrt(N n (n-1)) ) * sqrt( s1^2 - s2 )
        #   s1 = sum pj(1-pj);  s2 = sum pj(1-pj)(1-2pj)
        se = z = pval = float("nan")
        s1 = float((p_j * (1.0 - p_j)).sum())
        if s1 > 0 and N * n * (n - 1) > 0:
            s2 = float((p_j * (1.0 - p_j) * (1.0 - 2.0 * p_j)).sum())
            inner = s1 * s1 - s2
            if inner > 0:
                se = float(np.sqrt(2.0) / (s1 * np.sqrt(N * n * (n - 1))) * np.sqrt(inner))
                if se > 0 and kappa == kappa:
                    from scipy import stats
                    z = kappa / se
                    pval = float(2.0 * stats.norm.sf(abs(z)))

        # per-category CSV
        cat_df = pd.DataFrame(
            {
                "category": cat_labels,
                "proportion": np.round(p_j, 4),
                "kappa_category": [round(float(v), 4) if v == v else None for v in kappa_j],
            }
        )
        cat_df.to_csv(d / "fleiss_category_kappa.csv", index=False, encoding="utf-8")
        files.append("fleiss_category_kappa.csv")

        est_rows = [
            {"metric": "fleiss_kappa", "value": round(kappa, 4) if kappa == kappa else None},
            {"metric": "P_bar_observed", "value": round(P_bar, 4)},
            {"metric": "P_e_expected", "value": round(P_e, 4)},
            {"metric": "se", "value": round(se, 4) if se == se else None},
            {"metric": "z", "value": round(z, 4) if z == z else None},
            {"metric": "p_value", "value": round(pval, 6) if pval == pval else None},
            {"metric": "n_subjects", "value": float(N)},
            {"metric": "n_raters", "value": float(n)},
            {"metric": "n_categories", "value": float(q)},
        ]
        pd.DataFrame(est_rows).to_csv(d / "fleiss_estimates.csv", index=False, encoding="utf-8")
        files.append("fleiss_estimates.csv")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(5, q * 0.8), 4))
            vals = [v if v == v else 0.0 for v in kappa_j]
            ax.bar(cat_labels, vals, color="#4C72B0")
            ax.axhline(kappa if kappa == kappa else 0.0, color="#C44E52", ls="--", lw=1.0,
                       label=f"overall kappa={kappa:.3f}")
            ax.set_ylabel("per-category kappa")
            ax.set_title(f"Fleiss' kappa = {kappa:.3f}  ({N} subjects, {n} raters, {q} categories)")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "fleiss_category_kappa.png", dpi=150)
            plt.close(fig)
            files.append("fleiss_category_kappa.png")
        except Exception:
            pass

        estimates["fleiss_kappa"] = round(float(kappa), 4) if kappa == kappa else float("nan")
        estimates["P_bar_observed"] = round(P_bar, 4)
        estimates["P_e_expected"] = round(P_e, 4)
        if se == se:
            estimates["se"] = round(se, 4)
            estimates["z"] = round(z, 4)
            estimates["p_value"] = round(pval, 6)
        estimates["n_subjects"] = float(N)
        estimates["n_raters"] = float(n)
        estimates["n_categories"] = float(q)

        sig = (f"，z={z:.2f}, p={pval:.2g}（{'拒绝 κ=0' if pval < 0.05 else '不能拒绝 κ=0'}）"
               if se == se else "")
        warn_unequal = " ⚠ 各被试评分数原不一致，已按众数对齐删除不齐行。" if unequal else ""
        warn_questionnaire = (
            " ⚠ 此矩阵形似「受访者×题项」（问卷量表：列像题项名/多为有界评分量表），"
            "而非「评分者×被评对象」——Fleiss κ 假设列是可互换的评分者、行是被评对象，"
            "若这其实是问卷量表，κ 可能没有实质意义；若要一致性请确认矩阵确为评分者×对象"
            "（若是问卷信度，请改用 Cronbach's α）。"
            if questionnaire_like
            else ""
        )
        summary.append(
            f"{entry.method} 完成：{N} 个被试 × {n} 个评分者，{q} 个类别。"
            f"κ={kappa:.3f}（{_landis_koch(kappa)}）；P̄(观测一致)={P_bar:.3f}，P̄_e(期望一致)={P_e:.3f}{sig}。"
            "各类别 κ 见 fleiss_category_kappa.csv。"
            + warn_unequal
            + warn_questionnaire
            + " ⚠ Fleiss κ 假设评分者可互换（不要求每个被试是同一批评分者）、每个被试评分数相等、"
            "并做了随机校正；同样对类别患病率敏感（极偏类别拉低 κ）。"
        )
        code += [
            "import numpy as np  # Fleiss (1971) kappa",
            "# counts = subjects x categories (cell = # raters in category)",
            "p_j = counts.sum(0) / (N*n); P_i = (np.square(counts).sum(1) - n)/(n*(n-1))",
            "P_bar = P_i.mean(); P_e = np.square(p_j).sum()",
            "kappa = (P_bar - P_e) / (1 - P_e)",
        ]
    except Exception as err:
        summary.append(f"Fleiss' κ 失败：{err}")


# ===========================================================================
# 3. Bland-Altman  (two continuous methods)
# ===========================================================================

@register("bland_altman")
def _branch_bland_altman(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    excl = {fp.unit_col, fp.time_col}
    cont = [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]
    m1 = cfg.get("method1") if cfg.get("method1") in df.columns else None
    m2 = cfg.get("method2") if cfg.get("method2") in df.columns else None
    if m1 is None or m2 is None:
        rest = [c for c in cont if c not in {m1, m2}]
        if m1 is None:
            m1 = rest.pop(0) if rest else None
        if m2 is None:
            m2 = rest.pop(0) if rest else None
    if m1 is None or m2 is None or m1 == m2:
        summary.append("Bland-Altman 跳过：需要 2 个连续测量列（同一量的两种方法）。config 用 method1/method2 指定。")
        return
    try:
        sub = df[[m1, m2]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.shape[0] < 3:
            summary.append("Bland-Altman 跳过：成对删除缺失后样本不足 3 行。")
            return
        x1 = sub[m1].to_numpy(dtype=float)
        x2 = sub[m2].to_numpy(dtype=float)
        n = len(x1)
        diff = x1 - x2
        mean = (x1 + x2) / 2.0

        bias = float(diff.mean())
        sd = float(diff.std(ddof=1))
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd

        # --- 95% CI of bias and of each LoA (Bland & Altman 1986/1999) ---
        # SE(bias) = sd / sqrt(n); SE(LoA) ≈ sd * sqrt( 1/n + 1.96^2 / (2(n-1)) )
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
        se_bias = sd / np.sqrt(n)
        bias_ci_low = bias - t * se_bias
        bias_ci_high = bias + t * se_bias
        se_loa = sd * np.sqrt(1.0 / n + (1.96 ** 2) / (2.0 * (n - 1)))
        loa_low_ci = (loa_low - t * se_loa, loa_low + t * se_loa)
        loa_high_ci = (loa_high - t * se_loa, loa_high + t * se_loa)

        # --- proportional bias: regress diff on mean (OLS) ---
        slope, intercept, r_val, p_slope, std_err = stats.linregress(mean, diff)
        proportional = bool(p_slope < 0.05)

        # --- normality of differences (Shapiro) for the LoA assumption ---
        shapiro_p = float("nan")
        if 3 <= n <= 5000:
            try:
                shapiro_p = float(stats.shapiro(diff).pvalue)
            except Exception:
                shapiro_p = float("nan")

        # % within LoA (should be ~95% if differences ~ normal)
        within = float(np.mean((diff >= loa_low) & (diff <= loa_high)) * 100.0)

        # correlation (to contrast with agreement — the core ⚠)
        r_pearson = float(np.corrcoef(x1, x2)[0, 1]) if n > 1 else float("nan")

        # per-pair CSV
        pd.DataFrame(
            {m1: x1, m2: x2, "mean": np.round(mean, 6), "difference": np.round(diff, 6)}
        ).to_csv(d / "bland_altman_pairs.csv", index=False, encoding="utf-8")
        files.append("bland_altman_pairs.csv")

        est_rows = [
            {"metric": "bias", "value": round(bias, 6)},
            {"metric": "sd_diff", "value": round(sd, 6)},
            {"metric": "loa_lower", "value": round(loa_low, 6)},
            {"metric": "loa_upper", "value": round(loa_high, 6)},
            {"metric": "bias_ci_low", "value": round(bias_ci_low, 6)},
            {"metric": "bias_ci_high", "value": round(bias_ci_high, 6)},
            {"metric": "loa_lower_ci_low", "value": round(loa_low_ci[0], 6)},
            {"metric": "loa_lower_ci_high", "value": round(loa_low_ci[1], 6)},
            {"metric": "loa_upper_ci_low", "value": round(loa_high_ci[0], 6)},
            {"metric": "loa_upper_ci_high", "value": round(loa_high_ci[1], 6)},
            {"metric": "proportional_bias_slope", "value": round(float(slope), 6)},
            {"metric": "proportional_bias_p", "value": round(float(p_slope), 6)},
            {"metric": "pct_within_loa", "value": round(within, 2)},
            {"metric": "pearson_r", "value": round(r_pearson, 4) if r_pearson == r_pearson else None},
            {"metric": "n_pairs", "value": float(n)},
        ]
        pd.DataFrame(est_rows).to_csv(d / "bland_altman_estimates.csv", index=False, encoding="utf-8")
        files.append("bland_altman_estimates.csv")

        # --- Bland-Altman plot ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(mean, diff, alpha=0.6, color="#4C72B0", edgecolor="none")
            ax.axhline(bias, color="#C44E52", lw=1.5, label=f"bias = {bias:.3g}")
            ax.axhline(loa_high, color="#55A868", ls="--", lw=1.2,
                       label=f"+1.96 SD = {loa_high:.3g}")
            ax.axhline(loa_low, color="#55A868", ls="--", lw=1.2,
                       label=f"-1.96 SD = {loa_low:.3g}")
            ax.axhline(0.0, color="grey", lw=0.6)
            if proportional:
                xs = np.linspace(mean.min(), mean.max(), 50)
                ax.plot(xs, intercept + slope * xs, color="#8172B3", ls=":",
                        lw=1.2, label=f"trend (p={p_slope:.2g})")
            ax.set_xlabel(f"mean of methods  (({m1} + {m2}) / 2)")
            ax.set_ylabel(f"difference  ({m1} - {m2})")
            ax.set_title("Bland-Altman agreement plot")
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(d / "bland_altman_plot.png", dpi=150)
            plt.close(fig)
            files.append("bland_altman_plot.png")
        except Exception:
            pass

        estimates["bias"] = round(bias, 6)
        estimates["sd_diff"] = round(sd, 6)
        estimates["loa_lower"] = round(loa_low, 6)
        estimates["loa_upper"] = round(loa_high, 6)
        estimates["bias_ci_low"] = round(bias_ci_low, 6)
        estimates["bias_ci_high"] = round(bias_ci_high, 6)
        estimates["proportional_bias_slope"] = round(float(slope), 6)
        estimates["proportional_bias_p"] = round(float(p_slope), 6)
        estimates["pct_within_loa"] = round(within, 2)
        if r_pearson == r_pearson:
            estimates["pearson_r"] = round(r_pearson, 4)
        estimates["n_pairs"] = float(n)

        prop_txt = (
            f" ⚠ 存在比例偏差：差值随测量量级变化（斜率={slope:.3g}, p={p_slope:.2g}）——"
            "偏差非恒定，固定 LoA 会误导，建议对数变换或回归式 LoA。"
            if proportional
            else f"无明显比例偏差（斜率 p={p_slope:.2g}）。"
        )
        norm_txt = ""
        if shapiro_p == shapiro_p and shapiro_p < 0.05:
            norm_txt = f" ⚠ 差值偏离正态（Shapiro p={shapiro_p:.2g}），1.96·SD 的 LoA 覆盖率可能不准。"
        summary.append(
            f"{entry.method} 完成：{m1} vs {m2}（{n} 对）。"
            f"偏差(bias)={bias:.4g}（95% CI [{bias_ci_low:.4g}, {bias_ci_high:.4g}]）；"
            f"95% 一致性界限 LoA=[{loa_low:.4g}, {loa_high:.4g}]"
            f"（各界限 95% CI 见 estimates）；实际落入 LoA 的比例={within:.0f}%；"
            f"两法 Pearson r={r_pearson:.3f}。{prop_txt}{norm_txt}"
            " ⚠ Bland-Altman 评估的是「一致性」而非「相关」——高 r 也可能掩盖系统偏差/差的一致；"
            "「可接受」的差值大小需结合临床/领域判断；假设差值近似正态且方差恒定（已对比例偏差/异方差做检查）；要求配对测量。"
        )
        code += [
            "import numpy as np; from scipy import stats  # Bland-Altman (1986)",
            f"d = df['{m1}'] - df['{m2}']; m = (df['{m1}'] + df['{m2}'])/2",
            "bias = d.mean(); sd = d.std(ddof=1)",
            "loa = (bias - 1.96*sd, bias + 1.96*sd)  # 95% limits of agreement",
            "slope, intercept, r, p, se = stats.linregress(m, d)  # proportional bias",
        ]
    except Exception as err:
        summary.append(f"Bland-Altman 失败：{err}")
