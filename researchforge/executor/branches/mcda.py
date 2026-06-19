"""Branch handlers for the mcda family (migrated from the run.py monolith).

Each handler unpacks ctx into the same local names run_analysis used and runs the
original branch body verbatim. See executor/_branch_api.py.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import (
    _cost_mask,
    _entropy_weights,
    _mcda_direction_note,
    _mcda_inputs,
    _mcda_rank_plot,
    _minmax01,
)


def _resolve_weights(crit, cfg):
    """Resolve criterion weights for weighting-driven MCDA methods (VIKOR/PROMETHEE).

    cfg['weights'] (a list aligned with `crit`) overrides; it is coerced to floats,
    clamped to non-negative and renormalised to sum 1. If it is missing, malformed,
    the wrong length, or sums to 0, fall back to EQUAL weights. Returns
    (weights array, note string) where the note discloses which path was taken.
    """
    import numpy as np

    n = len(crit)
    raw = (cfg or {}).get("weights")
    if raw is not None:
        try:
            w = np.asarray([float(x) for x in raw], dtype=float)
            if w.shape[0] == n and np.all(w >= 0) and w.sum() > 0:
                w = w / w.sum()
                return w, "权重=用户 config['weights']（已归一）。"
        except (TypeError, ValueError):
            pass
    return np.ones(n) / n, "权重=等权（未提供有效 config['weights']）。"


@register("critic")
def _branch_critic(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"CRITIC 失败：{err}")
    else:
        import pandas as pd

        cost_mask, cost_names = _cost_mask(crit, cfg)
        Z = _minmax01(X, cost_mask)  # benefit-normalised [0,1] (cost cols flipped)
        sigma = Z.std(axis=0, ddof=1)  # contrast intensity per criterion
        # clip to [-1,1]: float noise can push r just past 1, making (1-r)<0 and
        # flipping weights negative when criteria are (near-)perfectly correlated.
        corr = np.clip(np.nan_to_num(np.corrcoef(Z, rowvar=False), nan=0.0), -1.0, 1.0)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        conflict = (1.0 - corr).sum(axis=1)  # conflict = Σ_k (1 - r_jk) ≥ 0
        info = sigma * conflict  # CRITIC information content C_j
        w = info / info.sum() if info.sum() > 0 else np.ones(len(crit)) / len(crit)
        composite = Z @ w
        res = pd.DataFrame({"alternative": labels, "critic_score": np.round(composite, 4)})
        res["rank"] = res["critic_score"].rank(ascending=False, method="min").astype(int)
        res = res.sort_values("rank").reset_index(drop=True)
        res.to_csv(d / "critic_scores.csv", index=False, encoding="utf-8")
        files.append("critic_scores.csv")
        pd.DataFrame({"criterion": crit, "critic_weight": np.round(w, 4)}).to_csv(
            d / "weights.csv", index=False, encoding="utf-8"
        )
        files.append("weights.csv")
        _mcda_rank_plot(res, "critic_score", "CRITIC-weighted ranking (top 20)", d / "critic_ranking.png")
        if (d / "critic_ranking.png").exists():
            files.append("critic_ranking.png")
        best = labels[int(np.argmax(composite))]
        estimates["top_score"] = round(float(composite.max()), 4)
        estimates["n_alternatives"] = float(len(labels))
        estimates["n_criteria"] = float(len(crit))
        summary.append(
            f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
            f"最优 [{best}]（CRITIC 加权得分 {composite.max():.3f}）；"
            "CRITIC 权重=对比度(标准差)×冲突性(1-相关) 客观赋权,见 weights.csv。"
            + _mcda_direction_note(cost_names)
        )
        code += [
            "import numpy as np  # CRITIC 客观赋权",
            "# w_j ∝ σ_j · Σ_k(1-r_jk); 综合得分 = Σ_j w_j · min-max(x_ij)",
        ]



@register("grey_relational")
def _branch_grey_relational(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"灰色关联分析失败：{err}")
    else:
        import pandas as pd

        cost_mask, cost_names = _cost_mask(crit, cfg)
        M = _minmax01(X, cost_mask)
        delta = np.abs(1.0 - M)  # distance to the ideal (benefit -> ideal = 1)
        dmin, dmax, rho = delta.min(), delta.max(), 0.5
        xi = (dmin + rho * dmax) / (delta + rho * dmax + 1e-12)  # grey relational coef
        grade = xi.mean(axis=1)  # grey relational grade (equal weight)
        res = pd.DataFrame({"alternative": labels, "relational_grade": np.round(grade, 4)})
        res["rank"] = res["relational_grade"].rank(ascending=False, method="min").astype(int)
        res = res.sort_values("rank").reset_index(drop=True)
        res.to_csv(d / "grey_relational.csv", index=False, encoding="utf-8")
        files.append("grey_relational.csv")
        _mcda_rank_plot(
            res, "relational_grade", "Grey relational ranking (top 20)",
            d / "grey_ranking.png",
        )
        if (d / "grey_ranking.png").exists():
            files.append("grey_ranking.png")
        best = labels[int(np.argmax(grade))]
        estimates["top_grade"] = round(float(grade.max()), 4)
        estimates["n_alternatives"] = float(len(labels))
        estimates["n_criteria"] = float(len(crit))
        summary.append(
            f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
            f"最优 [{best}]（关联度 {grade.max():.3f}，ρ=0.5，参考序列=各指标理想值）。"
            + _mcda_direction_note(cost_names)
        )
        code += [
            "import numpy as np  # 灰色关联分析(GRA)",
            "# Δ=|1-min-max|; ξ=(Δmin+0.5Δmax)/(Δ+0.5Δmax); 关联度=ξ 行均值",
        ]



@register("membership_function")
def _branch_membership_function(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"隶属函数法失败：{err}")
    else:
        import pandas as pd

        cost_mask, cost_names = _cost_mask(crit, cfg)
        M = _minmax01(X, cost_mask)  # membership degrees in [0,1] (cost cols flipped)
        composite = M.mean(axis=1)  # classic equal-weight average membership
        res = pd.DataFrame({"alternative": labels, "membership_score": np.round(composite, 4)})
        res["rank"] = res["membership_score"].rank(ascending=False, method="min").astype(int)
        res = res.sort_values("rank").reset_index(drop=True)
        res.to_csv(d / "membership_scores.csv", index=False, encoding="utf-8")
        files.append("membership_scores.csv")
        memb = pd.DataFrame(np.round(M, 4), columns=crit)
        memb.insert(0, "alternative", labels)
        memb.to_csv(d / "membership_matrix.csv", index=False, encoding="utf-8")
        files.append("membership_matrix.csv")
        _mcda_rank_plot(
            res, "membership_score", "Membership-function ranking (top 20)",
            d / "membership_ranking.png",
        )
        if (d / "membership_ranking.png").exists():
            files.append("membership_ranking.png")
        best = labels[int(np.argmax(composite))]
        estimates["top_score"] = round(float(composite.max()), 4)
        estimates["n_alternatives"] = float(len(labels))
        estimates["n_criteria"] = float(len(crit))
        summary.append(
            f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
            f"最优 [{best}]（隶属度均值 {composite.max():.3f}，等权）。"
            + _mcda_direction_note(cost_names)
        )
        code += [
            "import numpy as np  # 隶属函数法(等权)",
            "# M = min-max[0,1] 隶属度; 综合得分 = 各指标隶属度的等权平均",
        ]



@register("topsis")
def _branch_topsis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"TOPSIS 失败：{err}")
    else:
        import pandas as pd

        cost_mask, cost_names = _cost_mask(crit, cfg)
        Z = _minmax01(X, cost_mask)  # benefit-normalised to [0,1] (cost cols flipped)
        w = _entropy_weights(Z)
        V = Z * w
        a_best, a_worst = V.max(axis=0), V.min(axis=0)
        dp = np.sqrt(((V - a_best) ** 2).sum(axis=1))
        dn = np.sqrt(((V - a_worst) ** 2).sum(axis=1))
        score = dn / (dp + dn + 1e-12)
        res = pd.DataFrame({"alternative": labels, "score": np.round(score, 4)})
        res["rank"] = res["score"].rank(ascending=False, method="min").astype(int)
        res = res.sort_values("rank").reset_index(drop=True)
        res.to_csv(d / "topsis_scores.csv", index=False, encoding="utf-8")
        files.append("topsis_scores.csv")
        pd.DataFrame({"criterion": crit, "entropy_weight": np.round(w, 4)}).to_csv(
            d / "weights.csv", index=False, encoding="utf-8"
        )
        files.append("weights.csv")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = res.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.32)))
            ax.barh(top["alternative"].astype(str), top["score"], color="#4C72B0")
            ax.set_xlabel("TOPSIS closeness score")
            ax.set_title(f"Entropy-weighted TOPSIS ranking (top {len(top)})")
            fig.tight_layout()
            fig.savefig(d / "topsis_ranking.png", dpi=150)
            plt.close(fig)
            files.append("topsis_ranking.png")
        except Exception:
            pass
        best = labels[int(np.argmax(score))]
        estimates["top_score"] = round(float(score.max()), 4)
        estimates["n_alternatives"] = float(len(labels))
        estimates["n_criteria"] = float(len(crit))
        summary.append(
            f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
            f"最优方案 [{best}]（贴近度 {score.max():.3f}）；熵权见 weights.csv。"
            + _mcda_direction_note(cost_names)
        )
        code += [
            "import numpy as np  # 熵权-TOPSIS",
            "# Z=min-max[0,1]; w=熵权; V=Z*w; 距理想最优/最劣 -> 贴近度 dn/(dp+dn)",
        ]


@register("entropy_weight")
def _branch_entropy_weight(ctx: Ctx) -> None:
    # Shannon-entropy OBJECTIVE weighting (Shannon 1948; standard MCDA entropy weight
    # method, e.g. Zhu, Tian & Yan 2020, "Effectiveness of entropy weight method").
    # p_ij = x_ij / Σ_i x_ij on benefit-normalised [0,1] columns;
    # e_j = -k Σ_i p_ij ln p_ij with k = 1/ln(m); d_j = 1 - e_j; w_j = d_j / Σ_j d_j.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"熵权法失败：{err}")
        return
    if len(labels) < 2:
        summary.append("熵权法跳过：方案数 < 2，熵无法定义（k=1/ln(m)）。")
        return
    import pandas as pd

    cost_mask, cost_names = _cost_mask(crit, cfg)
    # Shift to non-negative & benefit-orient via min-max so cost criteria are flipped
    # and proportions p_ij are well-defined (entropy needs non-negative data).
    Z = _minmax01(X, cost_mask)  # [0,1], cost cols flipped
    # _entropy_weights implements exactly: P=Z/Σ_i Z; e=-Σ P lnP / ln(m); w∝1-e.
    w = _entropy_weights(Z)
    composite = Z @ w  # weighted-sum score on normalised criteria
    res = pd.DataFrame({"alternative": labels, "entropy_score": np.round(composite, 4)})
    res["rank"] = res["entropy_score"].rank(ascending=False, method="min").astype(int)
    res = res.sort_values("rank").reset_index(drop=True)
    res.to_csv(d / "entropy_scores.csv", index=False, encoding="utf-8")
    files.append("entropy_scores.csv")
    pd.DataFrame({"criterion": crit, "entropy_weight": np.round(w, 4)}).to_csv(
        d / "weights.csv", index=False, encoding="utf-8"
    )
    files.append("weights.csv")
    _mcda_rank_plot(
        res, "entropy_score", "Entropy-weight ranking (top 20)", d / "entropy_ranking.png"
    )
    if (d / "entropy_ranking.png").exists():
        files.append("entropy_ranking.png")
    best = labels[int(np.argmax(composite))]
    estimates["top_score"] = round(float(composite.max()), 4)
    estimates["n_alternatives"] = float(len(labels))
    estimates["n_criteria"] = float(len(crit))
    estimates["max_weight"] = round(float(w.max()), 4)
    summary.append(
        f"{entry.method} 完成：{len(labels)} 个方案 × {len(crit)} 个指标；"
        f"最优 [{best}]（加权得分 {composite.max():.3f}）；熵权见 weights.csv。"
        "⚠ 熵权奖励『离散度高』的指标，不代表『重要』；对归一化方式敏感；需非负数据"
        "（已用 min-max 归一保非负）。与 CRITIC（基于相关性的客观赋权）不同：本法只看单指标离散度。"
        + _mcda_direction_note(cost_names)
    )
    code += [
        "import numpy as np  # 熵权法(Shannon entropy objective weighting)",
        "# p=Z/Σ_i Z; e_j=-Σ p lnp / ln(m); d_j=1-e_j; w_j=d_j/Σ d; score=Z@w",
    ]


@register("vikor")
def _branch_vikor(ctx: Ctx) -> None:
    # VIKOR compromise ranking (Opricovic 1998; Opricovic & Tzeng 2004, EJOR 156).
    # f*_j best, f^-_j worst per criterion (benefit/cost aware);
    # S_i = Σ_j w_j (f*_j - f_ij)/(f*_j - f^-_j); R_i = max_j of that weighted term;
    # Q_i = v (S_i-S*)/(S^- -S*) + (1-v)(R_i-R*)/(R^- -R*), v=cfg['v'] (default 0.5).
    # Acceptance: C1 advantage Q(a2)-Q(a1) >= 1/(m-1); C2 stability: Q-best also best in S or R.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"VIKOR 失败：{err}")
        return
    m = len(labels)
    if m < 2:
        summary.append("VIKOR 跳过：方案数 < 2，无法做折中排序。")
        return
    import pandas as pd

    cost_mask, cost_names = _cost_mask(crit, cfg)
    w, wnote = _resolve_weights(crit, cfg)
    cmask = (
        np.zeros(len(crit), dtype=bool) if cost_mask is None else np.asarray(cost_mask, bool)
    )
    # f*_j / f^-_j respecting direction: benefit -> best=max, worst=min; cost -> reversed.
    col_max, col_min = X.max(axis=0), X.min(axis=0)
    f_best = np.where(cmask, col_min, col_max)
    f_worst = np.where(cmask, col_max, col_min)
    denom = f_best - f_worst
    safe = np.where(denom == 0, 1.0, denom)
    # normalised regret per (i,j); constant criterion (denom=0) contributes 0.
    d_norm = np.where(denom == 0, 0.0, (f_best - X) / safe)
    weighted = d_norm * w
    S = weighted.sum(axis=1)  # group utility
    R = weighted.max(axis=1)  # individual regret (max term)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    sden = (S_minus - S_star) or 1.0
    rden = (R_minus - R_star) or 1.0
    v_raw = (cfg or {}).get("v", 0.5)
    try:
        v = float(v_raw)
        if not (0.0 <= v <= 1.0):
            v = 0.5
    except (TypeError, ValueError):
        v = 0.5
    Q = v * (S - S_star) / sden + (1.0 - v) * (R - R_star) / rden
    res = pd.DataFrame(
        {
            "alternative": labels,
            "S": np.round(S, 4),
            "R": np.round(R, 4),
            "Q": np.round(Q, 4),
        }
    )
    res["rank"] = res["Q"].rank(ascending=True, method="min").astype(int)  # smaller Q = better
    res = res.sort_values(["rank", "Q"]).reset_index(drop=True)
    res.to_csv(d / "vikor_scores.csv", index=False, encoding="utf-8")
    files.append("vikor_scores.csv")
    # res is sorted best-first (lowest Q); the shared plot reverses head(20) so the best
    # (shortest Q bar) lands at the top. Q is a "lower=better" score — noted in the title.
    _mcda_rank_plot(res, "Q", "VIKOR Q (lower = better, top 20)", d / "vikor_ranking.png")
    if (d / "vikor_ranking.png").exists():
        files.append("vikor_ranking.png")
    # Acceptance conditions (rank by Q ascending).
    order = np.argsort(Q, kind="stable")
    a1 = order[0]
    best_alt = labels[int(a1)]
    s_best_idx = int(np.argmin(S))
    r_best_idx = int(np.argmin(R))
    c2_stable = (a1 == s_best_idx) or (a1 == r_best_idx)
    if m >= 2:
        a2 = order[1]
        dq = Q[a2] - Q[a1]
        c1_advantage = dq >= (1.0 / (m - 1))
    else:
        dq, c1_advantage = float("nan"), True
    if c1_advantage and c2_stable:
        verdict = f"推荐唯一折中解：[{best_alt}]（C1 显著优势 & C2 稳定性 均满足）。"
        compromise_set = [best_alt]
    elif not c1_advantage:
        # set = a1..aM while Q(aM)-Q(a1) < 1/(m-1)
        thr = 1.0 / (m - 1)
        kset = [int(order[0])]
        for j in range(1, m):
            if Q[order[j]] - Q[a1] < thr:
                kset.append(int(order[j]))
            else:
                break
        compromise_set = [labels[i] for i in kset]
        verdict = (
            f"C1（显著优势）不满足 -> 推荐折中解集合：{compromise_set}"
            f"（Q 差 < 1/(m-1)={thr:.3f}）。"
        )
    else:  # C1 ok but C2 fails
        compromise_set = [labels[int(a1)], labels[int(order[1])]]
        verdict = (
            f"C2（可接受稳定性）不满足 -> 推荐折中解集合：{compromise_set}"
            "（Q 最优者在 S/R 上均非最优）。"
        )
    estimates["top_Q"] = round(float(Q.min()), 4)
    estimates["n_alternatives"] = float(m)
    estimates["n_criteria"] = float(len(crit))
    estimates["v_strategy"] = round(v, 3)
    estimates["c1_advantage"] = float(bool(c1_advantage))
    estimates["c2_stability"] = float(bool(c2_stable))
    estimates["n_compromise"] = float(len(compromise_set))
    summary.append(
        f"{entry.method} 完成：{m} 个方案 × {len(crit)} 个指标；v={v}（策略权重）；"
        f"Q 最优 [{best_alt}]（Q={Q.min():.3f}）。{verdict} "
        "⚠ VIKOR 给『折中解』（到理想点的折中接近度），非简单加权和；"
        "v=0.5 为共识策略，v→1 偏多数效用、v→0 偏个体最大遗憾（否决式）；"
        "两条接受条件决定是唯一解还是解集。" + wnote
        + _mcda_direction_note(cost_names)
    )
    code += [
        "import numpy as np  # VIKOR 折中排序(Opricovic & Tzeng 2004)",
        "# S=Σ w(f*-f)/(f*-f^-); R=max_j 该项; Q=v(S-S*)/(S^- -S*)+(1-v)(R-R*)/(R^- -R*)",
        "# 接受条件 C1: Q(a2)-Q(a1)>=1/(m-1); C2: Q 最优者亦为 S 或 R 最优",
    ]


@register("promethee")
def _branch_promethee(ctx: Ctx) -> None:
    # PROMETHEE II complete ranking (Brans & Vincke 1985; Brans, Vincke & Mareschal 1986).
    # Per ordered pair (a,b) & criterion j: deviation d = oriented diff (benefit/cost);
    # preference P_j(a,b) via the LINEAR (V-shape, Type V) preference function with
    # indifference q and preference p thresholds (default q=0 -> reduces to V-shape):
    #   P = 0 if d<=q; (d-q)/(p-q) if q<d<p; 1 if d>=p.
    # π(a,b)=Σ_j w_j P_j(a,b); φ+ (a)=mean_b π(a,b); φ- (a)=mean_b π(b,a); φ=φ+ - φ-.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"PROMETHEE 失败：{err}")
        return
    m = len(labels)
    if m < 2:
        summary.append("PROMETHEE 跳过：方案数 < 2，无法成对比较。")
        return
    import pandas as pd

    cost_mask, cost_names = _cost_mask(crit, cfg)
    w, wnote = _resolve_weights(crit, cfg)
    ncrit = len(crit)
    cmask = (
        np.zeros(ncrit, dtype=bool) if cost_mask is None else np.asarray(cost_mask, bool)
    )
    # Orient so larger oriented value = better (flip cost columns): then d=f(a)-f(b)>0
    # means a is preferred to b on criterion j.
    Xo = np.where(cmask, -X, X)
    # Preference-function thresholds. Default linear/V-shape: q=0 and p = the per-criterion
    # value range (so a full-range advantage gives P=1). cfg can override globally
    # (cfg['q'], cfg['p']) or per-criterion (lists aligned with crit).
    rng = Xo.max(axis=0) - Xo.min(axis=0)
    p = np.where(rng == 0, 1.0, rng).astype(float)
    q = np.zeros(ncrit, dtype=float)
    func_note = "线性(V-shape)偏好函数，q=0、p=各指标极差"

    def _vec(key):
        val = (cfg or {}).get(key)
        if val is None:
            return None
        try:
            if isinstance(val, (list, tuple)):
                arr = np.asarray([float(x) for x in val], dtype=float)
                return arr if arr.shape[0] == ncrit else None
            return np.full(ncrit, float(val))
        except (TypeError, ValueError):
            return None

    qv, pv = _vec("q"), _vec("p")
    if qv is not None:
        q = np.maximum(qv, 0.0)
        func_note = "线性(V-shape)偏好函数，q=config['q']"
    if pv is not None:
        p = np.where(pv <= q, q + 1e-12, pv)
        func_note = func_note.replace("p=各指标极差", "p=config['p']")
        if qv is None:
            func_note = "线性(V-shape)偏好函数，q=0、p=config['p']"
    pq = np.where((p - q) == 0, 1.0, p - q)
    # Pairwise preference tensor P[a,b,j] = P_j(a,b) on oriented deviations.
    diff = Xo[:, None, :] - Xo[None, :, :]  # d_ab,j = f_a - f_b
    Pj = np.clip((diff - q) / pq, 0.0, 1.0)  # linear preference (0 below q, 1 at/above p)
    pi = (Pj * w).sum(axis=2)  # aggregated preference index π(a,b); diag π(a,a)=0 since d=0
    # Leaving / entering flows averaged over the OTHER m-1 alternatives.
    denom = m - 1
    phi_plus = pi.sum(axis=1) / denom  # Σ_b π(a,b)/(m-1) (diag is 0)
    phi_minus = pi.sum(axis=0) / denom  # Σ_b π(b,a)/(m-1)
    phi = phi_plus - phi_minus  # net flow (PROMETHEE II complete preorder)
    res = pd.DataFrame(
        {
            "alternative": labels,
            "phi_plus": np.round(phi_plus, 4),
            "phi_minus": np.round(phi_minus, 4),
            "phi_net": np.round(phi, 4),
        }
    )
    res["rank"] = res["phi_net"].rank(ascending=False, method="min").astype(int)
    res = res.sort_values("rank").reset_index(drop=True)
    res.to_csv(d / "promethee_flows.csv", index=False, encoding="utf-8")
    files.append("promethee_flows.csv")
    _mcda_rank_plot(
        res, "phi_net", "PROMETHEE II net flow (top 20)", d / "promethee_ranking.png"
    )
    if (d / "promethee_ranking.png").exists():
        files.append("promethee_ranking.png")
    best = labels[int(np.argmax(phi))]
    estimates["top_net_flow"] = round(float(phi.max()), 4)
    estimates["n_alternatives"] = float(m)
    estimates["n_criteria"] = float(ncrit)
    summary.append(
        f"{entry.method} 完成：{m} 个方案 × {len(crit)} 个指标；"
        f"净流量最优 [{best}]（φ={phi.max():.3f}）；流量见 promethee_flows.csv。"
        f"偏好函数：{func_note}。"
        "⚠ PROMETHEE II 以净流量 φ=φ⁺-φ⁻ 给出完全偏序；偏好函数与阈值(q/p)主导结果；"
        "净流量会掩盖 PROMETHEE I（偏序）能揭示的不可比性。" + wnote
        + _mcda_direction_note(cost_names)
    )
    code += [
        "import numpy as np  # PROMETHEE II(Brans & Vincke 1985)",
        "# P_j(a,b)=clip((d-q)/(p-q),0,1); π=Σ w P; φ+=mean_b π(a,b); φ-=mean_b π(b,a)",
        "# 完全排序按净流量 φ=φ+ - φ-",
    ]


@register("ahp")
def _branch_ahp(ctx: Ctx) -> None:
    # Analytic Hierarchy Process for CRITERIA WEIGHTS (Saaty 1980, 1987).
    # weights = principal right eigenvector (normalised) of the n×n reciprocal pairwise
    # matrix; λmax = max eigenvalue; CI=(λmax-n)/(n-1); CR=CI/RI (Saaty random index).
    # cfg['pairwise'] = expert n×n reciprocal matrix; else DATA PROXY a_ij = mean_i/mean_j.
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np

    try:
        X, crit, labels = _mcda_inputs(df, fp)
    except ValueError as err:
        summary.append(f"AHP 失败：{err}")
        return
    if len(labels) < 2:
        summary.append("AHP 跳过：方案数 < 2，无法排序。")
        return
    import pandas as pd

    n = len(crit)
    # Saaty random consistency index RI for n=1..10 (Saaty 1980).
    RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    cost_mask, cost_names = _cost_mask(crit, cfg)

    proxy = True
    A = None
    raw_pw = (cfg or {}).get("pairwise")
    if raw_pw is not None:
        try:
            cand = np.asarray(raw_pw, dtype=float)
            if cand.shape == (n, n) and np.all(cand > 0):
                A = cand
                proxy = False
        except (TypeError, ValueError):
            A = None
    if A is None:
        # DATA-DRIVEN PROXY (not real AHP): a_ij = mean(crit_i)/mean(crit_j) on
        # benefit-oriented, non-negative columns -> a perfectly consistent (rank-1)
        # matrix. Disclosed clearly: real AHP needs expert judgments.
        Zc = _minmax01(X, cost_mask)
        means = Zc.mean(axis=0)
        means = np.where(means <= 0, 1e-9, means)
        A = means[:, None] / means[None, :]
        proxy = True

    # Principal right eigenvector via the real eigenvalue with the largest real part.
    eigvals, eigvecs = np.linalg.eig(A)
    k = int(np.argmax(eigvals.real))
    lam_max = float(eigvals[k].real)
    vec = np.abs(eigvecs[:, k].real)
    w = vec / vec.sum() if vec.sum() > 0 else np.ones(n) / n
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    Z = _minmax01(X, cost_mask)
    composite = Z @ w
    res = pd.DataFrame({"alternative": labels, "ahp_score": np.round(composite, 4)})
    res["rank"] = res["ahp_score"].rank(ascending=False, method="min").astype(int)
    res = res.sort_values("rank").reset_index(drop=True)
    res.to_csv(d / "ahp_scores.csv", index=False, encoding="utf-8")
    files.append("ahp_scores.csv")
    pd.DataFrame({"criterion": crit, "ahp_weight": np.round(w, 4)}).to_csv(
        d / "weights.csv", index=False, encoding="utf-8"
    )
    files.append("weights.csv")
    _mcda_rank_plot(res, "ahp_score", "AHP-weighted ranking (top 20)", d / "ahp_ranking.png")
    if (d / "ahp_ranking.png").exists():
        files.append("ahp_ranking.png")
    best = labels[int(np.argmax(composite))]
    estimates["top_score"] = round(float(composite.max()), 4)
    estimates["n_alternatives"] = float(len(labels))
    estimates["n_criteria"] = float(n)
    estimates["lambda_max"] = round(lam_max, 4)
    estimates["CI"] = round(float(CI), 4)
    estimates["CR"] = round(float(CR), 4)
    src = "数据代理矩阵(均值比)" if proxy else "用户 config['pairwise'] 专家判断矩阵"
    cr_warn = "⚠ CR>0.1，判断矩阵不一致，权重不可靠。" if CR > 0.1 else "CR≤0.1，一致性可接受。"
    proxy_warn = (
        "⚠ 未提供 config['pairwise']：本结果用『指标均值比』构造的数据代理矩阵"
        "（rank-1，故 CR≈0），并非真正 AHP——真正 AHP 需专家成对判断。"
        if proxy
        else ""
    )
    judg = "数据代理（指标均值比）" if proxy else "用户提供的专家成对判断"
    summary.append(
        f"{entry.method} 完成：{len(labels)} 个方案 × {n} 个指标；权重来源={src}；"
        f"λmax={lam_max:.3f}，CI={CI:.3f}，CR={CR:.3f}；{cr_warn}"
        f"最优 [{best}]（加权得分 {composite.max():.3f}）；权重见 weights.csv。"
        f"{proxy_warn}"
        f"⚠ AHP 权重编码主观成对判断（此处为{judg}）；CR>0.1 表示判断不一致；"
        "AHP 存在已知的『等级逆转(rank reversal)』批评。" + _mcda_direction_note(cost_names)
    )
    code += [
        "import numpy as np  # AHP(Saaty 1980) 主特征向量法",
        "# w=主右特征向量(归一); λmax; CI=(λmax-n)/(n-1); CR=CI/RI; CR>0.1 不一致",
    ]

