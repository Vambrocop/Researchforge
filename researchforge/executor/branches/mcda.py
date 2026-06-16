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

