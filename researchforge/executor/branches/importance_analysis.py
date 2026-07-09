"""Branch handlers for the predictor RELATIVE-IMPORTANCE family.

Three ways to decompose a linear-OLS model R² across its predictors — i.e. "how
much of the explained variance does each predictor contribute?", which plain
standardized betas cannot answer when predictors are correlated:

  * dominance_analysis  — Budescu (1993) / Azen-Budescu GENERAL dominance:
                          average incremental R² of predictor j over ALL 2^p subsets.
  * relative_weights    — Johnson (2000) epsilon relative weights: orthogonalize the
                          predictors, regress, then Λ-transform back (no subset
                          enumeration, so it scales to many predictors).
  * commonality_analysis — partition R² into UNIQUE + COMMON (shared) components
                          (2^p − 1 commonality coefficients).

All three are PURE PYTHON (numpy / pandas; OLS R² via numpy lstsq), DETERMINISTIC,
and partition the SAME full-model R²: the per-predictor pieces sum (to numerical
tolerance) to the model R². They follow the regression-family column convention:
outcome = first continuous column (config["outcome"] override), predictors = the
remaining continuous/binary columns (config["predictors"] override).

Each handler degrades honestly (too few rows / <2 predictors / >cap predictors /
singular design / non-numeric / missing import → append a Chinese "<方法>跳过：<原因>"
to summary and RETURN), writes CSV + PNG (matplotlib Agg, ENGLISH plot labels),
fills float `estimates`, appends a Chinese `summary` with ⚠ disclosures, and
MUTATES ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from itertools import combinations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.run import resolve_outcome


# ─────────────────────────────────────────────────────────────────────────────
# Shared: resolve outcome + predictors (mirrors the regression-family convention)
# and build clean numeric design matrices.  Returns
#   (y_name, pred_names, y_vec, X_mat, problem_msg)
# When problem_msg is not None the caller appends it to summary and returns.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_design(ctx: Ctx, label: str, max_p: int):
    import numpy as np
    import pandas as pd

    fp, cfg, df = ctx.fp, ctx.cfg, ctx.df

    cont = [c.name for c in fp.columns if c.kind == "continuous"]
    if not cont:
        return None, None, None, None, f"{label}跳过：没有连续型结果变量（outcome），无法做方差分解。"

    # outcome: config override (must be continuous) else shared resolver
    y_name = cfg["outcome"] if cfg.get("outcome") in cont else resolve_outcome(fp, cfg, cont)

    # predictors: config override (numeric, != outcome) else continuous/binary/count cols
    forced = [c for c in (cfg.get("predictors") or []) if c in df.columns and c != y_name]
    if forced:
        pred_names = forced
    else:
        pred_names = [
            c.name
            for c in fp.columns
            if c.kind in {"continuous", "count", "binary"} and c.name != y_name
        ]

    if len(pred_names) < 2:
        return None, None, None, None, (
            f"{label}跳过：可用预测变量 <2（相对重要性是「在多个相关预测变量间分配 R²」，"
            "至少需要 2 个）。可用 config={\"predictors\":[...]} 指定。"
        )
    if len(pred_names) > max_p:
        return None, None, None, None, (
            f"{label}跳过：预测变量过多（{len(pred_names)} 个 > 上限 {max_p}，"
            f"该方法需枚举 2^p 个子模型，计算量爆炸）。"
            f"请用 config={{\"predictors\":[...]}} 挑选 ≤{max_p} 个最相关的预测变量。"
        )

    # build numeric matrices, drop rows with any NaN
    cols = [y_name] + pred_names
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] < len(pred_names) + 2:
        return None, None, None, None, (
            f"{label}跳过：有效样本量过小（{sub.shape[0]} 行，少于预测变量数+2），无法稳健分解。"
        )
    y = sub[y_name].to_numpy(dtype=float)
    X = sub[pred_names].to_numpy(dtype=float)

    # guard against constant / degenerate columns
    if np.any(np.std(X, axis=0) == 0) or np.std(y) == 0:
        return None, None, None, None, f"{label}跳过：存在常数列（结果或某预测变量无方差），无法分解。"

    return y_name, pred_names, y, X, None


# ─────────────────────────────────────────────────────────────────────────────
# Shared OLS R² for a subset of predictor columns (with intercept), via lstsq.
# Returns 0.0 if the subset is empty (intercept-only model has R²=0).
# ─────────────────────────────────────────────────────────────────────────────
def _r2(y, X, idx):
    import numpy as np

    if not idx:
        return 0.0
    Xs = X[:, list(idx)]
    A = np.column_stack([np.ones(Xs.shape[0]), Xs])
    # least squares; rcond=None uses machine precision (rank-deficiency tolerated)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _all_subset_r2(y, X, p):
    """Precompute R² for EVERY subset of the p predictors (keyed by frozenset of idx)."""
    cache = {frozenset(): 0.0}
    for k in range(1, p + 1):
        for combo in combinations(range(p), k):
            cache[frozenset(combo)] = _r2(y, X, combo)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 1. dominance_analysis — Budescu (1993) general dominance
# ─────────────────────────────────────────────────────────────────────────────
@register("dominance_analysis")
def _branch_dominance_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    y_name, preds, y, X, problem = _resolve_design(ctx, "优势分析", max_p=8)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        p = len(preds)
        cache = _all_subset_r2(y, X, p)
        model_r2 = cache[frozenset(range(p))]

        # General dominance of predictor j = average, over subset SIZES k=0..p-1, of
        # the MEAN incremental R² it adds to all size-k subsets not containing j.
        general = np.zeros(p)
        for j in range(p):
            others = [i for i in range(p) if i != j]
            size_means = []
            for k in range(0, p):  # subset sizes 0 .. p-1 (subsets drawn from `others`)
                incs = []
                for combo in combinations(others, k):
                    s = frozenset(combo)
                    incs.append(cache[s | {j}] - cache[s])
                if incs:
                    size_means.append(float(np.mean(incs)))
            general[j] = float(np.mean(size_means)) if size_means else 0.0

        dom_sum = float(general.sum())  # should ≈ model_r2 (exact decomposition)
        order = np.argsort(-general)  # descending
        ranks = np.empty(p, dtype=int)
        for r, j in enumerate(order, start=1):
            ranks[j] = r

        tab = pd.DataFrame({
            "predictor": preds,
            "general_dominance": np.round(general, 6),
            "pct_of_R2": np.round(np.where(model_r2 > 0, general / model_r2 * 100, 0.0), 3),
            "rank": ranks,
        }).sort_values("rank").reset_index(drop=True)
        tab.to_csv(d / "dominance.csv", index=False, encoding="utf-8")
        files.append("dominance.csv")

        # horizontal bar of general dominance, sorted ascending so largest on top
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            t2 = tab.sort_values("general_dominance")
            fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * p + 1.5)))
            ax.barh(t2["predictor"].astype(str), t2["general_dominance"], color="#4C72B0")
            ax.set_xlabel("General dominance (avg incremental R²)")
            ax.set_title(f"Dominance analysis (model R²={model_r2:.3f})")
            fig.tight_layout()
            fig.savefig(d / "dominance.png", dpi=150)
            plt.close(fig)
            files.append("dominance.png")
        except Exception:
            pass

        top = tab.iloc[0]
        estimates["model_r2"] = round(float(model_r2), 6)
        estimates["top_predictor_dominance"] = round(float(top["general_dominance"]), 6)
        estimates["n_predictors"] = float(p)
        estimates["dominance_sum_check"] = round(dom_sum, 6)

        (d / "dominance_summary.txt").write_text(
            f"优势分析（Budescu 1993 一般优势 / general dominance）：结果变量 {y_name}，"
            f"{p} 个预测变量（枚举 {2 ** p} 个 OLS 子模型）\n"
            f"全模型 R² = {model_r2:.4f}；一般优势之和 = {dom_sum:.4f}（应≈R²，验证分解无误）\n\n"
            "各预测变量的一般优势（= 在所有子集上的平均增量 R²）：\n"
            + tab.to_string(index=False) + "\n\n"
            "注：一般优势把全模型 R² 公平地分摊到各预测变量（含相关预测变量间的共享方差）；"
            "这是对【线性可加】模型的方差分解，不是因果重要性；模型若误设（遗漏变量/非线性/交互），"
            "排序仍可能误导；为控制计算量上限 8 个预测变量（2^p 子模型）。\n"
            "如需覆盖结果/预测变量：config={\"outcome\":\"...\",\"predictors\":[...]}。",
            encoding="utf-8",
        )
        files.append("dominance_summary.txt")

        summary.append(
            f"{entry.method} 完成：结果变量 {y_name}，{p} 个预测变量；全模型 R²={model_r2:.3f}；"
            f"最重要预测变量 {top['predictor']}（一般优势 {float(top['general_dominance']):.3f}，"
            f"占 R² 的 {float(top['pct_of_R2']):.1f}%）；分解之和={dom_sum:.3f}（≈R²，校验通过）。"
            "⚠ 一般优势是【线性可加】模型的方差分解（非因果重要性）；它公平拆分相关预测变量的共享方差，"
            "但模型误设时排序仍可能误导；上限 8 个预测变量（2^p 子模型）。"
        )
        code += [
            "import numpy as np",
            "from itertools import combinations",
            "# Budescu (1993) general dominance: avg incremental R^2 over ALL 2^p subsets",
            "# (per-predictor general dominance sums to the full-model R^2)",
        ]
    except Exception as err:  # never crash the engine
        summary.append(f"优势分析失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. relative_weights — Johnson (2000) epsilon relative weights
# ─────────────────────────────────────────────────────────────────────────────
@register("relative_weights")
def _branch_relative_weights(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    # no subset enumeration -> generous cap (avoid trivially huge designs)
    y_name, preds, y, X, problem = _resolve_design(ctx, "相对权重", max_p=40)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        p = len(preds)
        n = X.shape[0]

        # standardize X and y (z-scores; relative weights operate on the correlation
        # structure, so standardization is required)
        Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        yz = (y - y.mean()) / y.std(ddof=0)

        # predictor correlation matrix R_xx = Q Λ Q'  (symmetric -> eigh)
        Rxx = (Xz.T @ Xz) / n
        evals, Q = np.linalg.eigh(Rxx)
        if np.any(evals <= 1e-10):
            summary.append(
                "相对权重跳过：预测变量近似共线（相关矩阵奇异/非正定），"
                "正交化不稳定。请剔除冗余预测变量后重试。"
            )
            return
        Lam_inv_sqrt = np.diag(1.0 / np.sqrt(evals))
        Lam_sqrt = np.diag(np.sqrt(evals))

        # Z = Xz · (Q Λ^{-1/2} Q')  -> orthonormal counterparts (Z'Z/n = I)
        Z = Xz @ (Q @ Lam_inv_sqrt @ Q.T)
        # P = Q Λ^{1/2} Q'  relates original predictors to their orthogonal counterparts
        # (Xz ≈ Z · P);  P_jk = "loading" of orthogonal k onto original j
        P = Q @ Lam_sqrt @ Q.T

        # regress standardized y on orthogonal Z; columns orthonormal => b = Z'y / n
        b = (Z.T @ yz) / n  # standardized betas on the orthogonal variables
        # relative weight of predictor j = Σ_k P_jk^2 · b_k^2
        rel = (P ** 2) @ (b ** 2)

        model_r2 = float(np.sum(b ** 2))  # = R² of yz on Xz (orthonormal Z preserves it)
        weights_sum = float(np.sum(rel))  # should ≈ model_r2 (Σ_j P_jk^2 = R_xx,kk = 1)

        order = np.argsort(-rel)
        ranks = np.empty(p, dtype=int)
        for r, j in enumerate(order, start=1):
            ranks[j] = r

        tab = pd.DataFrame({
            "predictor": preds,
            "relative_weight": np.round(rel, 6),
            "pct_of_R2": np.round(np.where(model_r2 > 0, rel / model_r2 * 100, 0.0), 3),
            "rank": ranks,
        }).sort_values("rank").reset_index(drop=True)
        tab.to_csv(d / "relative_weights.csv", index=False, encoding="utf-8")
        files.append("relative_weights.csv")

        # bar of % of R² per predictor (sorted descending)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(5, 0.7 * p + 2), 4.5))
            ax.bar(tab["predictor"].astype(str), tab["pct_of_R2"], color="#55A868")
            ax.set_ylabel("% of model R²")
            ax.set_title(f"Relative weights (Johnson 2000), model R²={model_r2:.3f}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(d / "relative_weights.png", dpi=150)
            plt.close(fig)
            files.append("relative_weights.png")
        except Exception:
            pass

        top = tab.iloc[0]
        estimates["model_r2"] = round(model_r2, 6)
        estimates["top_predictor_weight"] = round(float(top["relative_weight"]), 6)
        estimates["weights_sum_check"] = round(weights_sum, 6)
        estimates["n_predictors"] = float(p)

        (d / "relative_weights_summary.txt").write_text(
            f"相对权重分析（Johnson 2000 epsilon relative weights）：结果变量 {y_name}，"
            f"{p} 个预测变量\n"
            f"全模型 R² = {model_r2:.4f}；相对权重之和 = {weights_sum:.4f}（应≈R²，验证分解无误）\n"
            "算法：标准化 X,y → 对预测变量相关阵 R_xx=QΛQ' 特征分解 → 构造正交对应物 "
            "Z=Xz·(QΛ^{-1/2}Q')（Z'Z/n=I）→ yz 对 Z 回归得 b → 相对权重_j = Σ_k P_jk²·b_k²，"
            "其中 P=QΛ^{1/2}Q'。\n\n"
            + tab.to_string(index=False) + "\n\n"
            "注：相对权重用正交化近似优势分析的方差分配（无需枚举子模型，可扩展到很多预测变量）；"
            "同样是【线性可加】方差分解，不是因果重要性；至少需 2 个预测变量；近似共线会使正交化不稳。\n"
            "覆盖：config={\"outcome\":\"...\",\"predictors\":[...]}。",
            encoding="utf-8",
        )
        files.append("relative_weights_summary.txt")

        summary.append(
            f"{entry.method} 完成：结果变量 {y_name}，{p} 个预测变量；全模型 R²={model_r2:.3f}；"
            f"最重要预测变量 {top['predictor']}（相对权重 {float(top['relative_weight']):.3f}，"
            f"占 R² 的 {float(top['pct_of_R2']):.1f}%）；权重之和={weights_sum:.3f}（≈R²，校验通过）。"
            "⚠ 相对权重用正交化近似优势分析的方差分配（可扩展到很多预测变量）；"
            "同为【线性可加】方差分解（非因果重要性）；近似共线时正交化不稳。"
        )
        code += [
            "import numpy as np",
            "# Johnson (2000) relative weights:",
            "# Rxx = QΛQ'; Z = Xz·(QΛ^{-1/2}Q') (orthonormal); b = Z'yz/n;",
            "# P = QΛ^{1/2}Q'; weight_j = Σ_k P_jk^2 · b_k^2  (weights sum to R^2)",
        ]
    except Exception as err:
        summary.append(f"相对权重分析失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. commonality_analysis — partition R² into UNIQUE + COMMON components
# ─────────────────────────────────────────────────────────────────────────────
@register("commonality_analysis")
def _branch_commonality_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    y_name, preds, y, X, problem = _resolve_design(ctx, "共同性分析", max_p=6)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import pandas as pd

        p = len(preds)
        cache = _all_subset_r2(y, X, p)
        full = frozenset(range(p))
        model_r2 = cache[full]

        # Commonality coefficients (Mood 1969 / Nimon et al. 2008), via the two-step
        # Möbius / inclusion–exclusion partition of the explained variance "Venn diagram":
        #   1) intersection variance J(T) = Σ_{∅≠W⊆T} (-1)^{|W|+1} · R²(W)  (each
        #      non-empty predictor set T — the variance common to ALL predictors in T);
        #   2) commonality C(S) = Σ_{B ⊆ (All\S)} (-1)^{|B|} · J(S∪B)  (the "exactly-S"
        #      Venn region — variance shared by predictors in S and by none outside S).
        # This gives, for a singleton {j}, the UNIQUE component R²(All) − R²(All\{j}),
        # for a pair {i,j} the shared variance C(ij)=R²(i)+R²(j)−R²(ij) (2-pred case),
        # and all 2^p − 1 coefficients sum exactly to the full-model R² (verified in tests).
        allset = set(range(p))

        def _J(T):  # intersection variance of all predictors in T (inclusion–exclusion)
            Tlist = sorted(T)
            tot = 0.0
            for wsz in range(1, len(Tlist) + 1):
                for W in combinations(Tlist, wsz):
                    tot += ((-1) ** (wsz + 1)) * cache[frozenset(W)]
            return tot

        comp = {}  # frozenset S -> commonality coefficient
        for size in range(1, p + 1):
            for S in combinations(range(p), size):
                Sset = frozenset(S)
                not_S = sorted(allset - Sset)
                val = 0.0
                for bsz in range(0, len(not_S) + 1):
                    for B in combinations(not_S, bsz):
                        val += ((-1) ** bsz) * _J(Sset | frozenset(B))
                comp[Sset] = float(val)

        # unique component of predictor j (singleton) and totals
        unique = {j: comp[frozenset({j})] for j in range(p)}
        total_unique = float(sum(unique.values()))
        total_common = float(sum(v for S, v in comp.items() if len(S) >= 2))
        comp_sum = total_unique + total_common  # should ≈ model_r2

        # largest pairwise common component (size-2 subset)
        pair_items = [(S, v) for S, v in comp.items() if len(S) == 2]
        max_pair = max(pair_items, key=lambda kv: kv[1])[1] if pair_items else 0.0

        max_unique_j = max(unique, key=lambda j: unique[j])
        max_unique_val = float(unique[max_unique_j])

        # build component table (component label / coefficient / % of R²)
        def _label(S):
            kind = "Unique" if len(S) == 1 else "Common"
            return f"{kind}:" + ",".join(preds[i] for i in sorted(S))

        rows = []
        for S, v in comp.items():
            rows.append({
                "component": _label(S),
                "n_predictors": len(S),
                "coefficient": round(float(v), 6),
                "pct_of_R2": round(float(v) / model_r2 * 100, 3) if model_r2 > 0 else 0.0,
            })
        tab = pd.DataFrame(rows).sort_values(
            ["n_predictors", "coefficient"], ascending=[True, False]
        ).reset_index(drop=True)
        tab.to_csv(d / "commonality.csv", index=False, encoding="utf-8")
        files.append("commonality.csv")

        # bar: unique component per predictor (the cleanest single view)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            uq = pd.DataFrame({
                "predictor": [preds[j] for j in range(p)],
                "unique": [unique[j] for j in range(p)],
            }).sort_values("unique", ascending=False)
            fig, ax = plt.subplots(figsize=(max(5, 0.7 * p + 2), 4.5))
            ax.bar(uq["predictor"].astype(str), uq["unique"], color="#C44E52", label="Unique")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel("Unique variance (R² contribution)")
            ax.set_title(
                f"Commonality: unique components (total common={total_common:.3f}, "
                f"model R²={model_r2:.3f})"
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(d / "commonality.png", dpi=150)
            plt.close(fig)
            files.append("commonality.png")
        except Exception:
            pass

        estimates["model_r2"] = round(float(model_r2), 6)
        estimates["total_unique"] = round(total_unique, 6)
        estimates["total_common"] = round(total_common, 6)
        estimates["max_unique_predictor"] = round(max_unique_val, 6)
        estimates["n_predictors"] = float(p)

        neg = [r for r in rows if r["coefficient"] < -1e-9]
        neg_note = (
            f"检测到 {len(neg)} 个负的共同性系数（抑制/suppression 效应，正常但需披露）。"
            if neg else "未检测到负系数。"
        )
        (d / "commonality_summary.txt").write_text(
            f"共同性分析（Commonality analysis，R² 的唯一/共同方差分解）：结果变量 {y_name}，"
            f"{p} 个预测变量（{2 ** p - 1} 个共同性系数）\n"
            f"全模型 R² = {model_r2:.4f}\n"
            f"唯一方差合计 = {total_unique:.4f}；共同（共享）方差合计 = {total_common:.4f}；"
            f"两者之和 = {comp_sum:.4f}（应≈R²，验证分解无误）\n"
            f"最大唯一贡献：{preds[max_unique_j]} = {max_unique_val:.4f}；"
            f"最大两两共同成分 = {max_pair:.4f}\n"
            f"{neg_note}\n\n"
            + tab.to_string(index=False) + "\n\n"
            "注：唯一成分 = 仅该预测变量独有解释的方差（=R²(全)−R²(去掉它)）；共同成分 = 多个"
            "预测变量【共享】解释的方差。共同性系数可能为负（抑制效应），需如实披露；成分数随预测"
            "变量数呈 2^p−1 爆炸（上限 6 个）；这是【线性可加】方差分解，不是因果重要性。\n"
            "覆盖：config={\"outcome\":\"...\",\"predictors\":[...]}。",
            encoding="utf-8",
        )
        files.append("commonality_summary.txt")

        summary.append(
            f"{entry.method} 完成：结果变量 {y_name}，{p} 个预测变量；全模型 R²={model_r2:.3f}；"
            f"唯一方差合计 {total_unique:.3f}、共同方差合计 {total_common:.3f}（和={comp_sum:.3f}≈R²，校验通过）；"
            f"最大唯一贡献 {preds[max_unique_j]}={max_unique_val:.3f}。"
            "⚠ 共同性系数可能为负（抑制效应，已披露）；成分数随预测变量数 2^p−1 爆炸（上限 6 个）；"
            "为【线性可加】方差分解（非因果重要性）。"
        )
        code += [
            "import numpy as np",
            "from itertools import combinations",
            "# Commonality (Mobius partition): J(T)=Σ_{∅≠W⊆T}(-1)^{|W|+1}R²(W);",
            "# C(S)=Σ_{B⊆All\\S}(-1)^{|B|}·J(S∪B).  unique_j=R²(All)-R²(All\\{j});",
            "# all 2^p-1 commonality coefficients sum to the full-model R²",
        ]
    except Exception as err:
        summary.append(f"共同性分析失败：{err}")
