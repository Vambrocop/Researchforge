"""Branch handlers for the field_trials family — DESIGN-AWARE experimental-design
analysis (pure Python: statsmodels / scipy / numpy / pandas, no R). These are the
analysis counterparts to `cli design` (which generates the randomized layouts):
design → collect → analyze, with the blocking structure declared, not guessed.

  * rcbd_anova         — Randomized Complete Block Design: y ~ C(treatment)+C(block),
                         Type-II ANOVA. Treatment F tested against residual after blocks
                         absorb nuisance variation; reports the relative efficiency of
                         blocking vs a CRD, treatment means + Tukey HSD.
  * latin_square_anova — Latin square (two-way blocking by row & column): y ~
                         C(treatment)+C(row)+C(col). Requires a t×t square; treatment
                         tested against residual; means + Tukey HSD.
  * split_plot_anova   — Split-plot: a balanced classical ANOVA with the TWO correct
                         error strata — the whole-plot factor A is tested against the
                         whole-plot error (block×A), while the sub-plot factor B and the
                         A×B interaction are tested against the sub-plot (residual) error.
                         Getting these strata right is the whole point (a naive two-way
                         ANOVA tests A against the wrong error and is badly anticonservative).

Roles are DECLARED (config wins) and otherwise inferred from column-name hints +
low cardinality. Each handler degrades honestly (missing roles / too few levels /
unbalanced / singular / missing import -> append a Chinese "<方法>跳过/失败：<原因>"
and return — never crash), writes CSV + PNG (matplotlib Agg, ENGLISH labels), fills
float `estimates`, appends a Chinese `summary` ending with ⚠ disclosures, and MUTATES
ctx (never rebinds). See executor/_branch_api.py and CLAUDE.md.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# A factor may profile as count/id (Likert/integer-coded) — accept those kinds.
_FACTOR_KINDS = {"categorical", "binary", "count", "id"}
_MAX_LEVELS = 20  # field-trial factors are small; guard against an id column sneaking in

_BLOCK_HINTS = ("block", "blk", "rep", "replicate", "replication", "plot")
_TRT_HINTS = ("treatment", "trt", "treat", "variety", "cultivar", "hybrid", "dose", "fert")
_ROW_HINTS = ("row",)
_COL_HINTS = ("col", "column")
_WHOLE_HINTS = ("wholeplot", "whole", "main", "mainplot", "irrigation", "tillage")
_SUB_HINTS = ("subplot", "sub", "split")


def _continuous(fp) -> list[str]:
    excl = {fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind == "continuous" and c.name not in excl]


def _factors(fp, df, excl: set) -> list[str]:
    """Low-cardinality factor columns (2..~20 levels), excluding given names."""
    out = []
    for c in fp.columns:
        if c.kind not in _FACTOR_KINDS or c.name in excl:
            continue
        try:
            k = int(df[c.name].nunique(dropna=True))
        except Exception:
            continue
        if 2 <= k <= _MAX_LEVELS:
            out.append((c.name, k))
    out.sort(key=lambda t: t[1])
    return [n for n, _ in out]


def _pick(df, fp, hints, excl, pool=None):
    """Pick a factor by name hint first, else the lowest-cardinality remaining factor."""
    cands = pool if pool is not None else _factors(fp, df, excl)
    for name in cands:
        if name in excl:
            continue
        low = str(name).strip().lower()
        if any(h in low for h in hints):
            return name
    for name in cands:
        if name not in excl:
            return name
    return None


def _resolve_response(cfg, fp):
    cont = _continuous(fp)
    y = cfg.get("response") or cfg.get("outcome")
    if y in cont:
        return y
    return cont[0] if cont else None


def _safe_plot(fn):
    try:
        fn()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# (A) rcbd_anova — Randomized Complete Block Design
# ─────────────────────────────────────────────────────────────────────────────
@register("rcbd_anova")
def _branch_rcbd_anova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    y = _resolve_response(cfg, fp)
    block = cfg.get("block") if cfg.get("block") in df.columns else None
    trt = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    excl = {y, fp.unit_col, fp.time_col}
    if trt is None:
        trt = _pick(df, fp, _TRT_HINTS, excl | ({block} if block else set()))
    if block is None:
        block = _pick(df, fp, _BLOCK_HINTS, excl | ({trt} if trt else set()))

    if y is None or trt is None or block is None or len({y, trt, block}) < 3:
        summary.append(
            "RCBD 方差分析失败：需要 1 个连续结果 + 1 个处理因子 + 1 个区组因子。"
            'config={"response":..,"treatment":..,"block":..} 指定。'
        )
        return

    sub = df[[y, trt, block]].dropna()
    try:
        sub = sub.astype({y: float})
    except (TypeError, ValueError):
        summary.append(f"RCBD 方差分析失败：结果列 {y} 非数值。")
        return
    t, b, n = int(sub[trt].nunique()), int(sub[block].nunique()), int(len(sub))
    if t < 2 or b < 2:
        summary.append(f"RCBD 方差分析失败：处理水平={t}、区组数={b}（各需 ≥2）。")
        return
    if n <= t + b - 1:
        summary.append(f"RCBD 方差分析失败：有效行={n} 太少，无残差自由度。")
        return

    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        data = sub.rename(columns={y: "_y", trt: "_trt", block: "_blk"})
        data["_trt"] = data["_trt"].astype(str)
        data["_blk"] = data["_blk"].astype(str)
        model = ols("_y ~ C(_trt) + C(_blk)", data=data).fit()
        aov = sm.stats.anova_lm(model, typ=2)

        f_trt = float(aov.loc["C(_trt)", "F"])
        p_trt = float(aov.loc["C(_trt)", "PR(>F)"])
        f_blk = float(aov.loc["C(_blk)", "F"])
        p_blk = float(aov.loc["C(_blk)", "PR(>F)"])
        ss_trt = float(aov.loc["C(_trt)", "sum_sq"])
        ss_blk = float(aov.loc["C(_blk)", "sum_sq"])
        ss_res = float(aov.loc["Residual", "sum_sq"])
        df_res = float(aov.loc["Residual", "df"])
        ms_res = ss_res / df_res if df_res > 0 else float("nan")
        ss_total = ss_trt + ss_blk + ss_res
        eta_trt = ss_trt / ss_total if ss_total > 1e-12 else float("nan")
        partial_eta_trt = ss_trt / (ss_trt + ss_res) if (ss_trt + ss_res) > 1e-12 else float("nan")

        # Relative efficiency of blocking vs a completely randomized design (Fisher):
        # RE = [(b-1)·MS_block + b(t-1)·MS_error] / [(bt-1)·MS_error]; >1 means blocking helped.
        ms_blk = ss_blk / (b - 1) if b > 1 else float("nan")
        re_block = (
            ((b - 1) * ms_blk + b * (t - 1) * ms_res) / ((b * t - 1) * ms_res)
            if ms_res > 1e-12 else float("nan")
        )

        estimates.update({
            "f_treatment": f_trt, "p_treatment": p_trt,
            "f_block": f_blk, "p_block": p_blk,
            "eta_squared_treatment": eta_trt, "partial_eta_squared_treatment": partial_eta_trt,
            "relative_efficiency_blocking": float(re_block),
            "n_treatments": float(t), "n_blocks": float(b), "n": float(n),
        })

        means = sub.groupby(trt, observed=True)[y].agg(["mean", "std", "count"])
        means.to_csv(d / "treatment_means.csv", encoding="utf-8")
        files.append("treatment_means.csv")
        aov.to_csv(d / "rcbd_anova_table.csv", encoding="utf-8")
        files.append("rcbd_anova_table.csv")

        # Tukey HSD on treatment
        tukey_sig = []
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            tuk = pairwise_tukeyhsd(sub[y].to_numpy(float), sub[trt].astype(str).to_numpy())
            tk = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
            tk.to_csv(d / "treatment_tukey_hsd.csv", index=False, encoding="utf-8")
            files.append("treatment_tukey_hsd.csv")
            tukey_sig = [f"{r['group1']}↔{r['group2']}"
                         for _, r in tk.iterrows()
                         if str(r.get("reject")).strip().lower() in {"true", "1"}]
        except Exception:
            pass

        def _plot():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m, sd, c = means["mean"], means["std"], means["count"]
            se = sd / np.sqrt(c.clip(lower=1))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(range(len(m)), m.to_numpy(), yerr=(1.96 * se).to_numpy(),
                        fmt="o", capsize=4, color="#4C72B0")
            ax.set_xticks(range(len(m)))
            ax.set_xticklabels([str(i) for i in m.index], rotation=30, ha="right")
            ax.set_xlabel(f"treatment ({trt})")
            ax.set_ylabel(f"mean {y} (95% CI)")
            ax.set_title("RCBD treatment means")
            fig.tight_layout()
            fig.savefig(d / "rcbd_means.png", dpi=150)
            plt.close(fig)
        if _safe_plot(_plot):
            files.append("rcbd_means.png")

        code += [
            "import statsmodels.api as sm",
            "from statsmodels.formula.api import ols",
            f"m = ols('Q(\"{y}\") ~ C(Q(\"{trt}\")) + C(Q(\"{block}\")), data=df).fit()",
            "print(sm.stats.anova_lm(m, typ=2))",
        ]

        sig = "显著" if p_trt < 0.05 else "不显著"
        re_txt = (f"区组相对效率≈{re_block:.2f}（>1 说明区组化比完全随机设计更有效）"
                  if np.isfinite(re_block) else "")
        tuk_txt = f"；Tukey 显著处理对：{'、'.join(tukey_sig)}" if tukey_sig else ""
        summary.append(
            f"{entry.method}：结果={y}，处理={trt}（{t} 水平），区组={block}（{b} 区组）。"
            f"处理效应 F={f_trt:.3f}, p={p_trt:.4g}（{sig}），η²={eta_trt:.3f}、偏η²={partial_eta_trt:.3f}；"
            f"区组 F={f_blk:.3f}, p={p_blk:.4g}。{re_txt}{tuk_txt}。"
            " ⚠ RCBD 假定**处理×区组无交互**（区组只移除可加性的区组效应）；"
            "处理在各区组内随机分配；结果近似正态、方差齐。区组化的代价是少量误差自由度——"
            "区组相对效率<1 时说明区组化反而不划算。"
        )
    except Exception as e:
        summary.append(f"RCBD 方差分析失败：{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# (B) latin_square_anova — two-way blocking by row & column (t×t square)
# ─────────────────────────────────────────────────────────────────────────────
@register("latin_square_anova")
def _branch_latin_square_anova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    y = _resolve_response(cfg, fp)
    trt = cfg.get("treatment") if cfg.get("treatment") in df.columns else None
    row = cfg.get("row") if cfg.get("row") in df.columns else None
    col = cfg.get("col") if cfg.get("col") in df.columns else None
    excl = {y, fp.unit_col, fp.time_col}
    if row is None:
        row = _pick(df, fp, _ROW_HINTS, excl | {trt, col} - {None})
    if col is None:
        col = _pick(df, fp, _COL_HINTS, excl | {trt, row} - {None})
    if trt is None:
        trt = _pick(df, fp, _TRT_HINTS, excl | {row, col} - {None})

    if y is None or None in (trt, row, col) or len({y, trt, row, col}) < 4:
        summary.append(
            "拉丁方方差分析失败：需要 1 个连续结果 + 处理 + 行区组 + 列区组。"
            'config={"response":..,"treatment":..,"row":..,"col":..} 指定。'
        )
        return

    sub = df[[y, trt, row, col]].dropna()
    try:
        sub = sub.astype({y: float})
    except (TypeError, ValueError):
        summary.append(f"拉丁方方差分析失败：结果列 {y} 非数值。")
        return
    t = int(sub[trt].nunique())
    nr, nc, n = int(sub[row].nunique()), int(sub[col].nunique()), int(len(sub))
    if not (t == nr == nc) or t < 3:
        summary.append(
            f"拉丁方方差分析失败：需 t×t 方阵（处理={t}、行={nr}、列={nc} 须相等且 ≥3）。"
        )
        return
    if n < t * t:
        summary.append(f"拉丁方方差分析失败：有效行={n} < t²={t * t}（方阵不完整）。")
        return
    # Verify it is a REAL Latin square (not just matching marginals): t² distinct
    # row×col cells each with one observation, and each treatment appearing exactly
    # once per row and once per column. Otherwise treatment is not orthogonal to the
    # blocks and y~C(t)+C(r)+C(c) would silently report a confounded, non-LSD analysis.
    cells = sub.groupby([row, col], observed=True).size()
    per_row = sub.groupby([row, trt], observed=True).size()
    per_col = sub.groupby([col, trt], observed=True).size()
    if (len(cells) != t * t or bool((cells != 1).any())
            or len(per_row) != t * t or bool((per_row != 1).any())
            or len(per_col) != t * t or bool((per_col != 1).any())):
        summary.append(
            "拉丁方方差分析跳过：非真正拉丁方——需 t² 个不重复的 行×列 单元、每单元一次观测，"
            "且每个处理在每行、每列各恰好出现一次（否则处理与行/列区组不正交，分析会被混淆）。"
        )
        return

    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        data = sub.rename(columns={y: "_y", trt: "_t", row: "_r", col: "_c"})
        for cc in ("_t", "_r", "_c"):
            data[cc] = data[cc].astype(str)
        model = ols("_y ~ C(_t) + C(_r) + C(_c)", data=data).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        f_trt = float(aov.loc["C(_t)", "F"])
        p_trt = float(aov.loc["C(_t)", "PR(>F)"])
        ss_trt = float(aov.loc["C(_t)", "sum_sq"])
        ss_res = float(aov.loc["Residual", "sum_sq"])
        ss_total = float(aov["sum_sq"].sum())
        eta_trt = ss_trt / ss_total if ss_total > 1e-12 else float("nan")
        partial_eta = ss_trt / (ss_trt + ss_res) if (ss_trt + ss_res) > 1e-12 else float("nan")

        estimates.update({
            "f_treatment": f_trt, "p_treatment": p_trt,
            "f_row": float(aov.loc["C(_r)", "F"]), "p_row": float(aov.loc["C(_r)", "PR(>F)"]),
            "f_col": float(aov.loc["C(_c)", "F"]), "p_col": float(aov.loc["C(_c)", "PR(>F)"]),
            "eta_squared_treatment": eta_trt, "partial_eta_squared_treatment": partial_eta,
            "square_size": float(t), "n": float(n),
        })

        means = sub.groupby(trt, observed=True)[y].agg(["mean", "std", "count"])
        means.to_csv(d / "treatment_means.csv", encoding="utf-8")
        files.append("treatment_means.csv")
        aov.to_csv(d / "latin_square_anova_table.csv", encoding="utf-8")
        files.append("latin_square_anova_table.csv")

        tukey_sig = []
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            tuk = pairwise_tukeyhsd(sub[y].to_numpy(float), sub[trt].astype(str).to_numpy())
            tk = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
            tk.to_csv(d / "treatment_tukey_hsd.csv", index=False, encoding="utf-8")
            files.append("treatment_tukey_hsd.csv")
            tukey_sig = [f"{r['group1']}↔{r['group2']}"
                         for _, r in tk.iterrows()
                         if str(r.get("reject")).strip().lower() in {"true", "1"}]
        except Exception:
            pass

        def _plot():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            m, sd, c = means["mean"], means["std"], means["count"]
            se = sd / np.sqrt(c.clip(lower=1))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(range(len(m)), m.to_numpy(), yerr=(1.96 * se).to_numpy(),
                        fmt="o", capsize=4, color="#55A868")
            ax.set_xticks(range(len(m)))
            ax.set_xticklabels([str(i) for i in m.index], rotation=30, ha="right")
            ax.set_xlabel(f"treatment ({trt})")
            ax.set_ylabel(f"mean {y} (95% CI)")
            ax.set_title(f"Latin square ({t}×{t}) treatment means")
            fig.tight_layout()
            fig.savefig(d / "latin_square_means.png", dpi=150)
            plt.close(fig)
        if _safe_plot(_plot):
            files.append("latin_square_means.png")

        code += [
            "import statsmodels.api as sm",
            "from statsmodels.formula.api import ols",
            f"m = ols('Q(\"{y}\") ~ C(Q(\"{trt}\")) + C(Q(\"{row}\")) + C(Q(\"{col}\")), data=df).fit()",
            "print(sm.stats.anova_lm(m, typ=2))",
        ]

        sig = "显著" if p_trt < 0.05 else "不显著"
        tuk_txt = f"；Tukey 显著处理对：{'、'.join(tukey_sig)}" if tukey_sig else ""
        summary.append(
            f"{entry.method}：{t}×{t} 拉丁方，结果={y}，处理={trt}，行区组={row}、列区组={col}。"
            f"处理效应 F={f_trt:.3f}, p={p_trt:.4g}（{sig}），η²={eta_trt:.3f}、偏η²={partial_eta:.3f}。"
            f"{tuk_txt}。"
            " ⚠ 拉丁方同时控制**行、列两个方向**的变异，假定处理与行/列**无交互**且为真正的 t×t 完整方阵；"
            "残差自由度仅 (t−1)(t−2)，小方阵（t=3/4）功效低——必要时用重复拉丁方。"
        )
    except Exception as e:
        summary.append(f"拉丁方方差分析失败：{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# (C) split_plot_anova — balanced classical split-plot with TWO error strata
# ─────────────────────────────────────────────────────────────────────────────
@register("split_plot_anova")
def _branch_split_plot_anova(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    y = _resolve_response(cfg, fp)
    block = cfg.get("block") if cfg.get("block") in df.columns else None
    whole = cfg.get("whole_plot") if cfg.get("whole_plot") in df.columns else None
    subf = cfg.get("sub_plot") if cfg.get("sub_plot") in df.columns else None
    excl = {y, fp.unit_col, fp.time_col}
    if block is None:
        block = _pick(df, fp, _BLOCK_HINTS, excl | {whole, subf} - {None})
    if whole is None:
        whole = _pick(df, fp, _WHOLE_HINTS, excl | {block, subf} - {None})
    if subf is None:
        subf = _pick(df, fp, _SUB_HINTS, excl | {block, whole} - {None})

    if y is None or None in (block, whole, subf) or len({y, block, whole, subf}) < 4:
        summary.append(
            "裂区方差分析失败：需要 1 个连续结果 + 区组(重复) + 主区因子 + 裂区因子。"
            'config={"response":..,"block":..,"whole_plot":..,"sub_plot":..} 指定。'
        )
        return

    sub = df[[y, block, whole, subf]].dropna()
    try:
        sub = sub.astype({y: float})
    except (TypeError, ValueError):
        summary.append(f"裂区方差分析失败：结果列 {y} 非数值。")
        return

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats

        r = int(sub[block].nunique())
        a = int(sub[whole].nunique())
        bb = int(sub[subf].nunique())
        if r < 2 or a < 2 or bb < 2:
            summary.append(f"裂区方差分析失败：区组={r}、主区={a}、裂区={bb}（各需 ≥2）。")
            return

        # Balanced requirement: every (block, whole, sub) cell has the SAME count m>=1.
        counts = sub.groupby([block, whole, subf], observed=True)[y].size()
        if len(counts) != r * a * bb or counts.nunique() != 1:
            summary.append(
                "裂区方差分析跳过：经典裂区 SS 分解要求**平衡**设计——每个"
                "(区组×主区×裂区)单元观测数相等且齐全。当前不平衡，"
                "请用平衡数据或混合模型（MixedLM）替代。"
            )
            return
        m = int(counts.iloc[0])
        N = r * a * bb * m

        gm = float(sub[y].mean())

        def _ss(group_cols, mult):
            mu = sub.groupby(group_cols, observed=True)[y].mean()
            return mult * float(((mu - gm) ** 2).sum())

        # marginal cell means for the interaction-style SS (computed via residual identity)
        ybar_block = sub.groupby(block, observed=True)[y].mean()
        ybar_a = sub.groupby(whole, observed=True)[y].mean()
        ybar_b = sub.groupby(subf, observed=True)[y].mean()
        ybar_ra = sub.groupby([block, whole], observed=True)[y].mean()
        ybar_ab = sub.groupby([whole, subf], observed=True)[y].mean()

        ss_total = float(((sub[y] - gm) ** 2).sum())
        ss_block = a * bb * m * float(((ybar_block - gm) ** 2).sum())
        ss_a = r * bb * m * float(((ybar_a - gm) ** 2).sum())
        # whole-plot error = block×A interaction: Σ (ȳ_ra - ȳ_r - ȳ_a + ȳ)²  · b·m
        wp = ybar_ra.reset_index()
        wp.columns = [block, whole, "_m"]
        wp["_exp"] = wp[block].map(ybar_block).values + wp[whole].map(ybar_a).values - gm
        ss_wp_err = bb * m * float(((wp["_m"] - wp["_exp"]) ** 2).sum())
        ss_b = r * a * m * float(((ybar_b - gm) ** 2).sum())
        ab = ybar_ab.reset_index()
        ab.columns = [whole, subf, "_m"]
        ab["_exp"] = ab[whole].map(ybar_a).values + ab[subf].map(ybar_b).values - gm
        ss_ab = r * m * float(((ab["_m"] - ab["_exp"]) ** 2).sum())
        ss_sub_err = ss_total - ss_block - ss_a - ss_wp_err - ss_b - ss_ab

        df_block = r - 1
        df_a = a - 1
        df_wp_err = (r - 1) * (a - 1)
        df_b = bb - 1
        df_ab = (a - 1) * (bb - 1)
        df_sub_err = (N - 1) - df_block - df_a - df_wp_err - df_b - df_ab
        if df_wp_err <= 0 or df_sub_err <= 0:
            summary.append("裂区方差分析失败：误差自由度不足（增加区组/重复）。")
            return

        ms_a = ss_a / df_a
        ms_wp_err = ss_wp_err / df_wp_err
        ms_b = ss_b / df_b
        ms_ab = ss_ab / df_ab
        ms_sub_err = ss_sub_err / df_sub_err

        # The two error strata: A tested against whole-plot error; B & A×B against sub-plot error.
        f_a = ms_a / ms_wp_err if ms_wp_err > 1e-12 else float("nan")
        f_b = ms_b / ms_sub_err if ms_sub_err > 1e-12 else float("nan")
        f_ab = ms_ab / ms_sub_err if ms_sub_err > 1e-12 else float("nan")
        p_a = float(stats.f.sf(f_a, df_a, df_wp_err)) if np.isfinite(f_a) else float("nan")
        p_b = float(stats.f.sf(f_b, df_b, df_sub_err)) if np.isfinite(f_b) else float("nan")
        p_ab = float(stats.f.sf(f_ab, df_ab, df_sub_err)) if np.isfinite(f_ab) else float("nan")

        estimates.update({
            "f_whole_plot": float(f_a), "p_whole_plot": p_a,
            "f_sub_plot": float(f_b), "p_sub_plot": p_b,
            "f_interaction": float(f_ab), "p_interaction": p_ab,
            "ms_whole_plot_error": float(ms_wp_err), "ms_sub_plot_error": float(ms_sub_err),
            "df_whole_plot_error": float(df_wp_err), "df_sub_plot_error": float(df_sub_err),
            "n_blocks": float(r), "n_whole": float(a), "n_sub": float(bb), "n": float(N),
        })

        # full ANOVA table with the two error lines explicit
        rows = [
            ("Block (rep)", df_block, ss_block, ""),
            (f"WholePlot A ({whole})", df_a, ss_a, "vs WP error"),
            ("WholePlot error (Block×A)", df_wp_err, ss_wp_err, ""),
            (f"SubPlot B ({subf})", df_b, ss_b, "vs sub error"),
            ("A×B interaction", df_ab, ss_ab, "vs sub error"),
            ("SubPlot error", df_sub_err, ss_sub_err, ""),
        ]
        tab = pd.DataFrame(
            [{"source": s, "df": dfx, "sum_sq": ssx,
              "mean_sq": (ssx / dfx if dfx else float("nan")), "tested_against": ta}
             for s, dfx, ssx, ta in rows]
        )
        tab.to_csv(d / "split_plot_anova_table.csv", index=False, encoding="utf-8")
        files.append("split_plot_anova_table.csv")

        cell = sub.groupby([whole, subf], observed=True)[y].mean().unstack()
        cell.to_csv(d / "cell_means.csv", encoding="utf-8")
        files.append("cell_means.csv")

        def _plot():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            for lvl in cell.index:
                ax.plot([str(c) for c in cell.columns], cell.loc[lvl].to_numpy(),
                        marker="o", label=f"{whole}={lvl}")
            ax.set_xlabel(f"sub-plot ({subf})")
            ax.set_ylabel(f"mean {y}")
            ax.set_title("Split-plot interaction (A × B)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(d / "split_plot_interaction.png", dpi=150)
            plt.close(fig)
        if _safe_plot(_plot):
            files.append("split_plot_interaction.png")

        code += [
            "# Balanced split-plot: A vs whole-plot error (Block×A); B & A×B vs sub-plot error.",
            "import pandas as pd, numpy as np; from scipy import stats",
            "# (see split_plot_anova_table.csv for the full decomposition)",
        ]

        def _v(p):
            return "显著" if (np.isfinite(p) and p < 0.05) else "不显著"
        summary.append(
            f"{entry.method}：区组(重复)={block}（{r}），主区 A={whole}（{a}），裂区 B={subf}（{bb}），"
            f"每单元 {m} 次观测。**两个误差层**——主区 A 用**主区误差(区组×A)**检验："
            f"F={f_a:.3f}, p={p_a:.4g}（{_v(p_a)}，df={df_a},{df_wp_err}）；"
            f"裂区 B 用**裂区误差**：F={f_b:.3f}, p={p_b:.4g}（{_v(p_b)}）；"
            f"A×B 交互用裂区误差：F={f_ab:.3f}, p={p_ab:.4g}（{_v(p_ab)}）。"
            " ⚠ 经典裂区**要求平衡设计**；主区因子的检验功效天然低于裂区（主区误差自由度少）——"
            "把更重要、需更高精度的因子放裂区。**切忌用普通两因素 ANOVA**（会把 A 对着裂区误差检验、"
            "严重高估显著性）。固定效应 F 检验还假定主区内**复合对称**（各裂区水平等相关）——"
            "裂区水平>2 且相关结构复杂时，考虑 MixedLM 或 GG 类校正。结果近似正态、方差齐。"
        )
    except Exception as e:
        summary.append(f"裂区方差分析失败：{type(e).__name__}: {e}")
