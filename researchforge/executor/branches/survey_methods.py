"""Branch handlers for the survey_methods family — design-based survey estimation.

Three design-based / weighting methods for complex-survey data (pure Python:
numpy / scipy / pandas; no R):

  * weighted_estimation — Horvitz-Thompson design-weighted population mean & total
    of a numeric variable using survey weights, with the design-based (ratio) SE,
    the Kish design effect (deff) and the effective sample size n_eff.
  * poststratification — adjust sample weights so the weighted category proportions
    of ONE post-strat variable match KNOWN population proportions (config pop_props).
  * raking — iterative proportional fitting (IPF) to calibrate weights so the
    weighted MARGINS of TWO+ categorical variables match known population margins
    simultaneously (config margins).

Each handler resolves its column roles from config + column kinds, degrades honestly
(missing required config — weights / pop_props / margins — non-positive weights, too
few rows, wrong kinds, missing import -> append a Chinese "<方法>跳过：<原因>（需 config[...]）"
and RETURN; NEVER fabricate population targets; never crash), writes CSV + PNG
(matplotlib Agg, ENGLISH plot labels), fills float `estimates`, appends a Chinese
`summary` ending with ⚠ disclosures, and MUTATES ctx (never rebinds).

See executor/_branch_api.py (Ctx fields) and CLAUDE.md. numpy/scipy/pandas installed.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Hints used to auto-detect a survey-weight column by name when config weight absent.
_WEIGHT_HINTS = ("weight", "wt", "pweight", "pwt", "finalwt", "sampwt", "svywt")


def _continuous_cols(fp, exclude=()):
    """Continuous numeric columns (excludes the profiler time_col & given names)."""
    return [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name != fp.time_col and c.name not in exclude
    ]


def _categorical_cols(fp, exclude=()):
    """Discrete grouping columns (categorical / binary / count / id) usable as strata.
    Excludes the profiler time_col & given names."""
    return [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary", "count", "id"}
        and c.name != fp.time_col
        and c.name not in exclude
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. weighted_estimation — Horvitz-Thompson design-weighted mean & total
# ─────────────────────────────────────────────────────────────────────────────
@register("weighted_estimation")
def _branch_weighted_estimation(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
    except Exception as err:  # pragma: no cover - numpy/pandas always present
        summary.append(f"设计加权估计跳过：缺少 numpy/pandas（{err}）。")
        return

    # ── resolve the VALUE column (config value, else first continuous) ──
    conts = _continuous_cols(fp)
    value = cfg.get("value")
    if value and value not in df.columns:
        value = None
    if value is None:
        value = conts[0] if conts else None
    if value is None:
        summary.append("设计加权估计跳过：找不到可估计的数值列（需 config[\"value\"] 指定连续变量）。")
        return

    # ── resolve the WEIGHT column ──
    #   config weight -> else a positive continuous column whose name hints weight
    #   -> else honest degrade (cannot invent weights).
    weight = cfg.get("weight")
    if weight and weight not in df.columns:
        weight = None
    if weight is None:
        for c in _continuous_cols(fp, exclude=(value,)):
            if any(h in c.lower() for h in _WEIGHT_HINTS):
                weight = c
                break
    if weight is None:
        summary.append(
            "设计加权估计跳过：未指定调查权重列，也没有名字含 weight/wt/pweight 的正值权重列"
            "（需 config[\"weight\"] 指定权重列）。"
        )
        return
    if weight == value:
        summary.append("设计加权估计跳过：权重列与数值列不能相同（需 config[\"weight\"] / config[\"value\"]）。")
        return

    # group/stratum column for optional per-group means (config group)
    group = cfg.get("group")
    if group and group not in df.columns:
        group = None

    try:
        cols = [value, weight] + ([group] if group else [])
        sub = df[cols].copy()
        sub[value] = pd.to_numeric(sub[value], errors="coerce")
        sub[weight] = pd.to_numeric(sub[weight], errors="coerce")
        sub = sub.dropna(subset=[value, weight])
        if len(sub) < 3:
            summary.append("设计加权估计跳过：有效行 <3，样本太少（检查 value/weight 列的缺失）。")
            return
        w = sub[weight].to_numpy(dtype=float)
        y = sub[value].to_numpy(dtype=float)
        if not np.all(w > 0):
            summary.append(
                "设计加权估计跳过：权重必须全为正（检测到 <=0 或缺失的权重）。"
                "请清洗权重或用 config[\"weight\"] 指定正确的权重列。"
            )
            return

        n = len(y)
        sw = float(w.sum())
        sw2 = float((w * w).sum())
        wmean = float((w * y).sum() / sw)
        wtotal = float((w * y).sum())

        # Design-based (ratio) variance of a weighted mean, with-replacement /
        # Kish approximation: treat z_i = w_i*(y_i - wmean) as the residual
        # contributions; Var(mean) = [n/(n-1)] * Σ z_i^2 / (Σ w_i)^2.
        z = w * (y - wmean)
        var_mean = (n / (n - 1)) * float((z * z).sum()) / (sw * sw)
        se_mean = float(np.sqrt(var_mean)) if var_mean > 0 else 0.0
        ci_low = wmean - 1.959963984540054 * se_mean
        ci_high = wmean + 1.959963984540054 * se_mean

        # Kish design effect from unequal weighting + effective sample size.
        deff = float(n * sw2 / (sw * sw))
        n_eff = float((sw * sw) / sw2)

        unweighted_mean = float(y.mean())

        # ── estimates table CSV ──
        est_df = pd.DataFrame(
            {
                "quantity": [
                    "n", "weighted_mean", "se_mean", "ci_low", "ci_high",
                    "weighted_total", "unweighted_mean", "design_effect", "n_eff",
                ],
                "value": [
                    n, wmean, se_mean, ci_low, ci_high,
                    wtotal, unweighted_mean, deff, n_eff,
                ],
            }
        )
        est_df.to_csv(d / "weighted_estimates.csv", index=False, encoding="utf-8")
        files.append("weighted_estimates.csv")

        # ── optional per-group weighted means ──
        group_df = None
        if group:
            rows = []
            for gval, gdf in sub.groupby(group):
                gw = gdf[weight].to_numpy(dtype=float)
                gy = gdf[value].to_numpy(dtype=float)
                if gw.sum() > 0:
                    rows.append(
                        {
                            "group": gval,
                            "n": int(len(gy)),
                            "weighted_mean": float((gw * gy).sum() / gw.sum()),
                            "weighted_total": float((gw * gy).sum()),
                            "sum_weights": float(gw.sum()),
                        }
                    )
            if rows:
                group_df = pd.DataFrame(rows)
                group_df.to_csv(d / "weighted_by_group.csv", index=False, encoding="utf-8")
                files.append("weighted_by_group.csv")

        # ── PNG: unweighted vs weighted mean (with CI on the weighted bar) ──
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            labels = ["unweighted mean", "weighted mean"]
            vals = [unweighted_mean, wmean]
            yerr = [0.0, 1.959963984540054 * se_mean]
            ax.bar(
                labels, vals, color=["#999999", "#4C72B0"],
                yerr=yerr, capsize=6, error_kw={"elinewidth": 1.5},
            )
            ax.set_ylabel(f"{value}")
            ax.set_title("Unweighted vs design-weighted mean (95% CI)")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            fig.savefig(d / "weighted_mean.png", dpi=150)
            plt.close(fig)
            files.append("weighted_mean.png")
        except Exception:
            pass

        estimates["weighted_mean"] = round(wmean, 6)
        estimates["ci_low"] = round(ci_low, 6)
        estimates["ci_high"] = round(ci_high, 6)
        estimates["se_mean"] = round(se_mean, 6)
        estimates["weighted_total"] = round(wtotal, 6)
        estimates["unweighted_mean"] = round(unweighted_mean, 6)
        estimates["design_effect"] = round(deff, 6)
        estimates["n_eff"] = round(n_eff, 4)
        estimates["n"] = float(n)

        deff_note = (
            f"设计效应 deff={round(deff, 3)}（>1 表示不等权抬高了方差，有效样本 n_eff="
            f"{round(n_eff, 1)} < n={n}）"
            if deff > 1.0001
            else f"设计效应 deff={round(deff, 3)}（≈1，权重近乎相等）"
        )
        (d / "weighted_summary.txt").write_text(
            f"设计加权估计（Horvitz-Thompson）：数值列 {value}，权重列 {weight}"
            + (f"，分组 {group}" if group else "") + "\n"
            f"有效样本 n={n}，Σw={round(sw, 3)}\n"
            f"加权均值 = Σwᵢyᵢ/Σwᵢ = {round(wmean, 6)}（95% CI [{round(ci_low, 6)}, {round(ci_high, 6)}]，"
            f"设计 SE={round(se_mean, 6)}）\n"
            f"未加权均值 = {round(unweighted_mean, 6)}\n"
            f"加权总量（人口/总体推断）= Σwᵢyᵢ = {round(wtotal, 4)}\n"
            f"{deff_note}\n"
            "注：设计 SE 始终采用 Kish/有放回近似（z_i=wᵢ(yᵢ-均值) 的残差贡献，乘 n/(n-1)）——"
            "**不纳入分层/PSU 聚类**（给了 group 只产出分组描述性均值，并不改变这个 SE），"
            "真实复杂抽样（分层/多阶）的方差可能更大；"
            "权重须为正；加权总量假设权重为膨胀权重（每单位代表的总体数）。\n\n"
            + (("分组加权均值：\n" + group_df.to_string(index=False) + "\n") if group_df is not None else ""),
            encoding="utf-8",
        )
        files.append("weighted_summary.txt")

        summary.append(
            f"{entry.method} 完成：数值列 {value}、权重列 {weight}"
            + (f"、分组 {group}" if group else "")
            + f"；加权均值 {round(wmean, 4)}（95% CI [{round(ci_low, 4)}, {round(ci_high, 4)}]），"
            f"加权总量 {round(wtotal, 2)}；未加权均值 {round(unweighted_mean, 4)}；"
            f"{deff_note}。"
            "⚠ 设计 SE 始终用 Kish/有放回近似、**不含分层/PSU 聚类**（group 仅给分组描述均值、不改 SE）；"
            "权重须为正；config 可指定 value/weight/group。"
        )
        code += [
            "import numpy as np",
            f"w = df[{weight!r}].to_numpy(float); y = df[{value!r}].to_numpy(float)",
            "wmean = (w*y).sum()/w.sum(); wtotal = (w*y).sum()",
            "z = w*(y - wmean); n = len(y)",
            "var = (n/(n-1))*(z*z).sum()/w.sum()**2  # design-based (Kish) variance",
            "deff = n*(w*w).sum()/w.sum()**2; n_eff = w.sum()**2/(w*w).sum()  # Kish deff",
        ]
    except Exception as err:
        summary.append(f"设计加权估计失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. poststratification — calibrate weights to KNOWN population proportions (1 var)
# ─────────────────────────────────────────────────────────────────────────────
@register("poststratification")
def _branch_poststratification(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
    except Exception as err:  # pragma: no cover
        summary.append(f"事后分层跳过：缺少 numpy/pandas（{err}）。")
        return

    # ── resolve the post-strat categorical column (config strata, else first categorical) ──
    cats = _categorical_cols(fp)
    strata = cfg.get("strata") or cfg.get("stratum")
    if strata and strata not in df.columns:
        strata = None
    if strata is None:
        strata = cats[0] if cats else None
    if strata is None:
        summary.append("事后分层跳过：找不到分层用的分类列（需 config[\"strata\"] 指定一个分类变量）。")
        return

    # ── known population proportions — REQUIRED; cannot be fabricated ──
    pop_props = cfg.get("pop_props") or cfg.get("pop_proportions")
    if not isinstance(pop_props, dict) or not pop_props:
        summary.append(
            "事后分层跳过：必须提供已知的总体比例 —— 引擎无法凭空捏造总体目标。"
            "请用 config[\"pop_props\"]={\"<类别>\": <比例>, ...} 给出每个类别的总体占比。"
        )
        return

    # optional base weight (else 1.0) and optional value column for adjusted mean
    base_w = cfg.get("weight")
    if base_w and base_w not in df.columns:
        base_w = None
    value = cfg.get("value")
    if value and value not in df.columns:
        value = None

    try:
        keep = [strata] + ([base_w] if base_w else []) + ([value] if value else [])
        sub = df[keep].copy()
        sub = sub.dropna(subset=[strata])
        if len(sub) < 3:
            summary.append("事后分层跳过：有效行 <3，样本太少。")
            return

        levels = sub[strata].astype(str)
        if base_w:
            bw = pd.to_numeric(sub[base_w], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.nan_to_num(bw, nan=-1) > 0):
                summary.append(
                    "事后分层跳过：基础权重必须全为正（检测到 <=0 或缺失）。"
                    "请清洗 config[\"weight\"] 列或省略它（默认每行权重 1）。"
                )
                return
        else:
            bw = np.ones(len(sub), dtype=float)

        # normalize the supplied population proportions to sum to 1 (be forgiving:
        # accept percentages or unnormalized weights); keys matched as strings.
        pop = {str(k): float(v) for k, v in pop_props.items()}
        if any(v < 0 for v in pop.values()):
            summary.append("事后分层跳过：总体比例不能为负（检查 config[\"pop_props\"]）。")
            return
        psum = sum(pop.values())
        if psum <= 0:
            summary.append("事后分层跳过：总体比例之和必须为正（检查 config[\"pop_props\"]）。")
            return
        pop = {k: v / psum for k, v in pop.items()}

        sample_levels = list(pd.unique(levels))
        # require every sample level to have a target (else we'd be inventing one)
        missing = [lv for lv in sample_levels if lv not in pop]
        if missing:
            summary.append(
                f"事后分层跳过：样本中的类别 {missing} 在 config[\"pop_props\"] 里没有给定总体比例 ——"
                "引擎不会替这些类别捏造目标。请补全所有类别的总体占比。"
            )
            return

        # base-weighted sample proportion per cell, then the post-strat factor.
        rows = []
        adj_weight = np.zeros(len(sub), dtype=float)
        total_bw = float(bw.sum())
        for lv in sample_levels:
            mask = (levels == lv).to_numpy()
            cell_bw = float(bw[mask].sum())
            samp_p = cell_bw / total_bw if total_bw > 0 else 0.0
            target_p = pop[lv]
            factor = (target_p / samp_p) if samp_p > 0 else 0.0
            adj_weight[mask] = bw[mask] * factor
            rows.append(
                {
                    "cell": lv,
                    "n": int(mask.sum()),
                    "sample_prop": samp_p,
                    "target_prop": target_p,
                    "adjustment_factor": factor,
                    "mean_adjusted_weight": float(adj_weight[mask].mean()) if mask.any() else 0.0,
                }
            )
        cell_df = pd.DataFrame(rows).sort_values("cell").reset_index(drop=True)
        cell_df.to_csv(d / "poststrat_cells.csv", index=False, encoding="utf-8")
        files.append("poststrat_cells.csv")

        n = len(sub)
        n_cells = len(sample_levels)
        factors = cell_df["adjustment_factor"].to_numpy(dtype=float)
        max_factor = float(factors.max())
        min_factor = float(factors[factors > 0].min()) if np.any(factors > 0) else 0.0
        # Kish weighting efficiency = (Σw)^2 / (n·Σw^2)  (= 1/deff of the new weights)
        sw = float(adj_weight.sum())
        sw2 = float((adj_weight * adj_weight).sum())
        weighting_eff = float((sw * sw) / (n * sw2)) if sw2 > 0 else 0.0

        # optional unadjusted vs post-stratified mean of a value column
        adj_value_mean = unadj_value_mean = None
        if value:
            yv = pd.to_numeric(sub[value], errors="coerce").to_numpy(dtype=float)
            ok = ~np.isnan(yv)
            if ok.sum() >= 1 and adj_weight[ok].sum() > 0:
                unadj_value_mean = float((bw[ok] * yv[ok]).sum() / bw[ok].sum())
                adj_value_mean = float((adj_weight[ok] * yv[ok]).sum() / adj_weight[ok].sum())

        # ── PNG: sample vs target proportion bars per cell ──
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xlabels = cell_df["cell"].astype(str).tolist()
            x = np.arange(len(xlabels))
            width = 0.38
            fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(xlabels)), 4.5))
            ax.bar(x - width / 2, cell_df["sample_prop"], width, label="sample", color="#999999")
            ax.bar(x + width / 2, cell_df["target_prop"], width, label="target (population)", color="#4C72B0")
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=30, ha="right")
            ax.set_ylabel("proportion")
            ax.set_title(f"Post-stratification on {strata}: sample vs target")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "poststrat_proportions.png", dpi=150)
            plt.close(fig)
            files.append("poststrat_proportions.png")
        except Exception:
            pass

        estimates["n_cells"] = float(n_cells)
        estimates["max_adjustment_factor"] = round(max_factor, 6)
        estimates["min_adjustment_factor"] = round(min_factor, 6)
        estimates["weighting_efficiency"] = round(weighting_eff, 6)
        estimates["n"] = float(n)
        if adj_value_mean is not None:
            estimates["adjusted_value_mean"] = round(adj_value_mean, 6)
            estimates["unadjusted_value_mean"] = round(unadj_value_mean, 6)

        extreme = max_factor > 3.0 or (0 < min_factor < 1 / 3.0)
        (d / "poststrat_summary.txt").write_text(
            f"事后分层（post-stratification）：分层变量 {strata}"
            + (f"，基础权重 {base_w}" if base_w else "（基础权重=1）")
            + (f"，值列 {value}" if value else "") + "\n"
            f"样本 n={n}，{n_cells} 个分层格\n"
            f"调整因子 = 总体比例 / 样本比例：max={round(max_factor, 4)}，min={round(min_factor, 4)}\n"
            f"加权效率（Kish）={round(weighting_eff, 4)}（越接近 1 越好；=1/deff）\n"
            + (
                f"{value} 未加权(基权)均值 {round(unadj_value_mean, 6)} → 事后分层后 {round(adj_value_mean, 6)}\n"
                if adj_value_mean is not None
                else ""
            )
            + (
                "⚠ 检测到极端调整因子（>3 或 <1/3），可能存在稀疏分层格，结果不稳。\n"
                if extreme
                else ""
            )
            + "注：事后分层需要外部已知的总体比例（config pop_props），无法凭空生成；"
            "它只校正单一边际（多变量请用 raking）；新权重 = 基础权重 × (总体比例/样本比例)。\n\n"
            "分层格明细：\n" + cell_df.to_string(index=False),
            encoding="utf-8",
        )
        files.append("poststrat_summary.txt")

        summary.append(
            f"{entry.method} 完成：分层变量 {strata}"
            + (f"、基础权重 {base_w}" if base_w else "（基权=1）")
            + f"；{n_cells} 格、n={n}；调整因子 max={round(max_factor, 3)} / min={round(min_factor, 3)}，"
            f"加权效率 {round(weighting_eff, 3)}"
            + (
                f"；{value} 均值 {round(unadj_value_mean, 4)}→{round(adj_value_mean, 4)}"
                if adj_value_mean is not None
                else ""
            )
            + ("。⚠ 检测到极端调整因子（稀疏格）" if extreme else "。")
            + "⚠ 事后分层必须提供已知总体比例（config pop_props），不可捏造；只校正单一边际"
            "（多边际用 raking）；config 可指定 strata/weight/pop_props/value。"
        )
        code += [
            "import numpy as np",
            f"levels = df[{strata!r}].astype(str)",
            "# new weight = base_weight * (target_prop / sample_prop)  per cell",
            "# target_prop comes from config['pop_props'] (KNOWN population, never invented)",
            "# weighting efficiency = (Σw)^2 / (n·Σw^2)  (= 1/deff)",
        ]
    except Exception as err:
        summary.append(f"事后分层失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. raking — iterative proportional fitting (IPF) on 2+ margins
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_margin(margin: dict) -> dict:
    """Normalize a {level: prop} target margin to sum to 1 (keys -> str)."""
    m = {str(k): float(v) for k, v in margin.items()}
    s = sum(m.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in m.items()}


@register("raking")
def _branch_raking(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    try:
        import numpy as np
        import pandas as pd
    except Exception as err:  # pragma: no cover
        summary.append(f"耙合(raking)跳过：缺少 numpy/pandas（{err}）。")
        return

    # ── resolve 2+ raking variables (config rake_vars, else categorical columns) ──
    cats = _categorical_cols(fp)
    rake_vars = cfg.get("rake_vars") or cfg.get("rake_variables")
    if isinstance(rake_vars, str):
        rake_vars = [rake_vars]
    if rake_vars:
        rake_vars = [v for v in rake_vars if v in df.columns]
    if not rake_vars:
        rake_vars = cats[:2]
    if len(rake_vars) < 2:
        summary.append(
            "耙合(raking)跳过：至少需要 2 个分类变量做边际校准"
            "（需 config[\"rake_vars\"]=[\"<var1>\", \"<var2>\", ...]）。"
        )
        return

    # ── known population margins — REQUIRED; cannot be fabricated ──
    margins = cfg.get("margins")
    if not isinstance(margins, dict) or not margins:
        summary.append(
            "耙合(raking)跳过：必须提供每个变量已知的总体边际 —— 引擎无法凭空捏造总体目标。"
            "请用 config[\"margins\"]={\"<var>\": {\"<级别>\": <比例>, ...}, ...} 给出各变量的总体占比。"
        )
        return
    # every rake variable needs a target margin
    missing_var = [v for v in rake_vars if v not in margins or not isinstance(margins[v], dict)]
    if missing_var:
        summary.append(
            f"耙合(raking)跳过：变量 {missing_var} 缺少总体边际（config[\"margins\"]）——不会替它们捏造目标。"
        )
        return

    base_w = cfg.get("weight")
    if base_w and base_w not in df.columns:
        base_w = None

    try:
        keep = list(rake_vars) + ([base_w] if base_w else [])
        sub = df[keep].copy().dropna(subset=list(rake_vars))
        if len(sub) < 3:
            summary.append("耙合(raking)跳过：有效行 <3，样本太少。")
            return

        # string-cast the rake variables; build & validate per-variable targets
        targets = {}
        for v in rake_vars:
            sub[v] = sub[v].astype(str)
            tg = _normalize_margin(margins[v])
            if not tg:
                summary.append(f"耙合(raking)跳过：变量 {v} 的总体边际之和必须为正（检查 config[\"margins\"]）。")
                return
            samp_levels = list(pd.unique(sub[v]))
            miss = [lv for lv in samp_levels if lv not in tg]
            if miss:
                summary.append(
                    f"耙合(raking)跳过：变量 {v} 的样本级别 {miss} 在 config[\"margins\"] 里没有目标比例 ——"
                    "不会替它们捏造目标。"
                )
                return
            targets[v] = tg

        if base_w:
            w0 = pd.to_numeric(sub[base_w], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.nan_to_num(w0, nan=-1) > 0):
                summary.append(
                    "耙合(raking)跳过：基础权重必须全为正（检测到 <=0 或缺失）。请清洗或省略 config[\"weight\"]。"
                )
                return
        else:
            w0 = np.ones(len(sub), dtype=float)

        N = len(sub)
        w = w0.copy()
        # precompute per-variable boolean masks for each level
        masks = {v: {lv: (sub[v] == lv).to_numpy() for lv in targets[v]} for v in rake_vars}

        max_iter = int(cfg.get("max_iter", 50))
        tol = float(cfg.get("tol", 1e-7))

        def _max_discrepancy(weights):
            err = 0.0
            tw = weights.sum()
            for v in rake_vars:
                for lv, tp in targets[v].items():
                    cur = weights[masks[v][lv]].sum() / tw if tw > 0 else 0.0
                    err = max(err, abs(cur - tp))
            return err

        iterations = 0
        converged = False
        for it in range(1, max_iter + 1):
            iterations = it
            for v in rake_vars:
                tw = w.sum()
                for lv, tp in targets[v].items():
                    m = masks[v][lv]
                    cur_w = w[m].sum()
                    target_w = tp * tw  # target total weight in this cell
                    if cur_w > 0:
                        w[m] *= target_w / cur_w
                    # cur_w==0 means level absent (cannot reach a >0 target) -> left as-is
            if _max_discrepancy(w) < tol:
                converged = True
                break

        max_margin_error = float(_max_discrepancy(w))

        # ── per-variable target vs achieved margin CSV ──
        rows = []
        tw = w.sum()
        for v in rake_vars:
            for lv, tp in targets[v].items():
                achieved = float(w[masks[v][lv]].sum() / tw) if tw > 0 else 0.0
                rows.append(
                    {
                        "variable": v,
                        "level": lv,
                        "target_prop": tp,
                        "achieved_prop": achieved,
                        "abs_error": abs(achieved - tp),
                    }
                )
        margin_df = pd.DataFrame(rows)
        margin_df.to_csv(d / "raking_margins.csv", index=False, encoding="utf-8")
        files.append("raking_margins.csv")

        # ── weight summary CSV (before vs after) ──
        sw = float(w.sum())
        sw2 = float((w * w).sum())
        raked_deff = float(N * sw2 / (sw * sw)) if sw > 0 else float("nan")
        wsum_df = pd.DataFrame(
            {
                "stat": ["n", "min", "max", "mean", "std", "sum", "design_effect"],
                "before": [
                    N, float(w0.min()), float(w0.max()), float(w0.mean()),
                    float(w0.std()), float(w0.sum()),
                    float(N * (w0 * w0).sum() / w0.sum() ** 2) if w0.sum() > 0 else float("nan"),
                ],
                "after": [
                    N, float(w.min()), float(w.max()), float(w.mean()),
                    float(w.std()), float(w.sum()), raked_deff,
                ],
            }
        )
        wsum_df.to_csv(d / "raking_weight_summary.csv", index=False, encoding="utf-8")
        files.append("raking_weight_summary.csv")

        # ── PNG: weight distribution before vs after raking ──
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            bins = min(30, max(5, N // 3))
            ax.hist(w0, bins=bins, alpha=0.5, label="before (base)", color="#999999")
            ax.hist(w, bins=bins, alpha=0.5, label="after (raked)", color="#4C72B0")
            ax.set_xlabel("weight")
            ax.set_ylabel("count")
            ax.set_title(f"Weight distribution before vs after raking (deff={raked_deff:.2f})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "raking_weights.png", dpi=150)
            plt.close(fig)
            files.append("raking_weights.png")
        except Exception:
            pass

        estimates["n_rake_vars"] = float(len(rake_vars))
        estimates["iterations"] = float(iterations)
        estimates["converged"] = 1.0 if converged else 0.0
        estimates["max_margin_error"] = round(max_margin_error, 8)
        estimates["raked_design_effect"] = round(raked_deff, 6)
        estimates["n"] = float(N)

        conv_note = (
            f"{iterations} 次迭代后收敛（最大边际误差 {max_margin_error:.2e} < tol={tol}）"
            if converged
            else f"{max_iter} 次迭代未收敛（最大边际误差 {max_margin_error:.2e}）——"
            "边际可能不相容或分层格稀疏"
        )
        (d / "raking_summary.txt").write_text(
            f"耙合 / 迭代比例拟合（IPF, raking）：变量 {', '.join(rake_vars)}"
            + (f"，基础权重 {base_w}" if base_w else "（基权=1）") + "\n"
            f"样本 n={N}\n"
            f"{conv_note}\n"
            f"耙合后权重设计效应 deff={round(raked_deff, 4)}（>1=方差被抬高，这是校准的代价）\n"
            "注：raking 需要每个变量已知的总体边际（config margins），无法凭空生成；"
            "当边际相互不相容或分层格稀疏时可能不收敛（已报告 converged 标志与最大误差）；"
            "校准会抬高设计效应（方差），已报告。\n\n"
            "目标 vs 达成边际：\n" + margin_df.round(6).to_string(index=False) + "\n\n"
            "权重分布（前/后）：\n" + wsum_df.round(4).to_string(index=False),
            encoding="utf-8",
        )
        files.append("raking_summary.txt")

        summary.append(
            f"{entry.method} 完成：{len(rake_vars)} 个变量 {', '.join(rake_vars)}"
            + (f"、基础权重 {base_w}" if base_w else "（基权=1）")
            + f"；{conv_note}；耙合后设计效应 deff={round(raked_deff, 3)}。"
            "⚠ raking 必须提供各变量已知总体边际（config margins），不可捏造；边际不相容/稀疏时可能不收敛"
            "（看 converged 与 max_margin_error）；校准抬高方差（已报 deff）；config 可指定 rake_vars/margins/weight。"
        )
        code += [
            "import numpy as np",
            "# IPF / raking: for each variable, rescale weights so each level's",
            "# weighted share matches its KNOWN target (config['margins']); iterate to tol.",
            "# targets are never invented; raked deff = n·Σw^2 / (Σw)^2 reports the variance cost.",
        ]
    except Exception as err:
        summary.append(f"耙合(raking)失败：{err}")
