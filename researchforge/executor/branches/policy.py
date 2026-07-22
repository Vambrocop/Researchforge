"""Branch handlers for the POLICY-EVALUATION family — Wave P4.

`pmc_index` — the PMC (Policy Modeling Consistency) index model (Ruiz Estrada 2011):
a quantitative consistency score for one or more policies, from a matrix of
researcher-coded secondary indicators.

Methodology (all steps disclosed in the run summary):
  1. Secondary indicators (二级指标) are 0/1 (or [0,1]) values a researcher assigned
     by reading each policy against a coding framework. Indicators are GROUPED into
     first-level variables (一级变量 X1..Xn) — via config ``groups`` (explicit), else by
     column-name prefix (``X1_scope``,``X1_forward`` → X1), else each indicator is its
     own variable (disclosed).
  2. Each first-level variable Xi = the MEAN of its secondary indicators for that
     policy → Xi ∈ [0,1].
  3. PMC index = Σ Xi over the n variables → PMC ∈ [0, n]; PMC ratio = PMC / n ∈ [0,1].
  4. Rating band (Ruiz Estrada convention, applied to the ratio): ≥0.90 完美 / ≥0.70
     优秀 / ≥0.50 可接受 / else 较低.
  5. PMC surface: the n variable scores are laid into a √n×√n matrix and drawn as a 3D
     surface — a dip marks a weak policy dimension. (Empty grid cells are padded with
     the policy's mean score so the surface is smooth, not artificially dented — this
     is disclosed.)

HONESTY BOUNDARY (the whole reason this method is scoped the way it is): the engine
does POST-CODING QUANTIFICATION, not automatic qualitative coding. It never judges
from the policy TEXT whether an indicator is 0 or 1 — that is the researcher's domain
judgement (the same boundary as three-level grounded-theory coding). Give it an
already-coded 政策×二级指标 matrix; it computes the index, the bands, the surface, and
the cross-policy comparison. It never fabricates codes.

Deterministic arithmetic + plotting (no statistical inference) — verified empirically.
Auto-registered by branches/__init__.py (pkgutil.walk_packages).
"""

from __future__ import annotations

import math
import re

from researchforge.executor._branch_api import Ctx, register

# Ruiz Estrada rating bands on the PMC RATIO (PMC / n_variables), highest first.
_PMC_BANDS: list[tuple[float, str]] = [
    (0.90, "完美 Perfect"),
    (0.70, "优秀 Good"),
    (0.50, "可接受 Acceptable"),
    (0.0, "较低 Low"),
]


def _rating(ratio: float) -> str:
    for thr, label in _PMC_BANDS:
        if ratio >= thr:
            return label
    return "较低 Low"


def _prefix(name: str) -> str | None:
    """First-level-variable prefix of an indicator column name, or None if it has no
    separator-delimited prefix. ``X1_scope`` → ``X1``; ``政策性质_1`` → ``政策性质``. Only a
    genuine separator (``_ - . space``) counts — trailing-digit stripping is too
    ambiguous to auto-group on."""
    parts = re.split(r"[_\-.\s]+", str(name).strip(), maxsplit=1)
    if len(parts) == 2 and parts[0]:
        return parts[0]
    return None


def _select_indicators(fp, df, cfg, exclude: set[str]) -> list[str]:
    """The secondary-indicator columns. config ``indicators`` overrides (used as given,
    normalized later); else auto = numeric columns whose non-null values all lie in
    [0,1] (0/1 coded indicators or fractional scores), excluding policy/id/time/unit."""
    import pandas as pd

    forced = cfg.get("indicators")
    if isinstance(forced, (list, tuple)) and forced:
        return [c for c in forced if c in df.columns]

    out: list[str] = []
    for c in fp.columns:
        if c.name in exclude:
            continue
        s = df[c.name]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            # booleans are fine as 0/1 — coerce below; here just skip non-numeric/datetime
            if pd.api.types.is_bool_dtype(s):
                out.append(c.name)
            continue
        nn = s.dropna()
        if len(nn) == 0:
            continue
        if float(nn.min()) >= 0.0 and float(nn.max()) <= 1.0:
            out.append(c.name)
    return out


def _resolve_groups(indicators: list[str], cfg) -> tuple[dict[str, list[str]], str]:
    """Map indicators → first-level variables. Returns (groups, how). config ``groups``
    (dict {var: [indicator,...]}) is authoritative; else group by column-name prefix
    when that yields ≥2 groups and every indicator has a prefix; else each indicator is
    its own variable (a disclosed fallback)."""
    forced = cfg.get("groups")
    if isinstance(forced, dict) and forced:
        groups: dict[str, list[str]] = {}
        for var, cols in forced.items():
            keep = [c for c in cols if c in indicators]
            if keep:
                groups[str(var)] = keep
        if groups:
            return groups, "config"

    prefixes = [_prefix(c) for c in indicators]
    if all(p is not None for p in prefixes) and len(set(prefixes)) >= 2:
        groups = {}
        for col, pre in zip(indicators, prefixes):
            groups.setdefault(pre, []).append(col)
        return groups, "prefix"

    return {c: [c] for c in indicators}, "each"


def _pmc_surface_png(values, var_names, title, path) -> bool:
    """3D PMC surface for one policy: variable scores laid row-major into a √n×√n grid,
    empty cells padded with the policy mean (disclosed). Best-effort; returns success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n = len(values)
        if n < 2:
            return False
        m = int(math.ceil(math.sqrt(n)))
        pad = float(np.mean(values))
        grid = np.full((m, m), pad, dtype=float)
        for i, v in enumerate(values):
            grid[i // m, i % m] = float(v)
        xs = np.arange(m)
        ys = np.arange(m)
        xg, yg = np.meshgrid(xs, ys)

        fig = plt.figure(figsize=(6.5, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            xg, yg, grid, cmap="viridis", vmin=0.0, vmax=1.0,
            edgecolor="0.35", linewidth=0.4, antialiased=True,
        )
        ax.set_zlim(0.0, 1.0)
        ax.set_zlabel("variable score")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for i, (v, name) in enumerate(zip(values, var_names)):
            ax.text(i % m, i // m, min(float(v) + 0.04, 1.0), str(name), fontsize=7, ha="center")
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


@register("pmc_index")
def _branch_pmc_index(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    import numpy as np
    import pandas as pd

    # --- policy-name column: config policy > an id/categorical column > row index -----
    policy_col = cfg.get("policy") if cfg.get("policy") in df.columns else None
    if policy_col is None:
        for c in fp.columns:
            if c.name in {fp.unit_col, fp.time_col}:
                continue
            if c.kind in {"id", "categorical"}:
                policy_col = c.name
                break

    exclude = {fp.unit_col, fp.time_col, policy_col} - {None}
    indicators = _select_indicators(fp, df, cfg, exclude)
    if len(indicators) < 2:
        summary.append(
            "PMC 指数模型跳过：需要至少 2 个二级指标列（研究者编码的 0/1 或 [0,1] 数值列）。"
            "可用 config indicators 指定二级指标列、groups 指定一级变量分组、policy 指定政策名列。"
        )
        return

    # --- coerce indicators to numeric in [0,1]; min-max normalize any out-of-range ----
    ind_df = df[indicators].apply(pd.to_numeric, errors="coerce")
    normalized: list[str] = []
    for c in indicators:
        col = ind_df[c].astype(float)
        cmin, cmax = float(np.nanmin(col)), float(np.nanmax(col))
        if cmin < 0.0 or cmax > 1.0:
            rng = cmax - cmin
            ind_df[c] = (col - cmin) / rng if rng > 0 else 0.0
            normalized.append(c)
    ind_df = ind_df.fillna(0.0)  # an uncoded cell counts as "dimension absent" (0), disclosed

    groups, how = _resolve_groups(indicators, cfg)
    var_names = list(groups.keys())
    n_vars = len(var_names)

    # --- per-policy first-level variable scores + PMC ---------------------------------
    if policy_col is not None:
        policy_names = df[policy_col].astype(str).tolist()
    else:
        policy_names = [f"policy_{i}" for i in range(len(df))]

    rows = []
    var_matrix = []  # n_policies x n_vars
    for r in range(len(df)):
        xi = [float(ind_df.iloc[r][groups[v]].mean()) for v in var_names]
        var_matrix.append(xi)
        pmc = float(sum(xi))
        ratio = pmc / n_vars if n_vars else 0.0
        row = {"policy": policy_names[r]}
        for v, val in zip(var_names, xi):
            row[f"X::{v}"] = round(val, 6)
        row["PMC"] = round(pmc, 6)
        row["PMC_ratio"] = round(ratio, 6)
        row["rating"] = _rating(ratio)
        rows.append(row)
    var_matrix = np.array(var_matrix, dtype=float)

    scores_df = pd.DataFrame(rows)
    scores_df.columns = [c.replace("X::", "X_") for c in scores_df.columns]
    scores_df.to_csv(d / "pmc_scores.csv", index=False, encoding="utf-8")
    files.append("pmc_scores.csv")

    # --- cross-policy variable means (which dimensions are systematically weak) --------
    var_means = var_matrix.mean(axis=0)
    means_df = pd.DataFrame({
        "variable": var_names,
        "mean_score": np.round(var_means, 6),
        "n_indicators": [len(groups[v]) for v in var_names],
    }).sort_values("mean_score")
    means_df.to_csv(d / "pmc_variable_means.csv", index=False, encoding="utf-8")
    files.append("pmc_variable_means.csv")

    # --- PNG: variable-means bar (weak dimensions) ------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(var_means)
        fig, ax = plt.subplots(figsize=(6.5, 0.4 * n_vars + 1.2))
        ax.barh(range(n_vars), var_means[order], color="#4C72B0")
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels([var_names[i] for i in order], fontsize=8)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("mean variable score (across policies)")
        ax.set_title("PMC first-level variable means", fontsize=10)
        fig.tight_layout()
        fig.savefig(d / "pmc_variable_means.png", dpi=150)
        plt.close(fig)
        files.append("pmc_variable_means.png")
    except Exception:
        pass

    # --- PNG: PMC-by-policy bar (quick comparison) ------------------------------------
    if len(df) <= 40:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pmc_vals = var_matrix.sum(axis=1)
            fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(df)), 4.2))
            colors = ["#55A868" if (v / n_vars) >= 0.7 else "#C44E52" if (v / n_vars) < 0.5 else "#DD8452" for v in pmc_vals]
            ax.bar(range(len(df)), pmc_vals, color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(policy_names, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, n_vars)
            ax.axhline(0.7 * n_vars, color="0.4", linestyle="--", linewidth=1)
            ax.set_ylabel(f"PMC (max {n_vars})")
            ax.set_title("PMC index by policy", fontsize=10)
            fig.tight_layout()
            fig.savefig(d / "pmc_by_policy.png", dpi=150)
            plt.close(fig)
            files.append("pmc_by_policy.png")
        except Exception:
            pass

    # --- PMC surface(s): one 3D surface per policy, capped -----------------------------
    try:
        max_surfaces = int(cfg.get("max_surfaces")) if cfg.get("max_surfaces") is not None else 6
        if max_surfaces < 1:
            max_surfaces = 6
    except (TypeError, ValueError):
        max_surfaces = 6
    n_surf = 0
    if n_vars >= 2:
        for r in range(min(len(df), max_surfaces)):
            xi = var_matrix[r].tolist()
            pmc = float(sum(xi))
            ratio = pmc / n_vars
            title = f"PMC surface — {policy_names[r]}\nPMC={pmc:.2f} ({ratio*100:.0f}%) {_rating(ratio)}"
            fname = f"pmc_surface_{r + 1}.png"
            if _pmc_surface_png(xi, var_names, title, d / fname):
                files.append(fname)
                n_surf += 1

    # --- estimates + summary ----------------------------------------------------------
    pmc_all = var_matrix.sum(axis=1)
    ratios = pmc_all / n_vars
    estimates["n_policies"] = float(len(df))
    estimates["n_variables"] = float(n_vars)
    estimates["n_indicators"] = float(len(indicators))
    estimates["mean_PMC"] = round(float(pmc_all.mean()), 6)
    estimates["max_PMC"] = round(float(pmc_all.max()), 6)
    estimates["min_PMC"] = round(float(pmc_all.min()), 6)
    estimates["mean_PMC_ratio"] = round(float(ratios.mean()), 6)

    weakest = var_names[int(np.argmin(var_means))]
    strongest = var_names[int(np.argmax(var_means))]
    how_txt = {
        "config": "分组来自 config groups",
        "prefix": "分组按列名前缀自动推断（如 X1_* → X1）",
        "each": "⚠ 未识别到分组，每个二级指标各自成一个一级变量（可用 config groups 指定真实分组）",
    }[how]
    norm_txt = (
        f"（指标 {', '.join(normalized[:4])} 值域超出 [0,1]，已按列 min-max 归一并披露）"
        if normalized else ""
    )
    if len(df) == 1:
        head = (
            f"该政策 PMC={pmc_all[0]:.2f}/{n_vars}（{ratios[0]*100:.0f}%，{_rating(ratios[0])}）"
        )
    else:
        n_perfect = int((ratios >= 0.9).sum())
        head = (
            f"{len(df)} 项政策：平均 PMC={pmc_all.mean():.2f}/{n_vars}"
            f"（{ratios.mean()*100:.0f}%），完美档 {n_perfect} 项"
        )
    summary.append(
        f"{entry.method} 完成（PMC 指数模型，{len(indicators)} 个二级指标 → {n_vars} 个一级变量；"
        f"{how_txt}{norm_txt}）：{head}。最弱维度={weakest}（均值 {var_means.min():.2f}）、"
        f"最强维度={strongest}（均值 {var_means.max():.2f}）。产出：每政策一级变量得分/PMC/评级"
        f"（pmc_scores.csv）、各维度均值（pmc_variable_means.csv + 柱状图）、"
        f"{n_surf} 张 PMC 曲面图（pmc_surface_*.png，凹陷=弱维）。"
        f"⚠ PMC 只做**编码后量化**：二级指标的 0/1 赋值须由研究者按政策文本人工编码（引擎不自动判断"
        f"政策是否具备某维度——那是研究者的领域判断，如同扎根理论三级编码）。空/缺编码格按「维度缺失」计 0。"
        f"评级档（≥0.9 完美/≥0.7 优秀/≥0.5 可接受/否则较低）是 Ruiz Estrada 约定阈值；"
        f"曲面图把 {n_vars} 个变量排进 √n×√n 方阵、空格用该政策均值填补（保曲面平滑、非人为凹陷）；"
        f"PMC 对一级/二级指标框架的设计高度敏感（换框架结果会变）——"
        f"可用 config indicators/groups/policy/max_surfaces 覆盖。"
    )
    code += [
        "# PMC index: policies × researcher-coded secondary indicators (0/1)",
        "import numpy as np",
        f"indicators = {indicators!r}",
        f"groups = {{{', '.join(repr(v) + ': ' + repr(groups[v]) for v in var_names[:3])}, ...}}",
        "Xi = {v: df[cols].mean(axis=1) for v, cols in groups.items()}  # first-level vars",
        "pmc = sum(Xi.values())  # PMC index per policy, in [0, n_variables]",
        "print(pmc, pmc / len(groups))  # ratio in [0,1] -> rating band",
    ]
