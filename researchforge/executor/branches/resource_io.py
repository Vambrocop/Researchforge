"""Branch handler for LEONTIEF INPUT-OUTPUT ANALYSIS (resource family / economics).

One method — ``input_output_analysis`` — for resource-flow multipliers in an
Energy-Water-Food nexus or regional-economy study. (A SEPARATE agent owns
``resource.py`` + ``resource.yaml`` in the same "resource" family; this module
coexists via auto-discovery and only owns ``input_output_analysis``.)

WHAT IT DOES
  Given a square n×n inter-sectoral transactions matrix Z (rows = supplying
  sector i, columns = using sector j, so z_ij = goods from i used by j), it
  derives the open static Leontief model:

    * technical coefficients   A,   a_ij = z_ij / x_j   (column-normalize by each
        using sector's total output x_j — the "recipe" of inputs per unit output)
    * Leontief inverse         L = (I − A)^{-1}          (total requirements matrix;
        l_ij = output of i needed, directly + indirectly, per unit final demand of j)
    * output multipliers       m_j = Σ_i L_ij           (column sums of L — total
        economy-wide output triggered per unit of final demand in sector j)
    * backward linkages        BL_j = m_j / mean_k(m_k) (normalized column sums of L
        — how much sector j PULLS from its suppliers, upstream)
    * forward linkages         FL_i = r_i / mean_k(r_k) (normalized row sums of L,
        r_i = Σ_j L_ij — how much sector i PUSHES to its users, downstream)
    * required total output    x = L · f                 (if a final-demand vector f
        is supplied via config['final_demand'])

MATRIX-DETECTION RULE (see ``_resolve_matrix``):
  1. Collect the numeric columns (all values coerce to finite floats). Take the
     first n_rows of them as candidate matrix columns; it must be SQUARE — the
     number of usable numeric columns equals the number of rows. A leading
     non-numeric column (or one named like sector/sector_name/industry/name) is
     taken as the sector LABELS.
  2. If config['total_output'] names a numeric column (the total output vector
     x), compute A = Z / x̂ (divide each column j by x_j). PATH = "Z + total_output".
  3. Else if the matrix already looks like a technical-coefficient matrix —
     every column sum < 1 AND every value in [0, 1) — treat the matrix itself AS
     A. PATH = "already coefficients".
  4. Else (a raw flow matrix with no total_output) — HONEST DEGRADE. Column-
     normalizing inter-industry flows by their own column sums gives a column-
     stochastic A (ρ=1, (I−A) singular): a degenerate, non-productive model. A
     productive A needs each sector's TOTAL output x_j (> the inter-industry column
     sum, by value-added / final demand), which the flows alone don't provide — so
     we ask for config['total_output'] rather than fabricate a degenerate model.

Productivity / invertibility guard: A must be productive — spectral radius
ρ(A) = max|eigenvalue(A)| < 1 — for L = (I − A)^{-1} to exist and be
economically meaningful (non-negative). If ρ(A) ≥ 1, or I − A is singular, we
DEGRADE HONESTLY ("跳过") rather than fabricate.

Pure Python (numpy / pandas / matplotlib Agg, ENGLISH plot labels). No R, no
heavy deps. Honest degrade on: not square / too few sectors / non-productive /
singular I − A / no numeric data.

Refs: Leontief "Input-Output Economics"; Miller & Blair "Input-Output Analysis:
Foundations and Extensions"; Rasmussen / Hirschman linkage indices.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Column names that, if a leading column matches, mark it as the sector-label column.
_LABEL_NAMES = {
    "sector", "sectors", "sector_name", "industry", "industries",
    "name", "names", "label", "labels", "id", "code",
}


def _is_numeric_col(s):
    """True if EVERY non-null value of the series coerces to a finite float."""
    import numpy as np
    import pandas as pd

    nn = s.dropna()
    if len(nn) == 0:
        return False
    coerced = pd.to_numeric(nn, errors="coerce")
    return bool(coerced.notna().all()) and bool(np.isfinite(coerced.to_numpy(dtype=float)).all())


def _resolve_matrix(ctx: Ctx):
    """Resolve (A, sector_names, path_note, err).

    Returns (A, names, path_note, None) on success or (None, None, None, msg) on
    honest failure. ``A`` is the n×n technical-coefficient matrix (float ndarray),
    ``names`` a list[str] of n sector labels, ``path_note`` the Chinese disclosure
    of which detection path was taken.
    """
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    n_rows = len(df.index)

    # ---- find the sector-label column (a leading non-numeric / named column) ----
    label_col = None
    sector_cfg = cfg.get("sectors")
    if sector_cfg in df.columns:
        label_col = sector_cfg
    else:
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in _LABEL_NAMES and not _is_numeric_col(df[c]):
                label_col = c
                break
        if label_col is None:
            # else: first non-numeric column, if any, is the label column.
            for c in df.columns:
                if not _is_numeric_col(df[c]):
                    label_col = c
                    break

    # ---- numeric (matrix) columns, in order, excluding the label column AND the
    # vector columns (total_output / a column-named final_demand) which are NOT
    # part of the n×n inter-industry block. ----
    aux_cols = set()
    to_name = cfg.get("total_output")
    if isinstance(to_name, str) and to_name in df.columns:
        aux_cols.add(to_name)
    fd_name = cfg.get("final_demand")
    if isinstance(fd_name, str) and fd_name in df.columns:
        aux_cols.add(fd_name)

    num_cols = [
        c for c in df.columns
        if c != label_col and c not in aux_cols and _is_numeric_col(df[c])
    ]
    n = n_rows
    if n < 2:
        return None, None, None, f"扇区数过少（行数 {n}<2），不构成可分析的投入产出表。"
    # spec rule: square iff #numeric matrix cols == #rows (vectors excluded above).
    if len(num_cols) != n:
        return None, None, None, (
            f"不是方阵：部门间数值列数 {len(num_cols)} ≠ 行数(扇区数) {n}，"
            "投入产出表需 n×n 的部门间流量矩阵（行=供给部门，列=使用部门；"
            "总产出/最终需求向量请用 config 指定，勿混入矩阵）。"
        )
    mat_cols = num_cols[:n]
    M = df[mat_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).to_numpy(dtype=float)
    if not np.isfinite(M).all():
        return None, None, None, "流量矩阵含缺失/非有限值，无法计算技术系数。"
    if M.shape[0] != M.shape[1]:
        return None, None, None, f"矩阵非方阵：{M.shape[0]}×{M.shape[1]}。"

    # ---- sector names ----
    if label_col is not None:
        names = [str(v) for v in df[label_col].tolist()[:n]]
    else:
        names = [str(c) for c in mat_cols]
    if len(names) != n:
        names = [f"S{i + 1}" for i in range(n)]

    # ---- which path: total_output? already-coefficients? raw flows? ----
    if isinstance(to_name, str) and to_name in df.columns and _is_numeric_col(df[to_name]):
        x = pd.to_numeric(df[to_name], errors="coerce").to_numpy(dtype=float)[:n]
        if not np.isfinite(x).all() or np.any(x <= 0):
            return None, None, None, (
                f"total_output 列 {to_name} 含非正/缺失总产出，无法按 A = Z / x̂ 列归一化。"
            )
        # A productive Leontief technical-coefficient matrix must be non-negative; with
        # x>0 this holds iff the inter-industry flows are non-negative. Reject negative
        # flows rather than build an economically-meaningless A (the later ρ(A)<1 guard
        # does not catch sign problems).
        if np.any(M < 0):
            return None, None, None, (
                f"投入产出流量矩阵含负值——技术系数 A 必须非负方为有意义的 Leontief 模型；"
                "请核对流量列（行=投入来源部门、列=消耗部门，均应≥0）。"
            )
        A = M / x[np.newaxis, :]  # column j divided by x_j
        path = (
            f"路径＝Z + total_output：以 {to_name} 列为各部门总产出 x，"
            "技术系数 A 按列归一化 a_ij = z_ij / x_j。"
        )
        return A, names, path, None

    colsum = M.sum(axis=0)
    looks_coef = bool(np.all(M >= 0)) and bool(np.all(M < 1.0)) and bool(np.all(colsum < 1.0))
    if looks_coef:
        A = M.copy()
        path = (
            "路径＝已是技术系数矩阵：矩阵各值∈[0,1) 且各列和<1，直接当作 A 使用"
            "（未做列归一化）。"
        )
        return A, names, path, None

    # raw flow matrix, no total_output, not already coefficients -> honest degrade.
    # NOTE: column-normalizing inter-industry flows by their OWN column sums yields a
    # column-stochastic A whose spectral radius is exactly 1 (every column sums to 1),
    # so (I − A) is singular and the economy is never "productive" — a degenerate
    # Leontief model. A productive A requires each sector's TOTAL output x_j (which
    # exceeds the inter-industry column sum by value-added / final demand), so we
    # cannot derive a valid model from the flows alone. Ask for total_output.
    return None, None, None, (
        "原始流量矩阵但未提供总产出向量：无法从行业间流量单独导出可生产的技术系数矩阵"
        "（按列和归一化会得到列随机矩阵，谱半径恒=1、(I−A) 奇异、Leontief 逆不存在）。"
        "请用 config['total_output'] 提供各部门总产出（含增加值/最终需求），"
        "或直接提供技术系数矩阵（各值∈[0,1) 且各列和<1）。"
    )


def _save_fig(d, fname, files, build):
    """best-effort matplotlib figure (Agg). build(plt) draws on the current figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        build(plt)
        plt.tight_layout()
        plt.savefig(d / fname, dpi=150)
        plt.close("all")
        files.append(fname)
    except Exception:
        pass


# ===========================================================================
# input_output_analysis — Leontief inverse, multipliers & linkages
# ===========================================================================
@register("input_output_analysis")
def _branch_input_output_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    A, names, path_note, err = _resolve_matrix(ctx)
    if err is not None:
        summary.append(f"input_output_analysis 跳过：{err}")
        return

    try:
        n = A.shape[0]
        eye = np.eye(n)

        # --- productivity / invertibility guard: spectral radius of A < 1 ---
        try:
            eigvals = np.linalg.eigvals(A)
            spectral_radius = float(np.max(np.abs(eigvals)))
        except Exception as exc:  # eigenvalue solver failed
            summary.append(f"input_output_analysis 跳过：特征值求解失败（{exc}）。")
            return
        if not np.isfinite(spectral_radius) or spectral_radius >= 1.0:
            summary.append(
                "input_output_analysis 跳过：经济不可生产（technical 矩阵 A 的谱半径 "
                f"ρ(A)={spectral_radius:.4f} ≥ 1），(I − A) 不可逆 / Leontief 逆不存在或非负性失效。"
                " 通常因为矩阵不是有效的技术系数矩阵（列和应<1）。"
            )
            return

        # --- Leontief inverse L = (I - A)^{-1} ---
        ImA = eye - A
        try:
            L = np.linalg.inv(ImA)
        except np.linalg.LinAlgError as exc:
            summary.append(f"input_output_analysis 跳过：(I − A) 奇异，不可逆（{exc}）。")
            return
        if not np.isfinite(L).all():
            summary.append("input_output_analysis 跳过：Leontief 逆含非有限值（矩阵病态）。")
            return

        # --- multipliers & linkages ---
        output_multipliers = L.sum(axis=0)          # column sums of L
        row_sums = L.sum(axis=1)                     # row sums of L
        mean_col = float(output_multipliers.mean())
        mean_row = float(row_sums.mean())
        backward_linkage = (output_multipliers / mean_col) if mean_col != 0 else np.full(n, np.nan)
        forward_linkage = (row_sums / mean_row) if mean_row != 0 else np.full(n, np.nan)

        # --- optional required total output x = L · f ---
        f_vec = cfg.get("final_demand")
        x_required = None
        fd_note = ""
        if f_vec is not None:
            try:
                if isinstance(f_vec, str) and f_vec in df.columns:
                    f_arr = pd.to_numeric(df[f_vec], errors="coerce").to_numpy(dtype=float)[:n]
                else:
                    f_arr = np.asarray(f_vec, dtype=float).ravel()
                if f_arr.size == n and np.isfinite(f_arr).all():
                    x_required = L @ f_arr
                    fd_note = "；已按 x = L·f 计算满足给定最终需求所需的各部门总产出（见表 required_output）。"
                else:
                    fd_note = (
                        f"；⚠ config['final_demand'] 维度({getattr(f_arr, 'size', '?')})≠扇区数({n})"
                        "或含非有限值，已忽略。"
                    )
            except Exception:
                fd_note = "；⚠ config['final_demand'] 解析失败，已忽略。"

        # --- multipliers table (CSV) ---
        tbl = pd.DataFrame({
            "sector": names,
            "output_multiplier": np.round(output_multipliers, 6),
            "backward_linkage": np.round(backward_linkage, 6),
            "forward_linkage": np.round(forward_linkage, 6),
        })
        if x_required is not None:
            tbl["required_output"] = np.round(x_required, 6)
        tbl.to_csv(d / "io_multipliers.csv", index=False, encoding="utf-8")
        files.append("io_multipliers.csv")

        # --- estimates (ALL plain floats) ---
        estimates.update({
            "n_sectors": float(n),
            "spectral_radius_A": round(spectral_radius, 6),
            "mean_output_multiplier": round(float(output_multipliers.mean()), 6),
            "max_output_multiplier": round(float(output_multipliers.max()), 6),
        })
        for nm, m in zip(names, output_multipliers):
            estimates[f"multiplier__{nm}"] = round(float(m), 6)

        # --- bar plot of output multipliers + linkages (ENGLISH labels) ---
        def _plot(plt):
            idx = np.arange(n)
            fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * n + 3.0), 4.6))
            w = 0.27
            ax.bar(idx - w, output_multipliers, w, color="#4C72B0", label="output multiplier")
            ax.bar(idx, backward_linkage, w, color="#55A868", label="backward linkage")
            ax.bar(idx + w, forward_linkage, w, color="#C44E52", label="forward linkage")
            ax.axhline(1.0, color="#888888", ls="--", lw=0.9)
            ax.set_xticks(idx)
            ax.set_xticklabels([str(s) for s in names], rotation=30, ha="right", fontsize=8)
            ax.set_xlabel("sector")
            ax.set_ylabel("multiplier / normalized linkage")
            ax.set_title("Leontief output multipliers & linkages")
            ax.legend(fontsize=8)

        _save_fig(d, "io_multipliers.png", files, _plot)

        # --- Chinese summary with ⚠ disclosures ---
        top_i = int(np.argmax(output_multipliers))
        x_note = ""
        if x_required is not None:
            x_note = f" 满足最终需求所需总产出 x = L·f，合计 {float(np.sum(x_required)):.6g}。"
        summary.append(
            f"{entry.method} 完成：{n} 个部门。{path_note} "
            f"谱半径 ρ(A)={spectral_radius:.4f}（<1，经济可生产）；已求 Leontief 逆 "
            "L=(I−A)⁻¹。产出乘数(列和) 均值 "
            f"{float(output_multipliers.mean()):.4f}、最大 {float(output_multipliers.max()):.4f}"
            f"（部门「{names[top_i]}」最高，每单位最终需求带动全经济 {float(output_multipliers[top_i]):.4f} 单位产出）。"
            "后向关联=归一化列和（部门对上游供应的拉动），前向关联=归一化行和（部门对下游使用的推动），"
            f">1 表示高于平均。明细见 io_multipliers.csv 与图。{x_note}{fd_note}"
            " ⚠ Leontief 假定：固定技术系数（投入比例不随产量变化）、规模报酬不变、单一线性技术、"
            "无供给/产能约束；乘数是需求驱动的。 ⚠ 本实现为开放(open)模型——家庭部门视为外生最终需求；"
            "闭合(closed)模型会把家庭消费内生化、乘数更大（口径不同）。 ⚠ 这是一张静态快照，"
            "未含价格调整、替代效应或技术进步。"
        )

        code += [
            "import numpy as np",
            "# A = technical coefficients (a_ij = z_ij / x_j); column-normalize Z by sector output",
            "# A = Z / total_output[None, :]            # if total_output given",
            "# A = Z / Z.sum(axis=0)[None, :]           # else assume col totals = output",
            "n = A.shape[0]",
            "rho = np.max(np.abs(np.linalg.eigvals(A)))   # spectral radius; need rho < 1 (productive)",
            "L = np.linalg.inv(np.eye(n) - A)             # Leontief inverse (I - A)^{-1}",
            "output_multipliers = L.sum(axis=0)           # column sums of L",
            "backward_linkage = output_multipliers / output_multipliers.mean()  # normalized col sums",
            "forward_linkage  = L.sum(axis=1) / L.sum(axis=1).mean()            # normalized row sums",
            "# x = L @ f                                   # required total output for final demand f",
        ]
    except Exception as exc:
        summary.append(f"input_output_analysis 计算失败：{exc}")
