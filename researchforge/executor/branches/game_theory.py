"""Branch handlers for the GAME-THEORY family (decision domain).

Two coherent methods spanning the non-cooperative / cooperative divide:

  - normal_form_game — 2-player normal-form (strategic) game analysis:
        * strictly / weakly dominated strategies for each player,
        * iterated elimination of strictly dominated strategies (IESDS),
        * pure-strategy Nash equilibria via the standard best-response test,
        * the 2x2 mixed-strategy NE in closed form (each player mixes to make the
          OTHER indifferent) + the game value,
        * for a zero-sum game, the minimax / maximin value and whether a pure
          saddle point exists.
  - shapley_value — cooperative game fair allocation: the exact Shapley value
        φ_i = Σ_{S⊆N\\{i}} [|S|!(n−|S|−1)!/n!]·(v(S∪{i})−v(S)) by enumerating all
        subsets (2^n; capped at n≤10 with an honest degrade beyond), the efficiency
        check (Σφ_i = v(N)), and each player's share %.

INPUT MODELS
------------
normal_form_game (see ``_resolve_bimatrix``):
  (a) config ``player1_payoff`` / ``player2_payoff`` — each names an equal-shape
      block of NUMERIC columns (rows = P1 strategies, cols = P2 strategies). The
      block can be given as a comma list of column names, OR (if a single name is
      given that is not a column) we fall back to splitting the numeric matrix in
      half by columns.
  (b) a "bimatrix" long form: columns p1_strategy, p2_strategy, payoff1, payoff2
      (one row per cell) — pivoted into two payoff matrices.
  (c) otherwise the single numeric matrix is treated as a ZERO-SUM game with
      P2 payoff = −P1 payoff.

shapley_value (see ``_resolve_characteristic_fn``):
  (a) config ``coalition`` / ``value`` columns: each row is a coalition (a bitmask
      like "101", a set string like "A,C" / "{A,C}", or a player-count integer)
      and its v(S).
  (b) one column per player (0/1 membership flags) plus a ``value`` column.

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法> 跳过：<原因>" appended to summary + return
    (never crash / fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort try/except),
    float ``estimates`` dict (plain floats only; nan for N/A), Chinese ``summary``
    with ⚠ assumption / bias disclosures.
  * The profiler may classify an integer payoff column as ``count`` or ``id``
    (all-distinct integers) — numeric-column resolution accepts continuous/count/id.

Pure Python (numpy / pandas / matplotlib). No heavy deps; the game-theory math is
hand-rolled and exact for the cases claimed.
"""

from __future__ import annotations

import itertools

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _numeric_cols(ctx: Ctx):
    """Names of numeric columns (continuous / count / id), in dataframe order.

    Accepts the ``id`` kind: an integer payoff column whose values happen to be
    all-distinct is misclassified as ``id`` by the profiler (CLAUDE.md「id 陷阱」),
    and a small integer payoff stream profiles as ``count`` — both are real numbers.
    """
    fp = ctx.fp
    names = {c.name: c.kind for c in fp.columns}
    out = []
    for col in ctx.df.columns:
        # binary (0/1) payoffs are legitimate numbers too — a payoff column that
        # happens to be all 0/1 profiles as ``binary`` and would otherwise be dropped.
        if names.get(col) in ("continuous", "count", "id", "binary"):
            out.append(col)
    return out


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


# --------------------------------------------------------------------------- #
# normal_form_game — input resolution
# --------------------------------------------------------------------------- #
def _block_cols(spec, df_cols):
    """Resolve a payoff-block spec into a list of column names.

    A spec may be (1) a list of column names, or (2) a comma-separated string of
    column names. Returns the list of names that actually exist in ``df_cols``
    (preserving order), or None if nothing resolves.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        names = [str(s) for s in spec]
    else:
        names = [s.strip() for s in str(spec).split(",")]
    have = [c for c in names if c in df_cols]
    return have if have else None


def _resolve_bimatrix(ctx: Ctx):
    """Resolve the two payoff matrices (P1, P2) for a 2-player normal-form game.

    Returns ``(p1, p2, is_zero_sum, note, None)`` on success (p1/p2 are float numpy
    arrays of identical shape; rows = P1 strategies, cols = P2 strategies) or
    ``(None, None, None, None, msg)`` on honest failure.
    """
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    df_cols = list(df.columns)

    def _num(cols):
        return df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).to_numpy(dtype=float)

    # (b) bimatrix long form: p1_strategy, p2_strategy, payoff1, payoff2
    bm_cols = {"p1_strategy", "p2_strategy", "payoff1", "payoff2"}
    if bm_cols.issubset(set(df_cols)):
        try:
            sub = df[["p1_strategy", "p2_strategy", "payoff1", "payoff2"]].copy()
            sub["payoff1"] = pd.to_numeric(sub["payoff1"], errors="coerce")
            sub["payoff2"] = pd.to_numeric(sub["payoff2"], errors="coerce")
            p1 = sub.pivot_table(index="p1_strategy", columns="p2_strategy",
                                 values="payoff1", aggfunc="mean")
            p2 = sub.pivot_table(index="p1_strategy", columns="p2_strategy",
                                 values="payoff2", aggfunc="mean")
            # align p2 to p1's index/columns ordering
            p2 = p2.reindex(index=p1.index, columns=p1.columns)
            a1 = p1.to_numpy(dtype=float)
            a2 = p2.to_numpy(dtype=float)
            if a1.size >= 1 and np.isfinite(a1).all() and np.isfinite(a2).all():
                zs = bool(np.allclose(a1, -a2))
                note = "（bimatrix 长表：p1_strategy×p2_strategy 透视为两张支付矩阵）"
                return a1, a2, zs, note, None
        except Exception:
            pass  # fall through to other layouts

    # (a) config player1_payoff / player2_payoff naming column blocks
    b1 = _block_cols(cfg.get("player1_payoff"), df_cols)
    b2 = _block_cols(cfg.get("player2_payoff"), df_cols)
    if b1 and b2 and len(b1) == len(b2):
        a1 = _num(b1)
        a2 = _num(b2)
        if a1.shape == a2.shape and np.isfinite(a1).all() and np.isfinite(a2).all():
            zs = bool(np.allclose(a1, -a2))
            note = (f"（P1 支付块={b1}、P2 支付块={b2}；行=P1 策略、列=P2 策略）")
            return a1, a2, zs, note, None

    # Build a single numeric matrix from the numeric columns for fallbacks below.
    nums = _numeric_cols(ctx)
    if not nums:
        return None, None, None, None, (
            "需要数值支付矩阵：config['player1_payoff']/['player2_payoff'] 指定两张等形矩阵、"
            "或 bimatrix 长表(p1_strategy,p2_strategy,payoff1,payoff2)、或单张数值矩阵(按零和处理)。"
        )
    M = _num(nums)
    M = M[np.isfinite(M).all(axis=1)]  # drop rows with any non-numeric cell
    if M.shape[0] < 1 or M.shape[1] < 1:
        return None, None, None, None, "数值支付矩阵无有效行/列。"

    # (a') player1_payoff given but as a single (non-column) name OR only one block:
    # split the numeric matrix in half by columns into P1 | P2 blocks (even #cols).
    if (cfg.get("player1_payoff") is not None or cfg.get("player2_payoff") is not None) \
            and M.shape[1] % 2 == 0 and M.shape[1] >= 2:
        half = M.shape[1] // 2
        a1 = M[:, :half]
        a2 = M[:, half:]
        zs = bool(np.allclose(a1, -a2))
        note = ("（⚠ 未能按列名解析支付块，已把数值矩阵按列对半拆为 P1|P2 两块；"
                "建议用列名列表显式指定 config['player1_payoff']/['player2_payoff']）")
        return a1, a2, zs, note, None

    # (c) single numeric matrix -> ZERO-SUM game: P2 payoff = -P1 payoff
    a1 = M
    a2 = -M
    note = ("（⚠ 未提供双方支付，已把单张数值矩阵当作零和博弈：P2 支付 = −P1 支付；"
            "行=P1 策略、列=P2 策略。用 config['player1_payoff']/['player2_payoff'] 指定非零和博弈）")
    return a1, a2, True, note, None


# --------------------------------------------------------------------------- #
# normal_form_game — game-theory computations (exact)
# --------------------------------------------------------------------------- #
def _dominated_rows(payoff, strict=True, tol=1e-9):
    """Indices of P1 strategies (rows) dominated by another pure row strategy.

    Row r is STRICTLY dominated by row k if payoff[k, :] > payoff[r, :] in every
    column (P1 chooses rows and maximises its own payoff). WEAKLY dominated: >= in
    every column AND > in at least one column. Returns a sorted list of dominated
    row indices (each dominated by SOME other row).
    """
    import numpy as np

    P = np.asarray(payoff, dtype=float)
    n_rows = P.shape[0]
    dominated = []
    for r in range(n_rows):
        for k in range(n_rows):
            if k == r:
                continue
            diff = P[k, :] - P[r, :]
            if strict:
                if np.all(diff > tol):
                    dominated.append(r)
                    break
            else:
                if np.all(diff >= -tol) and np.any(diff > tol):
                    dominated.append(r)
                    break
    return sorted(set(dominated))


def _dominated_cols(payoff2, strict=True, tol=1e-9):
    """Indices of P2 strategies (columns) dominated by another pure column strategy.

    P2 chooses columns and maximises ITS OWN payoff (``payoff2``). Column c is
    strictly dominated by column j if payoff2[:, j] > payoff2[:, c] in every row.
    """
    import numpy as np

    P = np.asarray(payoff2, dtype=float)
    n_cols = P.shape[1]
    dominated = []
    for c in range(n_cols):
        for j in range(n_cols):
            if j == c:
                continue
            diff = P[:, j] - P[:, c]
            if strict:
                if np.all(diff > tol):
                    dominated.append(c)
                    break
            else:
                if np.all(diff >= -tol) and np.any(diff > tol):
                    dominated.append(c)
                    break
    return sorted(set(dominated))


def _iesds(p1, p2, tol=1e-9):
    """Iterated Elimination of Strictly Dominated Strategies.

    Repeatedly removes strictly dominated rows (for P1 on ``p1``) and strictly
    dominated columns (for P2 on ``p2``) until none remain. Returns
    ``(surviving_row_idx, surviving_col_idx)`` as sorted lists of ORIGINAL indices.
    """
    import numpy as np

    rows = list(range(p1.shape[0]))
    cols = list(range(p1.shape[1]))
    changed = True
    while changed:
        changed = False
        if len(rows) > 1:
            sub1 = p1[np.ix_(rows, cols)]
            dom_r = _dominated_rows(sub1, strict=True, tol=tol)
            if dom_r:
                rows = [rows[i] for i in range(len(rows)) if i not in set(dom_r)]
                changed = True
        if len(cols) > 1:
            sub2 = p2[np.ix_(rows, cols)]
            dom_c = _dominated_cols(sub2, strict=True, tol=tol)
            if dom_c:
                cols = [cols[i] for i in range(len(cols)) if i not in set(dom_c)]
                changed = True
    return sorted(rows), sorted(cols)


def _pure_nash(p1, p2, tol=1e-9):
    """Pure-strategy Nash equilibria via the best-response test.

    Cell (r, c) is a pure NE iff:
      * P1 cannot improve by changing its row given P2 plays c:
        p1[r, c] >= max_k p1[k, c]  (P1's payoff is a column-max within column c), AND
      * P2 cannot improve by changing its column given P1 plays r:
        p2[r, c] >= max_j p2[r, j]  (P2's payoff is a row-max within row r).
    Returns a list of (r, c) index pairs.
    """
    import numpy as np

    A = np.asarray(p1, dtype=float)
    B = np.asarray(p2, dtype=float)
    nr, nc = A.shape
    col_max = A.max(axis=0)          # best P1 payoff achievable in each column
    row_max = B.max(axis=1)          # best P2 payoff achievable in each row
    eqs = []
    for r in range(nr):
        for c in range(nc):
            p1_br = A[r, c] >= col_max[c] - tol
            p2_br = B[r, c] >= row_max[r] - tol
            if p1_br and p2_br:
                eqs.append((r, c))
    return eqs


def _mixed_2x2(p1, p2, tol=1e-12):
    """Closed-form mixed-strategy NE for a 2x2 game (one player makes the other
    indifferent).

    P1 plays row 0 with prob p, row 1 with prob 1−p.
    P2 plays col 0 with prob q, col 1 with prob 1−q.

    q makes P1 INDIFFERENT between its two rows:
        q*p1[0,0] + (1−q)*p1[0,1] = q*p1[1,0] + (1−q)*p1[1,1]
        => q = (p1[1,1] − p1[0,1]) / (p1[0,0] − p1[0,1] − p1[1,0] + p1[1,1])
    p makes P2 INDIFFERENT between its two columns (using P2's payoffs p2):
        p*p2[0,0] + (1−p)*p2[1,0] = p*p2[0,1] + (1−p)*p2[1,1]
        => p = (p2[1,1] − p2[1,0]) / (p2[0,0] − p2[0,1] − p2[1,0] + p2[1,1])

    Returns ``(p, q, v1, v2, status)`` where p = P(P1 plays row 0),
    q = P(P2 plays col 0), v1/v2 are the equilibrium expected payoffs to P1/P2, and
    status in {"ok", "degenerate", "out_of_range"}. NaNs when not a proper interior
    mixed NE.
    """
    import numpy as np

    A = np.asarray(p1, dtype=float)
    B = np.asarray(p2, dtype=float)
    if A.shape != (2, 2):
        return float("nan"), float("nan"), float("nan"), float("nan"), "not_2x2"

    den_q = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    den_p = B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1]
    if abs(den_q) < tol or abs(den_p) < tol:
        return float("nan"), float("nan"), float("nan"), float("nan"), "degenerate"

    q = (A[1, 1] - A[0, 1]) / den_q          # P(P2 plays col 0)
    p = (B[1, 1] - B[1, 0]) / den_p          # P(P1 plays row 0)

    if not (-tol <= p <= 1 + tol and -tol <= q <= 1 + tol):
        return float(p), float(q), float("nan"), float("nan"), "out_of_range"

    p = min(1.0, max(0.0, p))
    q = min(1.0, max(0.0, q))
    pv = np.array([p, 1 - p])
    qv = np.array([q, 1 - q])
    v1 = float(pv @ A @ qv)
    v2 = float(pv @ B @ qv)
    return float(p), float(q), v1, v2, "ok"


def _minimax(p1, tol=1e-9):
    """Zero-sum minimax / maximin on P1's payoff matrix.

    maximin (P1's security level) = max_r min_c p1[r, c]
    minimax (P2's security level on P1's payoff) = min_c max_r p1[r, c]
    A pure saddle point exists iff maximin == minimax; then it equals the game value.
    Returns ``(maximin, minimax, has_saddle)``.
    """
    import numpy as np

    A = np.asarray(p1, dtype=float)
    row_mins = A.min(axis=1)
    col_maxs = A.max(axis=0)
    maximin = float(row_mins.max())
    minimax = float(col_maxs.min())
    has_saddle = abs(maximin - minimax) <= tol
    return maximin, minimax, has_saddle


# ===========================================================================
# 1) normal_form_game — dominance + pure NE + 2x2 mixed NE + zero-sum minimax
#    Refs: Osborne & Rubinstein "A Course in Game Theory"; Fudenberg & Tirole.
# ===========================================================================
@register("normal_form_game")
def _branch_normal_form_game(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    p1, p2, is_zero_sum, note, err = _resolve_bimatrix(ctx)
    if err is not None:
        summary.append(f"标准式博弈 跳过：{err}")
        return
    try:
        nr, nc = p1.shape

        # --- dominated strategies (strict & weak), per player ---
        p1_strict_dom = _dominated_rows(p1, strict=True)
        p1_weak_dom = _dominated_rows(p1, strict=False)
        p2_strict_dom = _dominated_cols(p2, strict=True)
        p2_weak_dom = _dominated_cols(p2, strict=False)

        # --- IESDS ---
        surv_rows, surv_cols = _iesds(p1, p2)

        # --- pure-strategy Nash equilibria ---
        pure = _pure_nash(p1, p2)

        # --- 2x2 mixed-strategy NE (closed form) ---
        mp, mq, mv1, mv2, mstatus = _mixed_2x2(p1, p2)
        is_2x2 = (nr == 2 and nc == 2)

        # --- zero-sum minimax / maximin ---
        if is_zero_sum:
            maximin, minimax, has_saddle = _minimax(p1)
        else:
            maximin = minimax = float("nan")
            has_saddle = False

        # ---- products: equilibria CSV ----
        eq_rows = []
        for (r, c) in pure:
            eq_rows.append({
                "type": "pure",
                "p1_strategy": f"R{r}",
                "p2_strategy": f"C{c}",
                "p1_prob_row0": float("nan"),
                "p2_prob_col0": float("nan"),
                "p1_payoff": round(float(p1[r, c]), 6),
                "p2_payoff": round(float(p2[r, c]), 6),
            })
        if is_2x2 and mstatus == "ok":
            eq_rows.append({
                "type": "mixed",
                "p1_strategy": f"p(R0)={mp:.4f}",
                "p2_strategy": f"q(C0)={mq:.4f}",
                "p1_prob_row0": round(mp, 6),
                "p2_prob_col0": round(mq, 6),
                "p1_payoff": round(mv1, 6),
                "p2_payoff": round(mv2, 6),
            })
        if not eq_rows:
            eq_rows.append({
                "type": "none", "p1_strategy": "", "p2_strategy": "",
                "p1_prob_row0": float("nan"), "p2_prob_col0": float("nan"),
                "p1_payoff": float("nan"), "p2_payoff": float("nan"),
            })
        pd.DataFrame(eq_rows).to_csv(d / "equilibria.csv", index=False, encoding="utf-8")
        files.append("equilibria.csv")

        # ---- estimates (plain floats only) ----
        estimates.update({
            "n_pure_ne": float(len(pure)),
            "n_p1_strategies": float(nr),
            "n_p2_strategies": float(nc),
            "n_p1_strictly_dominated": float(len(p1_strict_dom)),
            "n_p2_strictly_dominated": float(len(p2_strict_dom)),
            "iesds_rows_remaining": float(len(surv_rows)),
            "iesds_cols_remaining": float(len(surv_cols)),
            "is_zero_sum": 1.0 if is_zero_sum else 0.0,
            "has_saddle_point": 1.0 if (is_zero_sum and has_saddle) else 0.0,
            "mixed_p1_prob": round(mp, 6) if (is_2x2 and mstatus == "ok") else float("nan"),
            "mixed_p2_prob": round(mq, 6) if (is_2x2 and mstatus == "ok") else float("nan"),
            "game_value": (round(mv1, 6) if (is_2x2 and mstatus == "ok")
                           else (round(maximin, 6) if (is_zero_sum and has_saddle) else float("nan"))),
        })
        # game_value above is P1's equilibrium payoff. In a NON-zero-sum game the two
        # players' equilibrium payoffs differ, so surface P2's explicitly (in a
        # zero-sum game P2's value = -game_value by construction).
        estimates["game_value_p2"] = (
            round(mv2, 6) if (is_2x2 and mstatus == "ok")
            else (round(-maximin, 6) if (is_zero_sum and has_saddle) else float("nan"))
        )
        if is_zero_sum:
            estimates["maximin_value"] = round(maximin, 6)
            estimates["minimax_value"] = round(minimax, 6)

        # ---- payoff-matrix heatmap (English labels) ----
        def _plot(plt):
            fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * nc + 2), max(3.5, 0.8 * nr + 1.6)))
            im = ax.imshow(p1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(nc))
            ax.set_yticks(range(nr))
            ax.set_xticklabels([f"C{c}" for c in range(nc)])
            ax.set_yticklabels([f"R{r}" for r in range(nr)])
            ax.set_xlabel("P2 strategy (column)")
            ax.set_ylabel("P1 strategy (row)")
            ttl = "Payoff matrix (P1, P2)"
            if is_zero_sum:
                ttl = "Payoff matrix P1 (zero-sum: P2 = -P1)"
            ax.set_title(ttl)
            for r in range(nr):
                for c in range(nc):
                    star = " *" if (r, c) in pure else ""
                    ax.text(c, r, f"{p1[r, c]:.3g}, {p2[r, c]:.3g}{star}",
                            ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, label="P1 payoff")

        _save_fig(d, "payoff_heatmap.png", files, _plot)

        # ---- summary (Chinese, with disclosures) ----
        def _names(idx, prefix):
            return "、".join(f"{prefix}{i}" for i in idx) if idx else "无"

        pure_txt = ("、".join(f"(R{r},C{c})" for (r, c) in pure) if pure else "无（纯策略 NE 不存在）")

        if is_2x2 and mstatus == "ok":
            mixed_txt = (f"2x2 混合策略 NE：P1 以 p={mp:.4f} 选 R0、1−p={1 - mp:.4f} 选 R1；"
                         f"P2 以 q={mq:.4f} 选 C0、1−q={1 - mq:.4f} 选 C1；"
                         f"均衡期望支付 (P1,P2)=({mv1:.4f},{mv2:.4f})。")
        elif is_2x2 and mstatus == "degenerate":
            mixed_txt = "2x2 混合策略 NE：退化（差分分母≈0，存在弱占优/平局，无唯一内点混合解）。"
        elif is_2x2 and mstatus == "out_of_range":
            mixed_txt = ("2x2 混合策略 NE：闭式解落在 [0,1] 之外（说明存在严格占优策略，"
                         "无内点混合 NE，应看纯策略 NE）。")
        else:
            mixed_txt = ("混合策略 NE：本引擎仅对 2x2 给出闭式解；更大博弈的混合 NE 需要线性规划 / "
                         "Lemke-Howson 算法（未计算），此处仅报告纯策略 NE。")

        zs_txt = ""
        if is_zero_sum:
            sad = (f"存在鞍点（纯策略），博弈值={maximin:.4f}" if has_saddle
                   else "不存在纯策略鞍点（需混合策略，2x2 见上方混合 NE）")
            zs_txt = (f" 零和博弈：maximin（P1 安全值）={maximin:.4f}、"
                      f"minimax（P2 安全值）={minimax:.4f} → {sad}。")

        summary.append(
            f"{ctx.entry.method} 完成：{nr}×{nc} 双人标准式博弈{(' ' + note) if note else ''}。"
            f" P1 严格占优(被占优)策略：{_names(p1_strict_dom, 'R')}（弱占优：{_names(p1_weak_dom, 'R')}）；"
            f"P2 严格占优(被占优)策略：{_names(p2_strict_dom, 'C')}（弱占优：{_names(p2_weak_dom, 'C')}）。"
            f" IESDS（迭代剔除严格被占优）后剩余 P1 策略 {_names(surv_rows, 'R')}、P2 策略 {_names(surv_cols, 'C')}。"
            f" 纯策略 Nash 均衡（{len(pure)} 个）：{pure_txt}。 {mixed_txt}{zs_txt}"
            " 均衡与支付明细见 equilibria.csv、支付矩阵热力图见 payoff_heatmap.png。"
            " ⚠ 假定：参与人完全理性、完全信息（双方知道全部支付）、同时行动（一次性静态博弈）。"
            " ⚠ 纯策略 NE 用最优响应判定（己方在对方策略下无法单方改善）；混合策略 NE 仅对 2x2 闭式可解，"
            "更大博弈需 LP / Lemke-Howson。占优判定用 ±1e-9 容差。零和情形 P2 支付取 −P1。"
            " ⚠ 占优/IESDS 仅检验被**其它纯策略**占优，不检验被**混合策略**占优——故占优集与 IESDS "
            "幸存集可能偏大（完整 IESDS 需用 LP 检验混合占优）。"
        )
        code += [
            "import numpy as np",
            "# 纯策略 NE：最优响应检验",
            "col_max = p1.max(axis=0); row_max = p2.max(axis=1)",
            "pure = [(r,c) for r in range(p1.shape[0]) for c in range(p1.shape[1])",
            "        if p1[r,c] >= col_max[c]-1e-9 and p2[r,c] >= row_max[r]-1e-9]",
            "# 2x2 混合 NE：令对方无差异",
            "q = (p1[1,1]-p1[0,1])/(p1[0,0]-p1[0,1]-p1[1,0]+p1[1,1])  # P2 选 C0 概率",
            "p = (p2[1,1]-p2[1,0])/(p2[0,0]-p2[0,1]-p2[1,0]+p2[1,1])  # P1 选 R0 概率",
            "# 零和 minimax/maximin",
            "maximin = p1.min(axis=1).max(); minimax = p1.max(axis=0).min()",
        ]
    except Exception as exc:
        summary.append(f"标准式博弈 计算失败：{exc}")


# --------------------------------------------------------------------------- #
# shapley_value — input resolution
# --------------------------------------------------------------------------- #
def _parse_coalition(token, player_index):
    """Parse one coalition token into a frozenset of player names.

    Accepts:
      * a bitmask string over the player list, e.g. "101" -> {players[0], players[2]}
        (length must equal n; chars in {0,1}). MSB-left is players[0].
      * a set string: "A,C" / "{A,C}" / "A;C" / "A C" -> {"A","C"}.
      * the empty coalition: "", "0", "{}", "()", "-" -> frozenset().
    ``player_index`` maps player name -> position (and supplies the ordered list via
    its keys). Returns (frozenset, ok). When parsing a set string, names not in
    ``player_index`` are KEPT (they extend the player universe upstream).
    """
    s = str(token).strip()
    s = s.strip("{}()[]").strip()
    if s in ("", "0", "-", "none", "None", "empty", "∅"):
        return frozenset(), True
    players = list(player_index.keys())
    n = len(players)
    # bitmask form: all chars 0/1 and length == n (and n>0)
    if n > 0 and len(s) == n and set(s) <= {"0", "1"}:
        members = {players[i] for i, ch in enumerate(s) if ch == "1"}
        return frozenset(members), True
    # set form: split on , ; or whitespace
    import re

    parts = [p for p in re.split(r"[,;\s]+", s) if p]
    if parts:
        return frozenset(parts), True
    return frozenset(), True


def _resolve_characteristic_fn(ctx: Ctx):
    """Resolve the characteristic function v(S) for a cooperative game.

    Returns ``(players, v, note, None)`` on success where ``players`` is an ordered
    list of player names and ``v`` is a dict {frozenset(coalition): value}, or
    ``(None, None, None, msg)`` on honest failure. v(empty)=0 is always set.
    """
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    df_cols = list(df.columns)

    coal_col = cfg.get("coalition")
    val_col = cfg.get("value")

    # auto-detect a value column if not specified
    if val_col not in df_cols:
        for cand in ("value", "v", "payoff", "worth"):
            if cand in df_cols:
                val_col = cand
                break

    # ---- layout (b): one column per player (0/1 flags) + a value column ----
    if val_col in df_cols and (coal_col not in df_cols):
        player_cols = [c for c in df_cols if c != val_col]
        # keep only columns that look like 0/1 membership flags
        flag_cols = []
        for c in player_cols:
            ser = pd.to_numeric(df[c], errors="coerce")
            vals = set(ser.dropna().unique().tolist())
            if vals and vals <= {0.0, 1.0}:
                flag_cols.append(c)
        if len(flag_cols) >= 2:
            players = list(flag_cols)
            v = {frozenset(): 0.0}
            vals = pd.to_numeric(df[val_col], errors="coerce")
            for ridx in range(len(df)):
                if not pd.notna(vals.iloc[ridx]):
                    continue
                members = frozenset(
                    c for c in flag_cols
                    if float(pd.to_numeric(df[c], errors="coerce").iloc[ridx] or 0) == 1.0
                )
                v[members] = float(vals.iloc[ridx])
            note = (f"（每个参与人一列 0/1 成员标记 {players} + 价值列 {val_col}）")
            return players, v, note, None

    # ---- layout (a): coalition column + value column ----
    if coal_col not in df_cols:
        # auto: first non-numeric / non-value column as the coalition column
        for c in df_cols:
            if c == val_col:
                continue
            coal_col = c
            break
    if coal_col not in df_cols or val_col not in df_cols:
        return None, None, None, (
            "需要联盟特征函数 v(S)：config['coalition']+config['value'] 两列"
            "（每行一个联盟及其价值），或每个参与人一列 0/1 成员标记 + value 列。"
        )

    vals = pd.to_numeric(df[val_col], errors="coerce")
    # First pass: discover the universe of players (so bitmasks can be parsed and the
    # grand coalition identified). We collect set-string names AND infer bitmask width.
    # Build a provisional player index from set-string tokens; if tokens look like
    # bitmasks we defer to the longest bitmask's positional names P0..P{k-1}.
    import re as _re

    raw_tokens = [str(t).strip() for t in df[coal_col].tolist()]
    # The robust CSV reader coerces an all-bitmask column ("100","010","001",...) to
    # INTEGERS, stripping leading zeros (-> 100,10,1). If every token is all-0/1, treat
    # the column as bitmasks and ZERO-PAD each to the max width so positional parsing
    # (player Pi at position i) survives the lost leading zeros.
    _bare = [t.strip("{}()[]").strip() for t in raw_tokens]
    _all_bits = all(s != "" and set(s) <= {"0", "1"} for s in _bare) and len(_bare) > 0
    if _all_bits:
        _w = max(len(s) for s in _bare)
        raw_tokens = [s.zfill(_w) for s in _bare]
    set_names: list[str] = []
    max_bitmask = 0
    looks_bitmask = False
    for tok in raw_tokens:
        s = tok.strip("{}()[]").strip()
        if s and set(s) <= {"0", "1"} and len(s) >= 1:
            looks_bitmask = True
            max_bitmask = max(max_bitmask, len(s))
        else:
            for p in _re.split(r"[,;\s]+", s):
                if p and p not in ("0", "-", "none", "None", "empty"):
                    if p not in set_names:
                        set_names.append(p)

    if set_names:
        players = list(dict.fromkeys(set_names))
    elif looks_bitmask:
        players = [f"P{i}" for i in range(max_bitmask)]
    else:
        return None, None, None, "无法从 coalition 列识别参与人（既非位掩码也非集合字符串）。"

    player_index = {p: i for i, p in enumerate(players)}
    v = {frozenset(): 0.0}
    for ridx, tok in enumerate(raw_tokens):
        if not pd.notna(vals.iloc[ridx]):
            continue
        coal, ok = _parse_coalition(tok, player_index)
        if not ok:
            continue
        # extend the universe if a set-string introduced a new name
        for m in coal:
            if m not in player_index:
                players.append(m)
                player_index[m] = len(players) - 1
        v[coal] = float(vals.iloc[ridx])
    note = (f"（联盟列={coal_col}、价值列={val_col}；识别参与人 {players}）")
    return players, v, note, None


def _shapley(players, v):
    """Exact Shapley value by enumerating all subsets.

    φ_i = Σ_{S⊆N\\{i}} [|S|!·(n−|S|−1)!/n!]·(v(S∪{i}) − v(S)).

    ``v`` is a dict {frozenset: value}; missing coalitions default to 0.0 (a key
    modelling assumption, disclosed by the caller). Returns ``(phi, grand)`` where
    phi is a dict {player: value} and grand = v(N).
    """
    import math

    n = len(players)
    others = list(players)

    def vget(S):
        return float(v.get(frozenset(S), 0.0))

    phi = {p: 0.0 for p in players}
    factn = math.factorial(n)
    for i in players:
        rest = [p for p in others if p != i]
        # enumerate all subsets S of N\{i}
        for k in range(len(rest) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / factn
            for combo in itertools.combinations(rest, k):
                S = frozenset(combo)
                marginal = vget(S | {i}) - vget(S)
                phi[i] += weight * marginal
    grand = vget(frozenset(players))
    return phi, grand


# ===========================================================================
# 2) shapley_value — exact cooperative-game fair allocation
#    Refs: Shapley (1953) "A Value for n-Person Games"; Roth (ed.) "The Shapley
#          Value"; Peters "Game Theory: A Multi-Leveled Approach".
# ===========================================================================
@register("shapley_value")
def _branch_shapley_value(ctx: Ctx) -> None:
    d, files, summary, estimates, code = ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np  # noqa: F401  (kept for parity / potential plotting use)
    import pandas as pd

    players, v, note, err = _resolve_characteristic_fn(ctx)
    if err is not None:
        summary.append(f"Shapley 值 跳过：{err}")
        return
    try:
        n = len(players)
        if n < 1:
            summary.append("Shapley 值 跳过：未识别到任何参与人。")
            return
        if n > 10:
            summary.append(
                f"Shapley 值 跳过：参与人数 n={n} 超过精确枚举上限 10（精确解需枚举 2^n 个联盟，"
                "2^11=2048 起组合爆炸）。请减少参与人，或后续接入蒙特卡洛/抽样近似。"
            )
            return

        # count how many of the 2^n coalitions were actually supplied (excl. empty)
        total_coalitions = 2 ** n
        supplied = sum(1 for S in v.keys() if len(S) > 0)
        missing = total_coalitions - 1 - supplied  # minus the empty coalition

        phi, grand = _shapley(players, v)
        sum_phi = float(sum(phi.values()))
        efficiency_gap = sum_phi - grand
        # share % of the grand-coalition value (guard divide-by-zero)
        shares = {p: (phi[p] / grand * 100.0 if grand != 0 else float("nan")) for p in players}

        # ---- products: shapley CSV ----
        pd.DataFrame({
            "player": list(players),
            "shapley_value": [round(phi[p], 6) for p in players],
            "share_pct": [round(shares[p], 4) if shares[p] == shares[p] else float("nan")
                          for p in players],
        }).to_csv(d / "shapley_values.csv", index=False, encoding="utf-8")
        files.append("shapley_values.csv")

        # ---- estimates (plain floats only): shapley__<player> per player ----
        for p in players:
            key = f"shapley__{p}"
            estimates[key] = round(float(phi[p]), 6)
        estimates.update({
            "grand_coalition_value": round(float(grand), 6),
            "n_players": float(n),
            "shapley_sum": round(sum_phi, 6),
            "efficiency_gap": round(float(efficiency_gap), 6),
            "n_coalitions_supplied": float(supplied),
            "n_coalitions_missing": float(missing),
        })

        # ---- Shapley bar chart (English labels) ----
        def _plot(plt):
            fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * n + 2), 4.2))
            vals = [phi[p] for p in players]
            ax.bar(range(n), vals, color="#4C72B0", edgecolor="white")
            ax.axhline(0.0, color="#333333", lw=0.7)
            ax.set_xticks(range(n))
            ax.set_xticklabels([str(p) for p in players], rotation=0)
            ax.set_xlabel("player")
            ax.set_ylabel("Shapley value")
            ax.set_title(f"Shapley value allocation (v(N)={grand:.4g}, n={n})")
            for i, val in enumerate(vals):
                ax.text(i, val, f"{val:.3g}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8)

        _save_fig(d, "shapley_bar.png", files, _plot)

        eff_ok = abs(efficiency_gap) <= 1e-6
        alloc_txt = "、".join(
            f"{p}={phi[p]:.4f}"
            + (f"（{shares[p]:.1f}%）" if shares[p] == shares[p] else "")
            for p in players
        )
        summary.append(
            f"{ctx.entry.method} 完成：{n} 人合作博弈{(' ' + note) if note else ''}。"
            f" 大联盟价值 v(N)={grand:.6g}；各参与人 Shapley 值：{alloc_txt}。"
            f" 效率检验 Σφ_i={sum_phi:.6g} {'≈' if eff_ok else '≠'} v(N)={grand:.6g}"
            f"（差={efficiency_gap:.2e}{'，满足效率公理' if eff_ok else '，⚠ 偏离效率——通常因部分联盟价值缺失被当作 0'}）。"
            f" 共 2^{n}={total_coalitions} 个联盟，已提供 {supplied} 个、缺失 {missing} 个（按 0 处理）。"
            " 分配明细见 shapley_values.csv、柱状图见 shapley_bar.png。"
            " ⚠ Shapley 值精确解需要（理想情况下全部）联盟的价值 v(S)；缺失的联盟一律按 v(S)=0 处理"
            "（若实际并非 0，会使分配与效率检验失真——务必提供完整或近似完整的特征函数）。"
            " ⚠ 精确枚举为 2^n（已限 n≤10）。Shapley 值满足效率/对称/虚拟人/可加性公理，是唯一同时满足者；"
            "但要求博弈可转移效用(TU)、且对联盟形成顺序等概率——是规范性的「公平」分配，非实际谈判结果。"
        )
        code += [
            "import math, itertools",
            "# φ_i = Σ_{S⊆N\\{i}} |S|!(n-|S|-1)!/n! · (v(S∪{i}) − v(S))",
            "phi = {i: 0.0 for i in players}",
            "for i in players:",
            "    rest = [p for p in players if p != i]",
            "    for k in range(len(rest)+1):",
            "        w = math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)",
            "        for S in itertools.combinations(rest, k):",
            "            phi[i] += w * (v.get(frozenset(S)|{i},0) - v.get(frozenset(S),0))",
        ]
    except Exception as exc:
        summary.append(f"Shapley 值 计算失败：{exc}")
