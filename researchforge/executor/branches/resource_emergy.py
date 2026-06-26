"""Emergy analysis (Odum's environmental accounting) — `emergy_analysis`.

Converts every system input flow to solar emjoules (sej) via a transformity
(sej per unit), then aggregates into renewable (R), non-renewable (N) and
purchased/feedback (F) emergy and the standard emergy indicators (EYR, ELR, ESI,
%renewable). Transformities come from the user (config) or a small curated PUBLIC
library (`_emergy_transformities.py`, Odum/Brown/NEAD); honest degrade when none
resolves. Deterministic accounting (sums + ratios) — no statistical inference.

Engine conventions (CLAUDE.md): `@register` handler unpacks ctx and MUTATES
summary/estimates/files/code; estimates are plain floats; matplotlib Agg + English
plot labels; honest Chinese "跳过" degrade; never crash / never fabricate.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches._emergy_transformities import (
    BASELINE,
    DISCLAIMER,
    lookup_transformity,
)


def _to_float(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


@register("emergy_analysis")
def _branch_emergy_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    method = entry.method

    import numpy as np
    import pandas as pd

    excl = {fp.unit_col, fp.time_col}
    # candidate input flow columns: numeric-coercible, not the unit/time label
    cand = []
    for c in fp.columns:
        if c.name in excl:
            continue
        s = pd.to_numeric(df[c.name], errors="coerce")
        if s.notna().sum() > 0 and float(s.fillna(0.0).sum()) != 0.0:
            cand.append(c.name)
    if not cand:
        summary.append(f"{method} 跳过：未找到可用的数值输入流列。")
        return

    cfg_tr = cfg.get("transformities") if isinstance(cfg.get("transformities"), dict) else {}
    cfg_cat = cfg.get("categories") if isinstance(cfg.get("categories"), dict) else {}
    out_col = cfg.get("output") if cfg.get("output") in df.columns else (
        cfg.get("product") if cfg.get("product") in df.columns else None)
    money_col = cfg.get("money") if cfg.get("money") in df.columns else None
    reserved = {out_col, money_col}

    rows, unmatched, used_src = [], [], []
    for col in cand:
        if col in reserved:
            continue
        qty = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        tr = _to_float(cfg_tr.get(col))
        cat = cfg_cat.get(col)
        unit = "sej/unit"
        if tr is not None:
            src = "config transformities"
        else:
            hit = lookup_transformity(str(col))
            if hit is None:
                unmatched.append(col)
                continue
            tr, unit, lib_cat, meta = hit
            cat = cat or lib_cat
            src = f"{meta.get('source', '库')}"
        cat = str(cat or "F").upper()
        if cat not in {"R", "N", "F"}:
            cat = "F"
        emergy = qty * float(tr)
        rows.append({"input": col, "quantity": qty, "transformity": float(tr),
                     "unit": unit, "category": cat, "emergy": emergy})
        used_src.append(f"{col}={float(tr):.3g} {unit}[{cat}]")

    if not rows:
        summary.append(
            f"{method} 跳过：没有可解析转换率的输入流"
            "（用 config transformities={列:sej/单位}、categories={列:R/N/F} 指定，"
            "或让列名匹配内置转换率库，如 solar/wind/rain/electricity/natural_gas/labor）。"
        )
        return

    R = float(sum(r["emergy"] for r in rows if r["category"] == "R"))
    N = float(sum(r["emergy"] for r in rows if r["category"] == "N"))
    F = float(sum(r["emergy"] for r in rows if r["category"] == "F"))
    U = R + N + F

    def _safe(num, den):
        return float(num / den) if den not in (0, 0.0) else float("inf")

    eyr = _safe(U, F)            # emergy yield ratio
    elr = _safe(N + F, R)       # environmental loading ratio
    esi = _safe(eyr, elr) if np.isfinite(elr) and elr != 0 else float("inf")
    pct_renew = _safe(R, U) if U else float("nan")

    out_transformity, emr = float("nan"), float("nan")
    if out_col is not None:
        out_tot = float(pd.to_numeric(df[out_col], errors="coerce").fillna(0.0).sum())
        if out_tot > 0:
            out_transformity = U / out_tot
    if money_col is not None:
        money_tot = float(pd.to_numeric(df[money_col], errors="coerce").fillna(0.0).sum())
        if money_tot > 0:
            emr = U / money_tot

    estimates.update({
        "total_emergy_U": round(U, 6), "emergy_R": round(R, 6),
        "emergy_N": round(N, 6), "emergy_F": round(F, 6),
        "eyr": round(eyr, 6) if np.isfinite(eyr) else float("inf"),
        "elr": round(elr, 6) if np.isfinite(elr) else float("inf"),
        "esi": round(esi, 6) if np.isfinite(esi) else float("inf"),
        "pct_renewable": round(pct_renew, 6) if pct_renew == pct_renew else float("nan"),
        "n_inputs": float(len(rows)),
        "output_transformity": round(out_transformity, 6) if out_transformity == out_transformity else float("nan"),
        "emr": round(emr, 6) if emr == emr else float("nan"),
    })

    try:
        pd.DataFrame(rows).to_csv(d / "emergy_inputs.csv", index=False, encoding="utf-8")
        files.append("emergy_inputs.csv")
    except Exception:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.bar(["Renewable (R)", "Non-renewable (N)", "Purchased (F)"], [R, N, F],
               color=["#55A868", "#C44E52", "#4C72B0"])
        ax.set_ylabel("emergy (sej)")
        ax.set_title("Emergy by source (R / N / F)")
        for i, v in enumerate([R, N, F]):
            ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(d / "emergy_sources.png", dpi=150)
        plt.close("all")
        files.append("emergy_sources.png")
    except Exception:
        pass

    def _fmt(x):
        return "∞" if not np.isfinite(x) else f"{x:.3g}"

    summary.append(
        f"{method} 完成：{len(rows)} 个输入流折算为太阳能值(sej)。"
        f"R(可再生)={R:.3g}、N(不可再生)={N:.3g}、F(外购/反馈)={F:.3g}、总能值 U={U:.3g} sej。"
        f"能值产出率 EYR=U/F={_fmt(eyr)}、环境负载率 ELR=(N+F)/R={_fmt(elr)}、"
        f"可持续指数 ESI=EYR/ELR={_fmt(esi)}、可再生比例={pct_renew:.1%}。"
        + (f" 产出能值转换率=U/产出={_fmt(out_transformity)} sej/单位。" if out_transformity == out_transformity else "")
        + (f" 能值货币比 EMR={_fmt(emr)} sej/货币。" if emr == emr else "")
        + (f" 未匹配(已排除)：{', '.join(unmatched)}。" if unmatched else "")
        + f" 转换率来源：{'；'.join(used_src)}。"
        f" ⚠ 能值基线={BASELINE}；{DISCLAIMER} R/N/F 分类为建模选择，且应避免重复计算。"
    )
    code.append(
        "# emergy = Σ(input_quantity × transformity[sej/unit]); 按 R/N/F 聚合\n"
        "# EYR=U/F; ELR=(N+F)/R; ESI=EYR/ELR; %renewable=R/U  (U=R+N+F)"
    )
