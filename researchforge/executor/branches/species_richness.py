"""Branch handler for the ecology family: nonparametric species-richness
estimators (how many species are really there, including those never detected).

NEW family module (auto-discovered by branches/__init__'s walk_packages). Pure
Python (numpy) — no R, no network. Does NOT touch ecology.py.

Estimators (Chao & Chiu 2016, "Estimating and Comparing Species Richness ...";
Colwell EstimateS user guide; Chao 1984/1987; Chao & Lee 1992; Burnham &
Overton 1978/1979 for the jackknives):

Two regimes, chosen automatically:

ABUNDANCE-based (the cell values are individual counts). Pool counts across
rows -> per-species total abundance. With
    S_obs = #species with total > 0
    f1    = #species with total == 1   (SINGLETONS)
    f2    = #species with total == 2   (DOUBLETONS)
    n     = total individuals
  * Chao1 (bias-corrected form, valid when f2 may be 0):
        Chao1 = S_obs + f1*(f1-1) / (2*(f2+1))
  * ACE (Abundance-based Coverage Estimator; Chao & Lee 1992, EstimateS):
    split species into "rare" (abundance <= 10) and "abundant" (> 10).
        S_rare  = #rare species,        S_abund = #abundant species
        N_rare  = sum of abundances over rare species
        C_ace   = 1 - f1 / N_rare                              (sample coverage)
        gamma^2 = max( (S_rare/C_ace) * (sum_i i*(i-1)*f_i) / (N_rare*(N_rare-1)) - 1, 0 )
        ACE     = S_abund + S_rare/C_ace + (f1/C_ace) * gamma^2
    (falls back to Chao1 when C_ace == 0, i.e. every rare individual is a singleton.)
  * Chao1 log-normal 95% CI (Chao 1987 / Colwell): T = Chao1 - S_obs,
        var(Chao1) = f1*(f1-1)/(2*(f2+1)) + f1*(2*f1-1)^2/(4*(f2+1)^2)
                     + f1^2*f2*(f1-1)^2/(4*(f2+1)^4)            (f2 > 0 form)
    K = exp( 1.96 * sqrt( ln(1 + var/T^2) ) );  CI = [S_obs + T/K, S_obs + T*K].

INCIDENCE-based (the cell values are 0/1 presence, OR an abundance table that we
also binarize). Per-species incidence = #sites in which detected. With
    q1 = #species detected in EXACTLY 1 site   (UNIQUES)
    q2 = #species detected in EXACTLY 2 sites  (DUPLICATES)
    m  = #sites (sampling units)
  * Chao2 (bias-corrected):
        Chao2 = S_obs + ((m-1)/m) * q1*(q1-1) / (2*(q2+1))
  * 1st-order jackknife (Burnham & Overton):  Jack1 = S_obs + q1*(m-1)/m
  * 2nd-order jackknife:
        Jack2 = S_obs + q1*(2m-3)/m - q2*(m-2)^2 / (m*(m-1))

Regime decision: if every value is 0/1 -> incidence regime (Chao2 + jackknives,
abundance estimators NaN). Otherwise -> abundance regime (Chao1 + ACE primary)
AND additionally binarize to report the incidence estimators too (both labeled).

CLASSIC BUG GUARDED: f1/f2 (singletons/doubletons of *abundance*) are NOT the
same as q1/q2 (uniques/duplicates of *incidence*). They are computed separately.
"""

from __future__ import annotations

import math

from researchforge.executor._branch_api import Ctx, register

_NAN = float("nan")


def _numeric_species_cols(fp, df) -> list[str]:
    """Numeric species columns: count / binary / integer-like continuous / id,
    excluding unit & time roles. Site x species cells are abundances (count, or
    integer continuous) or presence (binary). Falls back to all-numeric if the
    profiler typed everything oddly."""
    excl = {fp.unit_col, fp.time_col}
    import pandas as pd

    cols: list[str] = []
    for c in fp.columns:
        if c.name in excl or c.name not in df.columns:
            continue
        if c.kind in {"count", "binary", "id", "continuous"}:
            s = pd.to_numeric(df[c.name], errors="coerce").dropna()
            if len(s) and bool((s >= 0).all()):  # abundances/incidences are non-negative
                cols.append(c.name)
    return cols


@register("species_richness")
def _branch_species_richness(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    # --- resolve species columns (config override else auto) ------------------
    requested = cfg.get("species")
    if requested:
        species = [c for c in list(requested) if c in df.columns]
    else:
        species = _numeric_species_cols(fp, df)

    if len(species) < 2:
        summary.append(
            "物种丰富度估计跳过：需要 ≥2 个物种列（站点×物种表，单元格为丰度计数或 0/1 出现）。"
            f"（自动检出物种列={species}；可用 config species=[...] 指定。）"
        )
        return

    mat = df[species].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    m = int(mat.shape[0])  # sites / samples
    if m < 1:
        summary.append("物种丰富度估计跳过：无有效站点行。")
        return

    # --- pooled abundance per species, and the regime decision ----------------
    per_species_abund = mat.sum(axis=0)          # total abundance per species
    incidence = (mat > 0).sum(axis=0)            # #sites detected per species
    present = per_species_abund > 0
    s_obs = int(present.sum())
    n_individuals = float(per_species_abund.sum())

    if s_obs < 1 or n_individuals <= 0:
        summary.append("物种丰富度估计跳过：所有物种丰度均为 0（无正计数/出现）。")
        return

    # all cells 0/1 -> the data themselves are incidence (presence/absence)
    is_incidence_data = bool(np.all((mat == 0) | (mat == 1)))
    regime = "incidence" if is_incidence_data else "abundance"

    # ===================== ABUNDANCE estimators ==============================
    chao1 = chao1_lo = chao1_hi = ace = _NAN
    f1 = f2 = _NAN
    if regime == "abundance":
        ab = per_species_abund[present]                       # >0 abundances
        f1 = float((ab == 1).sum())                          # SINGLETONS
        f2 = float((ab == 2).sum())                          # DOUBLETONS

        # Chao1 (bias-corrected; valid when f2 may be 0)
        chao1 = s_obs + f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))

        # Chao1 log-normal 95% CI (Chao 1987 / Colwell). T = extra species.
        T = chao1 - s_obs
        if T > 0:
            if f2 > 0:
                var = (
                    f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))
                    + f1 * (2.0 * f1 - 1.0) ** 2 / (4.0 * (f2 + 1.0) ** 2)
                    + f1 ** 2 * f2 * (f1 - 1.0) ** 2 / (4.0 * (f2 + 1.0) ** 4)
                )
            else:
                # doubleton-free form (Chao 1987)
                var = (
                    f1 * (f1 - 1.0) / 2.0
                    + f1 * (2.0 * f1 - 1.0) ** 2 / 4.0
                    - f1 ** 4 / (4.0 * chao1)
                )
            var = max(var, 0.0)
            if var > 0:
                K = math.exp(1.96 * math.sqrt(math.log(1.0 + var / (T * T))))
                chao1_lo = s_obs + T / K
                chao1_hi = s_obs + T * K
            else:
                chao1_lo = chao1_hi = chao1
        else:
            chao1_lo = chao1_hi = float(s_obs)

        # ACE (Chao & Lee 1992 / EstimateS): rare = abundance <= 10
        rare_mask = ab <= 10
        s_rare = float(rare_mask.sum())
        s_abund = float((~rare_mask).sum())
        n_rare = float(ab[rare_mask].sum())
        if s_rare == 0:
            ace = float(s_abund)                              # nothing rare -> all observed
        else:
            c_ace = 1.0 - (f1 / n_rare) if n_rare > 0 else 0.0
            if c_ace <= 0:
                ace = chao1                                   # degenerate coverage -> fall back
            else:
                # sum_i i*(i-1)*f_i over rare classes i=1..10
                ssum = 0.0
                for i in range(1, 11):
                    fi = float((ab == i).sum())
                    ssum += i * (i - 1.0) * fi
                if n_rare > 1:
                    gamma2 = max(
                        (s_rare / c_ace) * ssum / (n_rare * (n_rare - 1.0)) - 1.0, 0.0
                    )
                else:
                    gamma2 = 0.0
                ace = s_abund + s_rare / c_ace + (f1 / c_ace) * gamma2

    # ===================== INCIDENCE estimators ==============================
    # computed whenever m >= 2; in abundance regime we binarize to also report them
    chao2 = jack1 = jack2 = _NAN
    q1 = q2 = _NAN
    if m >= 2:
        inc = incidence[present]                              # >=1 incidences
        q1 = float((inc == 1).sum())                         # UNIQUES
        q2 = float((inc == 2).sum())                         # DUPLICATES
        mm = float(m)
        chao2 = s_obs + ((mm - 1.0) / mm) * q1 * (q1 - 1.0) / (2.0 * (q2 + 1.0))
        jack1 = s_obs + q1 * (mm - 1.0) / mm
        jack2 = (
            s_obs
            + q1 * (2.0 * mm - 3.0) / mm
            - q2 * (mm - 2.0) ** 2 / (mm * (mm - 1.0))
        )

    # --- primary estimator + completeness ------------------------------------
    if regime == "abundance":
        primary_val, primary_name = chao1, "Chao1"
        n_sing = f1
        n_doub = f2
    else:
        primary_val, primary_name = chao2, "Chao2"
        n_sing = q1
        n_doub = q2
    completeness = (
        float(s_obs) / primary_val if (primary_val and primary_val > 0 and not math.isnan(primary_val)) else _NAN
    )

    # --- estimates (plain floats; CI separate keys; NaN where N/A) ------------
    def _f(x):
        return float(x) if x is not None else _NAN

    estimates["s_observed"] = float(s_obs)
    estimates["chao1"] = _f(chao1)
    estimates["chao1_ci_low"] = _f(chao1_lo)
    estimates["chao1_ci_high"] = _f(chao1_hi)
    estimates["ace"] = _f(ace)
    estimates["chao2"] = _f(chao2)
    estimates["jackknife1"] = _f(jack1)
    estimates["jackknife2"] = _f(jack2)
    estimates["n_singletons"] = _f(n_sing)
    estimates["n_doubletons"] = _f(n_doub)
    estimates["completeness"] = _f(completeness)
    estimates["n_sites"] = float(m)
    estimates["n_individuals"] = float(n_individuals)

    # --- CSV: estimators side by side ----------------------------------------
    rows = [
        {"estimator": "S_observed", "richness": round(float(s_obs), 4),
         "ci_low": _NAN, "ci_high": _NAN, "regime": "—"},
        {"estimator": "Chao1", "richness": round(chao1, 4) if not math.isnan(chao1) else _NAN,
         "ci_low": round(chao1_lo, 4) if not math.isnan(chao1_lo) else _NAN,
         "ci_high": round(chao1_hi, 4) if not math.isnan(chao1_hi) else _NAN,
         "regime": "abundance"},
        {"estimator": "ACE", "richness": round(ace, 4) if not math.isnan(ace) else _NAN,
         "ci_low": _NAN, "ci_high": _NAN, "regime": "abundance"},
        {"estimator": "Chao2", "richness": round(chao2, 4) if not math.isnan(chao2) else _NAN,
         "ci_low": _NAN, "ci_high": _NAN, "regime": "incidence"},
        {"estimator": "Jackknife1", "richness": round(jack1, 4) if not math.isnan(jack1) else _NAN,
         "ci_low": _NAN, "ci_high": _NAN, "regime": "incidence"},
        {"estimator": "Jackknife2", "richness": round(jack2, 4) if not math.isnan(jack2) else _NAN,
         "ci_low": _NAN, "ci_high": _NAN, "regime": "incidence"},
    ]
    try:
        pd.DataFrame(rows).to_csv(
            d / "richness_estimators.csv", index=False, encoding="utf-8"
        )
        files.append("richness_estimators.csv")
    except Exception:
        pass

    # --- PNG: S_obs vs estimators (with Chao1 CI when abundance) -------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels, vals, errs, colors = [], [], [], []
        labels.append("S_obs"); vals.append(float(s_obs)); errs.append(0.0); colors.append("#4C72B0")
        if not math.isnan(chao1):
            labels.append("Chao1"); vals.append(chao1)
            lo = chao1_lo if not math.isnan(chao1_lo) else chao1
            hi = chao1_hi if not math.isnan(chao1_hi) else chao1
            errs.append(max(hi - chao1, chao1 - lo)); colors.append("#C44E52")
        if not math.isnan(ace):
            labels.append("ACE"); vals.append(ace); errs.append(0.0); colors.append("#C44E52")
        if not math.isnan(chao2):
            labels.append("Chao2"); vals.append(chao2); errs.append(0.0); colors.append("#55A868")
        if not math.isnan(jack1):
            labels.append("Jack1"); vals.append(jack1); errs.append(0.0); colors.append("#55A868")
        if not math.isnan(jack2):
            labels.append("Jack2"); vals.append(jack2); errs.append(0.0); colors.append("#55A868")

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        x = range(len(labels))
        # asymmetric CI only for Chao1; everything else has 0 error
        ax.bar(x, vals, color=colors, alpha=0.85)
        if not math.isnan(chao1) and not math.isnan(chao1_lo):
            ci_idx = labels.index("Chao1")
            ax.errorbar(
                [ci_idx], [chao1],
                yerr=[[chao1 - chao1_lo], [chao1_hi - chao1]],
                fmt="none", ecolor="black", capsize=4, lw=1.2,
            )
        ax.axhline(float(s_obs), color="grey", ls="--", lw=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("estimated species richness")
        ax.set_title(f"Richness estimators ({regime} regime, S_obs={s_obs})")
        fig.tight_layout()
        fig.savefig(d / "richness_estimators.png", dpi=150)
        plt.close(fig)
        files.append("richness_estimators.png")
    except Exception:
        pass

    # --- Chinese summary with disclosures ------------------------------------
    comp_pct = f"{completeness * 100:.1f}%" if not math.isnan(completeness) else "—"
    if regime == "abundance":
        ci_txt = (
            f"（95% CI [{chao1_lo:.1f}, {chao1_hi:.1f}]）"
            if not math.isnan(chao1_lo) else ""
        )
        head = (
            f"{entry.method} 完成（丰度法）：{len(species)} 个观测物种列 × {m} 个站点，"
            f"实测物种数 S_obs={s_obs}，个体总数 {int(n_individuals)}；"
            f"Chao1 估计真实物种数 ≈ {chao1:.1f}{ci_txt}"
            + (f"，ACE ≈ {ace:.1f}" if not math.isnan(ace) else "")
            + f"；样本完整度 S_obs/Chao1 ≈ {comp_pct}"
            f"（单种 f1={int(f1)}，双种 f2={int(f2)}）。"
        )
        if not math.isnan(chao2):
            head += f" 另按出现频率（二值化后）：Chao2 ≈ {chao2:.1f}，Jackknife1 ≈ {jack1:.1f}，Jackknife2 ≈ {jack2:.1f}。"
    else:
        head = (
            f"{entry.method} 完成（出现/频率法，数据为 0/1）：{len(species)} 个观测物种列 × {m} 个站点，"
            f"实测物种数 S_obs={s_obs}；Chao2 估计真实物种数 ≈ {chao2:.1f}，"
            f"Jackknife1 ≈ {jack1:.1f}，Jackknife2 ≈ {jack2:.1f}；"
            f"样本完整度 S_obs/Chao2 ≈ {comp_pct}（唯一种 q1={int(q1)}，二重种 q2={int(q2)}）。"
        )
    summary.append(
        head
        + " ⚠ 这些估计的是**包含未检出物种**的真实丰富度，重度依赖稀有物种"
        "（丰度法的单种/双种 f1/f2、出现法的唯一/二重种 q1/q2），对采样努力与 f1/f2(q1/q2) 的随机噪声敏感；"
        "⚠ Chao 估计量是真实丰富度的**下界**（采样不足时偏低）；"
        "⚠ 丰度法 vs 出现法基于不同信息（个体计数 vs 站点出现），二者数值不可直接互换、各自适用；"
        "⚠ 估计量不能替代充分采样——完整度低（如 <80%）时应加大采样而非仅靠外推。"
    )

    code += [
        "import numpy as np  # 非参数物种丰富度估计 (Chao & Chiu 2016 / EstimateS)",
        f"mat = df[{species!r}].fillna(0).clip(lower=0).to_numpy(float)",
        "ab = mat.sum(0); ab = ab[ab>0]            # 丰度法: 各物种总丰度",
        "f1=(ab==1).sum(); f2=(ab==2).sum(); S=len(ab)  # 单种/双种",
        "chao1 = S + f1*(f1-1)/(2*(f2+1))          # Chao1 偏差校正",
        "inc=(mat>0).sum(0); inc=inc[inc>0]; m=mat.shape[0]",
        "q1=(inc==1).sum(); q2=(inc==2).sum()       # 出现法: 唯一/二重种",
        "chao2 = S + ((m-1)/m)*q1*(q1-1)/(2*(q2+1)) # Chao2",
        "jack1 = S + q1*(m-1)/m                      # 一阶刀切",
        "jack2 = S + q1*(2*m-3)/m - q2*(m-2)**2/(m*(m-1))  # 二阶刀切",
    ]
