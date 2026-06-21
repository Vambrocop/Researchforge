"""Branch handlers for the Item Response Theory (IRT) psychometrics family.

Pure-Python latent-trait measurement on a respondent x binary-item 0/1 matrix,
estimated by ``girth`` via marginal maximum likelihood (MML — integrates the
person-ability theta out against a N(0,1) prior, so only item parameters are
free during fitting; abilities are scored afterwards by EAP).

  * ``irt_2pl``   — 2-parameter logistic model. Each item j has a discrimination
    a_j (slope — how sharply it separates abilities) and a difficulty b_j
    (the ability theta at which P(correct)=0.5). ICC_j(theta) = 1/(1+exp(-a_j(theta-b_j))).
  * ``irt_rasch`` — 1-parameter / Rasch model. ALL items share one discrimination
    (default a=1), only difficulty b_j varies, giving the model's signature
    *parallel* (non-crossing) item characteristic curves. We also fit a 2PL on the
    same data and report both log-likelihoods for an informal Rasch-holds check.

girth orientation (verified against girth's docstrings — see tests): the fitting
and ability functions take an **items x persons** binary matrix, i.e. the
TRANSPOSE of our respondents x items frame. Getting this backwards silently flips
every parameter, so the transpose is done once, explicitly, right after resolving
the item matrix, and the recovery tests assert correlation with a known truth.

Engine conventions (see CLAUDE.md「引擎约定」): handlers mutate
summary/estimates/files/code (never rebind); items default to binary 0/1 columns
and are overridable via config ``items``/``columns``; products are CSV + a
best-effort English-labelled Agg PNG; estimates hold floats; the summary is Chinese
and ends with ⚠ disclosures; on failure append a Chinese "<方法>失败/跳过：<reason>"
line and return.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Minimum respondents below which MML 2PL estimates are too unstable to trust.
_MIN_RESP_2PL = 100
_MIN_RESP_RASCH = 30  # Rasch has far fewer parameters, so it tolerates smaller n
_MIN_ITEMS = 3


# ---------------------------------------------------------------------------
# Shared helpers (local to this family)
# ---------------------------------------------------------------------------

def _resolve_irt_items(ctx: Ctx) -> list[str]:
    """Pick the binary item columns for the IRT model.

    Priority: config ``items`` then config ``columns`` (validated against the
    frame); otherwise the auto default of every ``binary`` column excluding the
    panel unit/time columns. A 0/1 dichotomous item profiles as ``binary``.
    """
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    forced = cfg.get("items") or cfg.get("columns")
    if forced:
        return [c for c in forced if c in df.columns]
    excl = {fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind == "binary" and c.name not in excl]


def _binary_item_matrix(ctx: Ctx, items: list[str]):
    """Return (respondents x items) INTEGER 0/1 numpy matrix or None if not 0/1.

    Coerces to numeric, drops rows with any missing item, and verifies every
    retained value is exactly 0 or 1 (IRT here is dichotomous — polytomous/Likert
    data must be dichotomised by the user first). Returns INT (not float): girth
    indexes its likelihood tables with the response matrix and raises
    "arrays used as indices must be of integer type" on a float matrix.
    """
    import numpy as np
    import pandas as pd

    sub = ctx.df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] == 0:
        return None
    X = sub.to_numpy(dtype=float)
    uniq = set(np.unique(X).tolist())
    if not uniq <= {0.0, 1.0}:
        return None
    return X.astype(int)


def _extract_params(result):
    """Pull (discrimination_array, difficulty_array) out of a girth result.

    girth's MML estimators return a dict with keys ``Discrimination`` and
    ``Difficulty`` (discrimination may be a scalar for Rasch/1PL — broadcast it).
    A defensive (disc, diff) tuple path is kept in case the installed build
    returns a bare tuple.
    """
    import numpy as np

    if isinstance(result, dict):
        disc = result.get("Discrimination")
        diff = result.get("Difficulty")
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        disc, diff = result[0], result[1]
    else:
        raise ValueError(f"unrecognised girth result type: {type(result)!r}")
    diff = np.atleast_1d(np.asarray(diff, dtype=float))
    disc = np.asarray(disc, dtype=float)
    if disc.ndim == 0:  # scalar discrimination (Rasch) -> broadcast to all items
        disc = np.full(diff.shape, float(disc))
    else:
        disc = np.atleast_1d(disc)
    return disc, diff


def _twopl_loglik(X_ip, disc, diff, theta) -> float:
    """2PL log-likelihood at the fitted item params and EAP thetas.

    X_ip is items x persons (girth orientation). Plugs the EAP point estimates of
    theta into the per-response Bernoulli log-likelihood — an informal model-fit
    number for comparing Rasch vs 2PL on the same data (NOT a formal LR test).
    """
    import numpy as np

    z = disc[:, None] * (theta[None, :] - diff[:, None])
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.sum(X_ip * np.log(p) + (1.0 - X_ip) * np.log(1.0 - p)))


def _estimate_abilities(X_ip, diff, disc):
    """EAP person abilities; X_ip is items x persons (girth orientation)."""
    import numpy as np

    import girth

    theta = np.asarray(girth.ability_eap(X_ip, diff, disc), dtype=float)
    return theta


def _plot_iccs(d, files, items, disc, diff, title: str, fname: str) -> None:
    """Item characteristic curves P(correct)=1/(1+exp(-a(theta-b))) over theta."""
    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theta_grid = np.linspace(-4.0, 4.0, 200)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        cmap = plt.get_cmap("viridis")
        k = len(items)
        for j, name in enumerate(items):
            p = 1.0 / (1.0 + np.exp(-disc[j] * (theta_grid - diff[j])))
            ax.plot(theta_grid, p, color=cmap(j / max(1, k - 1)), lw=1.4, label=str(name))
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("Ability (theta)")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        if k <= 14:
            ax.legend(fontsize=7, ncol=2, loc="lower right")
        fig.tight_layout()
        fig.savefig(d / fname, dpi=150)
        plt.close(fig)
        files.append(fname)
    except Exception:
        pass


# ===========================================================================
# 1. 2-parameter logistic IRT (girth.twopl_mml)
# ===========================================================================

@register("irt_2pl")
def _branch_irt_2pl(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    model = str(cfg.get("model", "2pl")).lower()
    if model not in {"2pl", "2-pl", "twopl"}:
        summary.append(f"2PL IRT 跳过：config model={model!r} 非 2PL（本分支仅支持 2pl）。")
        return

    items = _resolve_irt_items(ctx)
    if len(items) < _MIN_ITEMS:
        summary.append(f"2PL IRT 跳过：需要 ≥{_MIN_ITEMS} 个二值(0/1)题项，当前不足。")
        return

    X = _binary_item_matrix(ctx, items)
    if X is None:
        summary.append("2PL IRT 跳过：题项含缺失剔除后无数据，或题项不是 0/1 二值（多分/Likert 需先二分化）。")
        return
    n_resp, k = X.shape  # respondents x items
    if n_resp < 20:
        summary.append("2PL IRT 跳过：被试过少（<20），MML 无法收敛到可信解。")
        return

    try:
        import girth

        # girth wants items x persons -> transpose our respondents x items matrix.
        X_ip = X.T
        res = girth.twopl_mml(X_ip)
        disc, diff = _extract_params(res)
        if disc.shape[0] != k or diff.shape[0] != k:
            summary.append(
                f"2PL IRT 失败：girth 返回参数长度({disc.shape[0]}/{diff.shape[0]})与题项数({k})不符。"
            )
            return

        theta = _estimate_abilities(X_ip, diff, disc)
        ll = _twopl_loglik(X_ip, disc, diff, theta)

        # --- item parameter table ---
        item_df = pd.DataFrame(
            {
                "item": items,
                "discrimination_a": np.round(disc, 4),
                "difficulty_b": np.round(diff, 4),
                "p_correct": np.round(X.mean(axis=0), 4),  # raw proportion correct
            }
        )
        item_df.to_csv(d / "irt_2pl_item_params.csv", index=False, encoding="utf-8")
        files.append("irt_2pl_item_params.csv")

        ability_df = pd.DataFrame(
            {"respondent": np.arange(n_resp), "raw_score": X.sum(axis=1).astype(int), "theta_eap": np.round(theta, 4)}
        )
        ability_df.to_csv(d / "irt_2pl_abilities.csv", index=False, encoding="utf-8")
        files.append("irt_2pl_abilities.csv")

        _plot_iccs(d, files, items, disc, diff, "2PL Item Characteristic Curves", "irt_2pl_iccs.png")

        # --- item-fit notes: flag low-discrimination / extreme-difficulty items ---
        low_disc = [items[i] for i in range(k) if disc[i] < 0.4]
        extreme_b = [items[i] for i in range(k) if abs(diff[i]) > 3.0]

        estimates["n_items"] = float(k)
        estimates["n_respondents"] = float(n_resp)
        estimates["mean_discrimination"] = round(float(np.mean(disc)), 4)
        estimates["mean_difficulty"] = round(float(np.mean(diff)), 4)
        estimates["min_discrimination"] = round(float(np.min(disc)), 4)
        estimates["loglik_2pl"] = round(ll, 3)
        estimates["theta_mean"] = round(float(np.mean(theta)), 4)
        estimates["theta_sd"] = round(float(np.std(theta, ddof=1)), 4) if n_resp > 1 else 0.0

        stability = "" if n_resp >= _MIN_RESP_2PL else f"⚠ 被试仅 {n_resp}（<~{_MIN_RESP_2PL}），2PL 估计偏不稳。"
        msg = (
            f"{entry.method} 完成：{k} 个二值题项 × {n_resp} 个被试（girth MML）。"
            f"区分度 a 均值 {np.mean(disc):.3f}（范围 {np.min(disc):.2f}–{np.max(disc):.2f}）、"
            f"难度 b 均值 {np.mean(diff):.3f}（范围 {np.min(diff):.2f}–{np.max(diff):.2f}）；"
            f"能力 θ(EAP) 均值 {np.mean(theta):.2f}、SD {estimates['theta_sd']:.2f}；模型对数似然 {ll:.1f}。"
        )
        if stability:
            msg += " " + stability
        if low_disc:
            msg += f" ⚠ 低区分度题项（a<0.4，分辨力弱）：{', '.join(map(str, low_disc))}。"
        if extreme_b:
            msg += f" ⚠ 极端难度题项（|b|>3，几乎全对/全错）：{', '.join(map(str, extreme_b))}。"
        msg += (
            " ⚠ 2PL 假设单维(单一潜在能力)+局部独立(控制 θ 后题项作答互不相关)+单调性(P 随 θ 增)；"
            "区分度 a=题项斜率(越大越能区分高低能力)，难度 b=答对率 50% 处的能力水平；"
            "需 0/1 二值题项 + 足够被试(2PL 稳定通常需 ≥~100)；MML 把能力按 N(0,1) 先验积分掉、"
            "故 θ 以标准正态为尺度锚定(均值~0、SD~1)，θ 是事后 EAP 评分而非与题参联合估计。"
        )
        summary.append(msg)
        code += [
            "import girth, numpy as np  # 2PL IRT — marginal MML",
            f"X = df[{items!r}].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(float)  # respondents x items",
            "res = girth.twopl_mml(X.T)  # girth wants items x persons -> TRANSPOSE",
            "a, b = res['Discrimination'], res['Difficulty']  # per-item slope / difficulty",
            "theta = girth.ability_eap(X.T, b, a)  # EAP person abilities",
            "# ICC_j(theta) = 1/(1+exp(-a_j*(theta-b_j)))",
        ]
    except Exception as err:
        summary.append(f"2PL IRT 失败：{err}")


# ===========================================================================
# 2. Rasch (1-parameter) IRT (girth.rasch_mml)
# ===========================================================================

@register("irt_rasch")
def _branch_irt_rasch(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    items = _resolve_irt_items(ctx)
    if len(items) < _MIN_ITEMS:
        summary.append(f"Rasch IRT 跳过：需要 ≥{_MIN_ITEMS} 个二值(0/1)题项，当前不足。")
        return

    X = _binary_item_matrix(ctx, items)
    if X is None:
        summary.append("Rasch IRT 跳过：题项含缺失剔除后无数据，或题项不是 0/1 二值（多分/Likert 需先二分化）。")
        return
    n_resp, k = X.shape
    if n_resp < 20:
        summary.append("Rasch IRT 跳过：被试过少（<20），MML 无法收敛到可信解。")
        return

    try:
        import girth

        X_ip = X.T  # items x persons (girth orientation)

        # Rasch: one shared discrimination (default a=1), per-item difficulty.
        res_r = girth.rasch_mml(X_ip)
        disc_r, diff_r = _extract_params(res_r)
        if diff_r.shape[0] != k:
            summary.append(f"Rasch IRT 失败：girth 难度参数长度({diff_r.shape[0]})与题项数({k})不符。")
            return
        shared_a = float(np.mean(disc_r))  # Rasch broadcasts one a to every item

        theta = _estimate_abilities(X_ip, diff_r, disc_r)
        ll_rasch = _twopl_loglik(X_ip, disc_r, diff_r, theta)

        # Informal comparison: fit a 2PL on the SAME data and report its log-lik.
        # 2PL nests Rasch (frees the equal-discrimination constraint), so its
        # log-lik is >= Rasch's; a big gap suggests Rasch's equal-a assumption is
        # violated. We do NOT run a formal LR chi-square (df, regularity caveats).
        ll_2pl = float("nan")
        try:
            res2 = girth.twopl_mml(X_ip)
            disc2, diff2 = _extract_params(res2)
            if disc2.shape[0] == k and diff2.shape[0] == k:
                theta2 = _estimate_abilities(X_ip, diff2, disc2)
                ll_2pl = _twopl_loglik(X_ip, disc2, diff2, theta2)
        except Exception:
            ll_2pl = float("nan")

        # --- person / item separation reliability ---
        # Person separation reliability ~ (Var(theta) - mean SE^2) / Var(theta),
        # a Rasch analogue of classical reliability. girth gives no per-person SE,
        # so we approximate SE from the test information at each theta:
        #   I(theta) = sum_j a^2 * P_j(theta)(1-P_j(theta)),  SE = 1/sqrt(I).
        z = disc_r[:, None] * (theta[None, :] - diff_r[:, None])  # items x persons
        p = 1.0 / (1.0 + np.exp(-z))
        info = np.sum((disc_r[:, None] ** 2) * p * (1.0 - p), axis=0)  # per person
        info = np.clip(info, 1e-6, None)
        se2 = 1.0 / info  # per-person error variance
        var_theta = float(np.var(theta, ddof=1)) if n_resp > 1 else 0.0
        mean_se2 = float(np.mean(se2))
        if var_theta > mean_se2 > 0:
            person_rel = (var_theta - mean_se2) / var_theta
        else:
            person_rel = float("nan")

        item_df = pd.DataFrame(
            {
                "item": items,
                "difficulty_b": np.round(diff_r, 4),
                "discrimination_a_shared": np.round(disc_r, 4),
                "p_correct": np.round(X.mean(axis=0), 4),
            }
        )
        item_df.to_csv(d / "irt_rasch_item_params.csv", index=False, encoding="utf-8")
        files.append("irt_rasch_item_params.csv")

        ability_df = pd.DataFrame(
            {
                "respondent": np.arange(n_resp),
                "raw_score": X.sum(axis=1).astype(int),
                "theta_eap": np.round(theta, 4),
                "theta_se": np.round(np.sqrt(se2), 4),
            }
        )
        ability_df.to_csv(d / "irt_rasch_abilities.csv", index=False, encoding="utf-8")
        files.append("irt_rasch_abilities.csv")

        _plot_iccs(
            d, files, items, disc_r, diff_r,
            "Rasch Item Characteristic Curves (parallel)", "irt_rasch_iccs.png",
        )

        extreme_b = [items[i] for i in range(k) if abs(diff_r[i]) > 3.0]

        estimates["n_items"] = float(k)
        estimates["n_respondents"] = float(n_resp)
        estimates["shared_discrimination"] = round(shared_a, 4)
        estimates["mean_difficulty"] = round(float(np.mean(diff_r)), 4)
        estimates["loglik_rasch"] = round(ll_rasch, 3)
        estimates["loglik_2pl"] = round(ll_2pl, 3) if ll_2pl == ll_2pl else -1.0
        estimates["person_separation_reliability"] = (
            round(float(person_rel), 4) if person_rel == person_rel else -1.0
        )
        estimates["theta_mean"] = round(float(np.mean(theta)), 4)
        estimates["theta_sd"] = round(float(np.std(theta, ddof=1)), 4) if n_resp > 1 else 0.0

        rel_txt = f"{person_rel:.3f}" if person_rel == person_rel else "不可用"
        ll_gap = (ll_2pl - ll_rasch) if ll_2pl == ll_2pl else float("nan")
        cmp_txt = (
            f"同数据 2PL 对数似然 {ll_2pl:.1f}（Rasch {ll_rasch:.1f}，2PL−Rasch={ll_gap:.1f}）；"
            if ll_2pl == ll_2pl else
            f"Rasch 对数似然 {ll_rasch:.1f}；2PL 对照拟合失败。"
        )
        stability = "" if n_resp >= _MIN_RESP_RASCH else f"⚠ 被试仅 {n_resp}（<~{_MIN_RESP_RASCH}），估计偏不稳。"
        msg = (
            f"{entry.method} 完成：{k} 个二值题项 × {n_resp} 个被试（girth MML，共享区分度 a={shared_a:.3f}）。"
            f"难度 b 均值 {np.mean(diff_r):.3f}（范围 {np.min(diff_r):.2f}–{np.max(diff_r):.2f}）；"
            f"能力 θ(EAP) 均值 {np.mean(theta):.2f}、SD {estimates['theta_sd']:.2f}；"
            f"被试分离信度 ≈ {rel_txt}。{cmp_txt}"
        )
        if stability:
            msg += " " + stability
        if ll_2pl == ll_2pl and ll_gap > 2.0 * k:
            msg += " ⚠ 2PL 对数似然明显更高，提示各题区分度并不相等，Rasch 的等区分度假设可能不成立（改看 2PL）。"
        if extreme_b:
            msg += f" ⚠ 极端难度题项（|b|>3）：{', '.join(map(str, extreme_b))}。"
        msg += (
            " ⚠ Rasch(1PL) 假设各题区分度相等(仅难度变化)——比 2PL 更严格，故 ICC 为平行不交叉曲线；"
            "等区分度成立时 Rasch 享有特定客观性(specific objectivity)与等距量表解读，"
            "但若各题斜率实际不同则该解读失真(用 2PL 对数似然对照 informal 判断)；"
            "同样要求单维 + 局部独立 + 0/1 二值题项；分离信度与 θ 以 N(0,1) 先验为尺度锚定。"
        )
        summary.append(msg)
        code += [
            "import girth, numpy as np  # Rasch (1PL) IRT — marginal MML, equal discrimination",
            f"X = df[{items!r}].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(float)  # respondents x items",
            "res = girth.rasch_mml(X.T)  # girth wants items x persons -> TRANSPOSE",
            "b = res['Difficulty']; a = res['Discrimination']  # one shared a, per-item b",
            "theta = girth.ability_eap(X.T, b, a)  # EAP person abilities",
            "ll_2pl = ... # also fit girth.twopl_mml(X.T) to compare equal-discrimination",
        ]
    except Exception as err:
        summary.append(f"Rasch IRT 失败：{err}")
