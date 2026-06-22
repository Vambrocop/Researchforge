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

# Polytomous (GRM / PCM): each extra category adds a free threshold per item, so
# stable category-boundary estimation needs noticeably more respondents than the
# dichotomous models, plus enough responses per category.
_MIN_RESP_POLY = 150
_MIN_POLY_CATS = 3  # an ordinal item needs >=3 ordered categories to be polytomous


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


# ---------------------------------------------------------------------------
# Polytomous (GRM / PCM) helpers — local to this family
# ---------------------------------------------------------------------------

def _resolve_poly_items(ctx: Ctx) -> list[str]:
    """Pick the ordinal/polytomous item columns for a GRM/PCM model.

    Priority: config ``items`` then ``columns`` (validated against the frame);
    otherwise the auto default of every ``count`` column (a Likert 0..K integer
    item profiles as ``count`` — non-negative integers, non-unique) excluding the
    panel unit/time columns. We do NOT include ``binary`` by default: a binary
    column has only 2 categories, and polytomous IRT needs >=3 (caught later by
    the per-item category check, which yields an honest skip/note).
    """
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    forced = cfg.get("items") or cfg.get("columns")
    if forced:
        return [c for c in forced if c in df.columns]
    excl = {fp.unit_col, fp.time_col}
    return [c.name for c in fp.columns if c.kind == "count" and c.name not in excl]


def _poly_item_matrix(ctx: Ctx, items: list[str]):
    """Return ((respondents x items) INT matrix, n_categories) or (None, reason).

    Coerces to numeric, drops rows with any missing item, recodes each retained
    value to consecutive integers 0..K-1 (girth's polytomous estimators index
    their likelihood tables by response value, so categories must be 0-based,
    integer and contiguous across the WHOLE matrix — like the dichotomous path
    that returns INT to avoid girth's "arrays used as indices must be of integer
    type" IndexError). Requires >= _MIN_POLY_CATS distinct categories overall.

    girth treats the response set as global (max+1 categories shared by all
    items); recoding the pooled unique values keeps every item on the same
    0..K-1 scale even if a particular item never uses an extreme category.
    """
    import numpy as np
    import pandas as pd

    sub = ctx.df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] == 0:
        return None, "题项含缺失剔除后无数据"
    X = sub.to_numpy(dtype=float)
    # all polytomous responses must be (near-)integer ordinal codes
    if not np.allclose(X, np.round(X)):
        return None, "题项不是整数序数编码（GRM/PCM 需 0..K 有序整数）"
    Xr = np.round(X).astype(int)
    uniq = np.unique(Xr)
    if uniq.shape[0] < _MIN_POLY_CATS:
        return None, f"题项类别数不足（仅 {uniq.shape[0]} 类，需 ≥{_MIN_POLY_CATS}；二值数据请用 irt_2pl/irt_rasch）"
    # recode pooled unique values to contiguous 0..K-1
    remap = {v: i for i, v in enumerate(uniq.tolist())}
    Xc = np.vectorize(remap.get)(Xr).astype(int)
    return Xc, int(uniq.shape[0])


def _poly_thresholds(result):
    """Pull (discrimination_array, thresholds_2d) out of a girth GRM/PCM result.

    girth's polytomous MML estimators return a dict whose ``Difficulty`` is an
    (n_items x n_thresholds) array of ordered category boundaries, and
    ``Discrimination`` an (n_items,) array (PCM is Rasch-family so its values are
    ~equal / fixed; girth may also omit it -> broadcast 1.0). A defensive tuple
    path mirrors ``_extract_params``.
    """
    import numpy as np

    if isinstance(result, dict):
        disc = result.get("Discrimination")
        thr = result.get("Difficulty")
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        disc, thr = result[0], result[1]
    else:
        raise ValueError(f"unrecognised girth polytomous result type: {type(result)!r}")
    thr = np.atleast_2d(np.asarray(thr, dtype=float))  # n_items x n_thresholds
    n_items = thr.shape[0]
    if disc is None:
        disc = np.ones(n_items, dtype=float)
    else:
        disc = np.asarray(disc, dtype=float)
        if disc.ndim == 0:
            disc = np.full(n_items, float(disc))
        else:
            disc = np.atleast_1d(disc)
    return disc, thr


def _poly_abilities(X_ip, disc, thr, n_cats):
    """Person abilities for a polytomous fit; X_ip is items x persons (girth).

    girth has no public closed-form polytomous EAP scorer that matches every
    build, so we score abilities by EAP on a fixed N(0,1) quadrature grid using
    the fitted GRM/PCM category-response likelihood (same model family girth
    fitted). This keeps theta on the standard-normal metric the item params were
    estimated against. Returns a 1-D array of length n_persons.
    """
    import numpy as np

    nodes = np.linspace(-4.0, 4.0, 61)  # quadrature grid for theta
    prior = np.exp(-0.5 * nodes ** 2)
    prior /= prior.sum()
    n_items, n_persons = X_ip.shape
    # P(category=k | theta) per item: GRM cumulative-logit boundaries.
    # P(X>=k) = 1/(1+exp(-a*(theta - thr_k))); category prob = adjacent difference.
    # PCM's equal-a special case is a sub-model, so the same scorer applies.
    # cum[item][k] over the grid: shape n_items x (n_cats-1) x n_nodes
    log_post = np.tile(np.log(prior)[None, :], (n_persons, 1))  # persons x nodes
    for j in range(n_items):
        a = disc[j]
        b = thr[j]  # length n_cats-1 ordered thresholds
        cum = 1.0 / (1.0 + np.exp(-a * (nodes[None, :] - b[:, None])))  # (n_cats-1) x nodes
        # boundaries P(X>=k): prepend P(X>=0)=1, append P(X>=n_cats)=0
        upper = np.vstack([np.ones((1, nodes.size)), cum])            # k=0..n_cats-1 lower edge
        lower = np.vstack([cum, np.zeros((1, nodes.size))])           # k=0..n_cats-1 upper edge
        cat_p = np.clip(upper - lower, 1e-9, 1.0)                     # n_cats x nodes
        resp = X_ip[j]                                                # length n_persons
        log_post += np.log(cat_p[resp, :])                           # persons x nodes
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    theta = post @ nodes
    return np.asarray(theta, dtype=float)


def _plot_thresholds(d, files, items, disc, thr, title: str, fname: str) -> None:
    """Category-boundary / step-difficulty plot: thresholds per item on theta."""
    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = len(items)
        n_thr = thr.shape[1]
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.42 * k + 1.2)))
        cmap = plt.get_cmap("viridis")
        ypos = np.arange(k)
        for t in range(n_thr):
            ax.scatter(
                thr[:, t], ypos, s=42, color=cmap(t / max(1, n_thr - 1)),
                label=f"threshold {t + 1}", zorder=3,
            )
        for j in range(k):
            ax.plot(thr[j], [ypos[j]] * n_thr, color="grey", lw=0.8, zorder=1)
        ax.axvline(0.0, color="grey", ls=":", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([str(x) for x in items], fontsize=7)
        ax.set_xlabel("Category boundary on ability (theta)")
        ax.set_title(title)
        if n_thr <= 8:
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(d / fname, dpi=150)
        plt.close(fig)
        files.append(fname)
    except Exception:
        pass


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
        if ll_2pl == ll_2pl:
            # informational ONLY — this is a PLUG-IN (EAP-θ) log-lik gap, NOT a nested LR
            # statistic; it is uncalibrated and swings both signs even on genuine Rasch data
            # (~27% false-alarm at a fixed threshold), so we do NOT issue a "Rasch fails" verdict.
            msg += (f" 2PL−Rasch 对数似然差 = {ll_gap:.1f}（plug-in、非 LR 检验，仅供参考；"
                    "判定等区分度是否成立请看 2PL 各题 a 的离散程度，而非此差值）。")
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


# ===========================================================================
# 3. Graded Response Model — Samejima (girth.grm_mml), ordinal polytomous
# ===========================================================================

@register("irt_grm")
def _branch_irt_grm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    items = _resolve_poly_items(ctx)
    if len(items) < _MIN_ITEMS:
        summary.append(f"GRM 跳过：需要 ≥{_MIN_ITEMS} 个有序多分(Likert 0..K)题项，当前不足。")
        return

    X, info = _poly_item_matrix(ctx, items)
    if X is None:
        summary.append(f"GRM 跳过：{info}。")
        return
    n_cats = int(info)
    n_resp, k = X.shape
    if n_resp < 20:
        summary.append("GRM 跳过：被试过少（<20），MML 无法收敛到可信解。")
        return

    try:
        import girth

        # girth wants items x persons, integer 0..K-1 -> transpose our matrix.
        X_ip = X.T.astype(int)
        res = girth.grm_mml(X_ip)
        disc, thr = _poly_thresholds(res)  # disc:(k,), thr:(k, n_cats-1)
        if disc.shape[0] != k or thr.shape[0] != k:
            summary.append(
                f"GRM 失败：girth 返回参数行数({disc.shape[0]}/{thr.shape[0]})与题项数({k})不符。"
            )
            return

        # GRM requires the category thresholds of each item to be ORDERED
        # (monotone cumulative boundaries); flag any item that violates it.
        unordered = [
            items[j] for j in range(k)
            if not np.all(np.diff(thr[j][~np.isnan(thr[j])]) >= 0)
        ]

        theta = _poly_abilities(X_ip, disc, thr, n_cats)

        # --- item table: discrimination + one column per ordered threshold ---
        n_thr = thr.shape[1]
        item_cols = {"item": items, "discrimination_a": np.round(disc, 4)}
        for t in range(n_thr):
            item_cols[f"threshold_{t + 1}"] = np.round(thr[:, t], 4)
        item_df = pd.DataFrame(item_cols)
        item_df.to_csv(d / "irt_grm_item_params.csv", index=False, encoding="utf-8")
        files.append("irt_grm_item_params.csv")

        ability_df = pd.DataFrame(
            {"respondent": np.arange(n_resp), "raw_score": X.sum(axis=1).astype(int),
             "theta_eap": np.round(theta, 4)}
        )
        ability_df.to_csv(d / "irt_grm_abilities.csv", index=False, encoding="utf-8")
        files.append("irt_grm_abilities.csv")

        _plot_thresholds(
            d, files, items, disc, thr,
            "GRM Category Boundaries (ordered thresholds)", "irt_grm_thresholds.png",
        )

        low_disc = [items[i] for i in range(k) if disc[i] < 0.4]

        estimates["n_items"] = float(k)
        estimates["n_respondents"] = float(n_resp)
        estimates["n_categories"] = float(n_cats)
        estimates["mean_discrimination"] = round(float(np.mean(disc)), 4)
        estimates["min_discrimination"] = round(float(np.min(disc)), 4)
        estimates["n_unordered_threshold_items"] = float(len(unordered))
        estimates["theta_mean"] = round(float(np.mean(theta)), 4)
        estimates["theta_sd"] = round(float(np.std(theta, ddof=1)), 4) if n_resp > 1 else 0.0

        stability = "" if n_resp >= _MIN_RESP_POLY else f"⚠ 被试仅 {n_resp}（<~{_MIN_RESP_POLY}），多分阈值估计偏不稳。"
        msg = (
            f"{entry.method} 完成：{k} 个有序多分题项（{n_cats} 类）× {n_resp} 个被试（girth grm_mml MML）。"
            f"区分度 a 均值 {np.mean(disc):.3f}（范围 {np.min(disc):.2f}–{np.max(disc):.2f}）；"
            f"每题 {n_thr} 个有序类别阈值；能力 θ(EAP) 均值 {np.mean(theta):.2f}、SD {estimates['theta_sd']:.2f}。"
        )
        if stability:
            msg += " " + stability
        if low_disc:
            msg += f" ⚠ 低区分度题项（a<0.4）：{', '.join(map(str, low_disc))}。"
        if unordered:
            msg += f" ⚠ 类别阈值非单调题项（边界未严格递增，估计可疑）：{', '.join(map(str, unordered))}。"
        msg += (
            " ⚠ GRM(Samejima 分级反应模型)用于有序多分(Likert)作答：每题一个区分度 a + K 个有序类别边界(阈值)，"
            "P(X≥k|θ)=1/(1+exp(-a(θ-阈值_k)))，相邻累积差得各类别概率；阈值必须严格递增(否则模型设定可疑，见上 ⚠ 标记)；"
            "假设单维(单一潜在特质)+局部独立(控制 θ 后题项作答互不相关)；每个类别需足够作答样本，否则极端阈值不稳；"
            "θ 以 N(0,1) 先验为尺度锚定、事后 EAP 评分。"
        )
        summary.append(msg)
        code += [
            "import girth, numpy as np  # GRM (graded response) — polytomous MML",
            f"X = df[{items!r}].apply(pd.to_numeric, errors='coerce').dropna()",
            "X = X.round().astype(int).to_numpy()  # recode to 0..K-1 ordinal codes",
            "res = girth.grm_mml(X.T)  # girth wants items x persons (INT) -> TRANSPOSE",
            "a = res['Discrimination']; thr = res['Difficulty']  # per-item a + ordered thresholds",
            "# P(X>=k|theta)=1/(1+exp(-a*(theta-thr_k))); category prob = adjacent difference",
        ]
    except Exception as err:
        summary.append(f"GRM 失败：{err}")


# ===========================================================================
# 4. Partial Credit Model — Masters (girth.pcm_mml), Rasch-family polytomous
# ===========================================================================

@register("irt_pcm")
def _branch_irt_pcm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    items = _resolve_poly_items(ctx)
    if len(items) < _MIN_ITEMS:
        summary.append(f"PCM 跳过：需要 ≥{_MIN_ITEMS} 个有序多分(partial-credit 0..K)题项，当前不足。")
        return

    X, info = _poly_item_matrix(ctx, items)
    if X is None:
        summary.append(f"PCM 跳过：{info}。")
        return
    n_cats = int(info)
    n_resp, k = X.shape
    if n_resp < 20:
        summary.append("PCM 跳过：被试过少（<20），MML 无法收敛到可信解。")
        return

    try:
        import girth

        X_ip = X.T.astype(int)  # items x persons, INT 0..K-1
        res = girth.pcm_mml(X_ip)
        disc, thr = _poly_thresholds(res)  # PCM: disc ~ equal (Rasch family); thr = step difficulties
        if thr.shape[0] != k:
            summary.append(f"PCM 失败：girth 返回步难度行数({thr.shape[0]})与题项数({k})不符。")
            return
        disc_mean = float(np.mean(disc))
        disc_sd = float(np.std(disc, ddof=1)) if k > 1 else 0.0

        theta = _poly_abilities(X_ip, disc, thr, n_cats)

        # --- step-difficulty table (one column per step boundary) ---
        n_thr = thr.shape[1]
        # girth's pcm_mml estimates a FREE per-item discrimination (it is GPCM, not the
        # equal-a Masters PCM) — report each item's own a, not a single "shared" value.
        item_cols = {"item": items, "discrimination_a": np.round(disc, 4)}
        for t in range(n_thr):
            item_cols[f"step_{t + 1}"] = np.round(thr[:, t], 4)
        item_df = pd.DataFrame(item_cols)
        item_df.to_csv(d / "irt_pcm_item_params.csv", index=False, encoding="utf-8")
        files.append("irt_pcm_item_params.csv")

        ability_df = pd.DataFrame(
            {"respondent": np.arange(n_resp), "raw_score": X.sum(axis=1).astype(int),
             "theta_eap": np.round(theta, 4)}
        )
        ability_df.to_csv(d / "irt_pcm_abilities.csv", index=False, encoding="utf-8")
        files.append("irt_pcm_abilities.csv")

        _plot_thresholds(
            d, files, items, disc, thr,
            "GPCM Step Difficulties", "irt_pcm_steps.png",
        )

        estimates["n_items"] = float(k)
        estimates["n_respondents"] = float(n_resp)
        estimates["n_categories"] = float(n_cats)
        estimates["discrimination_mean"] = round(disc_mean, 4)
        estimates["discrimination_sd"] = round(disc_sd, 4)
        estimates["theta_mean"] = round(float(np.mean(theta)), 4)
        estimates["theta_sd"] = round(float(np.std(theta, ddof=1)), 4) if n_resp > 1 else 0.0

        stability = "" if n_resp >= _MIN_RESP_POLY else f"⚠ 被试仅 {n_resp}（<~{_MIN_RESP_POLY}），步难度估计偏不稳。"
        msg = (
            f"{entry.method} 完成：{k} 个有序多分题项（{n_cats} 类）× {n_resp} 个被试"
            f"（girth pcm_mml MML）；各题区分度 a 均值 {disc_mean:.3f}、SD {disc_sd:.3f}（每题见 CSV）。"
            f"每题 {n_thr} 个步难度(step difficulties)；能力 θ(EAP) 均值 {np.mean(theta):.2f}、"
            f"SD {estimates['theta_sd']:.2f}。"
        )
        if stability:
            msg += " " + stability
        msg += (
            " ⚠ 注意：girth 的 pcm_mml 实为**广义部分计分模型(GPCM)**——每题区分度【自由估计】"
            "（非经典 Masters PCM 的等区分度约束）；上表 discrimination_a 即各题估计斜率(SD 反映其离散)。"
            "步难度=相邻类别等概率处的能力水平(可非单调，与 GRM 的有序累积阈值含义不同)；"
            "若需严格等区分度的部分计分模型，需另行约束(girth 此函数不提供)；"
            "假设单维 + 局部独立 + 有序多分作答；θ 以 N(0,1) 先验为尺度锚定、事后 EAP 评分(SD 因 EAP 向先验收缩而偏小)。"
        )
        summary.append(msg)
        code += [
            "import girth, numpy as np  # PCM (partial credit) — Rasch-family polytomous MML",
            f"X = df[{items!r}].apply(pd.to_numeric, errors='coerce').dropna()",
            "X = X.round().astype(int).to_numpy()  # recode to 0..K-1 ordinal codes",
            "res = girth.pcm_mml(X.T)  # girth wants items x persons (INT) -> TRANSPOSE",
            "step = res['Difficulty']  # per-item step difficulties (equal discrimination)",
            "# compare girth.grm_mml(X.T)['Discrimination'] spread to check equal-a assumption",
        ]
    except Exception as err:
        summary.append(f"PCM 失败：{err}")


# ===========================================================================
# 5. Differential Item Functioning (DIF) — Mantel-Haenszel + logistic
# ===========================================================================

def _resolve_dif_group(ctx: Ctx, item_cols: list[str] | None = None) -> str | None:
    """Resolve the 2-level grouping (focal vs reference) column for DIF.

    Config ``group`` wins; else the lowest-cardinality binary/categorical column
    with exactly 2 levels (never a high-cardinality id), mirroring the bayesian
    family's ``_two_level_group`` idiom. When auto-defaulting we EXCLUDE the
    resolved item columns so the group resolver never silently co-opts a binary
    test item as the grouping variable — if the only 2-level columns are items,
    there is no genuine group and we honestly return None (the branch then skips).
    """
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    excl = {fp.unit_col, fp.time_col} | set(item_cols or [])
    chosen = cfg.get("group")
    if chosen and chosen in df.columns:
        return chosen
    cands = [
        c.name for c in fp.columns
        if c.kind in {"binary", "categorical"} and c.name not in excl
    ]
    cands = [c for c in cands if int(df[c].nunique(dropna=True)) == 2]
    cands.sort(key=lambda name: int(df[name].nunique(dropna=True)))
    return cands[0] if cands else None


def _mh_dif_item(item_resp, grp, score):
    """Mantel-Haenszel DIF for one binary item, stratified by total score.

    item_resp: 0/1 responses; grp: 0(reference)/1(focal); score: matching total.
    Returns (mh_or, mh_chi2, p_value, ets_class) with the ETS A/B/C delta-MH
    classification. The MH common odds ratio pools 2x2 (group x correct) tables
    across score strata; the MH chi-square (with continuity correction) tests
    OR=1 (no uniform DIF). ETS delta = -2.35*ln(OR_MH): |Δ|<1 -> A (negligible),
    1<=|Δ|<1.5 (or not significant) -> B (moderate), |Δ|>=1.5 & significant -> C.
    """
    import numpy as np
    from scipy import stats

    num = den = 0.0          # sum a*d/n and b*c/n for the common OR
    s_a = s_Ea = s_Va = 0.0  # MH chi-square accumulators
    for s in np.unique(score):
        m = score == s
        if m.sum() < 2:
            continue
        ir = item_resp[m]
        g = grp[m]
        # 2x2: rows = focal/reference, cols = correct(1)/incorrect(0)
        a = float(np.sum((g == 1) & (ir == 1)))  # focal correct
        b = float(np.sum((g == 1) & (ir == 0)))  # focal incorrect
        c = float(np.sum((g == 0) & (ir == 1)))  # ref correct
        d_ = float(np.sum((g == 0) & (ir == 0)))  # ref incorrect
        n = a + b + c + d_
        if n < 2:
            continue
        r1 = a + b  # focal total
        r2 = c + d_  # ref total
        c1 = a + c  # correct total
        if r1 == 0 or r2 == 0 or c1 == 0 or c1 == n:
            continue  # stratum carries no DIF information (degenerate margin)
        num += a * d_ / n
        den += b * c / n
        s_a += a
        s_Ea += r1 * c1 / n
        s_Va += (r1 * r2 * c1 * (n - c1)) / (n * n * (n - 1)) if n > 1 else 0.0

    if den <= 0 or num <= 0:
        mh_or = float("nan")
    else:
        mh_or = num / den
    if s_Va > 0:
        chi2 = (abs(s_a - s_Ea) - 0.5) ** 2 / s_Va  # continuity-corrected
        chi2 = max(chi2, 0.0)
        p = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = float("nan")
        p = float("nan")

    # ETS delta-MH classification
    if mh_or == mh_or and mh_or > 0:
        delta = -2.35 * np.log(mh_or)
    else:
        delta = float("nan")
    sig = (p == p) and (p < 0.05)
    if delta != delta:
        ets = "NA"
    elif abs(delta) < 1.0 or not sig:
        ets = "A" if abs(delta) < 1.0 else "B"
    elif abs(delta) < 1.5:
        ets = "B"
    else:
        ets = "C"
    return mh_or, chi2, p, ets, (delta if delta == delta else float("nan"))


def _logistic_dif_item(item_resp, grp, score):
    """Logistic-regression DIF for one binary item: item ~ score + group.

    Returns the group coefficient's p-value (uniform DIF test). A group*score
    interaction would detect NON-uniform DIF; we report the uniform test here and
    disclose the non-uniform extension. Returns nan if it fails to converge.
    """
    import numpy as np

    try:
        import statsmodels.api as sm

        z = (score - score.mean()) / (score.std() + 1e-9)
        Xd = np.column_stack([np.ones_like(z, dtype=float), z, grp.astype(float)])
        model = sm.Logit(item_resp.astype(float), Xd)
        fit = model.fit(disp=0, maxiter=100)
        return float(fit.pvalues[2])  # group coefficient p-value
    except Exception:
        return float("nan")


@register("dif_detection")
def _branch_dif_detection(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    items = _resolve_irt_items(ctx)  # binary 0/1 items (reuse the dichotomous resolver)
    if len(items) < _MIN_ITEMS:
        summary.append(f"DIF 检测 跳过：需要 ≥{_MIN_ITEMS} 个二值(0/1)题项，当前不足。")
        return

    group_col = _resolve_dif_group(ctx, item_cols=items)
    if group_col is None:
        summary.append("DIF 检测 跳过：未找到 2 水平分组列（设 config group 指定参照组/焦点组）。")
        return
    # drop the group column from items if it was swept in as a binary
    items = [c for c in items if c != group_col]
    if len(items) < _MIN_ITEMS:
        summary.append(f"DIF 检测 跳过：剔除分组列后二值题项不足（需 ≥{_MIN_ITEMS}）。")
        return

    sub = df[items + [group_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] == 0:
        # group may be categorical text -> coerce items numeric, keep group as labels
        sub = df[items + [group_col]].dropna()
        for c in items:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna()
    if sub.shape[0] == 0:
        summary.append("DIF 检测 跳过：题项含缺失剔除后无数据。")
        return

    Xitems = sub[items].to_numpy(dtype=float)
    uniq_items = set(np.unique(Xitems).tolist())
    if not uniq_items <= {0.0, 1.0}:
        summary.append("DIF 检测 跳过：题项不是 0/1 二值（多分题项请先二分化）。")
        return
    Xitems = Xitems.astype(int)

    glabels = sub[group_col]
    levels = sorted(pd.unique(glabels).tolist(), key=lambda v: (isinstance(v, str), v))
    if len(levels) != 2:
        summary.append(f"DIF 检测 跳过：分组列 {group_col} 不是恰好 2 个水平（DIF 比较参照组 vs 焦点组）。")
        return
    # reference = levels[0], focal = levels[1]
    grp = (glabels.to_numpy() == levels[1]).astype(int)
    n_resp, k = Xitems.shape
    if n_resp < 40 or grp.sum() < 10 or (n_resp - grp.sum()) < 10:
        summary.append(
            "DIF 检测 跳过：样本或某组过少（需总 ≥40、每组 ≥10），分层匹配无足够每层样本。"
        )
        return

    try:
        # Match on total score. For each item, compute the REST score (total minus
        # that item) as the matching variable — the standard MH-DIF practice that
        # avoids the studied item contaminating its own matching strata.
        total = Xitems.sum(axis=1)
        rows = []
        for j in range(k):
            rest = total - Xitems[:, j]  # rest score, the matching variable
            mh_or, chi2, p_mh, ets, delta = _mh_dif_item(Xitems[:, j], grp, rest)
            p_lr = _logistic_dif_item(Xitems[:, j], grp, rest)
            rows.append({
                "item": items[j],
                "p_correct_reference": round(float(Xitems[grp == 0, j].mean()), 4),
                "p_correct_focal": round(float(Xitems[grp == 1, j].mean()), 4),
                "mh_odds_ratio": round(mh_or, 4) if mh_or == mh_or else float("nan"),
                "mh_delta": round(delta, 4) if delta == delta else float("nan"),
                "mh_chi2": round(chi2, 4) if chi2 == chi2 else float("nan"),
                "mh_p_value": round(p_mh, 5) if p_mh == p_mh else float("nan"),
                "logistic_group_p": round(p_lr, 5) if p_lr == p_lr else float("nan"),
                "ets_dif_class": ets,
            })
        dif_df = pd.DataFrame(rows)
        dif_df.to_csv(d / "dif_detection.csv", index=False, encoding="utf-8")
        files.append("dif_detection.csv")

        # flagged items: significant MH (p<0.05) OR ETS class C
        flagged = dif_df[
            (dif_df["mh_p_value"] < 0.05) | (dif_df["ets_dif_class"] == "C")
        ]["item"].tolist()
        flagged_c = dif_df[dif_df["ets_dif_class"] == "C"]["item"].tolist()

        _plot_dif(d, files, dif_df, group_col, str(levels[0]), str(levels[1]))

        estimates["n_items"] = float(k)
        estimates["n_respondents"] = float(n_resp)
        estimates["n_reference"] = float(int((grp == 0).sum()))
        estimates["n_focal"] = float(int((grp == 1).sum()))
        estimates["n_flagged_dif"] = float(len(flagged))
        estimates["n_class_c_dif"] = float(len(flagged_c))
        valid_or = dif_df["mh_odds_ratio"].dropna()
        estimates["max_abs_log_or"] = (
            round(float(np.max(np.abs(np.log(valid_or[valid_or > 0])))), 4)
            if (valid_or > 0).any() else -1.0
        )

        flagged_txt = (
            f"被标记 DIF 的题项：{', '.join(map(str, flagged))}。"
            if flagged else "未检出显著 DIF 题项。"
        )
        msg = (
            f"{entry.method} 完成：{k} 个二值题项，按 {group_col} 分参照组「{levels[0]}」"
            f"(n={int((grp == 0).sum())}) vs 焦点组「{levels[1]}」(n={int((grp == 1).sum())})，"
            f"以总分(rest score)匹配能力。Mantel-Haenszel 共同比值比 + χ² + ETS A/B/C 分级；"
            f"逻辑回归(item~score+group)给一致性 DIF 的 group 系数 p。{flagged_txt}"
        )
        if flagged_c:
            msg += f" ⚠ 大 DIF(ETS C 级，公平性需重点关注)：{', '.join(map(str, flagged_c))}。"
        msg += (
            " ⚠ DIF=在匹配能力后题项跨组功能差异(测量偏差)，以总分作为能力匹配变量(rest score 防自污染)；"
            "Mantel-Haenszel 检测一致性(uniform)DIF——各能力层比值比同向；逻辑回归加 group×score 交互可检非一致性"
            "(non-uniform)DIF(本实现报一致性 group 主效应)；显著 DIF ⇒ 该题在两组测的不是同一构念(公平性问题，"
            "需复核题面/删题)；ETS 分级 A 可忽略/B 中等/C 大；需每组每分层有足够样本，否则分层 χ² 不稳。"
        )
        summary.append(msg)
        code += [
            "import numpy as np  # Mantel-Haenszel DIF — stratify on rest score",
            f"items = {items!r}; total = df[items].sum(axis=1)",
            "for j,item in enumerate(items):",
            "    rest = total - df[item]  # matching variable (rest score)",
            "    # pool 2x2 (group x correct) tables across rest-score strata:",
            "    # OR_MH = sum(a*d/n)/sum(b*c/n); MH chi2 tests OR=1; ETS delta=-2.35*ln(OR)",
        ]
    except Exception as err:
        summary.append(f"DIF 检测 失败：{err}")


def _plot_dif(d, files, dif_df, group_col, ref_label, focal_label) -> None:
    """Bar plot of MH delta per item, coloured by ETS DIF class."""
    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        deltas = dif_df["mh_delta"].to_numpy(dtype=float)
        items = dif_df["item"].astype(str).tolist()
        classes = dif_df["ets_dif_class"].tolist()
        cmap = {"A": "#4caf50", "B": "#ff9800", "C": "#e53935", "NA": "#9e9e9e"}
        colors = [cmap.get(c, "#9e9e9e") for c in classes]
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.42 * len(items) + 1.2)))
        ypos = np.arange(len(items))
        plot_d = np.nan_to_num(deltas, nan=0.0)
        ax.barh(ypos, plot_d, color=colors, zorder=3)
        ax.axvline(0.0, color="grey", lw=0.8)
        ax.axvline(1.0, color="grey", ls=":", lw=0.7)
        ax.axvline(-1.0, color="grey", ls=":", lw=0.7)
        ax.axvline(1.5, color="red", ls=":", lw=0.7)
        ax.axvline(-1.5, color="red", ls=":", lw=0.7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(items, fontsize=7)
        ax.set_xlabel("MH delta-DIF  (-=favours focal, +=favours reference)")
        ax.set_title(f"DIF by {group_col}: {ref_label} (ref) vs {focal_label} (focal)")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=cmap[c])
            for c in ["A", "B", "C"]
        ]
        ax.legend(handles, ["A negligible", "B moderate", "C large"], fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(d / "dif_detection.png", dpi=150)
        plt.close(fig)
        files.append("dif_detection.png")
    except Exception:
        pass
