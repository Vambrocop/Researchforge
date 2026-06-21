"""Branch handlers for the Bayesian conjugate-inference family.

Closed-form conjugate posteriors only — NO MCMC / PyMC. Pure Python (numpy/scipy).

Conjugate pairs used:
  * bayesian_ab_test      — Beta-Binomial (Beta prior, Binomial likelihood)
  * bayesian_proportion   — Beta-Binomial (single proportion)
  * bayesian_poisson_rate — Gamma-Poisson (Gamma prior, Poisson likelihood)

Engine conventions (CLAUDE.md「引擎约定」): each handler is
``@register("<id>") def _branch_<id>(ctx)``; unpack ctx into df/fp/entry/cfg/d +
files/summary/estimates/code and MUTATE them. Products: CSV + PNG (matplotlib Agg,
ENGLISH plot labels, best-effort), float estimates, a Chinese summary with ⚠
disclosures, and an honest skip + return when the data is unusable.

⚠ Bayesian disclosures live in each summary: priors are always stated; a posterior
probability (e.g. P(B>A), P(θ>θ0)) is NOT a frequentist p-value; a credible interval
is a direct probability statement about the parameter (≠ a confidence interval);
priors matter at small n. Any draw-based quantity uses a FIXED seed (disclosed).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

_SEED = 20240617  # fixed seed for all draw-based quantities (disclosed in summaries)
_N_DRAWS = 100_000


# --------------------------------------------------------------------------- #
# small shared helpers (live here — Bayesian-family-specific, not cross-family) #
# --------------------------------------------------------------------------- #
def _beta_credible_interval(a: float, b: float, level: float = 0.95):
    """Equal-tail credible interval for a Beta(a, b) posterior."""
    from scipy import stats

    lo = (1.0 - level) / 2.0
    return float(stats.beta.ppf(lo, a, b)), float(stats.beta.ppf(1.0 - lo, a, b))


def _beta_hpd(a: float, b: float, level: float = 0.95):
    """Highest-posterior-density interval for a Beta(a, b) posterior.

    Minimises interval width over the lower-tail mass by a 1-D scan; for a unimodal
    Beta this is the HPD. Falls back to equal-tail for U-shaped/degenerate Beta
    (a<=1 and b<=1), where the density has no interior mode.
    """
    from scipy import stats

    if a <= 1.0 or b <= 1.0:  # no interior mode -> HPD undefined / equal-tail
        return _beta_credible_interval(a, b, level)

    import numpy as np

    lows = np.linspace(1e-6, 1.0 - level - 1e-6, 2000)
    best = None
    for lo_mass in lows:
        lo = stats.beta.ppf(lo_mass, a, b)
        hi = stats.beta.ppf(lo_mass + level, a, b)
        width = hi - lo
        if best is None or width < best[2]:
            best = (lo, hi, width)
    return float(best[0]), float(best[1])


def _prob_b_gt_a(a1: float, b1: float, a2: float, b2: float, rng=None) -> tuple[float, bool]:
    """P(X2 > X1) for independent X1~Beta(a1,b1), X2~Beta(a2,b2).

    Returns (probability, used_draws). When a2 is a positive integer (the usual case:
    posterior a = prior_a + integer successes with an integer prior), uses Evan Miller /
    John Cook's exact closed form:

        P(X2 > X1) = Σ_{i=0}^{a2-1}  B(a1+i, b1+b2)
                                     ─────────────────────────────────
                                     (b2+i) · B(1+i, b2) · B(a1, b1)

    computed in log-space for numerical stability. Otherwise falls back to fixed-seed
    Monte-Carlo draws (used_draws=True, disclosed in the summary).
    """
    import numpy as np
    from scipy.special import betaln

    # exact closed form requires integer a2 and a manageable summation length
    if float(a2).is_integer() and 1 <= a2 <= 100_000:
        n = int(a2)
        i = np.arange(n)
        log_terms = (
            betaln(a1 + i, b1 + b2)
            - np.log(b2 + i)
            - betaln(1.0 + i, b2)
            - betaln(a1, b1)
        )
        total = float(np.sum(np.exp(log_terms)))
        return min(max(total, 0.0), 1.0), False

    # symmetric closed form on a1 if THAT is the integer (P(X2>X1)=1-P(X1>X2))
    if float(a1).is_integer() and 1 <= a1 <= 100_000:
        p_a_gt_b, _ = _prob_b_gt_a(a2, b2, a1, b1)
        return 1.0 - p_a_gt_b, False

    # fallback: fixed-seed Monte-Carlo
    if rng is None:
        rng = np.random.default_rng(_SEED)
    xa = rng.beta(a1, b1, _N_DRAWS)
    xb = rng.beta(a2, b2, _N_DRAWS)
    return float(np.mean(xb > xa)), True


def _resolve_binary_outcome(df, fp, cfg):
    """Pick a binary outcome column (config 'outcome' wins; else first binary col)."""
    excl = {fp.unit_col, fp.time_col}
    bin_cols = [c.name for c in fp.columns if c.kind == "binary" and c.name not in excl]
    chosen = cfg.get("outcome")
    if chosen and chosen in df.columns:
        return chosen, bin_cols
    return (bin_cols[0] if bin_cols else None), bin_cols


def _coerce_binary(series):
    """Map a 2-level (numeric/bool/text) column to 0/1 ints, dropping NaN.

    The larger / 'truthy' level becomes the success (1). Returns (int_array, level_map)
    where level_map describes which original level maps to 1.
    """
    import numpy as np
    import pandas as pd

    s = series.dropna()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).to_numpy(), {True: 1, False: 0}
    uniq = sorted(s.unique().tolist(), key=lambda v: (isinstance(v, str), v))
    if len(uniq) > 2:
        # not strictly binary — treat nonzero as success
        return (s != 0).astype(int).to_numpy(), {"!=0": 1}
    if len(uniq) == 2:
        lo, hi = uniq[0], uniq[1]
        mapped = (s == hi).astype(int).to_numpy()
        return mapped, {hi: 1, lo: 0}
    # constant column
    val = uniq[0] if uniq else 0
    return (np.zeros(len(s), dtype=int) if val == 0 else np.ones(len(s), dtype=int)), {val: 1}


def _two_level_group(df, fp, cfg):
    """Resolve a 2-level grouping column (config 'group' wins).

    Prefers a binary group, else the lowest-cardinality categorical/binary with
    exactly 2 levels, so a high-cardinality id is never picked.
    """
    excl = {fp.unit_col, fp.time_col}
    chosen = cfg.get("group")
    if chosen and chosen in df.columns:
        return chosen
    cands = [
        c.name
        for c in fp.columns
        if c.kind in {"binary", "categorical"} and c.name not in excl
    ]
    cands = [c for c in cands if int(df[c].nunique(dropna=True)) == 2]
    cands.sort(key=lambda name: int(df[name].nunique(dropna=True)))
    return cands[0] if cands else None


# --------------------------------------------------------------------------- #
# 1) Bayesian A/B test — Beta-Binomial conjugate                              #
# --------------------------------------------------------------------------- #
@register("bayesian_ab_test")
def _branch_bayesian_ab_test(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    prior_a = float(cfg.get("prior_a", 1.0))  # default uniform Beta(1,1)
    prior_b = float(cfg.get("prior_b", 1.0))
    level = float(cfg.get("ci", 0.95))

    # Two input modes:
    #  (A) successes/trials count columns via config: successes_a/trials_a/successes_b/trials_b
    #  (B) a binary outcome column + a 2-level group column
    sa = cfg.get("successes_a"); ta = cfg.get("trials_a")
    sb = cfg.get("successes_b"); tb = cfg.get("trials_b")
    counts_mode = all(v is not None for v in (sa, ta, sb, tb))

    if counts_mode:
        k_a, n_a = int(sa), int(ta)
        k_b, n_b = int(sb), int(tb)
        label_a, label_b = "A", "B"
        if n_a <= 0 or n_b <= 0 or k_a > n_a or k_b > n_b or k_a < 0 or k_b < 0:
            summary.append("贝叶斯 A/B 检验失败：successes/trials 配置不合法（需 0≤successes≤trials，trials>0）。")
            return
    else:
        outcome, bin_cols = _resolve_binary_outcome(df, fp, cfg)
        group_col = _two_level_group(df, fp, cfg)
        if outcome is None or group_col is None or outcome == group_col:
            summary.append(
                "贝叶斯 A/B 检验失败：需要一个二值结果列 + 一个 2 水平分组列"
                "（或通过 config 提供 successes_a/trials_a/successes_b/trials_b）。"
            )
            return
        sub = df[[outcome, group_col]].dropna()
        levels = sorted(sub[group_col].unique().tolist(), key=lambda v: (isinstance(v, str), v))
        if len(levels) != 2:
            summary.append(f"贝叶斯 A/B 检验失败：分组列 {group_col} 不是恰好 2 个水平。")
            return
        label_a, label_b = str(levels[0]), str(levels[1])
        ya, _ = _coerce_binary(sub.loc[sub[group_col] == levels[0], outcome])
        yb, _ = _coerce_binary(sub.loc[sub[group_col] == levels[1], outcome])
        k_a, n_a = int(ya.sum()), int(len(ya))
        k_b, n_b = int(yb.sum()), int(len(yb))
        if n_a == 0 or n_b == 0:
            summary.append("贝叶斯 A/B 检验失败：某一组样本量为 0。")
            return

    # Posteriors: Beta(prior_a + successes, prior_b + failures)
    a1, b1 = prior_a + k_a, prior_b + (n_a - k_a)  # arm A
    a2, b2 = prior_a + k_b, prior_b + (n_b - k_b)  # arm B
    mean_a = a1 / (a1 + b1)
    mean_b = a2 / (a2 + b2)
    ci_a = _beta_credible_interval(a1, b1, level)
    ci_b = _beta_credible_interval(a2, b2, level)

    # P(B > A): closed form (Beta inequality sum) when posterior a is integer, else draws
    p_b_gt_a, used_draws = _prob_b_gt_a(a1, b1, a2, b2)

    # Lift (B - A) posterior + expected loss: no simple closed form -> fixed-seed draws
    rng = np.random.default_rng(_SEED)
    draws_a = rng.beta(a1, b1, _N_DRAWS)
    draws_b = rng.beta(a2, b2, _N_DRAWS)
    lift = draws_b - draws_a
    lift_mean = float(np.mean(lift))
    lo = (1.0 - level) / 2.0
    lift_ci = (float(np.quantile(lift, lo)), float(np.quantile(lift, 1.0 - lo)))
    # Decision-theoretic expected loss of choosing each arm (loss = forgone conversion).
    loss_choose_a = float(np.mean(np.maximum(draws_b - draws_a, 0.0)))  # regret if A chosen but B better
    loss_choose_b = float(np.mean(np.maximum(draws_a - draws_b, 0.0)))

    # CSV: per-arm posterior table
    tbl = pd.DataFrame(
        {
            "arm": [label_a, label_b],
            "successes": [k_a, k_b],
            "trials": [n_a, n_b],
            "post_alpha": [a1, a2],
            "post_beta": [b1, b2],
            "post_mean": [mean_a, mean_b],
            "ci_low": [ci_a[0], ci_b[0]],
            "ci_high": [ci_a[1], ci_b[1]],
            "expected_loss": [loss_choose_a, loss_choose_b],
        }
    )
    tbl.to_csv(d / "bayesian_ab_test.csv", index=False, encoding="utf-8")
    files.append("bayesian_ab_test.csv")

    estimates["post_mean_a"] = mean_a
    estimates["post_mean_b"] = mean_b
    estimates["ci_low_a"], estimates["ci_high_a"] = ci_a
    estimates["ci_low_b"], estimates["ci_high_b"] = ci_b
    estimates["prob_b_gt_a"] = p_b_gt_a
    estimates["lift_mean"] = lift_mean
    estimates["lift_ci_low"], estimates["lift_ci_high"] = lift_ci
    estimates["expected_loss_a"] = loss_choose_a
    estimates["expected_loss_b"] = loss_choose_b
    estimates["ci_level"] = level

    # Plot: the two posterior densities
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats

        xs = np.linspace(0.0, 1.0, 500)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, stats.beta.pdf(xs, a1, b1), label=f"{label_a}  Beta({a1:.0f},{b1:.0f})")
        ax.plot(xs, stats.beta.pdf(xs, a2, b2), label=f"{label_b}  Beta({a2:.0f},{b2:.0f})")
        ax.axvline(mean_a, ls="--", lw=0.8, alpha=0.6)
        ax.axvline(mean_b, ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("Conversion rate")
        ax.set_ylabel("Posterior density")
        ax.set_title("Bayesian A/B test — Beta posteriors")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "bayesian_ab_posteriors.png", dpi=150)
        plt.close(fig)
        files.append("bayesian_ab_posteriors.png")
    except Exception:
        pass

    method = "解析闭式" if not used_draws else f"{_N_DRAWS:,} 次后验抽样(seed={_SEED})"
    summary.append(
        f"贝叶斯 A/B 检验完成（Beta-二项共轭）：{label_a} 后验均值 {mean_a:.4f} "
        f"[{ci_a[0]:.4f},{ci_a[1]:.4f}]，{label_b} 后验均值 {mean_b:.4f} "
        f"[{ci_b[0]:.4f},{ci_b[1]:.4f}]。P({label_b}>{label_a})={p_b_gt_a:.4f}（{method}）；"
        f"提升(B−A) 均值 {lift_mean:+.4f}，{int(level*100)}% 可信区间 "
        f"[{lift_ci[0]:+.4f},{lift_ci[1]:+.4f}]；选 {label_a} 的期望损失 {loss_choose_a:.4f}，"
        f"选 {label_b} 的期望损失 {loss_choose_b:.4f}。"
    )
    summary.append(
        f"⚠ 先验：Beta({prior_a:g},{prior_b:g})"
        + ("（默认均匀先验）" if (prior_a == 1.0 and prior_b == 1.0) else "（自定义）")
        + "。⚠ P(B>A) 是后验概率，不是 p 值；可信区间是对参数的直接概率陈述（≠置信区间）。"
        "⚠ 贝叶斯无需多看(peeking)校正，但小样本时先验影响大。⚠ 假定每次试验为 iid 伯努利。"
        "⚠ 提升区间/期望损失为固定种子蒙特卡洛抽样所得。"
    )

    code += [
        "from scipy import stats",
        f"a1, b1 = {prior_a}+k_a, {prior_b}+(n_a-k_a)  # arm A posterior Beta",
        f"a2, b2 = {prior_a}+k_b, {prior_b}+(n_b-k_b)  # arm B posterior Beta",
        "mean_b = a2/(a2+b2); ci_b = stats.beta.ppf([.025,.975], a2, b2)",
        "# P(B>A): closed form via Beta inequality sum, or posterior draws",
    ]


# --------------------------------------------------------------------------- #
# 2) Bayesian single-proportion estimation — Beta-Binomial conjugate          #
# --------------------------------------------------------------------------- #
@register("bayesian_proportion")
def _branch_bayesian_proportion(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy import stats

    # prior: default Beta(1,1) uniform; Jeffreys Beta(0.5,0.5) via config 'prior=jeffreys'
    prior_kind = str(cfg.get("prior", "uniform")).lower()
    if prior_kind in ("jeffreys", "jeffrey"):
        prior_a, prior_b = 0.5, 0.5
    else:
        prior_a, prior_b = 1.0, 1.0
    prior_a = float(cfg.get("prior_a", prior_a))
    prior_b = float(cfg.get("prior_b", prior_b))
    level = float(cfg.get("ci", 0.95))
    ref = float(cfg.get("ref", 0.5))  # reference value for P(θ > ref)
    interval_kind = str(cfg.get("interval", "equal_tail")).lower()  # or "hpd"

    outcome, bin_cols = _resolve_binary_outcome(df, fp, cfg)
    if outcome is None:
        summary.append("贝叶斯比例估计失败：未找到二值结果列（可用 config['outcome'] 指定）。")
        return

    y, _ = _coerce_binary(df[outcome])
    n = int(len(y))
    if n == 0:
        summary.append("贝叶斯比例估计失败：二值结果列在去除缺失后为空。")
        return
    k = int(y.sum())

    # Posterior Beta(prior_a + k, prior_b + n - k)
    a_post = prior_a + k
    b_post = prior_b + (n - k)
    post_mean = a_post / (a_post + b_post)
    post_median = float(stats.beta.median(a_post, b_post))
    # mode defined only for a,b > 1
    if a_post > 1.0 and b_post > 1.0:
        post_mode = (a_post - 1.0) / (a_post + b_post - 2.0)
    else:
        post_mode = float("nan")

    if interval_kind == "hpd":
        ci = _beta_hpd(a_post, b_post, level)
    else:
        ci = _beta_credible_interval(a_post, b_post, level)

    # P(θ > ref) = survival function (NOT 1-cdf: 1-cdf catastrophically cancels to 0.0
    # in the upper tail, e.g. Beta(141,61) at ref=0.95; sf keeps the true ~8.7e-31).
    p_gt_ref = float(stats.beta.sf(ref, a_post, b_post))

    tbl = pd.DataFrame(
        {
            "metric": [
                "successes", "trials", "post_alpha", "post_beta",
                "post_mean", "post_median", "post_mode",
                "ci_low", "ci_high", f"P(theta>{ref:g})",
            ],
            "value": [
                k, n, a_post, b_post,
                post_mean, post_median, post_mode,
                ci[0], ci[1], p_gt_ref,
            ],
        }
    )
    tbl.to_csv(d / "bayesian_proportion.csv", index=False, encoding="utf-8")
    files.append("bayesian_proportion.csv")

    estimates["successes"] = float(k)
    estimates["trials"] = float(n)
    estimates["post_mean"] = post_mean
    estimates["post_median"] = post_median
    estimates["post_mode"] = post_mode
    estimates["ci_low"], estimates["ci_high"] = ci
    estimates["ci_level"] = level
    estimates["prob_gt_ref"] = p_gt_ref
    estimates["ref"] = ref

    # Plot the posterior
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.linspace(0.0, 1.0, 500)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, stats.beta.pdf(xs, a_post, b_post),
                label=f"Posterior Beta({a_post:.1f},{b_post:.1f})")
        ax.axvline(post_mean, ls="--", lw=0.9, color="C1", label=f"mean={post_mean:.3f}")
        ax.axvspan(ci[0], ci[1], alpha=0.15, color="C0", label=f"{int(level*100)}% CrI")
        ax.axvline(ref, ls=":", lw=0.9, color="k", label=f"ref={ref:g}")
        ax.set_xlabel("Proportion (theta)")
        ax.set_ylabel("Posterior density")
        ax.set_title("Bayesian proportion — Beta posterior")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "bayesian_proportion_posterior.png", dpi=150)
        plt.close(fig)
        files.append("bayesian_proportion_posterior.png")
    except Exception:
        pass

    prior_label = "Jeffreys Beta(0.5,0.5)" if (prior_a == 0.5 and prior_b == 0.5) else (
        "均匀 Beta(1,1)" if (prior_a == 1.0 and prior_b == 1.0) else f"Beta({prior_a:g},{prior_b:g})"
    )
    int_label = "最高后验密度(HPD)" if interval_kind == "hpd" else "等尾"
    summary.append(
        f"贝叶斯比例估计完成（Beta-二项共轭）：成功 {k}/{n}，后验 Beta({a_post:g},{b_post:g})；"
        f"后验均值 {post_mean:.4f}、中位数 {post_median:.4f}"
        + (f"、众数 {post_mode:.4f}" if np.isfinite(post_mode) else "（众数未定义：a 或 b ≤ 1）")
        + f"；{int(level*100)}% {int_label}可信区间 [{ci[0]:.4f},{ci[1]:.4f}]；"
        f"P(θ>{ref:g})={p_gt_ref:.4f}。"
    )
    summary.append(
        f"⚠ 先验：{prior_label}。⚠ 可信区间是对 θ 的直接概率陈述（≠置信区间）。"
        "⚠ 小样本时先验敏感（均匀 vs Jeffreys 会移动结果），请同时报告先验。"
    )

    code += [
        "from scipy import stats",
        f"a_post, b_post = {prior_a}+k, {prior_b}+(n-k)  # Beta posterior",
        "post_mean = a_post/(a_post+b_post)",
        "ci = stats.beta.ppf([.025,.975], a_post, b_post)  # equal-tail credible interval",
        f"p_gt_ref = stats.beta.sf({ref}, a_post, b_post)  # survival fn (not 1-cdf: tail cancellation)",
    ]


# --------------------------------------------------------------------------- #
# 3) Bayesian Poisson rate estimation — Gamma-Poisson conjugate               #
# --------------------------------------------------------------------------- #
@register("bayesian_poisson_rate")
def _branch_bayesian_poisson_rate(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd
    from scipy import stats

    # prior Gamma(a0, b0) — default weakly informative Gamma(0.001, 0.001)
    a0 = float(cfg.get("prior_a", cfg.get("prior_shape", 0.001)))
    b0 = float(cfg.get("prior_b", cfg.get("prior_rate", 0.001)))
    level = float(cfg.get("ci", 0.95))

    # outcome: config 'outcome' wins; else first count column; accept name-locked id
    excl = {fp.unit_col, fp.time_col}
    count_cols = [c.name for c in fp.columns if c.kind == "count" and c.name not in excl]
    chosen = cfg.get("outcome")
    outcome = None
    if chosen and chosen in df.columns:
        outcome = chosen
    elif count_cols:
        outcome = count_cols[0]
    else:
        # name-locked fallback: an integer-but-all-unique column profiles as 'id'
        # (CLAUDE.md profiler "id" 陷阱) — accept it if named like a count.
        import re
        for c in fp.columns:
            if c.kind == "id" and re.search(r"count|num|n_|freq|events|incid", c.name.lower()):
                s = df[c.name].dropna()
                if pd.api.types.is_numeric_dtype(s) and bool((s % 1 == 0).all()) and bool((s >= 0).all()):
                    outcome = c.name
                    break

    if outcome is None:
        summary.append("贝叶斯泊松率估计失败：未找到计数列（可用 config['outcome'] 指定）。")
        return

    y_raw = df[outcome].dropna()
    y = y_raw.to_numpy(dtype=float)
    if len(y) == 0 or np.any(y < 0) or not np.all(y == np.round(y)):
        summary.append(f"贝叶斯泊松率估计失败：结果列 {outcome} 不是非负整数计数。")
        return

    # exposure / offset (optional): config 'exposure' column, else 1 per row
    exp_col = cfg.get("exposure")
    if exp_col and exp_col in df.columns:
        exposure = df.loc[y_raw.index, exp_col].fillna(0.0).to_numpy(dtype=float)
        exp_label = exp_col
    else:
        exposure = np.ones(len(y), dtype=float)
        exp_label = "1 per row"
    total_exposure = float(np.sum(exposure))
    if total_exposure <= 0:
        summary.append("贝叶斯泊松率估计失败：曝光量(exposure)合计 ≤ 0。")
        return

    sum_y = float(np.sum(y))
    # Posterior Gamma(a0 + Σy, b0 + Σexposure); mean = shape/rate
    a_post = a0 + sum_y
    b_post = b0 + total_exposure
    post_mean = a_post / b_post
    ci = (float(stats.gamma.ppf((1 - level) / 2, a_post, scale=1.0 / b_post)),
          float(stats.gamma.ppf(1 - (1 - level) / 2, a_post, scale=1.0 / b_post)))
    emp_rate = sum_y / total_exposure

    # overdispersion check: EXPOSURE-AWARE Pearson dispersion. Raw var(y)/mean(y) is wrong
    # when exposure varies (y_i~Poisson(rate·exp_i) is legitimately non-constant-mean even at
    # a constant rate, so raw var/mean false-positives). Pearson: λ̂_i = post_mean·exp_i,
    # dispersion = Σ(y_i−λ̂_i)²/λ̂_i / (n−1); reduces to ~var/mean when exposure is constant.
    n_obs = len(y)
    lam_hat = post_mean * exposure
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = float(np.sum(np.where(lam_hat > 0, (y - lam_hat) ** 2 / lam_hat, 0.0)))
    dispersion = pearson / (n_obs - 1) if n_obs > 1 else float("nan")
    overdispersed = bool(np.isfinite(dispersion) and dispersion > 1.5)

    estimates["sum_y"] = sum_y
    estimates["total_exposure"] = total_exposure
    estimates["post_shape"] = a_post
    estimates["post_rate"] = b_post
    estimates["post_mean_rate"] = post_mean
    estimates["ci_low"], estimates["ci_high"] = ci
    estimates["ci_level"] = level
    estimates["empirical_rate"] = emp_rate
    estimates["dispersion"] = dispersion
    estimates["overdispersed"] = 1.0 if overdispersed else 0.0

    rows = [
        {"metric": "sum_y", "value": sum_y},
        {"metric": "total_exposure", "value": total_exposure},
        {"metric": "post_shape", "value": a_post},
        {"metric": "post_rate", "value": b_post},
        {"metric": "post_mean_rate", "value": post_mean},
        {"metric": "ci_low", "value": ci[0]},
        {"metric": "ci_high", "value": ci[1]},
        {"metric": "empirical_rate", "value": emp_rate},
        {"metric": "dispersion(var/mean)", "value": dispersion},
    ]

    # Optional rate RATIO if a 2-level group is present
    ratio_note = ""
    group_col = _two_level_group(df, fp, cfg)
    if group_col is not None and group_col != outcome:
        cols = [outcome, group_col] + ([exp_col] if (exp_col and exp_col in df.columns) else [])
        sub = df[cols].dropna()
        levels = sorted(sub[group_col].unique().tolist(), key=lambda v: (isinstance(v, str), v))
        if len(levels) == 2:
            def _arm(lv):
                m = sub[group_col] == lv
                yy = sub.loc[m, outcome].to_numpy(dtype=float)
                if exp_col and exp_col in df.columns:
                    ee = sub.loc[m, exp_col].fillna(0.0).to_numpy(dtype=float)
                else:
                    ee = np.ones(len(yy))
                return float(yy.sum()), float(ee.sum())

            s1, e1 = _arm(levels[0])
            s2, e2 = _arm(levels[1])
            if e1 > 0 and e2 > 0:
                a1p, b1p = a0 + s1, b0 + e1
                a2p, b2p = a0 + s2, b0 + e2
                # rate ratio (group2 / group1): no closed form -> fixed-seed draws
                rng = np.random.default_rng(_SEED)
                r1 = rng.gamma(a1p, 1.0 / b1p, _N_DRAWS)
                r2 = rng.gamma(a2p, 1.0 / b2p, _N_DRAWS)
                ratio = r2 / r1
                ratio_mean = float(np.mean(ratio))
                rlo = float(np.quantile(ratio, (1 - level) / 2))
                rhi = float(np.quantile(ratio, 1 - (1 - level) / 2))
                p_ratio_gt1 = float(np.mean(ratio > 1.0))
                estimates["rate_ratio_mean"] = ratio_mean
                estimates["rate_ratio_ci_low"] = rlo
                estimates["rate_ratio_ci_high"] = rhi
                estimates["prob_ratio_gt1"] = p_ratio_gt1
                rows += [
                    {"metric": f"rate_{levels[0]}", "value": (a1p / b1p)},
                    {"metric": f"rate_{levels[1]}", "value": (a2p / b2p)},
                    {"metric": "rate_ratio_mean", "value": ratio_mean},
                    {"metric": "rate_ratio_ci_low", "value": rlo},
                    {"metric": "rate_ratio_ci_high", "value": rhi},
                    {"metric": "P(ratio>1)", "value": p_ratio_gt1},
                ]
                ratio_note = (
                    f" 率比({levels[1]}/{levels[0]}) 后验均值 {ratio_mean:.4f} "
                    f"[{rlo:.4f},{rhi:.4f}]，P(率比>1)={p_ratio_gt1:.4f}"
                    f"（{_N_DRAWS:,} 次抽样,seed={_SEED}）。"
                )

    pd.DataFrame(rows).to_csv(d / "bayesian_poisson_rate.csv", index=False, encoding="utf-8")
    files.append("bayesian_poisson_rate.csv")

    # Plot posterior rate density
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hi = max(ci[1] * 1.4, post_mean * 2.0, 1e-6)
        xs = np.linspace(0.0, hi, 500)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, stats.gamma.pdf(xs, a_post, scale=1.0 / b_post),
                label=f"Posterior Gamma({a_post:.3g},{b_post:.3g})")
        ax.axvline(post_mean, ls="--", lw=0.9, color="C1", label=f"mean={post_mean:.3g}")
        ax.axvspan(ci[0], ci[1], alpha=0.15, color="C0", label=f"{int(level*100)}% CrI")
        ax.set_xlabel("Rate (lambda)")
        ax.set_ylabel("Posterior density")
        ax.set_title("Bayesian Poisson rate — Gamma posterior")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "bayesian_poisson_rate_posterior.png", dpi=150)
        plt.close(fig)
        files.append("bayesian_poisson_rate_posterior.png")
    except Exception:
        pass

    prior_label = (
        "弱信息 Gamma(0.001,0.001)" if (a0 == 0.001 and b0 == 0.001)
        else f"Gamma({a0:g},{b0:g})"
    )
    summary.append(
        f"贝叶斯泊松率估计完成（Gamma-泊松共轭）：Σy={sum_y:g}，Σexposure={total_exposure:g}"
        f"（曝光={exp_label}）；后验 Gamma({a_post:g},{b_post:g})，后验均值率 {post_mean:.4f}，"
        f"{int(level*100)}% 可信区间 [{ci[0]:.4f},{ci[1]:.4f}]（经验率 {emp_rate:.4f}）。" + ratio_note
    )
    disp_msg = (
        f"⚠ 检测到过度离散（Pearson 离散度={dispersion:.2f}>1.5，已计入曝光）：泊松假定(率恒定/无过度离散)被违反，"
        "可信区间偏窄，建议改用负二项。"
        if overdispersed
        else (f"⚠ Pearson 离散度={dispersion:.2f}（接近 1，符合泊松假定）。" if np.isfinite(dispersion) else "")
    )
    summary.append(
        f"⚠ 先验：{prior_label}（默认弱信息，已声明）。⚠ 共轭 Gamma-泊松；假定 Poisson（率恒定、无过度离散）。"
        + disp_msg
        + ("⚠ 率比为固定种子蒙特卡洛抽样所得。" if ratio_note else "")
    )

    code += [
        "from scipy import stats",
        f"a_post, b_post = {a0}+sum_y, {b0}+total_exposure  # Gamma posterior",
        "post_mean = a_post/b_post  # posterior mean rate",
        "ci = stats.gamma.ppf([.025,.975], a_post, scale=1/b_post)",
    ]
