"""Branch handlers for the multilabel family — a data shape the engine had NO method for.

Dogfood measurement (600 rows, 6 features, 5 co-occurring labels, 81% of rows carrying
more than one label) on the catalog as it stood:

    catalog methods for multi-label            0 of 303
    likely_outcome                             'feat6'  (a FEATURE, low confidence)
    likely_treatment                           'politics' (a LABEL, low confidence)
    logistic_regression → outcome 'politics', the other FOUR labels silently unused
    random_forest       → outcome 'feat1', regressing one feature on the others, R²=0.88
    manova              → 6 features as DVs ~ label 'politics' as the FACTOR (reversed)

So the failure was not label leakage into the predictors (measured: the other labels are
not used as predictors). It was that a five-label problem was silently reduced to a
one-label problem, or drifted off-task entirely — and the outcome nudge pointed at a
feature, because "last numeric column" is a position heuristic and the labels are binary.

Three methods, describe → predict-baseline → predict-with-dependence:

  * multilabel_profile   — label cardinality/density, per-label prevalence and imbalance
    ratio, pairwise Jaccard + phi co-occurrence, observed vs possible label sets.
    This is what tells you whether modelling label DEPENDENCE can pay at all.
  * binary_relevance     — one classifier per label, k-fold CV. The workhorse baseline.
  * classifier_chain     — an ENSEMBLE of chains over random label orders, so the reported
    gain is not an artefact of one lucky order; compared against binary relevance on the
    same CV splits.

Why the metrics need care (and why every summary here carries the trivial baseline):
Hamming loss rewards a predictor that never predicts a rare label. On imbalanced label
sets a majority-class predictor can post a better Hamming loss than a real model, so
reporting Hamming loss alone flatters nonsense. Every run here reports BOTH the
subset (exact-match) accuracy and the MAJORITY-CLASS baseline for the same split, and
says so in the summary.

Engine conventions (CLAUDE.md, executor/_branch_api.py): handlers are
``@register("<id>") def _branch_<id>(ctx)``, unpack ctx then MUTATE
summary/estimates/files/code; honest degrade appends "<方法>跳过：<原因>" and returns;
products are best-effort try/except; plot labels are ENGLISH (translated centrally by
_init_mpl_style); `estimates` values are floats.
"""

from __future__ import annotations

import importlib.util

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor._helpers.core import resolve_predictors

_SEED = 0          # fixed random_state, disclosed
_MIN_LABELS = 3    # 2 binary columns is a flag pair, not a label set
_MIN_ROWS = 60
_DEFAULT_FOLDS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Label-set resolution — the role-binding question for this family.
#
# "Which columns are the labels?" is this family's equivalent of resolve_outcome:
# get it wrong and every number afterwards answers a different question. Returns
# (labels, Y, mapping_notes, problem) — when `problem` is not None the caller
# appends it and RETURNS.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_labels(df, fp, cfg, method: str):
    import numpy as np

    cfg = cfg or {}
    forced = [c for c in (cfg.get("labels") or []) if c in df.columns]
    cands = forced or list(getattr(fp, "binary_columns", None) or [])
    cands = [c for c in cands if c in df.columns]

    if len(cands) < _MIN_LABELS:
        return None, None, [], (
            f"{method}跳过：需要至少 {_MIN_LABELS} 个二值标签列，当前只找到 {len(cands)} 个"
            f"（用 config labels 显式指定标签列）"
        )

    # Coerce each label to 0/1. A label may arrive as yes/no, true/false, or a string
    # pair; the mapping is DISCLOSED rather than guessed silently.
    cols, notes = [], []
    for c in cands:
        s = df[c].dropna()
        uniq = sorted(set(s.tolist()), key=lambda v: str(v))
        if len(uniq) != 2:
            continue
        if set(uniq) <= {0, 1} or set(uniq) <= {0.0, 1.0}:
            cols.append(df[c].fillna(0).astype(float).to_numpy())
        else:
            pos = uniq[1]
            cols.append((df[c] == pos).astype(float).to_numpy())
            notes.append(f"{c}: '{pos}'=1")
    labels = [c for c in cands if c in df.columns][: len(cols)]
    if len(cols) < _MIN_LABELS:
        return None, None, [], (
            f"{method}跳过：可用的二值标签列不足 {_MIN_LABELS} 个（需恰好两个取值）"
        )

    Y = np.column_stack(cols)
    per_row = Y.sum(axis=1)

    # The guard that keeps this family honest: one-hot encoded MULTI-CLASS is not
    # multi-label. If no row carries two labels at once there is no dependence to
    # model and no multi-label problem — say so and name the right method.
    if float(per_row.max()) <= 1.0:
        return None, None, [], (
            f"{method}跳过：每行最多只有一个标签为 1，这是独热编码的**多分类**问题、"
            f"不是多标签问题（没有任何标签共现）——请改用 multinomial_logit / "
            f"random_forest 等单结果分类方法"
        )
    if len(df) < _MIN_ROWS:
        return None, None, [], f"{method}跳过：样本量 {len(df)} < {_MIN_ROWS}，多标签交叉验证不稳定"

    return labels, Y, notes, None


def _features(df, fp, cfg, labels, method: str):
    """Predictor matrix. Binary columns are NOT auto-selected here — on a multi-label
    frame a binary column is presumptively a LABEL, and quietly feeding label k into
    the model for label j is leakage. config predictors can force any column in."""
    import numpy as np

    cfg = cfg or {}
    anchor = labels[0]
    feats = resolve_predictors(
        fp, cfg, anchor, kinds=("continuous", "count"), cap=30, df=df, forced_cap=30
    )
    feats = [c for c in feats if c not in set(labels)]
    if not feats:
        return None, None, f"{method}跳过：没有可用的数值预测变量（标签列已排除）"
    X = df[feats].apply(lambda s: s.fillna(s.median())).to_numpy(dtype=float)
    if not np.isfinite(X).all():
        return None, None, f"{method}跳过：预测变量含无法填补的非有限值"
    return feats, X, None


def _sklearn_missing(method: str) -> str | None:
    if importlib.util.find_spec("sklearn") is None:
        return f"{method}跳过：未安装 scikit-learn"
    return None


def _cv_predict_br(X, Y, folds: int):
    """Binary relevance under k-fold CV: one independent classifier per label.

    Returns (Yhat, n_degenerate) where n_degenerate counts (fold, label) cells whose
    training slice had a single class — those predict that constant, and the count is
    disclosed rather than hidden."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=folds, shuffle=True, random_state=_SEED)
    Yhat = np.zeros_like(Y)
    degenerate = 0
    for tr, te in kf.split(X):
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if len(np.unique(y[tr])) < 2:
                Yhat[te, j] = float(y[tr][0]) if len(tr) else 0.0
                degenerate += 1
                continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[tr], y[tr])
            Yhat[te, j] = clf.predict(X[te])
    return Yhat, degenerate


def _cv_predict_chain(X, Y, folds: int, n_chains: int):
    """Ensemble of classifier chains over random label orders, k-fold CV.

    A single chain's result depends on an arbitrary label order, so one chain is not
    evidence. We average predicted probabilities over `n_chains` random orders and
    also return the per-chain subset accuracies so the SPREAD can be disclosed."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.multioutput import ClassifierChain

    rng = np.random.default_rng(_SEED)
    kf = KFold(n_splits=folds, shuffle=True, random_state=_SEED)
    L = Y.shape[1]
    proba_sum = np.zeros(Y.shape, dtype=float)
    per_chain = []
    for _ in range(n_chains):
        order = list(rng.permutation(L))
        p = np.zeros(Y.shape, dtype=float)
        ok = True
        for tr, te in kf.split(X):
            if any(len(np.unique(Y[tr, j])) < 2 for j in range(L)):
                ok = False
                break
            chain = ClassifierChain(
                LogisticRegression(max_iter=2000), order=order, random_state=_SEED
            )
            chain.fit(X[tr], Y[tr])
            p[te] = chain.predict_proba(X[te])
        if not ok:
            continue
        proba_sum += p
        per_chain.append(float(accuracy_score(Y, (p >= 0.5).astype(float))))
    if not per_chain:
        return None, []
    return (proba_sum / len(per_chain) >= 0.5).astype(float), per_chain


def _score_block(Y, Yhat, labels):
    """The metric set, with the trivial baseline that keeps Hamming loss honest."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss

    maj = np.tile((Y.mean(axis=0) >= 0.5).astype(float), (Y.shape[0], 1))
    out = {
        "hamming_loss": float(hamming_loss(Y, Yhat)),
        "hamming_loss_majority_baseline": float(hamming_loss(Y, maj)),
        "subset_accuracy": float(accuracy_score(Y, Yhat)),
        "subset_accuracy_majority_baseline": float(accuracy_score(Y, maj)),
        "f1_micro": float(f1_score(Y, Yhat, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y, Yhat, average="macro", zero_division=0)),
    }
    per = f1_score(Y, Yhat, average=None, zero_division=0)
    for name, v in zip(labels, np.atleast_1d(per)):
        out[f"f1_{name}"] = float(v)
    return out


def _write_label_csv(d, files, labels, Y, extra=None):
    import pandas as pd

    rows = {"label": labels, "prevalence": [float(Y[:, j].mean()) for j in range(len(labels))]}
    if extra:
        rows.update(extra)
    p = d / "labels.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    files.append(p.name)


# ─────────────────────────────────────────────────────────────────────────────
@register("multilabel_profile")
def _branch_multilabel_profile(ctx: Ctx) -> None:
    df, fp, cfg, d = ctx.df, ctx.fp, ctx.cfg, ctx.d
    files, summary, estimates = ctx.files, ctx.summary, ctx.estimates
    import numpy as np

    labels, Y, notes, problem = _resolve_labels(df, fp, cfg, "多标签结构画像")
    if problem:
        summary.append(problem)
        return

    n, L = Y.shape
    per_row = Y.sum(axis=1)
    prevalence = Y.mean(axis=0)
    cardinality = float(per_row.mean())
    density = cardinality / L
    n_sets = int(len({tuple(r) for r in Y.tolist()}))
    imbalance = float(prevalence.max() / max(prevalence.min(), 1.0 / n))

    # pairwise structure: phi (binary correlation) and Jaccard
    phi = np.corrcoef(Y, rowvar=False) if L > 1 else np.array([[1.0]])
    jac = np.ones((L, L))
    for a in range(L):
        for b in range(L):
            inter = float(((Y[:, a] == 1) & (Y[:, b] == 1)).sum())
            union = float(((Y[:, a] == 1) | (Y[:, b] == 1)).sum())
            jac[a, b] = inter / union if union else 0.0
    off = [(abs(phi[a, b]), phi[a, b], labels[a], labels[b])
           for a in range(L) for b in range(a + 1, L) if np.isfinite(phi[a, b])]
    off.sort(reverse=True)

    estimates.update({
        "n_labels": float(L),
        "label_cardinality": round(cardinality, 4),
        "label_density": round(density, 4),
        "distinct_label_sets": float(n_sets),
        "possible_label_sets": float(2 ** L),
        "imbalance_ratio": round(imbalance, 3),
        "rows_with_multiple_labels": round(float((per_row > 1).mean()), 4),
    })
    for name, v in zip(labels, prevalence):
        estimates[f"prevalence_{name}"] = round(float(v), 4)

    try:
        import pandas as pd

        _write_label_csv(d, files, labels, Y,
                         extra={"imbalance_ratio": [round(float(prevalence.max() / max(v, 1.0 / n)), 3)
                                                    for v in prevalence]})
        p = d / "label_cooccurrence.csv"
        pd.DataFrame(jac, index=labels, columns=labels).to_csv(p)
        files.append(p.name)
    except Exception:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].bar(range(L), prevalence, color="#4C72B0")
        axes[0].set_xticks(range(L))
        axes[0].set_xticklabels(labels, rotation=30, ha="right")
        axes[0].set_ylabel("Prevalence")
        axes[0].set_title("Label prevalence")
        im = axes[1].imshow(jac, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_xticks(range(L))
        axes[1].set_xticklabels(labels, rotation=30, ha="right")
        axes[1].set_yticks(range(L))
        axes[1].set_yticklabels(labels)
        axes[1].set_title("Label co-occurrence (Jaccard)")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        p = d / "multilabel_profile.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        files.append(p.name)
    except Exception:
        pass

    top = ("；".join(f"{a}–{b} φ={v:.2f}" for _, v, a, b in off[:3])) if off else "无"
    summary.append(
        f"多标签结构画像完成：{L} 个标签、n={n}。标签基数（每行平均标签数）="
        f"{cardinality:.2f}，密度={density:.3f}，{100 * (per_row > 1).mean():.1f}% 的行带 >1 个标签。"
        f"观测到 {n_sets} 种标签组合（理论上限 {2 ** L}）。最不平衡标签的患病率比 = {imbalance:.1f}。"
        f"标签间关联最强的三对：{top}。"
    )
    if off and abs(off[0][1]) >= 0.2:
        summary.append(
            "标签之间存在明显关联 → 值得试 classifier_chain（建模标签依赖），"
            "并与 binary_relevance 基线比较 subset accuracy。"
        )
    else:
        summary.append(
            "⚠ 标签之间关联很弱 → classifier_chain 相对 binary_relevance 大概率没有增益，"
            "链式方法的额外复杂度可能挣不回来。"
        )
    if notes:
        summary.append("⚠ 非 0/1 标签的取值映射：" + "；".join(notes))
    summary.append(
        "⚠ 这是描述性结构画像，不做任何预测或推断；基数/密度/共现都是样本量的函数，"
        "小样本下罕见标签的共现估计极不稳定。"
    )


# ─────────────────────────────────────────────────────────────────────────────
@register("binary_relevance")
def _branch_binary_relevance(ctx: Ctx) -> None:
    df, fp, cfg, d = ctx.df, ctx.fp, ctx.cfg, ctx.d
    files, summary, estimates = ctx.files, ctx.summary, ctx.estimates

    method = "Binary relevance 多标签分类"
    miss = _sklearn_missing(method)
    if miss:
        summary.append(miss)
        return
    labels, Y, notes, problem = _resolve_labels(df, fp, cfg, method)
    if problem:
        summary.append(problem)
        return
    feats, X, problem = _features(df, fp, cfg, labels, method)
    if problem:
        summary.append(problem)
        return

    folds = max(2, int((cfg or {}).get("folds") or _DEFAULT_FOLDS))
    Yhat, degenerate = _cv_predict_br(X, Y, folds)
    sc = _score_block(Y, Yhat, labels)
    estimates.update({k: round(v, 4) for k, v in sc.items()})
    estimates["n_labels"] = float(len(labels))
    estimates["cv_folds"] = float(folds)

    try:
        import numpy as np
        _write_label_csv(d, files, labels, Y,
                         extra={"cv_f1": [round(sc[f"f1_{c}"], 4) for c in labels]})

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        idx = np.arange(len(labels))
        ax.bar(idx - 0.2, [sc[f"f1_{c}"] for c in labels], width=0.4,
               label="CV F1", color="#4C72B0")
        ax.bar(idx + 0.2, [float(Y[:, j].mean()) for j in range(len(labels))], width=0.4,
               label="Prevalence", color="#DD8452")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Per-label cross-validated F1 vs prevalence")
        ax.legend()
        fig.tight_layout()
        p = d / "binary_relevance_f1.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        files.append(p.name)
    except Exception:
        pass

    beats = sc["subset_accuracy"] > sc["subset_accuracy_majority_baseline"]
    summary.append(
        f"{method}完成：{len(labels)} 个标签、{len(feats)} 个预测变量、{folds} 折交叉验证。"
        f"Hamming loss={sc['hamming_loss']:.4f}（多数类基线 {sc['hamming_loss_majority_baseline']:.4f}），"
        f"subset(完全匹配)准确率={sc['subset_accuracy']:.4f}"
        f"（基线 {sc['subset_accuracy_majority_baseline']:.4f}），"
        f"micro-F1={sc['f1_micro']:.4f}、macro-F1={sc['f1_macro']:.4f}。"
    )
    if not beats:
        summary.append(
            "⚠ **模型的完全匹配准确率没有超过「每个标签都猜多数类」的平凡基线**——"
            "这里的预测变量对标签集没有可用信息，别把 Hamming loss 的数值当成成绩。"
        )
    summary.append(
        "⚠ Binary relevance 按构造**独立**拟合每个标签，完全不建模标签间依赖；"
        "若 multilabel_profile 显示标签相关，classifier_chain 可能在 subset accuracy 上更好"
        "（Hamming loss 上通常差别很小——两个指标衡量的不是一回事）。"
    )
    summary.append(
        "⚠ Hamming loss 会奖励「从不预测罕见标签」的模型，所以此处同时报告了多数类基线；"
        f"随机种子固定为 {_SEED}。"
    )
    if degenerate:
        summary.append(
            f"⚠ 有 {degenerate} 个 (折×标签) 单元的训练片只含单一类别（标签过于罕见），"
            "这些单元退化为常数预测。"
        )
    if notes:
        summary.append("⚠ 非 0/1 标签的取值映射：" + "；".join(notes))


# ─────────────────────────────────────────────────────────────────────────────
@register("classifier_chain")
def _branch_classifier_chain(ctx: Ctx) -> None:
    df, fp, cfg, d = ctx.df, ctx.fp, ctx.cfg, ctx.d
    files, summary, estimates = ctx.files, ctx.summary, ctx.estimates

    method = "Classifier chain 多标签分类"
    miss = _sklearn_missing(method)
    if miss:
        summary.append(miss)
        return
    labels, Y, notes, problem = _resolve_labels(df, fp, cfg, method)
    if problem:
        summary.append(problem)
        return
    feats, X, problem = _features(df, fp, cfg, labels, method)
    if problem:
        summary.append(problem)
        return

    cfg = cfg or {}
    folds = max(2, int(cfg.get("folds") or _DEFAULT_FOLDS))
    n_chains = max(1, int(cfg.get("chains") or 10))

    Yhat_chain, per_chain = _cv_predict_chain(X, Y, folds, n_chains)
    if Yhat_chain is None:
        summary.append(
            f"{method}跳过：某一折的训练片里有标签只含单一类别，链式模型无法拟合"
            "（标签过于罕见——可减小 folds 或先看 multilabel_profile）"
        )
        return
    Yhat_br, degenerate = _cv_predict_br(X, Y, folds)

    sc = _score_block(Y, Yhat_chain, labels)
    br = _score_block(Y, Yhat_br, labels)
    import numpy as np

    d_subset = sc["subset_accuracy"] - br["subset_accuracy"]
    spread = (float(np.min(per_chain)), float(np.max(per_chain)))

    estimates.update({k: round(v, 4) for k, v in sc.items()})
    estimates.update({
        "binary_relevance_subset_accuracy": round(br["subset_accuracy"], 4),
        "binary_relevance_hamming_loss": round(br["hamming_loss"], 4),
        "subset_accuracy_gain_over_br": round(d_subset, 4),
        "n_chains": float(len(per_chain)),
        "chain_subset_accuracy_min": round(spread[0], 4),
        "chain_subset_accuracy_mean": round(float(np.mean(per_chain)), 4),
        "chain_subset_accuracy_max": round(spread[1], 4),
        "cv_folds": float(folds),
    })

    try:
        _write_label_csv(d, files, labels, Y,
                         extra={"chain_cv_f1": [round(sc[f"f1_{c}"], 4) for c in labels],
                                "br_cv_f1": [round(br[f"f1_{c}"], 4) for c in labels]})

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.bar([0, 1], [br["subset_accuracy"], sc["subset_accuracy"]],
               color=["#8C8C8C", "#4C72B0"], width=0.5)
        ax.errorbar([1], [np.mean(per_chain)],
                    yerr=[[np.mean(per_chain) - spread[0]], [spread[1] - np.mean(per_chain)]],
                    fmt="o", color="#C44E52", capsize=5, label="Single-chain range")
        ax.axhline(br["subset_accuracy_majority_baseline"], ls="--", color="#555",
                   label="Majority baseline")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Binary relevance", "Chain ensemble"])
        ax.set_ylabel("Subset accuracy (CV)")
        ax.set_title("Does modelling label dependence pay?")
        ax.legend()
        fig.tight_layout()
        p = d / "classifier_chain_vs_br.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        files.append(p.name)
    except Exception:
        pass

    # A gain smaller than the spread ACROSS label orders is not a gain — it is the same
    # noise the ensemble was built to average out. Measured on conditionally-independent
    # data: Δ=+0.0040 with an order spread of ~0.02, i.e. the headline would have claimed
    # a win that the disclosure two lines below then denied. The verdict has to agree
    # with the evidence, not just be qualified by it.
    noise = spread[1] - spread[0]
    inconclusive = abs(d_subset) <= noise
    verdict = (
        "与 binary_relevance 基线无实质差别" if inconclusive
        else ("高于 binary_relevance 基线" if d_subset > 0 else "低于 binary_relevance 基线")
    )
    summary.append(
        f"{method}完成：{len(labels)} 个标签、{len(feats)} 个预测变量、{folds} 折交叉验证，"
        f"{len(per_chain)} 条随机标签顺序的链集成。subset(完全匹配)准确率="
        f"{sc['subset_accuracy']:.4f}，{verdict}"
        f"（{br['subset_accuracy']:.4f}，Δ={d_subset:+.4f}）；"
        f"Hamming loss={sc['hamming_loss']:.4f}（binary relevance {br['hamming_loss']:.4f}，"
        f"多数类基线 {sc['hamming_loss_majority_baseline']:.4f}）。"
    )
    if d_subset <= 0 or inconclusive:
        summary.append(
            "⚠ **建模标签依赖在这份数据上没有带来增益**——零结果照报。"
            "这通常意味着标签之间的关联可以被预测变量本身解释掉（条件独立），"
            "此时 binary_relevance 更简单、更该被选用。"
        )
    summary.append(
        f"⚠ 单条链的结果依赖于**任意的标签顺序**：{len(per_chain)} 条随机顺序单独看时，"
        f"subset 准确率在 {spread[0]:.4f}–{spread[1]:.4f} 之间波动"
        f"（跨度 {spread[1] - spread[0]:.4f}）。"
        f"{'这个跨度比上面的 Δ 还大，所以「链更好」这个结论在顺序噪声之内、不成立。' if (spread[1] - spread[0]) > abs(d_subset) else '报告的是集成结果，不是某条幸运的链。'}"
    )
    if sc["subset_accuracy"] < spread[0]:
        # Measured on a dependence-dominant frame: ensemble 0.5413 < every single chain
        # (min 0.5437). Not a bug — averaging PROBABILITIES optimises each label's marginal,
        # while subset accuracy scores the whole vector. Marginal averaging destroys the joint
        # coherence an individual chain maintains. Say so instead of letting the headline
        # understate the approach.
        summary.append(
            f"⚠ 集成的 subset 准确率（{sc['subset_accuracy']:.4f}）**低于每一条单链**"
            f"（最低 {spread[0]:.4f}）：概率平均优化的是各标签的**边际**，而 subset 准确率"
            f"考的是整个标签向量全中——平均会破坏单链维持的联合一致性。"
            f"若你的目标指标是 subset 准确率，单链（或对标签向量投票的集成）可能更合适；"
            f"Hamming loss / micro-F1 这类边际指标不受此影响。"
        )
    summary.append(
        "⚠ 链式方法在训练时用真实标签、预测时用预测标签（exposure bias），"
        f"误差会沿链传播；随机种子固定为 {_SEED}。"
    )
    if degenerate:
        summary.append(f"⚠ binary relevance 基线里有 {degenerate} 个 (折×标签) 单元退化为常数预测。")
    if notes:
        summary.append("⚠ 非 0/1 标签的取值映射：" + "；".join(notes))
