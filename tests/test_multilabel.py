"""multilabel 家族——引擎此前对这个数据形状 0 个方法。

Dogfood measurement that motivated the family (600 rows, 6 features, 5 co-occurring
labels, 81% of rows carrying >1 label), on the catalog as it stood:

    catalog methods for multi-label   0 of 303
    logistic_regression  -> outcome 'politics', the other FOUR labels silently unused
    random_forest        -> outcome 'feat1', regressing one feature on the others (R²=0.88)
    manova               -> the six features as DVs ~ label 'politics' as the FACTOR

The tests below are built around the one question that decides whether this family is
worth anything: **can the chain method actually DETECT label dependence, and does it
say so honestly when there is none?** A comparison that can only ever come out one way
is decoration, so both directions are pinned:

  * `_dependent()`  — labels are noisy copies of each other, the dependence is NOT
    recoverable from the features → the chain must BEAT binary relevance.
  * `_conditionally_independent()` — labels are independent given a latent factor that
    the features carry → the chain must NOT claim a win, and must say why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _dependent(n=500, seed=1):
    """y2..y4 are noisy copies of y1; the features cannot explain that link."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.normal(0, 1, n) for _ in range(3))
    y1 = (1 / (1 + np.exp(-(1.6 * x1 + 0.9 * x2))) > rng.random(n)).astype(int)

    def flip(y, p):
        return np.where(rng.random(n) < p, 1 - y, y)

    return pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "x3": x3.round(3),
                         "y1": y1, "y2": flip(y1, 0.08), "y3": flip(y1, 0.10),
                         "y4": flip(y1, 0.12)})


def _conditionally_independent(n=500, seed=0):
    """Labels are independent GIVEN a latent z, and the features carry z — so once you
    condition on the features there is no label dependence left to exploit."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, 3))
    X = z @ rng.normal(0, 1, (3, 4)) + rng.normal(0, 0.5, (n, 4))
    W = rng.normal(0, 1.4, (3, 4))
    df = pd.DataFrame({f"feat{i + 1}": X[:, i].round(3) for i in range(4)})
    for j, name in enumerate(["politics", "economy", "sport", "health"]):
        p = 1 / (1 + np.exp(-(z @ W[:, j] - 0.3)))
        df[name] = (rng.random(n) < p).astype(int)
    return df


def _onehot_multiclass(n=300, seed=5):
    """Exactly one label per row — multi-CLASS wearing multi-label clothes."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 3, n)
    return pd.DataFrame({"x1": rng.normal(0, 1, n).round(3),
                         "x2": rng.normal(0, 1, n).round(3),
                         "cls_a": (k == 0).astype(int),
                         "cls_b": (k == 1).astype(int),
                         "cls_c": (k == 2).astype(int)})


# ── the guard that keeps the family honest ───────────────────────────────────
@pytest.mark.parametrize("mid", ["multilabel_profile", "binary_relevance", "classifier_chain"])
def test_onehot_multiclass_is_refused_and_the_right_method_named(mid, tmp_path):
    """One-hot multi-class is not multi-label — no row carries two labels, so there is
    no dependence to model. Running anyway would answer a question nobody asked."""
    fp = _fp(_onehot_multiclass(), tmp_path, name="oh.csv")
    res = run_analysis(fp, _CAT.by_id(mid), output_root=str(tmp_path / ("o" + mid)))
    assert "跳过" in res.summary, res.summary[:200]
    assert "多分类" in res.summary
    assert "multinomial_logit" in res.summary, "an honest refusal names the alternative"
    assert not res.estimates


def test_two_binary_columns_are_a_flag_pair_not_a_label_set(tmp_path):
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({"x": rng.normal(0, 1, n).round(3),
                       "a": rng.integers(0, 2, n), "b": rng.integers(0, 2, n)})
    fp = _fp(df, tmp_path, name="two.csv")
    res = run_analysis(fp, _CAT.by_id("multilabel_profile"), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary and "至少 3" in res.summary


# ── profile ──────────────────────────────────────────────────────────────────
def test_the_profile_measures_the_structure_it_claims(tmp_path):
    df = _conditionally_independent()
    fp = _fp(df, tmp_path, name="prof.csv")
    res = run_analysis(fp, _CAT.by_id("multilabel_profile"), output_root=str(tmp_path / "o"))
    labels = ["politics", "economy", "sport", "health"]
    Y = df[labels].to_numpy()
    e = res.estimates
    assert e["n_labels"] == 4.0
    assert e["label_cardinality"] == pytest.approx(float(Y.sum(1).mean()), abs=1e-3)
    assert e["label_density"] == pytest.approx(float(Y.sum(1).mean()) / 4, abs=1e-3)
    assert e["possible_label_sets"] == 16.0
    assert e["rows_with_multiple_labels"] == pytest.approx(float((Y.sum(1) > 1).mean()), abs=1e-3)
    for c in labels:
        assert e[f"prevalence_{c}"] == pytest.approx(float(df[c].mean()), abs=1e-3)
    assert "label_cooccurrence.csv" in res.files


# ── the two directions that make the chain comparison mean something ─────────
@pytest.mark.slow
def test_the_chain_beats_binary_relevance_when_dependence_is_real(tmp_path):
    """If the chain can never win, the comparison it reports is decoration. Measured on
    this frame: BR subset 0.5212, chain ensemble 0.5413 (Δ=+0.0200)."""
    fp = _fp(_dependent(), tmp_path, name="dep.csv")
    res = run_analysis(fp, _CAT.by_id("classifier_chain"), output_root=str(tmp_path / "o"))
    e = res.estimates
    gain = e["subset_accuracy_gain_over_br"]
    spread = e["chain_subset_accuracy_max"] - e["chain_subset_accuracy_min"]
    assert gain > 0, e
    assert gain > spread, f"a real gain must clear the label-order noise: {gain} vs {spread}"
    assert "高于 binary_relevance 基线" in res.summary


@pytest.mark.slow
def test_no_gain_is_reported_as_no_gain_with_the_reason(tmp_path):
    """Labels independent given the features → the honest answer is "use the simpler
    method", not a manufactured improvement.

    My first version of this test asserted the gain is <= 0. Wrong premise, and the
    measurement said so: the gain came out at **+0.0040** — noise around zero, not a
    negative number. What matters is that +0.0040 sits INSIDE the spread across label
    orders (~0.02), i.e. it is the very noise the ensemble exists to average out. So the
    thing to pin is that the report does not CLAIM a win it cannot support."""
    fp = _fp(_conditionally_independent(), tmp_path, name="ci.csv")
    res = run_analysis(fp, _CAT.by_id("classifier_chain"), output_root=str(tmp_path / "p"))
    e = res.estimates
    gain = e["subset_accuracy_gain_over_br"]
    spread = e["chain_subset_accuracy_max"] - e["chain_subset_accuracy_min"]
    assert abs(gain) <= spread, f"gain {gain} vs order spread {spread}"
    assert "高于 binary_relevance 基线" not in res.summary, res.summary[:300]
    assert "无实质差别" in res.summary
    assert "没有带来增益" in res.summary
    assert "条件独立" in res.summary
    assert "binary_relevance 更简单" in res.summary


@pytest.mark.slow
def test_the_single_chain_spread_is_reported_and_can_veto_the_headline(tmp_path):
    """One chain's result depends on an arbitrary label order, so the spread across
    orders is part of the evidence, not a footnote."""
    fp = _fp(_conditionally_independent(), tmp_path, name="sp.csv")
    res = run_analysis(fp, _CAT.by_id("classifier_chain"), output_root=str(tmp_path / "q"))
    e = res.estimates
    assert e["chain_subset_accuracy_min"] <= e["chain_subset_accuracy_mean"]
    assert e["chain_subset_accuracy_mean"] <= e["chain_subset_accuracy_max"]
    assert "任意的标签顺序" in res.summary
    if (e["chain_subset_accuracy_max"] - e["chain_subset_accuracy_min"]) > abs(
        e["subset_accuracy_gain_over_br"]
    ):
        assert "在顺序噪声之内" in res.summary


# ── binary relevance: the metrics, and the baseline that keeps them honest ───
def test_the_majority_baseline_is_always_reported(tmp_path):
    """Hamming loss rewards never predicting a rare label. Quoting it without the
    trivial baseline flatters nonsense, so both are always present."""
    fp = _fp(_dependent(), tmp_path, name="br.csv")
    res = run_analysis(fp, _CAT.by_id("binary_relevance"), output_root=str(tmp_path / "o"))
    e = res.estimates
    assert "hamming_loss_majority_baseline" in e
    assert "subset_accuracy_majority_baseline" in e
    assert "多数类基线" in res.summary
    assert e["subset_accuracy"] > e["subset_accuracy_majority_baseline"], (
        "features genuinely predict these labels"
    )


def test_a_label_is_never_used_as_a_predictor_for_another_label(tmp_path):
    """Leakage check: binary columns are presumptively LABELS on this shape, so the
    auto predictor set must not contain any of them."""
    df = _dependent()
    fp = _fp(df, tmp_path, name="leak.csv")
    res = run_analysis(fp, _CAT.by_id("binary_relevance"), output_root=str(tmp_path / "o"))
    assert "4 个标签、3 个预测变量" in res.summary, res.summary[:200]


def test_uninformative_features_do_not_produce_a_fake_success(tmp_path):
    """Pure noise predictors: Hamming loss can still look 'fine', so the summary must
    say the model failed to beat the trivial baseline."""
    rng = np.random.default_rng(9)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(0, 1, n).round(3), "x2": rng.normal(0, 1, n).round(3)})
    base = rng.integers(0, 2, n)
    for c in ["l1", "l2", "l3"]:
        df[c] = np.where(rng.random(n) < 0.25, 1 - base, base)
    fp = _fp(df, tmp_path, name="noise.csv")
    res = run_analysis(fp, _CAT.by_id("binary_relevance"), output_root=str(tmp_path / "o"))
    e = res.estimates
    if e["subset_accuracy"] <= e["subset_accuracy_majority_baseline"]:
        assert "没有超过" in res.summary, res.summary[:300]


# ── config ───────────────────────────────────────────────────────────────────
def test_config_labels_and_predictors_are_honoured(tmp_path):
    df = _dependent()
    fp = _fp(df, tmp_path, name="cfg.csv")
    res = run_analysis(fp, _CAT.by_id("binary_relevance"), output_root=str(tmp_path / "o"),
                       config={"labels": ["y1", "y2", "y3"], "predictors": ["x1"], "folds": 3})
    assert res.estimates["n_labels"] == 3.0
    assert res.estimates["cv_folds"] == 3.0
    assert "3 个标签、1 个预测变量、3 折" in res.summary, res.summary[:200]
    assert "f1_y4" not in res.estimates
