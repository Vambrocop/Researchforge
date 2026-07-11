"""Wave K · F3 — 可疑完美 / 数据泄漏 事后诊断器单元测试。

验证 ``executor/_helpers/diagnostics.suspicious_fit_warnings`` 三条保守判据各自命中/不误伤，
以及 narrate 红线（警告串本身不含背书措辞）。（注：``test_diagnostics.py`` 是另一模块
``recommender.diagnostics`` 的测试，与本文件无关。）
"""

from __future__ import annotations

from researchforge.executor._helpers.diagnostics import (
    ENDORSING_PHRASES,
    suspicious_fit_warnings,
)


# ① 分类准确率高得可疑 ─────────────────────────────────────────────
def test_leaky_high_accuracy_fires():
    w = suspicious_fit_warnings(cv_accuracy=0.995)
    assert len(w) == 1 and "可疑地完美" in w[0] and "泄漏" in w[0]


def test_genuinely_strong_accuracy_does_not_fire():
    # 0.95 是真强模型的常见成绩，不该狼来了
    assert suspicious_fit_warnings(cv_accuracy=0.95) == []


def test_nan_or_none_accuracy_ignored():
    assert suspicious_fit_warnings(cv_accuracy=None) == []
    assert suspicious_fit_warnings(cv_accuracy=float("nan")) == []


def test_imbalanced_useless_model_does_not_fire():
    # 不均衡域（churn/fraud）：只猜多数类的无用模型 acc 也 >0.97，但 lift≈0 → 不该喊泄漏
    assert suspicious_fit_warnings(cv_accuracy=0.981, baseline_rate=0.980) == []
    assert suspicious_fit_warnings(cv_accuracy=0.995, baseline_rate=0.994) == []


def test_real_leak_still_fires_despite_high_baseline():
    # 真泄漏：准确率显著超过基线（lift 大）→ 仍报，基线率感知不该压掉真信号
    w = suspicious_fit_warnings(cv_accuracy=1.0, baseline_rate=0.69)  # churn fixture 基线
    assert len(w) == 1 and "可疑地完美" in w[0]
    assert len(suspicious_fit_warnings(cv_accuracy=0.99, baseline_rate=0.5)) == 1


# ② 单特征重要度碾压 ─────────────────────────────────────────────
def test_single_feature_dominance_fires_with_name():
    w = suspicious_fit_warnings(
        importances=[0.95, 0.02, 0.03],
        feature_names=["refund_amount", "tenure", "monthly_fee"],
    )
    assert len(w) == 1 and "refund_amount" in w[0] and "泄漏" in w[0]


def test_balanced_importances_do_not_fire():
    assert suspicious_fit_warnings(importances=[0.4, 0.35, 0.25]) == []


def test_single_importance_column_does_not_trivially_fire():
    # 只有 1 个特征时占比恒为 100%，不能据此报泄漏
    assert suspicious_fit_warnings(importances=[0.99]) == []


# ③ 完美分离 ─────────────────────────────────────────────────────
def test_separation_flag_fires():
    w = suspicious_fit_warnings(separation=True)
    assert len(w) == 1 and "完美分离" in w[0]


def test_separation_heuristic_fires_on_huge_se_and_flat_p():
    w = suspicious_fit_warnings(coefs=[0.5, 1.0], ses=[60.0, 0.3], pvalues=[0.999, 0.04])
    assert len(w) == 1 and "完美分离" in w[0]


def test_separation_heuristic_fires_when_pvalues_missing():
    # 分离时 p 常为 NaN/缺失；巨大 SE 单独也应触发
    w = suspicious_fit_warnings(coefs=[0.5], ses=[80.0])
    assert len(w) == 1 and "完美分离" in w[0]


def test_normal_glm_does_not_fire():
    assert suspicious_fit_warnings(coefs=[0.5, 1.0], ses=[0.2, 0.3], pvalues=[0.6, 0.04]) == []


# 组合 & 红线 ─────────────────────────────────────────────────────
def test_multiple_signatures_accumulate():
    w = suspicious_fit_warnings(
        cv_accuracy=0.99, importances=[0.93, 0.04, 0.03], feature_names=["leak", "a", "b"]
    )
    assert len(w) == 2


def test_no_evidence_returns_empty():
    assert suspicious_fit_warnings() == []


def test_warnings_do_not_narrate_as_honest_generalization():
    # narrate 红线：诊断警告本身绝不能出现背书措辞（样本外/泛化/可常规解读/诚实…）
    w = suspicious_fit_warnings(
        cv_accuracy=0.99, importances=[0.95, 0.03, 0.02], separation=True
    )
    joined = "".join(w)
    for bad in ENDORSING_PHRASES:
        assert bad not in joined, f"诊断警告不该含背书措辞 {bad!r}: {joined!r}"


# 铺点集成：ml.py rf/xgboost 在泄漏数据上必须报警而非背书 ─────────────────────
def test_leaky_churn_rf_flags_suspicious_end_to_end(tmp_path):
    # Wave K-F3 铺点：P5 churn 含 refund_amount 泄漏列，rf 拿到 ~100% 准确率——
    # 引擎必须报一等 ⚠（可疑地完美/泄漏），而不是把它 narrate 成诚实泛化估计。
    from fixtures.dogfood import build_p5_churn  # noqa: E402
    from researchforge.catalog import Catalog
    from researchforge.executor import run_analysis
    from researchforge.profiler import profile_dataset

    df = build_p5_churn()
    csv = tmp_path / "churn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, Catalog.load().by_id("random_forest"),
                       output_root=str(tmp_path / "o"), config={"outcome": "churn"})
    assert "可疑地完美" in res.summary or "泄漏" in res.summary, res.summary


def test_leaky_churn_gbm_fires_and_drops_endorsement(tmp_path):
    # Wave K-F3 红线：gbm 在泄漏 churn 上命中时，摘要必须报 ⚠ 且不得残留背书词（诚实泛化/样本外）。
    from fixtures.dogfood import build_p5_churn  # noqa: E402
    from researchforge.catalog import Catalog
    from researchforge.executor import run_analysis
    from researchforge.profiler import profile_dataset

    df = build_p5_churn()
    csv = tmp_path / "churn.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, Catalog.load().by_id("gradient_boosting"),
                       output_root=str(tmp_path / "o"), config={"outcome": "churn"})
    assert "可疑地完美" in res.summary or "泄漏" in res.summary, res.summary
    assert "诚实泛化" not in res.summary and "样本外" not in res.summary, res.summary
