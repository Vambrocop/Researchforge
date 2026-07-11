"""可疑完美 / 数据泄漏 事后诊断（Wave K · F3）。

dogfooding P5 逮到最伤"诚实"招牌的事故：churn 数据里 refund_amount 是**结果发生后**才产生
的泄漏字段，模型据它拿到近乎 100% 的交叉验证准确率，而引擎把这个成绩 narrate 成"诚实的样本外
泛化估计"——零警告。本模块提供一个**保守**的事后检测器：只在三条清晰的泄漏/完美分离特征上发声，
不对真正强的模型狼来了。命中时由调用方把结果**降级**为一等 ⚠、并**禁止**再出现"样本外泛化/
可常规解读/泛化性能"等背书措辞（narrate 红线，见 docs/dogfood-findings.md 的 K-F3）。

注：这是**事后**启发式（拿到 cv 成绩/重要度/系数之后再看），不是特征工程级的泄漏预扫
（结果-特征近完美关联 / 结果后字段识别）——那个更重，排在 Wave L。
"""

from __future__ import annotations

import numpy as np

# ── 保守阈值（宁可漏报也别狼来了；调这里别散落到各分支） ───────────────────────
_LEAK_ACC = 0.97       # 分类准确率高到这个程度：可疑，提示查泄漏
_MIN_LIFT = 0.05       # …但还须比多数类基线率高出这么多才发声（防不均衡域误报）
_DOMINANCE = 0.90      # 单个特征占了 ≥90% 的重要度：典型泄漏特征信号
_SEP_SE = 50.0         # 系数标准误大到这个量级：完美分离把 SE 吹爆的典型标志
_SEP_P = 0.99          # 且 p≈1：配合巨大 SE 判完美分离（真微弱效应 p 不会同时 ≈1）

# 背书性措辞黑名单：命中诊断时，摘要里禁止出现这些词（narrate 红线，供调用方自检/测试引用）
ENDORSING_PHRASES = ("样本外", "泛化", "可常规解读", "泛化性能", "诚实")


def suspicious_fit_warnings(
    *,
    cv_accuracy: float | None = None,
    baseline_rate: float | None = None,
    importances=None,
    feature_names=None,
    coefs=None,
    ses=None,
    pvalues=None,
    separation: bool = False,
) -> list[str]:
    """返回一列一等 ⚠ 警告串（中文）；空列表 = 无可疑迹象。

    调用方按自己手头有的证据传参（分类传 cv_accuracy/importances；GLM 传 coefs/ses/pvalues
    或 separation 旗标）。**命中任一条 → 调用方须把结果降级为 ⚠ 且抹掉背书措辞**（见模块 docstring）。

    保守设计：三条判据各自独立、阈值偏严，宁可漏也别误伤真正强的模型。
    """
    flags: list[str] = []

    # ① 分类：准确率高得可疑，且明显超过多数类基线 → 优先怀疑结果后泄漏字段。
    #    看 lift（准确率−基线率）而非裸准确率：不均衡域（churn/fraud，正类≤3%）里一个只
    #    猜多数类的无用模型也会破 0.97，不减基线就会对合法模型误喊"泄漏"。baseline_rate
    #    缺省（None）时退回裸阈值（保守），调用方应传多数类占比以启用防误报。
    if cv_accuracy is not None and np.isfinite(cv_accuracy) and cv_accuracy > _LEAK_ACC:
        _lift = (cv_accuracy - baseline_rate
                 if baseline_rate is not None and np.isfinite(baseline_rate) else None)
        if _lift is None or _lift > _MIN_LIFT:
            flags.append(
                f"⚠ 结果可疑地完美（预测准确率高达 {cv_accuracy:.3f}）——高度警惕结果后泄漏字段"
                "（尤其在结果发生之后才产生的量，如退款/结案/注销之后才写入的记录）；"
                "请逐列核查是否有由结果反推得到的特征，剔除后重估。"
            )

    # ② 单个特征几乎独力决定预测 → 典型泄漏特征
    if importances is not None:
        # 泄漏特征=大**正**重要度；置换重要度可为负，clip 掉负值免得大负值被当"主导"
        imp = np.clip(np.asarray(importances, dtype=float), 0, None)
        if imp.size > 1 and np.isfinite(imp).any():
            total = float(np.nansum(imp))
            if total > 0:
                top = int(np.nanargmax(imp))
                share = float(imp[top]) / total
                if share >= _DOMINANCE:
                    who = ""
                    if feature_names is not None and top < len(feature_names):
                        who = f"（{feature_names[top]}）"
                    flags.append(
                        f"⚠ 单个特征{who}占了 {share:.0%} 的重要度、近乎独力决定预测——典型泄漏特征信号，"
                        "请核查该列是否由结果反推得到（若是，剔除后模型才反映真实可预测性）。"
                    )

    # ③ GLM 完美分离：系数未真正收敛、不可解读。GLM 调用方须同时传 pvalues——靠 p≈1
    #    区分"真分离"（bse 爆大且 p→1）与"合法的大 SE"（微尺度变量 bse 大但 p 显著）。
    hit_sep = bool(separation)
    if not hit_sep and coefs is not None and ses is not None:
        c = np.asarray(coefs, dtype=float)
        s = np.asarray(ses, dtype=float)
        p = np.asarray(pvalues, dtype=float) if pvalues is not None else np.full(s.shape, np.nan)
        with np.errstate(invalid="ignore"):
            big_se = np.isfinite(s) & (np.abs(s) > _SEP_SE)
            p_flat = ~np.isfinite(p) | (p > _SEP_P)  # p≈1 或 p 缺失（分离时常 NaN）
            hit_sep = bool(np.any(big_se & p_flat))
    if hit_sep:
        flags.append(
            "⚠ 完美分离：模型未真正收敛，系数与标准误不可解读——切勿据此报 OR / 效应量或声称显著。"
        )

    return flags
