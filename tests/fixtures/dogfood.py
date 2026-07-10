"""Synthetic fixtures reproducing the STRUCTURE of the 6 Wave-J dogfooding personas.

Each builder returns a small, deterministic (fixed-seed) ``pd.DataFrame`` that recreates
the column names/types/roles that tripped up the auto-selector in ``docs/dogfood-findings.md``
— NOT the full-size datasets from ``e:/tmp/dogfood/`` (those are scratch files that will
disappear; these live in the repo as the Wave K regression lock). Row counts are shrunk to
the minimum that still exercises the target methods (profiler/EFA/ANOVA/logistic all run
fine on n=90-300; nothing here needs the original n=250-800).

See ``docs/dogfood-findings.md`` (Wave K plan) and the archived ``ANSWER_KEY.md`` judge notes
for the full DGP + true effect sizes each builder is modeled on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_p1_likert(n: int = 120, seed: int = 101) -> pd.DataFrame:
    """P1 — 医学生问卷: 性别/年龄 + 满意度1..8 (Likert 1-5), two independent latent factors
    (items 1-4 -> FactorA, items 5-8 -> FactorB, loading 0.7), items 3 and 7 REVERSE-CODED
    (raw = 6 - aligned), ~4% MCAR missing per item. Chinese column names throughout.
    Modeled on ANSWER_KEY.md dataset 1 (survey_likert.xlsx, n=250, seed=101), shrunk to n=120.
    """
    rng = np.random.default_rng(seed)
    factor_a = rng.normal(0, 1, n)
    factor_b = rng.normal(0, 1, n)
    loading = 0.7
    items: dict[str, np.ndarray] = {}
    for i in range(1, 5):
        latent = loading * factor_a + np.sqrt(1 - loading**2) * rng.normal(0, 1, n)
        items[f"满意度{i}"] = np.clip(np.round(3 + latent * 1.15), 1, 5).astype(int)
    for i in range(5, 9):
        latent = loading * factor_b + np.sqrt(1 - loading**2) * rng.normal(0, 1, n)
        items[f"满意度{i}"] = np.clip(np.round(3 + latent * 1.15), 1, 5).astype(int)
    # reverse-keyed items (silent trap: naive scoring/alpha/EFA must reverse 6-x first)
    items["满意度3"] = 6 - items["满意度3"]
    items["满意度7"] = 6 - items["满意度7"]
    age = np.clip(np.round(rng.normal(23, 3, n)), 18, 40).astype(int)
    gender = np.where(rng.random(n) < 0.55, "女", "男")
    df = pd.DataFrame({"性别": gender, "年龄": age, **items})
    for col in [f"满意度{i}" for i in range(1, 9)]:
        mask = rng.random(n) < 0.04
        df.loc[mask, col] = np.nan
    return df


def build_p2_cohort(n: int = 300, seed: int = 10) -> pd.DataFrame:
    """P2 — 流行病 cohort: disease(0/1) placed FIRST (not last), smoking(0/1), age, sex(0/1),
    bmi. logit P(smoking)=f(age) (confounder), logit P(disease)=f(smoking, age, sex, bmi) with
    true smoking OR=2.0. Binary-heavy table (disease/smoking/sex all 0/1) so a naive "outcome =
    first continuous column" heuristic misses the real (binary, first-column) outcome entirely.
    Modeled on ANSWER_KEY.md dataset 2 (cohort.csv, n=800, seed=10), shrunk to n=300.
    """
    rng = np.random.default_rng(seed)
    age = np.round(rng.uniform(20, 80, n))
    sex = rng.binomial(1, 0.5, n)
    bmi = np.clip(rng.normal(25, 4, n), 15, 45)
    p_smoke = 1 / (1 + np.exp(-(-1.2 + 0.035 * (age - 50))))
    smoking = rng.binomial(1, p_smoke)
    p_dis = 1 / (1 + np.exp(-(-2.2 + np.log(2.0) * smoking + 0.04 * (age - 50)
                               + 0.3 * sex + 0.05 * (bmi - 25))))
    disease = rng.binomial(1, p_dis)
    return pd.DataFrame({"disease": disease, "smoking": smoking, "age": age,
                         "sex": sex, "bmi": bmi})


def build_p3_panel(n_firms: int = 20, n_years: int = 6, seed: int = 15) -> pd.DataFrame:
    """P3 — 公司-年面板: firm_id x year, cashflow correlated with an omitted firm fixed
    effect (which also drives investment), so pooled OLS on cashflow is upward-biased vs the
    within/FE estimator; `size` is a plausible-looking but functionally irrelevant distractor
    covariate. Modeled on ANSWER_KEY.md dataset 3 (panel_firms.csv, 60x8=480, seed=15), shrunk
    to 20 firms x 6 years = 120 rows.
    """
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    fe = np.repeat(rng.normal(0, 2, n_firms), n_years)  # row-length firm fixed effect
    mean_cf = 10 + 0.5 * fe
    cashflow = mean_cf + rng.normal(0, 2.38, n_firms * n_years)
    investment = 2 + 0.6 * cashflow + fe + rng.normal(0, 1, n_firms * n_years)
    size = 50 + 3 * fe + 0.5 * year + rng.normal(0, 5, n_firms * n_years)  # distractor
    return pd.DataFrame({"firm_id": firm, "year": year, "investment": investment.round(3),
                         "cashflow": cashflow.round(3), "size": size.round(3)})


def build_p4_rcbd(seed: int = 404) -> pd.DataFrame:
    """P4 — 农学田间试验 RCBD: 区组 (block1..block4, English-word+number labels) x 处理
    (对照/处理A/处理B/处理C/处理D, Chinese categorical) x 产量, 5 reps -> 100 rows. Only 处理B
    has a real ~15% yield lift; 区组 has a real (nuisance) block effect. Role-detection hints
    are English-only in the engine, so an auto-detector that name-matches will find nothing
    and must fall back sanely (处理 has 5 levels, 区组 has 4). Modeled on ANSWER_KEY.md dataset
    4 (field_trial.csv, 4x5x5=100, seed=404) — same size, no shrink needed.
    """
    rng = np.random.default_rng(seed)
    treatments = {"对照": 500.0, "处理A": 505.0, "处理B": 575.0, "处理C": 495.0, "处理D": 510.0}
    blocks = [f"block{i}" for i in range(1, 5)]
    block_eff = {b: rng.normal(0, 20) for b in blocks}
    rows = []
    for b in blocks:
        for trt, base in treatments.items():
            for _ in range(5):
                rows.append({"区组": b, "处理": trt,
                             "产量": base + block_eff[b] + rng.normal(0, 25)})
    return pd.DataFrame(rows)


def build_p5_churn(n: int = 300, seed: int = 483) -> pd.DataFrame:
    """P5 — churn 预测: customer_id (unique id, first col), tenure, monthly_fee,
    support_calls, region (uninformative), churn (binary outcome, 2nd-to-last col — NOT
    first/last), refund_amount (LEAKAGE: computed FROM churn — only churners get a refund).
    Honest ceiling ~70-75% accuracy on tenure/monthly_fee/support_calls; including
    refund_amount gives near-perfect separation. Modeled on ANSWER_KEY.md dataset 5
    (churn_predict.csv, n=600, seed=483), shrunk to n=300.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.uniform(0, 36, n)
    monthly_fee = np.clip(rng.normal(70, 15, n), 15, 200)
    support_calls = rng.poisson(2.0, n)
    region = rng.choice(["North", "South", "East", "West"], n)
    logit = -1.5 - 0.08 * tenure + 0.02 * monthly_fee + 0.35 * support_calls
    p = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, p)
    refund = np.where(churn == 1, rng.uniform(50, 200, n), 0.0) + rng.normal(0, 5, n)
    refund = np.clip(refund, 0, None)
    customer_id = np.arange(10000, 10000 + n)
    return pd.DataFrame({"customer_id": customer_id, "tenure": tenure.round(2),
                         "monthly_fee": monthly_fee.round(2), "support_calls": support_calls,
                         "region": region, "churn": churn, "refund_amount": refund.round(2)})


def build_p6_messy(n: int = 150, seed: int = 648) -> pd.DataFrame:
    """P6 — 脏行政表: 日期 (text, non-zero-padded "2024/M/D"), 城市/区域 (12 cities -> 4
    regions, 5 rare cities <1%), 销售额(万元) (text with comma thousands separators),
    成本(万元) (~6% MCAR missing), 备注 (mostly-empty free text), 年份 (CONSTANT column =
    2024). 华东 region sales run ~20% higher than the other 3 regions. 5 exact duplicate rows
    appended. Modeled on ANSWER_KEY.md dataset 6 (messy_admin.xlsx, n=300, seed=648), shrunk to
    n=150 (+5 duplicates = 155 rows).
    """
    rng = np.random.default_rng(seed)
    cities_regions = {
        "上海": "华东", "杭州": "华东", "南京": "华东", "苏州": "华东",
        "广州": "华南", "深圳": "华南", "厦门": "华南",
        "北京": "华北", "天津": "华北", "石家庄": "华北",
        "成都": "西南", "重庆": "西南",
    }
    common = ["上海", "杭州", "广州", "深圳", "北京", "天津", "成都"]
    rare = ["南京", "苏州", "厦门", "石家庄", "重庆"]
    cities = common + rare
    weights = np.array([0.135] * len(common) + [0.011] * len(rare))
    weights = weights / weights.sum()
    city = rng.choice(cities, n, p=weights)
    region = np.array([cities_regions[c] for c in city])
    base_mean = np.where(region == "华东", 1200.0, 1000.0)
    sales = rng.normal(base_mean, 150)
    cost = sales * rng.uniform(0.55, 0.65, n) + rng.normal(0, 30, n)
    cost = cost.astype(float)
    missing_mask = rng.random(n) < 0.06
    cost[missing_mask] = np.nan
    days = rng.integers(1, 366, n)
    dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(d) - 1) for d in days]
    date_text = [f"2024/{d.month}/{d.day}" for d in dates]  # non-zero-padded text
    remarks_pool = ["客户投诉", "补货延迟", "促销活动", "新店开业", "设备维修"]
    remark = np.where(rng.random(n) < 0.07, rng.choice(remarks_pool, n), "")
    year = np.full(n, 2024)  # constant column
    sales_text = [f"{s:,.1f}" for s in sales]  # comma thousands separator, text dtype
    out = pd.DataFrame({
        "日期": date_text, "城市": city, "区域": region,
        "销售额(万元)": sales_text, "成本(万元)": cost, "备注": remark, "年份": year,
    })
    dup_idx = rng.choice(n, 5, replace=False)
    out = pd.concat([out, out.loc[dup_idx]], ignore_index=True)
    return out
