"""profiler.types.infer_kind 类型推断锁测（真实数据 dogfood 逮到的 count 误判）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchforge.profiler.types import infer_kind


def test_whole_valued_float_high_cardinality_is_continuous():
    # 真数据 dogfood: sklearn diabetes 的 target 是 float64、214 个不同值、范围 25–346——
    # 一个用 .0 写出的连续测量(疾病进展评分),不是事件计数。旧逻辑"非负整数值即 count"
    # 把它误判 count→推 Poisson/NB;现 whole-valued float + 高基数(>15) → continuous。
    rng = np.random.default_rng(3)
    progression = pd.Series(np.round(rng.uniform(25, 346, 300)).astype(float), name="progression")
    assert infer_kind(progression) == "continuous"


def test_int_typed_genuine_count_still_count():
    # 未误伤:int 型真事件计数(Poisson 抽样)仍判 count(小-中值、maxのfew-hundred)。
    rng = np.random.default_rng(4)
    events = pd.Series(rng.poisson(3, 300).astype(int), name="events")
    assert infer_kind(events) == "count"
    events_hi = pd.Series(rng.poisson(40, 300).astype(int), name="events_hi")  # 高率计数 max~60
    assert infer_kind(events_hi) == "count"


def test_int_large_magnitude_high_cardinality_is_continuous():
    # 真数据 dogfood ②:整数金额/年龄天数(max≫1000、高基数)是连续测量,不是事件计数——
    # 没人 Poisson 建模均值上万的量。max≥1000 + 高基数 → continuous。
    rng = np.random.default_rng(6)
    amount = pd.Series(rng.integers(100, 99999, 300), name="amount_yuan")
    assert infer_kind(amount) == "continuous"
    age_days = pd.Series(rng.integers(6000, 30000, 300), name="age_days")
    assert infer_kind(age_days) == "continuous"


def test_low_cardinality_whole_float_still_count():
    # 阈值以下(≤15 唯一)的 whole-float 仍当 count(如 0–5 的次数,浮点存储也算计数)。
    rng = np.random.default_rng(5)
    n = pd.Series(np.round(rng.uniform(0, 5, 300)).astype(float), name="n")
    assert infer_kind(n) == "count"


def test_binary_and_id_unchanged():
    assert infer_kind(pd.Series([0, 1, 1, 0, 1], name="flag")) == "binary"
    assert infer_kind(pd.Series([10, 11, 12, 13, 14], name="row_id")) == "id"
