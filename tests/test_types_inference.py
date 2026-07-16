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
    # 未误伤:int 型真事件计数(Poisson 抽样)仍判 count。
    rng = np.random.default_rng(4)
    events = pd.Series(rng.poisson(3, 300).astype(int), name="events")
    assert infer_kind(events) == "count"


def test_low_cardinality_whole_float_still_count():
    # 阈值以下(≤15 唯一)的 whole-float 仍当 count(如 0–5 的次数,浮点存储也算计数)。
    rng = np.random.default_rng(5)
    n = pd.Series(np.round(rng.uniform(0, 5, 300)).astype(float), name="n")
    assert infer_kind(n) == "count"


def test_binary_and_id_unchanged():
    assert infer_kind(pd.Series([0, 1, 1, 0, 1], name="flag")) == "binary"
    assert infer_kind(pd.Series([10, 11, 12, 13, 14], name="row_id")) == "id"
