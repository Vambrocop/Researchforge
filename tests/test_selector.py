"""Tests for the fast, goal-aware method selector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.profiler import profile_dataset
from researchforge.recommender import select_top
from researchforge.recommender.goals import entry_matches_goal, resolve_goal


def _rich_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(20):
        for t in range(5):
            rows.append({"unit": u, "year": 2000 + t, "y": rng.normal(0, 1),
                         "treat": int(t >= 3 and u % 2 == 0), "x": rng.normal(0, 1), "grp": u % 3})
    p = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_resolve_goal() -> None:
    assert resolve_goal("causal") == "causal"
    assert resolve_goal("CAUSAL") == "causal"
    assert resolve_goal("regression") == "relate"   # keyword hit
    assert resolve_goal("回归") == "relate"           # label substring (Chinese)
    assert resolve_goal("nonsense-xyz") is None
    assert resolve_goal(None) is None


def test_entry_matches_predict_via_catalog_goal_field() -> None:
    # Wave K-F2: 一个 catalog goal='predict' 的方法在 --goal predict 下应匹配（信 entry.goal），
    # 即便它不在硬编码的 predict id/family 表里（dogfood 发现 P5）。
    from researchforge.catalog import Catalog

    preds = [e for e in Catalog.load().entries if getattr(e, "goal", None) == "predict"]
    assert preds, "expected catalog entries with goal=predict"
    assert all(entry_matches_goal(e, "predict") for e in preds)


def test_has_design_signal_double_gate() -> None:
    # Wave L-C1: has_design_signal 双证门槛——treatment 词(处理/dose)单独可判"有设计信号";
    # 单个 block 词(site/区组)不够、需 ≥2 设计词(防单歧义词把观测数据误判为设计实验)。
    from types import SimpleNamespace as NS

    from researchforge.recommender.goals import has_design_signal

    def _fp(cols):
        return NS(columns=[NS(name=c) for c in cols])

    assert has_design_signal(_fp(["处理", "区组", "产量"]))       # treatment 词 → True
    assert has_design_signal(_fp(["site", "batch", "y"]))         # ≥2 设计词 → True
    assert not has_design_signal(_fp(["site", "x", "y"]))         # 单个 block 词 → False(双证)
    assert not has_design_signal(_fp(["城市", "区域", "销售额"]))  # 观测无设计词 → False


def test_select_top_caps_and_filters(tmp_path: Path) -> None:
    fp = profile_dataset(_rich_csv(tmp_path))
    top6 = select_top(fp, top=6)
    assert 1 <= len(top6) <= 6                       # focused, non-empty
    # goal filtering is correct and narrows the pool
    for goal in ("causal", "relate", "compare"):
        picks = select_top(fp, goal=goal, top=20)
        assert all(entry_matches_goal(r.entry, goal) for r in picks)
    assert len(select_top(fp, goal="causal", top=100)) <= len(select_top(fp, top=100))


def test_cli_recommend_goal(tmp_path: Path, capsys) -> None:
    from researchforge.cli import main

    rc = main(["recommend", str(_rich_csv(tmp_path)), "--goal", "causal", "--top", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "推荐 top" in out and "因果" in out
