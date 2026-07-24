"""Tests for the SEMANTIC NETWORK method (Wave P2 — keyword co-occurrence graph).

Corpus: documents drawn from three disjoint-vocabulary themes whose words co-occur only
within a theme, so greedy-modularity community detection should recover the themes.
networkx is required (importorskip). Also covers the Chinese char-bigram path and the
honest no-text-column degrade.
"""

from __future__ import annotations

import random as _random
from pathlib import Path

import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

pytest.importorskip("networkx", reason="semantic_network needs networkx")


def _entry():
    return Catalog.load().by_id("semantic_network")


_EN_THEMES = {
    "space": ["telescope", "galaxy", "orbit", "rocket", "planet", "star", "cosmic", "satellite"],
    "cook": ["garlic", "onion", "butter", "sugar", "flour", "bake", "recipe", "sauce"],
    "market": ["stock", "investor", "market", "bond", "inflation", "rate", "trade", "equity"],
}
_ZH_THEMES = {
    "教育": ["学校", "教育", "学生", "老师", "课程", "学习", "培养", "素质"],
    "医疗": ["医院", "医生", "病人", "健康", "治疗", "药物", "诊断", "护理"],
    "交通": ["道路", "交通", "车辆", "公交", "地铁", "出行", "拥堵", "运输"],
}


def _corpus_df(themes, per_theme=14, seed=2) -> pd.DataFrame:
    rng = _random.Random(seed)
    rows = []
    for th, ws in themes.items():
        for _ in range(per_theme):
            rows.append({"review": " ".join(rng.sample(ws, 6)), "cat": th})
    return pd.DataFrame(rows)


def test_catalog_loads():
    e = _entry()
    assert e is not None
    assert e.executor_ref == "py::semantic_network"
    assert isinstance(e.biases, list) and len(e.biases) >= 4
    assert e.params and all(p.name for p in e.params)


def test_recovers_theme_communities(tmp_path):
    csv = tmp_path / "c.csv"
    _corpus_df(_EN_THEMES).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "semantic_network_nodes.csv").exists()
    assert (out / "semantic_network_edges.csv").exists()
    assert res.estimates["vocab_size"] >= 6
    assert res.estimates["n_edges"] > 0
    assert res.estimates["n_communities"] >= 2

    nodes = pd.read_csv(out / "semantic_network_nodes.csv").set_index("word")
    # two same-theme words should land in the same community (within-theme co-occurrence)
    space = [w for w in _EN_THEMES["space"] if w in nodes.index]
    assert len(space) >= 2
    comms = {int(nodes.loc[w, "community"]) for w in space}
    # not all space words scattered into distinct communities
    assert len(comms) < len(space)


def test_chinese_fallback(tmp_path, monkeypatch):
    import researchforge.executor.branches.text_mining as tm
    monkeypatch.setattr(tm, "_jieba_available", lambda: False)

    csv = tmp_path / "zh.csv"
    _corpus_df(_ZH_THEMES).to_csv(csv, index=False, encoding="utf-8")
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "semantic_network_nodes.csv").exists()
    nodes = set(pd.read_csv(out / "semantic_network_nodes.csv")["word"])
    assert nodes & {"教育", "医生", "交通", "学校", "健康", "车辆"}, nodes
    assert res.estimates["n_communities"] >= 2
    assert "字符二元组" in res.summary


def test_degrade_no_text_column(tmp_path):
    df = pd.DataFrame({"x": range(30), "y": range(30, 60), "g": [0, 1] * 15})
    csv = tmp_path / "nums.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "未找到自由文本列" in res.summary


def test_degrade_high_thresholds(tmp_path):
    """Thresholds so high no pair co-occurs enough -> honest 跳过, no fabricated graph."""
    csv = tmp_path / "c.csv"
    _corpus_df(_EN_THEMES).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"min_cooccur": 9999})
    assert "跳过" in res.summary
    assert "n_edges" not in res.estimates
