"""Tests for CONTENT ANALYSIS (Wave P5) — dictionary coding + inter-coder Cohen's κ.

Covers: dictionary-based coding on English and (short, no-space) Chinese corpora, the
Cohen's κ agreement path cross-checked against sklearn, and the honest-degrade path when
neither a dictionary nor coder columns are supplied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry():
    return Catalog.load().by_id("content_analysis")


_EN_DOCS = [
    "the telescope observed distant galaxies and bright stars",
    "astronauts launched a rocket into orbit around mars",
    "she chopped garlic and onion to make pasta sauce",
    "bake the bread with flour butter and sugar",
    "the stock market rallied as investors bought shares",
    "traders sold equities amid fears of inflation",
]
_EN_DICT = {
    "space": ["telescope", "galaxies", "rocket", "orbit", "stars", "mars"],
    "food": ["garlic", "onion", "bake", "flour", "sugar", "sauce", "bread"],
    "finance": ["stock", "market", "investors", "shares", "inflation", "equities"],
}

_ZH_DOCS = [
    "政策大力推进教育公平与改革发展",
    "医疗卫生体系保障人民群众健康",
    "推动教育资源均衡促进学生培养",
    "完善交通基础设施建设与运输网络",
    "加大教育投入深化课程改革发展",
    "强化医疗服务提升诊断治疗水平",
    "优化公共交通缓解城市道路拥堵",
    "职业教育培养技能型人才发展",
    "健康中国战略推进医疗药物保障",
]
_ZH_DICT = {
    "教育": ["教育", "学生", "培养", "课程"],
    "医疗": ["医疗", "健康", "诊断", "药物"],
    "交通": ["交通", "道路", "运输", "拥堵"],
}


def test_catalog_loads():
    e = _entry()
    assert e is not None
    assert e.executor_ref == "py::content_analysis"
    assert isinstance(e.biases, list) and len(e.biases) >= 4
    assert e.params and all(p.name for p in e.params)


def test_dictionary_english(tmp_path):
    csv = tmp_path / "e.csv"
    pd.DataFrame({"review": _EN_DOCS}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"dictionary": _EN_DICT})
    out = Path(res.output_dir)

    assert (out / "content_category_totals.csv").exists()
    assert (out / "content_doc_categories.csv").exists()
    assert res.estimates["n_categories"] == 3.0
    assert res.estimates["total_hits"] > 0

    totals = pd.read_csv(out / "content_category_totals.csv").set_index("category")
    # each planted theme should register hits
    for cat in ("space", "food", "finance"):
        assert totals.loc[cat, "total_hits"] > 0
    # dominant category of the space document should be space
    doc = pd.read_csv(out / "content_doc_categories.csv")
    assert doc.loc[0, "dominant_category"] == "space"


def test_dictionary_chinese_short_docs(tmp_path):
    """Short (no-space) Chinese answers are found (CJK short-text detection) and coded by
    substring match — no jieba needed."""
    csv = tmp_path / "z.csv"
    pd.DataFrame({"内容": _ZH_DOCS}).to_csv(csv, index=False, encoding="utf-8")
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"dictionary": _ZH_DICT})
    out = Path(res.output_dir)

    assert (out / "content_category_totals.csv").exists()
    assert res.estimates["n_docs"] == 9.0
    totals = pd.read_csv(out / "content_category_totals.csv").set_index("category")
    assert totals.loc["教育", "total_hits"] >= totals.loc["交通", "total_hits"]
    assert "子串匹配" in res.summary


def test_kappa_matches_sklearn(tmp_path):
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    c1 = rng.integers(0, 3, 60)
    c2 = c1.copy()
    flip = rng.random(60) < 0.2
    c2[flip] = rng.integers(0, 3, flip.sum())
    csv = tmp_path / "k.csv"
    pd.DataFrame({"coderA": c1, "coderB": c2}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"coder1": "coderA", "coder2": "coderB"})

    assert (Path(res.output_dir) / "content_coder_agreement.csv").exists()
    direct = float(cohen_kappa_score(c1.astype(str), c2.astype(str)))
    assert abs(res.estimates["cohen_kappa"] - direct) < 1e-5
    assert res.estimates["n_agreement"] == 60.0


def test_degrade_no_scheme(tmp_path):
    """Neither a dictionary nor coder columns -> honest 跳过, no fabricated categories."""
    csv = tmp_path / "e.csv"
    pd.DataFrame({"review": _EN_DOCS}).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "dictionary" in res.summary
    assert "n_categories" not in res.estimates
