"""Machine-readable config-param spec + run-time validation (v0.9).

Covers the ParamSpec model on AnalysisEntry, the validator's warnings (unknown
key / bad choice / wrong column / wrong type), its graceful silence when no spec
is declared, and the end-to-end surfacing of a bad override in run_analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from researchforge.catalog import Catalog
from researchforge.catalog.config_schema import validate_config
from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _bare_entry(params=None) -> AnalysisEntry:
    return AnalysisEntry(
        id="x", method="m", domain="statistics", family="statistics",
        goal="explore", params=params or [],
    )


def _two_numeric_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": range(20), "b": [i * 1.3 + 2 for i in range(20)],
                       "c": [i * 0.7 for i in range(20)]})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    return csv


# --------------------------------------------------------------------------- #
# Spec loads from the catalog
# --------------------------------------------------------------------------- #
def test_params_load_from_catalog() -> None:
    cat = Catalog.load()
    pearson = cat.by_id("pearson_correlation")
    assert pearson is not None
    names = {p.name for p in pearson.params}
    assert names == {"x", "y"}

    matrix = cat.by_id("correlation_matrix")
    method = next(p for p in matrix.params if p.name == "method")
    assert method.type == "choice"
    assert set(method.choices) == {"pearson", "spearman"}

    cohens = cat.by_id("cohens_d")
    assert {p.name for p in cohens.params} == {"outcome", "group", "ci"}


# --------------------------------------------------------------------------- #
# validate_config
# --------------------------------------------------------------------------- #
def test_no_spec_is_silent() -> None:
    # entry with no declared params -> cannot validate keys, stays silent
    assert validate_config(_bare_entry(), {"whatever": 1}) == []


def test_empty_config_is_silent() -> None:
    cat = Catalog.load()
    assert validate_config(cat.by_id("pearson_correlation"), None) == []
    assert validate_config(cat.by_id("pearson_correlation"), {}) == []


def test_unknown_key_warns() -> None:
    cat = Catalog.load()
    warns = validate_config(cat.by_id("pearson_correlation"), {"xx": "a"})
    assert len(warns) == 1
    assert "未知参数" in warns[0] and "xx" in warns[0]


def test_known_keys_no_warning() -> None:
    cat = Catalog.load()
    assert validate_config(cat.by_id("pearson_correlation"), {"x": "a", "y": "b"}) == []


def test_bad_choice_warns() -> None:
    cat = Catalog.load()
    warns = validate_config(cat.by_id("correlation_matrix"), {"method": "nope"})
    assert len(warns) == 1
    assert "method" in warns[0]


def test_wrong_column_warns_with_fingerprint(tmp_path: Path) -> None:
    fp = profile_dataset(_two_numeric_csv(tmp_path))
    cat = Catalog.load()
    warns = validate_config(cat.by_id("pearson_correlation"), {"x": "nonexistent"}, fp)
    assert any("不在数据中" in w for w in warns)
    # a real column produces no warning
    assert validate_config(cat.by_id("pearson_correlation"), {"x": "a"}, fp) == []


def test_columns_type_must_be_list() -> None:
    cat = Catalog.load()
    warns = validate_config(cat.by_id("correlation_matrix"), {"columns": "a"})
    assert any("列名列表" in w for w in warns)


def test_ci_float_type_check() -> None:
    cat = Catalog.load()
    assert validate_config(cat.by_id("cohens_d"), {"ci": 0.9}) == []
    warns = validate_config(cat.by_id("cohens_d"), {"ci": "high"})
    assert any("数值" in w for w in warns)


# --------------------------------------------------------------------------- #
# End-to-end through run_analysis
# --------------------------------------------------------------------------- #
def test_run_surfaces_bad_config_key(tmp_path: Path) -> None:
    fp = profile_dataset(_two_numeric_csv(tmp_path))
    entry = Catalog.load().by_id("correlation_matrix")
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"),
                       config={"method": "pearson", "typoo": 1})
    assert "配置参数提示" in res.summary
    assert "typoo" in res.summary


def test_run_clean_config_no_warning(tmp_path: Path) -> None:
    fp = profile_dataset(_two_numeric_csv(tmp_path))
    entry = Catalog.load().by_id("correlation_matrix")
    res = run_analysis(fp, entry, output_root=str(tmp_path / "o"),
                       config={"method": "spearman"})
    assert "配置参数提示" not in res.summary
