"""Wave K · B4a — statsmodels formula 标识符安全 helper 单测。"""

from __future__ import annotations

from researchforge.executor._helpers.formula import is_formula_safe, safe_formula_terms


def test_legal_names_pass_through_unchanged():
    terms, rename = safe_formula_terms(["age", "bmi", "x1", "log.dose"])
    assert terms == ["age", "bmi", "x1", "log.dose"]
    assert rename == {}


def test_chinese_names_aliased_and_mapped_back():
    terms, rename = safe_formula_terms(["年龄", "满意度3", "bmi"])
    # 中文列换别名，合法列原样；顺序保持
    assert terms == ["v1", "v2", "bmi"]
    assert rename == {"v1": "年龄", "v2": "满意度3"}


def test_spaces_and_operators_are_unsafe():
    assert not is_formula_safe("total sales")
    assert not is_formula_safe("a-b")
    assert not is_formula_safe("x(1)")
    assert not is_formula_safe("年龄")
    assert is_formula_safe("age_at_entry")
    assert is_formula_safe("log.dose")


def test_alias_avoids_collision_with_existing_column():
    # 已有合法列恰好叫 v1 → 中文列的别名必须跳过它
    terms, rename = safe_formula_terms(["v1", "价格"])
    assert terms[0] == "v1"                 # 原样透传
    assert terms[1] == rename_key(rename)   # 别名不是 v1
    assert terms[1] != "v1"
    assert rename[terms[1]] == "价格"


def rename_key(rename: dict) -> str:
    return next(iter(rename))


def test_order_and_length_preserved():
    cols = ["x", "变量二", "z", "第 四 列"]
    terms, rename = safe_formula_terms(cols)
    assert len(terms) == len(cols)
    assert terms[0] == "x" and terms[2] == "z"
    assert set(rename.values()) == {"变量二", "第 四 列"}


def test_empty_input():
    assert safe_formula_terms([]) == ([], {})
