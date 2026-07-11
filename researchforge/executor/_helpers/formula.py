"""statsmodels formula 标识符安全（Wave K · B4a）。

中文/含空格/含运算符的列名直接进 statsmodels/patsy formula 会被解析器切碎或报错
（dogfooding P4/B 线：中文列名回归/混合模型静默丢变量或崩）。本 helper 把列名映射成安全
别名 `v1,v2,…`（抄 rbridge 的标识符守卫），并给回原名的映射，供结果表/摘要还原展示——
单一收口，别在各分支各自搓（B4 统一收口点）。

职责单一：**只管标识符安全**。分类列的 `C()` 包裹 / 哑变量化由消费点自行决定。
"""

from __future__ import annotations

import re

# 与 rbridge.r_names_safe 同一守卫：合法 formula 标识符 = 字母/点起头 + 字母数字/点/下划线
_IDENT = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")


def is_formula_safe(col: str) -> bool:
    """列名能否原样进 statsmodels/patsy formula（不被解析器切碎）。"""
    return bool(_IDENT.fullmatch(str(col)))


def safe_formula_terms(cols: list[str]) -> tuple[list[str], dict[str, str]]:
    """把列名转成 formula 安全的项。

    返回 ``(terms, rename_map)``：
      - ``terms``：与 ``cols`` 等长、同序的安全项列表——合法列名原样透传，
        非法（中文/空格/运算符）列名换成别名 ``v1,v2,…``；
      - ``rename_map``：``{别名: 原名}``，仅含被改名的列，供把 DataFrame 列 rename
        成别名跑模型、再把结果/摘要里的别名还原回中文原名展示。

    别名与任何输入列名去重（若某列本就叫 ``v1``，别名跳到 ``v2``），避免撞名。
    分类列的 ``C()`` 包裹不在此处——helper 只保证标识符安全（职责单一）。
    """
    existing = {str(c) for c in cols}
    terms: list[str] = []
    rename_map: dict[str, str] = {}
    counter = 0
    for c in cols:
        c = str(c)
        if is_formula_safe(c):
            terms.append(c)
            continue
        counter += 1
        alias = f"v{counter}"
        while alias in existing or alias in rename_map:
            counter += 1
            alias = f"v{counter}"
        terms.append(alias)
        rename_map[alias] = c
    return terms, rename_map
