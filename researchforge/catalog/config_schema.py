"""Config validation against an analysis's machine-readable ParamSpec.

One source of truth (``AnalysisEntry.params``), consumed here at run time so a
typo'd key or wrong column name is *surfaced* instead of silently ignored (the
old failure mode: ``cfg.get("outom")`` misses, the analysis quietly falls back
to its auto default, and the user never learns their override did nothing).

Honest + non-blocking: validation only **warns**. The analysis still runs on its
auto defaults. An entry that declares no ``params`` is not key-validated (we
don't know its keys yet) — so this is safe to roll out incrementally.
"""

from __future__ import annotations

from typing import Optional

from researchforge.catalog.schema import AnalysisEntry, ParamSpec
from researchforge.profiler.fingerprint import DataFingerprint


def validate_config(
    entry: AnalysisEntry,
    config: Optional[dict],
    fp: Optional[DataFingerprint] = None,
) -> list[str]:
    """Return human-readable (Chinese) warnings about ``config`` vs the entry's
    declared params. Empty list = nothing to flag. Never raises."""
    if not config:
        return []
    if not entry.params:
        return []  # spec not declared yet — cannot key-validate, stay silent

    allowed = {p.name: p for p in entry.params}
    known_cols = {c.name for c in fp.columns} if fp is not None else None
    warns: list[str] = []

    for key, val in config.items():
        spec = allowed.get(key)
        if spec is None:
            opts = "、".join(allowed) or "（无）"
            warns.append(f"未知参数 '{key}'：{entry.id} 仅接受 {opts}；已忽略、回退默认。")
            continue
        warns.extend(_check_value(entry, spec, key, val, known_cols))

    return warns


def _check_value(
    entry: AnalysisEntry, spec: ParamSpec, key: str, val, known_cols
) -> list[str]:
    out: list[str] = []
    t = spec.type

    if t == "choice" and spec.choices and val not in spec.choices:
        out.append(
            f"参数 '{key}'={val!r} 不在允许取值 {spec.choices} 内；已忽略、回退默认。"
        )
    elif t == "int" and (isinstance(val, bool) or not isinstance(val, int)):
        out.append(f"参数 '{key}' 应为整数，收到 {type(val).__name__}；已忽略、回退默认。")
    elif t == "float" and (isinstance(val, bool) or not isinstance(val, (int, float))):
        out.append(f"参数 '{key}' 应为数值，收到 {type(val).__name__}；已忽略、回退默认。")
    elif t == "bool" and not isinstance(val, bool):
        out.append(f"参数 '{key}' 应为布尔值，收到 {type(val).__name__}；已忽略、回退默认。")
    elif t == "column":
        if not isinstance(val, str):
            out.append(f"参数 '{key}' 应为单个列名，收到 {type(val).__name__}；已忽略、回退默认。")
        elif known_cols is not None and val not in known_cols:
            out.append(f"参数 '{key}' 指定的列 '{val}' 不在数据中；已忽略、回退默认。")
    elif t == "columns":
        if not isinstance(val, (list, tuple)):
            out.append(f"参数 '{key}' 应为列名列表，收到 {type(val).__name__}；已忽略、回退默认。")
        elif known_cols is not None:
            missing = [c for c in val if c not in known_cols]
            if missing:
                out.append(f"参数 '{key}' 含数据中不存在的列 {missing}；这些将被忽略。")

    return out
