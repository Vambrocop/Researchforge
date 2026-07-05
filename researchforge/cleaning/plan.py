"""Generate, apply, and log a cleaning plan derived from quality diagnostics.

Philosophy: never silently destroy information. Duplicates / constant columns /
imputable gaps are handled; outliers are *flagged for review*, not removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from researchforge.profiler.fingerprint import DataFingerprint

_NUMERIC_KINDS = {"continuous", "count"}


class CleaningStep(BaseModel):
    # drop_duplicates | impute_median | impute_mode | drop_column | flag_outliers
    # | collapse_rare (rare categorical levels -> "Other") | flag_high_cardinality (advisory)
    action: str
    column: Optional[str] = None
    reason: str = ""


def make_cleaning_plan(fp: DataFingerprint) -> list[CleaningStep]:
    kinds = {c.name: c.kind for c in fp.columns}
    n = fp.n_rows
    steps: list[CleaningStep] = []
    for issue in fp.issues:
        if issue.kind == "duplicate_rows":
            steps.append(CleaningStep(action="drop_duplicates", reason=issue.detail))
        elif issue.kind == "constant":
            steps.append(
                CleaningStep(action="drop_column", column=issue.column, reason="constant column")
            )
        elif issue.kind == "missing":
            ratio = issue.count / n if n else 0.0
            if ratio > 0.5:
                steps.append(
                    CleaningStep(
                        action="drop_column",
                        column=issue.column,
                        reason=f"missing {ratio:.0%} — too sparse to impute",
                    )
                )
            elif kinds.get(issue.column) in _NUMERIC_KINDS:
                steps.append(
                    CleaningStep(action="impute_median", column=issue.column, reason=issue.detail)
                )
            else:
                steps.append(
                    CleaningStep(action="impute_mode", column=issue.column, reason=issue.detail)
                )
        elif issue.kind == "outliers":
            steps.append(
                CleaningStep(
                    action="flag_outliers",
                    column=issue.column,
                    reason=issue.detail + " (flagged for review, not auto-removed)",
                )
            )
        elif issue.kind == "rare_categories":
            steps.append(
                CleaningStep(
                    action="collapse_rare",
                    column=issue.column,
                    reason=issue.detail + " → 合并入 'Other'（稳住下游模型、收窄独热）",
                )
            )
        elif issue.kind == "high_cardinality":
            steps.append(
                CleaningStep(
                    action="flag_high_cardinality",
                    column=issue.column,
                    reason=issue.detail + " (像标识符/自由文本，建议分析时排除，不自动删除)",
                )
            )
    return steps


def apply_cleaning_plan(
    df: pd.DataFrame, steps: list[CleaningStep]
) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    log: list[dict] = []
    for step in steps:
        entry = {
            "action": step.action,
            "column": step.column,
            "reason": step.reason,
            "applied": False,
            "detail": "",
        }
        if step.action == "drop_duplicates":
            before = len(out)
            out = out.drop_duplicates().reset_index(drop=True)
            entry.update(applied=True, detail=f"removed {before - len(out)} duplicate rows")
        elif step.action == "drop_column":
            if step.column in out.columns:
                out = out.drop(columns=[step.column])
                entry.update(applied=True, detail="dropped column")
        elif step.action == "impute_median":
            if step.column in out.columns:
                med = out[step.column].median()
                out[step.column] = out[step.column].fillna(med)
                entry.update(applied=True, detail=f"imputed missing with median={med}")
        elif step.action == "impute_mode":
            if step.column in out.columns:
                mode = out[step.column].mode(dropna=True)
                if len(mode):
                    out[step.column] = out[step.column].fillna(mode.iloc[0])
                    entry.update(applied=True, detail=f"imputed missing with mode={mode.iloc[0]}")
        elif step.action == "flag_outliers":
            entry.update(applied=False, detail="flagged only — left unchanged")
        elif step.action == "collapse_rare":
            if step.column in out.columns:
                vc = out[step.column].value_counts(dropna=True)
                thresh = max(2, round(0.01 * len(out)))
                rare = list(vc[vc < thresh].index)
                if len(rare) >= 2:  # only worth it when a real tail collapses
                    col = out[step.column].astype("object")
                    out[step.column] = col.where(~col.isin(rare), "Other")
                    entry.update(
                        applied=True,
                        detail=f"collapsed {len(rare)} rare levels (<{thresh} rows) into 'Other'",
                    )
        elif step.action == "flag_high_cardinality":
            entry.update(applied=False, detail="flagged only — likely identifier/free text, left unchanged")
        log.append(entry)
    return out, log


def write_cleaning_log(log: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
