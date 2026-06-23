"""Data-quality diagnostics — flags issues the Cleaning stage can act on."""

from __future__ import annotations

import pandas as pd

from researchforge.profiler.fingerprint import Issue


def _severity(ratio: float) -> str:
    if ratio >= 0.20:
        return "high"
    if ratio >= 0.05:
        return "medium"
    return "low"


def diagnose(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    n = len(df)
    if n == 0:
        return issues

    # Disclose any numeric coercions the robust reader applied (never silent).
    for col, note in (df.attrs.get("rf_coercions") or {}).items():
        issues.append(
            Issue(kind="coerced_numeric", severity="low",
                  detail=f"文本列已转为数值：{note}", column=str(col))
        )

    dup = int(df.duplicated().sum())
    if dup:
        issues.append(
            Issue(kind="duplicate_rows", severity=_severity(dup / n),
                  detail=f"{dup} duplicate rows", count=dup)
        )

    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        if miss:
            issues.append(
                Issue(kind="missing", severity=_severity(miss / n),
                      detail=f"{miss} missing values", column=str(col), count=miss)
            )
        nuniq = int(s.nunique(dropna=True))
        if nuniq <= 1:
            issues.append(
                Issue(kind="constant", severity="low",
                      detail="constant / single-value column", column=str(col), count=n)
            )
        # High-cardinality text column (likely free text / identifier): a poor
        # grouping factor and a memory risk for one-hot. Numbers/dates exempt.
        elif (
            not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_bool_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s)
            and nuniq > max(50, 0.5 * n)
        ):
            issues.append(
                Issue(kind="high_cardinality", severity="low",
                      detail=f"{nuniq} distinct text values ({nuniq / n:.0%} of rows)",
                      column=str(col), count=nuniq)
            )
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            nn = s.dropna()
            if len(nn) >= 8:
                q1, q3 = nn.quantile(0.25), nn.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    n_out = int(((nn < q1 - 1.5 * iqr) | (nn > q3 + 1.5 * iqr)).sum())
                    if n_out:
                        issues.append(
                            Issue(kind="outliers", severity=_severity(n_out / len(nn)),
                                  detail=f"{n_out} IQR outliers", column=str(col), count=n_out)
                        )
    return issues
