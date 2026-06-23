"""Robust ingestion — read real-world messy CSV/Excel without crashing or
mis-typing.

Three hardening layers over a bare ``pd.read_csv``:

1. **Encoding** — try ``utf-8-sig`` (plain UTF-8 + strips a BOM), then ``gb18030``
   (Chinese GBK/GB2312 superset), then ``latin-1`` (never raises). A BOM no longer
   pollutes the first column name; a GBK export no longer ``UnicodeDecodeError``s.
2. **Delimiter** — sniff ``, ; \\t |`` from the header line, so a semicolon/tab
   export no longer loads as a single column.
3. **Numeric coercion** — a column that is *really* numeric but stored as text
   (thousands separators ``1,234``, currency ``$5`` / ``￥5``, trailing ``12%``,
   stray missing-tokens ``-`` / ``missing``) is coerced to numbers. This is
   **conservative** (only when ≥90% of non-missing values parse) so genuine
   categoricals are never turned into numbers, and **never silent** — every
   coercion is recorded in ``df.attrs['rf_coercions']`` for the Profiler to
   disclose as an Issue.

Assumptions (disclosed): the thousands separator is ``,`` (US/international
grouping); European decimal-comma (``1,5`` = 1.5) is left untouched rather than
risk corrupting it. See docs/deferred-log.md for the i18n / charset-detector
follow-ups.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Tried in order: BOM-aware UTF-8, Chinese, never-fail fallback.
_ENCODINGS = ("utf-8-sig", "gb18030", "latin-1")
_DELIMS = (",", ";", "\t", "|")

# Missing-tokens pandas does NOT treat as NA by default (it already covers
# '', 'NA', 'N/A', 'NULL', 'null', 'None', 'NaN', 'n/a', 'nan', '<NA>', …).
# Applied ONLY inside numeric coercion, so a real categorical level like "-"
# is preserved when the column is not actually numeric.
_EXTRA_NA = {
    "-", "--", "---", "—", "–", "?", "??", ".", "..",
    "missing", "unknown", "n.a.", "na.", "none.", "tbd", "n/d", "nd",
}
# A comma that groups thousands: between a digit and exactly three digits at a
# word boundary (matches 1,234 and the right comma of 1,234,567; leaves 1,5 alone).
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_CURRENCY = re.compile(r"^[\$¥€£￥]\s*")
_PCT = re.compile(r"%\s*$")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/Excel robustly. Returns a DataFrame whose ``.attrs`` may carry
    ``rf_coercions`` (a dict of column -> coercion note) and ``rf_encoding`` /
    ``rf_sep`` for transparency. Drop-in replacement for the old reader."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        df.attrs["rf_encoding"] = "excel"
        df.attrs["rf_sep"] = None
    else:
        df, enc, sep = _read_csv_robust(path)
        df.attrs["rf_encoding"] = enc
        df.attrs["rf_sep"] = sep
    coercions = _coerce_numeric_columns(df)
    df.attrs["rf_coercions"] = coercions
    return df


def _read_csv_robust(path: Path) -> tuple[pd.DataFrame, str, str]:
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            sep = _sniff_sep(path, enc)
            df = pd.read_csv(path, encoding=enc, sep=sep)
            return df, enc, sep
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    # latin-1 above never raises UnicodeDecodeError, so we normally never get
    # here; keep an explicit byte-level fallback just in case.
    raw = path.read_bytes().decode("latin-1", errors="replace")
    from io import StringIO

    sep = _sniff_text_sep(raw.splitlines()[0] if raw else "")
    return pd.read_csv(StringIO(raw), sep=sep), "latin-1(replace)", sep


def _sniff_sep(path: Path, enc: str) -> str:
    with open(path, encoding=enc) as f:
        first = f.readline()
    return _sniff_text_sep(first)


def _sniff_text_sep(header_line: str) -> str:
    """Pick the delimiter that appears most in the header line; default comma."""
    if not header_line:
        return ","
    counts = {d: header_line.count(d) for d in _DELIMS}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def _coerce_numeric_columns(df: pd.DataFrame) -> dict[str, str]:
    """Coerce text columns that are really numeric. Mutates df in place; returns
    {column: human note} for every column actually coerced."""
    coercions: dict[str, str] = {}
    for col in list(df.columns):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        coerced, note = _try_coerce(s)
        if coerced is not None:
            df[col] = coerced
            coercions[str(col)] = note
    return coercions


def _try_coerce(s: pd.Series):
    """Return (numeric_series, note) if ≥90% of non-missing values parse as
    numbers after light cleaning; else (None, '')."""
    text = s.astype("string").str.strip()
    # Stray missing-tokens become NA for the parse (only matters if numeric).
    extra_na = text.str.lower().isin(_EXTRA_NA)
    text = text.mask(extra_na)

    work = (
        text.str.replace(_CURRENCY, "", regex=True)
        .str.replace(_THOUSANDS, "", regex=True)
    )
    is_pct = work.str.contains(_PCT, regex=True, na=False)
    work = work.str.replace(_PCT, "", regex=True).str.strip()

    parsed = pd.to_numeric(work, errors="coerce")

    denom = int(text.notna().sum())          # real values (excl. blanks + extra-NA)
    if denom == 0:
        return None, ""
    ok = int((parsed.notna() & text.notna()).sum())
    if ok / denom < 0.9 or ok == 0:
        return None, ""

    note_bits = []
    # Whole column is percentages -> divide by 100 so 12% -> 0.12.
    pct_on_real = is_pct[text.notna()]
    if len(pct_on_real) and bool(pct_on_real.all()):
        parsed = parsed / 100.0
        note_bits.append("百分号→比例(÷100)")
    else:
        note_bits.append("文本→数值")
    if int(extra_na.sum()):
        note_bits.append(f"{int(extra_na.sum())} 个缺失标记→NaN")
    note = f"{ok}/{denom} 解析为数（{'、'.join(note_bits)}）"
    return parsed.astype(float), note
