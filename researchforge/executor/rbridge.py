"""Optional R bridge — run gold-standard CRAN packages (lavaan, QCA, gstat,
spdep, …) via subprocess Rscript.

Design rules:
- **Optional & graceful**: every caller checks `r_available()` / `r_package_available()`
  first and falls back to a pure-Python implementation when R (or the package) is
  missing, so portability is never lost.
- **Vetted, not fetched**: R code wired here has been reviewed; nothing is
  downloaded or executed from the network at runtime.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache


@lru_cache(maxsize=1)
def find_rscript() -> str | None:
    """Locate Rscript cross-platform; None if R is not installed."""
    exe = shutil.which("Rscript")
    if exe:
        return exe
    cands: list[str] = []
    # Windows: pick the highest installed R version
    for pat in (
        r"C:/Program Files/R/R-*/bin/x64/Rscript.exe",
        r"C:/Program Files/R/R-*/bin/Rscript.exe",
        r"C:/Program Files/Microsoft/R Open/R-*/bin/x64/Rscript.exe",
    ):
        cands += sorted(glob.glob(pat), reverse=True)
    # macOS / Linux
    cands += [
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
        "/usr/bin/Rscript",
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def r_available() -> bool:
    return find_rscript() is not None


def run_r(r_code: str, timeout: int = 120) -> str:
    """Run R code through a temporary script and return stdout.

    Raises RuntimeError if R is missing or the script exits non-zero.
    """
    rscript = find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found")
    fd, path = tempfile.mkstemp(suffix=".R")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(r_code)
        proc = subprocess.run(
            [rscript, "--vanilla", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"R error: {proc.stderr.strip()[:600]}")
        return proc.stdout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@lru_cache(maxsize=64)
def r_package_available(pkg: str) -> bool:
    """True if R is present and the named CRAN package is installed (cached)."""
    if not r_available():
        return False
    try:
        return "TRUE" in run_r(f'cat(requireNamespace("{pkg}", quietly=TRUE))', timeout=60)
    except Exception:
        return False
