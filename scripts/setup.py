"""Cross-platform setup for ResearchForge (Windows / macOS / Linux).

Installs the engine in editable mode with dev deps and creates working dirs.
Run: python scripts/setup.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if sys.version_info < (3, 11):
        print("Python 3.11+ required.", file=sys.stderr)
        return 1

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[dev]"])

    for sub in ("outputs", "data/raw"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    print("ResearchForge setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
