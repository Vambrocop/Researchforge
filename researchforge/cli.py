"""Command-line entry point for ResearchForge."""

from __future__ import annotations

import argparse

from researchforge import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="researchforge")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(f"researchforge {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
