"""Command-line entry point for ResearchForge."""

from __future__ import annotations

import argparse
import sys

from researchforge import __version__

_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
_ASCII = {"green": "[OK]", "yellow": "[! ]", "red": "[X ]"}


def _ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def _markers() -> dict[str, str]:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return _EMOJI if "utf" in enc else _ASCII


def _cmd_recommend(path: str) -> int:
    from researchforge.profiler import profile_dataset
    from researchforge.recommender import recommend

    fp = profile_dataset(path)
    print(
        f"数据：{fp.n_rows} 行 × {fp.n_cols} 列 | "
        f"面板={fp.is_panel} 时序={fp.is_timeseries} "
        f"时间列={fp.time_col} 单位列={fp.unit_col}"
    )
    if fp.issues:
        print(f"质量问题：{len(fp.issues)} 项（运行清洗可处理）")
    print("\n可做的分析（按严谨度排序，红灯需知情覆盖）：")
    mark = _markers()
    for r in recommend(fp):
        print(f"  {mark[r.rigor.light]} [{r.rigor.score:3d}] {r.entry.method} — {r.rigor.note}")
    return 0


def _cmd_run(path: str, analysis_id: str) -> int:
    from researchforge.catalog import Catalog
    from researchforge.executor import run_analysis
    from researchforge.profiler import profile_dataset

    fp = profile_dataset(path)
    entry = Catalog.load().by_id(analysis_id)
    if entry is None:
        print(f"未知分析 id：{analysis_id}")
        return 1
    res = run_analysis(fp, entry)
    print(f"已执行：{res.method}")
    print(f"摘要：{res.summary}")
    print(f"产物目录：{res.output_dir}")
    for f in res.files:
        print(f"  - {f}")
    return 0


def _cmd_ingest() -> int:
    from researchforge.ingestion import ingest_inbox

    items = ingest_inbox()
    if not items:
        print("skills_inbox 无可登记项。")
        return 0
    print(f"已登记 {len(items)} 项：")
    for i in items:
        desc = (i.description[:60] + "…") if len(i.description) > 60 else i.description
        print(f"  - [{i.kind}] {i.name} — {desc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(prog="researchforge")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")
    rec = sub.add_parser("recommend", help="profile data and list feasible analyses")
    rec.add_argument("path", help="path to a CSV/Excel file")
    run_p = sub.add_parser("run", help="run a chosen analysis and save outputs")
    run_p.add_argument("path", help="path to a CSV/Excel file")
    run_p.add_argument("analysis", help="analysis id from the catalog (e.g. did)")
    sub.add_parser("ingest", help="process skills_inbox into the catalog manifest")
    args = parser.parse_args(argv)

    if args.version:
        print(f"researchforge {__version__}")
        return 0
    if args.command == "recommend":
        return _cmd_recommend(args.path)
    if args.command == "run":
        return _cmd_run(args.path, args.analysis)
    if args.command == "ingest":
        return _cmd_ingest()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
