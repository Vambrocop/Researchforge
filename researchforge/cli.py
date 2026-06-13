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
        s = r.score
        print(f"  {mark[r.rigor.light]} [{r.rigor.score:3d}] {r.entry.method} — {r.rigor.note}")
        print(
            f"        方法学评分 总{s.overall} | 契合{s.fit} 流行{s.popularity} "
            f"可发表{s.publishability} 美观{s.aesthetics} 新颖{s.novelty} 难度{s.difficulty}"
        )
    return 0


def _cmd_run(path: str, analysis_id: str, config: str | None = None) -> int:
    import json

    from researchforge.catalog import Catalog
    from researchforge.executor import run_analysis
    from researchforge.profiler import profile_dataset

    cfg = None
    if config:
        try:
            cfg = json.loads(config)
        except json.JSONDecodeError as err:
            print(f"--config 不是合法 JSON：{err}")
            return 1
    fp = profile_dataset(path)
    entry = Catalog.load().by_id(analysis_id)
    if entry is None:
        print(f"未知分析 id：{analysis_id}")
        return 1
    res = run_analysis(fp, entry, config=cfg)
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


def _cmd_benchmark() -> int:
    from researchforge.benchmark import run_benchmark, save_report

    rep = run_benchmark()
    print(f"ResearchForge benchmark v{rep.version}（{rep.n_cases} 例）")
    print(f"  画像准确率   ：{rep.profile_accuracy:.0%}")
    print(f"  推荐命中分   ：{rep.recommendation_score:.0%}")
    print(f"  估计回收通过 ：{rep.recovery_pass_rate:.0%}  (MAE={rep.recovery_mae})")
    print(f"  已存档       ：{save_report(rep)}")
    return 0


def _cmd_discover(persist: bool = False) -> int:
    from researchforge.catalog.discover import discover_candidates

    found = discover_candidates(persist=persist)
    if not found:
        print("自我进化：未发现新方法（候选都已在目录/队列中）。")
        return 0
    print(f"自我进化发现 {len(found)} 个候选方法（按优先级排序，均为 pending，不自动上线）：")
    for m in found:
        b = m.breakdown
        print(
            f"  [{m.priority:3d}] {m.method}（{m.family}）— 新颖{b.get('novelty')}"
            f"/可发表{b.get('publishability')}/流行{b.get('popularity')}"
        )
        print(f"        {m.rationale}  来源: {', '.join(m.sources) or '—'}")
    if persist:
        print("\n已写入候选队列（candidate_queue/discovered.yaml）。"
              "用 `researchforge candidates` 查看；实现+测试后方可 promote。")
    else:
        print("\n（加 --persist 写入候选队列）")
    return 0


def _cmd_candidates() -> int:
    from researchforge.catalog.candidates import load_candidates

    cands = load_candidates()
    if not cands:
        print("候选队列为空。")
        return 0
    print(f"候选 {len(cands)} 条：")
    for c in cands:
        print(f"  - [{c.status}] {c.entry.id}（{c.entry.family}）— 来源:{c.source or '—'}")
    return 0


def _cmd_web(port: int) -> int:
    import uvicorn

    url = f"http://127.0.0.1:{port}"
    print(f"ResearchForge Web UI → {url}")
    uvicorn.run("researchforge.web.app:app", host="127.0.0.1", port=port)
    return 0


def _cmd_promote(candidate_id: str) -> int:
    from researchforge.catalog.candidates import promote_candidate

    try:
        entry = promote_candidate(candidate_id)
        print(f"已提升进正式 catalog：{entry.id}（{entry.method}）")
        return 0
    except Exception as err:
        print(f"提升失败：{err}")
        return 1


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
    run_p.add_argument(
        "--config",
        default=None,
        help='JSON of substantive overrides, e.g. \'{"outcome":"yield","predictors":["rain","fert"]}\'',
    )
    sub.add_parser("ingest", help="process skills_inbox into the catalog manifest")
    sub.add_parser("benchmark", help="score engine quality on known cases")
    sub.add_parser("candidates", help="list catalog candidates (self-growth queue)")
    disc = sub.add_parser("discover", help="self-evolution: discover + score new candidate methods")
    disc.add_argument("--persist", action="store_true", help="write discoveries into the candidate queue")
    promo = sub.add_parser("promote", help="promote a ready candidate into the live catalog")
    promo.add_argument("candidate_id", help="candidate analysis id")
    web_p = sub.add_parser("web", help="launch the ResearchForge web UI")
    web_p.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    args = parser.parse_args(argv)

    if args.version:
        print(f"researchforge {__version__}")
        return 0
    if args.command == "recommend":
        return _cmd_recommend(args.path)
    if args.command == "run":
        return _cmd_run(args.path, args.analysis, args.config)
    if args.command == "ingest":
        return _cmd_ingest()
    if args.command == "benchmark":
        return _cmd_benchmark()
    if args.command == "candidates":
        return _cmd_candidates()
    if args.command == "discover":
        return _cmd_discover(args.persist)
    if args.command == "promote":
        return _cmd_promote(args.candidate_id)
    if args.command == "web":
        return _cmd_web(args.port)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
