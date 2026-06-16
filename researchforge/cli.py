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


def _cmd_scorecard(save: bool = False) -> int:
    import datetime as _dt
    from pathlib import Path

    from researchforge.quality import compute_scorecard

    sc = compute_scorecard()
    print(f"ResearchForge 项目自评分卡 — 总分 {sc.overall}/100\n")
    print(sc.table())
    if save:
        doc = Path(__file__).resolve().parent.parent / "docs" / "scorecard.md"
        order = ["completeness", "correctness", "rigor", "honesty", "design",
                 "novelty", "performance", "usability"]
        header = (
            "# 项目自评分卡历史（Project Scorecard History）\n\n"
            "> `cli scorecard --save` 每次追加一行；分数随项目改进而动，用于追踪提升。\n\n"
            "| 日期 | 总分 | 完整性 | 准确性 | 严谨 | 诚实 | 设计 | 新颖 | 快速 | 可用 | 方法数 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        if not doc.exists():
            doc.write_text(header, encoding="utf-8")
        row = (
            f"| {_dt.date.today().isoformat()} | **{sc.overall}** | "
            + " | ".join(str(sc.dimensions[k]) for k in order)
            + f" | {int(sc.metrics['n_methods'])} |\n"
        )
        with doc.open("a", encoding="utf-8") as f:
            f.write(row)
        print(f"\n已追加到 {doc}（版本历史，可看趋势）。")
    else:
        print("\n（加 --save 把这次评分追加进 docs/scorecard.md 版本历史）")
    return 0


def _cmd_status() -> int:
    """Live project front-door: health + scale + git + next-up + what-to-improve, all
    computed from current repo signals so it never goes stale. Run it first thing."""
    import re
    import subprocess
    from pathlib import Path

    from researchforge import __next_milestone__, __version__
    from researchforge.quality.scorecard import DIM_LABELS, MODULE_LINE_LIMIT, compute_scorecard

    repo = Path(__file__).resolve().parent.parent
    sc = compute_scorecard()
    m = sc.metrics
    weak = sorted(sc.dimensions.items(), key=lambda kv: kv[1])

    def _git(*a: str) -> str:
        try:
            r = subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, timeout=10)
            return r.stdout.strip()
        except Exception:
            return ""

    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    dirty = bool(_git("status", "--porcelain"))
    _u = _git("rev-list", "--count", "@{u}..HEAD") or _git("rev-list", "--count", "origin/main..HEAD")
    n_unpushed = int(_u) if _u.isdigit() else None

    nexts: list[str] = []
    dlog = repo / "docs" / "deferred-log.md"
    if dlog.exists():
        seg = dlog.read_text(encoding="utf-8").split("🔜 下一波")
        if len(seg) > 1:
            block = seg[1].split("\n## ")[0]
            nexts = re.findall(r"^\d+\.\s*\*\*(.+?)\*\*", block, re.M)

    print("ResearchForge — 状态速览  (researchforge status)")
    print("=" * 56)
    print(f"里程碑 引擎 v{__version__} → 下一站 {__next_milestone__}  (路线图 docs/roadmap.md)")
    print(f"健康  总分 {sc.overall}/100   最弱: "
          + " · ".join(f"{DIM_LABELS[k]} {v}" for k, v in weak[:3]))
    print(f"规模  {int(m['n_methods'])} 方法 / {int(m['n_families'])} 族 / "
          f"{int(m['n_test_files'])} 测试文件 / 最大模块 {int(m.get('max_module_lines', 0))} 行 (护栏 {MODULE_LINE_LIMIT})")
    print(f"Git   分支 {branch} · 未推送 {n_unpushed if n_unpushed is not None else '?'} · 工作树 {'有改动' if dirty else '干净'}")

    print("\n🔜 下一波 (docs/deferred-log.md):")
    if nexts:
        for i, t in enumerate(nexts, 1):
            print(f"  {i}. {t}")
    else:
        print("  见 docs/deferred-log.md 顶部「🔜 下一波」/ 记忆 next-batch")

    print("\n需改进 (自动探测):")
    for k, v in weak[:2]:
        print(f"  - {DIM_LABELS[k]} {v} —— {sc.notes[k].split('；')[0].split('（')[0]}")
    warn_at = int(MODULE_LINE_LIMIT * 0.8)
    for rel, n in sc.large_modules:
        print(f"  - {'⚠ 逼近' if n >= warn_at else '留意'}护栏: {rel} ({n}/{MODULE_LINE_LIMIT} 行)")
    if n_unpushed:
        print(f"  - {n_unpushed} 个 commit 未 push（用户说『今天 ok』才推）")
    if dirty:
        print("  - 工作树有未提交改动")

    print("\n提速: 全量 `pytest -n 2` · 快循环 `pytest -m \"not slow\"`（别用 -n auto，R worker OOM）")
    print("加分析: 进 branches/<family>.py 的 @register；真推断派 inference-reviewer 双审；push 等『今天 ok』")
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
    scd = sub.add_parser("scorecard", help="project self-assessment scorecard (versioned)")
    scd.add_argument("--save", action="store_true", help="append this score to docs/scorecard.md history")
    promo = sub.add_parser("promote", help="promote a ready candidate into the live catalog")
    promo.add_argument("candidate_id", help="candidate analysis id")
    web_p = sub.add_parser("web", help="launch the ResearchForge web UI")
    web_p.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    sub.add_parser("status", help="live front-door: health + next-up + what to improve (run first)")
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
    if args.command == "scorecard":
        return _cmd_scorecard(args.save)
    if args.command == "promote":
        return _cmd_promote(args.candidate_id)
    if args.command == "web":
        return _cmd_web(args.port)
    if args.command == "status":
        return _cmd_status()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
