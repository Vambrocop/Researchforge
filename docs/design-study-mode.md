# Study Mode 设计规格（Wave I）— Fable 5 定稿 2026-07-06

> 北极星的产品形态收口：`丢数据 → 一份诚实的合并研究报告`。所有零件已存在
> （select_top / run_analysis / --clean / report_narrative / scorecard / manifest 思路），
> 本波只加**编排层**。设计判断已全部做完——按本文实现，别重新发明；
> 实现中若某决策与现实冲突，按「STOP 点」处理而不是自行改设计。

## 1. 入口

- CLI：`py -3 -m researchforge.cli study <data> [--goal X] [--top K] [--clean] [--config JSON]`
  - `--top` 默认 **3**（实质方法数，不含第0节的 descriptive 基线）。
  - `--clean` 语义与 `run --clean` 完全一致（可选、默认关、复用 `_auto_clean_before_run`）。
  - `--config` 全局应用到每个方法（outcome/predictors 这类键跨方法同义）。
    **v1 不做 per-method config**（记 deferred；需要时再加 `{"per_method": {id: {...}}}`）。
- Web：`POST /api/study`（file_id + goal/top/clean/config），返回 study 目录 + 报告文本。
  实现放 `researchforge/study.py`（新顶层模块，CLI 与 web/service 都薄封装调它）。

## 2. 编排流（run_study(fp_path, goal, top, clean, config) -> StudyResult）

1. profile → 质量 nudge；`--clean` 开且有实质步骤 → 应用+披露+重新画像（同 run --clean）。
2. `select_top(fp, goal=goal, top=12, diagnostic_aware=True)` 取候选池。
3. **多样性过滤**（定案）：按现有排序贪心，每 family 最多取 1 个，直到凑满 K；
   若 family 数不足 K，再按 fit 次序回填第二名。`descriptive_stats/correlation/
   correlation_matrix/summary_statistics`（_PICK_SKIP 集）不占 K。
4. **第0节基线**：无条件先跑 `descriptive_stats`（不计入 K，报告作 §0 数据概览）。
5. 逐方法 `run_analysis(fp, entry, output_root=<study_dir>, config=config)`，
   **每个包 try/except**：单方法失败 → 该节写「失败+原因」，study 继续；
   全部 K 失败 → 报告仍写出，CLI 退出码非 0。绝不编造。
6. 写 `study_meta.json`（引擎版本 / 数据文件 sha256 / n_rows×n_cols / goal / K /
   实际所跑方法列表及其 run 目录 / config / 清理是否应用 / 时间戳）——
   这是 roadmap ⑤ run manifest 的 study 级先行版（per-run manifest 仍是独立小项）。
7. 产出目录：`outputs/<ts>_study/`，内含 `study_report.md` + `study_meta.json` +
   各方法自己的常规 run 子目录（复用现有命名，零改动 run_analysis）。

## 3. 报告结构（study_report.md，定案）

- **§0 数据与质量**：行×列 / 结构（面板/时序/geo）/ 角色提示（likely_outcome+置信度+依据）/
  质量发现与清理披露（若 --clean 应用了，逐条 ✓/⚠）。
- **§选法依据**：K 方法一览表——严谨度灯 / fit / 一行"为何选它"（命中的诊断 finding；
  无诊断则写 family 亲和依据）。
- **§1..K 每方法一节**：run 的 summary 原文（自带全部 ⚠ 披露，**不改写不美化**）/
  关键数值表（estimates 按现有 salience 规则取前若干）/ 图（相对路径嵌入）/ 产物清单。
  失败方法：节保留，写失败原因 + 指引（缺包/前提不满足时给替代建议）。
- **§跨方法收敛信号**（诚实规则，不硬凑可比性）：仅当 ≥2 方法产出**同名 estimate 键**
  （如同一预测变量系数）时，报告符号/量级是否一致；否则明说
  「各方法回答不同问题，不做数值横比」。**纯规则实现，运行时零 LLM 零联网**。
- **§方法学附录**：每方法 6 维评分卡 + catalog biases + 严谨度 note。
- **§披露汇总**：全文所有 ⚠ 行的聚合清单（自动 grep 各节）。

## 4. 测试计划

- e2e ×3 合成数据（回归形 / 二值结果 / 计数结果）：study 跑通、report 存在、
  §0+K 节齐、meta.json 键齐、多样性约束成立（无同 family 重复，除非 family 不足）。
- **失败注入**：monkeypatch 某 handler 抛异常 → 该节含"失败"、其余节完好、退出码语义对。
- --clean 路径 ×1：脏数据 study，报告 §0 含清理披露。
- 时长控制：合成数据小（n≤200）；若单条 e2e >60s，进 SLOW_MODULES。

## 5. 规模与分工

- `researchforge/study.py`（编排+报告生成，预估 ~350-450 行；**逼近 800 行就把
  report 生成拆 `study_report.py`**）+ cli/web 薄封装 + tests/test_study.py。
- Sonnet 建（本文即规格）→ 主脑亲验（跑 3 个 e2e + 读报告全文）→ fresh 冷审
  （重点：失败语义诚实性 / 收敛信号是否硬凑 / 披露是否完整传递）→ commit。

## 6. STOP 点（建造者遇到即停下汇报，别自行决定）

1. 收敛信号规则在真数据上频繁误报"一致/矛盾" → 停，宁可退化为纯列表不下结论。
2. 多样性过滤与 goal 过滤叠加后凑不满 K → 按实际数出报告并披露，不放宽严谨度灯。
3. 任何需要改 run_analysis 本体才能实现的点 → 停（本波零改 executor 是硬边界）。

## 7. 明确不做（v1 边界）

per-method config / HTML 报告（md 先行，web 端渲染 md 即可）/ LLM 叙事段
（report_narrative 现有规则叙事可复用，不新增智能）/ 并行跑方法（K=3 串行足够，
避免 R 桥并发内存问题——见 pytest -n auto OOM 教训）。
