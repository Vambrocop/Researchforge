# ResearchForge — 骨架阶段设计文档（Spec v0.1）

- 日期：2026-06-12
- 状态：待用户评审
- 范围：本文档只定义**骨架/MVP 阶段**。完整多领域平台是愿景，分期实现（"先骨架，后慢慢填满"）。

## 1. 背景与愿景

**一句话定位**：给 ResearchForge 一份数据，它自动读懂数据、推荐能做的研究分析、由你选定、再自动执行并出图出报告。

**核心体验循环**：
> 自动识别数据 → 自动推荐分析可能性 → 你来选 → 自动执行 + 自动作图出报告

**普适性来源**：分析的路由由**数据类型/结构**驱动，而非写死某个学科——所以天然能跨学科泛化、可"一点点加领域"。

**起步领域**：农业经济学 / 通用统计学（已有强执行器）。后续扩：土壤学、生态学、微生物学。

**最终形态**：网页应用（换任何电脑/设备开浏览器即用）。
**当前形态（本期）**：可移植的 **Git 仓库 + Python 引擎**，经 Claude Code 驱动。网页是后续阶段的增量。

## 2. 架构：三层 + 两个横切机制

```
                ┌─────────────────────────────────────────┐
   数据文件 ──▶ │ ① Profiler 数据画像                       │
  (CSV/Excel)   │   输出: DataFingerprint (结构化指纹 JSON)  │
                └───────────────────┬─────────────────────┘
                                    ▼
                ┌─────────────────────────────────────────┐
                │ ② Recommender 推荐（混合路线 C）          │
                │   规则目录匹配 + LLM 排序/解释             │
                │   输出: 排序的 AnalysisRecommendation[]    │
                └───────────────────┬─────────────────────┘
                         你来选 ◀────┤
                                    ▼ (选中项)
                ┌─────────────────────────────────────────┐
                │ ③ Executor 执行                           │
                │   调对应 skill/脚本 → 表 + 图 + 报告        │
                └─────────────────────────────────────────┘

  横切 A: Skill Ingestion（随时添加）   横切 B: Portability（便携性）
```

### 各层职责与接口契约

- **① Profiler**：输入数据文件，输出 `DataFingerprint` —— 一份结构化指纹：变量列表与类型（连续/分类/计数/二值/日期/ID/地理坐标）、缺失情况、是否存在 `单位×时间` 面板结构、是否存在处理组/对照组、样本量、分布概况。**纯只读、确定性、不调 LLM。**
- **② Recommender（混合 C）**：输入 `DataFingerprint`（+ 可选研究问题），输出排序的 `AnalysisRecommendation[]`，每条含：方法名、前提是否满足、预期产出、对应执行器引用、推荐理由。规则目录给候选与硬前提，LLM 负责排序、解释、补目录未覆盖的情况。**每条推荐必须透明可解释。**
- **③ Executor**：输入选中的推荐 + 数据，调对应 skill/脚本执行，产出表格 + 图 + 报告。本期复用 `empirical-analysis-python`。

### 横切机制

- **A. Skill Ingestion（随时添加）**：`skills_inbox/` 投递 → 读 `SKILL.md` → 提炼（领域/方法/前提/产出）→ 登记进分析目录（catalog）→ 原件归档 `_processed/`。让"随时加 skill"成为一等公民。
- **B. Portability（便携性）**：一切进 Git 仓库；引擎依赖的 skill 通过 `setup` 脚本安装到 `~/.claude/skills/`（或仓库内打包）；换电脑 = `git clone` + `setup`。

## 3. 推荐引擎：混合路线 C 怎么落地

- **规则目录（catalog）**：一组结构化条目（YAML/JSON），每条 = `{method, domain, preconditions, produces, executor_ref}`。
- **匹配**：用 `DataFingerprint` 比对每条的 `preconditions` → 得到候选集（硬前提保证不推荐数据不支持的方法）。
- **LLM 层**：对候选排序、生成人话解释、并对目录未覆盖的情况给补充建议（标注"非目录项"）。
- **可靠性兜底**：硬前提由规则把关，LLM 只在已通过前提的候选上做软判断，降低乱推风险。

## 4. 本期范围（骨架 MVP）

**做（In Scope）**：
1. 仓库结构 + `setup` 脚本骨架（依赖与 skill 安装）。
2. `DataFingerprint` schema + Profiler（先支持 CSV/Excel 表格类）。
3. Catalog 数据结构 + 农业计量的**初始条目**（面板/截面/DID/固定效应/IV 等少量）。
4. Recommender（规则匹配 + LLM 排序），输出可解释推荐菜单。
5. **一条完整纵切跑通**：省份×年份面板 CSV → 画像 → 推荐 → 选「双向固定效应/DID」→ 调 `empirical-analysis-python` 执行 → 出回归表 + 图 + 简报。
6. Skill ingestion 跑通一次（投一个 skill → 进 catalog → 推荐中可见）。

**不做（Out of Scope，后续阶段）**：
- 网页前端与云部署；
- 生态/土壤/微生物的原始数据执行器（需封装 vegan/lme4/agricolae/phyloseq 等）；
- 空间数据、文本/文献语料的画像；
- 大而全的多领域目录。

## 5. 技术选型

- **引擎核心**：Python 包（`profiler` / `catalog` registry / `recommender`）。
- **执行**：复用 `empirical-analysis-python`（Python 计量执行器）；后续接 R 执行器。
- **编排入口**：Claude Code 的 master skill / CLI 命令，把 ①②③ 串起来。
- **LLM 推理**：经 Claude（推荐层的排序与解释）。

## 6. 成功标准（可验证）

1. 丢一份省份×年份面板 CSV 进去 → 引擎正确识别为**面板数据**（识别出单位、时间、可能的处理变量）。
2. 列出 **≥3 个可行分析**，每个都说明"前提是否满足 + 预期产出"。
3. 选定一个 → 自动产出**回归表 + 图 + 简报**（复用 empirical-analysis-python）。
4. 在**另一台电脑** `git clone` + `setup` 后，能复现上述结果。
5. 往 `skills_inbox/` 投一个新 skill → 能被登记进 catalog 并在后续推荐中可见。

## 7. 风险与未决

- **R 环境便携性**比 Python 难（生态/土壤执行器用 R 时要处理）——本期不涉及，后期再定。
- **skill 打包**的体积/许可：本期用 setup 脚本安装而非内嵌，规避。
- **LLM 推荐可靠性**：靠规则硬前提兜底；推荐菜单始终展示前提满足情况，人来终审。

## 8. 相关文档

- 能力盘点与缺口：`docs/analysis-catalog/skills-inventory.md`
- skill 投递协议：`skills_inbox/README.md`
