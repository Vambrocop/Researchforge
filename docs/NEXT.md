# ResearchForge — 下次继续（Next session）

> 2026-06-12 收尾时的状态快照 + 下次优先级。新会话说一句"继续 ResearchForge"，我会读 memory + 本文件 + `docs/backlog.md` 接着干。

## 当前状态
- **引擎完整可用**：Profiler（类型/面板/时序识别 + 质量诊断 + 清洗）→ Recommender（🟢🟡🔴 严谨度评审 + 偏差披露 + 知情覆盖）→ Executor（出代码/图/表/报告到 `outputs/`）。
- **catalog 16 个分析**：descriptive_stats · correlation · ols_regression · panel_fixed_effects · did · iv_regression(诚实占位) · logistic_regression · group_comparison · random_forest · xgboost · kmeans_clustering · pca · arima · mixed_effects · poisson_regression · negative_binomial_regression。覆盖 经济计量 / 统计 / 机器学习 / 时序 / 生态(部分)。
- **自生长闭环跑通**：候选队列 `catalog/candidate_queue/` → 质量门（Fable 规格 + Sonnet 实现 + Fable+Opus 双审 + benchmark）→ `rf promote` 上线。已用此流程并入 xgboost / pca / mixed_effects / poisson / NB。
- **CLI**：`rf recommend` / `run` / `ingest` / `benchmark` / `candidates` / `promote`。
- **benchmark** v0.0.1 基线 100% / 100% / 100%（画像 / 推荐 / 估计回收，MAE 0.16）。70 测试全过。
- **质量纪律**：分层模型编排（Fable 计划+复审 · Sonnet 实现 · Haiku 走量 · Opus 给推断方法双审）；自动 push 钩子；跨平台（Win/mac/Linux，用 `py -3`）；中文为主、英文翻译已按"保持精简"删除。
- **backlog**：P1 + P2 已完成；P3（小项）待办。

## 下次优先级（我的建议）
1. **清尾**：接最后一个 pending 候选 **diversity_indices**（Shannon/Simpson 多样性指数，纯 numpy，低风险，Fable 自审）→ promote，清空当前 harvest 队列。
2. **C — 做界面（最高价值）**：把"命令行能用"升级成"点点能用"。先做**网页应用**（上传/选数据 → 看画像+质量诊断 → 红绿灯推荐菜单 → 点选 → 在线看/下载表图报告），或先做"超级 skill"封装。这是最终形态、对你和他人最实用。
3. **扩域 / 自生长继续**：再 harvest 一轮（土壤 / 微生物 / GIS / LCA / 调研问卷 / 系统动力学，均在 spec §8 预留），按"一个个来"过门并入。
4. **P3 收尾（可穿插）**：group_comparison 空组 try/except、infer_kind 文本边界、arima 自动选阶、给老分支补执行器级测试。
5. **可选大件**：把 harvester 做成运行时功能（cron 前沿监视 + GitHub 采集，带"采集代码不自动执行"红线）。

## 入口指引
- 设计定稿：`docs/superpowers/specs/2026-06-12-researchforge-skeleton-design.md`（含 §7.5 自生长闭环、§8 架构预留）
- 实现计划：`docs/superpowers/plans/2026-06-12-researchforge-mvp.md`
- 补强清单：`docs/backlog.md`
