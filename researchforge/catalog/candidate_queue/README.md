# catalog/candidate_queue — 候选队列（自生长种子）

发现/起草的分析方法先进这里当**候选**，不直接上线。每条候选 = 一个 `AnalysisEntry` + `source`（来源）+ `status` + `notes`。

- `status: pending` —— 已草拟，未验证（执行器没接 / 前提没核实）。
- `status: ready` —— 已配齐前提 + 能跑的执行器 + 测试/benchmark 覆盖（过了 Fable+Opus 双审质量门）。
- `status: rejected` —— 评估后不采纳。

**只有 `ready` 的候选**能用 `promote_candidate(id)` / `rf promote <id>` 提升进正式 catalog（写入 `entries/promoted.yaml`）。这从数据层强制"分析必须能跑才被推荐"。

候选文件放成 `*.yaml`（每个是候选列表）。
