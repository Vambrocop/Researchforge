# ResearchForge Backlog（Opus 历史分支补强扫描，2026-06-12）

> 来源：Opus 对已合并执行器分支的 supplement sweep。优先级 P1>P2>P3。
> 每条 = 问题 + 一句理由 + 涉及文件。

## P1 — 尽快修（静默错误结果 / 正确性风险） ✅ 已完成（2026-06-12，Opus 双审 LGTM）

- [ ] **`_regression` 第一个连续列被当因变量**（ols/panel_fixed_effects/did）：多连续列数据上会静默选错 DV（同随机森林修过的那类 bug）。→ 加目标列选择透明 + 更稳的选法。`executor/run.py`
- [ ] **`group_comparison` 可能拿 `unit` id 当分组变量**：未排除 `unit_col`/`time_col`，高基数 id 当组 → 几十个单例组的无意义 ANOVA。`executor/run.py`
- [ ] **`logistic_regression` 拿第一个二值当结果**：而二值通常是处理标志（`treatment_candidates` = 所有二值列）→ 把处理当结果回归。`executor/run.py`
- [ ] **`iv_regression` 在目录里却无执行器**：落到 else 占位、空跑，却仍被推荐为可行 → 误导、伤信任。→ 接执行器，或在接好前从推荐中 gate 掉。`executor/run.py`

## P2 — 应修 ✅ 完成（2026-06-12）：透明(随P1) / 零预测警告 / 高基数提示 / 命名统一 / did 处理检测(within-unit, Opus 双审 LGTM)

- [ ] 各分支**不透明**：未在 report 说明"选了哪列当结果/目标"（仅 RF 有注释）。→ summary 加一行"因变量/目标列选择"。`executor/run.py` + `_report`
- [ ] **零解释变量回归**静默拟合截距模型（`~ 1`）→ 应警告"无可用解释变量"。`executor/run.py`
- [ ] **did 的处理变量**取自"所有二值列"→ 因果估计可能挂错变量。→ 收紧处理检测（组×期交互/名称启发）。`profiler/profile.py` + `run.py`
- [ ] **descriptive_stats 无高基数守护**（`describe(include="all")` + min_rows 1）→ 宽表慢且不可读；profiler 已检测 high_cardinality 却未用。`executor/run.py`
- [ ] **estimates 键 / 产物命名不一致**（`feature_importance.png` 单数 vs `feature_importances.csv` 复数；键约定各异）→ 下游(benchmark/报告)需可预测。`executor/run.py`

## P3 — 锦上添花

- [ ] 给 logistic/group_comparison/arima/random_forest 补**执行器级单测**（达到 kmeans 的覆盖标准，尤其结果选择边界）。`tests/`
- [ ] **group_comparison 空组会在 scipy 内部报错**（无 per-branch try/except，与其他分支不一致）。`executor/run.py`
- [ ] **infer_kind**：任意两值文本→binary、全唯一文本→id，会放大上面的选择 bug。`profiler/types.py`
- [ ] **arima 硬编码 order=(1,1,1) + 固定 10 期**，无平稳性检验。→ auto_arima/ADF 选阶。`executor/run.py`
