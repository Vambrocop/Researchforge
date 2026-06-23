# ResearchForge 路线图（Roadmap）— 活文档

> **1.0 不是终点,是里程碑。** 北极星 = 方法学大杂烩 + 越来越聪明的自动选模 —— 永不"完工",
> **持续优化是常态**;版本号只是路上的桩。当前 **v0.7**。本文件是活的:做完一档就更新、补新方法波。
>
> **路线选择:厚 1.0（用户 2026-06-16 选 B,大格局派)** —— 1.0 不只"能用",还要"方法够全"。
> 所以 1.0 前不止做 UI/鲁棒,还把下面方法波的高需求部分(实验设计/测量/因果/ML)做进去;
> **慢慢优化,不赶**。
>
> - 一眼现状:`py -3 -m researchforge.cli status`（显示当前版本 + 下一里程碑）
> - 战术「下一波」:`docs/deferred-log.md` 顶部 / 记忆 `next-batch`
> - 本文件 = 战略里程碑阶梯(往哪走、为什么)

## 运营模式（持续优化环 ∞）
挑一项 → 按宪法实现(`branches/<family>.py` 处理器 + 测试 + 真推断派 `inference-reviewer` 双审)
→ `cli scorecard --save` 记录提升 → push(用户说「今天 ok」)→ 重复。
**自评分卡趋势线 = "一直优化"的度量。**

## 通往「厚 1.0」的路（B 方案,不赶,慢慢做）

### 阶段 ① v0.8 — 产品壳（让人能用）✅ 达成 2026-06-23
- [x] **Web 前端 MVP**:上传 → 推荐(🟢🟡🔴 + 6 维评分卡) → 跑 → 报告/图。`web/static/index.html` 单页应用 + `web/app.py`(analyze/clean/reanalyze/run/download) + `cli web` 启动 + test_web 9 绿；`/api/run` 已接 config 覆盖。**可用性 58→83、总分→93**（旧 scorecard 探测错路径 templates/→已改 static/index.html）。

### 阶段 ② v0.9 — 硬化 + 触达 ✅ 达成（2026-06-23，引擎 v0.9.0）
> 主体达成：脏数据鲁棒读取 + README/pip + 推断 backlog + 机器可读 config schema。剩 discover phase-2（自标暂缓）与 config 回填（~30 族）作为持续优化。
- [x] **真实脏数据鲁棒**(编码/分隔符/数值强转/缺失标记/高基数列)—— 1.0 隐形门槛。**2026-06-23**:`profiler/ingest.py` 鲁棒读取门(utf-8-sig/gb18030/latin-1 编码回退 + `,;\t|` 分隔符嗅探 + **保守数值强转**:文本列"1,234"/"$5"/"12%"/杂缺失标记 ≥90% 可解析才转、记 `df.attrs['rf_coercions']` 非静默)；`diagnose` 加 coerced_numeric/high_cardinality 披露；12 个真实脏数据测试端到端不崩；全量回归护好。〔余:i18n 小数逗号/charset 检测器/不规则行容错见 deferred-log〕
- [x] 用户 **README + pip 打包**(2026-06-23 波⑦:README 重写 + pyproject 动态版本/scripts/extras/package-data)
- [ ] **discover 真抓取**(阶段2):`fetch_fn` 接 CRAN/PyPI/GitHub,带降级（roadmap 自标"暂缓:选得准>加得多",优先级最低）
- [x] **推断 backlog**:DML/causal_forest CV R² + 多重比较校正、BART holdout R²、GAMM 非高斯（#4a-#4d 已做）
- [x] **config schema(机器可读,每 analysis entry)** —— 2026-06-23：`AnalysisEntry.params: list[ParamSpec]` + `config_schema.validate_config`(非阻塞警告) + run 接入 + `cli params` + recommend payload；correlation_suite/effect_sizes 范例已填，其余 ~30 族渐进回填（Web UI / 推荐 / 运行校验共一份规格）

### 阶段 ③ 方法熔炉扩张（折入 1.0,使其"厚"）
高需求方法族,多为纯 Python(`/add-analysis` + 双审):
- [ ] **实验设计/经典统计**:ANOVA(单/双/重复测量)、ANCOVA、MANOVA、功效/样本量、Tukey、卡方/Fisher、非参(KW/Friedman)
- [ ] **设计感知田间试验**(农学高频、误分析风险高,Codex 审提):RCBD、split-plot/split-split、nested、repeated —— 做成**强制声明 block/plot/subplot/repeated 角色的 mixed-model wrapper**,而非裸 ANOVA
- **★ 新能力·北极星:实验设计顾问 / DoE**(用户 2026-06-16 提):**数据前** —— 给因子/水平/约束 → 推荐设计 + 算重复数/功效 + 出随机化布局 + 配套分析模板。新模式(问题→设计)。
  - [x] **首切片已上线(2026-06-16)**:`power_analysis`(所需样本量,避事后功效坑)+ `researchforge.design` 模块 + **`cli design` 命令**(rcbd/factorial/latin_square 随机化布局,确定性可复现,指向对应分析)。
  - [ ] 续:从数据画像**自动推荐**设计、analysis-template 串联、Web 表单入口、更多设计(split-plot 布局/不完全区组/响应面)。
- [ ] **测量/信度**:Cronbach α、ICC、Cohen's κ、Bland-Altman、IRT/Rasch
- [ ] **因果扩张**:模糊 RDD、事件研究、交错 DiD(Callaway-Sant'Anna)、合成 DiD、2SLS/IV(现占位)、PSM/IPW/AIPW
- [ ] **ML/预测**:lasso/ridge/elasticnet、SVM、梯度提升、SHAP、Prophet/ETS、GARCH、changepoint、UMAP/t-SNE

### 🎯 v1.0 — 可用 + 综合 + 稳定 + 有文档（里程碑,非终点）
研究者拿真实数据端到端出可信报告;**方法库已大幅扩充(~100+ 法)**;稳定、错误处理、文档齐。

## 1.0 之后(继续慢慢优化)

### v1.1 → v1.5 — 余下方法波 + 选模变聪明
- [ ] **贝叶斯/生存/缺失**:贝叶斯回归·分层(待 Stan/JAGS)、竞争风险、时变协变量、MICE 多重插补
  - **领域聚焦(用户 2026-06-16 定)**:主战场 = **生态 / 农学 / 经济 / 环境 / 社科**;**医学暂缓**(MICE/竞争风险/生存缺失等临床向后推,不进 1.0 前序)。方法扩张优先这五域高频法。
- [ ] **文本/网络/空间**:LDA 主题、情感、ERGM、空间面板、GWR、网络 meta
- [x] **快速选择器已上线(2026-06-16)**:目标感知 `recommend --goal <compare/relate/causal/…>` → 聚焦 top-N + 讲清"为什么"(14 目标分类 `recommender/goals.py` + `select_top`)。治"方法多到不知道用啥"。
- [ ] **自动选模更聪明(续)**:precondition/严谨度规则细化、按数据画像直接荐目标、Web UI 接选择器。
- [ ] ~~自我进化 discover 真抓取~~ **暂缓(用户 2026-06-16 定)**:选得准 > 加得多;75 法对五域够用,等选择器+Web 成形再回头。

### v2.0 — 愿景达成（仍不是终点）
真·大杂烩 + 极聪明自动选模 + 趋势引擎闭环(真实流行度/更新喂回评分卡)。

### v2.0+ — 持续优化（∞）
方法继续加、选模继续变聪明、UX/性能/真实数据覆盖继续打磨。**永远在优化。**

## 横切的持续优化（任何版本都在做）
- **性能**:R 测试更快(已 `-n 2`;可缓存 R 进程)、分析提速
- **鲁棒**:真实数据边界、友好错误、`config` schema 校验
- **质量护栏**:每真推断方法双审、模块 ≤1500 行护栏、评分卡逐版追踪
