# ResearchForge 路线图（Roadmap）— 活文档

> **1.0 不是终点,是里程碑。** 北极星 = 方法学大杂烩 + 越来越聪明的自动选模 —— 永不"完工",
> **持续优化是常态**;版本号只是路上的桩。当前 **v1.0.0**。本文件是活的:做完一档就更新、补新方法波。
>
> **🎯 v1.0.0 已达成（2026-06-24）**：可用(CLI+Web+Python API) + 综合(225 法 / 33 族) + 稳定
> (987 测绿 + 4 真实公开数据集端到端验证零崩溃) + 有文档(README + USER_GUIDE + loop-decisions + roadmap)。
> **1.0 后持续优化**：更聪明自动选模(语义化 outcome 检测,治整数目标被判 count 的默认坑)、Web 纵深、
> discover phase-2、更多方法族(UMAP/ERGM/AIPW/Bayesian)。
> **已达成（v0.9.0，2026-06-23）**：脏数据鲁棒、config schema+CI guard、ML 可解释性、空间依赖、中介扩展、设计感知田间试验。
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
- [x] **discover 真抓取**(阶段2) —— **2026-06-25 做了**：`catalog/trends.py` 活的趋势引擎——`fetch_trend` 真抓 PyPI(下载量)/CRAN(在册)/GitHub(星标+活跃,按名搜索带含义护栏)→ 0-100 `momentum`(真数字、编辑权重、log 归一)。`build_live_fetch_fn` + `score_candidate` 把 momentum 融进发现优先级；`cli discover --live` 重排 + 写趋势快照(`~/.researchforge/`)；`score_method` 读快照精修 popularity（**热路径只读缓存、零网络**，缺则回退编辑先验并诚实披露）。可选 + 优雅降级（离线/无 requests/限流→回退），缓存 7d。镜像 R 桥模式、绝不进分析运行时。13 个 mock-HTTP 确定性测试 + 真网 smoke(可跳) + conftest hermetic fixture。**闭环：真实流行度/活跃度 → 趋势引擎 → 评分卡**。
- [x] **推断 backlog**:DML/causal_forest CV R² + 多重比较校正、BART holdout R²、GAMM 非高斯（#4a-#4d 已做）
- [x] **config schema(机器可读,每 analysis entry)** —— 2026-06-23：`AnalysisEntry.params: list[ParamSpec]` + `config_schema.validate_config`(非阻塞警告) + run 接入 + `cli params` + recommend payload；correlation_suite/effect_sizes 范例已填，其余 ~30 族渐进回填（Web UI / 推荐 / 运行校验共一份规格）

### 阶段 ③ 方法熔炉扩张（折入 1.0,使其"厚"）
高需求方法族,多为纯 Python(`/add-analysis` + 双审):
- [x] **实验设计/经典统计**:ANOVA(单/双/重复测量)、ANCOVA、MANOVA、功效/样本量、卡方/Fisher、非参(KW/Friedman) —— 已上线 `anova_oneway`/`ancova`/`repeated_measures_anova`(experimental_stats)、`manova`(multivariate)、`power_analysis`(experimental_design)、`chi_square_test`/`fisher_exact`(categorical_tests)、`kruskal_wallis`/`friedman_test`(nonparametric_tests)。〔Tukey 事后多重比较仍可作后续细化〕
- [x] **设计感知田间试验**(农学高频、误分析风险高,Codex 审提):RCBD、latin-square、split-plot —— 已上线 `rcbd_anova`/`latin_square_anova`/`split_plot_anova`(field_trials.py),声明 block/plot/subplot 角色的设计感知 ANOVA;并配套 `experimental_design/` 布局族(rcbd/latin_square/split_plot/factorial_anova/response_surface/ammi/gge_biplot)。〔nested/split-split 仍可续扩〕
- **★ 新能力·北极星:实验设计顾问 / DoE**(用户 2026-06-16 提):**数据前** —— 给因子/水平/约束 → 推荐设计 + 算重复数/功效 + 出随机化布局 + 配套分析模板。新模式(问题→设计)。
  - [x] **首切片已上线(2026-06-16)**:`power_analysis`(所需样本量,避事后功效坑)+ `researchforge.design` 模块 + **`cli design` 命令**(rcbd/factorial/latin_square 随机化布局,确定性可复现,指向对应分析)。
  - [x] 续:更多设计(split-plot 布局/响应面/factorial)已上线(`experimental_design/` 布局族);Web 表单入口已接(config 表单)。〔余:从数据画像**自动推荐**设计、analysis-template 串联、不完全区组仍待做〕
- [x] **测量/信度**:Cronbach α、McDonald ω、ICC、Cohen's κ、Bland-Altman、IRT/Rasch —— 已上线 `cronbach_alpha`/`mcdonald_omega`/`icc`(psychometrics)、`cohens_kappa`/`bland_altman`(agreement)、IRT 全家(`irt_2pl`/`irt_rasch`/`irt_grm`/`irt_pcm`)。
- [x] **因果扩张(部分)**:模糊 RDD、事件研究、交错 DiD(Callaway-Sant'Anna)、PSM/IPW —— 已上线 `fuzzy_rdd`/`event_study`(causal/)、`callaway_santanna`(causal_did)、`psm`/`ipw`(causal/)。〔仍开:合成 DiD(需编译器)、2SLS/IV 占位转实、AIPW —— 见 1.0 之后/暂缓〕
- [x] **ML/预测(部分)**:lasso/ridge/elasticnet、SVM、梯度提升、SHAP、ETS、GARCH、changepoint、t-SNE —— 已上线 `regularized_regression`/`svm_model`/`gradient_boosting`(ml_supervised)、`shap_values`(interpretability)、`exponential_smoothing`(forecasting)、`garch`/`structural_breaks`(timeseries,后者即 changepoint)、`tsne`(dimensionality_extra);ML 可解释性另含 `partial_dependence`(PDP)/`accumulated_local_effects`(ALE)/`feature_interaction`(H 统计)/`surrogate_model`/`quantile_intervals`。〔仍开:Prophet、UMAP —— 见暂缓清单〕

### 🎯 v1.0 — 可用 + 综合 + 稳定 + 有文档（里程碑,非终点）✅ 达成 2026-06-24（v1.0.0）
研究者拿真实数据端到端出可信报告 ✅(tests/test_end_to_end.py 在 sklearn diabetes/breast_cancer、statsmodels co2/grunfeld 四真实数据集跑通管线、26+ 分析零崩溃);**方法库 225 法 / 33 族 / 987 测绿**(远超原定 ~100+);稳定 + 诚实错误处理 + 文档齐(README + docs/USER_GUIDE.md)。**1.0 后续(持续优化,非门槛)**:Web 前端深度、discover phase-2(自标暂缓)、更聪明自动选模。

## 1.0 之后(继续慢慢优化)

### v1.1 → v1.5 — 余下方法波 + 选模变聪明
- [~] **贝叶斯/生存/缺失**:**贝叶斯回归·分层(2026-06-25 做了)** —— 原标"待 Stan/JAGS",**现代 PyMC NUTS 无需编译器**(Windows 无 g++ 也秒级采样),`bayesian_mcmc.py`:`bayesian_regression`/`bayesian_logistic_regression`/`bayesian_hierarchical`(变截距部分汇集),HDI/R-hat/ESS/ICC,可选+优雅降级。〔共轭先验贝叶斯 `bayesian_ab_test`/`bayesian_proportion`/`bayesian_poisson_rate` 仍在〕。**仍开**:竞争风险(已有 competing_risks)、时变协变量(已有 time_varying_cox)、MICE 多重插补、贝叶斯 SEM(blavaan)。
  - **领域聚焦(用户 2026-06-16 定)**:主战场 = **生态 / 农学 / 经济 / 环境 / 社科**;**医学暂缓**(MICE/竞争风险/生存缺失等临床向后推,不进 1.0 前序)。方法扩张优先这五域高频法。
- [~] **文本/网络/空间**:空间侧已大幅上线 —— GWR(`gwr`)、空间依赖(`moran_i`/`local_moran`/`getis_ord_gi`/`getis_ord`/`bivariate_moran`/`local_geary`/`skater`/`ripleys_k`/`join_count`)、空间回归(`spatial_regression`)、网络科学(`community_detection`/`centrality_suite`/`epidemic_model`);**仍开**:文本挖掘(LDA 主题/情感)、ERGM、网络 meta。
- [x] **快速选择器已上线(2026-06-16)**:目标感知 `recommend --goal <compare/relate/causal/…>` → 聚焦 top-N + 讲清"为什么"(14 目标分类 `recommender/goals.py` + `select_top`)。治"方法多到不知道用啥"。
- [ ] **自动选模更聪明(续)**:precondition/严谨度规则细化、按数据画像直接荐目标、Web UI 接选择器。
- [x] **自我进化 discover 真抓取(2026-06-25 解禁并完成)**:原暂缓("选得准>加得多")——现选择器+Web 已成形，回头做了。`catalog/trends.py` 真抓 PyPI/CRAN/GitHub→momentum 喂发现优先级 + 评分卡 popularity（详见阶段②）。**趋势引擎闭环（v2.0 愿景的一块）已落地雏形。**

### v2.0 — 愿景达成（仍不是终点）
真·大杂烩 + 极聪明自动选模 + 趋势引擎闭环(真实流行度/更新喂回评分卡)。

### v2.0+ — 持续优化（∞）
方法继续加、选模继续变聪明、UX/性能/真实数据覆盖继续打磨。**永远在优化。**

## 横切的持续优化（任何版本都在做）
- **性能**:R 测试更快(已 `-n 2`;可缓存 R 进程)、分析提速
- **鲁棒**:真实数据边界、友好错误、`config` schema 校验
- **质量护栏**:每真推断方法双审、模块 ≤1500 行护栏、评分卡逐版追踪
