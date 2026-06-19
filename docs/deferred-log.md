# 未做 / 降级 / 待优化 日志（Deferred & Optimization Log）

> 这里记录**没做完、降级处理、或受环境限制绕过**的事项，方便后续回头优化。
> 每条：**项** · **为何没做/降级** · **当前可用替代/现状** · **如何补全/优化**。
> 由 AI 自走时持续追加（受 CPU/GPU/装包/后端限制时尤其要写在这）。

## 🔜 下一波（优先级，2026-06-19 更新）

> 新会话/下一波接着做这些（记忆 `next-batch` 也指向这里）。**已完成（别重做）**：run.py 巨石全拆、测试提速（`pytest -n 2`；别用 `-n auto` OOM）、conformal、**因果/计量方法波**（PSM/IPW/event_study/fuzzy_rdd/staggered_did）、**推断 backlog #4a-#4d**（causal_forest BH/FDR、BART holdout R²、DML nuisance CV R²、GAMM 非高斯族）、**A 风格 web 前端**（接引擎 + goals 后端供给）、**时间序列波**（协整+VECM/GARCH/结构突变/STL/ARDL）、**并行波1**（农学 AMMI/GGE/RSM + 空间面板 SAR/SEM/SDM + 生态 Mantel/IndVal/RDA）、**DesignSync**（组件库推 claude.ai/design "ResearchForge UI"）、**并行波2**（生存 competing_risks/parametric_survival/rmst + MCDA entropy_weight/vikor/promethee/ahp + 多元 manova/discriminant/canonical_correlation/hotelling_t2）、**并行波3**（心理测量 cronbach_alpha/mcdonald_omega/icc + 金融 value_at_risk/extreme_value/risk_adjusted_return + 非参 permutation_test/bootstrap_ci/robust_regression + 状态空间 unobserved_components/markov_switching/dynamic_factor）+ **引擎审计**（只读,无正确性 bug/无死代码,注册表↔catalog 100 id 双向对齐）。全部 inference-reviewer 双审或确定性实测。**全量 414 绿（115 方法）**。

1. 🔄 **并行 subagent 集成队列（2026-06-19，compact 后用 `git worktree list` 找回）**：
   - ✅ **设计流（组件库）**：已审+合并+推送（main 6b267b8，web/static/components/ 8 文件）。**待办：DesignSync 推到 claude.ai/design**（需用户登录）。
   - ✅ **农学设计波（已审+整合 6/19）**：`ammi`/`gge_biplot`/`response_surface`（纯 Python，experimental_design.py 1230 行）。inference-reviewer **APPROVE**——手搓 SVD/canonical 数学经对数核验全部正确（AMMI 双中心化+秩 min(g-1,e-1)+ASV 方向；RSM 驻点 -0.5B⁻¹b + Hessian 特征值分类；GGE 列中心化+which-won-where 投影）。已合并 main，全量 292 绿。**剩 nice-to-have（无优先级，必清 backlog）**：(a) AMMI 中 `n_axes>=2` 守卫因前置 g,e≥3 永真、可简化（无害）；(b) AMMI `interaction_norm` 是 `inter` 行 Frobenius 范数（原始交互幅度，与 catalog 标签"interaction-residual norm"一致，仅记录）。
   - ✅ **空间面板（已审+整合 6/19）**：`spatial_panel` SAR/SEM/SDM（R splm，个体 FE within）。inference-reviewer **APPROVE-WITH-FIXES**——核心数学(LeSage-Pace impacts S=(I-ρW)⁻¹(Ib+Wθ)、`mean(rowSums)=1/(1-ρ)` for 行标准化 W、W↔数据单位排序、SDM Durbin、SAR/SDM ρ 提取)实测 R 4.6.0/splm 1.6.5 全部核验正确。**已修 MUST-FIX**：SEM(model="error")的 λ 恒报 0.0000——within FE 下 `m$errcomp` 为 NULL，误差参数 `rho` 实住 `m$coefficients`；改按模型分支提取 + 加 SEM 回收测试。已合并 main，全量 298 绿。**剩 backlog（无优先级，必清）**：(a) SHOULD-FIX：LM lag-vs-error 用非稳健 `lml/lme`（两者对任一依赖都拒），换稳健 `rlml/rlme` 判别更准（仅 advisory，默认按 config）；(b) NICE：lon/lat 无名匹配时回退 `geo[-1]/geo[0]` 可能轴互换（欧氏 k-NN 影响小、可 config 覆盖），可加回退告警；(c) impacts 仅点估计、无模拟 SE（splm impacts() 跨版本碎，已披露"未附 SE/条件于点估计"），补 delta/bootstrap SE 仍待做。
   - ✅ **生态波（已审+整合 6/19）**：`mantel_test`/`indicator_species`（纯 Python）+ `rda`（R vegan）。inference-reviewer **APPROVE**——Mantel r 匹配 scipy 到浮点精度 + 联合行列置换是正确 Mantel 零分布且校准良好；IndVal A×B×100 完美匹配 Dufrene-Legendre(1997)（完美指示种=100、均匀=低）；RDA 端到端匹配直跑 R（total/constrained/F=274/p=.001 完全一致）。已合并 main，全量 307 绿。**已应用 3 个 nice-to-have**：(a) RDA 真给调整 R²(RsquareAdj，原只提示)；(b) Mantel summary 补空间自相关告警；(c) IndVal summary 补稀有类不稳告警。剩：无（#2/#5 经核为非问题）。
   - **整合顺序（全部完成 ✅ 6/19）**：✅① 农学 a4079af → ✅② 空间面板 a0ee0a3 → ✅③ 生态 a5c9ed1。三波 inference-reviewer 双审（空间面板抓 1 个 SEM λ MUST-FIX 已修+回归测试，农学/生态 APPROVE），逐个 merge + 全量绿 + push。全量 283 → **307 绿**（+9 农学 +6 空间 +9 生态）。**剩唯一待办：DesignSync 推组件库到 claude.ai/design（需用户登录）。**
   - 编排约定见记忆 [[design-tooling-and-orchestration]]：worktree 隔离、不相交文件、主脑双审、push 等用户。**红线：未经 inference-reviewer 审的真推断方法不合并/不推。**
1b. ✅ **并行 subagent 集成队列 — 第二批（2026-06-19，全部完成）**：DesignSync（组件库 8 文件推 claude.ai/design 项目 "ResearchForge UI" 41510f35，卡片由 @dsCard 标记自动建）+ 3 方法波。**本批环境注意**：建造 subagent 的沙箱**禁止跑 Python/git**（与第一批不同），故 3 个 subagent 都没自验测试也没提交——主脑兜底跑测试 + 提交 + 修 bug（见 [[design-tooling-and-orchestration]]）。
   - ✅ **生存波**：`competing_risks`（lifelines AalenJohansen CIF + 可选 R cmprsk Gray 检验）/`parametric_survival`（Weibull/LogNormal/LogLogistic AFT，AIC 选优）/`rmst`（受限平均生存时间 + 组间 z 检验）。inference-reviewer **APPROVE**。**主脑修 1 个真 MUST-FIX**：RMST 组间 SE 误用 lifelines `restricted_mean_survival_time(return_variance=True)`（那是受限寿命**分布**方差 ~θ²，不随 n 收缩），致 SE 大 ~11×、p=0.607 掩盖真实差异（真 p≈6e-9）；改用 Klein-Moeschberger/survRM2 估计量方差（新 helper `_rmst_with_se`），经 R `survival::survfit` 核验到 6 位小数。**剩 nice-to-have**：competing_risks 头条 CIF 是 pooled（summary 可注明）；AFT 可改用 lifelines 原生 `exp(coef)` CI 列（数值同）。
   - ✅ **MCDA 波**：`entropy_weight`（Shannon 熵客观赋权）/`vikor`（折衷排序 + C1/C2 接受条件）/`promethee`（PROMETHEE II 净流，V 型偏好）/`ahp`（层次分析特征向量权重 + 一致性比 CR；config pairwise 专家判断，缺则数据代理并披露）。确定性纯 numpy，18 测断言手算值绿；AHP CR 对数核验正确（测试不一致矩阵原 CR=0.056 太一致、已修为 CR=0.117）。
   - ✅ **多元波**：`manova`（四统计量 Wilks/Pillai/HL/Roy）/`discriminant_analysis`（LDA/QDA + 分层 CV 准确率）/`canonical_correlation`（白化 SVD 典型相关 + Bartlett/Wilks 序贯）/`hotelling_t2`（两样本 T²→F）。inference-reviewer **APPROVE**——Hotelling T²→F + CCA Wilks 序贯经数值核验正确（对比 sklearn CCA / 特征分解）。**已应用披露润色**：CCA 小样本上偏入 summary、MANOVA "Box's M" 措辞不暗示已检验。
1c. ✅ **并行波3 + 审计（2026-06-20，全部完成，414 绿）**：4 波各派 inference-reviewer（全 APPROVE/-WITH-FIXES）。**沙箱又禁 subagent 跑 Python/git**——主脑兜底跑测试时抓到多个真问题（见下），并发现 finance.yaml 的 YAML 语法错（描述含 `: ` 未引号，subagent 没跑全 catalog 故漏）。**主脑修的真 bug/问题**：① 状态空间 markov `res.param_names`→`res.model.param_names`（拟合即失败）；② 状态空间 DFM 传 `Z.to_numpy()` 丢列名→载荷全 NaN→方差解释塌成 0；③ 金融 EVT ξ→0 分支比率写反（高估 ~166%，inference-reviewer MUST-FIX）；④ 金融价格检测加滞后1自相关（小漂移价格漏判）；⑤ 状态空间 DFM 方差解释改 1-σ²（平滑因子方差弱信号低估 30-60%）；⑥ 非参 BCa z0 改 tie-aware（离散统计量方向错）+ 退化守卫；⑦ 心理测量 min_numeric_cols 预条件（Likert/count 数据原永不被推荐）+ omega 标准化α 键改名；⑧ finance.yaml YAML 修。**剩 backlog（无优先级，必清）**：
   - 审计未修项：csQCA/fsQCA "二值列直接用"文案与"仅选连续列"实现不符（R-gated）；`guidance_only` catalog 标志（iv_regression/pls_sem/bayesian_sem 是 guidance-only 却被当可跑推荐）；survival_analysis 把 matplotlib 当硬依赖（plot 未隔离）；GARCH omega 缩放单位披露；spatial_panel predictors `[:5]` 优先级。
   - 波3 nice-to-have：心理测量 ICC 死守卫(`not in (0,nan)`)/Feldt CI 未夹≤1/p1 单因子F 未导出/omega_hierarchical(多维);金融 n_tail_obs 只反映历史/CF 非单调披露;状态空间 N1 冗余 smp 分支/N2 死 markov 回退/N3 全NaN nanmean 警告守卫;非参小样本 BCa(<20) 量化告警/置换精确性注;心理测量 pilot/全不同整数 id-trap 鲁棒（小样本量表）。
   - **自愈护栏想法**：加一个快测「逐个 safe_load 所有 catalog/entries/*.yaml」，让 YAML 语法错快速失败而非 51 个级联（本次教训）。
2. **discover 真抓取（阶段2）**：`catalog/discover.py` 现离线 SEED，已留 `fetch_fn` 注入点；接真实 CRAN/PyPI/GitHub，带诚实降级；流行度/更新喂回 `recommender/scoring.py`。
3. **更多方法波**（域聚焦 生态/农学/经济/环境/社科）：农学实验设计(AMMI/GGE/RSM)、生态(RDA/CCA/SDM)、MCDA(AHP/熵权/VIKOR)、合成DiD/de Chaisemartin、贝叶斯状态空间…（见 method-melting-pot-roadmap 记忆）。
   - 〔GAMM nice-to-have：无 config 时结果意图是二值/计数却默认高斯，可加 summary 提示。UX 优化、非 bug。〕

---

## 方法层（method gaps）

| 项 | 为何降级/未做 | 当前替代/现状 | 如何补全 |
|---|---|---|---|
| **bayesian_sem** | 本机无 blavaan、无 RTools 编译器（brms 实测 `make not found`）、无 JAGS；不自动触发重型工具链装 | 诚实降级 → 指向 `sem`（频率派 CB-SEM）/`efa` | 装 `blavaan` + JAGS（或 RTools/Stan）→ 接 `bsem()` 真后验路径（载荷/路径可信区间） |
| **差异丰度 ANCOM-BC** | 专用桥未接（需 TreeSummarizedExperiment 构造） | ALDEx2 金标准已接（`da_method=aldex2`）；ancombc 诚实降级 | 接 ANCOMBC：构 TSE → ancombc2() → 解析 |
| **GAMM 非高斯** | 目前仅高斯族（连续结果） | 高斯 GAMM 已上线 | 扩 binomial/poisson GAMM（mgcv family=）+ 双审 |
| ~~**RDD 模糊断点**~~ ✅ | ~~仅 sharp RDD~~ | **已上线**（2026-06-18）：`fuzzy_rdd`，rdrobust `fuzzy=`，报 LATE + 第一阶段弱工具检测；inference-reviewer 审「correct as-is」 | — |
| **需用户测量模型的 SEM 族** | sem/pls_sem 的结构引擎不能猜 | sem 支持 `config model_spec`；pls_sem 诚实降级 | 文档+示例引导用户写 lavaan 语法 |

## 设计驱动（需 config 指定，非自动）
RDD（running/cutoff）、synthetic_control（treated_unit/treatment_time）、changes_in_changes（treated_group/periods）、joint model（marker/event_time/event）、double_ml/causal_forest（treatment）——这些设计性角色引擎不猜，靠 config（已在 docs/loop-decisions.md 速查 + 各分支诚实降级提示）。**优化**：可加更强的列名启发式 / profiler 增强（如识别"事件时间""驱动变量"列名）。

## 工程层（engineering）

| 项 | 现状 | 优化 |
|---|---|---|
| 全量测试慢（~2–4 min，R 重型方法多） | 后台跑规避阻塞 | pytest-xdist 并行 / 给 R 慢测打 `@pytest.mark.slow` 分层 / 缓存 R 进程 |
| discover 仍靠离线 SEED | SEED 手写、id 已对齐 catalog | **阶段2**：接真实 CRAN/PyPI/GitHub 抓取（带降级回退离线） |
| 无 web 前端 | 有 FastAPI service | **阶段3**：建前端（上传→推荐+评分卡→跑→报告） |
| 评分卡流行/新颖维是离线编辑先验 | scoring.py 规则 | 趋势引擎接通后用真实流行度/更新喂回 |
| BART 样本内 R²（无 CV） | 已披露"偏乐观" | 可加 holdout/CV R² |
| ~~**`run.py` 巨石**：7935 行 / `run_analysis` ~5500 行 / 67 分支 elif~~ | ✅ **已解决（2026-06-16）** | 拆成 15 个 `executor/branches/*.py`（注册表分发，`_branch_api.py`）+ helper 迁 `_helpers/{core,backends}.py`（run.py re-export）。`run_analysis` 只剩 setup+dispatch+teardown。**run.py 7935→148 行、最大模块 1436<1500**；70/70 id 注册、0 缺失、无循环；全量 227 绿 + `test_module_size` 护栏锁定 ≤1500。「prompt too long」元凶根除 |
| 48 处静默 `except Exception: pass` | 多为绘图 best-effort（合规，CLAUDE.md 允许）；0 处裸 `except:`（好） | 抽查**非绘图**的静默吞错（包住 CSV/文件写/估计计算的），至少 `summary.append("…失败")` 或记日志，别静默丢 |
| sharp `rdd` 的 outcome 解析未排除 running 列 | `_branch_rdd`(causal.py:574)：若用户把 `config["outcome"]` 设成 running 列会接受 → 自回归无意义（fuzzy_rdd 已加守卫排除 running/treatment） | 同样守卫 sharp rdd：`outcome = cfg["outcome"] if cfg.get("outcome") in cont and cfg.get("outcome") != running else next(...)`（inference-reviewer 标 nice-to-have，非回归） |
| `causal.py` 逼近软上限（1234 行 > ~1200 提包约定，仍 <1500 硬限） | 单文件 11 个 causal 分支 | 逼近 1500 前提升为包 `branches/causal/`（每分支一模块 + `pkgutil` 自动发现，CLAUDE.md 扩展约定）；当前不阻塞 |

## 环境/装包（本机已装，便于复现）
Py：rdrobust, doubleml, econml, networkx, dbarts(R), pysyncon, factor_analyzer, lifelines, linearmodels, semopy。
R：lavaan, QCA, SetMethods, frontier, plm, gstat, spdep, vegan, cna, metafor, mgcv, lme4, qte, JM, ALDEx2, ANCOMBC, dbarts, brms+rstan（**但无编译器、不可用**）。
**缺**：blavaan, JAGS, RTools/C++ 编译器（→ Stan 类方法当前不可跑）。

## 好点子 / 优化灵感（Good ideas — 多来自双审与建造时的发现）

> 审核/建造时冒出的、值得以后做的点子。不一定现在做，但记下来别丢。

**来自 inference-reviewer 双审的建议（已记、择机做）：**
- ~~**DML 旁加 CV R²**~~ ✅ **已做（#4c，2026-06-19）**：报 nuisance 干扰项交叉拟合(样本外) R²（PLR: E[Y|X]+E[D|X]；IRM: 结果模型 g(D,X)，倾向为分类器不报）；披露为诊断非有效性门槛。inference-reviewer 审「correct as-is」（逐源核对 out-of-fold + ml_l=E[Y|X]）。〔仍可做：交叉拟合 `n_rep>1` 多切分取平均更稳；若将来开放 n_rep 需在 flatten 前 assert n_rep==1 或按 rep 平均；causal_forest 的 holdout/CV R² 仍可加。〕
- ~~**causal_forest**：`frac_significant` 多重比较校正~~ ✅ **已做（#4a，2026-06-18）**：BH/FDR 校正 + `fdr_method=fdr_by` 切换 + 双份披露；SE 从 CI 反推（与 econml.stderr 机器精度一致）。
- ~~**BART**：holdout/CV R²~~ ✅ **已做（#4b，2026-06-18）**：80/20 holdout R²（dbarts x.test），样本内仍报但以 holdout 为准。〔split-share 重要性换 SHAP/permutation 仍可做。〕
- **GAMM**：扩 **非高斯族**（binomial/poisson）；RE 的 p 是近似自由度检验，可注明。
- **meta_regression**：加**亚组分析 / trim-and-fill** 补缺、**多层 meta**（3 层）。
- **joint model**：基线风险可选 **spline-PH**（更灵活）；`event_time` 自动检测可优先**按列名**（time/surv/fu/followup）而非"首个常量连续"。
- **meta / CiC**：小 k（<10）偏倚检验功效不足（已披露）；FE 下 I²/τ² 无意义（已改）。
- **staggered_did（Sun-Abraham，2026-06-18 审）**：① 预趋势可加**联合 Wald 检验**(所有 lead=0，`model.f_test`)作为单一原则性统计量(现为逐 lead p<0.05 标记，无多重校正，已披露功效有限)；② 总体 ATT 现为处理后事件期**简单平均**，可加**按格元样本量加权**(对齐 Callaway-Sant'Anna `aggte(simple)`)作为可选聚合(已披露二者略异)。〔should-fix 的 per-(g,e) 观测加权 + pname 精确匹配已当场应用〕

**方法可运行性验证（用户点子 2026-06-16）：**
- 原则：每个方法都用**真实示例数据**跑通 + 出图，**留代码**作为"可运行"凭证。
- 小 demo（KB 级）：committed + 驱动测试（合成数据常不收敛，真实小样本更稳）。
- 大数据（MB+）：**下载→跑通→作图→删数据留代码**（省空间），验证留痕记本日志（来源+结果，便于复现）。
- 极致版：测试**从 R 包内置数据现取**（gsynth simdata / JM aids…），数据不进仓库、只留代码。

**通用工程灵感：**
- **profiler 增强**：按列名识别设计性角色（事件时间、驱动变量、处理组），减少 RDD/JM/synth 对 config 的依赖。
- **测试提速**：pytest-xdist 并行 + R 慢测分层（`@pytest.mark.slow`）。
- **inference-reviewer 子代理**本身是高价值资产——R 后端真推断几乎每个都能挑出真 bug（本阶段抓了 ~10 个 must-fix），值得保持每个真推断方法都派。
- **评分卡**：趋势引擎接通后，用真实流行度/更新喂回 popularity/novelty 维（现为离线先验）。
- **config 机制**：可做一个 `config schema` 校验 + 友好报错（现各分支自查）。

**全量代码审核发现（2026-06-16, Opus max）：**
- ✅ **头号结构债 = `run.py` 巨石 — 已解决（2026-06-16）**：拆成 15 个 `branches/*.py` 注册表处理器（工具 `_migrate_branches.py` 正则搬体、断言防腐，已用毕删除），run.py 7935→2442 行，全量 227 绿。手法：handler 从 `ctx` 解包同名局部变量后跑原分支体逐字不变（全链审计 67/67 只 mutate 不 rebind，故行为保持）。helper 仍留 run.py、family 模块按需 import。**「prompt too long」根治**：读单族文件即可，不必读整 run.py。
- **静默吞错抽查**（见上工程表）：48 处 `except Exception: pass`，多数是绘图 best-effort（合规），但需抽查包住文件/估计的少数。
- **正向确认**：0 处裸 `except:`；83 个测试文件；自评分卡（86 分）已诚实指出最弱两维 = **可用性 58（无 web 前端，阶段3）** 与 **快速性 62（R 测试慢，pytest-xdist/分层）**——与本审核独立结论一致，二者是当前最高优先优化项。

**Codex 跨族审查补充（2026-06-16，nice-to-have；结构性的已写进 roadmap）：**
- **config schema 前移为 v0.9 门槛**（非横切）：每 analysis entry 一份机器可读参数规格,Web UI/推荐解释/运行错误共消费,别各自猜。
- **discover 趋势分 ≠ 包热度**:加 领域归一化 / 维护状态 / 最近发布 / 引用·教程信号 / license·安装可行性,保留人工 gate。
- **Web MVP 要薄**:上传→推荐→config→运行→报告即可;把 schema/错误提示/产物浏览做扎实 > 追 UI 完整度。
- **实验设计要"设计感知"**:RCBD/split-plot/nested/repeated 做成强制声明 block/plot/subplot 角色的 mixed-model wrapper（已入 roadmap）。
- Codex 审查任务书在 `docs/codex-review-brief.md`（之前+接下来的阅读清单 + 红线）。

- **spatial_panel（SAR/SEM/SDM, splm）impacts 的 SE**：当前 direct/indirect/total 用 LeSage-Pace **解析点估计**（exact，`S=(I-ρW)^{-1}(Ib+Wθ)`），**未附模拟标准误/CI**。原因：splm 自带 `impacts()` 在本机 spdep/spatialreg 版本下易碎（`trW`、`as_dgRMatrix_listw` 已移位/改名，且 `impacts(spml)` 报 `have_factor_preds` 断言失败），故绕过自带实现、自算点估计。**补全**：用 ρ、β 的协方差（`vcov(m)`）做 delta-method 或参数自助（draw ρ/β ~ N(est, vcov)，重算 S，取分位）给 impacts 的 SE/CI。
- **spatial_panel 的 W 仅 k-NN**：默认行标准化 k-NN（k=6，欧氏经纬度）。**待优化**：可配距离阈值/contiguity/反距离权重，以及真测地距离（与截面 spatial_regression 同一 backlog 第 6 条）。
- **spatial_panel FE 仅个体（within, individual）**：未做双向（个体+时间）或随机效应空间面板，也未自动跑 Hausman 选 FE/RE-spatial。**补全**：接 `effect="twoways"` 与 `spml` 的 RE 变体 + `sphtest`（空间 Hausman）。

---
*持续追加。受硬件/装包限制绕过的、以及审核时的好点子，都在此留痕。*
