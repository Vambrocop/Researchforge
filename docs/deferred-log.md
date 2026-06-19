# 未做 / 降级 / 待优化 日志（Deferred & Optimization Log）

> 这里记录**没做完、降级处理、或受环境限制绕过**的事项，方便后续回头优化。
> 每条：**项** · **为何没做/降级** · **当前可用替代/现状** · **如何补全/优化**。
> 由 AI 自走时持续追加（受 CPU/GPU/装包/后端限制时尤其要写在这）。

## 🔜 下一波（优先级，2026-06-19 更新）

> 新会话/下一波接着做这些（记忆 `next-batch` 也指向这里）。**已完成（别重做）**：run.py 巨石全拆、测试提速（`pytest -n 2`；别用 `-n auto` OOM）、conformal、**因果/计量方法波**（PSM/IPW/event_study/fuzzy_rdd/staggered_did）、**推断 backlog #4a-#4d**（causal_forest BH/FDR、BART holdout R²、DML nuisance CV R²、GAMM 非高斯族）、**A 风格 web 前端**（接引擎 + goals 后端供给）、**时间序列波**（协整+VECM/GARCH/结构突变/STL/ARDL）。全部 inference-reviewer 双审或确定性实测。全量 283 绿。

1. 🔄 **并行 subagent 集成队列（2026-06-19，compact 后用 `git worktree list` 找回）**：
   - ✅ **设计流（组件库）**：已审+合并+推送（main 6b267b8，web/static/components/ 8 文件）。**待办：DesignSync 推到 claude.ai/design**（需用户登录）。
   - 🔍 **农学设计波（待审+整合）**：worktree 分支 `worktree-agent-a4079af329142421f`——`ammi`/`gge_biplot`/`response_surface`（纯 Python，experimental_design.py 1228 行，13 测过+12 无回归）。**待办：inference-reviewer 审（AMMI SVD/稳定性、GGE 中心化、RSM 典范分析）→ 合并 → 全量 → push。**
   - 🔍 **空间面板（待审+整合）**：worktree `worktree-agent-a0ee0a3a38f2b817e`（commit fef8de9）——`spatial_panel` SAR/SEM/SDM（R splm，econometrics.py 608 行，9 测过；rho 0.5→0.51 回收，impacts=b/(1-rho) 对得上）。**待办：inference-reviewer 审（rho/impacts 解析式 LeSage-Pace、W 选择、FE）→ 合并 → 全量 → push。** 注：worktree 里有未跟踪 scratch `tests/test_spatial_panel_debug.py`（已排除提交，整合时删）。impacts 仅点估计无模拟 SE（splm impacts() 版本碎，已 deferred-log delta/bootstrap 补全）。
   - 🔍 **生态波（待审+整合）**：worktree `worktree-agent-a5c9ed115ed07182a`（commit 84549f3）——`mantel_test`/`indicator_species`（纯 Python）+ `rda`（R vegan，已装，实测 93.7% 约束方差/F=274/p=.001）；新增 catalog `ecology.yaml`；ecology.py 1164 行，13 测过+13 无回归。**待办：inference-reviewer 审（Mantel 置换/RDA vegan 解析）→ 合并 → 全量 → push。**
   - **整合顺序（compact 后逐个做）**：① 农学 a4079af ② 空间面板 a0ee0a3 ③ 生态 a5c9ed1。各：派 inference-reviewer→据审修→merge worktree 分支→`pytest -n 2` 全量绿→push。注意空间面板 worktree 有未跟踪 scratch 文件整合时删。
   - 编排约定见记忆 [[design-tooling-and-orchestration]]：worktree 隔离、不相交文件、主脑双审、push 等用户。**红线：未经 inference-reviewer 审的真推断方法不合并/不推。**
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

---
*持续追加。受硬件/装包限制绕过的、以及审核时的好点子，都在此留痕。*
