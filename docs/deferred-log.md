# 未做 / 降级 / 待优化 日志（Deferred & Optimization Log）

> 这里记录**没做完、降级处理、或受环境限制绕过**的事项，方便后续回头优化。
> 每条：**项** · **为何没做/降级** · **当前可用替代/现状** · **如何补全/优化**。
> 由 AI 自走时持续追加（受 CPU/GPU/装包/后端限制时尤其要写在这）。

## 方法层（method gaps）

| 项 | 为何降级/未做 | 当前替代/现状 | 如何补全 |
|---|---|---|---|
| **bayesian_sem** | 本机无 blavaan、无 RTools 编译器（brms 实测 `make not found`）、无 JAGS；不自动触发重型工具链装 | 诚实降级 → 指向 `sem`（频率派 CB-SEM）/`efa` | 装 `blavaan` + JAGS（或 RTools/Stan）→ 接 `bsem()` 真后验路径（载荷/路径可信区间） |
| **差异丰度 ANCOM-BC** | 专用桥未接（需 TreeSummarizedExperiment 构造） | ALDEx2 金标准已接（`da_method=aldex2`）；ancombc 诚实降级 | 接 ANCOMBC：构 TSE → ancombc2() → 解析 |
| **GAMM 非高斯** | 目前仅高斯族（连续结果） | 高斯 GAMM 已上线 | 扩 binomial/poisson GAMM（mgcv family=）+ 双审 |
| **RDD 模糊断点** | 仅 sharp RDD | sharp 已上线、披露 fuzzy 需另接 | 接 fuzzy RDD（rdrobust fuzzy 参数）+ 双审 |
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
| **`run.py` 巨石**：7935 行，`run_analysis` 单函数 ~5500 行（line 2391→EOF）/ 66 分支 `elif` | 能跑、测试全绿，但**读整文件会撑爆上下文（疑似 VS Code "prompt too long" 元凶）**、难维护、定位慢 | 按方法族拆 `executor/branches/*.py` + `id→handler` 注册表（registry dispatch），`run_analysis` 只做分发与公共脚手架；**以全量测试为护栏分批迁移**（高收益、需独立一轮谨慎做） |
| 48 处静默 `except Exception: pass` | 多为绘图 best-effort（合规，CLAUDE.md 允许）；0 处裸 `except:`（好） | 抽查**非绘图**的静默吞错（包住 CSV/文件写/估计计算的），至少 `summary.append("…失败")` 或记日志，别静默丢 |

## 环境/装包（本机已装，便于复现）
Py：rdrobust, doubleml, econml, networkx, dbarts(R), pysyncon, factor_analyzer, lifelines, linearmodels, semopy。
R：lavaan, QCA, SetMethods, frontier, plm, gstat, spdep, vegan, cna, metafor, mgcv, lme4, qte, JM, ALDEx2, ANCOMBC, dbarts, brms+rstan（**但无编译器、不可用**）。
**缺**：blavaan, JAGS, RTools/C++ 编译器（→ Stan 类方法当前不可跑）。

## 好点子 / 优化灵感（Good ideas — 多来自双审与建造时的发现）

> 审核/建造时冒出的、值得以后做的点子。不一定现在做，但记下来别丢。

**来自 inference-reviewer 双审的建议（已记、择机做）：**
- **DML / causal_forest**：交叉拟合用 `n_rep>1` 多次切分取平均，更稳；ATE 旁可加 **holdout/CV R²**（样本内偏乐观）。
- **causal_forest**：`frac_significant`（逐行 95% CI）应做**多重比较校正**（已加披露，未做校正）；可上 BH/同时置信带。
- **BART**：加 **holdout/CV R²**（现仅样本内）；split-share 重要性可换 SHAP/permutation。
- **GAMM**：扩 **非高斯族**（binomial/poisson）；RE 的 p 是近似自由度检验，可注明。
- **meta_regression**：加**亚组分析 / trim-and-fill** 补缺、**多层 meta**（3 层）。
- **joint model**：基线风险可选 **spline-PH**（更灵活）；`event_time` 自动检测可优先**按列名**（time/surv/fu/followup）而非"首个常量连续"。
- **meta / CiC**：小 k（<10）偏倚检验功效不足（已披露）；FE 下 I²/τ² 无意义（已改）。

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
- **头号结构债 = `run.py` 巨石**（见上工程表）。直接后果：①上下文易爆（"prompt too long"）②新增分析要在 5500 行函数里翻找。**建议下一个独立 effort 专门做拆分重构**，全量测试当护栏，每迁一族跑一次。
- **静默吞错抽查**（见上工程表）：48 处 `except Exception: pass`，多数是绘图 best-effort（合规），但需抽查包住文件/估计的少数。
- **正向确认**：0 处裸 `except:`；83 个测试文件；自评分卡（86 分）已诚实指出最弱两维 = **可用性 58（无 web 前端，阶段3）** 与 **快速性 62（R 测试慢，pytest-xdist/分层）**——与本审核独立结论一致，二者是当前最高优先优化项。

---
*持续追加。受硬件/装包限制绕过的、以及审核时的好点子，都在此留痕。*
