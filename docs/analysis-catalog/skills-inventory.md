# ResearchForge 能力盘点 v0.1（搜集阶段）

> 一次盘清"我们已有什么"，映射到三层架构，标出缺口。避免每次重整。
> 三层架构：① 数据画像 → ② 推荐（混合路线 C）→ ③ 执行 + 出图出报告

角色标记：**[执行器]** 能真跑代码出结果 ｜ **[规划/综述]** 给方法或做文献综述 ｜ **[写作/出图]**

## 一、已有 skill ×（领域 / 角色）

### 经济学 / 农业经济学 — 覆盖强 ✅
- **empirical-analysis-python** [执行器·核心]：完整 8 步计量流水线（清洗→构造→描述→诊断→基线→稳健→机制/异质→出表出图），AER/QJE 风格多列回归表 + 事件研究/系数图。另含 **Mode A 流行病学**（IPTW/TMLE/MR/生存）、**Mode B 因果机器学习**（DML/因果森林/meta-learner/policy tree）。
- **empirical-analysis-R** [执行器]、**stata** [执行器/参考]
- **statspai-skill**、**the-effect-book**、**results-analysis**
- buffett 等一批投资/估值"思维框架" skill — 偏定性，非数据执行器

### 通用统计学 — 覆盖强 ✅
- **openclaw-statistical-analysis** [执行器]：t检验/ANOVA/回归/贝叶斯/功效分析/假设检查/APA 报告
- **openclaw-data-stats-analysis** [执行器]
- **openclaw-bio-experimental-design-power-analysis / -sample-size**（实验设计/样本量）

### 土壤学 / 农学（田间试验） — 部分覆盖 ⚠️
- 有：openclaw-statistical-analysis 可做 ANOVA/回归/混合模型基础；environment-life-review-forge 管文献综述端
- **缺**：专门的田间试验执行器（RCBD/裂区/混合效应 lme4·nlme、agricolae 风格）、土壤剖面/空间地统计（kriging）

### 生态学 — 原始数据分析端缺口明显 ❌
- 有：**environment-life-review-forge**[规划/综述]（PECO/PICO、异质性、风险偏倚）、meta-analysis-forge
- **缺**：群落生态执行器（排序 NMDS/PCA、PERMANOVA、α/β 多样性、vegan 风格）、物种分布模型

### 微生物学 — 缺口 ❌
- 有：openclaw-bio-causal-genomics-mediation-analysis（基因组中介）、实验设计 skill
- **缺**：扩增子/微生物组流水线（DADA2/QIIME2、α/β 多样性、差异丰度 DESeq2/ANCOM）

### 文献 / 证据综述 — 覆盖强 ✅
- deep-research（13-agent）、academic-pipeline、evidence-synthesis-forge、meta-analysis-forge、environment-life-review-forge、literature-review、umbrella-review-skeptic、meta-ml-screener、nature-academic-search、paper-search-mcp、daily-paper-generator

### 写作 / 出图 / 出片 — 覆盖强 ✅
- academic-paper（12-agent）、academic-paper-reviewer、ml-paper-writing、nature-writing/polishing/figure/citation/response/reader、academic-plotting、humanizer-academic、avoid-ai-writing、chinese-de-aigc、paper-slide-deck、k-dense-*

## 二、映射到三层 → 要建什么

| 层 | 现状 | 结论 |
|----|------|------|
| ① 数据画像 | 无现成 skill | **要新写**（轻量 Python：结构探测 + 变量类型 + 面板/处理组识别） |
| ② 推荐（混合 C） | 无现成 | **核心要建**：分析目录知识库 + LLM 推理排序 |
| ③ 执行 | 经济/统计/综述/写作可复用 | 生态/土壤/微生物**原始数据执行器是主要缺口**（封装 R 包或新写） |

## 三、缺口总结（待建清单）
1. **数据画像引擎**（全新，轻）
2. **分析目录 + 混合推荐引擎**（全新，项目心脏）
3. **生态/土壤/微生物原始数据执行器**（封装 vegan/lme4/nlme/agricolae/phyloseq/DADA2，或新写）
4. **编排骨架**（把 ①②③ 串起来）

## 四、外部可吸收的学术 skill（待评估）
- **Academic Research Skills suite**（v3.7, 2026；research→write→review→revise→finalize，插件市场 30 秒装）— 与我们已有 academic-pipeline/academic-paper 高度重叠，可对比取长
- **awesome-econ-ai-stuff**（经济学 AI 工具清单：Stata 清洗→LaTeX→DID 固定效应 + 系数图）
- **claudeskills.info `/category/research`**（研究类 skill 目录）
- GitHub `imbad0202/academic-research-skills`

## 五、论文配套代码（replication code）去哪儿找
- **经济/社科**：AEA Data & Code Repository @ openICPSR（AER/AEJ 强制提交，复现包金矿）、Harvard Dataverse（含 REStat）、OSF、Zenodo、ICPSR
- **生态/农学/环境**：Dryad（自动同步 Zenodo）、Zenodo（给 DOI）、figshare、KNB、DataONE、rOpenSci（R 包生态）
- **微生物/生信**：Zenodo、NCBI/SRA（数据）、Bioconductor（包）、QIIME2 论坛
- **通用**：Papers with Code、Code Ocean、OSF

## 六、已知约束
- **便携性**：用户可能换电脑 → 整个项目（引擎 + 目录 + 所需 skill）应放进 **Git 仓库**，换机 = `git clone`。注意：现有 skill 在 `~/.claude/skills/`（机器相关），需在仓库内打包或用安装脚本，否则换机后引擎找不到执行器。
