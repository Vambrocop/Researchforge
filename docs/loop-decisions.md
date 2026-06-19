# 待用户决策清单（自走 loop 累积）

自走建设时，引擎对一些**实质性选择**做了合理默认（已在每个分析的产出里披露）。这些默认能让分析跑起来，但你可能想按领域知识覆盖。

> **✅ 用户可配置机制已上线（2026-06-13）**：`run_analysis(..., config={...})`，CLI `run <data> <id> --config '{"key":"val"}'`，web service 同步透传。各分支读 `cfg.get(<key>)` 覆盖默认、否则回退自动默认（默认仍能跑）。**九条决策已全部接入**（#1–#8 完全可配；#9 提供可配的纯 Python 检验 + 金标准诚实降级，R 专用桥待接）。下表「状态」逐条标注。
>
> **config 键速查**（按分析）：
> - 回归族(#5)：`outcome`(结果列)、`predictors`(预测列表)
> - MCDA TOPSIS/CRITIC/隶属/灰关联(#2)：`cost_criteria`(成本型列名列表)
> - DEA/SFA/Malmquist(#1)：`inputs`、`outputs`(列名列表)；Malmquist 另 `periods`(#8, [起,末])
> - QCA fsQCA/必要性(#3)：`anchors`([低,中,高] 分位)、`incl_cut`((0,1])；csQCA 仅 `incl_cut`
> - 空间 Moran/LISA/Gi*/空间回归(#6)：`knn_k`(近邻数)
> - 动态面板 GMM(#7)：`endogenous`(内生协变量列表)、`gmm_lags`([lo,hi])
> - SEM(#4)：`model_spec`(lavaan/semopy 语法字符串)
> - 差异丰度(#9)：`da_method`(clr_mw / clr_welch / aldex2[R 金标准已接]；ancombc 降级)
> - RDD 断点回归：`running`(驱动变量,必填)、`cutoff`(断点,必填)、`outcome`
> - 双重机器学习 DML：`treatment`(处理列)、`controls`(混杂列表)、`outcome`、`n_folds`(默认5)、`seed`(默认0,固定交叉拟合切分)
> - 因果森林 causal_forest：`treatment`、`effect_modifiers`(异质特征列表)、`outcome`、`n_folds`、`seed`、`fdr_method`(`fdr_bh` 默认 / `fdr_by` 任意相关下保守)
> - Meta 回归 meta_regression：`moderators`(调节变量列表)、`measure`、`method`(同 meta_analysis)
> - GAMM：`outcome`、`predictors`、`group`(随机截距分组列)、`family`(`gaussian` 默认/`binomial`/`poisson`；缺省按结果列类型自动:二值→binomial、计数→poisson)
> - changes-in-changes：`outcome`、`treatment`、`time`、`treated_group`(=1 的处理组值,定方向)、`periods`[前,后]、`probs`
> - 网络分析 network_analysis：`source`、`target`(边两端节点列)、`weight`(可选边权)、`directed`(默认 False)
> - 保形预测 conformal_prediction：`outcome`(结果,默认首个连续列)、`predictors`(预测变量列表)、`alpha`(误覆盖率,默认0.1→90%区间)、`seed`(默认0,固定切分+RF)
> - 空间面板 spatial_panel(SAR/SEM/SDM, R splm)：`unit`/`time`(面板索引)、`outcome`、`predictors`、`lon`/`lat`(每单位坐标,建 W)、`model`(`lag`=SAR 默认 / `error`=SEM / `sdm`=空间杜宾)、`knn_k`(空间权重近邻数,默认6,行标准化)；LM 检验(slmtest)仅给 lag-vs-error 建议、不自动切换；impacts(direct/indirect/total)用 LeSage-Pace 解析式

> 说明：以下都是「默认能用、但你说了算」的点，不是 bug。按重要性排序。

| # | 涉及分析 | 引擎默认 | 你可能想改成 | 状态 |
|---|---|---|---|---|
| 1 | **DEA / Malmquist / SFA** | 首个数值列=产出，其余=投入 | 明确指定哪些是投入/产出（最关键） | ✅ DEA 可配 `inputs`/`outputs`；Malmquist/SFA 待接 |
| 2 | **TOPSIS/隶属函数/灰关联/CRITIC** | 所有指标=效益型（越大越好） | 标出哪些是成本型（越小越好，需反向） | ✅ 四法均可配 `cost_criteria`（成本型反向，标准 (hi-x)/rng 变换） |
| 3 | **fsQCA/csQCA/QCA必要性** | 校准锚点：fuzzy 用 0.1/0.5/0.9 分位、crisp 用中位数；incl.cut=0.8(必要性 0.9) | 按理论设锚点 + 一致性阈值 | ✅ 可配 `anchors`(3 分位)/`incl_cut`；csQCA 仅 incl_cut（中位二分待接） |
| 4 | **SEM** | 单因子 CFA 模板 | 你的理论结构（多因子/路径/中介/完整 SEM） | ✅ 可配 `model_spec`(lavaan/semopy 语法,多因子/路径);列名从 spec 自动提取,两后端通用 |
| 5 | **回归族/NCA/分位/有序…** | 结果变量=首个连续列 | 指定真正的结果变量 | ✅ 回归族可配 `outcome`/`predictors`；其余分支待接 |
| 6 | **空间(Moran/Gi*/LISA/空间回归)** | k-NN 权重 k=8(R路径6)，经纬度欧氏距离 | 改 k / 距离度量 / 真测地距离 / 其他权重 | ✅ 四法均可配 `knn_k`（近邻数）；距离度量/权重型待接 |
| 7 | **动态面板 GMM** | 协变量设为严格外生；工具滞后 2-4 | 标出内生/前定变量；调工具滞后深度 | ✅ 可配 `endogenous`(内生协变量)/`gmm_lags`[lo,hi]；滞后DV强制 lag≥2 |
| 8 | **Malmquist** | 取首末两期 | 逐期链式 / 指定基期 | 待批 |
| 9 | **差异丰度** | CLR + Mann-Whitney + BH-FDR（纯 Py 筛查法） | 上 R 金标准 ALDEx2 / ANCOM-BC（组成性更严谨） | ✅ 可配 `da_method`：clr_mw(默认)/clr_welch/**aldex2(R 金标准已接, MC-CLR+Welch)**；ancombc 桥待接(需 TreeSummarizedExperiment)→诚实降级 |
| 10 | **空间面板 (SAR/SEM/SDM, splm)** | 模型=lag(SAR)；W=k-NN k=6 行标准化；个体FE(within)；LM检验仅建议不自动切换 | 选 model(lag/error/sdm)、改 k、指定 unit/time/outcome/predictors/lon/lat | ✅ 可配 `unit`/`time`/`outcome`/`predictors`/`lon`/`lat`/`model`(lag/error/sdm)/`knn_k`；impacts 用 LeSage-Pace 解析式(direct/indirect/total)，绕过 splm 自带 impacts() 跨版本易碎(trW/as_dgRMatrix_listw 移位) |
| 11 | **生存扩展 (competing_risks/parametric_survival/rmst)** | duration/event/group 按列名+profiler 自动；RMST 的 **tau 默认=各组最大【事件】时间的最小值**（保证共同支撑，比 survRM2 的"最大观测时间"更稳，避免末端外推）；竞争事件 0=删失/1=兴趣/≥2=竞争 | 指定列角色；设 tau；event_of_interest | ✅ 可配 `duration`/`event`/`group`/`tau`/`event_of_interest`；用户 tau 超组随访自动夹到共同支撑并披露 |
| 12 | **MCDA 新增 (entropy_weight/vikor/promethee/ahp)** | 指标=连续列、权重缺则等权或客观导出、全效益型；VIKOR v=0.5；PROMETHEE V型偏好(p=各指标极差)；AHP 缺 pairwise 则数据代理(均值比)并披露非真 AHP | 标成本型；给 weights；VIKOR 调 v；AHP 给 `pairwise` 专家判断矩阵 | ✅ 可配 `weights`/`cost_criteria`/`v`(VIKOR)/`q`/`p`(PROMETHEE)/`pairwise`(AHP) |
| 13 | **多元 (manova/discriminant/canonical_correlation/hotelling_t2)** | 结果变量=全部连续列、分组=最低基数分类列；**CCA 默认把连续列对半切 set_x/set_y（任意，强烈建议 config 指定两个概念变量集）** | 指定 outcomes/group；CCA 指定 set_x/set_y | ✅ 可配 `outcomes`/`group`/`factors`(MANOVA)/`predictors`(判别)/`set_x`/`set_y`(CCA) |
| 14 | **心理测量 (cronbach_alpha/mcdonald_omega/icc)** | 题项=连续+计数列(Likert 常判 count;新 min_numeric_cols 预条件让其可被推荐);ICC 报 Shrout-Fleiss 6 形式 | 指定题项;选 ICC 形式判读(随机/固定评分者) | ✅ 可配 `items`/`columns`;omega 报标准化α(键 cronbach_alpha_standardized,与 raw α 区分) |
| 15 | **金融 (value_at_risk/extreme_value/risk_adjusted_return)** | 单序列(首连续列);价格自动判别(全正且趋势相关高或 ρ₁≈1)转对数收益;VaR α=[.95,.99];EVT 阈值=损失95分位;年化 ppy=252,rf=0;损失为正 | 设 is_returns/α/阈值/evt_alpha/ppy/rf | ✅ 可配 `column`/`is_returns`/`alpha`/`threshold`/`threshold_quantile`/`evt_alpha`/`periods_per_year`/`rf` |
| 16 | **非参 (permutation_test/bootstrap_ci/robust_regression)** | 结果=首连续、分组=最低基数;BCa 统计量=mean;n_perm/n_boot=9999;种子=0(可配) | 设统计量/重采样数/置信水平/种子 | ✅ 可配 `outcome`/`group`/`n_perm`/`statistic`(mean/median/std/correlation)/`column`/`column2`/`ci`/`n_boot`/`predictors`/`seed` |
| 17 | **状态空间 (unobserved_components/markov_switching/dynamic_factor)** | UC level=局部线性趋势,季节按周期自动/config;markov k_regimes=2、switching_variance=True、按均值排序解标签切换;DFM k_factors=1、标准化、首因子载荷和≥0定符号;单序列键=`column`(异于 timeseries 族的 `value`) | 选 level/seasonal_period;k_regimes/order;k_factors/factor_order;columns | ✅ 可配 UC `column`/`level`/`seasonal_period`;markov `column`/`k_regimes`/`switching_variance`/`order`/`search_reps`;DFM `columns`/`k_factors`/`factor_order` |

---

## 诚实降级 / 待办（装包或后端门槛）

- **bayesian_sem（贝叶斯 SEM）**：诚实降级。需 R `blavaan` + JAGS 或 Stan(C++ 编译)后端 + 理论测量模型；本机 blavaan 未装、无 RTools 编译器（brms 实测 `make not found` 无法编译）、无 JAGS。**不自动触发重型/易碎的工具链安装**。可运行替代：`sem`（频率派 CB-SEM，config model_spec）/ `efa`。**待办**：装 blavaan + JAGS（或 RTools/Stan）后，接 `bsem()` 真后验路径。
- **差异丰度 #9 的 ANCOM-BC**：桥待接（需 TreeSummarizedExperiment）；ALDEx2 已接。

---
**怎么用**：你回来逐条说「第 N 条改成 X」，或「都先这样」。我据此把对应分析升级成接受参数/换默认，并重跑。
（此清单由自走 loop 维护，新增方法若有新默认会续加。）
