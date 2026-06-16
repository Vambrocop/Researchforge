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
> - 因果森林 causal_forest：`treatment`、`effect_modifiers`(异质特征列表)、`outcome`、`n_folds`、`seed`
> - Meta 回归 meta_regression：`moderators`(调节变量列表)、`measure`、`method`(同 meta_analysis)
> - GAMM：`outcome`、`predictors`、`group`(随机截距分组列)
> - changes-in-changes：`outcome`、`treatment`、`time`、`treated_group`(=1 的处理组值,定方向)、`periods`[前,后]、`probs`
> - 网络分析 network_analysis：`source`、`target`(边两端节点列)、`weight`(可选边权)、`directed`(默认 False)

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

---
**怎么用**：你回来逐条说「第 N 条改成 X」，或「都先这样」。我据此把对应分析升级成接受参数/换默认，并重跑。
（此清单由自走 loop 维护，新增方法若有新默认会续加。）
