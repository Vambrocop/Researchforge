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
> - Cohen's κ cohens_kappa：`rater1`/`rater2`(两位评分者列,默认前两个类别/序数列)、`weights`(`none` 默认 / `linear` / `quadratic`,序数量表加权口径,二者均会算出报告,此键仅决定摘要头条)
> - Fleiss' κ fleiss_kappa：`raters`(评分者列表,默认全部类别/二值/计数列)、`count_matrix`(默认 False;True 时各列=按类别已计数的「被试×类别」矩阵)；评分数不齐时按众数 n 对齐删行
> - Bland-Altman bland_altman：`method1`/`method2`(两种连续测量列,默认前两个连续列)；LoA=bias±1.96·SD,各界限 95% CI 用 Bland-Altman SE(LoA)≈SD·√(1/n+1.96²/(2(n-1)))

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
| 18 | **计数 (zero_inflated_poisson/zero_inflated_negbin/tweedie_glm)** | 结果=首个计数列(tweedie 取首连续,半连续判 continuous);预测=连续/二值(≤5);ZI 膨胀模型默认仅常数;Tweedie var_power=1.5 | 指定 outcome/predictors;ZI 给 inflation 预测列;调 var_power | ✅ 可配 `outcome`/`predictors`/`inflation`(ZI 膨胀预测列)/`var_power`(Tweedie p,默认1.5) |
| 19 | **GWR 地理加权回归** | 核=双平方(bisquare);带宽=自适应k,按 AICc 网格选;非平稳=描述性指标(局部IQR>2·全局SE,会高估、非正式检验) | 设 kernel/bw/adaptive_k;指定 lon/lat/outcome/predictors | ✅ 可配 `lon`/`lat`/`outcome`/`predictors`/`kernel`(bisquare/gaussian)/`bw`(固定带宽)/`adaptive_k`(近邻数) |
| 20 | **贝叶斯共轭 (bayesian_ab_test/bayesian_proportion/bayesian_poisson_rate)** | 先验默认弱信息(Beta(1,1)/Gamma(0.001,0.001))并声明;P(B>A)闭式;P(θ>ref) 用 sf;过离散用曝光感知 Pearson | 设先验/参考值/曝光列;计数模式给 successes/trials | ✅ 可配 AB `outcome`/`group`/`successes_*`/`trials_*`/`prior_a`/`prior_b`/`ci`;比例 `outcome`/`prior`(jeffreys)/`prior_a`/`prior_b`/`ci`/`ref`/`interval`(hpd);泊松 `outcome`/`exposure`/`group`/`prior_a`/`prior_b`/`ci` |
| 21 | **IRT (irt_2pl/irt_rasch, girth)** | 题项=二值 0/1 列(`min_categorical_cols:3`);MML 估计;能力=EAP(N(0,1) 先验);仅二分类(多分类须先二分) | 指定 items;model | ✅ 可配 `items`/`columns`(题项列)/`model`(irt_2pl 的 2pl) |
| 22 | **潜类别 (latent_class_analysis/latent_profile_analysis, stepmix)** | LCA 指标=分类/二值/计数(`min_categorical_cols:2`)、measurement 二值→binary 否则 categorical;LPA 指标=连续、gaussian_diag(自由方差);k 按 BIC 选 2..5;熵=relative_entropy;类按大小排(解标签切换);n_init 多启动 | 设 n_classes;indicators | ✅ 可配 `indicators`/`n_classes`(固定 k,否则 BIC 选) |
| 23 | **混合/异常 (gaussian_mixture/dbscan_clustering/mahalanobis_outliers, sklearn)** | 特征=连续列、标准化(马氏除外,尺度不变);GMM k 按 BIC 选 1..6;DBSCAN min_samples=2p、eps 按 k 距离肘选;马氏 MCD 稳健协方差 + 卡方(p,.975) 阈 | 设 k_range/covariance_type;eps/min_samples;alpha | ✅ 可配 `features`;GMM `k_range`/`covariance_type`/`n_init`;DBSCAN `eps`/`min_samples`;马氏 `alpha`(默认0.975) |
| 24 | **Callaway-Sant'Anna 交错 DiD (callaway_santanna, R did)** | gname=各单位首次处理期(0=从不处理),无则由 0/1 `treatment` 推导;control_group=nevertreated(无则回退 notyettreated);est_method=dr(双稳健);聚合 simple/dynamic(事件研究)/group | 指定 unit/time/outcome/gname 或 treatment;control_group/est_method | ✅ 可配 `unit`/`time`/`outcome`/`gname`/`treatment`/`control_group`(nevertreated/notyettreated)/`est_method`(dr/ipw/reg) |
| 25 | **因果敏感性 (oster_delta/evalue/rosenbaum_bounds)** | oster R_max=min(1.3·R̃,1);evalue RR 转换 rare→OR、common→sqrt(OR)、连续→exp(.91d) Chinn,连续暴露>2级按中位二分;**rosenbaum** 协变量匹配对(倾向得分 logit 最近邻+0.2·SD caliper,标准化协变量距离回退,绝不按结果配对)+signed-rank Γ 边界,Γ 网格1..6步0.05,临界 Γ*=上界 p 首超0.05 | 指定 outcome/treatment(exposure)/controls;调 r_max;rosenbaum 指定 covariates | ✅ 可配 oster `outcome`/`treatment`/`controls`/`r_max`;evalue `outcome`/`exposure`(或 treatment)/`controls`;rosenbaum `treatment`/`outcome`/`covariates` |
| 26 | **多项 IRT + DIF (irt_grm/irt_pcm/dif_detection, girth)** | 题项=count 序数列 0..K(`min_categorical_cols:3`);GRM=自由区分度+有序阈值;**PCM 实为 GPCM(girth 自由每题区分度,非等区分度)**;DIF 按总分(rest score)匹配做 MH+逻辑回归 | 指定 items;DIF 指定 group | ✅ 可配 `items`/`columns`;dif_detection 另 `group` |
| 27 | **降维/对应 (mds/correspondence_analysis/pls_regression)** | MDS 度量+标准化欧氏距离,stress-1;CA(2 分类列)/MCA(≥3),total_inertia 满秩(=χ²/n);PLS CV 选成分+VIP | MDS 设 n_components/metric;CA 指定 columns;PLS 指定 outcome/predictors/n_components | ✅ 可配 MDS `features`/`n_components`/`metric`/`label`;CA `columns`;PLS `outcome`/`predictors`/`n_components` |
| 28 | **网络科学 (community_detection/centrality_suite/epidemic_model, networkx+louvain)** | 边表 source/target(+weight/directed);Louvain 社团+模块度;5 中心性(权重语义:度不计权、介数/接近=距离、特征向量/PageRank=强度);SIR/SIS β=.05/γ=.1、n_runs=10、seed | 设 source/target/weight;SIR 设 model/beta/gamma/steps | ✅ 可配 `source`/`target`/`weight`/`directed`;centrality `size_by`/`top_n`;epidemic `model`/`beta`/`gamma`/`initial_infected`/`steps`/`n_runs`/`seed` |
| 29 | **列联表 (loglinear/cmh_test/ordinal_association)** | 因子=低基数 分类/二值/计数(自动选择有基数上限,避全不同整数 id-trap;config 强指不受限);loglinear 取前 2 因子做 2×2 独立性 G²,残差用**调整 Pearson 残差**(~N(0,1),\|r\|>2 标显著);cmh 二值暴露×结局分层于第三列,MH 合并 OR+连续性校正 χ²+**未校正 Breslow-Day**(无 Tarone);ordinal 类别序取排序标签,γ 报 ASE1 的 95% CI(非 z 检验),显著性看 τ-b 的 p | 指定 factors/exposure/outcome/stratum/var1/var2 | ✅ 可配 loglinear `factors`;cmh `exposure`/`outcome`/`stratum`;ordinal `var1`/`var2` |
| 30 | **分布拟合 (distribution_fit/goodness_of_fit/qq_analysis)** | 单数值列(首连续,可 config);fit 候选 norm/lognorm/gamma/weibull/expon(数据≤0 跳正支撑分布并披露),MLE+AIC/BIC 选优;GoF 默认 norm,KS(估参→保守 Lilliefors)+AD(仅特定分布,抑制 scipy1.17 warning)+CvM(估参同保守)+Shapiro;QQ 用 Hazen 绘图位(i-.5)/n,对**拟合分布**的 Q-Q(斜率≈1/截距≈0),PPCC(非 Filliben 中位秩) | 指定 column;GoF/QQ 设 dist | ✅ 可配 `column`/`dist`(GoF/QQ,默认 norm) |
| 31 | **调节过程 (moderated_mediation/johnson_neyman)** | 角色从连续列按列序自动(Y=首连续,余为 X/M/W 或 X/W;**强烈建议 config 指定**,X-M-W 不对称);X/W 均值中心化;MM=Hayes model7,index=a3·b,条件间接(a1+a3(w-锚))·b 于 mean±SD,bootstrap B=2000 百分位 CI(每次重抽用各自中心化锚点),seed=20240607;JN θ(w)=b1+b3·w 方差由 cov_params,边界解二次式 t_crit(df=n−4),根映回原尺度,越界标外推 | 指定 x/m/y/w | ✅ 可配 MM `x`/`m`/`y`/`w`;JN `x`/`w`/`y` |
| 32 | **现代 DiD 稳健性 (goodman_bacon/honest_did/chaisemartin_did)** | 面板 unit/time/outcome + 处理时点(gname 或 0/1 treatment 推 onset,与 callaway 同);GB 平衡为相交面板、坏比较权重诊断、真 TWFE 系数=`twfe_did_direct`(分解和=`twfe_did_decomp` 简化权重重构、不强对齐);honest_did 事件研究(ref=-1、按单位聚类)、RM 单后期界 |bias|≤M̄×**最大前期一阶差分**(与 HonestDiD 包同尺度)、breakdown M̄、grid[0,.5,1,1.5,2]、无 never+全交错则诚实拒绝;dCDH DID_M 稳定对照(仅未切换)、单位 bootstrap seed=12345 | 指定 unit/time/outcome/gname 或 treatment;honest 设 window/post_period;dCDH 设 bootstrap | ✅ 可配 `unit`/`time`/`outcome`/`gname`/`treatment`;honest_did 另 `window`/`post_period`;chaisemartin 另 `bootstrap` |
| 33 | **生存补全 (time_varying_cox/stratified_cox/cox_ph_diagnostics, lifelines)** | TVC=计数过程长表(id/start/stop/event,event 仅事件区间=1),HR=exp(coef)、**无 concordance**(CoxTimeVaryingFitter 不提供);分层 Cox=各层独立基线风险+共同系数(无层×协变量交互);PH 诊断=scaled Schoenfeld(time_transform="rank"),per-covariate p(global_ph_p 为 nan 因 lifelines 只给逐项),p<0.05 标违背→建议 stratified/time-varying | TVC 指定 id/start/stop/event/covariates;分层指定 strata;诊断指定 duration/event/covariates | ✅ 可配 TVC `id`/`start`/`stop`/`event`/`covariates`;stratified `duration`/`event`/`strata`/`covariates`;diagnostics `duration`/`event`/`covariates` |
| 34 | **面板补全 (mundlak/hausman_test/first_difference, linearmodels)** | 实体/时间/结果/预测=econometrics 同惯例;时不变预测变量自动剔(FE 吸收/Mundlak 共线)并披露;Mundlak=RE 加协变量实体均值+对均值系数 clustered Wald(稳健版 Hausman);**hausman_test=PSD 门控**——V_FE−V_RE 正定用经典式(无调整协方差)、否则回退**回归式(Mundlak)Hausman**(无调整,与经典同基准、有限样本恒可定义);FD=FirstDifferenceOLS(去实体效应、需时变回归元、clustered SE) | 指定 unit/time/outcome/predictors | ✅ 可配三者 `unit`/`time`/`outcome`/`predictors` |
| 35 | **空间补全 (ripleys_k/getis_ord/join_count, 手搓 numpy/scipy/networkx)** | 坐标=config x/y 或 geo lon/lat 或前两连续列;Ripley K 边校正≈包围盒圆内比例(36 采样角)、CSR 蒙卡包络 seed 固定;Gi*=k-NN(默认8,含自身)、**Gi* 任意实数可用,全局 G 仅非负变量(含负值跳过+披露,改用 moran_i)**;join-count k-NN(默认6)对称化、BB/WW/BW 自由抽样 z(WW 与 BB 对称) | Ripley 设 x/y/n_sim;Gi* 设 value/x/y/knn_k;join 设 value/x/y/knn_k | ✅ 可配 ripleys_k `x`/`y`/`n_sim`;getis_ord `value`/`x`/`y`/`knn_k`;join_count `value`/`x`/`y`/`knn_k` |
| 36 | **回归诊断 (vif_multicollinearity/heteroskedasticity_test/influence_diagnostics, statsmodels)** | 结果=首连续、预测=其余(config 可覆盖);VIF 含截距设计、条件数=**含截距的单位长度缩放全设计**(BKW,>30 阈)；异方差=BP+White(元组序 lm/lm_p/f/f_p,White df 用辅助设计**实际秩**避二值平方丢列高估);影响点=Cook's D(4/n)/leverage(2p/n)/DFFITS(2√(p/n)),p 含截距 | 三者指定 outcome/predictors | ✅ 可配三者 `outcome`/`predictors` |
| 37 | **相对重要性 (dominance_analysis/relative_weights/commonality_analysis)** | 结果=首连续、预测=其余;dominance=全子集(2^p,p≤8)一般优势=Shapley、sum 到 R²;relative_weights=Johnson(标准化→R_xx=QΛQ'→正交对应物 Z→权重 Σ P²b²,sum 到 R²);commonality=Möbius 唯一+共性分解(2^p−1,p≤6,可负=抑制),C({j})=R²(全)−R²(全\{j}) | 三者指定 outcome/predictors | ✅ 可配三者 `outcome`/`predictors` |
| 38 | **有序回归 (proportional_odds_logit/ordered_probit/brant_test, statsmodels OrderedModel)** | 结果=3..10 级有序列(排序取序);proportional_odds=比例优势 logit、报 OR;probit=潜正态尺度系数;**阈值用 transform_threshold_params 还原为真切点**(OrderedModel `.params` 阈值是 [c1,log(c2-c1),…] 增量空间,切点 SE 需 delta 法故略);McFadden 伪R²+LR;brant=各切点二元 logit 的**近似** Brant(对角方差、忽略跨切协方差→保守)。注：基础 `ordered_logit` 在 statistics.py(已同步修阈值) | 指定 outcome/predictors;brant 同 | ✅ 可配三者 `outcome`/`predictors` |
| 39 | **密度/尾部 (finite_mixture/kernel_density/tail_index)** | 单数值列;finite_mixture=1D 高斯混合 EM、BIC 选 k(≤max_k,seed)、ΔBIC>10=多峰;kernel_density=scipy gaussian_kde,报 Silverman(0.9·robust)+**Scott(1.0·sd·n^-1/5=scipy 默认)**、网格找峰,数值 bw 覆盖=bw/std;tail_index=Hill 估计、稳定区(k 中段 20-60% 中位)、α<4=重尾、需正值尾(下尾翻号)、n≥50 | finite 设 max_k;kde 设 bandwidth;tail 设 tail/k_frac | ✅ 可配三者 `column`;finite `max_k`;kde `bandwidth`;tail `tail`/`k_frac` |
| 40 | **抽样设计 (weighted_estimation/poststratification/raking)** | 设计加权;weighted=HT 均值/总量+Kish deff/n_eff+**有放回设计 SE(始终不含分层/PSU 聚类,group 仅给分组描述均值不改 SE)**;poststrat/raking **必须 config 提供总体比例/边际、绝不杜撰**,缺则诚实跳过;raking=IPF 迭代到收敛、报 converged+边际误差+raked deff | weighted 设 value/weight/group;poststrat 设 strata/pop_props/weight/value;raking 设 rake_vars/margins/weight | ✅ 可配 weighted `value`/`weight`/`group`;poststratification `strata`/`pop_props`/`weight`/`value`;raking `rake_vars`/`margins`/`weight` |
| 41 | **经典 ANOVA (anova_oneway/ancova/repeated_measures_anova, statsmodels)** | 结果=首连续、组=低基数分类;oneway 报 F+η²+ω²+Levene(Brown-Forsythe 中位)+Welch 回退+Tukey HSD;ancova=OLS C(组)+协变量 Type II、调整均值在协变量**总均值**、斜率齐性查交互(协变量键用精确 `Q("名")` 匹配);RM=AnovaRM,Mauchly(χ² 一阶 Box 近似,已披露)+GG ε 还原(操作以 GG-校正 p 为准),不平衡被试丢弃+披露 | oneway 设 outcome/group;ancova 设 outcome/group/covariates;RM 设 subject/within/outcome | ✅ 可配 oneway `outcome`/`group`;ancova `outcome`/`group`/`covariates`;rm `subject`/`within`/`outcome` |
| 42 | **非参组检验 (kruskal_wallis/friedman_test/mann_whitney, scipy)** | KW=H(tie 校正)+**η²[H]=(H-k+1)/(n-k)**(rstatix eta2[H],非 Tomczak ε²)+Dunn(tie 校正方差+Bonferroni);Friedman=Q+Kendall's W+Nemenyi(studentized range,完整块,丢不全被试);MWU=U+**rank-biserial r=2U1/(n1n2)-1**(r>0=组1占优)+Hodges-Lehmann+分布无关 CI;均测随机优势(同形才=中位) | KW/MWU 设 outcome/group;Friedman 设 subject/within/outcome | ✅ 可配 kruskal/mann `outcome`/`group`;friedman `subject`/`within`/`outcome` |
| 43 | **监督 ML (regularized_regression/svm_model/gradient_boosting, sklearn)** | 结果检测同 ml.py(首连续→回归;**分类目标须 config outcome**,否则分类列当特征);全部 **CV 交叉验证(StandardScaler 在 pipeline 内、无泄漏;指标全留出样本非样本内)**;regularized=Lasso/Ridge/ElasticNetCV(嵌套选 α)报 cv_r2/RMSE+标准化系数稀疏;svm=SVC/SVR(rbf,C/gamma 默认不调)报 CV acc+macroF1 或 R²;gbm=GBM+**置换重要性(留出集、非杂质)** | 指定 outcome/predictors;regularized 设 method/alphas;svm 设 kernel/C/gamma;gbm 设 n_estimators 等 | ✅ 可配三者 `outcome`/`predictors`;regularized `method`/`alphas`/`l1_ratio`;svm `kernel`/`C`/`gamma`;gbm `n_estimators`/`learning_rate`/`max_depth` |

---

## 诚实降级 / 待办（装包或后端门槛）

- **bayesian_sem（贝叶斯 SEM）**：诚实降级。需 R `blavaan` + JAGS 或 Stan(C++ 编译)后端 + 理论测量模型；本机 blavaan 未装、无 RTools 编译器（brms 实测 `make not found` 无法编译）、无 JAGS。**不自动触发重型/易碎的工具链安装**。可运行替代：`sem`（频率派 CB-SEM，config model_spec）/ `efa`。**待办**：装 blavaan + JAGS（或 RTools/Stan）后，接 `bsem()` 真后验路径。
- **差异丰度 #9 的 ANCOM-BC**：桥待接（需 TreeSummarizedExperiment）；ALDEx2 已接。

---
**怎么用**：你回来逐条说「第 N 条改成 X」，或「都先这样」。我据此把对应分析升级成接受参数/换默认，并重跑。
（此清单由自走 loop 维护，新增方法若有新默认会续加。）
