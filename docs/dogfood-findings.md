# Wave J · Dogfooding 发现 + Wave K 修复规划

> 2026-07-10。6 个 persona 扮真实用户用引擎、**只用不修**、记摩擦。数据在 `e:/tmp/dogfood/`（含
> `ANSWER_KEY.md` 真效应量，供判读对答案）。已强互证的来自 P1 医学生(Likert 问卷) + P2 流行病学者
> (cohort)，各独立重跑 3 次结论收敛；P3-P6(面板/RCBD/churn泄漏/脏Excel)部分跑过（cleaned 文件为证）。
> Wave K 规划由 **Fable 5 参谋长**（`Agent(model="fable")`，2026-07-10 起可派）出，Opus 照办。

## 战略判读（Fable）
不是主战场判断错位，是**打完了没清扫**。C/E/G/H3/H4 修执行层"用对列"是必要功；但边际战场已转移到
**profiler 的 `count` kind 是语义垃圾桶**——同时装：真事件计数(Poisson)、有界评分(Likert)、普通离散量
(年龄/年份)。早先造了 `is_ordinal_like`/`has_rater_block`/`has_count_outcome`/`_RATER_FLOOR` 把"评分"从
"计数"剥出，但**只接进 2 个消费点，漏了 3 个仍吃裸 `count` 的路径**，且"离散量≠计数结果"这第二刀根本没切。
**Wave K = 把已造好的 ordinal/true-count 区分，穿透到所有仍吃裸 count 的选择路径，并补第二刀。**

## 互证发现（P1×3 + P2×3）
1. **推荐层系统性误路由（头号）**：Likert 问卷 → top 推 PERMANOVA/Indicator-species/Fleiss-κ(契合 85-92)，
   cronbach/EFA 沉到 rank 7+ 或不可达；二值 cohort → top 推 NB/ZIP(把整数 age 当过离散计数结果)，
   logistic/epi_risk_measures 全程不进 top6。
2. **`study` 旗舰照跑误路由方法、给自信胡话**：Fleiss-κ 把"250人×10列"当"10评分者×21类别"(κ=-0.006 无警告)；
   PERMANOVA 把满意度当群落组成。agreement/ecology 无前置门，"有数值矩阵"就跑。
3. **默认列选择悄悄坑人**：cronbach 默认"全部连续/计数"→把年龄当第9题(α 从 0.7 塌到 ~0)；
   logistic 默认预测变量只取 continuous/count → 悄悄丢二值暴露 smoking + 混杂 sex。
4. **承诺/产物错配**：logistic catalog 写 `produces: odds ratios`，产物只给 log-odds(β)，OR 要手动 exp。
5. 跨方法收敛信号靠字段名撞车比不相关 p_value；中文列名进 statsmodels formula 未做标识符守卫(R 桥有先例)；
   分类预测变量不自动哑变量化。

## P3 补充（经济学生/面板，计量·推断质量向，与上面根因桶不同）
6. **`pick` 生成的命令覆盖出比分支默认更差的结果**：pick 把 likely_outcome 塞进 `--config outcome=`，
   当它是低置信/误判时（面板里 size），反而覆盖了引擎本来对的默认（第一连续列 investment）→ 跟着工具给的
   命令跑反而错、不带 config 跑反而对。**反直觉，头号陷阱。** → pick 命令不应用比默认更差的猜测覆盖它。
7. **面板回归默认只 HC1、不按 unit_col 聚类 SE**：明知 `fp.unit_col`，`panel_fixed_effects`/`ols`/`did`
   共享分支仍 HC1 → p 值假性极小（1e-187），只 `random_effects` 分支做对了聚类。→ 检测到 unit_col 默认聚类。
8. **`ols_regression` 不感知面板**：同次 profiling 已判 `is_panel=True`，ols 偏差披露却只字不提、说"可常规解读"，
   pooled 偏差（+25%）静默通过。→ ols 偏差文本交叉核对 `is_panel`，检测到主动提示 FE。
9. **`study` 把崩溃方法计入"N/N 跑了"**：panel_qca R 崩仍算进 methods_run（Wave I judgment call #1 现被真实
   用户确认）。→ 总结行区分成功/失败计数。
10. rigor 排名不奖励更严谨 SE：random_effects(Hausman+聚类,做对)排在 panel_fixed_effects(HC1,错)之下。
> ⚠ 6-10 是 P3 新增、Fable 的 Wave K 规划成于 P1/P2 之前——**这几条需 Fable 参谋长回炉折进 Wave K 优先级**
> （7/8 是"面板感知"一族、便宜高价值；6 是 pick/nudge 层；9 接 Wave I methods_run 根治）。

## P4 补充（农学/RCBD，角色层——中文列名 + 分类因子，新主题）
11. **角色检测在中文列名上静默反转（阻断级）**：`run rcbd` 不带 config 把 **区组当 treatment、处理当 block**——
    `_BLOCK_HINTS`/`_TRT_HINTS`(`experimental_design/_shared.py`、`field_trials.py` 两处独立定义)只认英文子串，
    都不中时按列声明顺序兜底、**无披露**；真处理效应 F=46,p≈1e-21 被埋没。
12. **`group_comparison` 反语义挑分组（阻断级·最危险单点）**：`cat_cols.sort(key=nunique)`(`statistics.py:547`)
    纯按类别数最少取首个 → 选区组(4级)而非处理(5级)→ 假阴性 p=0.156(真 p≈1e-21)，**无"角色自动猜测"披露**。
13. **`mixed_effects` 吞掉多水平分类固定效应（阻断级）**：predictors 过滤器只收 continuous/count/binary，
    categorical 处理因子被丢 → 退化纯截距、照打"完成"(`branches/statistics.py:711`)。→ 退化到零预测变量必须报错。
14. **分类预测变量不自动哑变量化**（与 P1 finding #5、K2⑤同根）：proportional_odds/logistic 等遇分类协变量
    直接"含非数值无法转矩阵"跳过——性别/处理这类协变量是刚需。
15. RCBD 双实现、config key 不一(`rcbd` 用 outcome / `rcbd_anova` 用 response)；pick/recommend(含 `--goal design`)
    从不把 rcbd 排第一，即便数据是其教科书场景。
> ⚠ **11-14 是新根因层（角色检测=英文-only + 反语义启发式 + 分类因子被丢），与 count 桶(K0/K1)平行、同等重要。**
> 中文命名是本项目核心用户群（CLAUDE.md 自承 Windows 中文研究者），角色层英文-only 是系统性失败。

## P6 补充（行政/脏 Excel，非专家视角——结构过度检测 + 无人话结论）
16. **结构过度检测压过组间比较**：有日期列 → 判"时序" → `pick`/`study` 无 goal 推 ARIMA/VaR/EVT(金融黑话)、
    只字不提区域；`recommend --goal compare` 里真正对口的 `group_comparison` 排**第 8 名**(top-6 看不到)，
    前 6 全是 DoE 怪兽(factorial/split-plot/RCBD/Latin/AMMI)。观测数据无实验设计信号却让复杂 DoE 系统压过朴素比较。
17. **`factorial_anova` 嵌套因子静默 NaN（统计错+丑）**：城市**嵌套**于区域(非交叉) → 区域行 `F=nan,p=nan`，
    报告仍"完成"，还把 statsmodels "covariance...not full rank" 警告泄漏到终端。→ 检测完全嵌套两分类列应拒跑/降级。
18. **组间比较结果不合成人话结论**：`study --goal compare` 算对了(华东 F=51.65,p=3.7e-22)但"关键看点"写
    "本次无实质结论行可供提炼"——判出显著却不说"华东均值最高、高 ~20%、显著"，用户得自己开 CSV/看图。
    → group_comparison 类应自动合成一句结论。**这是 P6 唯一能达成"可信一句话"判据的缺口。**
19. **likely_outcome 位置启发式选中常量列**：原始文件上提示"结果变量=年份"(常量 2024)——最没意义的数字列。
    → last-numeric 启发式遇常量列应跳过。
20. **✅ 正面（clean 层是强项）**：`clean --apply` 逗号数字("1,011.6"→1011.6 非误读 1.2345)/常量列删/93%缺失列删/
    10 重复行精删/稀有城市→Other/**日期诚实标"像标识符"不硬解析**——全对。且分支自己会解析逗号数字，
    多数分析可跳过 clean 直跑（值得写进文档当卖点）。

## P5 补充（ML 工程师/churn 泄漏，最严重诚实缺口 + 两个真 bug）
21. **无泄漏检测器，引擎把泄漏结果 narrate 成"诚实估计"（headline·最危险）**：refund_amount 是结果后泄漏特征
    (只有 churner 有退款)，gradient_boosting/discriminant/logistic **默认全吃** → cv_acc 0.975-1.000、单特征
    importance 碾压其余，报告写"诚实泛化估计"、**零警告**。profiler 无任何 leakage/too-good-to-be-true 检查(grep 过)。
    → 便宜高价值：cv_acc>0.97 / 单特征 importance 碾压 / logistic PerfectSeparation / SE≫coef 时，出一等 ⚠
    "结果可疑地完美——查是否信息泄漏(尤其结果发生后才产生的字段)"。
22. **`ml.py` random_forest/xgboost config-outcome override BUG（真 defect，连 H4 接线）**：即便
    `--config outcome=churn`，仍 regress on tenure——"prefer continuous outcome" tier 在 resolve_outcome 之前，
    churn 不在 cont 候选里、config 被静默吞。违反 CLAUDE.md config 契约(override 应赢)。`ml_supervised._resolve_xy`
    做对了、`ml.py:518-650` 没有。→ config["outcome"] 检查须在 cont-vs-binary tier 决策**之前**。
23. **`entry_matches_goal` 忽略 `entry.goal`（goal 路由 bug）**：gradient_boosting 声明 `goal: predict` 却对
    `--goal predict` 完全不可见(`recommender/goals.py:24-26,64-71` 用 whitelist+字面关键词，description 无"predict"字样)——
    最该服务该目标、且唯一正确 honor config+分层 CV+报 F1 的方法反而消失。→ entry_matches_goal 应先信 entry.goal。
24. **logistic 完美分离不动态诊断**：SE 爆到 1.9e8、p≈1，报告只有静态 `biases: perfect separation` boilerplate
    (每次都印，非诊断)、无动态"没收敛"flag。→ 动态检测 SE≫coef/p≈1 pattern 绑回"未真正收敛"。
25. gamm 作 `--goal predict` top pick 但截面数据无分组变量直接失败(阻断)。
26. **✅ 正面**：customer_id(id-kind)正确排除、churn(第5列)高置信检测**对**——检测对，断在**binding 到执行**(#22)。
    手动排除 refund_amount 后 logistic 近乎精确恢复真 DGP(tenure -0.068 vs 真 -0.08…)——底层统计引擎可信。
> ⚠ **21 是新根因线 F（泄漏/可疑完美=零检测，最伤"诚实"招牌）；22/23 是可直接修的真 bug（config 契约+goal 路由）。**

## 待办：Wave K 需 Fable 回炉定稿
Fable 的下方 Wave K 规划成于 P1/P2，**未含 P3-P6(6-26)**。**6/6 persona 全到齐。下一步应让 Fable 参谋长读齐
全部 26 条发现、重排 Wave K**。六条根因线：
A. count 语义桶(P1/P2) · B. 角色层英文-only+反语义+分类因子丢(P4) · C. 结构过度检测(时序/DoE/goal 路由压过朴素, P3/P6) ·
D. 面板/推断质量(SE/pick 命令, P3) · E. 执行层诚实(OR/丢暴露/嵌套 NaN/无人话结论/完美分离, P1/P2/P6) ·
**F. 泄漏/可疑完美=零检测 + config 契约/goal 路由真 bug(P5，最伤诚实招牌)**。
**头号元判据（6 persona 印证）：分析层可信、clean 层是强项，但"自动挡"(recommend/pick/study 自动选法+角色检测)
在最常见数据形态上系统误路由、常静默给自信错答——最狠是 P5：把数据泄漏的 100% 准确率 narrate 成"诚实估计"。
直接戳北极星"自动选模越聪明"，对新手是帮倒忙。修选择/角色/诚实-防护层，优先于加任何新方法。**


## Wave K 修复规划（Fable，可直接派工；下面成于 P1/P2，尚未含 P3 的 6-10）
**代码坐标**：`recommender/affinity.py:145-217`(信号定义+漏接点) · `scoring.py:189`(min_count_cols 裸门) ·
`match.py:51-65`(可行性门=K0/K1 主战场) · `profiler/types.py:62-84`(`is_ordinal_like`，只消费别动) ·
`executor/branches/regression.py`(logistic OR+预测变量+标识符守卫) · `branches/psychometrics.py`(cronbach 选题) ·
`catalog/entries/{ecology,count_models,agreement}.yaml`。

- **K0 根因·一改多治·支点**：把"真计数 = `count ∧ ¬ordinal_like` 且是结果非协变量"穿透 3 个漏接点：
  ① `affinity._available_outcomes` 用 `has_count_outcome` 不用裸 `has_count`；② `scoring._precond_bonus` 的
  `min_count_cols` 门挂 `has_count_outcome`；③ **`match.py` 的 `requires_count_outcome`/`min_count_cols`
  可行性门排除 `ordinal_like` 列**（关键——indicator_species/mantel/rda/zip 在 Likert 上判 feasible 的根子）。
  一个概念、~4 处 edit + 回归测试。堵门后 study 自然不选它们。
- **K1 第二刀·需判断**：`count` 列只有当**是 likely_outcome 或列名带 count/n_/events/cases** 时才算计数结果
  (喂 Poisson/NB/ZIP)，否则是协变量。收口进 `has_count_outcome` 定义。防误伤真 Poisson=纯判断活。
- **K2 便宜·高价值·机械可并行**：① logistic 出 `exp(β)`+95%CI(OR)；② cronbach 默认选题=ordinal_like 列；
  ③ logistic 默认预测变量纳入 binary+分类(哑变量化)；④ 中文列名进 formula 前抄 R 桥标识符守卫+映射回原名；
  ⑤ 分类预测变量自动哑变量化(同 formula 批)。
- **K3 纵深防御·执行层前置门**：agreement/ecology 处理器自检结构信号，缺则拒跑或带 ⚠ 降级(fleiss 在
  respondents×items 上点明、不无声输出 κ)。K0 落地后 study 已不选它们，故优先级低于 K0，但 study 是旗舰仍要做。
- **by-design 不该动**：① rater-block floor 同抬 cronbach(对)+fleiss(存疑)是真·可识别性极限
  (respondents×items ≡ raters×targets 同形)——别堆启发式硬压 fleiss，交 K3 诚实警告；② `is_ordinal_like`
  保持 `count` 上的 flag、**别升新 kind**（波及 60+ 分支裸判）——穿透 flag，别 re-kind。
- **必配护栏**：用 P1(250×10 Likert)、P2(二值 cohort)真 fixture 写**选择顺序黄金测试**——断言
  logistic/cronbach/factor 进 top-N、ecology/count-model 不 feasible。防回归的锁。

**分层派工**：Opus 亲做(判断+过 inference-reviewer)=K0 概念收口设计 / K1 计数判定规则 / K3 拒跑-vs-警告语义；
Sonnet 机械=K0 的 4 处 edit+黄金测试 / K2 整批(可 fan-out) / fixture 黄金测试。
**顺序**：K0 先(支点)，K2 并行(不依赖 K0)，K1 紧随 K0(同收口点)，K3 收尾。本地 commit，等"今天 ok"再 push。
