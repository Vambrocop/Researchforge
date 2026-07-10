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

## 六条根因线 + 头号元判据（6 persona 印证）
A. count 语义桶(P1/P2) · B. 角色层英文-only+反语义+分类因子丢(P4) · C. 结构过度检测(时序/DoE/goal 路由压过朴素, P3/P6) ·
D. 面板/推断质量(SE/pick 命令, P3) · E. 执行层诚实(OR/丢暴露/嵌套 NaN/无人话结论/完美分离, P1/P2/P6) ·
**F. 泄漏/可疑完美=零检测 + config 契约/goal 路由真 bug(P5，最伤诚实招牌)**。
**头号元判据：分析层可信、clean 层是强项，但"自动挡"(recommend/pick/study 自动选法+角色检测)在最常见数据
形态上系统误路由、常静默给自信错答——最狠是 P5：把数据泄漏的 100% 准确率 narrate 成"诚实估计"。
直接戳北极星"自动选模越聪明"，对新手是帮倒忙。修选择/角色/诚实-防护层，优先于加任何新方法。**

---

## Wave K 定稿（Fable 5 参谋长，2026-07-10 读齐 26 条 + 核对 15 处坐标后重排；可直接派工）
**总判断：F 线不"排队"而"并行第一梯队"**——F1/F2 两个真 bug 契约级、各 <10 行、有正确范式照抄、零依赖，性价比全场最高；F3 泄漏检测器紧随（"诚实背书的假完美"用户会信，击穿立身之本）。F 批与 A 批文件零重叠，插队即并行。**诚实异议**：原 K3(agreement/ecology 前置门)降级并入批 4 收尾（A1 堵门后 study 已不选它们，只剩手动 run 踩坑）；原 K2 的哑变量化升格为 **B4 统一 formula helper**（P4 的 13/14 证明是阻断级、非 nice-to-have）。

**批 0 · 验收锚（先行，所有批的锁）** — `K-0` 6 persona 合成 fixtures 进 `tests/fixtures/`（小型化，**别引用 e:/tmp 会消失**，照 ANSWER_KEY.md 造）+ 黄金选择测试骨架（P1→cronbach/EFA 进 top-N 且 ecology/count 不 feasible；P2→logistic/epi；P4→rcbd 角色对；P5→config outcome 必 bind）。**先红后绿**，初始允许 xfail、后续批逐个转绿。〔Sonnet；只加测试不改引擎〕

**批 1 · F 线真 bug + 检测器（与批 2 并行）**
- `K-F1` `ml.py` rf/xgboost 在 tier 决策**前**加 `forced_y=cfg.get("outcome")`，照抄 `ml_supervised.py:65-69`。〔Sonnet 机械免双审；STOP：别顺手改特征选择/exclude，记 deferred〕
- `K-F2` `goals.py` `entry_matches_goal` 先信 `entry.goal`。**Fable 已判：只接 predict↔predict 一条**（entry.goal 值域 describe/explain/predict 与 goal keys 仅 predict 重合，别把 explain 泛化映射到 relate/causal 会大改路由）；顺手 gamm entry 加分组 precondition(发现25)。〔Sonnet；**碰 golden ratchet** 更新断言；STOP：想建 explain→relate 映射就停下报告〕
- `K-F3` 可疑完美动态诊断 helper（合并发现 21+24）：`cv_acc>0.97`/单特征 importance 碾压/`SE≫coef 且 p≈1`→一等 ⚠「结果可疑地完美——查泄漏字段」/「完美分离未真收敛」。铺 ml.py+ml_supervised+regression logistic。〔**Opus 定阈值语义→Sonnet 铺点→inference-reviewer 冷审**；helper 进**新文件 `_helpers/diagnostics.py`（别塞 core.py，尺寸金丝雀）**；STOP：只做保守三条〕

**批 2 · A 线支点（与批 1 并行；A2 依赖 A1 串行同收口点）**
- `K-A1` `affinity._available_outcomes` 用 `has_count_outcome`；`scoring.py:189` 门挂它；`match.py:51-58` 两处可行性门排除 `ordinal_like`。〔Opus 已定概念，Sonnet 4 处 edit；**碰 golden ratchet**（PERMANOVA/ZIP 从 Likert 消失，批0 断言转绿）；STOP：某 yaml `min_count_cols` 语义存疑就列清单报告〕
- `K-A2` count 列仅当「是 likely_outcome 或列名带 count/n_/events/cases」才算计数结果，收口进 `affinity.py:191` 的 `has_count_outcome`。〔**Opus 亲做**（误伤真 Poisson=判断活）+ 真 Poisson fixture 验证；STOP：任何让现存 Poisson/NB 测试转红的判定先报告〕
- `K-A3` likely_outcome 位置启发式跳过常量列(`n_unique>1`)。〔Sonnet 极小，`profiler/roles.py` 一处〕

**批 3 · B 线角色层（阻断级 ×3；B4 helper 设计先行）**
- `K-B1` 两处 `_BLOCK_HINTS`/`_TRT_HINTS`(`field_trials.py:36-37`、`experimental_design/_shared.py:10-11`)各加中文子串(区组/重复/处理/品种/剂量/水平…)+兜底命中时 summary 加「⚠ 角色按列序猜测，可 config 覆盖」。〔Sonnet；**STOP：别合并两处定义**，统一化记 deferred〕
- `K-B2` `statistics.py:547` group_comparison：选组先按「非 block-hint 名」分层，nunique 排序降为层内 tie-breaker+披露+接 `config["group"]` 覆盖键(记 loop-decisions)。〔Sonnet 建，**Fable/Opus 审启发式分层**；新增 RCBD 执行测试；STOP：发现其他调用方依赖现排序即停〕
- `K-B3` `statistics.py:711` mixed_effects：categorical 固定效应哑变量化进模型；**退化到零预测变量必须报失败**不许打"完成"。〔Sonnet 建→**inference-reviewer 冷审**〕
- `K-B4` 统一 formula 收口 helper(守卫非法/中文标识符+映射回原名，抄 rbridge 先例；分类预测自动 `C()`/哑变量化)。**两 commit**：B4a=helper+核心 3 处(logistic/ols/mixed_effects)；B4b=普查铺点(照 H4 大普查打法)。〔**Opus 设计接口**→Sonnet 铺点；helper 进 `_helpers/diagnostics.py` 或独立模块；STOP：普查>15 消费点报回分批。⚠ `statistics.py` 现 1021 行，B2/B3 加码后**逼近 1200 软顶即报告预拆别自拆**〕

**批 4 · E 线执行层诚实（机械，全并行）** — `K-E1` logistic 出 OR+95%CI〔Sonnet→inference-reviewer〕· `K-E2` cronbach 默认选题=ordinal_like〔Sonnet〕· `K-E3` logistic 默认预测纳入 binary(分类随 B4b)〔Sonnet〕· `K-E4` factorial_anova 检两分类列完全嵌套→拒跑/降级+吞警告进 summary(发现17)〔Sonnet〕· `K-E5` group_comparison 显著时合成一句人话结论(发现18)〔Sonnet；STOP：只做比较族别泛化〕· `K-E6` study 的 methods_run 分成功/失败计数(发现9)〔Sonnet〕· `K-E7` agreement/ecology 前置自检+⚠ 降级(原 K3 收尾；fleiss 保持诚实警告不硬压=by-design)〔Sonnet〕

**批 5 · D 线面板/推断质量（D1 审重）** — `K-D1` 检测 `fp.unit_col`+is_panel→panel_fixed_effects/ols/did 默认按 unit 聚类 SE(参照 random_effects 已做对的)+披露〔Sonnet 建→**inference-reviewer 强制审 SE 语义**；更新 p 值断言前列清单报告〕· `K-D2` ols 偏差文本交叉核对 is_panel、主动提示 FE〔Sonnet〕· `K-D3` pick 命令仅 **high-confidence** likely_outcome 才写进 `--config outcome=`(复用 `resolve_outcome` 的 medium/low 不 bind 原则，发现6)〔Sonnet〕· `K-D5` rcbd/rcbd_anova config key 统一(双键向后兼容，发现15)〔Sonnet〕

**批 6 · C 线结构过度检测（最需判断、动排序、殿后）** — `K-C1` `--goal compare` 下朴素 group_comparison 不被 DoE 族压制：观测数据(无 block/trt 命名信号)时 DoE 族 affinity 降权或 compare ids 提权(发现16后半)。〔**Opus 亲做**排序权衡→**golden ratchet 全量复验**；只动 goal=compare 路径〕

**明确砍到 Wave L（诚实异议，代价写明）**：① 无 goal「日期列→全推时序」根治(发现16前半)——需 ColumnSemantics 层，K-C1 修通 `--goal compare` 后有逃生门；② rigor 奖励聚类 SE(发现10)——K-D1 落地后排序倒挂自消解大半；③ profiler 级泄漏预扫——K-F3 事后检测先兜住；④ 跨方法收敛信号字段名撞车(发现5)——E5 先点状兜人话。

**依赖与并行**：批0(先行)→[批1(F1∥F2∥F3) ∥ 批2(A1→A2, ∥A3)]→批3(B1∥B2∥B3, B4a→B4b)→批4(E 全并行)→批5(D)→批6(C1)。碰 golden ratchet：F2/A1/A2/C1(串行过 ratchet 每次更新断言)。碰尺寸护栏：statistics.py(1021→留意1200)、新建 `_helpers/diagnostics.py`(别塞 core.py)。**四处必审**(inference-reviewer)：F3/B3/E1/D1。约 20 commit，Sonnet ~15 机械项可 fan-out，Opus 亲做 3 处判断(A2/B4a 接口/C1)。本地 commit，等"今天 ok"再 push。

## Wave L+ 滚动路线（Fable 5）
- **Wave L · 语义层扎根（防护加固）**：ColumnSemantics 统一层——把散落的角色检测(outcome/treatment/block/group/id/date，中英双语+置信度)收成一个架构件，给 B 线 hints 补丁和 C 线结构判定一个家；回收无 goal 智能路由、profiler 泄漏预扫、rigor 聚类奖励、收敛信号字段语义化。**理由：Wave K 是止血，B/C 两线补丁都在喊同一个架构缺件，不建它下轮 dogfood 还会长同类发现。**
- **Wave M · 验收+固化**：dogfooding 二轮(同 6 persona 场景重跑，验 26 条清零、逮新回归)+ dogfood fixtures 升格常驻选择回归 suite + 把"clean 层直跑"写进文档当卖点(发现20)。
- **Wave N · 方法扩张恢复**：回 melting-pot 清单(MCDA/DEA/SFA/面板计量/QCA 族)，让 `discover` 接推荐质量信号。**理由：先聪明再多——在自动挡失信的引擎上加方法是给误路由扩弹药。**

## 一句话元判断（Fable 5）
修完 Wave K，自动挡从「系统误路由+自信胡话」升到「常见数据形态(问卷/cohort/面板/RCBD/churn)选得对、config 契约必兑现、错和可疑时会自己说话」——**从演示级到有监督可用级：引擎变得"错得诚实"**；还差 Wave L 的 ColumnSemantics 才能"选得聪明"（中文角色是补丁非架构、泄漏只有事后启发式、无 goal 全自动路由仍会过度检测），那之前别宣称无监督可信。
