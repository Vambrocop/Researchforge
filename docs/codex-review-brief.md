# Codex 审查任务书（Review Brief）

> 给独立审查者（Codex / GPT 系，冷启动新视角，审查者≠建造者）。在**仓库根目录**打开,读完本文件,按顺序看 + 跑命令,最后按「交付格式」回报。
> 两块要审:**A 之前的(本轮代码改动)** 和 **B 接下来的(路线计划)**。

---

## 0. 项目是什么(30 秒)
ResearchForge = **方法学大杂烩引擎**:丢数据 → 自动识别类型/质量 → 推荐可行分析(带 🟢🟡🔴 严谨度 + 偏差披露 + 6 维评分卡) → 自动执行出代码/图/表/报告。已 70 个分析。**先读 `CLAUDE.md`(项目宪法)**——所有约定/红线在那。当前 v0.7,自评分卡 89/100。

先跑一眼活体现状:
```
py -3 -m researchforge.cli status
```

---

## A. 审「之前的」——本轮改动

**范围**:本会话 12 个 commit，`git diff a6e2f68..75768b1`（HEAD）。主线是**把 7935 行的 `run.py` 巨石拆解** + 新增 conformal_prediction + 测试提速 + status 前门。

### ⚠ 范围边界(省你时间,务必看)
- **大头是「逐字搬运」,不是新逻辑**:`run.py` 7935→149 行,66+ 个分析分支的**函数体原封不动**搬进 `executor/branches/<family>.py`,helper 搬进 `executor/_helpers/{core,backends}.py`。**别逐方法重审这些方法体的统计正确性**——它们是已上线代码的relocation。只需判:**搬运是否保真**(行为有无被改)+ 架构是否合理。
- **已审过、别重复**:① conformal_prediction 已派 inference-reviewer,抓 1 must-fix(小校准集下覆盖保证不可达却误报"达标")已修;② `cli status` + `scorecard` 路径已过 /simplify 4-agent 清扫(3×遍历→1)。求你**新视角**,不是复述。

### A1. 先读懂新架构(入口文件)
- `researchforge/executor/_branch_api.py`(59 行)— `Ctx` / `register` / `BRANCH_REGISTRY`,分发基座。
- `researchforge/executor/run.py`(149 行,**整读**)— setup → 注册表分发 → teardown + 从 `_helpers` re-export helper。
- `researchforge/executor/branches/__init__.py` — `pkgutil.walk_packages` 自动发现注册。
- `researchforge/executor/_helpers/{core,backends}.py`(头部 + 结构)— core=计算/绘图/纯 Python 方法;backends=R 桥/econml/doubleml/semopy 委托。
- 抽一个 family 看搬运范式:`researchforge/executor/branches/regression.py`(最短,51 行)+ `soil.py`。

### A2. 重点审(真·新逻辑 / 设计决策)
| 文件 | 看什么 |
|---|---|
| `_branch_api.py` + `run.py` 分发 + bottom import | 注册表设计、**循环导入安全性**(run.py 末尾 import branches、branches 从 run re-export helper)、re-export 显式列表 vs `*`(我们故意不用 `*`,因 helper 全下划线会被泄漏) |
| `executor/_helpers/core.py` 的 `_conformal_prediction` + `branches/ml.py` 的 `_branch_conformal_prediction` + `tests/test_conformal_prediction.py` | 分裂保形预测正确性(已 inference 审过,你做交叉确认):分位 `ceil((n_cal+1)(1-α))`、小校准集降级披露 |
| `researchforge/quality/scorecard.py`(`_measure` 单遍走 + `large_modules` + `DIM_LABELS` + `MODULE_LINE_LIMIT`) | 度量逻辑;`_measure` 返回 `(dict, large)` 改动面;单遍走是否漏算 |
| `researchforge/cli.py` 的 `_cmd_status` | 组合 scorecard+git+roadmap 的活体状态;markdown 抓取健壮性 |
| `tests/conftest.py`(`SLOW_MODULES` + collection hook) | slow 分层是否合理;集中清单 vs 逐文件 pytestmark 的取舍 |
| `tests/test_module_size.py` | ≤1500 行护栏(防巨石复发) |

### A3. 给 Codex 的具体问题(A)
1. 注册表/分发架构有没有隐患(导入顺序、注册时机、重复注册)?可扩展到 100+ 方法吗?
2. 搬运保真:有没有哪个分支在迁移中被悄悄改了行为?(ctx 解包后跑原体——分支只 mutate `summary/estimates/files/code` 从不 rebind,是迁移正确的前提,帮我复核这个前提是否真成立)
3. `run.py` re-export 的 60 个名字手列——可维护性?有更好的稳定导入面方案吗(但别建议 `*`,理由见上)?
4. conformal 的统计与披露,你独立判 correct 吗?
5. 任何正确性 bug(这是重点,/simplify 只管质量不抓 bug)。

---

## B. 审「接下来的」——路线计划

**读这三个**:
- `docs/roadmap.md` — 战略阶梯:**厚 1.0(B 方案)** → v1.1-1.5 → v2.0 → ∞(持续优化)。用户大格局派,选了"1.0 前就把高需求方法波折进去"。
- `docs/deferred-log.md` — 顶部「🔜 下一波」+ 方法/工程 backlog + 好点子(含多个 inference-reviewer 提的优化)。
- `docs/scorecard.md` — 评分卡趋势(最弱维 = 可用 58 / 快速 70,指向要改的)。

### 给 Codex 的具体问题(B)
1. **厚 1.0 的方法波次顺序合理吗?** 现有 70 法(回归/因果/SEM/空间/生态/MCDA/效率/生存/时序),roadmap 要补实验设计/测量/因果扩张/ML——优先级对吗?有没有更该先做的高需求方法?
2. **漏了哪些常用方法**(尤其农学/社科/生物医学高频的)?
3. **"先 web UI 还是先方法"**——用户选 B(方法折入 1.0)。你认同吗?有没有更优排序?
4. 计划里的**风险/隐藏门槛**(我们已标"真实脏数据鲁棒"为隐形门槛——还有别的吗)?
5. 自我进化(discover 真抓取 阶段2)、趋势引擎的设计有没有坑?

---

## 红线 / 审查标准(照这些判,摘自 CLAUDE.md + 工作doctrine)
- **架构**:一分析 = 一模块(`branches/<family>.py` 的 `@register`,**别往 run.py 加 elif**);family 逼近 ~1200 行就提升为包;**模块 ≤1500 行**(有 `test_module_size` 护栏)。
- **R 后端**:可选 + **优雅降级**(缺 R/包→回退纯 Python 或诚实提示);列名进 R formula 前过标识符守卫 `re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", c)`(防注入);temp CSV `finally` 删;**R 代码经审才接,运行时不联网取**。
- **默认能独立跑**:`config` 覆盖实质默认,缺则回退自动默认(默认必须仍能跑)。
- **产物**:matplotlib **图标签用英文**(默认字体无 CJK);填 `estimates`;写中文 `summary` 含 **⚠ 偏差/假定披露**;缺包 best-effort 不中断。
- **诚实门禁**:零结果照报、不编数、不确定标 ⚠;golden/oracle 类文件不手改历史;不绕过 pytest/门禁。
- **流程**:真统计推断方法须**双审(审者≠建者)**;push 门禁(用户说"今天 ok"才推)。

## 怎么跑(Windows)
```
py -3 -m researchforge.cli status            # 活体现状
py -3 -m researchforge.cli recommend <data.csv>
py -3 -m pytest -n 2 -q                       # 全量 ~2:49（别用 -n auto：R worker 重会 MemoryError）
py -3 -m pytest -m "not slow" -q              # 快循环 ~51s（跳重模型测试）
```
环境:`py`(非 python)、`$env:PYTHONUTF8='1'`、PowerShell 5.1(无 `&&`)。R 已装(lavaan/QCA/gstat/mgcv/lme4/metafor/JM/dbarts… 但无编译器→Stan 类不可跑)。

## 交付格式
分 **A(代码)** 和 **B(计划)** 两段。每条发现:`文件:行` · 严重度(**must-fix** / nice-to-have) · 一句症结 + 具体改法。代码以**正确性 bug** 和**架构可扩展性**为主(质量/简化已做过);计划以**方法优先级/缺口/风险**为主。建造者可在有证据时驳回(会说明)。
