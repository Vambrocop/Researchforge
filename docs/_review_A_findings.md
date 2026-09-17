# 冷审 A — 角色绑定线（处理变量侧）· 审查者 ≠ 建造者

范围：`8971c50..72667cd` + `81144be`。只读审查，未改任何代码。
所有数字均为实测；每条给最小可复现代码。

共用最小数据集（下文引用为 `arm_frame`）：

```python
# 真值：接受处理 → severity 降低 8；age 是混杂（同时影响处理概率与结局）
import numpy as np, pandas as pd
def arm_frame(arm_name="treated", n=600, seed=3, invert=False):
    rng = np.random.default_rng(seed)
    age = rng.integers(20,70,n); bmi = rng.normal(26,4,n).round(2)
    treated = (rng.random(n) < 1/(1+np.exp(-(0.03*(age-45))))).astype(int)
    y = 50 - 8*treated + 0.2*age + 0.5*bmi + rng.normal(0,5,n)
    col = (1-treated) if invert else treated
    return pd.DataFrame({"severity": y.round(2), arm_name: col, "age": age, "bmi": bmi})
# 正确绑定时的参照：IPW ate = -7.652, PSM att = -7.564
```

---

## Q1 · 还有哪些列名会被错误绑定成处理变量

### MUST-FIX 1 — NEVER 规则只降权、不否决：control/placebo 仍会被绑定，符号翻转

**症状**　建造者声称 control|ctrl|placebo|comparator|sham 一律 strength 0「覆盖一切」。
实际上 strength 0 只把该列踢出 `resolve_treatment` 的 **tier 2**（命名列）。
tier 3（`fp.likely_treatment`）与 tier 4（第一个非 outcome 候选）**没有任何强度过滤**，
所以当对照标记是**唯一/第一个**二值列时，它照样被绑定，**且零警告**。
`roles.detect_roles` 有同一个洞（`elif treat_cands: likely_treatment = treat_cands[0]`）。

**复现**
```python
from researchforge.profiler import profile_dataset
from researchforge.executor import run_analysis
from researchforge.catalog import Catalog
df = arm_frame("control_arm", invert=True)     # 只有这一个二值列
df.to_csv("d.csv", index=False)
res = run_analysis(profile_dataset("d.csv"), Catalog.load().by_id("ipw"), output_root="out")
print(res.treatment, res.estimates["ate"])
```

**实测**（真值 −8；正确绑定 = −7.652 / −7.564）

| 列名 | strength | 绑定 | IPW ate | PSM att | |
|---|---|---|---|---|---|
| control_arm | 0 | control_arm | **+7.652** | **+7.042** | 符号翻转 |
| placebo | 0 | placebo | **+7.652** | **+7.042** | 符号翻转 |
| untreated | 0 | untreated | **+7.652** | **+7.042** | 符号翻转 |
| treated（参照） | 2 | treated | −7.652 | −7.564 | 正确 |

报告原文（`control_arm`，`likely_treatment_confidence` = low）：

```
💡 本方法把 'control_arm' 当作处理/暴露变量——如需改用其他列，用 config treatment 指定。
… ATE=7.6519（HC1 SE=0.5001, p=7.62e-53，显著）；重叠 良好。
```

p = 7.6e-53 的「显著、自信、符号相反」。现有测试
`test_a_control_flag_does_not_flip_the_sign_end_to_end` 之所以绿，是因为它的 frame 里
**同时存在** `treated`——只钉住了排序，没钉住「唯一候选」这一支。

**根因**　`researchforge/executor/_helpers/core.py:136-144`：NEVER 规则活在
`treatment_name_strength` 里，只影响 `named` 列表；tier 3/4 直接取列，不过滤。

**建议修法**（推荐 b，或 a+b）
- (a) tier 3/4 取列前先滤掉 `_TREATMENT_NEVER_RE` 命中的列；全被滤光时再退回它们（否则
  「只有对照标记」的数据会直接降级），但必须附警告。
- (b) `run.py` 处理变量披露块加硬警告：
  ```python
  from researchforge.profiler.semantics import _TREATMENT_NEVER_RE
  if _TREATMENT_NEVER_RE.search(bound_treatment):
      _tnote += "；⚠ 该列名像**对照/安慰剂**标记——若 1=对照，上面的效应**符号与处理效应相反**"
  ```

---

### MUST-FIX 2 — 否定式处理词拿到 strength 2，并**压过**正确命名的 `treated`

**症状**　强词表 `_TREATMENT_BIND_RE` 对**否定前缀**毫无感知。带分隔符的否定写法一律 strength 2。

**复现**
```python
from researchforge.profiler.semantics import treatment_name_strength as s
for n in ["no_treatment","non_treated","non_exposed","never_exposed","treatment_naive",
          "baseline_arm","dose_0","pre_treatment","post_treatment","non_treatment",
          "untreated","unexposed","nontreated"]:
    print(s(n), n)
```
输出：前 10 个全是 **2**；`untreated / unexposed / nontreated` = 0——后三个得 0 纯属**巧合**
（词根前没有分隔符，`\b` 不成立），不是设计。同一语义类，错误剖面自相矛盾。

**实测 A：否定列是唯一的处理标记**（`arm_frame(col, invert=True)`，真值 −8）

| 列名 | strength | IPW ate | PSM att |
|---|---|---|---|
| no_treatment | 2（conf=**high**） | **+7.652** | **+7.042** |
| non_exposed | 2 | **+7.652** | **+7.042** |
| treatment_naive | 2 | **+7.652** | **+7.042** |
| baseline_arm | 2 | **+7.652** | **+7.042** |
| dose_0 | 2 | **+7.652** | **+7.042** |

**实测 B：否定列与正确的 `treated` 并存**，列序 `[severity, <decoy>, treated, age, bmi]`，
decoy = 1−treated：`no_treatment / non_exposed / treatment_naive / baseline_arm / dose_0 /
pre_treatment / post_treatment / arm_b` **全部**绑定 decoy → IPW **+6.435**（正确绑定 −6.435）。
（该 frame 的 PSM 给 nan——decoy 与 treated 完全镜像导致完美分离；用实测 C 的非退化设计复测。）

**实测 C：非退化三臂剂量试验** `[severity, dose_0, dose_low, dose_high, age, bmi]`，
真值 dose_high vs dose_0 = −8 → 绑定 `dose_0`，IPW ate = **+6.351**。

**根因**　`researchforge/profiler/semantics.py:56-60`（无否定/时相前缀处理）+
`_helpers/core.py:137`（同强度并列按文件序取 `named[0]`，无第二信号）。

**建议修法**
1. 否定/时相前缀并入否决规则：
   ```python
   _TREATMENT_NEVER_RE = re.compile(
       r"(?:^|_|\b)(control|ctrl|placebo|comparator|sham|baseline|"
       r"no|non|never|without|naive|pre|post|prior)(?:$|_|\b)"
       r"|(?:^|_|\b)(un|non)(treated|exposed)|(?:^|_|\b)dose_?0(?:$|_|\b)", re.I)
   ```
2. 并列 strength 2 时，命中否定/时相规则的一方让位。

---

### MUST-FIX 3 — 第二个「结局名」二值列会被当成处理变量（outcome 排除只排一列）

**症状**　`resolve_treatment` 的 outcome 排除是 `c != fp.likely_outcome`——**只排一列**。
数据里有两个结局型二值列时，第二个若同时带处理词（`treatment_failure` /
`treatment_response` / `post_treatment_relapse`），就以 strength 2 压过 `treated`。

**复现**
```python
# [severity, relapse, treatment_failure, treated, age]
# 真值：treated -> severity = -8；treatment_failure 是**处理后结局**（对 severity +6）
rng = np.random.default_rng(7); n=600
age = rng.integers(20,70,n); t = rng.integers(0,2,n)
fail = (rng.random(n) < 0.5 - 0.25*t).astype(int)
relapse = (rng.random(n) < 0.3 + 0.3*fail).astype(int)
y = 50 - 8*t + 6*fail + 0.2*age + rng.normal(0,5,n)
```

**实测**　`fp.likely_outcome = relapse`（文件序第一个结局名二值列）→ 绑定
`treatment_failure`；**IPW ate = +6.689，PSM att = +7.022**；
`config={"treatment":"treated"}` 时 ate = **−8.17**。
披露行只说「本方法把 'treatment_failure' 当作处理/暴露变量」，
而且因为 `fp.likely_treatment` 也是它，**连「角色检测另有意见」的 ⚠ 都不出现**。

**根因**　`_helpers/core.py:136` 单列排除。

**建议修法**　把 `roles._BIN_OUTCOME_RE` 命中的**全部**二值列从处理候选剔除（不只
`likely_outcome` 那一个），或让结局词命中的列 strength 归 0。
`prior_*/baseline_*/history_*` 前缀列本来就不是处理候选，不冲突。

---

### MUST-FIX 4 — 绑定词表是纯英文的；中文 / 西语 / 德语 / CDISC 列名一律零信号

**症状**　`semantics.py` 模块 docstring 自称 "ONE **bilingual** role-vocabulary registry"，
`ROLE_HINTS["treatment"]` 里也确有 `处理/品种/剂量/水平/组别`，但
`treatment_name_strength()` **完全不查 ROLE_HINTS**，只用三条英文正则。
于是中文数据集的处理列拿 0 分，引擎退回「文件序第一个二值列」。

**复现**（`[severity, sex, <arm>, age, bmi]`——人口学二值列排在前；真值 arm −8、sex +1）

**实测**

| arm 列名 | strength | conf | 实际绑定 | IPW ate |
|---|---|---|---|---|
| treated / trt / ARM | 2 | high | 正确 | **−7.843** |
| TRT01PN | 0 | low | `sex` | +1.385 |
| TRTA | 0 | low | `sex` | +1.385 |
| armcd | 0 | low | `sex` | +1.385 |
| 处理组 / 干预组 / 治疗组 / 实验组 / 是否用药 | 0 | low | `sex` | +1.385 |
| grupo_tratado / tratamiento / behandelt | 0 | low | `sex` | +1.385 |

真实的 −7.84 被完全错过，报出来的是 sex 的 +1.385（真值 +1.0），且显著。
同一份中文数据上 `has_design_signal(fp)` = **True**——引擎的另一半**已经认识**「处理」，
只是绑定这条路不查。

**注释与行为不符（本身即一条 finding）**　`semantics.py:63-68` 用
「CDISC standardises on TRT01P/TRTA」论证 `tx` 可以只做整名匹配，暗示这两个名字被
`trt` 子串覆盖。实测 `treatment_name_strength` 对 `TRT01P` / `TRTA` / `armcd` **全是 0**
（`(?:$|_|\b)` 后界在 `trt`+数字/字母处不成立）。CDISC 一个都没覆盖。

**建议修法**
- `treatment_name_strength` 在英文正则之外补一条 `role_hint(name, "treatment")` 中文通道
  （给 2，但先过中文否决词 `对照|安慰剂|空白|未处理|未用药|基线`）。
- 补 `trt\d*[a-z]?` / `armcd` / `actarm` 到强词表（锚在词首，`re.I`）。
- 若不打算做多语言，至少**改掉 docstring 的 "bilingual" 断言**，并在
  `likely_treatment_confidence == "low"` 时于运行摘要明说「无名称证据」（见 Q3）。

---

### Q1 · 我验证过**是对的**的部分
- `tx` 只整名匹配、`trt` 做子串：`tx_date/tx_id/tx_amount/tx_online/tx_count/post_tx` 全 0；
  `trt/pre_trt` = 2；`part/sort/trtn` = 0。payments 域实测绑定 `promo_shown`（正确）。
- `age_group/blood_group/risk_group/income_group/group_id/group_size/region_group/
  skin_condition/patient_group` 全 0；`study_group/case_group/experimental_group/
  case_cohort/study_grp` = 1（弱，不压强词）；`intervention_group/exposure_group/
  treatment_group/dose_group/arm_group/treat_cohort` = 2。
- 强度排序确实生效：`[score, group, treated, age]` → 绑 `treated`（−ATT）。
- 二值结局 `treatment_response` 作为 `likely_outcome` 时，确实被 tier 2 的 `c != lo` 拦住。

---

## Q2 · 处理后协变量的排除范围是否正确

结论：**「只排 survival 事件列」这个口径本身站得住**（名称无法一般性地识别处理后变量，
乱删会删掉真混杂因子）。但**三件事**是真问题，其中两件已量到会翻符号 / 83% 衰减。

### MUST-FIX 5 — 中介变量自动进倾向模型，PSM 直接翻符号、AIPW 衰减 83%

**症状**　psm/ipw/aipw 的 AUTO 协变量 = 「除 unit/time/outcome/treatment/survival-event
以外的**全部**数值列」。随机化试验里一个处理后中介（依从性 / 实际剂量 / 随访时长）
落进来，就是教科书式的 over-adjustment。

**复现**
```python
def med_frame(n=800, seed=17, share=0.75):
    """treated -> adherence（随机化后测量的中介） -> severity。真值总 ATT = -8"""
    rng = np.random.default_rng(seed)
    age = rng.integers(20,70,n); bmi = rng.normal(26,4,n).round(2)
    t = rng.integers(0,2,n)                                  # 随机化：本来无需任何调整
    adh = (60 + 30*t + rng.normal(0,10,n)).round(2)          # 处理后
    y = 50 - 8*(1-share)*t - (8*share/30)*adh + 0.2*age + 0.5*bmi + rng.normal(0,5,n)
    return pd.DataFrame({"severity": y.round(2), "treated": t,
                         "adherence": adh, "age": age, "bmi": bmi})
```

**实测**（真值 −8；右列为 `config={"covariates":["age","bmi"]}`）

| 中介占比 | AUTO · IPW | AUTO · PSM | AUTO · AIPW | 正确协变量 · IPW |
|---|---|---|---|---|
| 0.25 | −7.842 | −4.412 | — | −7.770 |
| 0.50 | −6.298 | −2.543 | — | −7.723 |
| 0.75 | −4.754 | **−0.673** | −3.055，CI [−3.79, −2.32] | −7.676 |
| 1.00 | −3.207 | **+1.197（符号翻转）** | **−1.358，CI [−2.04, −0.68]，p=9.3e-05** | −7.628 |

AIPW 最糟：双稳健估计量给出一个 **95% CI 完全不覆盖真值**、p<1e-4 的「显著」结论。
（它同时打印了 `朴素均值差=-7.53`，那才是随机化下的正确答案——没有任何一行提示这个
5 倍的落差值得怀疑。）

**部分安全网（已验证存在）**：IPW/AIPW 的重叠诊断会叫（extreme ps 69%，ESS=54/800）；
PSM 的平衡诊断会报「⚠ 残留不平衡」。但三者的**标题数字照样以「显著」呈现**。

**根因**　`branches/causal/{psm,ipw,aipw}.py:39-41`、`sensitivity.py`（rosenbaum）——
AUTO 集是「全部数值列」，唯一的语义排除是 survival 事件列。

**建议修法**（不需要猜谁是中介，做诚实披露即可）
1. **把 AUTO 协变量名单写进 summary**（现在 IPW/AIPW **一个字都没有**，PSM 只在
   `balance.csv` 里间接可见）。例：
   `⚠ 自动协变量：age、bmi、adherence（按数值列自动选取）——其中任何**在处理之后测量**的变量
   （中介/依从性/实际剂量/随访时长/脱落）都属处理后调整，会使 ATT 衰减甚至翻符号；
   请用 config covariates 显式指定。`
2. 可选的启发式守卫：AUTO 集中若存在与处理相关性极强的连续列（|SMD| 或 |corr| 超阈），
   单独点名提示「疑似处理后变量」。**不要自动删**——删真混杂是另一种偏差。

### SHOULD-FIX 6 — 同族另外三个分支根本没接这条排除，而且还在用「第一个二值列」

**症状**　`g_computation` / `double_ml` / `causal_forest` 各自构建 AUTO 协变量集，
**既没有** survival 事件排除、**也没有**任何披露；而且它们的处理变量仍是
`next(c for c in fp.columns if c.kind == "binary")`——正是 H4d 声称已在全族修掉的缺陷。

**复现 / 实测**（H4d 旗舰 frame `[duration, event, treatment, age, biomarker]`，
真值：treatment 对 duration 的 ATE ≈ **−7.0**；按 `event` 分组的均值差 = **+1.24**）

| 分析 | `RunResult.treatment` | ATE | 摘要原文 |
|---|---|---|---|
| psm | treatment | −3.528 | 排除 event 已披露 |
| ipw | treatment | −3.917 | 同上 |
| aipw | treatment | −4.263 | 同上 |
| **g_computation** | **None** | **+1.240** | — |
| **double_ml** | **None** | **−15.798** | 「处理 **event** → duration」 |
| **causal_forest** | **None** | **+1.341** | 「处理 **event** → duration」 |
| rosenbaum_bounds | treatment | — | 已接 |
| evalue | treatment | — | 已接（结局取 event，对 E-value 合理） |

三个分支在**同一份数据**上把删失指示变量当成了干预，并给出 +1.24 / −15.8 / +1.34。
这同时证明了建造者声称 8（`RunResult.treatment is None` 是审计工具）**是有效的**——
只是这次审计没人跑。

**建议修法**　这三处改用 `resolve_treatment(fp, cfg, fp.binary_columns or bins, df=df)`，
并接上与 psm/ipw/aipw 相同的 `survival_event_column` 排除 + 披露。

### NICE-TO-HAVE 7 — 其余处理后变量实测偏差温和（记录以免后人重复实验）

同样口径下量了两类常被点名的处理后变量，**不构成 must-fix**：

- 处理后二值结局 `relapse` 进协变量：IPW −6.947 / PSM −7.730（正确 −8.616 / −8.760），偏差 13% / 12%。
- 碰撞变量 `dropout`（被处理与结局共同驱动）：IPW −8.581 / PSM −9.333（正确 −7.649），偏差 12% / 22%。

即：**中介**是会翻符号的那一类，二值结局与碰撞变量在本设定下只是中等偏差。

---

## Q3 · `likely_treatment_confidence` 参与绑定了吗？

**答案：没有。它是纯披露，而且披露还没到运行摘要里。**

- 消费点只有三处显示：`cli.py:66-69`、`study_report.py:78-81`、`web/service.py:51`。
- `resolve_treatment` 的 tier 3 是 `elif lt in candidates: chosen = lt`——**不看 confidence**；
  tier 4 更是纯列序。所以 low 置信照绑，且**绑定路径与 confidence 无因果关系**。
- `run.py` 的处理变量披露块（新加的那段）**根本没读** `likely_treatment_confidence`：
  它只说「本方法把 'X' 当作处理/暴露变量——如需改用其他列」。结果 CLI 会说
  「无名称信号，按列序推定」，而真正跑分析时的报告里这句消失了。

**这可接受吗**：作为**守卫**不可接受（它没守任何东西），作为**披露**目前也不完整——
最需要看到「这只是列序猜测」的场合恰恰是运行报告。

### SHOULD-FIX 8 — low 置信绑定在报告里读起来和高置信一模一样

**什么数据形状下会产出「看起来完全正常、实则无意义的 ATT」**（已实测，见 MUST-FIX 4）：
凡是**处理列名不在英文词表内**、而**另一个二值列（人口学/地区/是否城市…）排在它前面**的数据。

**实测**　`[severity, sex, 处理组, age, bmi]`（真值：处理 −8、sex +1）
→ `likely_treatment='sex'`，conf='low' → 绑定 `sex` → **ATE = +1.385，p 极小，重叠「良好」**，
报告全绿、无任何 ⚠。用户看到的是一个方法学上「干净」的因果估计，
内容却是「性别对 severity 的加权均值差」，而真正的处理效应 −7.84 一行都没出现。

**建议修法**　`run.py` 披露块按 confidence 分级：
```python
_conf = getattr(fp, "likely_treatment_confidence", "")
if not _explicit and bound_treatment == getattr(fp, "likely_treatment", None) and _conf == "low":
    _tnote = ("——⚠ **该列没有任何名称证据**，是按列序推定的第一个二值列；"
              "若它不是干预，请用 config treatment 指定（当前估计将毫无因果含义）")
```
（更进一步：low 置信 + 存在 ≥2 个二值候选时，在 summary 里把候选列表都列出来。）

---

## Q4 · `resolve_treatment` 的阶梯有没有顺序错误？

**结论：阶梯顺序本身没错，但 tier 3 基本是死代码，而且 tier 3/4 缺了两个过滤。**

1. **「命名列优先于 `fp.likely_treatment`」是对的**。两者现在用的是同一个
   `treatment_name_strength` 排序：`roles.detect_roles` 在**全部二值列**上排，
   `resolve_treatment` 在**调用方给的候选子集**上排。子集更具体（例如 DiD 只给
   within-unit switcher），所以 tier 2 应当优先。实测在 DiD 族上正是靠这个顺序，
   `policy_on` 才能压过切换率更高的 `post`。
2. **tier 3 几乎永远与 tier 2/4 同解**。有名称信号时 tier 2 已命中；无名称信号时
   tier 3 给「全体二值列中第一个非 outcome」，tier 4 给「候选中第一个非 outcome」——
   仅当候选顺序被重排（DiD 按切换率排序）时两者才可能不同，且没有证据说哪个更可信。
3. **tier 3/4 缺失的两个过滤**（这才是实质问题，已在 Q1 记为 MUST-FIX 1）：
   没有 `_TREATMENT_NEVER_RE` 否决，也没有强度下限。

### SHOULD-FIX 9 — outcome 排除只认 `fp.likely_outcome`，不认 `config["outcome"]` → 处理列 == 结果列 → 裸崩

**症状**　tier 2/4 的排除项是 `lo = fp.likely_outcome`，**从不看 `cfg["outcome"]`**。
用户把 outcome 指到一个二值列时，同一列可能被选成处理变量。

**复现**
```python
# [died, relapse, age, bmi]；无任何处理词
rng = np.random.default_rng(3); n=500
relapse = (rng.random(n) < 0.35).astype(int)
died = (rng.random(n) < 0.2+0.3*relapse).astype(int)
df = pd.DataFrame({"died": died, "relapse": relapse,
                   "age": rng.integers(20,70,n), "bmi": rng.normal(26,4,n).round(2)})
run_analysis(fp, CAT.by_id("psm"), output_root=..., config={"outcome": "relapse"})
```

**实测**　`fp.likely_outcome='died'` 被排除 → tier 4 取 `relapse` = 用户的结果列。
psm / ipw / aipw 三者全部：
```
⚠ psm 执行失败：TypeError: '<' not supported between instances of 'str' and 'NoneType'
```
不是错数字，是**不诚实的降级**：裸 TypeError，没有任何「处理列与结果列相同」的提示。

**根因**　`_helpers/core.py:126` `lo = getattr(fp, "likely_outcome", None)`；
psm/ipw/aipw 里 treatment 在 outcome **之前**解析，所以分支自己也来不及去重。

**建议修法**　`resolve_treatment` 取
`lo = (cfg.get("outcome") if cfg.get("outcome") in cols else None) or fp.likely_outcome`；
并在 psm/ipw/aipw 解析完 outcome 后加一条
`if treatment == outcome: summary.append("…失败：处理列与结果列相同，请用 config 指定"); return`。

---

## Q5 · `_pick_did_treatment` 返回全部 switcher —— 其余调用点对不对？

**6 个真实调用点全部正确**（都是把列表整个喂给 `resolve_treatment` 当候选池）：
`branches/causal/event_study.py:30`、`gsynth.py:27`、`staggered_did.py:33`、
`synthetic_control.py:28`、`branches/causal_did.py:164`、`branches/did_advanced.py:77`。
且都有 `unit and time` 前置守卫。实测在两个 switcher 的面板上
（`post` 切换率 1.0 vs `policy_on` 0.5）三者都绑定 `policy_on`，ATT 落在 2.80–2.95（真值 +3.0）。

**唯一的 `[:1]` 消费点是错的**：

### MUST-FIX 10 — `did` 分支取 `_pick_did_treatment(...)[:1]`，等于重新引入 argmax bug

**症状**　`_helpers/core.py:258`
```python
rhs_vars = _pick_did_treatment(df, fp)[:1] or fp.binary_columns[:1]
```
`[:1]` 取的是**切换率最高**的那一列，**完全绕过名称阶梯**——正是 delta-review D1
所修的那个错误，只是搬进了 `_regression`。而且它不经过 `resolve_treatment`，
所以既**不记录**（`RunResult.treatment` 永远 None）、也**不读 `config["treatment"]`**。

**复现（会出错数字的那一支）**
```python
# 面板，真值 policy_on 的 ATT = +3.0；hospitalized 是被政策驱动的时变二值**结局**
for u in range(30):
    g = 6 if u%3==0 else (10 if u%3==1 else 10**9)
    for t in range(14):
        d = 1 if t>=g else 0
        rows.append({"unit": f"u{u:02d}", "year": 2005+t,
                     "y": 10+fe+3.0*d+0.1*t+noise, "policy_on": d,
                     "hospitalized": int(rng.random() < 0.3+0.2*d)})
```

**实测**
```
_pick_did_treatment -> ['hospitalized', 'policy_on']         # 按切换率
resolve_treatment(同一列表) -> 'policy_on'                    # 名称阶梯是对的
did         : RunResult.treatment=None, estimates={'hospitalized': 0.2744}
              摘要「因变量 y，关键系数 hospitalized = 0.2744 (p=0.00575)」
staggered_did: treatment='policy_on', att_overall = 2.798     # 真值 +3.0
```
即 `did` 报了一个**显著的 0.2744**，真值是 **+3.0**，而正确的列就在同一张表里。

**复现（会降级的那一支）**　把 `hospitalized` 换成日历 dummy `post`：
`[:1]` 取 `post`，它与时间固定效应共线 → 分支以秩亏消息诚实失败。
诚实，但**本不该失败**——`policy_on` 就在候选里。

**config 被忽略（实测）**
```
did config={}                        -> {'hospitalized': 0.2744}
did config={'treatment':'policy_on'} -> {'hospitalized': 0.2744}   # 完全没反应
```

**根因**　`_helpers/core.py:255-258`：did 路径自己挑列，没走共享解析器。

**建议修法**
```python
if entry.id == "did" and fp.binary_columns:
    _sw = _pick_did_treatment(df, fp, unit=fp.unit_col, time=fp.time_col)
    _t = resolve_treatment(fp, cfg, _sw or fp.binary_columns, df=df)   # 名称阶梯 + config + 记录
    rhs_vars = [_t] if _t else fp.binary_columns[:1]
```
这一行同时修好三件事：错数字、config 失效、`RunResult.treatment` 空洞。

### SHOULD-FIX 11 — `_pick_did_treatment` 的无面板回退丢掉了 cfg

`_helpers/core.py:168`：`t = resolve_treatment(fp, None, fp.binary_columns)` —— 传的是
`None` 而不是调用方的 cfg。所有 DiD 族调用点都有 unit/time 守卫，所以目前**只有
`_regression` 的 did 路径能走到这里**；一旦修了 MUST-FIX 10 这条就不再可达，
但留着就是下一个 `config` 静默失效的陷阱。建议改签名收 `cfg` 并透传。

---

## Q6 · `RunResult.treatment is None` 这个审计不变式有没有假阴性？

**不变式本身成立**（「is None ⇒ 该分支没走共享解析器」），而且**这次审计真的抓到东西**：
g_computation / double_ml / causal_forest 三个分支 `treatment=None`，实测在旗舰 frame 上
把 `event` 当成了干预（见 SHOULD-FIX 6 的表）。这条不变式值得保留并接进门禁。

已核对的边界情形：

| 情形 | 结论 |
|---|---|
| `did`（`_regression`）选了处理列却不记录 | **不是假阴性**——它确实没用共享解析器，None 是诚实的。但用户因此**看不到任何处理变量披露**（见 MUST-FIX 10） |
| `_pick_did_treatment` 无面板回退里记录了一次 | **潜在假阳性**：记录的是 `cfg=None` 下的结果，与分支随后真正使用的列可能不同（first-wins）。当前只有 `did` 非面板路径可达，实测记录值恰好与所用列一致，**但这是巧合** |
| `gsynth`：`_cands` 为空时走 `cfg.get("treatment")` | **假阴性**：用了配置列却不记录（`gsynth.py:29` 的 else 分支没有 `_record_bound_treatment`） |
| psm/ipw/aipw 记录后又因前置条件失败 return | 记录仍在 → `treatment` 非 None 而 `estimates` 空。测试用 `_ran()` 过滤，可接受；但审计脚本要同时看 estimates |
| `evalue` 显式 exposure | 已在 `sensitivity.py:278` 补记录，实测 `config={"exposure":"treatment"}` → `RunResult.treatment='treatment'` ✓ |

**建议修法**　(1) `gsynth.py:29` 的 else 分支补 `_record_bound_treatment`；
(2) 把「因果族分析跑成功但 `RunResult.treatment is None`」做成门禁断言，
    这样 g_computation/double_ml/causal_forest 这类漂移下次会自己报出来。

---

## 跨问题 · 本波的旗舰测试**当前全部处于休眠**（自己的披露文案关掉了自己的测试）

### MUST-FIX 12 — `_ran()` 把新披露里的「失败」二字当成了降级信号，10 个测试静默 skip

**症状**　两个新测试文件都用
```python
_DEGRADED = ("失败", "跳过", "暂未接入", "未检测到")
def _ran(res): return bool(res.estimates) and not any(k in res.summary for k in _DEGRADED)
```
而**同一个 commit 新增的**处理后协变量披露文案里有：
「…实测会把 ATT 衰减 33-73%，事件率饱和时甚至直接**失败**）。」
于是在旗舰 survival frame 上，`_ran()` 对 psm/ipw/aipw/rosenbaum **恒为 False**，
所有断言在 `pytest.skip` 处被跳过——**分析其实跑成功了**（`estimates` 非空）。

**复现**
```python
_DEGRADED = ("失败","跳过","暂未接入","未检测到")
for cid in ("psm","ipw","aipw"):
    r = run_analysis(fp_surv, CAT.by_id(cid), output_root=...)
    print(cid, bool(r.estimates), [k for k in _DEGRADED if k in r.summary])
# psm True ['失败']   ipw True ['失败']   aipw True ['失败']
#   命中来源：「…事件率饱和时甚至直接失败）。若确需纳入…」
```

**实测：当前 main 上休眠的测试**（`pytest -rs` 输出）
```
test_treatment_binding.py::test_causal_family_binds_the_named_treatment[psm|ipw|aipw|rosenbaum_bounds]
test_treatment_binding.py::test_psm_recovers_the_real_effect_not_the_event_contrast   ← 本波的旗舰回归测试
test_treatment_binding.py::test_config_treatment_still_wins_end_to_end
test_treatment_binding.py::test_did_formula_takes_only_one_treatment_term
test_treatment_binding.py::test_did_uses_exactly_one_of_several_switchers
test_treatment_covariates_and_hints.py::test_the_event_indicator_stays_out_of_the_auto_covariates[psm|ipw|aipw]  ← 特性①的全部测试
test_treatment_covariates_and_hints.py::test_excluding_the_event_moves_the_estimate_materially
```
连 ratchet `test_wired_branches_never_run_without_reporting_their_treatment` 也因
`if _ran(res) and res.treatment is None` 短路而对这 4 个分支不生效。
原始输出：`82 passed, 12 skipped`。

**把判据收紧后复跑**（把 `"失败"`/`"跳过"` 换成 `"失败："`/`"跳过："`，测试文件复制到
scratchpad 运行，仓库未改动）：**92 passed, 2 skipped**。
好消息：这 10 条断言**本身是对的**，一旦跑就通过；坏消息：这是个**已经脱扣的棘轮**。
剩下的 2 个 skip 正是 `did` 的那两条——也就是 MUST-FIX 10 能一直没人发现的原因。

**根因**　用中文摘要做子串匹配来判断「跑没跑成」，而摘要里本来就会出现这些词。

**建议修法**
- 判据改成结构化的：所有分支的失败/跳过消息形如 `"<方法>失败：" / "跳过："`（带冒号），
  用 `"失败："`/`"跳过："` 即可区分；更稳的是让 `RunResult` 带一个 `degraded: bool`
  字段由 teardown 统一置位，测试改判这个字段。
- 加一条**元测试**：对旗舰 frame 断言 `_ran(res) is True`（否则本波测试再次整体休眠）。

---

## 我验证过**是对的**的机制（主脑不必重复劳动）

1. **DiD 族的 switcher→名称阶梯两级选择**：6 个调用点全部正确传列表；
   `[post(1.0), policy_on(0.5)]` → 绑 `policy_on`；staggered_did att 2.95 / event_study 2.91
   / Sun-Abraham 2.80（真值 +3.0）。时不变强命名诱饵 `treatment_arm` 也压不过 switcher。
2. **`treatment_name_strength` 的强/弱分离**：`age_group`/`group_id`/`skin_condition`/
   `tx_*` 全部 0；`group`/`tx`/`study_group` = 1 且从不压过强词；payments frame 绑
   `promo_shown`（−ATT）而非 `tx_online`。
3. **`fp.binary_columns` 改名**：`treatment_candidates` 属性已彻底移除，语义正确（形状事实）。
4. **survival 事件列排除 + 常数协变量守卫**：在 psm/ipw/aipw/rosenbaum 上确实生效并有披露；
   事件率饱和（event 恒为 1、profiler 判成 count）时确实被常数守卫接住，不再裸 `Singular matrix`。
5. **explicit config 优先 + `df=` 加宽**：`config={"treatment":"bmi"}`（连续列）能绑上并被记录；
   不传 `df=` 时不加宽。`evalue` 的显式 exposure 也被记录。
6. **`RunResult.treatment` 作为审计工具**：不变式成立，并且真的抓到了 3 个漂移分支（见 SHOULD-FIX 6）。
7. **importance_analysis 排除面板时间列**（claim 9）：我验证的是**载荷部分**——
   `resolve_predictors` 在 `[firm, year, sales, x1, x2]` 面板上返回 `['x1','x2']`，
   `config={"predictors":[...,"year"]}` 仍能显式包含 `year`。
   端到端的 dominance 运行**未能验证**：当前工作树里主脑在建的未跟踪文件
   `researchforge/catalog/entries/multilabel.yaml` 是坏 YAML
   （`line 16, column 372: mapping values are not allowed here`），
   `Catalog.load()` 整体抛异常——**这是主脑的在飞工作，不属于本次审查范围，但树目前是红的**。

---

## 建造者声称的 9 条 · 逐条判定

| # | 声称 | 判定 |
|---|---|---|
| 1 | 两套词表、错误剖面相反 | **成立**（弱表只用于 skip、强表用于 bind），但强表本身有否定盲区（MUST-FIX 2） |
| 2 | NEVER 规则「覆盖一切」 | **不成立**：只覆盖 tier 2；唯一候选时照绑，+7.652 vs −8（MUST-FIX 1） |
| 3 | 强度阶梯 config > 命名 > likely_treatment > 非 outcome 首列 > 首列 | **成立**（顺序对），但 tier 3/4 缺否决与强度下限（Q4） |
| 4 | 处理后协变量只排 survival 事件列、如实披露 | **口径成立**，但同族三个分支没接、且 AUTO 协变量名单从不披露（MUST-FIX 5 / SHOULD-FIX 6） |
| 5 | 常数协变量守卫 | **成立**（实测接住饱和事件率） |
| 6 | `likely_treatment_confidence` 既显示又参与绑定 | **半不成立**：它**不参与**绑定（绑定不看 confidence），而且运行摘要里根本没显示（SHOULD-FIX 8） |
| 7 | `_pick_did_treatment` 返回全部 switcher、did 取 `[:1]` | **前半成立、后半是 bug**：`[:1]` = argmax，绕过名称阶梯，报 0.2744 vs 真值 3.0（MUST-FIX 10） |
| 8 | `RunResult.outcome/treatment` 是审计工具 | **成立且有效**（据此抓到 3 个分支） |
| 9 | importance_analysis 排除面板时间列 | **载荷部分已验证**；端到端因树内坏 YAML 未跑（见上） |
