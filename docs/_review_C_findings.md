# 冷审 C — `repeated_measures_anova` 长表自动检测（Q9–Q13）

审查者：Opus 冷审（reviewer ≠ builder）。只读审查，未改任何代码。
日期：2026-09-18。代码位置：`researchforge/executor/branches/experimental_stats.py:436-724`；
测试 `tests/test_repeated_measures_long_format.py`。
对照工具：R 4.6.0 + `car`（`Rscript` = `C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe`）；
本机**无** `ez`、**无** `pingouin`（已实测确认）。

---

## Q9 — 自动检测的前提（平衡完整设计）是否被验证

### 基线复现（建造者声称的数字，已核实为真）
60 受试者 × 5 周长表 `[subject, week, arm, pain_score]`：

| | 引擎 | R `car::Anova(idata,idesign=~week)` |
|---|---|---|
| F | **139.386** | **139.39** |
| num/den df | 4 / 236 | 4 / 236 |
| p | 6e-61 | < 2.2e-16 |
| 偏 η² | 0.703 | —（R 不直接给） |
| Mauchly W | 0.403218 | 0.40322 |
| GG ε | **0.6323637** | **0.6323637** |
| GG 校正 p | 1.592761e-39 | 1.592761e-39 |

→ **建造者宣称的 F(4,236)=139.39 / 偏 η²=0.703 / Mauchly 拒绝 → GG，全部属实。**

### 四种病态形状的实测行为

| 形状 | 结果 | 自由度 | 判定 |
|---|---|---|---|
| 随机删 5% 行（285 行） | 完成：48 受试者，**去掉 12 名**（披露了） | F(4,188)，(n-1)(k-1)=47*4=188 ✅ | 诚实降级 |
| 随机删 20% 行（240 行） | 完成：22 受试者，**去掉 38 名**（63% 样本） | F(4,84)=59.065，21*4=84 ✅ | 数字对，但**样本损失量级未预警**（见 S1） |
| 结构性缺格（30 人缺 week=4） | 完成：30 受试者，去掉 30 名 | F(4,116)，29*4=116 ✅ | 数字对，**缺失机制未披露**（见 S1） |
| 同一格 2 个观测（10 人 week=2 重复） | **失败**：`需要 长表… 受试者=None` | — | 诚实失败但**理由说错了**（见 S2） |
| 只有 1 名受试者完整 | **失败**：「只剩 1 名完整受试者（需 ≥2）」 | — | ✅ 诚实降级 |
| 只有 3 名受试者完整 | **完成**，且 **Mauchly 谎报「球形度未拒绝」** | F(4,8) 对 | ❌ **MUST-FIX M1** |

结论：**自由度本身没算错**（AnovaRM 的 (n-1)(k-1) 在每种形状下都与手算/R 一致），
complete-case 过滤（`experimental_stats.py:534-538`）确实挡住了不平衡与重复格。
但**秩亏时球形度诊断静默给出反向结论**——见 M1。

---

## MUST-FIX

### M1 — 受试者数 < 条件数时，Mauchly 静默谎报「球形度良好」，并把用户推向错误的 p
**位置**：`researchforge/executor/branches/experimental_stats.py:679-707`（`_sphericity`），
消费处 `:578, :633-639`。

**症状**：当完整受试者数 `n` 满足 `n-1 < k-1`（即 n < k），条件协方差阵投影 `S*` 秩亏，
真 Mauchly `W = 0`（不可估）。代码第 679 行 `eig = eig[eig > 1e-12]` **把零特征值丢掉**，
随后仍用 `p = k-1` 归一化（第 696-699 行），于是 W 被凭空抬成正数，p 接近 1。

**最小复现**（60 人数据里只留 3 人完整）：
```
q9_only3complete: 3 名受试者 × 5 个条件
  引擎：Mauchly p = 0.997246 →「Mauchly 球形度检验未拒绝(p≈0.997)，未校正 p 可用」
        未校正 p = 0.011966（显著）；GG p = 0.088564（不显著）
```
**实测内部数字**（直接调 `_sphericity`）：
```
S* 特征值 = [-0., 0., 6.13178847, 29.72696487]   (kept 2/4)
W(含零特征值, 真值) = -6.54e-32   W(过滤后, 引擎实际用的) = 0.0282226
```
**R 对照**（同一份 3×5 矩阵）：
- `car::Anova(...)` → `sphericity.tests` **空表**，`GG eps = NA`，并显式告警
  `one or more error SSP matrix: corresponding non-sphericity tests and corrections not available`
- `stats::mauchly.test` → `W = 1.1974e-30, p = 0.07095`（W 本身已经是 0，p 无意义）

即：R 的两条路都**拒绝**给出球形度结论，只有本引擎给出「p=0.997，球形度良好」。

**后果**：这不是小数点问题——它**翻转了给用户的方法学指令**。引擎说「未校正 p 可用」，
于是用户报 p=0.012 显著；GG 校正 p=0.0886 并不显著。且三名受试者的 RM-ANOVA 本就不该做。

**同一 bug 的更极端形态**（蒙特卡洛，n=5/k=6，3296 次抽样中最坏几例）：
```
n=5 k=6  engine Mauchly p = 1.0000   真值(R 二阶公式, 基于真 W) = 4.25e-09
n=5 k=6  engine Mauchly p = 0.9999   真值 = 2.22e-09
```
**引擎报「完美球形」，真值是极端违反球形**。全样本里 2.5% 的抽样发生 0.05 判定翻转，
且全部集中在 `n <= k` 的秩亏区。

**根因**：`eig > 1e-12` 的过滤对 GG ε 无害（零特征值对 `(Σλ)²/((k-1)Σλ²)` 的分子分母都
不贡献），但对 **W = Πλ / (Σλ/p)^p 是致命的**——乘积里丢掉零因子等于把 det 从 0 改成正数。
作者把同一个过滤后的 `eig` 同时喂给了两个公式。

**建议修法**（`_sphericity` 内）：
```python
eig_all = np.linalg.eigvalsh(Sstar)
rank_ok = (ns - 1) >= (k - 1) and int((eig_all > 1e-9 * max(eig_all.max(), 1.0)).sum()) == k - 1
eig = eig_all[eig_all > 1e-12]
...
# GG/HF 照旧（零特征值无害），但：
if not rank_ok:
    return float("nan"), float("nan"), float("nan")   # 与 car 一致：不可估就不报
```
并在分支里（`:633-639`）为 `mauchly_p` 为 nan 的情况已有分支，但**必须同时把 ε 与 GG p
也标成不可估**，且 summary 明说「完整受试者数(n) < 条件数(k)，球形度诊断与 GG 校正不可估
（R `car` 在同一数据上亦拒绝给出）——结果仅供描述，请改用 mixed_effects」。
另建议在 `n < k` 时干脆**拒绝出 F**（或至少强制警告）：3 名受试者 × 5 条件的 RM-ANOVA
不是一个可报告的推断。

---

### M2 — 组间因子被静默丢弃：在建造者自己的 RCT 测试数据上，引擎答错了问题、并**凭空造出一个球形度违反**
**位置**：`experimental_stats.py:482-484`（`long_df = df[[subject, within, outcome]]` —— 只取三列，
其余全丢）、`:640-646`（summary 从不提被丢的列）。

**症状**：本分支只做**单因子组内** RM-ANOVA。任何组间因子（RCT 的 `arm`）被静默从模型里剔除，
且 summary 一个字都不提。对 RCT 而言，真正的处理效应是 **arm×week 交互**，而不是 week 主效应。

**最小复现**：就用 `tests/test_repeated_measures_long_format.py::_long_rm()` 的那份数据
（60 人 × 5 周 × arm，真值：drug 组每周额外 -2.5）。

**实测对照**（R `car::Anova`，同一份 60×5 矩阵；脚本 `check2.R`）：

| | 引擎（组内单因子） | R 正确的裂区(split-plot)模型 `lm(Y ~ arm)` + `idesign=~week` |
|---|---|---|
| week F | **F(4,236)=139.39** | **F(4,232)=260.30** |
| week 误差 SS | 1850.5 | **1007.1** |
| arm | 未估 | F(1,58)=30.20, p=9.1e-07 |
| **arm:week（RCT 的处理效应）** | **未估** | **F(4,232)=48.57, p<2.2e-16** |
| GG ε | **0.6324** | **0.9531** |
| HF ε | 0.6630（算了但没报） | 1.0285（>1 截为 1） |
| Mauchly | **p=4.23e-08 → 「⚠ 球形假定违反，采用 GG 校正」** | 正确模型下 ε≈0.95，**球形度基本没问题** |

三重后果，全部实测：
1. **丢掉估计量**：RCT 的处理效应 `arm:week` (F=48.57) 根本没被估过，用户拿不到。
2. **主效应被稀释**：被省略的交互项被倒进误差项（SS_error 1007 → 1850），week 的 F 从 260.3 掉到 139.4。
3. **伪造了一个球形度违反**：两组随时间发散被误读成条件协方差异质，ε 从 0.953 掉到 0.632，
   于是引擎**大声警告「⚠ Mauchly 球形度检验被拒绝」并让用户改用 GG 校正 p**——这个警告
   在正确设定的模型下**根本不存在**。这不是「少报一个交互项」，是**模型误设导致的假诊断**。

**根因**：`:482-484` 硬取三列；分支从未检查「有没有在受试者内恒定、但在受试者间变化」的列。
讽刺的是**这个判据分支自己已经算过**（`:474` 的 `groupby(subject)[c].nunique() > 1`），
只是永远不报告否定的那一侧。

**建议修法**（最小代价，不改估计器）：在 `:528` 之后加
```python
_between = [c for c in df.columns
            if c not in {subject, within, outcome}
            and df.groupby(subject, observed=True)[c].nunique(dropna=True).max() == 1
            and df[c].nunique(dropna=True) > 1]
if _between:
    summary.append(
        f"⚠ 检测到组间因子 {_between} 未纳入模型——本分支只做单因子组内 RM-ANOVA。"
        f"若这是随机对照试验，真正的处理效应是 {_between[0]}×{within} 交互（裂区/混合 ANOVA），"
        "本结果无法估计它；并且省略它会把交互项并进误差项，"
        "使组内 F 偏小、并可能把组间发散误判成球形度违反。请改用 mixed_effects。")
```
（更彻底的做法是支持 `config["between"]` 做裂区 ANOVA，但那是新功能；上面这条披露是止血线。）

---

## Q10 — `fp.time_col` 当 within 因子是否总是对的

### 实测四种形状

| 形状 | `fp.unit_col` / `time_col` | 引擎行为 | 判定 |
|---|---|---|---|
| 连续随访天（每人 12.3/14.7/…，300 个不同值） | `subject` / `day` | **诚实失败**：「within 因子=None」 | ✅ 基数守卫 `_nu > len(df)//2`（`:471`）挡住了 |
| 日历日期不对齐（每人自己的起始日） | `subject` / `visit_date` | **诚实失败**：「只剩 0 名完整受试者」 | ✅ 不产数；但理由指错（见 S3） |
| **真 within 因子不是时间**（45 人 × 3 种刺激 `[participant, stimulus, rt_ms]`） | **None / None** | **失败**：「受试者=None、within 因子=None」 | ❌ **MUST-FIX M3** |
| 同上 + 第二个组内因子 `session` | None / None | 同上失败 | ❌ 同 M3 |

### M3 — 最经典的组内设计（非时间的 within 因子）根本触发不了自动检测，且报错指错方向
**位置**：`experimental_stats.py:462-463`（`subject` 只从 `fp.unit_col` 取）×
`researchforge/profiler/profile.py:87-90`。

**症状**：`profile.py:89-90` 是 `if time_col is None: return` —— **没有时间列就绝不设 `unit_col`**。
而分支的 within 检测循环整体被 `if within is None and subject is not None`（`:464`）门住。
于是**任何没有时间列的组内设计（心理学/感官/神经科学的标准长表）自动检测 100% 失效**。

**最小复现**（45 名被试 × 3 种刺激，教科书 within-subjects 设计）：
```python
df = pd.DataFrame({"participant": [...], "stimulus": ["neutral","happy","fearful",...], "rt_ms": [...]})
# 实测输出：
重复测量方差分析失败：需要 长表(...) 或 宽表(...)。（已自动查找：受试者=None、
  within 因子=None、结果='rt_ms'；within 因子须在同一受试者内取到 ≥2 个值——组间因子如试验臂不算）
```
`participant` 每人恰好 3 行、`stimulus` 每人恰好 3 个水平，是一份**完美平衡**的 RM 数据，
引擎却说「没找到受试者」。

**建造者断言的证伪**：测试文件 docstring 与 commit 叙事都说「修后长表不再需要 config」。
**实测边界是：只有当数据被 profiler 判成「面板」（有时间列 + 唯一 (unit,time) 对）时才不需要 config。**
纯组内实验设计仍然 100% 需要 config。这条断言被**过度推广**了 —— 本身就是一个 finding
（注释/测试叙事与实际行为不符）。

**建议修法**：`subject` 的兜底不要只认 `fp.unit_col`。加一条纯结构判据：
```python
if subject is None:
    # 组内设计不一定有时间列 -> profiler 不会给 unit_col。用结构直接找：
    # 一个候选 subject 列要重复出现、且组数远小于行数
    for c in (x.name for x in fp.columns if x.kind in {"categorical","id","count"}):
        nu = int(df[c].nunique(dropna=True))
        if 2 <= nu <= len(df)//2 and (len(df)/nu) >= 2:
            subject = c; break
```
并在 within 循环里对「每个受试者内取到的水平数」要求**中位数 ≥2**（而非仅 `.any()`，见 S4）。
注意：加了这条之后，M2 的组间因子披露和 S4 的守卫收紧**必须同时上线**，否则 M3 的修复会
把更多 `q12b` 那类误检数据放进来。

---

## Q11 — Mauchly / GG / HF 数值核对（对照 R 4.6.0 `car::Anova` + `stats::mauchly.test`）

逐位对照（`sph.py` 直接调 `_sphericity`，`check.R` 跑 R）：

| 量 | N=60 引擎 | N=60 R | N=8 引擎 | N=8 R |
|---|---|---|---|---|
| Mauchly **W** | 0.403218 | 0.40322 | 0.081282 | 0.081282 |
| Mauchly **p** | **4.23403e-08** | **4.30324e-08** | **0.137482** | **0.154307** |
| **GG ε** | 0.6323637 | 0.6323637 ✅ | 0.4706406 | 0.4706406 ✅ |
| **HF ε** | 0.6630327 | 0.6630327 ✅ | 0.6380391 | 0.6380391 ✅ |
| GG 校正 p | 1.5927610589927074e-39 | 1.592761e-39 ✅ | — | 0.0007763511 |
| F / df | 139.38617904 / (4,236) | 139.39 / (4,236) ✅ | 13.346 / (4,28) | 同 ✅ |
| 偏 η² | 0.7026002502580152 | =4372/(4372+1850.5) ✅ | — | — |

→ **W、GG ε、HF ε、GG 校正 p、F、df、偏 η² 全部逐位正确。** `_orthonormal_contrasts` 的
QR 构造（`:712-724`）也因此被间接证实（W 与 R 完全一致）。

### S0（SHOULD-FIX）— Mauchly p 只用了一阶卡方近似，小样本系统性偏小
`_sphericity:700-706` 用 `p = chi2.sf(-(n-1)·d·lnW, f)`。R 的 `stats:::mauchly.test.SSD`
（以及 SPSS）用**二阶修正**：
```
w2 = (p+2)(p-1)(p-2)(2p^3+6p^2+3p+2) / (288 (n·p·rho)^2)
pval = Pr(chi2_f > z) + w2 * (Pr(chi2_{f+4} > z) - Pr(chi2_f > z))
```
**实测偏差**：N=60 差 1.6%（4.234e-08 vs 4.303e-08，无所谓）；**N=8 差 11%**（0.1375 vs 0.1543）。
蒙特卡洛（3000 次抽样，已**排除**秩亏区 n≤k）：**0.8% 的抽样在 0.05 上判定翻转**，且
**方向一律是引擎 p 偏小 → 过度拒绝球形度**：
```
n=6 k=5  engine 0.04611  R 0.06929   (引擎判「违反」，R 判「不违反」)
n=8 k=6  engine 0.04963  R 0.07115
n=8 k=6  engine 0.04124  R 0.06028
```
后果是**保守**（多用一次 GG 校正 → 损失功效，不制造假阳），所以不是 MUST-FIX；
但 summary 里那句「未拒绝，未校正 p 可用 / 被拒绝 → 用 GG」的**指令会翻**。
修法就是照抄上面两行 `w2` 公式（本 finding 已给出可直接粘贴的实现）。

### S5（SHOULD-FIX）— HF ε 算了、算对了，然后被扔掉
`:578` `gg_eps, hf_eps, mauchly_p = _sphericity(M)` —— `hf_eps` 此后**再未被使用**：
不进 `estimates`（`:587-593`）、不进 summary、不进 CSV。而它的值与 R 逐位一致（0.6630327）。
标准做法（Girden）是 **ε̂ < 0.75 用 GG，≥0.75 用 HF**，因为 GG 在 ε 大时过度保守；
引擎无条件用 GG。建议：把 `hf_epsilon` 与 `hf_corrected_p` 一并写进 estimates 并在 summary
按 0.75 规则给建议。（注：元组解包不触发 ruff F841，所以门禁抓不到这个死变量。）

---

## Q12 — within 守卫（「须在同一受试者内取到 ≥2 个值」）的漏杀与误杀

### M4（MUST-FIX）— 漏杀：平衡的**无关**组内变量会被当成设计因子，静默出一份完整 ANOVA
**位置**：`experimental_stats.py:464-476`。守卫只要求 `.any()`（**至少一名**受试者内有 ≥2 个水平）
+ 基数 `2 ≤ nu ≤ len(df)//2`。它**不检查该列是不是一个设计因子**。

**最小复现**（60 人 × 5 次随访；随访时间是连续天数所以被基数守卫排除；
`operator_id` = 每次测量的技师，按拉丁方轮换，因此**完美平衡**；真值是**时间**效应 -1.5/次）：
```python
rows.append({"subject": f"p{s:03d}", "day": 14*t + noise,          # continuous -> rejected
             "operator_id": int(perm_of_1_to_5[t]),                 # nuisance, perfectly balanced
             "pain": 50 + re_s - 1.5*t + noise})
```
**实测输出**（`q12b_balanced_nuisance`，无任何警告）：
```
完成（长表）：60 名受试者 × 5 个条件（operator_id，结果 pain）；
组内主效应 F(4,236)=1.787, p=0.132（不显著），偏 η²=0.029。
Mauchly 球形度检验未拒绝(p≈0.991)，未校正 p 可用；GG 校正 p=0.133 备查。
```
用户拿到一份干净、显著性明确的「无效应」结论——**回答的是他从没问过的问题**（技师之间有没有
差别），而**真正的时间效应从未被检验**。加重情节：`outcome` 的自动选取会发一条
💡「已自动选取 'pain' 为结果变量…如需改用其他列，用 config outcome 指定」，
但 **`subject` / `within` 的自动选取一个提示都没有**——同一分支里两套披露标准。

**建议修法**（两条一起）：
1. 守卫从 `.any()` 收紧为「**多数受试者**都取到 ≥2 个水平」，并要求「每个受试者内的水平数
   与总水平数相符」：
```python
_per = df.groupby(subject, observed=True)[_c].nunique(dropna=True)
if float((_per >= 2).mean()) < 0.5 or float(_per.median()) < min(2, _nu):
    continue
```
2. **无条件补一条自动选取披露**，与 outcome 的 💡 提示对齐：
```python
if cfg.get("within") is None:
    summary.append(f"💡 within 因子由引擎自动选取为 {within!r}（依据：它在受试者内变化且水平数={n_cond}）；"
                   f"若真正的组内因子是别的列，用 config={{\"within\":..}} 指定。")
```

### 误杀：**没有**发生（已实测）
- 一名受试者在 week 0 后脱落 → 检测照常命中 `week`，该受试者被平衡过滤剔除，
  **59 名受试者 × 5 条件，F(4,232)=135.068**，并披露「去除 1 名」。✅ 不误杀。
  （检测循环用 `.any()`，一名完整受试者就够，所以脱落者挡不住检测。）
- 60 名受试者**各缺一个不同的 week**（240 行）→ complete-case 全军覆没 →
  **诚实失败**「只剩 0 名完整受试者」。✅ 不出错数，但见 S1。

---

## Q13 — 它作为「推荐出口」称职吗？

### 建造者的 B1 修复：**已验证为真**
`regression.py:96-128`：`_keep = [v for v in rhs_vars if v not in _absorbed]`；
若 `_keep` 非空 → 走 `:97-118` 的剔除后重估路径并清掉 `_degenerate`；
「改用 repeated_measures_anova / mixed_effects」这句被门在 `_absorbed and not _keep`（`:127`）。
**实测确认**：firm×year 面板 + 唯一预测变量 `treated` 时不变 → 无幸存项 → 建议出现。
建议只在关键项真不可估时才给 —— **驳回不了，建造者这条改对了。**

### S6（SHOULD-FIX）— 但顺着这条建议走过去，拿到的是**对另一个问题的显著答案**
**最小复现**（40 家公司 × 6 年，`treated` 时不变，真值 +1.2）：
```
step1 panel_fixed_effects → 失败：设计矩阵秩亏（秩 45 < 列数 46）——被完全解释的是预测变量 treated…
        「…若为重复测量设计，请改用 repeated_measures_anova / mixed_effects；」
step2 repeated_measures_anova（照建议做，同一份数据）→
        完成（长表）：40 名受试者 × 6 个条件（year，结果 roa）；
        组内主效应 F(5,195)=14.200, p=7.56e-12（显著），偏 η²=0.267。
step3 mixed_effects（同一份数据）→ 完成：估出 treated = 2.17
```
用户问的是「`treated` 有没有效应」。RM-ANOVA **把 `treated` 整列丢掉**（M2），自动挑了 `year`
当 within 因子，然后给出 **p=7.56e-12 的强显著结果**——那是**年份主效应**，不是处理效应。
既没有一句「你的处理变量没进模型」，也没有一句「本方法估不了被单位固定效应吸收的时不变处理」。
而 `mixed_effects` 在同一份数据上**直接估出了 treated**。

结构上这是必然的：被单位固定效应吸收 ⇔ 该变量在受试者内不变 ⇔ 它是**组间**因子
⇔ **单因子组内 RM-ANOVA 在定义上就估不了它**。所以把 `repeated_measures_anova` 和
`mixed_effects` 并列推荐是把用户往沟里带。

**建议修法**（`regression.py:124-127`）：
```python
+ ("关键预测变量本身不可估（最常见：处理变量在每个单位内不随时间变化，被单位固定效应吸收）——"
   "这类时不变处理属于**组间**因子，单因子组内 repeated_measures_anova 在定义上估不了它"
   "（它只会改测时间主效应）；请用 mixed_effects（随机截距可估组间处理），"
   "或 did（若有处理前后时点）；"
   if _absorbed and not _keep else "")
```
即：**把 `repeated_measures_anova` 从这条建议里去掉**，或降级为「只想看时间主效应时才用」。
（本条 + M2 的披露补上后，即便用户误入，也会看到「⚠ 检测到组间因子 treated 未纳入模型」。）

---

## 其余 SHOULD-FIX / NICE-TO-HAVE

### S1（SHOULD-FIX）— complete-case 删除的**规模**与**机制**未预警；全军覆没时是死胡同
`:534-545`。实测：随机删 20% 的**行** → **删掉 60 名受试者里的 38 名（63%）**，
summary 只有一句括号「（去除 38 名缺条件/重复的受试者）」，随后照常宣告
「22 名受试者…F(4,84)=59.065, p=1.24e-23（显著）」。
- 缺 listwise deletion 在 MNAR/脱落下有偏的披露（临床重复测量里脱落几乎从不是 MCAR）。
- 缺「丢弃比例过大」的阈值预警。
- 「每人各缺一个不同 week」→ 0 名完整 → 失败消息是死胡同，**不指向能处理不平衡数据的
  `mixed_effects`**——而 `mixed_effects` 恰恰不需要平衡。

建议：把 `n_dropped` / `drop_frac` 写进 `estimates`；`drop_frac > 0.2` 时加 ⚠；
`n_subj < 2` 的失败消息追加「RM-ANOVA 需完全平衡；不平衡/脱落数据请改用 mixed_effects
（它用全部可用观测，不做 listwise 删除）」。

### S2（SHOULD-FIX）— 有重复格时自动检测整体失效，且**报错理由是错的**
实测（10 名受试者在 week=2 有两行）：
```
重复测量方差分析失败：需要 长表(...)（已自动查找：受试者=None、within 因子=None、结果='pain_score'；
  within 因子须在同一受试者内取到 ≥2 个值——组间因子如试验臂不算）
```
`subject` 明明在表里。根因在 `profiler/profile.py:121` 的 `pair.duplicated().sum() == 0`：
**(unit,time) 必须严格唯一**才设 `unit_col`；一旦有重复格，`fp.unit_col` 变 None，
分支的自动检测就永远到不了自己 `:534-538` 那个专门修重复格的平衡过滤器。
**显式 config 时过滤器工作正常**（实测：删掉 10 名，50 名 × 5 条件，F(4,196)=123.378）；
`statsmodels.AnovaRM` 自己也会拦（实测报 `ValueError: The data set contains more than one
observation per subject and cell…`），所以**不产错数**——纯粹是诊断指错方向。
建议：失败消息里对「subject 候选存在但 (subject,within) 有重复」单独给一句话，
并提示 `AnovaRM(aggregate_func='mean')` 是另一条合法路（每格多次试验的设计很常见）。

### S3（NICE-TO-HAVE）— 日期型 within 因子的失败理由不解释「水平数≈行数」
实测（每人自己的起始日，117 个不同日期）：只说「只剩 0 名完整受试者」，
没说「within 因子取了日历日期、117 个水平、每人各不相同」。
建议在 `n_cond > n_subj` 或 `n_cond` 大于某阈值时先给一句结构诊断。
另注：`:471` 的基数上限 `len(df)//2` 相当宽松（300 行可放行 150 个水平），
`:535` 的 `groupby(...).size().unstack()` 会展开成 `n_subj × n_cond` 的稠密矩阵——
大表上是个未验证的内存风险（**未验证的怀疑**，本次没测到 OOM）。

### N1（NICE-TO-HAVE）— `estimates` 缺关键字段
`:587-593` 只写了 8 个键。缺 `num_df`/`den_df`（只存在于 summary 字符串里）、
`mauchly_W`、`hf_epsilon`、`n_dropped`。下游报表/网页无法复核自由度。

### N2（NICE-TO-HAVE）— `tests/test_repeated_measures_long_format.py:86-93` 的断言是恒真式
```python
assert ("n_conditions" not in res.estimates) or res.estimates["n_conditions"] == 2
```
注释写「honest failure, not a fake ANOVA」，但这个 `or` 的第二个分支**恰好放行了注释所说的
那个 bug**（若真产出一份 2 条件的假 ANOVA，断言照样通过）。
实测真实行为是诚实失败（`estimates == {}`），所以**当前行为没问题、测试没保护住它**。
建议改成 `assert res.estimates == {} and "失败" in res.summary`。

---

## 我验证过是对的（不要动）

1. **建造者宣称的基线数字全部属实**：F(4,236)=139.386、p=5.998e-61、偏 η²=0.7026、
   Mauchly 拒绝 → GG，与 R `car::Anova` 逐位一致。
2. **自由度在每一种病态形状下都算对了**：(n-1)(k-1) 与手算/R 一致 ——
   285 行→F(4,188)、240 行→F(4,84)、结构缺格→F(4,116)、1 名脱落→F(4,232)、3 人→F(4,8)。
   **没有出现「静默给出错的自由度」**（本次审查最担心的那种失败模式没有发生）。
3. **GG ε / HF ε / GG 校正 p 的数学是对的**，与 R 逐位一致（0.6323637 / 0.6630327 /
   1.592761e-39），`_orthonormal_contrasts` 的 QR 构造由 W 与 R 完全一致间接证实。
4. **偏 η² 公式对**：SS_cond/(SS_cond+SS_error) = 4372/(4372+1850.5) = 0.7026，与 R 的
   `Sum Sq` / `Error SS` 吻合。
5. **平衡过滤器（`:534-538`）在显式 config 下是真守卫，不是摆设**：缺格、重复格、
   完全重复行三种都正确剔除并披露；`AnovaRM` 自己也会对重复格抛 `ValueError`，双保险。
6. **within 守卫挡住了 `arm`**：组间因子不会被选作 within（`q9_base` 取 `week` 而非 `arm`），
   显式 `config={"within":"arm"}` 也是诚实失败（`estimates == {}`），不出假 ANOVA。
7. **不误杀脱落者**：单个受试者脱落不会阻断检测，只被平衡过滤器剔除并披露。
8. **`n_cond` 没有过期问题**（我原本怀疑它在 `:529` 早算、过滤后不刷新）：
   `(cell == 1).all(axis=1)` 要求每个幸存者在**所有**水平上各 1 次，
   所以幸存集合的水平数恒等于过滤前的 `n_cond`。**这条怀疑被证伪，不是 bug。**
9. **连续时间被基数守卫正确挡住**（`:471`），不会退化成每格一个观测。
10. **B1 的修复是真的**：秩亏建议确实只在 `_absorbed and not _keep`（关键项不可估）时才出现。
11. `tests/test_repeated_measures_long_format.py` + `tests/test_repeated_measures_anova.py`
    **11 passed**（EXIT=0）。

## 复现材料
探针脚本在
`C:\Users\Vambr\AppData\Local\Temp\claude\e--my-projects-Researchforge\987b032e-b5d9-4979-afc8-3f55d0a682b1\scratchpad\rmq\`
（`harness.py` / `q9a.py` `q9b.py` `q9c.py` / `q10.py` / `q11b.py` / `q12.py` `q12b.py` / `q13.py`
/ `sph.py` / `sweep.py` / `check.R` `check2.R`）。R 用
`C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe --vanilla`。
