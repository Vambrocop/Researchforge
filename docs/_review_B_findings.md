# 冷审报告 · 范围 B（时序成本模型 + 回归族守卫 + 重复测量）

审查者 ≠ 建造者。提交：`bd9fff9`、`5a11430`、`a90d4ac`、`9c6604e`。
**只读审查**：本文件是唯一写入物，未改任何代码、未 commit、未 push。

所有数字均为本机实测（statsmodels 0.14.6，R 4.6.0 + car）。未实测的推断一律标注「未验证的怀疑」。

---

## A. ARIMA 成本模型（Q1–Q5）

### A0 —— Q1 硬结论：`k_states` 的真实公式

**去查了 statsmodels 源码**
`…/site-packages/statsmodels/tsa/statespace/sarimax.py`：

```python
# L425-426
self._k_order = max(self.k_ar + self.k_seasonal_ar,
                    self.k_ma + self.k_seasonal_ma + 1)
# L453-457
k_states = self._k_order
if not self.simple_differencing:
    k_states += (self.seasonal_periods * self._k_seasonal_diff + self._k_diff)
```

其中 `k_ar=p`、`k_seasonal_ar=sp*P`、`k_ma=q`、`k_seasonal_ma=sp*Q`（L397-405）。
`simple_differencing` 默认 **False**，分支也没传它，所以差分项**确实进状态向量**。

真实状态维数：

```
k_states = max(p + sp*P,  q + sp*Q + 1) + sp*D + d
```

实测验证（12/12 全中，与 `SARIMAX(...).k_states` 逐一比对）：

| order | sorder | 真实 k_states | 上式 | 建造者 `max(p,q+1)+sp(P+Q)` |
|---|---|---|---|---|
| (0,1,0) | (0,1,0,52) | 54 | 54 | **1** |
| (0,1,1) | (0,1,0,52) | 55 | 55 | **2** |
| (0,1,0) | (0,1,1,52) | 106 | 106 | **53** |
| (1,1,1) | (1,1,1,52) | 107 | 107 | 106 |
| (2,1,2) | (0,1,0,12) | 16 | 16 | **3** |
| (0,1,1) | (0,1,1,12) | 27 | 27 | **14** |
| (1,0,1) | (1,0,1,12) | 14 | 14 | **26** |
| (2,1,2) | (1,1,1,12) | 28 | 28 | 27 |

**结论：`max(p, q+1) + sp*(P+Q)` 不是 statsmodels 的状态维数**，两处错：

1. **漏掉 `sp*D + d`**。季节差分**是**带在状态里的。建造者的注释
   「Seasonal DIFFERENCING is applied to the data, not carried in the state」
   （`timeseries.py:178-181`）**与源码相反**。他用「(0,1,0)(0,1,0,52) 只要 0.8s 而
   (0,1,0)(0,1,1,52) 要 213s，差 266 倍，所以 D 不该计」来反推——这个推理把
   **优化器迭代次数**的差异错算到了状态维数头上（前者 `k_params=1`，后者 `k_params=2`
   且要在 52 阶滞后上搜季节 MA，实测 nfev 15 vs 212）。维数其实是 54 vs 106（差 3.85×）。
2. **把 AR 与 MA 两侧相加**（`sp*(P+Q)`），真实是**取 max**（`max(p+sp*P, q+sp*Q+1)`）。
   P、Q 同时 >0 时维数高估约 2×，成本高估约 4×。

**哪些组合会被错判**（budget=5e6，默认季节网格 p,q≤2 / P,Q≤1 / D=1）：

- **错误放行（该贵却放过去 → 仍然吃时间）**：所有 `P=Q=0` 的季节候选。
  建造者模型给 `dim = max(p,q+1) ≤ 3` → 成本 `n*9`，即「白送」；真实维数是 `sp+d+max(p,q+1)`。
  实测 `(2,1,2)(0,1,0,52)` @ n=2000：建造者成本 **1.8e4**（预算的 0.36%），真实 n·dim²=6.27e6
  （超预算），**实际单次拟合 10.97s**。co2 形状（n=2225, sp=52）上这类候选有 9 个 ——
  这正是「修完仍要 22.6s」的来源：成本闸对它们完全无感，真正拦住它们的是墙钟预算。
- **错误跳过（该便宜却被拦）**：`P=Q=1` 的乘性季节模型被高估 ~4×。
  实测 `(2,1,2)(1,1,1,12)` @ n=8000：建造者成本 5.83e6 > 5e6 → **被跳过**，
  而**实际只要 22.7s**，且 **AICc = 11636.9 vs 被保留的 `(2,1,2)(0,1,0,12)` 的 16411.7**
  ——跳掉的是好 4775 个 AICc 点的模型。
- **排序错乱**：`(0,1,1)(0,1,1,52)` 与 `(2,1,2)(1,1,1,52)` @ n=400 单次似然评估
  **实测都是 145ms**（145.65 / 145.33），建造者成本却是 1.17e6 vs 4.58e6（差 3.9×）。

**建议修法（A0）**：`_cost` 换成真实维数，指数提到 ~4（标定见 A2）：

```python
def _cost(order, sorder):
    _p, _d, _q = order
    _P, _D, _Q, _s = sorder
    # statsmodels SARIMAX (simple_differencing=False) k_states — sarimax.py:425 / 453-457
    dim = max(_p + _s * _P, _q + _s * _Q + 1) + _s * _D + _d
    return float(_n) * float(dim) ** 4
```

---

### A1 【MUST-FIX】「回退也受预算约束」是假的：分支把 `_auto_order` 的廉价回退扔了

**症状**：`_auto_order` 在 `best is None` 时算了一个受预算约束的廉价回退
（`timeseries.py:229-237`，还设了 `info["fallback_cheap"]`），**调用点无条件覆盖它**，
改去拟合 `(1,d,1)(1,D,1,sp)` —— 正是 commit message 里那个 219s 的模型。
`9c6604e` 的 commit message 第 ③ 条、`tests/test_arima_cost_budget.py` 模块 docstring
（"the FALLBACK order has to respect the budget too"）**与实际行为不符，本身就是 finding**。

**根因**（`researchforge/executor/branches/timeseries.py:297-302`）：

```python
order, sorder, model, oinfo = _auto_order(y.to_numpy(), sp, cfg)
if model is None:                      # every candidate failed -> honest fallback
    order = (1, oinfo["d"], 1)                       # <- 覆盖了 _auto_order 的返回值
    sorder = (1, oinfo["D"], 1, sp) if sp else (0, 0, 0, 0)
    model = _fit_sarimax(y.to_numpy(), order, sorder)
```

**最小复现**（`fit_cost_budget` 是**已声明的 config 键**，用户可达）：

```python
import numpy as np, pandas as pd, tempfile, os
from researchforge.executor.branches.timeseries import _auto_order
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

rng = np.random.default_rng(3); t = np.arange(72)
y = 100 + 0.5*t + np.array([-8,-6,-2,2,6,9,8,5,1,4,12,20])[t % 12] + rng.normal(0, 2, 72)
print(_auto_order(y, 12, {"fit_cost_budget": 10.0})[:2])     # -> ((0,1,1), (0,1,0,12))

df = pd.DataFrame({"month": pd.date_range("2019-01-01", periods=72, freq="MS").strftime("%Y-%m"),
                   "revenue": y.round(2)})
tmp = tempfile.mkdtemp(); csv = os.path.join(tmp, "m.csv"); df.to_csv(csv, index=False)
res = run_analysis(profile_dataset(csv), Catalog.load().by_id("arima"),
                   output_root=os.path.join(tmp, "o"), config={"fit_cost_budget": 10.0})
print(res.estimates["p"], res.estimates["P"], res.estimates["Q"])
```

**实测**：
- `_auto_order` 返回 `order=(0,1,1) sorder=(0,1,0,12)`，`model=None`，`n_fits=0`，
  `skipped_costly=36`，`fallback_cheap=True`。
- **分支实际拟合的是 `SARIMA(1,1,1)(1,1,1)[12]`**（`p=1.0, P=1.0, Q=1.0`）。
  在 sp=52 的形状上这就是那个 219s 的模型 —— 挂死保护在这条路径上**完全失效**。

**同一处还有两句假话（一并算 MUST-FIX，它们进了报告正文）**：
1. summary 同时出现「实拟合 **0** 个候选」和「当前阶数是 **可承受候选中的最优**」。
   零个候选被拟合时它不是任何东西的最优，它是回退阶数，而且是**最贵**的那个形状。
2. degrade_note 写「自动定阶网格**全部拟合失败**」，真实原因是**一个都没试**（全被成本闸跳过）。

**建议修法**：
```python
order, sorder, model, oinfo = _auto_order(y.to_numpy(), sp, cfg)
if model is None:
    model = _fit_sarimax(y.to_numpy(), order, sorder)     # 用 _auto_order 已给的受预算阶数
    degrade_note += (" ⚠ 自动定阶未能拟合任何候选（"
                     + ("全部候选超出计算预算" if oinfo["skipped_costly"] and not oinfo["n_fits"]
                        else "网格全部拟合失败")
                     + f"），已回退 {order}{sorder[:3]} —— 该阶数**未经 AICc 比较**。")
```
并把「可承受候选中的最优」限定为 `oinfo["n_fits"] > 0` 才出现。

---

### A2 【SHOULD-FIX】预算 5e6 不是时间单位：同一个数字对应 8s / 31s / 151s（Q2）

**症状**：`_FIT_COST_BUDGET` 的注释说「n=2225/sp=52 投影 ~2.4e7、n=60/sp=12 投影 ~3.5e4，
5e6 两侧各有两个数量级余量」。余量是**维数空间**的，不是**时间**的。

**实测**（单次完整 `fit()` 墙钟秒）：

| n | sp | order | sorder | 建造者成本 | 闸门 | 实测单次拟合 |
|---|---|---|---|---|---|---|
| 600 | 12 | (0,1,1) | (0,1,1,12) | 1.18e5 | 放行 | **0.42s** |
| 2000 | 12 | (2,1,2) | (1,1,1,12) | 1.46e6 | 放行 | **6.25s** |
| 2000 | 52 | (2,1,2) | (0,1,0,52) | 1.8e4 | 放行 | **10.97s** |
| 600 | 52 | (0,1,1) | (0,1,1,52) | 1.75e6 | **放行** | **51.10s** |
| 7000 | 24 | (0,1,1) | (0,1,1,24) | 4.73e6 | **放行**（94% 预算） | **29.29s** |
| 8000 | 12 | (2,1,2) | (1,1,1,12) | 5.83e6 | **跳过** | 22.71s |

把「成本恰好 = 5e6」的边界换算成秒：
- sp=12, P+Q=1 → n≈25500 → **约 8s**
- sp=24, P+Q=1 → n≈7400 → **约 31s**（n=7000 实测 29.29s）
- sp=52, P+Q=1 → n≈1780 → **约 151s**（n=600 实测 51.1s，时间近似线性于 n）

**同一个预算数字对应 8s–151s，19 倍散布**，而且方向是反的：季节周期越大越放行得越久。

**统计后果（实测，非推理）**：
- n=8000 / sp=12：`(2,1,2)(1,1,1,12)` 被跳过，实际只要 22.7s，
  **AICc 11636.9 vs 保留下来的 16411.7**，差 4775 点。
- n=7000 / sp=24（≈10 个月小时级数据，很常见）：`(0,1,1)(0,1,1,24)` 成本 4.73e6
  卡在预算内被放行，**AICc 10197.1 vs `(0,1,1)(0,1,0,24)` 的 14799.1**；
  但 n 一涨到 ≥7400（约 10.3 个月）就翻过 5e6 被跳掉 —— **同一份数据多攒 10 天，
  季节项被整体砍掉、AICc 断崖 4600 点**。这就是「一整类合理季节数据被无谓砍掉季节项」。

**根因**：成本量纲是 `n·dim²`，而实测拟合时间 ≈ `c · n · k_states^4`
（用 sp=12/24/52 三个边界点回归，两段指数 4.09 与 4.06 一致；`c ≈ 6.5e-10` 用
n=600/sp=52/k=107/51.1s 标定，回代 n=7000/sp=24/k=51 预测 30.8s、实测 29.29s）。
二次幂低估了大周期的增长，所以同一阈值在小 sp 上太紧、在大 sp 上太松。

**建议修法**：预算改成**秒**，让 config 键有物理意义：
```python
_FIT_SECONDS_BUDGET = 30.0          # 单次拟合目标上限（秒）
_KALMAN_SEC_PER_UNIT = 6.5e-10      # 实测标定：t ≈ c · n · k_states**4
```
保留 `fit_cost_budget` 作别名以免破坏已声明的 config 契约；披露里报**预计秒数**
而不是无量纲的 5e6（标定常数随机器变化，文案注明「按本机基准估算」）。

---

### A3 【SHOULD-FIX】墙钟预算兜不住单次拟合：成本闸放行的一次拟合就能花 51s（Q4）

**症状**：`if best is not None and elapsed > _budget: break` 只在**拟合之间**检查。
建造者已知这点（迭代 ①），认为成本闸能兜住。**兜不住**。

**实测**：n=600 / sp=52 / `(0,1,1)(0,1,1,52)`，建造者成本 1.75e6 < 5e6 → **放行**，
单次拟合 **51.10s**。按简约排序它是第 2 个被试的候选（p+q+P+Q=1 组第一个），
所以默认 20s 预算下真实耗时 ≈ 51s+，超预算 2.5 倍；n=1780 时同一形状约 151s（超 7.5 倍）。

非季节侧同样没有闸：sp=0 时 `dim = max(p,q+1) ≤ 4`，`n*16 > 5e6` 需要 n > 312500，
即**任何现实 n 都不会被跳过**，只能靠事后 break。（n≥3e5 的单序列现实与否：
**未验证的怀疑**，我没跑那么大的 n。）

**建议修法**：给单次拟合加**硬**截止。statsmodels 支持 `fit(callback=...)`，
callback 里超时抛异常即可中断（外层已有 `except Exception: continue`）：
```python
def _fit_sarimax(y, order, seasonal_order, deadline=None):
    def _cb(params):
        if deadline and _time.perf_counter() > deadline:
            raise TimeoutError("per-fit deadline")
    ... .fit(disp=False, callback=_cb if deadline else None)
```
（**未验证**：我没实测 callback 中断在 lbfgs 下能否干净退出；按代码路径它会抛出并被
`except Exception: continue` 吞掉，应当安全，但请建造者实测一次再落地。
低风险替代：按投影成本收紧 `maxiter`。）

---

### A4 【SHOULD-FIX】跳过候选的披露里有一个写死的假数字（Q3）

**症状**：`timeseries.py:385-388` 的跳过文案写死
「（季节周期 {sp} 下单次拟合可达**数百秒**）」，与 sp/n 无关地断言。

**实测**：A1 的复现里 sp=12 / n=72，报告照样印出「季节周期 12 下单次拟合可达数百秒」；
而 sp=12 的季节拟合实测是 **0.42s（n=600）/ 6.25s（n=2000）**。报告里出现了一个凭空的量级。

**建议修法**：把估算值算出来再印（`f"约 {_cost_seconds(...):.0f}s"`），或只说
「投影计算成本超预算」，不给编造的秒数。

**另**：`elapsed_s` 只在 `timed_out` 分支出现；跳过但没超时的运行（A1 情形）不报耗时，
用户无从判断预算是否值得放宽。建议无条件报 `elapsed_s` 与 `n_fits/n_candidates`。

---

### A5 【NICE-TO-HAVE】三处注释/测试与代码不符

1. `timeseries.py:52` 常量注释写 `dim ≈ max(p, q+1) + sp·(P+Q+D)`，而 `_cost`（L182）算的是
   `max(p, q+1) + sp*(P+Q)` —— **注释停留在被推翻的第 ② 版**。
2. `tests/test_arima_cost_budget.py::test_seasonal_differencing_is_not_priced_like_a_seasonal_arma_term`
   **在测试内部重新实现了一遍 `cost()`**，没有 import 被测函数。它测的是测试自己的副本，
   `_cost` 怎么改这条测试都绿 —— 一条**永远不会失败的守卫**。
3. 分支写出的 `analysis_code.py`（`timeseries.py:420-428`）用
   `enforce_stationarity=False, enforce_invertibility=False` 且**不带 `trend='c'`**，
   而实际拟合走 `_fit_sarimax`（True/True + d==0 时 `trend='c'`）。
   `_fit_sarimax` 的 docstring 明写这两个设置「load-bearing … must not be relaxed」
   （上一轮冷审的 MUST-FIX），**导出的复现代码恰好复现了那两个被修掉的 bug**。
   此项早于本批，但属 arima 分支。

### Q5 结论（AICc 的解释边界）

未触发 A1 那条路径时文案是诚实的：超时说「**已试候选中的最优**」、跳过说
「**可承受候选中的最优**」，并已披露「d 由单位根检验固定、跨 d 不可比」与
「有界网格、非完整 Hyndman-Khandakar」。**这一块我验证过是对的**。
要补的限定只有两处：A1 的 `n_fits==0` 情形；以及把「被跳过的候选可能才是最优」写实
（实测 AICc 11637 vs 16412，确实更优）。

---

## B. 回归族秩亏守卫（Q6–Q8）

### B1 【MUST-FIX】一个再普通不过的面板（含一个时不变协变量）现在整份分析归零

**症状**：`panel_fixed_effects` 对「firm × year 面板 + 一个时不变协变量（baseline / 性别 /
地区码 / 基线值）」直接 `失败` 返回、`estimates={}`。而该面板的处理效应**完全可估**。

**最小复现**：
```python
import numpy as np, pandas as pd, tempfile, os
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
rng = np.random.default_rng(11); rows = []
for i in range(40):
    baseline = rng.normal(50, 10); a = rng.normal(0, 2)          # 时不变协变量
    for t in range(6):
        treat = 1 if (i % 2 == 0 and t >= 3) else 0
        rows.append(dict(firm=f"F{i:02d}", year=2010+t, baseline=round(baseline,2),
                         capex=round(rng.normal(20,5),2), treat=treat,
                         profit=round(10+a+0.5*t+3.0*treat+0.2*rng.normal(),3)))
df = pd.DataFrame(rows); tmp = tempfile.mkdtemp(); csv = os.path.join(tmp,"p.csv")
df.to_csv(csv, index=False)
res = run_analysis(profile_dataset(csv), Catalog.load().by_id("panel_fixed_effects"),
                   output_root=os.path.join(tmp,"o"))
print(res.summary[-260:], res.estimates)
```

**实测**：
```
Panel two-way fixed effects 失败：设计矩阵秩亏（秩 47 < 列数 48）——说明某个预测变量被其它项
完全解释（最常见：处理变量在每个单位内不随时间变化，被单位固定效应吸收）。…
若为重复测量设计，请改用 repeated_measures_anova / mixed_effects；…
estimates = {}
```
而**同一个秩亏拟合里，可估方向的估计完全正确**：
- 秩亏拟合（statsmodels pinv）：`treat = 2.9788, se = 0.0599`
- 删掉被吸收的 `baseline` 重拟合：`treat = 2.9788, se = 0.0599`（真值 3.0）—— **逐位相同**。
- 只有 `baseline` 一个系数是垃圾（0.06698, se = 7.09e-4）。

**根因**：`researchforge/executor/branches/regression.py:25-60` 用「整个设计矩阵是否满秩」
当判据然后 `return`。但秩亏只毁掉**一个方向**；`resolve_predictors` 自动抓前 5 个
连续/计数/二值 列，面板数据里**几乎必然**抓到时不变列（基线值、性别、地区码、行业码），
于是旗舰面板方法在最常见的形状上直接不出结果。`did` 只把处理变量放进 rhs 所以幸免、
`ols_regression` 无 FE 项所以幸免 —— 死的恰好是面板方法。

附带：文案把原因钉死在「处理变量被单位固定效应吸收」并推荐
`repeated_measures_anova / mixed_effects`，对 firm-year 面板是**错误建议**
（此处正解是「时不变协变量本来就被 FE 吸收，丢掉即可」）。

**建议修法（R `lm` 语义：丢别名列 + 报其余 + 强披露）**：
```python
import numpy as np
X = np.asarray(model.model.exog, float); names = list(model.model.exog_names)
if int(np.linalg.matrix_rank(X)) < X.shape[1]:
    absorbed = []
    for v in rhs_vars:            # 谁被吸收了：逐个检验「能否被其余列完全解释」
        for j, nm in enumerate(names):
            if nm == f"Q('{v}')" or nm.startswith(f"Q('{v}')["):
                rest = np.delete(X, j, axis=1)
                r = X[:, j] - rest @ np.linalg.lstsq(rest, X[:, j], rcond=None)[0]
                if float(r @ r) <= 1e-18 * float(X[:, j] @ X[:, j]):
                    absorbed.append(v); break
    if rhs_vars and rhs_vars[0] in absorbed:
        ...   # 关键项本身不可估 -> 维持现在的诚实失败
    else:
        # 丢掉被吸收的预测变量重拟合，照常报告 + 强披露
        summary.append(f"⚠ 预测变量 {absorbed} 与其它项完全共线（面板里最常见：时不变列被"
                       f"单位固定效应吸收），已剔除后重估；其余系数不受影响。")
```
注意：**别用 `scipy.linalg.qr(pivoting=True)` 的主元顺序来点名**——我实测它在这个设计上
把 `Intercept` 报成别名列（主元按列范数排序），对用户毫无意义。
最低限度：只在**关键项**被吸收时整体失败，其余情况照报并点出被吸收的列名（现在连是哪一列都不说）。

---

### B2 【SHOULD-FIX】守卫只在 cond ≈ 2e14 才触发；真正危险的 1e5–1e12 区间无人看守（Q6）

**最小复现**：
```python
import numpy as np, statsmodels.api as sm
rng = np.random.default_rng(0); n = 200; x1 = rng.normal(0,1,n)
for eps in (1e-1,1e-3,1e-5,1e-7,1e-9,1e-12,1e-14,0.0):
    x2 = x1 + eps*rng.normal(0,1,n); X = np.column_stack([np.ones(n), x1, x2])
    y = 1 + 2*x1 + rng.normal(0,1,n); m = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"{eps:8.0e} cond={np.linalg.cond(X):9.3g} rank={np.linalg.matrix_rank(X)}/3 "
          f"b1={m.params[1]:10.3g} se={m.bse[1]:10.3g}")
```

**实测**（真值 b1 = 2.0）：

| eps | cond | rank | 守卫 | b̂₁ | SE |
|---|---|---|---|---|---|
| 1e-3 | 1.9e3 | 3/3 | 不触发 | 22.2 | 66.2 |
| 1e-5 | 2.0e5 | 3/3 | 不触发 | **-1.4e4** | 8.1e3 |
| 1e-7 | 2.0e7 | 3/3 | 不触发 | 6.6e5 | 8.2e5 |
| 1e-9 | 1.9e9 | 3/3 | 不触发 | -6.9e7 | 6.7e7 |
| 1e-12 | 1.8e12 | 3/3 | 不触发 | -3.4e10 | 6.4e10 |
| 1e-14 | 2.1e14 | **2/3** | **触发** | — | — |

**判断（重要，别过度反应）**：这一整段**不是无效推断**——OLS 仍无偏，SE 同步爆炸，
t/p 值是诚实的（b̂₁=-1.4e4、SE=8.1e3 → t=-1.7，正确地说「测不准」）。
所以不是「引擎报了错的数」，而是**报了一个量级荒谬的头条数字却不提共线性**。
`_degenerate` 的 SE 比值兜底（`se < 1e-8*|b|`）在这段**永不触发**（实测 se/|b| ≈ 0.5–3）。

**建议修法**：不要塞进 fail 分支，加一条**披露**：条件数（`np.linalg.cond(exog)`）或
关键项 VIF 超阈值时追加「⚠ 设计矩阵条件数 {cond:.1e}，预测变量高度共线：系数量级不稳定
（换一批样本可能翻符号），区间宽度已反映这一点；需要单独效应请改用岭回归/主成分回归或删冗余列」。

---

### B3 【SHOULD-FIX】误杀面（Q7）：成分数据与 n<p 被同一条守卫按同一套说辞拒绝

**实测**：
- **量纲悬殊但满秩**：dollars(1e3–1e12) + 比例(0–1) 的设计，cond 到 **4.08e12 仍不触发**；
  要到 scale=1e14（cond 4.06e14）才误判。**数值容差在现实量纲下不会误杀** —— 这点建造者是对的
  （`matrix_rank` 默认 tol = max(M,N)·eps·σ_max，需要 ~1e13 的比例失衡才咬）。
- **成分数据（sand+silt+clay = 100）+ 截距**：cond 5.79e17 → **触发**。数学上确实奇异
  （守卫没错），但文案说「处理变量被单位固定效应吸收 … 请改用 repeated_measures_anova /
  mixed_effects」——**对成分数据是彻底错误的建议**（正解是 ilr/alr 对数比变换或删一个分量）。
  而 `a90d4ac` 自己的 dogfood 清单里就有 compositional 这个域。
- **n < p**（10×13）：触发。这个是对的（OLS 本就不可识别），但同样给错建议。

**建议修法**：失败文案按**诊断出的原因**分叉：
① 某预测变量被 FE 吸收 → 现在这套话术（正确）；
② 多个预测变量线性相加为常数（闭合/成分）→ 指向对数比变换 / 删一个分量；
③ `n < 列数` → 指向正则化或减少预测变量；
并且**任何一种都应先尝试 B1 的「丢别名列 + 报其余」**，只有关键项本身不可估时才整体失败。
