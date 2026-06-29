# CLAUDE.md — ResearchForge 项目宪法

> 给在本仓库工作的 AI（及人类协作者）的约定。每轮先读它。

## 这是什么
ResearchForge = **方法学大杂烩引擎**：丢数据 → 自动识别类型/结构/质量 → 推荐可行分析（带 🟢🟡🔴 严谨度判语 + 偏差披露 + 知情覆盖 + 6 维方法学评分卡）→ 自动执行 → 出代码/图/表/报告。北极星：方法越全 + 自动选模型越聪明。已有 60+ 分析。自我进化：`cli discover` 发现+评分趋势方法→候选队列（不自动上线）。

## 运行
- **状态速览（接手先跑这个）**：`py -3 -m researchforge.cli status` —— 一眼看 健康分 / 🔜下一波 / 自动探测的待改进（live 从仓库算，不会过期）。新会话第一件事。
- **`py -3`**（不是 `python`，Windows 上裸 `python` 会挂）；UTF-8：`PYTHONUTF8=1`。
- 测试：`py -3 -m pytest -q`。**查退出码用 `>log 2>&1; echo EXIT=$?`，别用 `| tail` 屏蔽 pytest 的退出码**。
  - **提速**：全量并行 `py -3 -m pytest -n 2 -q`（~2:49 vs 串行 ~4:29）。**别用 `-n auto`**——R worker 内存重，会 `MemoryError`/报错；`-n 2` 是本机安全档。
  - **快循环** `py -3 -m pytest -m "not slow" -q`（跳 35 个重模型测试文件，~51s/138 测，验证引擎plumbing 够用）；慢测清单在 `tests/conftest.py` 的 `SLOW_MODULES`（加重方法测试时同步更新，刷新看 `pytest -n 2 --durations=25`）。
- 跑分析：`py -3 -m researchforge.cli run <data.csv> <analysis_id>`。

## 加一个分析（标准流程）
一个分析 = ① `executor/branches/<family>.py` 里一个 `@register("<id>")` 处理器（`def _branch_<id>(ctx): ...`；见下「引擎架构」。**别再往 `run.py` 加 elif**——巨石正在拆）② `catalog/entries/*.yaml` 一条目（preconditions/produces/biases）③ `tests/test_<id>.py`。然后：测试 → Opus 双审 → 据 CHANGES 修 → 全量测试 → 本地 commit。
**有脚手架别手搓**：用 `/add-analysis` 技能起步（含分支/条目/测试模板 + R 桥降级骨架）；推断双审派 `inference-reviewer` 子代理（审者≠建者，见 `.claude/agents/`）。改 `*.py` 后有 PostToolUse 钩子跑 `py_compile` 即时查语法（不必等 3–5min 全量套件；钩子在 settings.local.json）。

### 引擎约定（照抄，别重新发明）
- **用户可配置覆盖**：`run_analysis(fp, entry, output_root=..., config={...})` 的 `config` dict 携带用户对实质默认的覆盖（列角色/锚点/参数）。分支里 `cfg = config or {}`，按 `cfg.get("<key>")` 读、缺则回退自动默认（**默认必须仍能独立跑**）。入口：CLI `run <data> <id> --config '{...}'`(JSON)、`web/service.run_for_path(..., config=)`。键名按分析记在 `docs/loop-decisions.md`。已接：回归族 `outcome`/`predictors`。**新分支若有实质默认，顺手接 config 键并记文档**。
- **结果变量惯例**：回归族取「第一个连续列」为结果(outcome)；其余连续/二值为预测变量（可被 `config["outcome"]`/`["predictors"]` 覆盖）。
- **R 后端**：经 `executor/rbridge.py`，**可选 + 优雅降级** —— 先 `rbridge.r_available()` + `r_package_available(pkg)`，缺则回退纯 Python 或诚实提示（指向纯 Python 替代）。列名进 R formula 前过标识符守卫 `re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", c)`（防注入/解析）。temp CSV 写输出目录、`finally` 里删。R 代码经审才接，**运行时不联网取**。
- **产物**：CSV + PNG（matplotlib `Agg`）；填 `estimates` dict；写中文 `summary`（含 ⚠ 偏差/假定披露）。best-effort try/except 包图，缺 matplotlib 不中断。
  - **图标签字体**：**CJK 已支持**（阶段0）——`_init_mpl_style`（`_helpers/core.py`，每次 run 的统一入口，run_analysis 在分支前调用）检测到系统中文字体即自动接入字体回退链 + `axes.unicode_minus=False`，中文标签可正常渲染、无字体则优雅回退英文。**英文标签仍是最稳妥默认**（可移植到无 CJK 字体的 headless/CI）；要中文须确保目标环境有中文字体。别在分支里各自设字体——统一走 `_init_mpl_style`。
- **profiler "id" 陷阱**：整数值且全不同的列会被判为 `id` 类（非 count/continuous）。按列名锁定的检测（如 sand/silt/clay、duration）应接受 `id` 类。
- **survival 等**：时长列可能被 profiler 当 `time_col`——这类分支别排除 time_col。

### 引擎架构 & 读码纪律（怎么避免「prompt too long」）
- **分发 = 纯注册表**：`run_analysis`（`executor/run.py`，现仅 ~148 行）只做 setup（读数据/cfg/输出目录/`summary`…）→ `BRANCH_REGISTRY.get(entry.id)` 命中则 `handler(ctx)`、否则 else 占位 → teardown（写 code/report、返回 `RunResult`）。elif 链已全部迁出。
- **分析逻辑住在 `executor/branches/<family>.py`**：每分支是 `@register("<id>") def _branch_<id>(ctx)`，开头 `df, fp, entry, cfg, d = ctx.df, …；files, summary, estimates, code = ctx.files, …` 解包后写逻辑（**mutate** summary/estimates/files/code，别 rebind）。`Ctx`/`register`/`BRANCH_REGISTRY` 在 `executor/_branch_api.py`。
- **helper 在 `executor/_helpers/{core,backends}.py`**（core=计算/绘图/纯 Python 方法；backends=R 桥/econml/doubleml/semopy 委托），由 `run.py` **re-export**，故 family 模块与测试仍 `from researchforge.executor.run import _xxx` 不变。`branches/__init__.py` import 各 family 触发注册；`run.py` **末尾**才 import branches 包（避免循环）。
- **已拆解（历史教训）**：run.py 曾 7935 行 / `run_analysis` ~5500 行 / 67 分支 elif——**整文件读一次就撑爆上下文（VS Code「prompt too long」元凶）**。现最大模块 <1500 行。**护栏**：`tests/test_module_size.py` 强制 `researchforge/` 与 `tests/` 的 *.py ≤1500 行（超了就拆），评分卡设计维也据此给分。
- **防巨石复发的扩展约定（熔炉要长很多方法，必须可扩展）**：① **一个分析 = 一个模块**；② family 先是单文件 `branches/<family>.py`，**逼近 ~1200 行就提升为包** `branches/<family>/`（每方法一模块 `<id>.py`，并把该方法的 helper 从 `_helpers/` 搬进来同住）；③ `branches/__init__.py` 用 `pkgutil.walk_packages` **自动发现注册**——丢个文件进对应目录就自动注册，不用改 `__init__`、不抢 merge；④ 跨 family 共用的 helper 才留 `_helpers/`。护栏是强制力，本约定是「逼近时怎么拆」。当前金丝雀：`_helpers/backends.py`(1436)、`branches/statistics.py`(1021) 最先到顶。
- **读码纪律**：**别整文件读 `run.py` / 大文件**——用 `Grep` 定位 + 带 `offset/limit` 的 `Read` 只读片段；长会话定期 `/compact`；新分支进 `branches/` 让单文件保持小而专。（`/add-analysis` 技能的 run.py-elif 模板待更新为 branches/ 处理器。）

## 红线 & 工作流（不可逆动作守紧）
- **push gating**：自由本地 commit，但**只有用户说「今天 ok」才 push**（自动 push 钩子已移除）。用户忘了就查 `git log origin/main..HEAD` 提醒，别擅自推。
- **双审**：有真统计推断的方法 → 派 Opus 子代理审推断正确性（审者≠建者）；确定性方法（纯算/复用已审 helper）可实测验证、不必派审。**建造者可在有证据时驳回审查者**（如暴力验证），但要在汇总里告诉用户。Fable 5 的工作流/计划当**导师基准**；Agent 工具调不动 `claude-fable-5`（无权限）。
- **实质决策**（投入产出/校准锚点/X-M-Y 路径等）：默认 + 披露 + 追加到 `docs/loop-decisions.md`，**别阻塞**，用户异步拍板。
- **状态假设 & 先推断**：默认推进时把所做假设**一句话写明**（别静默填空）；能从代码/数据/已给指令推断的（列名/语言/已下的指令）先推断，别为可推断的事去问。
- **停-条件**：只在 ①踩红线 ②数据不可行 ③门禁挂 才打断用户，其余按推荐自走。
- **汇报从简**：让产物（表/图/报告/摘要）说话，配简短说明（含 ⚠ 披露），少冗长后记。
- **诚实**：零结果照报、不编数据、不确定就标 ⚠。
- **留痕**：受硬件/装包/后端限制**绕过或降级**的、以及双审/建造时冒出的**好点子**，都追加到 `docs/deferred-log.md`（未做事项 + 优化灵感日志，供后续回看）。

## 关键文件
- `executor/run.py`（setup + 注册表分发 + teardown + helper；分析分支正迁往 branches/）、`executor/_branch_api.py`（`Ctx`/`register`/`BRANCH_REGISTRY`）、`executor/branches/<family>.py`（各分析 `@register` 处理器）、`executor/rbridge.py`（R 桥）、`catalog/entries/*.yaml`（方法库）、`catalog/schema.py`（Precondition/AnalysisEntry）、`catalog/discover.py`（自我进化发现引擎）、`recommender/match.py`（precondition 匹配）、`recommender/scoring.py`（6 维方法学评分卡）、`profiler/`（指纹/类型/质量）、`web/`（FastAPI）。
- 本地自动化（`.claude/`，gitignored）：`agents/inference-reviewer.md`（推断双审子代理）、`skills/add-analysis/`（加分析脚手架）、`settings.local.json` 的 PostToolUse py_compile 钩子。
- 记忆见 `~/.claude/.../memory/MEMORY.md`（项目定位/自主权/路线图/R桥策略/评分feature 等）。
