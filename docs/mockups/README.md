# ResearchForge 前端原型（design mockups）

**用途**：在动手建真前端前，先用自包含 HTML 把布局 / 风格 / 流程定下来（"Artifacts 思路"，但留在仓库里、浏览器直接打开）。

**重要**：这些是**纯静态原型**——数据是假的、**没连引擎**。Artifacts/沙箱浏览器跑不了 Python/R，所以**不能**做生产前端；真前端会是 FastAPI 服务的 web app（HTML+模板 或 JS 框架），调用 `researchforge/web/` 的后端跑真分析。原型只用来定**设计**。

## 两个方向（浏览器打开对比）

| 文件 | 方向 | 气质 |
|---|---|---|
| [researchforge-ui-A-minimal.html](researchforge-ui-A-minimal.html) | **A · clean-minimal** | 留白多、单列分步（上传→推荐→报告）、暖绿单色、卡片式；像一份干净的研究助手 |
| [researchforge-ui-B-dashboard.html](researchforge-ui-B-dashboard.html) | **B · dashboard-dense** | 左侧栏 + 密集表格、冷色控制台风、信息密度高；像一个分析工作台 |

两者**内容一致**（同一份农业面板假数据 farm_panel.csv，目标=因果效应），只是视觉语言不同，都覆盖真实引擎概念：
- 🟢🟡🔴 严谨度灯 + 按严谨度排序
- 6 维方法学评分卡（流行 / 可发表 / 美观 / 难度=COST / 契合 / 新颖）
- 14 个分析目标的目标选择器
- ⚠ 偏差披露 / 知情覆盖
- 结果报告：摘要（含 ⚠）+ 估计表 + 事件研究图 + 可下载产物（CSV/PNG/code.py/报告）

## 怎么用
浏览器直接打开 `.html`；点步骤 / 目标 / "运行" 体验流程。挑一个方向（或 A/B 混搭点），定下来我再建真前端（接 FastAPI）。
