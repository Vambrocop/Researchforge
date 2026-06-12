# ResearchForge

数据驱动的自动研究引擎：识别数据 → 诊断/清洗 → 按"数据结构 + 研究目标"推荐分析（带严谨度评审与知情覆盖）→ 你选 → 自动执行并保存代码/图/表/报告。

- 设计：`docs/superpowers/specs/2026-06-12-researchforge-skeleton-design.md`
- 实现计划（分阶段）：`docs/superpowers/plans/2026-06-12-researchforge-mvp.md`
- 能力盘点：`docs/analysis-catalog/skills-inventory.md`

## 快速开始

```bash
python scripts/setup.py        # 安装引擎 + 依赖，创建 outputs/ data/
python -m pytest -q            # 跑测试
python -m researchforge.cli --version
```

## 开发提示

- **跨平台**：代码用 `pathlib`、不写死盘符；目标兼容 Windows / macOS / Linux。
- **Windows 注意**：若 `python` 命令异常（可能被 Microsoft Store 的占位 stub 拦截），改用 **`py -3`**：
  `py -3 -m pytest -q`、`py -3 -m researchforge.cli --version`。
- 测试通过 `pyproject.toml` 的 `pythonpath = ["."]` 直接 import 本地包，无需 editable 安装。
