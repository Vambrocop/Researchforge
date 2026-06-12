# ResearchForge

Data-driven automated research engine: identify data → diagnose/clean → recommend analyses based on "data structure + research objective" (with rigor review and informed override) → you select → auto-execute and save code/figures/tables/reports.

- Design: `docs/superpowers/specs/2026-06-12-researchforge-skeleton-design.md`
- Implementation plan (staged): `docs/superpowers/plans/2026-06-12-researchforge-mvp.md`
- Capabilities inventory: `docs/analysis-catalog/skills-inventory.md`

## Quick Start

```bash
python scripts/setup.py        # Install engine + dependencies, create outputs/ data/
python -m pytest -q            # Run tests
python -m researchforge.cli --version
```

## Development Tips

- **Cross-platform**: Code uses `pathlib`, no hardcoded drive letters; targets compatibility with Windows / macOS / Linux.
- **Windows note**: If `python` command fails (may be intercepted by Microsoft Store's placeholder stub), use **`py -3`** instead:
  `py -3 -m pytest -q`, `py -3 -m researchforge.cli --version`.
- Tests pass `pythonpath = ["."]` via `pyproject.toml` for direct local package import; editable install not required.
