# ResearchForge — Skeleton Design Document (Spec v0.1)

- Date: 2026-06-12
- Status: Pending user review
- Scope: This document defines only **skeleton/MVP phase**. Full multi-domain platform is vision, implemented in phases ("skeleton first, then fill in gradually").

## 1. Background and Vision

**One-sentence positioning**: Give ResearchForge a dataset, it auto-reads the data, recommends possible research analyses, you select, then auto-executes and outputs figures and reports.

**Core experience loop**:
> Auto-identify data → Auto-recommend analysis possibilities → You select → Auto-execute + auto-generate figures and reports

**Source of generality**: Analysis routing is driven by **data type/structure**, not hardcoded to any discipline — so naturally cross-disciplinary generalizable and "add domain one bit at a time".

**Source of novelty**: ① Cross-disciplinary method transfer (data-driven routing naturally brings methods from one domain to data of another); ② "Novelty lens" at recommendation stage — run one literature scan to tell you which angles existing literature covers, where blank spaces are; ③ Optional "frontier monitor" runs on auto-loop, continuously track new papers; ④ "GitHub collector" — auto-discovers new skills/paper replication code/new method packages and registers them. Details in §7.

**Starting domains**: Agricultural economics / general statistics (already have strong executors). Later expand: soil science, ecology, microbiology, **space/GIS and global macroeconomic research**.

**Cross-disciplinary vision**: Facing "**agri · bio · envir · econ**" (agronomy · biology/life sciences · environment/ecology · economics) cross-disciplinary combination research — "cross-disciplinary method transfer" (§7.1) is exactly for this.

**Final form (multi-modal delivery, shared same engine)**: ① One "super" **Claude skill** (use directly in Claude Code); ② **Web application** (open browser and use); ③ **Mini-program / desktop widget**.
**Current form (this phase)**: Portable **Git repository + Python engine**, driven by Claude Code; web/skill/mini-program wrapping are follow-up increments.

## 2. Architecture: Three Layers + Three Cross-cutting Mechanisms

```
                ┌─────────────────────────────────────────┐
   Data files ──▶ │ ① Profiler Data Profiling                │
  (CSV/Excel)    │   Output: DataFingerprint (structured JSON) │
                └───────────────────┬─────────────────────┘
                                    ▼
                ┌─────────────────────────────────────────┐
                │ ② Recommender Recommendations (hybrid C) │
                │   Rule catalog matching + LLM ranking/  │
                │   explanation                            │
                │   Output: ranked AnalysisRecommendation[]│
                └───────────────────┬─────────────────────┘
                         You select ◀────┤
                                    ▼ (selected item)
                ┌─────────────────────────────────────────┐
                │ ③ Executor Execution                      │
                │   Call corresponding skill/script →      │
                │   tables + figures + reports             │
                └─────────────────────────────────────────┘

  Cross-cutting A: Skill Ingestion (anytime add)  Cross-cutting B: Portability  Cross-cutting C: Frontier & Novelty
```

### Each Layer's Responsibilities and Interface Contracts

- **① Profiler (Data Profiling)**: Input data (**scan specified local data folder** for files, or fetch from public data sources, default prefer latest data — see §8), output `DataFingerprint` — a structured fingerprint: variable list and types (continuous/categorical/count/binary/datetime/ID/**geographic coordinates**), missing data, presence of `unit×time` panel structure, **whether time series (single/panel time series)**, **whether contains space/GIS dimension (coordinates/region/raster)**, presence of treatment/control groups, sample size, distribution sketch, and **mark data quality issues** (missing, outliers, duplicates, type/unit inconsistencies). **Read-only, deterministic, no LLM calls.**
- **①.5 Cleaning (data cleaning, enabled by default)**: Based on quality issues marked by Profiler, generate a **cleaning plan** by data type for your confirmation; after confirmation execute and retain **cleaning log**, ensure reproducibility.
- **② Recommender (Hybrid C)**: Input `DataFingerprint` + **optional research target/question** (explanation/causation vs prediction/forecasting vs classification/clustering — same data routed to different method families per target, e.g., panel data→DID or XGBoost prediction), output ranked `AnalysisRecommendation[]`, each item contains: method name, **prerequisite check (🟢met / 🟡caveat / 🔴unmet)**, **introduced statistical biases**, **rigor score**, expected output, executor reference, recommendation rationale. Rule catalog provides candidates and hard prerequisites; LLM handles ranking, explanation, and supplements for cases not covered by catalog. **Every recommendation must be transparent and explainable.** Even 🔴 unmet allows **informed override** (if you insist, do it, report records warnings/biases/precedents). Details in §3.
- **③ Executor (pluggable multi-backend)**: Input selected recommendation + data, via unified interface call corresponding backend executor (Python / R / Stata / deep learning framework), produce tables + figures + reports. **All artifacts — generated analysis code, images, tables, reports — land in `outputs/` and retained** (reproducible, traceable). This phase reuses `empirical-analysis-python` (Python); other backends see §8 reserved.

### Cross-cutting Mechanisms

- **A. Ingestion (anytime add, multi-channel)**: `skills_inbox/` ingestion → extract (domain/method/prerequisites/output) → register in analysis catalog → archive originals in `_processed/`. Make "anytime add" a first-class citizen. **Three ingestion channels**: ① **Manual** skill (`SKILL.md`) into inbox; ② **Paper absorption** — place PDF/DOI, extract its method/data prerequisite/analysis and register (can locate replication code side-by-side); ③ **GitHub collector** (§7.4) auto-discovers skill/code/methods then enters.
- **B. Portability (cross-platform)**: Everything into Git repo; engine-dependent skills installed via `setup` script to `~/.claude/skills/` (or bundled in repo); machine change = `git clone` + `setup`. **Must be cross-platform compatible Windows / macOS / Linux** — Python uses `pathlib`, no hardcoded drive paths; setup provides cross-platform versions (Python script, or `.sh` + `.ps1`). **Lightweight data principle**: repo holds code + synthetic data generator (`synth`) + network data references; real/large data goes `.gitignore` not in repo. **GitHub is single source of truth**, keep local minimal, easy to move, if lost can recover from cloud.
- **C. Frontier & Novelty (innovation)**: Hang "novelty lens" at recommendation layer (literature scan mark blank spaces), and optionally run "frontier monitor" on auto-loop to keep track of new papers continuously. Details in §7.

### Implementation Style: AI-agent Driven + Dual-mode Acquisition (Automatic / Manual)

System's "soft reasoning" parts — recommendation ranking/explanation, novelty lens, frontier monitoring, GitHub collection, paper absorption — all implementable by **dedicated AI agents** (reuse deep-research multi-agent pipeline, Claude Code subagents, agent teams), not hardcoded rules. Rules only handle hard prerequisites and deterministic steps (profiling, cleaning, execution).

**Dual-mode acquisition**: Each knowledge acquisition channel (skill / paper / frontier / GitHub) simultaneously supports ① **AI-agent automatic** (assist search and update) and ② **manual addition**, both coexist, switch anytime.

**Layered model orchestration ("foreman" mode)**: Orchestrator (Fable 5 / Opus 4.8 etc. high-tier models) handles planning, task assignment, review, quality control; bulk low-level work (read large volumes of papers, scan GitHub, profile multiple files, extract register, draft templates) handed to cheap model (Haiku / Sonnet) subagents. **Directly implementable** — Claude Code's Agent/subagent support model assignment (haiku/sonnet/opus/fable). Saves tokens and keeps quality with high-tier model gate.

> **Compute/scale note**: User is **Max plan**, can bear agent-intensive and periodic search heavier functions; architecture designed "as large as possible" (see §8 reserved). But **build order still first punch through one vertical cut, then expand surface** — goal is verify closed loop first, lower risk, unrelated to compute.

## 3. Recommendation Engine: How Hybrid Route C Lands

- **Rule catalog**: A set of structured entries (YAML/JSON), each = `{method, domain, preconditions, produces, executor_ref}`.
- **Matching**: Use `DataFingerprint` against each entry's `preconditions` → get candidate set (hard prerequisites ensure won't recommend unsupported methods).
- **LLM / agent layer**: Rank candidates, generate human-readable explanations, and give supplementary suggestions for cases not covered by catalog (mark "non-catalog item"). Can be implemented by dedicated agent (see §2 implementation style).
- **Reliability fallback**: Hard prerequisites gated by rules, LLM only does soft judgment on candidates already passing prerequisites, lower careless recommendation risk.
- **Method rigor review (Rigor & Review, review perspective)**: Each analysis (system-recommended or you force-specified) gets three-color conclusion — 🟢assumptions met / 🟡caveat / 🔴unmet — plus: ① which **statistical biases** introduced and direction/severity; ② **precedent in literature** ("has anyone done this", judge by paper search); ③ **rigor score** (data-method match degree). **Allow informed override**: you insist on 🔴 still can do, but report fully records warning, biases and precedent. Reuse academic-paper-reviewer / umbrella-review-skeptic "review / devil's advocate" perspective + empirical-analysis diagnostic tests. (Scoring rubric and auto-precedent lookup to refine later.)

## 4. This Phase Scope (Skeleton MVP)

**In Scope**:
1. Repository structure + `setup` script skeleton (dependencies and skill installation).
2. `DataFingerprint` schema + Profiler (**scan specified local data folder** for CSV/Excel): **data type identification** (include time series detection) + **data quality diagnosis** + **basic cleaning** (generate cleaning plan by diagnosis → you confirm → execute + log).
3. Catalog data structure + initial entries for **agricultural econometrics** (panel/cross-section/DID/fixed-effects/IV etc., limited set).
4. Recommender (rule matching + LLM ranking), output interpretable recommendation menu.
5. **One complete vertical cut running through**: province×year panel CSV → profiling → recommendation → select "two-way fixed-effects/DID" → call `empirical-analysis-python` to execute → output regression table + figures + brief report, **artifacts (code/figures/tables/reports) land in `outputs/`**.
6. Skill ingestion run once (ingest a skill → enter catalog → visible in recommendation).
7. **Novelty lens (lightweight version)**: At recommendation stage hang one literature scan on selected analysis, output "literature covered it or not, where blanks are" hints.
8. **Method rigor review (basic version)**: Recommended/selected analysis gets three-color prerequisite conclusion + main bias hints + **allow informed override** (report records warnings).

**Out of Scope (later phases)**:
- **Frontier monitor** (auto-loop continuous new paper tracking, leave for later);
- **GitHub collector** (auto-discover + read GitHub skill/code/methods, leave for later; architecture reserves entry point. Safety redline: don't auto-execute collected code);
- Web frontend and cloud deployment;
- Ecology/soil/microbiology raw data executors (need wrap vegan/lme4/agricolae/phyloseq etc.);
- **Machine learning / deep learning / time series executors** (RF·XGBoost, CNN·LSTM·Transformer, ARIMA·Prophet etc. — architecture reserved, see §8);
- **R / Stata executor backends** (agronomy field trials, ecology, microbiology mostly need R — architecture reserved, see §8);
- **Space / GIS and global macroeconomic analysis** (spatial econometrics, geostatistics, remote sensing raster — architecture reserved, see §8);
- Text/literature corpus profiling;
- Large comprehensive multi-domain catalog.

## 5. Technology Selection

- **Engine core**: Python package (`profiler` / `catalog` registry / `recommender`).
- **Execution**: pluggable multi-backend, unified interface. This phase Python (`empirical-analysis-python`); reserved R / Stata / deep learning frameworks (see §8).
- **Orchestration entry**: Claude Code master skill / CLI command, link ①②③ together.
- **LLM reasoning**: Via Claude (recommendation layer ranking and explanation).

## 6. Success Criteria (Verifiable)

1. Drop a province×year panel CSV → engine correctly identifies as **panel data** (identify unit, time, possible treatment variables).
2. List **≥3 viable analyses**, each stating "prerequisites met + expected output".
3. Select one → auto-produce **regression table + figures + brief report** (reuse empirical-analysis-python).
4. On **another computer (including macOS)** after `git clone` + `setup`, can reproduce above results.
5. Put a new skill into `skills_inbox/` → can be registered in catalog and visible in later recommendations.
6. Selected analysis gets at least **one "literature blank space" hint** (novelty lens runs once).
7. After uploading data can give **data quality diagnosis + one confirmable cleaning plan** and execute (include cleaning log).
8. For an **assumption-unmet** analysis, can give 🔴 warning + bias explanation, and **allow informed override** with report recording it.
9. After execution, **generated analysis code + images + tables + reports** all saved in `outputs/` (retained, reproducible, not just displayed).

## 7. Innovation: Frontier Awareness and Novelty Assessment

ResearchForge's innovation relies on three mechanisms, each plugged at different location:

1. **Cross-disciplinary method transfer (structural innovation)**: Data-type-driven routing naturally brings mature methods from one domain to another domain's data (e.g., causal ML/DID to agronomy, ecology data). This is architecture-native creative source, no extra building needed.
2. **Novelty lens (plugged to ② recommendation layer)**: When giving recommendation menu, do one literature scan on candidate angles, mark "existing literature covered or not, to what depth, where blanks are". So recommendation upgrades from "what's viable" to "what's viable and novel/publishable". Reuse existing skills: deep-research, paper-search-mcp, nature-academic-search, literature-review, research-ideation, k-dense-hypothesis. **This phase do lightweight version.**
3. **Frontier monitor (independent, optional)**: Every interval auto-search user's domain for new papers/methods (arXiv, journals, GitHub), filter relevance → deduplicate → rank, new methods auto-enter catalog, related discoveries auto-push alerts. **Feasibility confirmed** — search stack uses deep-research / paper-search-mcp / nature-academic-search / WebSearch; timer has two options: `/loop` (in-session periodic check) or **cron cloud scheduled agent (runs even when computer off, suits "auto daily/weekly patrol")**; `daily-paper-generator` already exemplifies same type. Not just list new papers, also do **innovation assessment** — combined bibliometric trends (what emerging, hotspot shift, blanks) judge "what's latest / worth doing direction", novel-lens agent outputs. **This phase don't do, leave for later.**
4. **GitHub collector (Skill / Code / Method Discovery)**: Auto-search GitHub for target domain's new skills, paper replication code, new method packages (rank by topic / stars / update time), read and register in catalog — is auto version of cross-cutting A "anytime add", also high-ROI frontier source (public repos directly fetchable / cloneable). **Safety boundary (redline): only auto "read + register"; executing collected code must pass manual review / sandbox, never auto-run strangers' code. This phase don't do, leave for later; cross-cutting A reserves entry interface.**

**Reliability boundary**: LLM + search novelty judgment may miss/misflag, positioned as "clues and initial screening for researchers", human does final review, not publication conclusion.

## 8. Architecture Reserved (Reserved Extension Points)

To support "skeleton first, gradually fill in", following all **reserve interfaces, not implement this phase**. Each added item = register in catalog / registry, **no three-layer structure change**.

**Executor backends (unified interface)**
- Python ✅ (this phase, empirical-analysis-python)
- R (reserved) — agronomy field trials (lme4 / nlme / agricolae), ecology community (vegan), microbiome (phyloseq / DESeq2)
- Stata (reserved) — reuse stata skill
- Deep learning frameworks (reserved) — PyTorch / TensorFlow

**Analysis families (enter catalog, routed by "research target")**
- Econometrics / statistical models / causal inference ✅ (this phase start, **quantitative**)
- Predictive machine learning (reserved) — random forest, XGBoost / LightGBM (tabular prediction/classification)
- Deep learning (reserved) — CNN, LSTM, Transformer (image/sequence/time series)
- Time series / forecasting (reserved) — ARIMA, ETS, Prophet, darts / sktime
- Space / GIS (reserved) — spatial econometrics, geostatistics (kriging), GWR, remote sensing raster, spatial autocorrelation; facing **global macroeconomic research** (country spatial panel, land use/climate raster)
- Ecology community (reserved) — ordination, PERMANOVA, α/β diversity
- Microbiome (reserved) — DADA2 / QIIME2, differential abundance
- **Qualitative / quality analysis (reserved)** — thematic analysis, content analysis, grounded theory, qualitative coding (interviews/text data); and **mixed methods** (quantitative + qualitative)
- **Systematic review / evidence synthesis** ✅ Already have skills (evidence-synthesis-forge, meta-analysis-forge, environment-life-review-forge, deep-research, umbrella-review-skeptic) — can register in catalog, not this phase's vertical cut
- **Bibliometrics (reserved)** — citation networks, co-citation, keyword co-occurrence, research trend map (bibliometrix / VOSviewer / CiteSpace / pybliometrics); for novelty lens and frontier monitor "trend / frontier" signals
- **Survey / psychological scale analysis (reserved)** — Likert data reliability (Cronbach α), factor analysis (EFA/CFA), ordinal / Likert regression, item analysis

**Domain processes / system models (reserved, mostly external software wrapping + unified interface)**
- Life cycle assessment LCA (openLCA / Brightway2), **emergy analysis** (systems ecology)
- Crop / farm models (DSSAT, APSIM, AquaCrop)
- Agricultural / regional economic models (CGE, partial equilibrium, GTAP, input-output)
- Environment / ecosystem models (SWAT watershed, InVEST ecosystem services, integrated assessment IAM)
- **System dynamics** (stock-flow simulation: PySD / BPTK-Py / Vensim)

**Research design / tool generation (reserved)**
- **Survey questionnaire / Likert scale generation** (item and scale design); experiment / sampling / power design (already have openclaw design skills)
- Generated design can accept **novelty lens** for innovation identification (see §7.2)

**Data modalities (enter Profiler)**
- Tabular CSV/Excel ✅ (this phase)
- Space / GIS (reserved) — coordinates / shapefile / raster / remote sensing
- Text / literature corpus (reserved)

**Public data sources / data acquisition (reserved, connectors; default prefer latest data)**
- Economics / social science: World Bank, IMF, FRED, OECD, Eurostat, openICPSR, Harvard Dataverse
- Agriculture: FAOSTAT, USDA NASS / QuickStats
- Environment / remote sensing: Google Earth Engine, Copernicus, NASA EarthData, SoilGrids
- Biology / species: NCBI / SRA, GBIF
- Local folder scan is **this phase capability**; remote connectors leave for later

**Knowledge sources / ingestion channels (cross-cutting A)**
- Manual skill ingestion ✅ (this phase)
- Paper absorption (reserved) — PDF/DOI → extract method/prerequisite/analysis into catalog
- GitHub collector (reserved) — see §7.4
- Frontier monitor periodic search (reserved) — see §7.3

> **Key design**: Routing looks not only at **data structure**, but also **research target** (explanation/causation vs prediction/forecasting vs classification). Same panel data, target "identify policy effect" → DID/fixed-effects; target "predict yield" → XGBoost/LSTM. After reserving these families, Recommender can give cross-paradigm menus per target.
>
> **Architecture robustness self-check**: To date, user-proposed R packages, ML/DL, time series, GIS/global macro, paper absorption, periodic search, qualitative/quality analysis, bibliometrics, systematic reviews, agri-bio-envir-econ cross, method rigor review, LCA/emergy, domain process models, system dynamics, survey questions/scales — all fit as "registration items" with **no three-layer structure change** — this is evidence this architecture holds.

## 9. Risks and Open Questions

- **R environment portability** harder than Python (ecology/soil/agronomy executors using R need handling) — not this phase, decide later.
- **Deep learning / GIS dependencies heavy** (PyTorch, GDAL / geospatial stack) — reserve as independent optional backend, not in MVP base install, avoid dragging down portability.
- **Skill packaging** volume/license: this phase use setup script to install rather than embed, sidestep.
- **LLM recommendation / novelty reliability**: rules gate hard prerequisites; recommendations and novelty hints both show reasoning, human final review.
- **Periodic search cost**: cloud scheduled agent running searches has token/compute cost, need configurable frequency and scope.
- **Cross-platform (including macOS)**: avoid OS-specific paths/dependencies; setup provides Windows / macOS / Linux versions; R / GIS / deep learning heavy dependencies install variance across platforms need testing.
- **Public data source connectors**: each source has different API/auth/quota, remote fetch reserved for later and wrap by source.

## 10. Related Documents

- Capabilities inventory and gaps: `docs/analysis-catalog/skills-inventory.md`
- Skill ingestion protocol: `skills_inbox/README.md`
