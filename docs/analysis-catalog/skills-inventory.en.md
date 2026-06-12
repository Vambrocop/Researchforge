# ResearchForge Capabilities Inventory v0.1 (Collecting phase)

> A single sweep of "what we already have", mapped to the three-layer architecture, marking gaps. Avoid re-doing this each time.
> Three-layer architecture: ① Data profiling → ② Recommendation (hybrid route C) → ③ Execution + figure/report generation

Role markers: **[Executor]** can run code and produce results ｜ **[Planning/synthesis]** gives methods or does literature surveys ｜ **[Writing/visualization]**

## One: Existing Skills × (Domain / Role)

### Economics / Agricultural Economics — Strong coverage ✅
- **empirical-analysis-python** [Executor·Core]: Complete 8-step econometric pipeline (cleaning→construction→description→diagnosis→baseline→robustness→mechanism/heterogeneity→tables/figures), AER/QJE-style multi-column regression tables + event studies/coefficient plots. Also contains **Mode A epidemiology** (IPTW/TMLE/MR/survival), **Mode B causal machine learning** (DML/causal forests/meta-learners/policy trees).
- **empirical-analysis-R** [Executor], **stata** [Executor/reference]
- **statspai-skill**, **the-effect-book**, **results-analysis**
- buffett and a batch of investment/valuation "thinking framework" skills — qualitative, not data executors

### General Statistics — Strong coverage ✅
- **openclaw-statistical-analysis** [Executor]: t-test/ANOVA/regression/Bayesian/power analysis/assumption checking/APA reporting
- **openclaw-data-stats-analysis** [Executor]
- **openclaw-bio-experimental-design-power-analysis / -sample-size** (experimental design/sample size)

### Soil Science / Agronomy (Field Trials) — Partial coverage ⚠️
- Have: openclaw-statistical-analysis can do basic ANOVA/regression/mixed models; environment-life-review-forge manages literature synthesis
- **Missing**: dedicated field trial executor (RCBD/split-plot/mixed effects lme4·nlme, agricolae style), soil profile/spatial geostatistics (kriging)

### Ecology — Raw data analysis gap obvious ❌
- Have: **environment-life-review-forge** [Planning/synthesis] (PECO/PICO, heterogeneity, risk bias), meta-analysis-forge
- **Missing**: community ecology executor (ordination NMDS/PCA, PERMANOVA, α/β diversity, vegan style), species distribution models

### Microbiology — Gap ❌
- Have: openclaw-bio-causal-genomics-mediation-analysis (genomic mediation), experimental design skill
- **Missing**: amplicon/microbiome pipeline (DADA2/QIIME2, α/β diversity, differential abundance DESeq2/ANCOM)

### Literature / Evidence Synthesis — Strong coverage ✅
- deep-research (13-agent), academic-pipeline, evidence-synthesis-forge, meta-analysis-forge, environment-life-review-forge, literature-review, umbrella-review-skeptic, meta-ml-screener, nature-academic-search, paper-search-mcp, daily-paper-generator

### Writing / Visualization / Video — Strong coverage ✅
- academic-paper (12-agent), academic-paper-reviewer, ml-paper-writing, nature-writing/polishing/figure/citation/response/reader, academic-plotting, humanizer-academic, avoid-ai-writing, chinese-de-aigc, paper-slide-deck, k-dense-*

## Two: Mapping to Three Layers → What to Build

| Layer | Current State | Conclusion |
|-------|------|------|
| ① Data profiling | No off-the-shelf skill | **Need to build new** (lightweight Python: structure detection + variable types + panel/treatment group identification) |
| ② Recommendation (hybrid C) | None available | **Core to build**: analysis catalog knowledge base + LLM reasoning for ranking |
| ③ Execution | Economics/statistics/synthesis/writing reusable | Ecology/soil/microbiology **raw data executors are main gap** (wrap R packages or build new) |

## Three: Gap Summary (To-build List)

1. **Data profiling engine** (wholly new, lightweight)
2. **Analysis catalog + hybrid recommendation engine** (wholly new, project heart)
3. **Ecology/soil/microbiology raw data executors** (wrap vegan/lme4/nlme/agricolae/phyloseq/DADA2, or build new)
4. **Orchestration skeleton** (link ①②③ together)

## Four: External Academic Skills to Assess

- **Academic Research Skills suite** (v3.7, 2026; research→write→review→revise→finalize, plugin marketplace 30-second install) — high overlap with our existing academic-pipeline/academic-paper, can compare/trade-off
- **awesome-econ-ai-stuff** (economics AI tools checklist: Stata cleaning→LaTeX→DID fixed-effects + coefficient plots)
- **claudeskills.info `/category/research`** (research skills directory)
- GitHub `imbad0202/academic-research-skills`

## Five: Paper Replication Code — Where to Find

- **Economics/social science**: AEA Data & Code Repository @ openICPSR (AER/AEJ mandatory submission, replication package goldmine), Harvard Dataverse (includes REStat), OSF, Zenodo, ICPSR
- **Ecology/agronomy/environment**: Dryad (auto-syncs to Zenodo), Zenodo (gives DOI), figshare, KNB, DataONE, rOpenSci (R ecosystem)
- **Microbiology/bioinformatics**: Zenodo, NCBI/SRA (data), Bioconductor (packages), QIIME2 forums
- **General**: Papers with Code, Code Ocean, OSF

## Six: Known Constraints

- **Portability**: users may change computers → entire project (engine + catalog + required skills) should fit in **Git repository**; machine change = `git clone`. Note: existing skills live in `~/.claude/skills/` (machine-dependent); need to bundle in repo or use install script, otherwise after move engine can't find executors.
