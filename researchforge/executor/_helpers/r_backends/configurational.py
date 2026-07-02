"""R-bridge configurational/QCA delegators (QCA/cna: csQCA, fsQCA, panel QCA, CNA, necessity)."""

from __future__ import annotations


def _csqca_via_r(csv_path, outcome: str, conditions: list[str], incl_cut=0.8):
    """Crisp-set QCA via R QCA: dichotomise (binary kept; continuous cut at the
    median) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x) {\n"
        "  nn <- x[!is.na(x)]; u <- unique(nn)\n"
        "  if (length(u) <= 2 && all(u %in% c(0,1))) return(as.integer(x))\n"
        "  as.integer(x > median(nn))\n"  # crisp threshold at the median
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), incl.cut={float(incl_cut)}, show.cases=FALSE)\n'
        "sol <- minimize(tt, details=TRUE)\n"
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##IC\\n"); ic <- sol$IC$incl.cov\n'
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclS[i], ic$covS[i], ic$covU[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##IC":
            section = "I"
        elif section == "S" and s:
            sol_str = s
        elif section == "I" and "|" in s:
            rows.append(s.split("|"))
    if not sol_str or not rows:
        raise RuntimeError("QCA 未返回充分配置（无解 / 有限多样性）")
    tab = pd.DataFrame(rows, columns=["configuration", "consistency", "raw_coverage", "unique_coverage"])
    for c in ("consistency", "raw_coverage", "unique_coverage"):
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return sol_str, tab


def _fsqca_via_r(csv_path, outcome: str, conditions: list[str], anchors=(0.1, 0.5, 0.9), incl_cut=0.8):
    """Run fsQCA with R's QCA package: direct fuzzy calibration (percentile
    anchors) -> truth table -> Boolean minimization. Returns (solution_str,
    incl.cov DataFrame). Raises so the caller can report an honest message."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        # skewed/discrete columns can tie the anchor quantiles; calibrate
        # needs strictly increasing anchors, so nudge ties by a tiny epsilon.
        "calib <- function(x) {\n"
        f"  thr <- as.numeric(quantile(x, c({a0},{a1},{a2}), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib))\n"
        "names(cal) <- cols\n"
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), '
        f"incl.cut={float(incl_cut)}, show.cases=FALSE)\n"
        "sol <- minimize(tt, details=TRUE)\n"
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##IC\\n"); ic <- sol$IC$incl.cov\n'
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclS[i], ic$covS[i], ic$covU[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##IC":
            section = "I"
        elif section == "S" and s:
            sol_str = s
        elif section == "I" and "|" in s:
            rows.append(s.split("|"))
    if not sol_str or not rows:
        raise RuntimeError("QCA 未返回充分配置（可能无解 / 有限多样性）")
    tab = pd.DataFrame(
        rows, columns=["configuration", "consistency", "raw_coverage", "unique_coverage"]
    )
    for c in ("consistency", "raw_coverage", "unique_coverage"):
        # R prints NA (e.g. unique coverage of a single-term solution) -> NaN
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return sol_str, tab


def _panel_qca_via_r(csv_path, outcome, conditions, unit, anchors, incl_cut):
    """Panel/clustered fsQCA via R SetMethods::cluster (Garcia-Castro & Ariño):
    fuzzy-calibrate, derive the pooled fsQCA solution, then decompose each
    solution term's consistency into POOLED vs BETWEEN-unit vs WITHIN-unit, with
    the between→pooled (dBP) and within→pooled (dWP) distances. A large distance
    means the configuration does NOT hold uniformly across units / over time
    (clustered heterogeneity). Returns (solution_str, terms_df). Raises on no
    result. Column names go through an identifier guard upstream."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages({library(SetMethods); library(QCA)})\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x){ thr<-as.numeric(quantile(x,c("
        f"{a0},{a1},{a2}),na.rm=TRUE)); rng<-diff(range(x,na.rm=TRUE)); "
        "eps<-if(rng>0) rng*1e-6 else 1e-6; if(thr[2]<=thr[1]) thr[2]<-thr[1]+eps; "
        'if(thr[3]<=thr[2]) thr[3]<-thr[2]+eps; calibrate(x,type="fuzzy",thresholds=thr) }\n'
        "cal <- as.data.frame(lapply(d[,cols], calib)); names(cal) <- cols\n"
        f'cal[["UNIT"]] <- as.character(d[["{unit}"]])\n'
        f'tt <- truthTable(cal, outcome="{outcome}", conditions=c({conds_r}), incl.cut={float(incl_cut)})\n'
        "sol <- minimize(tt, details=TRUE)\n"
        f'cl <- cluster(data=cal, results=sol, outcome="{outcome}", unit_id="UNIT", cluster_id="UNIT")\n'
        'cat("##SOL\\n"); cat(paste(sol$solution[[1]], collapse=" + "), "\\n")\n'
        'cat("##TERMS\\n")\n'
        "for (t in names(cl$output)) { o <- cl$output[[t]]; "
        'cat(sprintf("%s|%.4f|%.4f|%.4f|%.4f\\n", t, o$POCOS, o$Coverages$pooled, o$dBP, o$dWP)) }\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, sol_str, rows = None, "", []
    for line in out.splitlines():
        s = line.strip()
        if s == "##SOL":
            section = "S"
        elif s == "##TERMS":
            section = "T"
        elif section == "S" and s:
            sol_str = s
        elif "|" in s and section == "T":
            rows.append(s.rsplit("|", 4))
    if not rows:
        raise RuntimeError("SetMethods cluster 未返回分解（可能无解/单解项/有限多样性）")
    terms = pd.DataFrame(
        rows, columns=["term", "pooled_consistency", "pooled_coverage", "dist_between", "dist_within"]
    )
    for c in ("pooled_consistency", "pooled_coverage", "dist_between", "dist_within"):
        terms[c] = pd.to_numeric(terms[c], errors="coerce")
    return sol_str, terms


def _cna_via_r(csv_path, factors, outcome, con, cov, anchors, fuzzy):
    """Coincidence Analysis via R `cna` — finds configurational causal structures
    (Boolean solution formulas) and, unlike QCA, can recover structures with
    MULTIPLE outcomes (it does not require pre-designating one). Crisp (0/1) data
    is used directly (type="cs"); continuous factors are fuzzy-calibrated by
    percentile anchors (type="fs"). Returns (asf_df, n_csf). asf columns:
    outcome, condition, consistency, coverage, complexity. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    facs_r = ", ".join(f'"{c}"' for c in factors)
    out_arg = f', outcome=c("{outcome}")' if outcome else ""
    typ = "fs" if fuzzy else "cs"
    calib = (
        # fuzzy-calibrate non-binary columns to [0,1]; leave 0/1 columns as-is
        "calib <- function(x){ u<-unique(x[!is.na(x)]); "
        "if(length(u)<=2 && all(u %in% c(0,1))) return(x); "
        f"thr<-as.numeric(quantile(x,c({a0},{a1},{a2}),na.rm=TRUE)); "
        "rng<-diff(range(x,na.rm=TRUE)); eps<-if(rng>0) rng*1e-6 else 1e-6; "
        "if(thr[2]<=thr[1]) thr[2]<-thr[1]+eps; if(thr[3]<=thr[2]) thr[3]<-thr[2]+eps; "
        'calibrate(x,type="fuzzy",thresholds=thr) }\n'
        "d <- as.data.frame(lapply(d[,facs], calib)); names(d)<-facs\n"
        if fuzzy
        else "d <- d[, facs]\n"
    )
    rcode = (
        "suppressMessages(library(cna))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"facs <- c({facs_r})\n"
        + calib
        + f'x <- cna(d, type="{typ}", con={float(con)}, cov={float(cov)}{out_arg})\n'
        "a <- asf(x)\n"
        'cat("##ASF\\n")\n'
        "if (nrow(a)) for (i in seq_len(nrow(a))) cat(sprintf('%s|%s|%.4f|%.4f|%d\\n', "
        "a$outcome[i], a$condition[i], a$con[i], a$cov[i], a$complexity[i]))\n"
        'cat("##CSF\\n"); cat(sprintf("n|%d\\n", nrow(csf(x))))\n'
    )
    out = rbridge.run_r(rcode, timeout=180)
    section, rows, n_csf = None, [], 0
    for line in out.splitlines():
        s = line.strip()
        if s == "##ASF":
            section = "A"
        elif s == "##CSF":
            section = "C"
        elif "|" in s and section == "A":
            rows.append(s.rsplit("|", 4))
        elif s.startswith("n|") and section == "C":
            n_csf = int(s.split("|", 1)[1])
    if not rows:
        raise RuntimeError("cna 未返回解（可能 con/cov 阈值过高或无配置性结构）")
    asf = pd.DataFrame(rows, columns=["outcome", "condition", "consistency", "coverage", "complexity"])
    for c in ("consistency", "coverage", "complexity"):
        asf[c] = pd.to_numeric(asf[c], errors="coerce")
    return asf, n_csf


def _qca_necessity_via_r(csv_path, outcome: str, conditions: list[str], anchors=(0.1, 0.5, 0.9), incl_cut=0.9):
    """QCA necessity analysis via R superSubset on fuzzy-calibrated data. Returns
    a DataFrame [expression, inclN(consistency), RoN, covN(coverage)]. RoN flags
    trivially-necessary (always-high) conditions. Raises on no result."""
    import pandas as pd

    from researchforge.executor import rbridge

    a0, a1, a2 = (float(x) for x in anchors)
    csv_r = str(csv_path).replace("\\", "/")
    cols_r = ", ".join(f'"{c}"' for c in [outcome, *conditions])
    conds_r = ", ".join(f'"{c}"' for c in conditions)
    rcode = (
        "suppressMessages(library(QCA))\n"
        f'd <- read.csv("{csv_r}")\n'
        f"cols <- c({cols_r})\n"
        "calib <- function(x) {\n"
        f"  thr <- as.numeric(quantile(x, c({a0},{a1},{a2}), na.rm=TRUE))\n"
        "  rng <- diff(range(x, na.rm=TRUE)); eps <- if (rng>0) rng*1e-6 else 1e-6\n"
        "  if (thr[2] <= thr[1]) thr[2] <- thr[1] + eps\n"
        "  if (thr[3] <= thr[2]) thr[3] <- thr[2] + eps\n"
        '  calibrate(x, type="fuzzy", thresholds=thr)\n'
        "}\n"
        "cal <- as.data.frame(lapply(d[, cols], calib)); names(cal) <- cols\n"
        f'ss <- superSubset(cal, outcome="{outcome}", conditions=c({conds_r}), '
        f"incl.cut={float(incl_cut)}, cov.cut=0.5)\n"
        "ic <- ss$incl.cov\n"
        'for (i in seq_len(nrow(ic))) cat(sprintf("%s|%.4f|%.4f|%.4f\\n", '
        "rownames(ic)[i], ic$inclN[i], ic$RoN[i], ic$covN[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=180)
    rows = [s.split("|") for s in out.splitlines() if "|" in s and not s.strip().startswith("#")]
    if not rows:
        raise RuntimeError("superSubset 未返回必要条件（无满足一致性阈值的必要项）")
    tab = pd.DataFrame(rows, columns=["expression", "consistency_inclN", "RoN", "coverage_covN"])
    for c in ("consistency_inclN", "RoN", "coverage_covN"):
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return tab
