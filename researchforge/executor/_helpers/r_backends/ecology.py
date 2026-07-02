"""R-bridge differential-abundance delegator (ALDEx2)."""

from __future__ import annotations


def _diff_abundance_aldex2_via_r(csv_path, taxa: list[str], group: str):
    """Differential abundance via R ALDEx2 (compositional gold standard): CLR with
    Monte-Carlo Dirichlet sampling of the counts, Welch t per taxon over the MC
    instances, BH-FDR. ALDEx2 expects features (taxa) as ROWS, samples as COLUMNS.
    Returns a DataFrame [taxon, effect, diff_btw, p_value, q_value]. Raises so the
    caller can degrade honestly."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    taxa_r = ", ".join(f'"{t}"' for t in taxa)
    rcode = (
        "suppressMessages(library(ALDEx2))\n"
        f'd <- read.csv("{csv_r}", check.names=FALSE)\n'
        f"taxa <- c({taxa_r})\n"
        "counts <- t(as.matrix(d[, taxa]))\n"  # features x samples
        f'conds <- as.character(d[["{group}"]])\n'
        # 128 MC Dirichlet instances (ALDEx2 default); Welch t with effect sizes
        'x <- aldex(round(counts), conds, mc.samples=128, test="t", effect=TRUE, denom="all", verbose=FALSE)\n'
        'cat("##R\\n")\n'
        'for (i in seq_len(nrow(x))) cat(sprintf("%s|%.6f|%.6f|%.6g|%.6g\\n", '
        "rownames(x)[i], x$effect[i], x$diff.btw[i], x$we.ep[i], x$we.eBH[i]))\n"
    )
    out = rbridge.run_r(rcode, timeout=300)
    rows = []
    section = None
    for line in out.splitlines():
        s = line.strip()
        if s == "##R":
            section = "R"
        elif section == "R" and "|" in s:
            rows.append(s.split("|"))
    if not rows:
        raise RuntimeError("ALDEx2 未返回结果")
    res = pd.DataFrame(rows, columns=["taxon", "effect", "diff_btw", "p_value", "q_value"])
    for c in ("effect", "diff_btw", "p_value", "q_value"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


