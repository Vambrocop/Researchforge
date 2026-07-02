"""ergm — exponential random graph model (R statnet/ergm) + pure-Python CUG degrade."""

from __future__ import annotations

import re
from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.network_science._common import _build_graph, _resolve_edges


_ERGM_TERMS = {
    "edges", "mutual", "triangle", "triangles", "gwesp", "gwdegree", "gwdsp", "gwnsp",
    "kstar", "istar", "ostar", "twopath", "density", "isolates", "balance",
    "transitiveties", "cyclicalties", "degree", "idegree", "odegree",
}
_ERGM_TERM_RE = re.compile(r"^[a-zA-Z]+(\([0-9.,=\sA-Za-z]*\))?$")


def _ergm_terms_ok(terms: str) -> bool:
    """Validate a user-supplied ergm RHS like 'edges + gwesp(0.25, fixed=TRUE)' against
    the curated term allowlist + a strict token charset (no quotes/semicolons/backticks
    → no R injection through the formula)."""
    parts = [p.strip() for p in str(terms).split("+") if p.strip()]
    if not parts:
        return False
    for p in parts:
        if not _ERGM_TERM_RE.match(p):
            return False
        base = p.split("(", 1)[0]
        if base not in _ERGM_TERMS:
            return False
    return True


def _ergm_via_r(csv_path, directed, terms):
    """Fit an ERGM via R's statnet/ergm (MCMC-MLE). Returns (coef DataFrame[term,
    estimate,std_err,p_value], diag dict). Raises on failure so the caller degrades."""
    import pandas as pd

    from researchforge.executor import rbridge

    csv_r = str(csv_path).replace("\\", "/")
    dir_r = "TRUE" if directed else "FALSE"
    rcode = (
        "suppressMessages(library(ergm))\n"
        f'el <- read.csv("{csv_r}", colClasses="character")\n'
        f'net <- network(as.matrix(el[, c("src","tgt")]), matrix.type="edgelist", directed={dir_r})\n'
        f"m <- ergm(net ~ {terms}, control=control.ergm(seed=1, MCMC.samplesize=4096))\n"
        "s <- summary(m); ct <- s$coefficients\n"
        'pcol <- ncol(ct)\n'  # last column is Pr(>|z|) in ergm's coefficient table
        'cat("##COEF\\n")\n'
        'for (nm in rownames(ct)) cat(sprintf("%s|%.6f|%.6f|%.6g\\n", nm, ct[nm,1], ct[nm,2], ct[nm,pcol]))\n'
        'cat("##DIAG\\n")\n'
        'cat(sprintf("aic|%.6f\\nbic|%.6f\\n", AIC(m), BIC(m)))\n'
    )
    out = rbridge.run_r(rcode, timeout=300)
    section, crows, diag = None, [], {}
    for line in out.splitlines():
        s = line.strip()
        if s == "##COEF":
            section = "C"
        elif s == "##DIAG":
            section = "D"
        elif "|" in s and section == "C":
            crows.append(s.rsplit("|", 3))
        elif "|" in s and section == "D":
            k, v = s.split("|", 1)
            try:
                diag[k] = float(v)
            except ValueError:
                pass
    if not crows:
        raise RuntimeError("ergm 未返回系数")
    coef = pd.DataFrame(crows, columns=["term", "estimate", "std_err", "p_value"])
    for c in ("estimate", "std_err", "p_value"):
        coef[c] = pd.to_numeric(coef[c], errors="coerce")
    return coef, diag


def _cug_transitivity_test(UG, n_sim, seed):
    """Pure-Python conditional-uniform-graph (CUG) test: compare the observed global
    transitivity to its null distribution under random graphs with the SAME node count
    and edge count (G(n,m)). Returns (observed, null_mean, null_sd, z, p_one_sided)."""
    import networkx as nx
    import numpy as np

    n = UG.number_of_nodes()
    m = UG.number_of_edges()
    obs = nx.transitivity(UG)
    rng = np.random.RandomState(seed)
    null = np.empty(n_sim)
    for i in range(n_sim):
        Gr = nx.gnm_random_graph(n, m, seed=int(rng.randint(0, 2**31 - 1)))
        null[i] = nx.transitivity(Gr)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else float("nan")
    p = float((np.sum(null >= obs) + 1) / (n_sim + 1))  # one-sided (clustering above random)
    return float(obs), mu, sd, z, p


@register("ergm")
def _branch_ergm(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    from researchforge.executor import rbridge

    source, target, weight, directed, problem = _resolve_edges(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import networkx as nx

        G, UG = _build_graph(df, source, target, weight, directed)
        UG = nx.Graph(UG)
        UG.remove_edges_from(nx.selfloop_edges(UG))
        n = UG.number_of_nodes()
        m = UG.number_of_edges()

        terms = str(cfg.get("terms", "edges + gwesp(0.25, fixed=TRUE)"))
        if not _ergm_terms_ok(terms):
            terms = "edges + gwesp(0.25, fixed=TRUE)"

        # ── Full ERGM via R statnet/ergm (MCMC-MLE) when available ──────────────
        if rbridge.r_available() and rbridge.r_package_available("ergm"):
            import pandas as pd

            sub = df[[source, target]].dropna()
            sub = sub[sub[source] != sub[target]]
            sub.columns = ["src", "tgt"]
            csv = d / "_ergm_input.csv"
            sub.to_csv(csv, index=False)
            try:
                coef, diag = _ergm_via_r(csv, directed, terms)
                coef.to_csv(d / "ergm_coefficients.csv", index=False, encoding="utf-8")
                files.append("ergm_coefficients.csv")

                for _, r in coef.iterrows():
                    estimates[f"coef_{r['term']}"] = round(float(r["estimate"]), 4)
                    estimates[f"p_{r['term']}"] = round(float(r["p_value"]), 4)
                estimates["n_nodes"] = float(n)
                estimates["n_edges"] = float(m)
                if "aic" in diag:
                    estimates["aic"] = round(diag["aic"], 3)
                if "bic" in diag:
                    estimates["bic"] = round(diag["bic"], 3)

                lines = [f"{r['term']}: 估计={round(float(r['estimate']), 4)}（SE={round(float(r['std_err']), 4)}, "
                         f"p={round(float(r['p_value']), 4)}）" for _, r in coef.iterrows()]
                (d / "ergm_summary.txt").write_text(
                    f"指数随机图模型 ERGM（R statnet/ergm, MCMC-MLE）：边 {source}→{target}"
                    f"（{'有向' if directed else '无向'}）\n"
                    f"节点 {n}，边 {m}\n公式：net ~ {terms}\n"
                    + "\n".join(lines) + "\n"
                    + (f"AIC={round(diag.get('aic', float('nan')), 3)}, BIC={round(diag.get('bic', float('nan')), 3)}\n"
                       if "aic" in diag else "")
                    + "注：系数为对数几率尺度——edges≈基线密度（类似截距）；gwesp>0=超出随机的"
                    "三元闭合/传递性（聚集）；正的结构项表示该构型比随机更常见。ERGM 为 MCMC-MLE，"
                    "需检查退化(degeneracy)与 MCMC 收敛；这里用固定 seed 与默认控制参数。\n",
                    encoding="utf-8",
                )
                files.append("ergm_summary.txt")

                edges_row = coef[coef["term"] == "edges"]
                gw = coef[coef["term"].str.startswith("gwesp")]
                msg = f"{entry.method} 完成（R ergm, MCMC-MLE）：边 {source}→{target}；{n} 节点、{m} 边；公式 net ~ {terms}。"
                if not edges_row.empty:
                    msg += f" edges（基线密度）log-odds={round(float(edges_row['estimate'].iloc[0]), 3)}；"
                if not gw.empty:
                    gv = float(gw["estimate"].iloc[0]); gp = float(gw["p_value"].iloc[0])
                    msg += (f"gwesp（传递性/三元闭合）={round(gv, 3)}（p={round(gp, 3)}，"
                            f"{'显著高于随机的聚集' if (gv > 0 and gp < 0.05) else '不显著'}）；")
                if "aic" in diag:
                    msg += f"AIC={round(diag['aic'], 1)}。"
                msg += " ⚠ ERGM 为 MCMC-MLE，需检查模型退化与 MCMC 收敛（固定 seed/默认控制）；系数为对数几率。"
                summary.append(msg)
                code += [
                    "library(ergm)  # R statnet",
                    'net <- network(el, matrix.type="edgelist", directed=FALSE)',
                    f"m <- ergm(net ~ {terms})  # MCMC-MLE; edges=baseline density, gwesp=transitivity",
                    "summary(m)  # coefficients on the log-odds scale",
                ]
                return
            finally:
                try:
                    csv.unlink()
                except OSError:
                    pass

        # ── Degrade: pure-Python CUG test on transitivity ──────────────────────
        import numpy as np
        import pandas as pd

        if n < 5 or m < 3:
            summary.append("ERGM 失败：网络过小，无法做 ERGM 或 CUG 检验。")
            return
        n_sim = int(cfg.get("n_sim", 300))
        seed = int(cfg.get("seed", 0))
        obs, mu, sd, z, p = _cug_transitivity_test(UG, n_sim, seed)
        density = nx.density(UG)

        cug = pd.DataFrame([{"statistic": "transitivity", "observed": round(obs, 5),
                             "null_mean": round(mu, 5), "null_sd": round(sd, 5),
                             "z": round(z, 4) if z == z else float("nan"),
                             "p_one_sided": round(p, 5)}])
        cug.to_csv(d / "ergm_cug_test.csv", index=False, encoding="utf-8")
        files.append("ergm_cug_test.csv")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rng = np.random.RandomState(seed)
            null = np.array([nx.transitivity(nx.gnm_random_graph(n, m, seed=int(rng.randint(0, 2**31 - 1))))
                             for _ in range(n_sim)])
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.hist(null, bins=30, color="#B0B0B0", alpha=0.85, label="null G(n,m)")
            ax.axvline(obs, color="#C44E52", lw=2, label=f"observed ({obs:.3f})")
            ax.set_xlabel("global transitivity")
            ax.set_ylabel("count")
            ax.set_title(f"CUG test: transitivity vs random graphs (n={n}, m={m}, {n_sim} sims)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "ergm_cug_test.png", dpi=150)
            plt.close(fig)
            files.append("ergm_cug_test.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(n)
        estimates["n_edges"] = float(m)
        estimates["density"] = round(float(density), 5)
        estimates["transitivity_observed"] = round(obs, 5)
        estimates["transitivity_null_mean"] = round(mu, 5)
        estimates["transitivity_z"] = round(z, 4) if z == z else float("nan")
        estimates["transitivity_p"] = round(p, 5)

        verdict = ("聚集显著高于随机（存在三元闭合/传递性）" if (p < 0.05 and obs > mu)
                   else "聚集与随机图无显著差异")
        (d / "ergm_summary.txt").write_text(
            f"指数随机图模型 ERGM —— 降级为 CUG 条件均匀图检验（未检测到 R 的 ergm 包）：边 {source}→{target}\n"
            f"节点 {n}，边 {m}，密度={round(float(density), 5)}\n"
            f"观测全局传递性={round(obs, 5)}；同节点同边数随机图 G(n,m) 的零分布："
            f"均值={round(mu, 5)}、SD={round(sd, 5)}（{n_sim} 次模拟）\n"
            f"z={round(z, 4) if z == z else 'NaN'}，单侧 p={round(p, 5)} —— {verdict}\n"
            "注：完整 ERGM（系数/SE/退化诊断）需要 R 的 statnet/ergm（install.packages('ergm')）；"
            "这里给出纯 Python 的 CUG 检验作为替代——它检验「观测聚集是否超出仅由规模与密度决定的随机水平」，"
            "正是 ergm 的 gwesp 项所刻画的传递性效应的单变量版本（不控制其他结构项、无系数尺度）。\n",
            encoding="utf-8",
        )
        files.append("ergm_summary.txt")

        summary.append(
            f"{entry.method}（降级为 CUG 检验，未检测到 R 的 ergm 包）：边 {source}→{target}；"
            f"{n} 节点、{m} 边、密度 {round(float(density), 4)}；观测传递性 {round(obs, 4)} vs "
            f"随机零分布均值 {round(mu, 4)}（z={round(z, 3) if z == z else 'NaN'}, p={round(p, 4)}）—— {verdict}。"
            "⚠ 完整 ERGM（系数/退化诊断）需 R 的 statnet/ergm（install.packages('ergm')）；"
            "CUG 是单变量替代（仅检验聚集是否超随机，不控制其他结构项、无系数）。"
        )
        code += [
            "import networkx as nx, numpy as np",
            "# Full ERGM needs R statnet/ergm; pure-Python degrade = CUG test:",
            "obs = nx.transitivity(G)  # vs null distribution of G(n,m) random graphs",
            "# p = P(null transitivity >= observed); tests clustering beyond random density",
        ]
    except Exception as err:
        summary.append(f"ERGM/CUG 失败：{err}")
