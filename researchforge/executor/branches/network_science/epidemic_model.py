"""epidemic_model — network-based stochastic SIR/SIS diffusion simulation."""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register
from researchforge.executor.branches.network_science._common import _build_graph, _resolve_edges


def _simulate_epidemic(UG, model, beta, gamma, initial_infected, steps, n_runs, seed):
    """Discrete-time stochastic SIR/SIS on the network UG, averaged over n_runs.

    Each step (synchronous update): every S node becomes I with prob
    1-(1-beta)^(#infected neighbours); every I node recovers with prob gamma
    (SIR -> R, immune; SIS -> S, susceptible again).
    Returns (mean_curve dict t->(S,I,R), peak_I, time_to_peak, attack_rate)."""
    import random

    nodes = list(UG.nodes())
    N = len(nodes)
    neighbors = {u: list(UG.neighbors(u)) for u in nodes}

    runs_curves = []
    attack_rates = []
    for run in range(n_runs):
        rng = random.Random(seed + run)
        seeds = rng.sample(nodes, min(initial_infected, N))
        state = {u: "S" for u in nodes}
        for s in seeds:
            state[s] = "I"
        ever_infected = set(seeds)

        curve = []  # list of (S, I, R)
        for _t in range(steps + 1):
            S = sum(1 for u in nodes if state[u] == "S")
            I = sum(1 for u in nodes if state[u] == "I")
            R = N - S - I
            curve.append((S, I, R))
            if I == 0:
                # epidemic died out: pad remaining steps with the steady state
                last = (S, I, R)
                while len(curve) < steps + 1:
                    curve.append(last)
                break
            # compute new states (synchronous update)
            new_state = dict(state)
            for u in nodes:
                if state[u] == "I":
                    if rng.random() < gamma:
                        new_state[u] = "R" if model == "sir" else "S"
                elif state[u] == "S":
                    inf_nb = sum(1 for v in neighbors[u] if state[v] == "I")
                    if inf_nb:
                        p = 1.0 - (1.0 - beta) ** inf_nb
                        if rng.random() < p:
                            new_state[u] = "I"
                            ever_infected.add(u)
            state = new_state
        runs_curves.append(curve)
        attack_rates.append(len(ever_infected) / N)

    # average the curves across runs (all padded to steps+1)
    T = steps + 1
    mean_curve = {}
    for t in range(T):
        s = sum(c[t][0] for c in runs_curves) / n_runs
        i = sum(c[t][1] for c in runs_curves) / n_runs
        r = sum(c[t][2] for c in runs_curves) / n_runs
        mean_curve[t] = (s, i, r)

    peak_I = max(mean_curve[t][1] for t in range(T))
    time_to_peak = max(range(T), key=lambda t: mean_curve[t][1])
    attack_rate = sum(attack_rates) / n_runs
    return mean_curve, peak_I, time_to_peak, attack_rate


@register("epidemic_model")
def _branch_epidemic_model(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code

    source, target, weight, directed, problem = _resolve_edges(ctx)
    if problem is not None:
        summary.append(problem)
        return
    try:
        import numpy as np
        import pandas as pd

        # spread runs on the undirected contact graph
        _, UG = _build_graph(df, source, target, weight, directed)
        N = UG.number_of_nodes()

        model = str(cfg.get("model", "sir")).lower()
        if model not in {"sir", "sis"}:
            model = "sir"
        beta = float(cfg.get("beta", 0.05))
        gamma = float(cfg.get("gamma", 0.1))
        initial_infected = max(1, int(cfg.get("initial_infected", 1)))
        steps = int(cfg.get("steps", 60))
        n_runs = max(1, int(cfg.get("n_runs", 10)))
        seed = int(cfg.get("seed", 0))

        mean_curve, peak_I, time_to_peak, attack_rate = _simulate_epidemic(
            UG, model, beta, gamma, initial_infected, steps, n_runs, seed
        )

        T = steps + 1
        curve_df = pd.DataFrame({
            "t": list(range(T)),
            "S": [round(mean_curve[t][0], 3) for t in range(T)],
            "I": [round(mean_curve[t][1], 3) for t in range(T)],
            "R": [round(mean_curve[t][2], 3) for t in range(T)],
        })
        curve_df.to_csv(d / "epidemic_curve.csv", index=False, encoding="utf-8")
        files.append("epidemic_curve.csv")

        # R0 proxy = beta * <k> / gamma  (rough; true threshold ~ <k^2>/<k>)
        degs = [dg for _, dg in UG.degree()]
        mean_k = float(np.mean(degs)) if degs else 0.0
        r0_proxy = (beta * mean_k / gamma) if gamma > 0 else float("inf")

        # epidemic curve plot
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(curve_df["t"], curve_df["S"], label="S (susceptible)", color="#4C72B0")
            ax.plot(curve_df["t"], curve_df["I"], label="I (infected)", color="#C44E52")
            if model == "sir":
                ax.plot(curve_df["t"], curve_df["R"], label="R (recovered)", color="#55A868")
            ax.axvline(time_to_peak, ls="--", lw=1, color="grey", alpha=0.7)
            ax.set_xlabel("time step")
            ax.set_ylabel("number of nodes")
            ax.set_title(f"Network {model.upper()} epidemic curve (mean of {n_runs} runs)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "epidemic_curve.png", dpi=150)
            plt.close(fig)
            files.append("epidemic_curve.png")
        except Exception:
            pass

        estimates["n_nodes"] = float(N)
        estimates["mean_degree"] = round(mean_k, 3)
        estimates["beta"] = round(beta, 4)
        estimates["gamma"] = round(gamma, 4)
        estimates["peak_infected"] = round(float(peak_I), 3)
        estimates["peak_infected_frac"] = round(float(peak_I) / N, 4) if N else 0.0
        estimates["time_to_peak"] = float(time_to_peak)
        estimates["attack_rate"] = round(float(attack_rate), 4)
        estimates["r0_proxy"] = round(float(r0_proxy), 4) if r0_proxy != float("inf") else float("inf")

        spread_note = "可能扩散（R0 代理 >1）" if r0_proxy > 1 else "趋于熄灭（R0 代理 ≤1）"
        (d / "epidemic_summary.txt").write_text(
            f"网络传播模拟（{model.upper()}，基于接触网络的离散时间随机过程）：边 {source}→{target}\n"
            f"节点 {N}，平均度 <k>={round(mean_k, 3)}\n"
            f"参数：beta（每接触每步传播概率）={beta}，gamma（每步康复概率）={gamma}，"
            f"初始感染 {initial_infected} 个随机种子（已固定 seed={seed}），"
            f"模拟 {steps} 步、{n_runs} 次取平均\n"
            f"峰值感染 {round(float(peak_I), 2)}（占 {round(float(peak_I) / N * 100, 1) if N else 0}%），"
            f"达峰时间 t={time_to_peak}\n"
            f"最终攻击率（曾被感染比例）={round(float(attack_rate), 4)}\n"
            f"R0 代理 = beta·<k>/gamma = {round(float(r0_proxy), 3) if r0_proxy != float('inf') else 'inf'} —— {spread_note}\n"
            "注：网络 SIR/SIS 取决于接触结构（度的异质性驱动传播——hub 加速扩散）；"
            "beta/gamma 是假定参数（已报告）；过程是随机的（多次取平均、seed 已固定）；"
            "R0 代理是粗略值，真实流行阈值取决于度分布方差 <k^2>/<k>。\n\n"
            "流行曲线（前 20 步）：\n" + curve_df.head(20).to_string(index=False),
            encoding="utf-8",
        )
        files.append("epidemic_summary.txt")

        summary.append(
            f"{entry.method} 完成（网络 {model.upper()}）：边 {source}→{target}；{N} 节点、平均度 {round(mean_k, 3)}；"
            f"beta={beta}, gamma={gamma}；峰值感染 {round(float(peak_I), 1)}"
            f"（占 {round(float(peak_I) / N * 100, 1) if N else 0}%）于 t={time_to_peak}；"
            f"最终攻击率 {round(float(attack_rate), 4)}；R0 代理 "
            f"{round(float(r0_proxy), 3) if r0_proxy != float('inf') else 'inf'}（{spread_note}）。"
            "⚠ 依赖接触结构（hub 加速扩散）；beta/gamma 为假定（已报告）；随机（多次取平均、seed 固定）；R0 代理粗略。"
        )
        code += [
            "import networkx as nx, random",
            f"G = nx.from_pandas_edgelist(df, {source!r}, {target!r})",
            f"# discrete-time stochastic {model.upper()}: S->I w.p. 1-(1-beta)^(#inf nb), I->{'R' if model == 'sir' else 'S'} w.p. gamma",
            f"# beta={beta}, gamma={gamma}, {n_runs} runs averaged (seed fixed); R0 proxy = beta*<k>/gamma",
        ]
    except Exception as err:
        summary.append(f"传播模拟失败：{err}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. link_prediction — neighbourhood similarity scores + held-out-edge AUC + top-K
# ─────────────────────────────────────────────────────────────────────────────
