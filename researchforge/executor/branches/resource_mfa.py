"""Branch handler for MATERIAL FLOW ANALYSIS (MFA) — resource family / sustainability.

One method — ``material_flow_analysis`` — the PHYSICAL mass-balance accounting
piece of an Energy-Water-Food (EWF) nexus toolkit. It complements the monetary
Leontief input-output analysis (``resource_io.py``) and cost / cost-effectiveness
methods (other agents' modules) with the *economy-wide / facility material balance*
principle from industrial ecology:

    INPUTS  =  OUTPUTS  +  NET STOCK ADDITION

i.e. everything that physically enters a system either leaves it (as product,
emission, waste, export, …) or accumulates as a net addition to in-use stock. The
balancing residual NAS = inputs − outputs is, for a *closed* mass balance, the
genuine physical stock build-up; for an *open* accounting it lumps stock change
together with any unaccounted flow / measurement error (both interpretations are
disclosed — they cannot be distinguished from the totals alone).

(SEPARATE agents own ``resource.py`` / ``resource_io.py`` / ``resource_econ.py`` and
their yamls; all coexist in the "resource" family via catalog auto-discovery — this
module only owns ``material_flow_analysis`` and never touches the others.)

WHAT IT DOES
------------
Given numeric flow columns, each classified as an INPUT flow or an OUTPUT flow, it
computes the standard MFA totals and indicators over the whole dataset:

  * Total inputs   TI = Σ (all input flows, summed over every cell)
  * Total outputs  TO = Σ (all output flows, summed over every cell)
  * Net stock addition / balancing item   NAS = TI − TO
        (the mass-balance residual: physical accumulation in stock, OR
         unaccounted flow / measurement error — disclosed; ≈0 or = genuine stock
         build-up for a true, consistent-unit balance)
  * Throughput     = TI   (the material throughput of the system)
  * Balance ratio  = TO / TI   (how much of the input mass leaves as output;
        1.0 ⇔ perfectly balanced with zero net stock change)
  * Resource / eco efficiency = useful product output / TI   (only if a single
        output flow is flagged as the useful product via config ``product_flow``;
        else reported as overall TO / TI and flagged as the generic output ratio)
  * Recycling rate = recycled input / TI   (only if an input flow is flagged via
        config ``recycled_flow``)
  * Per-flow shares: each input flow as a % of TI, each output flow as a % of TO.

CLASSIFICATION RULE
-------------------
1. If config ``input_flows`` / ``output_flows`` (lists of column names) are given,
   they win (only their columns that are numeric & present are used).
2. Otherwise classify each numeric flow column by NAME (case-insensitive substring):
     inputs  : input | extraction | import | intake | supply | consumption |
               withdrawal | use
     outputs : output | emission | waste | export | discharge | disposal |
               loss | product
   A column matching BOTH lists (e.g. "input_waste") is resolved by config first,
   else it falls to whichever keyword bucket; ties (matches both name buckets) are
   reported as ambiguous and treated as INPUT with disclosure. A column matching
   NEITHER is reported "unclassified" and EXCLUDED from the balance (disclosed).
The period/entity label column (config ``unit`` or fp.unit_col / fp.time_col) is
never treated as a flow.

NaN HANDLING (NOTE — STOP-AND-REPORT choice)
--------------------------------------------
Flow tables are typically SPARSE (not every period/entity reports every flow), so a
blank flow cell means "no flow", not "missing". We therefore **treat NaN flow cells
as 0** for summation (the standard convention for sparse physical-flow accounts),
rather than dropping rows. This is documented in the Chinese summary. A column that
is *entirely* NaN / non-numeric contributes 0 and is reported.

Pure Python (numpy / pandas / matplotlib Agg, ENGLISH plot labels). No R, no heavy
deps. Honest degrade ("跳过") when we cannot identify ≥1 input flow AND ≥1 output
flow — never crash / fabricate.

Refs: Eurostat "Economy-wide Material Flow Accounts (EW-MFA) Handbook"; Brunner &
Rechberger "Practical Handbook of Material Flow Analysis"; OECD material-flow
indicators. ⚠ A mass balance is only meaningful if every flow is in the SAME mass
unit (e.g. tonnes); MFA requires consistent units across all flows.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register

# Name-based classification keywords (case-insensitive substring match).
_INPUT_KEYWORDS = (
    "input", "extraction", "import", "intake", "supply",
    "consumption", "withdrawal", "use",
)
_OUTPUT_KEYWORDS = (
    "output", "emission", "waste", "export", "discharge",
    "disposal", "loss", "product",
)


def _save_fig(d, fname, files, build) -> None:
    """best-effort matplotlib figure (Agg). build(plt) draws on the current figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        build(plt)
        plt.tight_layout()
        plt.savefig(d / fname, dpi=150)
        plt.close("all")
        files.append(fname)
    except Exception:
        pass


def _numeric_flow_cols(ctx: Ctx) -> list[str]:
    """Candidate flow columns: numeric kinds, excluding the unit / time label cols.

    Accepts continuous, count AND id kinds — an integer flow with all-distinct
    values is misclassified ``id`` by the profiler (CLAUDE.md「id 陷阱」); a small
    integer stream profiles as ``count``. The period/entity label column (config
    ``unit`` or fp.unit_col) and fp.time_col are excluded.
    """
    fp, cfg = ctx.fp, ctx.cfg
    excl = {fp.unit_col, fp.time_col}
    unit_cfg = cfg.get("unit")
    if isinstance(unit_cfg, str):
        excl.add(unit_cfg)
    return [
        c.name
        for c in fp.columns
        if c.name not in excl and c.kind in ("continuous", "count", "id")
    ]


def _classify_flows(ctx: Ctx, candidates: list[str]):
    """Return (inputs, outputs, unclassified, ambiguous, note).

    config ``input_flows`` / ``output_flows`` (lists of names) win; else classify by
    NAME keyword. A name matching BOTH buckets is 'ambiguous' -> treated as INPUT
    (disclosed). A name matching NEITHER is 'unclassified' -> excluded.
    """
    cfg = ctx.cfg
    cand_set = set(candidates)

    def _as_name_list(val):
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        try:
            return [str(v) for v in val]
        except TypeError:
            return [str(val)]

    cfg_in = [c for c in _as_name_list(cfg.get("input_flows")) if c in cand_set]
    cfg_out = [c for c in _as_name_list(cfg.get("output_flows")) if c in cand_set]

    inputs: list[str] = []
    outputs: list[str] = []
    unclassified: list[str] = []
    ambiguous: list[str] = []

    if cfg_in or cfg_out:
        # config-driven; columns named in BOTH lists go to input (input wins).
        in_set, out_set = set(cfg_in), set(cfg_out)
        for c in candidates:
            if c in in_set:
                inputs.append(c)
            elif c in out_set:
                outputs.append(c)
            else:
                unclassified.append(c)
        note = (
            f"分类来源＝config：input_flows={inputs}、output_flows={outputs}"
            + (f"；未分类（排除）={unclassified}" if unclassified else "")
        )
        return inputs, outputs, unclassified, ambiguous, note

    # name-based heuristic classification
    for c in candidates:
        cl = str(c).strip().lower()
        is_in = any(k in cl for k in _INPUT_KEYWORDS)
        is_out = any(k in cl for k in _OUTPUT_KEYWORDS)
        if is_in and is_out:
            ambiguous.append(c)
            inputs.append(c)  # tie -> treat as INPUT (disclosed)
        elif is_in:
            inputs.append(c)
        elif is_out:
            outputs.append(c)
        else:
            unclassified.append(c)
    note = (
        f"分类来源＝按列名启发式：input_flows={inputs}、output_flows={outputs}"
        + (f"；⚠ 名称同时命中进出口关键词、按 INPUT 处理={ambiguous}" if ambiguous else "")
        + (f"；未分类（排除）={unclassified}" if unclassified else "")
    )
    return inputs, outputs, unclassified, ambiguous, note


# ===========================================================================
# material_flow_analysis — economy-wide / facility mass-balance accounting
# ===========================================================================
@register("material_flow_analysis")
def _branch_material_flow_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import numpy as np
    import pandas as pd

    candidates = _numeric_flow_cols(ctx)
    if not candidates:
        summary.append(
            "material_flow_analysis 跳过：未找到任何数值型流量列"
            "（用 config input_flows / output_flows 指定）"
        )
        return

    inputs, outputs, unclassified, ambiguous, cls_note = _classify_flows(ctx, candidates)
    if not inputs or not outputs:
        summary.append(
            "material_flow_analysis 跳过：未能同时识别出 ≥1 个输入流与 ≥1 个输出流"
            f"（候选数值列={candidates}；识别到 inputs={inputs}、outputs={outputs}）"
            "（用 config input_flows / output_flows 指定）"
        )
        return

    try:
        # --- column sums, treating NaN flow cells as 0 (sparse-table convention) ---
        def _col_sum(col: str) -> float:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            return float(np.nansum(s.to_numpy(dtype=float)))

        in_sums = {c: _col_sum(c) for c in inputs}
        out_sums = {c: _col_sum(c) for c in outputs}

        total_input = float(sum(in_sums.values()))
        total_output = float(sum(out_sums.values()))
        net_stock_addition = float(total_input - total_output)
        throughput = float(total_input)
        balance_ratio = float(total_output / total_input) if total_input != 0 else float("nan")

        # --- resource / eco efficiency: useful product output / total input ---
        product_flow = cfg.get("product_flow")
        resource_efficiency = float("nan")
        eff_note = ""
        if isinstance(product_flow, str) and product_flow in out_sums:
            resource_efficiency = (
                float(out_sums[product_flow] / total_input) if total_input != 0 else float("nan")
            )
            eff_note = (
                f"资源/生态效率＝有用产品产出「{product_flow}」/ 总输入 = {resource_efficiency:.6g}。"
            )
        else:
            if isinstance(product_flow, str) and product_flow not in out_sums:
                eff_note = (
                    f"⚠ config product_flow='{product_flow}' 不在输出流中（{list(out_sums)}），已忽略；"
                )
            eff_note += (
                f"未指定有用产品(product_flow)，资源效率以总输出/总输入比代报 = "
                f"{balance_ratio:.6g}（非真正的有用产出效率）。"
            )

        # --- recycling rate: recycled input / total input ---
        recycled_flow = cfg.get("recycled_flow")
        recycling_rate = float("nan")
        rec_note = ""
        if isinstance(recycled_flow, str) and recycled_flow in in_sums:
            recycling_rate = (
                float(in_sums[recycled_flow] / total_input) if total_input != 0 else float("nan")
            )
            rec_note = (
                f"再生(循环)率＝再生输入「{recycled_flow}」/ 总输入 = {recycling_rate:.6g}。"
            )
        elif isinstance(recycled_flow, str) and recycled_flow not in in_sums:
            rec_note = (
                f"⚠ config recycled_flow='{recycled_flow}' 不在输入流中（{list(in_sums)}），已忽略，再生率=NaN。"
            )

        # --- per-flow shares + flow table (CSV) ---
        rows = []
        for c in inputs:
            share = (in_sums[c] / total_input * 100.0) if total_input != 0 else float("nan")
            rows.append({"flow": c, "type": "input", "value": round(in_sums[c], 6),
                         "share_pct": round(share, 4)})
        for c in outputs:
            share = (out_sums[c] / total_output * 100.0) if total_output != 0 else float("nan")
            rows.append({"flow": c, "type": "output", "value": round(out_sums[c], 6),
                         "share_pct": round(share, 4)})
        for c in unclassified:
            rows.append({"flow": c, "type": "unclassified", "value": round(_col_sum(c), 6),
                         "share_pct": float("nan")})
        flow_tbl = pd.DataFrame(rows, columns=["flow", "type", "value", "share_pct"])
        flow_tbl.to_csv(d / "mfa_flows.csv", index=False, encoding="utf-8")
        files.append("mfa_flows.csv")

        # --- estimates (ALL plain floats; NaN for N/A) ---
        estimates.update({
            "total_input": round(total_input, 6),
            "total_output": round(total_output, 6),
            "net_stock_addition": round(net_stock_addition, 6),
            "balance_ratio": round(balance_ratio, 6) if np.isfinite(balance_ratio) else float("nan"),
            "throughput": round(throughput, 6),
            "resource_efficiency": (round(resource_efficiency, 6)
                                    if np.isfinite(resource_efficiency) else float("nan")),
            "recycling_rate": (round(recycling_rate, 6)
                               if np.isfinite(recycling_rate) else float("nan")),
            "n_input_flows": float(len(inputs)),
            "n_output_flows": float(len(outputs)),
        })

        # --- flow-balance bar plot: inputs vs outputs vs NAS (ENGLISH labels) ---
        def _plot(plt):
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            labels = ["Total inputs", "Total outputs", "Net stock addition"]
            vals = [total_input, total_output, net_stock_addition]
            colors = ["#4C72B0", "#C44E52", "#55A868" if net_stock_addition >= 0 else "#DD8452"]
            ax.bar(labels, vals, color=colors)
            ax.axhline(0.0, color="#888888", lw=0.9)
            ax.set_ylabel("mass flow (consistent unit, e.g. tonnes)")
            ax.set_title("Material flow balance: inputs = outputs + net stock addition")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.4g}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)

        _save_fig(d, "mfa_balance.png", files, _plot)

        # --- Chinese summary with ⚠ disclosures ---
        nas_pct = (abs(net_stock_addition) / total_input * 100.0) if total_input != 0 else float("nan")
        summary.append(
            f"{entry.method} 完成：{len(inputs)} 个输入流、{len(outputs)} 个输出流。{cls_note} "
            f"总输入(吞吐量) TI={total_input:.6g}、总输出 TO={total_output:.6g}；"
            f"净存量增加(平衡残差) NAS = TI − TO = {net_stock_addition:.6g}"
            f"（占总输入 {nas_pct:.2f}%），平衡比 TO/TI={balance_ratio:.6g}。"
            f"{eff_note}{rec_note} 各流明细与占比见 mfa_flows.csv 与平衡图。"
            " ⚠ 质量平衡要求所有流量使用一致的质量单位（如吨）——MFA 把不同单位的流相加无意义，"
            "请确认各列同单位。 ⚠ 残差 NAS 同时含「真实的在用存量净增加」与「未核算流/测量误差」，"
            "二者无法从总量区分（封闭一致的物料平衡里 NAS≈0 或＝真实存量积累）。 ⚠ 进/出流的"
            "名称启发式分类只是默认，请用 config input_flows/output_flows 核对或覆盖；未分类列已排除并披露。"
            " ⚠ NaN 流量单元按 0 计入求和（稀疏流量表惯例，非缺失插补）。"
        )

        code += [
            "import numpy as np, pandas as pd",
            "# inputs / outputs = lists of flow column names (config or name-keyword classified)",
            "# NaN flow cells -> 0 (sparse-table convention)",
            "in_sums  = {c: np.nansum(pd.to_numeric(df[c], errors='coerce').fillna(0)) for c in inputs}",
            "out_sums = {c: np.nansum(pd.to_numeric(df[c], errors='coerce').fillna(0)) for c in outputs}",
            "total_input  = sum(in_sums.values())          # throughput",
            "total_output = sum(out_sums.values())",
            "net_stock_addition = total_input - total_output   # mass-balance residual (NAS)",
            "balance_ratio = total_output / total_input        # 1.0 == perfectly balanced",
            "# resource_efficiency = out_sums[product_flow] / total_input   # if product_flow given",
            "# recycling_rate      = in_sums[recycled_flow] / total_input   # if recycled_flow given",
        ]
    except Exception as exc:
        summary.append(f"material_flow_analysis 计算失败：{exc}")
