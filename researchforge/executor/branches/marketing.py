"""Branch handlers for the MARKETING-ANALYTICS family (customer / commercial
analytics). Three workhorse methods every marketing-analytics stack ships with:

  - rfm_segmentation      — Recency / Frequency / Monetary customer scoring +
                            quintile RFM scores + standard named segments.
  - customer_lifetime_value — Customer Lifetime Value (historical + a simple
                            discounted-margin predictive projection).
  - market_basket         — association-rule mining (hand-rolled Apriori) with
                            support / confidence / lift.

INPUT MODELS
  * rfm_segmentation    : transaction rows — a customer id (config ``customer``),
                          a date (config ``date``), an amount (config ``amount``);
                          else inferred (id-like col / datetime col / amount-like
                          numeric col).
  * customer_lifetime_value : per-customer OR transaction rows — customer id +
                          amount (+ optional date for lifespan). Historical CLV =
                          total monetary per customer; a discounted-margin
                          projection is added under config margin/retention/discount.
  * market_basket       : transaction baskets — long form (config ``transaction``
                          id + ``item``) OR a one-hot 0/1 item matrix.

Conventions (CLAUDE.md「引擎约定」):
  * Honest degrade -> Chinese "<方法> 跳过：<原因>" appended to summary + return
    (never crash / fabricate).
  * Products: CSV + PNG (matplotlib Agg, ENGLISH plot labels, best-effort
    try/except), float ``estimates`` dict (plain floats only; nan for N/A),
    Chinese ``summary`` with ⚠ assumption / bias disclosures.
  * The profiler may classify a customer id as ``id`` / ``categorical`` / ``count``
    and a date as ``datetime`` / ``id`` — detection is tolerant of all of these.

Pure Python (numpy / pandas / matplotlib). ``mlxtend`` is used ONLY if importable;
otherwise the hand-rolled Apriori runs. No method REQUIRES a heavy dependency.
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _save_fig(d, fname, files, build):
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


def _kind_of(fp, name):
    c = fp.column(name)
    return c.kind if c is not None else None


def _pick_customer(ctx):
    """Resolve the customer-id column.

    Order: config ``customer`` -> profiler unit_col -> first id/categorical column
    (a customer id profiles as ``id`` when all-distinct, ``categorical`` when
    repeated, occasionally ``count``). Returns the column name or None.
    """
    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    c = cfg.get("customer")
    if c in df.columns:
        return c
    if fp.unit_col and fp.unit_col in df.columns:
        return fp.unit_col
    for col in fp.columns:
        if col.name in df.columns and col.kind in ("id", "categorical", "count"):
            return col.name
    return None


def _pick_date(ctx):
    """Resolve a transaction-date column.

    Order: config ``date`` -> profiler time_col -> first datetime-kind column ->
    any column that parses as datetime on >90% of non-null rows. Returns a parsed
    pandas Series (datetime64) aligned to df, or None.
    """
    import pandas as pd

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg

    def _try(col):
        if col is None or col not in df.columns:
            return None
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().mean() > 0.9:
            return s
        return None

    s = _try(cfg.get("date"))
    if s is not None:
        return s
    s = _try(fp.time_col)
    if s is not None:
        return s
    for col in fp.columns:
        if col.kind == "datetime":
            s = _try(col.name)
            if s is not None:
                return s
    # last resort: any object/text column that parses as dates
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = _try(col)
            if s is not None:
                return s
    return None


def _pick_amount(ctx, exclude=None):
    """Resolve a monetary amount column.

    Order: config ``amount`` (or ``monetary``) -> a numeric column whose name looks
    monetary (amount/value/sales/revenue/price/spend/total/monetary) -> first
    continuous/count numeric column (excluding the customer id). Returns the column
    name or None.
    """
    import pandas as pd

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg
    exclude = set(exclude or [])

    c = cfg.get("amount") or cfg.get("monetary")
    if c in df.columns:
        return c

    money_words = ("amount", "value", "sales", "revenue", "price", "spend",
                   "spent", "total", "monetary", "amt", "cost", "payment")
    numeric_cols = [
        col.name for col in fp.columns
        if col.name in df.columns and col.name not in exclude
        and col.kind in ("continuous", "count")
    ]
    for col in numeric_cols:
        low = str(col).lower()
        if any(w in low for w in money_words):
            return col
    if numeric_cols:
        return numeric_cols[0]
    # tolerant fallback: any column that coerces to numeric
    for col in df.columns:
        if col in exclude:
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.8:
            return col
    return None


def _quintile_score(series, reverse=False):
    """Score a numeric series into 1..5 by quintiles (rank-based, robust to ties).

    Higher value -> higher score by default. ``reverse=True`` flips it (used for
    Recency: a SMALLER recency = more recent = a HIGHER score). Uses a percentile
    rank cut into 5 equal-width buckets so it degrades gracefully when fewer than
    5 distinct values exist (qcut would raise). Returns an int Series 1..5.
    """
    import numpy as np
    import pandas as pd

    s = pd.Series(series).astype(float)
    n = len(s)
    if n == 0:
        return pd.Series([], dtype=int)
    # average-rank percentile in (0,1]; ties share a rank.
    pct = s.rank(method="average", pct=True)
    # 5 equal-width percentile buckets -> 1..5 (higher value -> higher bucket)
    score = np.ceil(pct * 5).clip(1, 5).astype(int)
    if reverse:
        score = 6 - score
    return pd.Series(score.to_numpy(), index=s.index, dtype=int)


def _rfm_segment(r, f, m):
    """Map an (R,F,M) quintile triple (each 1..5) to a standard named segment.

    Standard RFM segmentation rules (the widely-used "RFM matrix" mapping; see e.g.
    Putler / Optimove segment definitions). FM = average of Frequency & Monetary.
    """
    fm = (f + m) / 2.0
    # Standard RFM matrix mapping. Order matters: most specific cells first so each
    # (R, FM) triple lands in exactly one segment.
    if r >= 4 and fm >= 4:
        return "Champions"           # bought recently, often, big spend
    if r <= 1 and fm >= 4:
        return "Cant Lose Them"      # were best customers, long lapsed
    if r <= 2 and fm >= 3:
        return "At Risk"             # good value, slipping away
    if r >= 3 and fm >= 3:
        return "Loyal Customers"
    if r >= 4 and fm >= 2:
        return "Potential Loyalist"  # recent, building frequency/spend
    if r >= 4 and fm < 2:
        return "New Customers"       # very recent, low frequency/spend
    if r == 3 and fm == 2:
        return "Need Attention"
    if r <= 1 and fm <= 2:
        return "Lost"
    if r <= 2 and fm <= 2:
        return "Hibernating"
    if r >= 3 and fm < 3:
        return "Promising"
    return "Others"


# ===========================================================================
# 1) rfm_segmentation — Recency / Frequency / Monetary scoring
#    Refs: Hughes "Strategic Database Marketing"; standard RFM quintile model.
# ===========================================================================
@register("rfm_segmentation")
def _branch_rfm_segmentation(ctx: Ctx) -> None:
    d, files, summary, estimates, code = (
        ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code)
    import numpy as np
    import pandas as pd

    df = ctx.df
    cust = _pick_customer(ctx)
    if cust is None:
        summary.append("RFM 跳过：未找到客户 id 列（config['customer'] 或一个 id/类别列）。")
        return
    amount_col = _pick_amount(ctx, exclude=[cust])
    if amount_col is None:
        summary.append("RFM 跳过：未找到金额列（config['amount'] 或一个数值列）。需交易级数据。")
        return

    date_series = _pick_date(ctx)
    if date_series is None:
        summary.append("RFM 跳过：未找到日期列（config['date'] 或一个可解析的日期列）。"
                       "Recency 需要交易日期。")
        return

    try:
        work = pd.DataFrame({
            "customer": df[cust].astype(str),
            "amount": pd.to_numeric(df[amount_col], errors="coerce"),
            "date": date_series,
        }).dropna(subset=["date"])
        if work.empty:
            summary.append("RFM 跳过：日期解析后无有效交易行。")
            return
        ref_date = work["date"].max()

        grp = work.groupby("customer")
        recency = (ref_date - grp["date"].max()).dt.days.astype(float)
        frequency = grp.size().astype(float)
        monetary = grp["amount"].sum().astype(float)

        rfm = pd.DataFrame({
            "recency_days": recency,
            "frequency": frequency,
            "monetary": monetary,
        })
        if len(rfm) < 1:
            summary.append("RFM 跳过：聚合后无客户。")
            return

        # quintile scores: recency reversed (more recent = smaller days = score 5).
        rfm["R"] = _quintile_score(rfm["recency_days"], reverse=True)
        rfm["F"] = _quintile_score(rfm["frequency"], reverse=False)
        rfm["M"] = _quintile_score(rfm["monetary"], reverse=False)
        rfm["RFM_score"] = (rfm["R"].astype(str) + rfm["F"].astype(str)
                            + rfm["M"].astype(str))
        rfm["segment"] = [
            _rfm_segment(int(r), int(f), int(m))
            for r, f, m in zip(rfm["R"], rfm["F"], rfm["M"])
        ]
        rfm = rfm.sort_values(["R", "F", "M"], ascending=False)

        rfm.reset_index().rename(columns={"index": "customer"}).to_csv(
            d / "rfm_customers.csv", index=False, encoding="utf-8")
        files.append("rfm_customers.csv")

        seg_counts = rfm["segment"].value_counts()
        seg_counts.rename_axis("segment").reset_index(name="count").to_csv(
            d / "rfm_segment_counts.csv", index=False, encoding="utf-8")
        files.append("rfm_segment_counts.csv")

        n_customers = int(len(rfm))
        n_champions = int((rfm["segment"] == "Champions").sum())
        n_at_risk = int((rfm["segment"] == "At Risk").sum())

        estimates.update({
            "n_customers": float(n_customers),
            "n_champions": float(n_champions),
            "n_at_risk": float(n_at_risk),
            "avg_recency_days": round(float(rfm["recency_days"].mean()), 4),
            "avg_frequency": round(float(rfm["frequency"].mean()), 4),
            "avg_monetary": round(float(rfm["monetary"].mean()), 4),
        })

        def _plot(plt):
            sc = seg_counts.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(3.2, 0.45 * len(sc) + 1)))
            ax.barh(sc.index.astype(str), sc.to_numpy(), color="#4C72B0",
                    edgecolor="white")
            ax.set_xlabel("Number of customers")
            ax.set_ylabel("RFM segment")
            ax.set_title("RFM segment sizes")
            for i, v in enumerate(sc.to_numpy()):
                ax.text(v, i, f" {int(v)}", va="center", fontsize=8)

        _save_fig(d, "rfm_segments.png", files, _plot)

        summary.append(
            f"RFM 分群：{n_customers} 位客户（参考日 {ref_date.date()}）。"
            f"Champions {n_champions} 位、At Risk {n_at_risk} 位；"
            f"平均 Recency {estimates['avg_recency_days']:.1f} 天、"
            f"平均 Frequency {estimates['avg_frequency']:.2f}、"
            f"平均 Monetary {estimates['avg_monetary']:.2f}。"
        )
        summary.append(
            "⚠ 五分位切点是数据相对的（按本数据集的客户分布定 1–5 分），"
            "不可跨数据集直接比较；客户=" + str(cust) + "、金额=" + str(amount_col)
            + "、Recency 相对数据内最大日期计算。需交易级（逐笔）数据。"
        )

        code.append("# RFM segmentation")
        code.append(f"cust, amount, date = {cust!r}, {amount_col!r}, <date col>")
        code.append("work = df[[cust, amount, date]].copy()")
        code.append("work[date] = pd.to_datetime(work[date], errors='coerce')")
        code.append("ref = work[date].max()")
        code.append("g = work.groupby(cust)")
        code.append("recency = (ref - g[date].max()).dt.days")
        code.append("frequency = g.size(); monetary = g[amount].sum()")
        code.append("# R reversed (recent=5); F,M ascending; score in 1..5 quintiles")
        code.append("")
    except Exception as exc:  # pragma: no cover - defensive
        summary.append(f"RFM 跳过：计算异常（{type(exc).__name__}: {exc}）。")
        return


# ===========================================================================
# 2) customer_lifetime_value — CLV (historical + discounted-margin projection)
#    Refs: Gupta & Lehmann "Managing Customers as Investments"; Fader & Hardie.
# ===========================================================================
@register("customer_lifetime_value")
def _branch_customer_lifetime_value(ctx: Ctx) -> None:
    d, files, summary, estimates, code = (
        ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code)
    import numpy as np
    import pandas as pd

    df, cfg = ctx.df, ctx.cfg
    cust = _pick_customer(ctx)
    if cust is None:
        summary.append("CLV 跳过：未找到客户 id 列（config['customer'] 或一个 id/类别列）。")
        return
    amount_col = _pick_amount(ctx, exclude=[cust])
    if amount_col is None:
        summary.append("CLV 跳过：未找到金额列（config['amount'] 或一个数值列）。")
        return

    try:
        amount = pd.to_numeric(df[amount_col], errors="coerce")
        work = pd.DataFrame({"customer": df[cust].astype(str), "amount": amount})
        work = work.dropna(subset=["amount"])
        if work.empty:
            summary.append("CLV 跳过：金额列无有效数值。")
            return

        grp = work.groupby("customer")["amount"]
        # historical CLV = total monetary per customer (backward-looking).
        hist = grp.sum().astype(float)
        n_orders = grp.size().astype(float)
        aov = (hist / n_orders).replace([np.inf, -np.inf], np.nan)

        clv_tbl = pd.DataFrame({
            "historical_clv": hist,
            "n_orders": n_orders,
            "avg_order_value": aov,
        })
        n_customers = int(len(clv_tbl))

        # --- optional discounted-margin predictive projection ------------------
        # CLV = Σ_t (m·r^t)/(1+d)^t = m·r/(1+d−r)   for retention r, margin m,
        # discount d, infinite horizon (geometric series; the standard closed form).
        # m defaults to the mean historical AOV (a per-period contribution proxy).
        def _f(key, default):
            try:
                v = cfg.get(key)
                return float(v) if v is not None else float(default)
            except (TypeError, ValueError):
                return float(default)

        retention = _f("retention", float("nan"))
        discount = _f("discount", 0.10)
        margin_cfg = cfg.get("margin")
        proj_clv = float("nan")
        formula_note = ""
        if retention == retention and 0.0 < retention < 1.0 and (1.0 + discount - retention) > 0:
            if margin_cfg is not None:
                try:
                    m = float(margin_cfg)
                except (TypeError, ValueError):
                    m = float(aov.mean())
            else:
                m = float(aov.mean())
            proj_clv = m * retention / (1.0 + discount - retention)
            formula_note = (
                f"预测 CLV = m·r/(1+d−r) = {m:.2f}·{retention:.2f}"
                f"/(1+{discount:.2f}−{retention:.2f}) = {proj_clv:.2f}"
                "（m=每期边际贡献，r=留存率，d=折现率，无限期等比级数）。"
            )

        clv_tbl = clv_tbl.sort_values("historical_clv", ascending=False)
        clv_tbl.reset_index().rename(columns={"index": "customer"}).to_csv(
            d / "clv_customers.csv", index=False, encoding="utf-8")
        files.append("clv_customers.csv")

        # top-decile CLV share = sum(CLV of top 10% customers) / total CLV.
        total = float(hist.sum())
        sorted_clv = np.sort(hist.to_numpy())[::-1]
        k = max(1, int(np.ceil(0.10 * n_customers)))
        top_decile_share = (
            float(sorted_clv[:k].sum() / total) if total > 0 else float("nan"))

        estimates.update({
            "mean_clv": round(float(hist.mean()), 4),
            "median_clv": round(float(hist.median()), 4),
            "top_decile_share": (round(top_decile_share, 4)
                                 if top_decile_share == top_decile_share else float("nan")),
            "n_customers": float(n_customers),
            "retention_used": (round(retention, 4)
                               if retention == retention else float("nan")),
            "discount_used": round(float(discount), 4),
            "projected_clv": (round(proj_clv, 4)
                              if proj_clv == proj_clv else float("nan")),
        })

        def _plot(plt):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            vals = hist.to_numpy()
            bins = min(30, max(5, int(np.sqrt(max(1, n_customers)))))
            ax.hist(vals, bins=bins, color="#55A868", edgecolor="white")
            ax.axvline(float(hist.mean()), color="#C44E52", lw=1.6,
                       label=f"mean = {hist.mean():.1f}")
            ax.axvline(float(hist.median()), color="#4C72B0", lw=1.4, ls="--",
                       label=f"median = {hist.median():.1f}")
            ax.set_xlabel("Historical CLV (total monetary per customer)")
            ax.set_ylabel("Number of customers")
            ax.set_title("Customer Lifetime Value distribution")
            ax.legend()

        _save_fig(d, "clv_distribution.png", files, _plot)

        msg = (f"CLV：{n_customers} 位客户，历史 CLV（累计消费）均值 "
               f"{estimates['mean_clv']:.2f}、中位 {estimates['median_clv']:.2f}；"
               f"前 10% 客户贡献 ")
        msg += (f"{top_decile_share*100:.1f}% 的总价值。"
                if top_decile_share == top_decile_share else "（不可计算）。")
        if formula_note:
            msg += " " + formula_note
        summary.append(msg)
        summary.append(
            "⚠ 历史 CLV 是回顾性的（仅累计已发生消费）；预测 CLV 依赖留存率 r 与折现率 d 假设"
            "（config retention/discount/margin），假设变动会显著改变结果。"
            "客户=" + str(cust) + "、金额=" + str(amount_col) + "。"
        )

        code.append("# Customer Lifetime Value (historical + projection)")
        code.append(f"cust, amount = {cust!r}, {amount_col!r}")
        code.append("g = df.groupby(cust)[amount]")
        code.append("historical_clv = g.sum(); n_orders = g.size()")
        code.append("aov = historical_clv / n_orders")
        code.append("# projection: CLV = m*r/(1+d-r)  (m=margin, r=retention, d=discount)")
        code.append("")
    except Exception as exc:  # pragma: no cover - defensive
        summary.append(f"CLV 跳过：计算异常（{type(exc).__name__}: {exc}）。")
        return


# ===========================================================================
# 3) market_basket — association-rule mining (hand-rolled Apriori)
#    Refs: Agrawal & Srikant "Fast Algorithms for Mining Association Rules".
# ===========================================================================
def _baskets_from_data(ctx):
    """Return (list_of_baskets, n_transactions, note, err).

    Two input forms:
      1. long form: config ``transaction`` id + ``item`` -> group items per
         transaction id.
      2. one-hot matrix: 2+ columns each ~ 0/1 (binary), each row a basket -> the
         columns where the row is 1 are the items.
    Each basket is a set of distinct item labels (str). ``err`` is non-None on
    honest failure.
    """
    import pandas as pd

    df, fp, cfg = ctx.df, ctx.fp, ctx.cfg

    tcol = cfg.get("transaction")
    icol = cfg.get("item")
    # auto-detect long form: an id/categorical 'transaction-like' + a categorical
    # 'item-like' column, only when not explicitly given.
    if (tcol not in df.columns or icol not in df.columns):
        tname = iname = None
        for col in fp.columns:
            low = str(col.name).lower()
            if col.name not in df.columns:
                continue
            if tname is None and any(w in low for w in
                                     ("transaction", "order", "invoice", "basket", "receipt", "ticket")):
                tname = col.name
            if iname is None and any(w in low for w in
                                     ("item", "product", "sku", "article", "good")):
                iname = col.name
        if tcol not in df.columns and tname is not None:
            tcol = tname
        if icol not in df.columns and iname is not None:
            icol = iname

    if tcol in df.columns and icol in df.columns:
        long = df[[tcol, icol]].dropna()
        if long.empty:
            return None, 0, "", "长表（transaction+item）无有效行。"
        baskets = (long.groupby(tcol)[icol]
                   .apply(lambda s: set(str(x) for x in s.unique()))
                   .tolist())
        baskets = [b for b in baskets if b]
        note = f"长表：交易={tcol}、商品={icol}。"
        return baskets, len(baskets), note, None

    # one-hot detection: binary columns (0/1) — each row is a basket.
    binary_cols = []
    for col in fp.columns:
        if col.name not in df.columns:
            continue
        if col.kind == "binary":
            binary_cols.append(col.name)
        else:
            ser = pd.to_numeric(df[col.name], errors="coerce").dropna()
            if len(ser) > 0 and set(ser.unique().tolist()) <= {0, 1}:
                binary_cols.append(col.name)
    if len(binary_cols) >= 2:
        sub = df[binary_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        baskets = []
        for _, row in sub.iterrows():
            items = {str(c) for c in binary_cols if row[c] == 1}
            if items:
                baskets.append(items)
        if not baskets:
            return None, 0, "", "独热矩阵每行均为空篮（无 1）。"
        note = f"独热矩阵：{len(binary_cols)} 个商品列。"
        return baskets, len(baskets), note, None

    return (None, 0, "",
            "需要长表（config transaction+item）或独热 0/1 商品矩阵（≥2 列）。")


def _apriori_rules(baskets, min_support, min_confidence, max_len):
    """Hand-rolled Apriori: frequent itemsets up to ``max_len`` then 1->many rules.

    support(X)    = (# baskets containing X) / N
    confidence(A->B) = support(A∪B) / support(A)
    lift(A->B)    = confidence(A->B) / support(B)
                  = support(A∪B) / (support(A)·support(B))

    Returns (rules_list, itemset_supports). Each rule is a dict with antecedent,
    consequent (frozensets), support, confidence, lift. itemset_supports maps each
    frequent frozenset -> support. Candidate generation is capped at ``max_len`` to
    bound the combinatorial cost.
    """
    from itertools import combinations

    n = len(baskets)
    if n == 0:
        return [], {}
    baskets = [frozenset(b) for b in baskets]

    # --- level 1: frequent single items ------------------------------------
    counts1 = {}
    for b in baskets:
        for it in b:
            counts1[it] = counts1.get(it, 0) + 1
    supports = {}
    L = {}  # size -> set of frequent frozensets
    freq1 = set()
    for it, c in counts1.items():
        sup = c / n
        if sup >= min_support:
            fs = frozenset([it])
            supports[fs] = sup
            freq1.add(fs)
    L[1] = freq1

    # --- levels 2..max_len: generate, count, prune by min_support ----------
    k = 2
    max_len = max(1, int(max_len))
    while k <= max_len and L.get(k - 1):
        prev_items = sorted({it for fs in L[k - 1] for it in fs})
        candidates = set()
        # generate k-itemsets from frequent items; downward-closure prune: every
        # (k-1)-subset must be frequent.
        for combo in combinations(prev_items, k):
            cset = frozenset(combo)
            if all(frozenset(sub) in supports
                   for sub in combinations(combo, k - 1)):
                candidates.add(cset)
        if not candidates:
            break
        cand_counts = {c: 0 for c in candidates}
        for b in baskets:
            for c in candidates:
                if c <= b:
                    cand_counts[c] += 1
        Lk = set()
        for c, cnt in cand_counts.items():
            sup = cnt / n
            if sup >= min_support:
                supports[c] = sup
                Lk.add(c)
        L[k] = Lk
        k += 1

    # --- rule generation: split each frequent itemset (size>=2) ------------
    rules = []
    for itemset, sup in supports.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        # enumerate non-empty proper antecedents
        for r in range(1, len(items)):
            for ant in combinations(items, r):
                A = frozenset(ant)
                B = itemset - A
                if not B:
                    continue
                supA = supports.get(A)
                supB = supports.get(B)
                if not supA:
                    continue
                conf = sup / supA
                if conf < min_confidence:
                    continue
                lift = conf / supB if supB else float("nan")
                rules.append({
                    "antecedent": A,
                    "consequent": B,
                    "support": sup,
                    "confidence": conf,
                    "lift": lift,
                })
    return rules, supports


@register("market_basket")
def _branch_market_basket(ctx: Ctx) -> None:
    d, files, summary, estimates, code = (
        ctx.d, ctx.files, ctx.summary, ctx.estimates, ctx.code)
    import numpy as np
    import pandas as pd

    cfg = ctx.cfg

    baskets, n_tx, note, err = _baskets_from_data(ctx)
    if err is not None:
        summary.append(f"购物篮 跳过：{err}")
        return

    def _f(key, default):
        try:
            v = cfg.get(key)
            return float(v) if v is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    min_support = _f("min_support", 0.01)
    min_confidence = _f("min_confidence", 0.3)
    try:
        max_len = int(cfg.get("max_len", 3))
    except (TypeError, ValueError):
        max_len = 3
    max_len = max(2, min(max_len, 5))  # need >=2 for rules; cap cost.

    try:
        rules, supports = None, None
        # mlxtend ONLY if importable; otherwise the hand-rolled Apriori.
        used_backend = "hand-rolled Apriori"
        try:
            from mlxtend.frequent_patterns import apriori as _mlx_apriori
            from mlxtend.frequent_patterns import association_rules as _mlx_rules
            from mlxtend.preprocessing import TransactionEncoder

            te = TransactionEncoder()
            arr = te.fit([list(b) for b in baskets]).transform([list(b) for b in baskets])
            onehot = pd.DataFrame(arr, columns=te.columns_)
            fi = _mlx_apriori(onehot, min_support=min_support,
                              max_len=max_len, use_colnames=True)
            if fi.empty:
                rules = []
                supports = {frozenset(s): float(sp)
                            for s, sp in zip(fi.get("itemsets", []), fi.get("support", []))}
            else:
                supports = {frozenset(s): float(sp)
                            for s, sp in zip(fi["itemsets"], fi["support"])}
                rdf = _mlx_rules(fi, metric="confidence",
                                 min_threshold=min_confidence)
                rules = [{
                    "antecedent": frozenset(row["antecedents"]),
                    "consequent": frozenset(row["consequents"]),
                    "support": float(row["support"]),
                    "confidence": float(row["confidence"]),
                    "lift": float(row["lift"]),
                } for _, row in rdf.iterrows()]
            used_backend = "mlxtend"
        except Exception:
            rules, supports = _apriori_rules(
                baskets, min_support, min_confidence, max_len)

        n_frequent = int(len(supports)) if supports is not None else 0
        rules = rules or []
        # sort rules by lift then confidence, descending.
        rules.sort(key=lambda x: (
            x["lift"] if x["lift"] == x["lift"] else -1.0, x["confidence"]),
            reverse=True)

        def _fmt(fs):
            return "{" + ", ".join(sorted(str(x) for x in fs)) + "}"

        rows = [{
            "antecedent": _fmt(r["antecedent"]),
            "consequent": _fmt(r["consequent"]),
            "support": round(r["support"], 6),
            "confidence": round(r["confidence"], 6),
            "lift": (round(r["lift"], 6) if r["lift"] == r["lift"] else float("nan")),
        } for r in rules]
        rules_df = pd.DataFrame(
            rows, columns=["antecedent", "consequent", "support", "confidence", "lift"])
        rules_df.to_csv(d / "market_basket_rules.csv", index=False, encoding="utf-8")
        files.append("market_basket_rules.csv")

        top_lift = float(rules[0]["lift"]) if rules else float("nan")
        top_conf = float(rules[0]["confidence"]) if rules else float("nan")

        estimates.update({
            "n_frequent_itemsets": float(n_frequent),
            "n_rules": float(len(rules)),
            "top_lift": (round(top_lift, 6) if top_lift == top_lift else float("nan")),
            "top_confidence": (round(top_conf, 6) if top_conf == top_conf else float("nan")),
            "n_transactions": float(n_tx),
        })

        if rules:
            def _plot(plt):
                top = rules[:min(10, len(rules))][::-1]
                labels = [f"{_fmt(r['antecedent'])} -> {_fmt(r['consequent'])}"
                          for r in top]
                lifts = [r["lift"] for r in top]
                fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.5 * len(top) + 1)))
                ax.barh(range(len(top)), lifts, color="#8172B3", edgecolor="white")
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(labels, fontsize=7)
                ax.axvline(1.0, color="#C44E52", lw=1.0, ls="--",
                           label="lift = 1 (independence)")
                ax.set_xlabel("Lift")
                ax.set_title("Top association rules by lift")
                ax.legend(fontsize=8)

            _save_fig(d, "market_basket_lift.png", files, _plot)

        if rules:
            r0 = rules[0]
            summary.append(
                f"购物篮（{used_backend}）：{n_tx} 笔交易，{n_frequent} 个频繁项集、"
                f"{len(rules)} 条规则（min_support={min_support}, "
                f"min_confidence={min_confidence}, max_len={max_len}）。"
                f"最强规则 {_fmt(r0['antecedent'])} → {_fmt(r0['consequent'])}："
                f"support={r0['support']:.3f}、confidence={r0['confidence']:.3f}、"
                f"lift={r0['lift']:.3f}。"
            )
        else:
            summary.append(
                f"购物篮（{used_backend}）：{n_tx} 笔交易，{n_frequent} 个频繁项集，"
                f"但在 min_support={min_support}、min_confidence={min_confidence} "
                "下未产生规则（可调低阈值）。"
            )
        summary.append(
            "⚠ lift>1 表示正向关联（非因果）；support/confidence 阈值会改变结果；"
            f"组合成本由 max_len={max_len} 上限控制。" + note
        )

        code.append("# Market-basket association rules (Apriori)")
        code.append("# support(X)=#baskets_with_X/N; conf(A->B)=sup(AuB)/sup(A);")
        code.append("# lift(A->B)=conf/sup(B)=sup(AuB)/(sup(A)*sup(B))")
        code.append(f"min_support, min_confidence, max_len = "
                    f"{min_support}, {min_confidence}, {max_len}")
        code.append("")
    except Exception as exc:  # pragma: no cover - defensive
        summary.append(f"购物篮 跳过：计算异常（{type(exc).__name__}: {exc}）。")
        return
