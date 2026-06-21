"""Branch handlers for the latent-variable mixture family (LCA / LPA via stepmix).

Two finite-mixture / latent-class methods built on stepmix (sklearn-compatible EM):

* ``latent_class_analysis`` — LCA for CATEGORICAL / BINARY indicators
  (Bernoulli / Multinoulli measurement). Recovers latent classes from
  item-response patterns.
* ``latent_profile_analysis`` — LPA = the continuous analogue (Gaussian mixture
  with free per-class diagonal variances). Recovers latent profiles from
  continuous indicators.

Both: select k by BIC over a config range, report mixing proportions, per-class
profiles (item probabilities / indicator means), the scaled relative entropy
(0-1 classification certainty), the BIC/AIC curve + selected k, and posterior
class membership per row. Honest skip when indicators / rows are insufficient.

stepmix API used (verified against installed v3.0.0):
  StepMix(n_components=k, measurement=..., n_init=..., random_state=..., max_iter=...,
          progress_bar=0).fit(X)
  .bic(X) / .aic(X)            -> float  (lower is better)
  .relative_entropy(X)         -> float in [0,1]  (1 - entropy/(n*ln K); Ramaswamy 1993)
  .predict(X)                  -> hard modal class labels (n,)
  .predict_proba(X)            -> posterior P(class|X) matrix (n, k)
  .get_parameters()["weights"] -> mixing proportions (k,)
  binary:   get_parameters()["measurement"]["pis"]  shape (k, n_items) = P(item=1|class)
  gaussian: get_parameters()["measurement"]["means"] / ["covariances"]  shape (k, n_items)
  categorical (multinoulli): per-outcome item probs via get_mm_df() (param "pis",
  variables expanded to "<item>_<outcome>").

⚠ A latent class is a STATISTICAL construct (a mixture component), not an externally
validated group; k is a modeling choice (we report the BIC curve); class identity is
arbitrary (label switching — we order by size); EM can hit local optima (n_init>1).
"""

from __future__ import annotations

from researchforge.executor._branch_api import Ctx, register


# --------------------------------------------------------------------------- #
# shared helpers (local to this family)
# --------------------------------------------------------------------------- #
def _k_range_from_cfg(cfg: dict, n_rows: int) -> tuple[list[int], int | None]:
    """Resolve the candidate k range and an optional forced k from config.

    Returns (candidate_ks, forced_k). ``n_classes`` (or ``k``) forces a single k;
    otherwise ``k_min``..``k_max`` (default 2..5), capped so each class can hold a
    few rows.
    """
    forced = cfg.get("n_classes", cfg.get("k"))
    hard_cap = max(2, min(8, n_rows // 5))  # need ~5 rows/class to estimate
    if forced is not None:
        try:
            kf = int(forced)
            if kf >= 2:
                return [min(kf, hard_cap)], min(kf, hard_cap)
        except (TypeError, ValueError):
            pass
    try:
        k_min = max(2, int(cfg.get("k_min", 2)))
    except (TypeError, ValueError):
        k_min = 2
    try:
        k_max = int(cfg.get("k_max", 5))
    except (TypeError, ValueError):
        k_max = 5
    k_max = min(k_max, hard_cap)
    if k_max < k_min:
        k_max = k_min
    return list(range(k_min, k_max + 1)), None


def _order_by_size(weights, labels, proba):
    """Stable, deterministic class relabeling: order classes by descending size.

    Returns (perm, new_labels, new_proba, new_weights) where ``perm[new]=old``.
    Tackles label switching — class identity is arbitrary in mixture models, so we
    fix a canonical order (largest class = 0) for reproducible reporting.
    """
    import numpy as np

    counts = np.bincount(labels, minlength=len(weights))
    # sort by count desc, tie-break by weight desc for stability
    order = sorted(range(len(weights)), key=lambda c: (-counts[c], -float(weights[c])))
    old_to_new = {old: new for new, old in enumerate(order)}
    new_labels = np.array([old_to_new[int(l)] for l in labels])
    new_proba = proba[:, order]
    new_weights = np.asarray(weights)[order]
    return order, new_labels, new_proba, new_weights


def _fit_select(make_model, X, ks, n_init, seed):
    """Fit a StepMix model per k, pick the k with the lowest BIC.

    ``make_model(k)`` builds a fresh StepMix. Returns
    (best_model, best_k, curve) where curve is a list of dicts per k.
    Skips ks that fail to fit (e.g. too few rows / degenerate)."""
    best = None  # (bic, k, model)
    curve = []
    for k in ks:
        try:
            m = make_model(k)
            m.fit(X)
            bic = float(m.bic(X))
            aic = float(m.aic(X))
        except Exception:
            continue
        curve.append({"k": k, "bic": round(bic, 3), "aic": round(aic, 3)})
        if best is None or bic < best[0]:
            best = (bic, k, m)
    if best is None:
        return None, None, curve
    return best[2], best[1], curve


def _plot_bic_curve(curve, selected_k, png_path, title):
    """BIC/AIC vs k line plot (English labels). Best-effort."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [c["k"] for c in curve]
        bics = [c["bic"] for c in curve]
        aics = [c["aic"] for c in curve]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, bics, "o-", label="BIC")
        ax.plot(ks, aics, "s--", label="AIC", alpha=0.7)
        if selected_k in ks:
            ax.axvline(selected_k, color="red", ls=":", alpha=0.6, label=f"selected k={selected_k}")
        ax.set_xlabel("number of classes (k)")
        ax.set_ylabel("information criterion (lower = better)")
        ax.set_title(title)
        ax.set_xticks(ks)
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 1) Latent Class Analysis — categorical / binary indicators
# --------------------------------------------------------------------------- #
@register("latent_class_analysis")
def _branch_latent_class_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    # Indicators: categorical / binary / count (rater/item codes profile as count).
    # The profiler's "id" trap can hit small-integer items; allow config override.
    auto_ind = [
        c.name
        for c in fp.columns
        if c.kind in {"categorical", "binary", "count"}
        and c.name not in {fp.unit_col, fp.time_col}
    ]
    forced = [c for c in (cfg.get("indicators") or []) if c in df.columns]
    indicators = forced or auto_ind

    if importlib.util.find_spec("stepmix") is None:
        summary.append("潜在类别分析（LCA）需要 stepmix 包（未检测到）。安装：pip install stepmix。")
        return
    if len(indicators) < 2:
        summary.append(
            "潜在类别分析（LCA）跳过：需要 ≥2 个分类/二值/计数指标"
            "（用 config={\"indicators\":[...]} 指定）。"
        )
        return

    try:
        import numpy as np
        import pandas as pd
        from stepmix import StepMix

        sub = df[indicators].dropna()
        if len(sub) < 20:
            summary.append(f"潜在类别分析（LCA）跳过：有效样本仅 {len(sub)} 行（<20），不足以稳健估计。")
            return

        # Integer-encode each indicator to 0-indexed categories (stepmix
        # categorical/binary measurement expects integer codes). pd.factorize
        # gives 0..L-1; binary stays 0/1 if already coded so.
        enc = pd.DataFrame(index=sub.index)
        levels = {}
        for col in indicators:
            codes, uniques = pd.factorize(sub[col], sort=True)
            enc[col] = codes
            levels[col] = list(uniques)
        n_levels = {c: len(levels[c]) for c in indicators}

        if any(v < 2 for v in n_levels.values()):
            const = [c for c, v in n_levels.items() if v < 2]
            summary.append(f"潜在类别分析（LCA）跳过：指标 {const} 为常数（仅 1 个取值）。")
            return

        X = enc.values.astype(int)
        all_binary = all(v == 2 for v in n_levels.values())
        measurement = "binary" if all_binary else "categorical"

        try:
            n_init = max(1, int(cfg.get("n_init", 10)))
        except (TypeError, ValueError):
            n_init = 10
        try:
            seed = int(cfg.get("seed", cfg.get("random_state", 42)))
        except (TypeError, ValueError):
            seed = 42

        ks, forced_k = _k_range_from_cfg(cfg, len(sub))

        def make(k):
            return StepMix(
                n_components=k,
                measurement=measurement,
                n_init=n_init,
                random_state=seed,
                max_iter=1000,
                progress_bar=0,
                verbose=0,
            )

        model, best_k, curve = _fit_select(make, X, ks, n_init, seed)
        if model is None:
            summary.append("潜在类别分析（LCA）失败：所有候选 k 均无法收敛（数据可能过于稀疏）。")
            return

        proba = np.asarray(model.predict_proba(X))
        labels = np.asarray(model.predict(X)).astype(int)
        weights = np.asarray(model.get_parameters()["weights"]).ravel()

        # label-switching fix: canonical order = largest class first
        _order, labels, proba, weights = _order_by_size(weights, labels, proba)
        rel_entropy = float(model.relative_entropy(X))
        if not np.isfinite(rel_entropy):
            rel_entropy = float("nan")

        # --- membership CSV (posterior + modal class) ---
        mem = pd.DataFrame({"row": sub.index, "class": labels})
        for c in range(best_k):
            mem[f"posterior_class{c}"] = proba[:, c]
        mem["max_posterior"] = proba.max(axis=1)
        mem.to_csv(d / "class_membership.csv", index=False, encoding="utf-8")
        files.append("class_membership.csv")

        # --- per-class item-response profiles ---
        # Build a tidy profile table: rows = item (and outcome for multinoulli),
        # cols = class0..k-1, value = P(response | class). Reorder cols to size order.
        prof_rows = []
        if measurement == "binary":
            pis = np.asarray(model.get_parameters()["measurement"]["pis"])  # (k, n_items)
            pis = pis[_order, :]  # apply canonical class order
            for j, item in enumerate(indicators):
                row = {"item": item, "response": f"P({item}={levels[item][1]})"}
                for c in range(best_k):
                    row[f"class{c}"] = round(float(pis[c, j]), 4)
                prof_rows.append(row)
        else:
            # categorical/multinoulli: use the long-form measurement DF (per outcome)
            try:
                mm = model.get_mm_df(x_names=indicators)  # index (model_name,param,variable)
                mm = mm.reset_index()
                mm = mm[mm["param"] == "pis"]
                # reorder class columns by canonical order
                class_cols = [c for c in mm.columns if c not in {"model_name", "param", "variable"}]
                ordered_cols = [class_cols[i] for i in _order if i < len(class_cols)]
                for _, r in mm.iterrows():
                    row = {"item": r["variable"], "response": "P(=outcome)"}
                    for new_c, old_col in enumerate(ordered_cols):
                        row[f"class{new_c}"] = round(float(r[old_col]), 4)
                    prof_rows.append(row)
            except Exception:
                prof_rows = []

        if prof_rows:
            prof_df = pd.DataFrame(prof_rows)
        else:
            prof_df = pd.DataFrame()
        # mixing proportions row appended as metadata-free separate CSV
        class_sizes = np.bincount(labels, minlength=best_k)
        size_df = pd.DataFrame({
            "class": list(range(best_k)),
            "mixing_proportion": [round(float(w), 4) for w in weights],
            "n_assigned": [int(s) for s in class_sizes],
        })
        size_df.to_csv(d / "class_sizes.csv", index=False, encoding="utf-8")
        files.append("class_sizes.csv")
        if not prof_df.empty:
            prof_df.to_csv(d / "class_profiles.csv", index=False, encoding="utf-8")
            files.append("class_profiles.csv")

        # --- BIC curve CSV ---
        pd.DataFrame(curve).to_csv(d / "bic_curve.csv", index=False, encoding="utf-8")
        files.append("bic_curve.csv")

        # --- plots ---
        if len(curve) >= 2:
            if _plot_bic_curve(curve, best_k, d / "bic_curve.png", "LCA — BIC / AIC vs k"):
                files.append("bic_curve.png")
        # item-probability profile plot (binary: one prob per item per class)
        try:
            if measurement == "binary":
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                pis = np.asarray(model.get_parameters()["measurement"]["pis"])[_order, :]
                fig, ax = plt.subplots(figsize=(max(6, len(indicators) * 0.8), 4))
                x = np.arange(len(indicators))
                for c in range(best_k):
                    ax.plot(x, pis[c, :], "o-", label=f"class {c} (n={class_sizes[c]})")
                ax.set_xticks(x)
                ax.set_xticklabels(indicators, rotation=30, ha="right")
                ax.set_ylabel("P(item = 1 | class)")
                ax.set_ylim(-0.02, 1.02)
                ax.set_title("LCA — item-response profiles by class")
                ax.legend()
                fig.tight_layout()
                fig.savefig(d / "class_profiles.png", dpi=150)
                plt.close(fig)
                files.append("class_profiles.png")
        except Exception:
            pass

        # --- estimates ---
        bic_sel = next((c["bic"] for c in curve if c["k"] == best_k), float("nan"))
        aic_sel = next((c["aic"] for c in curve if c["k"] == best_k), float("nan"))
        estimates["n_classes"] = float(best_k)
        estimates["bic"] = round(float(bic_sel), 3)
        estimates["aic"] = round(float(aic_sel), 3)
        estimates["entropy"] = round(rel_entropy, 4) if np.isfinite(rel_entropy) else float("nan")
        estimates["n"] = float(len(sub))
        estimates["largest_class_share"] = round(float(weights.max()), 4)

        # --- summary (Chinese, with ⚠ disclosures) ---
        ent_txt = (
            f"熵={rel_entropy:.3f}（{'分离良好' if rel_entropy >= 0.8 else '分离一般/偏低'}，1=完美分类）"
            if np.isfinite(rel_entropy) else "熵=不可用"
        )
        sel_txt = "（config 指定 k）" if forced_k is not None else "（BIC 选取）"
        sizes_pct = "、".join(f"类{c} {weights[c]:.0%}" for c in range(best_k))
        (d / "lca_summary.txt").write_text(
            f"潜在类别分析（LCA，stepmix，measurement={measurement}，n_init={n_init}，seed={seed}）\n"
            f"指标（{len(indicators)} 个）：{indicators}\n"
            f"选定潜在类别数 k={best_k} {sel_txt}；样本 n={len(sub)}\n"
            f"混合比例（按规模降序）：{sizes_pct}\n"
            f"BIC={bic_sel:.2f}，AIC={aic_sel:.2f}，{ent_txt}\n\n"
            "类别画像（每类的条目响应概率，见 class_profiles.csv）——这是定义各潜在类别的特征。\n\n"
            "⚠ 披露：\n"
            "1) 潜在类别是统计构念（混合成分），不是经外部验证的真实分组；\n"
            "2) k 是建模选择，已报告 BIC/AIC 曲线（bic_curve.csv/png），不同准则可能选不同 k；\n"
            "3) 类别编号本身无意义（标签切换）——已按规模降序固定编号（类0 最大）；\n"
            f"4) EM 可能陷入局部最优，已用 n_init={n_init} 次随机初始化 + 固定 seed={seed} 缓解；\n"
            "5) LCA 需要分类/二值指标；连续指标请用潜在剖面分析（LPA）。\n",
            encoding="utf-8",
        )
        files.append("lca_summary.txt")

        summary.append(
            f"{entry.method} 完成（stepmix，{measurement}）：{len(indicators)} 个指标上识别出 "
            f"k={best_k} 个潜在类别{sel_txt}；混合比例 {sizes_pct}；{ent_txt}；"
            f"BIC={bic_sel:.1f}（n={len(sub)}）。"
            "⚠ 类别为统计构念（非外部验证分组）；k 由 BIC 选（见曲线）；标签按规模降序固定；"
            f"EM 局部最优已用 n_init={n_init}+seed 缓解。"
        )
        code += [
            "from stepmix import StepMix  # 潜在类别分析 LCA（EM）",
            f"# X = integer-encoded indicators {indicators}",
            f"m = StepMix(n_components={best_k}, measurement={measurement!r}, n_init={n_init}, random_state={seed})",
            "m.fit(X); labels = m.predict(X); proba = m.predict_proba(X)",
            "print('BIC', m.bic(X), 'relative_entropy', m.relative_entropy(X))",
            "# get_parameters()['weights'] = mixing props; ['measurement']['pis'] = item probs",
        ]
    except Exception as err:
        summary.append(f"潜在类别分析（LCA）执行失败：{err}")


# --------------------------------------------------------------------------- #
# 2) Latent Profile Analysis — continuous indicators (Gaussian mixture)
# --------------------------------------------------------------------------- #
@register("latent_profile_analysis")
def _branch_latent_profile_analysis(ctx: Ctx) -> None:
    df, fp, entry, cfg, d = ctx.df, ctx.fp, ctx.entry, ctx.cfg, ctx.d
    files, summary, estimates, code = ctx.files, ctx.summary, ctx.estimates, ctx.code
    import importlib.util

    auto_ind = [
        c.name
        for c in fp.columns
        if c.kind == "continuous" and c.name not in {fp.unit_col, fp.time_col}
    ]
    forced = [c for c in (cfg.get("indicators") or []) if c in df.columns]
    indicators = forced or auto_ind

    if importlib.util.find_spec("stepmix") is None:
        summary.append("潜在剖面分析（LPA）需要 stepmix 包（未检测到）。安装：pip install stepmix。")
        return
    if len(indicators) < 2:
        summary.append(
            "潜在剖面分析（LPA）跳过：需要 ≥2 个连续指标"
            "（用 config={\"indicators\":[...]} 指定）。"
        )
        return

    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from stepmix import StepMix

        sub = df[indicators].dropna()
        if len(sub) < 20:
            summary.append(f"潜在剖面分析（LPA）跳过：有效样本仅 {len(sub)} 行（<20），不足以稳健估计。")
            return

        # near-constant indicators break Gaussian variance estimation
        stds = sub.std(numeric_only=True)
        const_cols = [c for c in indicators if not np.isfinite(stds.get(c, 0)) or stds.get(c, 0) == 0]
        if const_cols:
            summary.append(f"潜在剖面分析（LPA）跳过：指标 {const_cols} 近似常数，无法估计方差。")
            return

        # Standardize indicators (scale-sensitive Gaussian mixture); keep the
        # scaler so we can report profiles on the ORIGINAL scale too.
        scaler = StandardScaler()
        Xs = scaler.fit_transform(sub.values.astype(float))

        try:
            n_init = max(1, int(cfg.get("n_init", 10)))
        except (TypeError, ValueError):
            n_init = 10
        try:
            seed = int(cfg.get("seed", cfg.get("random_state", 42)))
        except (TypeError, ValueError):
            seed = 42

        ks, forced_k = _k_range_from_cfg(cfg, len(sub))

        # gaussian_diag = free per-class, per-indicator diagonal variances (the
        # standard LPA "free variances" / class-varying model).
        def make(k):
            return StepMix(
                n_components=k,
                measurement="gaussian_diag",
                n_init=n_init,
                random_state=seed,
                max_iter=1000,
                progress_bar=0,
                verbose=0,
            )

        model, best_k, curve = _fit_select(make, Xs, ks, n_init, seed)
        if model is None:
            summary.append("潜在剖面分析（LPA）失败：所有候选 k 均无法收敛。")
            return

        proba = np.asarray(model.predict_proba(Xs))
        labels = np.asarray(model.predict(Xs)).astype(int)
        params = model.get_parameters()
        weights = np.asarray(params["weights"]).ravel()
        means_std = np.asarray(params["measurement"]["means"])  # (k, n_ind), standardized
        covs_std = np.asarray(params["measurement"]["covariances"])  # (k, n_ind) for diag

        _order, labels, proba, weights = _order_by_size(weights, labels, proba)
        means_std = means_std[_order, :]
        covs_std = covs_std[_order, :]
        rel_entropy = float(model.relative_entropy(Xs))
        if not np.isfinite(rel_entropy):
            rel_entropy = float("nan")

        # back-transform means to original scale: mean_orig = mean_std*scale + center
        scale_ = scaler.scale_
        center_ = scaler.mean_
        means_orig = means_std * scale_ + center_
        # variances on original scale: var_orig = var_std * scale^2
        vars_orig = covs_std * (scale_ ** 2)

        # --- membership CSV ---
        mem = pd.DataFrame({"row": sub.index, "class": labels})
        for c in range(best_k):
            mem[f"posterior_class{c}"] = proba[:, c]
        mem["max_posterior"] = proba.max(axis=1)
        mem.to_csv(d / "class_membership.csv", index=False, encoding="utf-8")
        files.append("class_membership.csv")

        # --- profiles: per-class means + variances (original scale) ---
        prof_rows = []
        for j, item in enumerate(indicators):
            row = {"indicator": item}
            for c in range(best_k):
                row[f"class{c}_mean"] = round(float(means_orig[c, j]), 4)
                row[f"class{c}_var"] = round(float(vars_orig[c, j]), 4)
            prof_rows.append(row)
        pd.DataFrame(prof_rows).to_csv(d / "class_profiles.csv", index=False, encoding="utf-8")
        files.append("class_profiles.csv")

        class_sizes = np.bincount(labels, minlength=best_k)
        size_df = pd.DataFrame({
            "class": list(range(best_k)),
            "mixing_proportion": [round(float(w), 4) for w in weights],
            "n_assigned": [int(s) for s in class_sizes],
        })
        size_df.to_csv(d / "class_sizes.csv", index=False, encoding="utf-8")
        files.append("class_sizes.csv")

        pd.DataFrame(curve).to_csv(d / "bic_curve.csv", index=False, encoding="utf-8")
        files.append("bic_curve.csv")

        # --- plots ---
        if len(curve) >= 2:
            if _plot_bic_curve(curve, best_k, d / "bic_curve.png", "LPA — BIC / AIC vs k"):
                files.append("bic_curve.png")
        # mean-profile line plot (standardized means, so indicators are comparable)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(6, len(indicators) * 0.8), 4))
            x = np.arange(len(indicators))
            for c in range(best_k):
                ax.plot(x, means_std[c, :], "o-", label=f"class {c} (n={class_sizes[c]})")
            ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(indicators, rotation=30, ha="right")
            ax.set_ylabel("standardized mean (z)")
            ax.set_title("LPA — class mean profiles (standardized)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(d / "class_profiles.png", dpi=150)
            plt.close(fig)
            files.append("class_profiles.png")
        except Exception:
            pass

        # --- estimates ---
        bic_sel = next((c["bic"] for c in curve if c["k"] == best_k), float("nan"))
        aic_sel = next((c["aic"] for c in curve if c["k"] == best_k), float("nan"))
        estimates["n_classes"] = float(best_k)
        estimates["bic"] = round(float(bic_sel), 3)
        estimates["aic"] = round(float(aic_sel), 3)
        estimates["entropy"] = round(rel_entropy, 4) if np.isfinite(rel_entropy) else float("nan")
        estimates["n"] = float(len(sub))
        estimates["largest_class_share"] = round(float(weights.max()), 4)

        ent_txt = (
            f"熵={rel_entropy:.3f}（{'分离良好' if rel_entropy >= 0.8 else '分离一般/偏低'}，1=完美分类）"
            if np.isfinite(rel_entropy) else "熵=不可用"
        )
        sel_txt = "（config 指定 k）" if forced_k is not None else "（BIC 选取）"
        sizes_pct = "、".join(f"类{c} {weights[c]:.0%}" for c in range(best_k))
        (d / "lpa_summary.txt").write_text(
            f"潜在剖面分析（LPA，stepmix，measurement=gaussian_diag/类内对角自由方差，"
            f"n_init={n_init}，seed={seed}）\n"
            f"指标（{len(indicators)} 个，已标准化）：{indicators}\n"
            f"选定潜在剖面数 k={best_k} {sel_txt}；样本 n={len(sub)}\n"
            f"混合比例（按规模降序）：{sizes_pct}\n"
            f"BIC={bic_sel:.2f}，AIC={aic_sel:.2f}，{ent_txt}\n\n"
            "各剖面均值/方差（原始量纲，见 class_profiles.csv）——这是定义各潜在剖面的特征。\n\n"
            "⚠ 披露：\n"
            "1) LPA 假定类内服从（多元）正态；本实现用对角协方差（gaussian_diag）"
            "——类内方差按类、按指标自由估计，但忽略类内指标间相关；\n"
            "2) 潜在剖面是统计构念，不是经外部验证的真实分组；\n"
            "3) k 是建模选择，已报告 BIC/AIC 曲线；不同准则可能选不同 k；\n"
            "4) 类别编号无意义（标签切换）——已按规模降序固定（类0 最大）；\n"
            f"5) EM 可能陷入局部最优，已用 n_init={n_init} + 固定 seed={seed} 缓解；\n"
            "6) 指标已标准化（量纲差异敏感）；均值/方差已回算到原始量纲报告。\n",
            encoding="utf-8",
        )
        files.append("lpa_summary.txt")

        summary.append(
            f"{entry.method} 完成（stepmix，gaussian_diag）：{len(indicators)} 个连续指标上识别出 "
            f"k={best_k} 个潜在剖面{sel_txt}；混合比例 {sizes_pct}；{ent_txt}；"
            f"BIC={bic_sel:.1f}（n={len(sub)}）。"
            "⚠ 假定类内正态（对角协方差，忽略类内相关）；剖面为统计构念（非外部验证分组）；"
            f"k 由 BIC 选；标签按规模降序固定；指标已标准化；EM 局部最优已用 n_init={n_init}+seed 缓解。"
        )
        code += [
            "from sklearn.preprocessing import StandardScaler",
            "from stepmix import StepMix  # 潜在剖面分析 LPA = 连续型 LCA（高斯混合，EM）",
            f"# Xs = StandardScaler().fit_transform(df[{indicators}].dropna())",
            f"m = StepMix(n_components={best_k}, measurement='gaussian_diag', n_init={n_init}, random_state={seed})",
            "m.fit(Xs); labels = m.predict(Xs); proba = m.predict_proba(Xs)",
            "print('BIC', m.bic(Xs), 'relative_entropy', m.relative_entropy(Xs))",
            "# get_parameters()['measurement']['means'] / ['covariances'] = per-class profiles",
        ]
    except Exception as err:
        summary.append(f"潜在剖面分析（LPA）执行失败：{err}")
