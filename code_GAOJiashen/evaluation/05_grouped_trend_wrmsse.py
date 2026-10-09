"""Compare constant, linear, and damped grouped trends on rolling M5 origins.

The constant forecast exactly reuses the frozen state × category rho profiles.
Trend slopes use eight complete aggregate weeks and the same group rho weights.
The global damping factor is selected on development origins, then evaluated on
confirmation and audit origins. Held-out d_1914..d_1941 is read only after the
final forecast has been written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "evaluation" / "grouped_trend"
RHO_DIR = ROOT / "outputs" / "evaluation" / "rho_selection"
HORIZON = 28
ORIGINS = tuple(range(121, 1886, 28))
DEV, CONFIRM, AUDIT = ORIGINS[:32], ORIGINS[32:48], ORIGINS[48:]
PHI_GRID = (0.0, .25, .5, .65, .75, .8, .85, .9, .95, 1.0)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_groups(meta: pd.DataFrame, rho_table: pd.DataFrame):
    keys = pd.MultiIndex.from_frame(meta[["state_id", "cat_id"]])
    codes, unique = pd.factorize(keys, sort=True)
    labels = [" × ".join(x) for x in unique.tolist()]
    rho_map = rho_table.set_index("group").rho.to_dict()
    missing = set(labels) - set(rho_map)
    if missing:
        raise ValueError(f"Missing learned rho groups: {sorted(missing)}")
    return codes.astype(np.int16), labels, np.array([rho_map[x] for x in labels], dtype=float)


def estimate_group_trends(train, origin, codes, rhos, leaf_base):
    """WLS on group daily totals / 7 for the last eight complete weeks."""
    # Calendar weeks here mean aligned seven-day blocks ending at each origin.
    # All rolling origins are 7-day aligned; the oldest block starts origin-55.
    days = origin - np.arange(55, -1, -1)
    if days.min() < 1:
        raise ValueError(f"Origin d_{origin} lacks eight complete history weeks")
    groups = int(codes.max()) + 1
    daily = np.asarray(train[:, days - 1], dtype=np.float64)
    totals = np.zeros((groups, 56), dtype=np.float64)
    for g in range(groups):
        totals[g] = daily[codes == g].sum(axis=0)
    daily_by_week = totals.reshape(groups, 8, 7)
    week_means = daily_by_week.mean(axis=2)
    # Newest week has index 7. Match the group's already selected rho.
    weights = np.power(rhos[:, None], np.arange(7, -1, -1)[None, :])
    weights[rhos == 0, :-1] = 0
    weights /= weights.sum(axis=1, keepdims=True)
    a = (weights * week_means).sum(axis=1)
    t = np.arange(-7, 1, dtype=float)
    tbar = (weights * t[None, :]).sum(axis=1)
    denom = (weights * (t[None, :] - tbar[:, None]) ** 2).sum(axis=1)
    b = (weights * (t[None, :] - tbar[:, None]) * (week_means - a[:, None])).sum(axis=1)
    b = np.divide(b, denom, out=np.zeros(groups), where=denom > 0)
    ell0 = a - b * tbar
    # Ratio denominator is precisely the constant leaf forecast's group daily mean.
    base_mean = np.zeros(groups, dtype=float)
    for g in range(groups):
        base_mean[g] = leaf_base[codes == g].sum() / 7.0
    weekday_effect = (weights[:, :, None] *
                      (daily_by_week - week_means[:, :, None])).sum(axis=1)
    # Should agree apart from release truncation / boundary-week details. Use the
    # actual baseline as denominator to guarantee ratio=1 under the constant model.
    return {"a": a, "b": b, "ell0": ell0, "base_mean": base_mean,
            "week_means": week_means, "weights": weights,
            "weekday_effect": weekday_effect}


def trend_levels(fit, phi: float):
    j = np.arange(1, 5, dtype=float)
    if phi == 1:
        return fit["ell0"][:, None] + fit["b"][:, None] * j[None, :]
    factors = np.array([sum(phi ** r for r in range(1, h + 1)) for h in range(1, 5)])
    return fit["ell0"][:, None] + fit["b"][:, None] * factors[None, :]


def clipped_group_weeks(fit, phi: float):
    """Return count and group indices where a raw projected level is negative."""
    raw = trend_levels(fit, phi)
    mask = raw < 0
    return int(mask.sum()), np.flatnonzero(mask.any(axis=1)).tolist()


def trend_forecast(leaf_base, codes, fit, phi):
    levels = trend_levels(fit, phi)
    levels = np.maximum(levels, 0.0)
    ratio = np.divide(levels, fit["base_mean"][:, None],
                      out=np.ones_like(levels), where=fit["base_mean"][:, None] > 0)
    result = leaf_base[:, :, None] * ratio[codes, None, :]
    return result.transpose(0, 2, 1).reshape(len(leaf_base), 28)


def save_final_diagnostics(fit, labels, rhos, chosen_phi, final_model, final_phi, weekdays):
    """Write group-level final-origin components and a compact 3×3 panel figure."""
    groups = len(labels)
    linear = np.maximum(trend_levels(fit, 1.0), 0.0)
    damped = np.maximum(trend_levels(fit, chosen_phi), 0.0)
    constant = np.repeat(fit["base_mean"][:, None], 4, axis=1)
    selected = (constant if final_model == "constant" else linear if final_model == "linear"
                else np.maximum(trend_levels(fit, final_phi), 0.0))
    rows = []
    for g, label in enumerate(labels):
        row = {"group": label, "rho": float(rhos[g]), "a": float(fit["a"][g]),
               "b_per_week": float(fit["b"][g]), "ell0": float(fit["ell0"][g]),
               "constant_baseline_daily_mean": float(fit["base_mean"][g]),
               "development_selected_phi": float(chosen_phi),
               "final_selected_model": final_model}
        row.update({f"history_wk_{k-7}_mean": float(fit["week_means"][g, k]) for k in range(8)})
        row.update({f"history_wk_{k-7}_weight": float(fit["weights"][g, k]) for k in range(8)})
        for j in range(4):
            row[f"constant_week_{j+1}"] = float(constant[g, j])
            row[f"linear_week_{j+1}"] = float(linear[g, j])
            row[f"damped_dev_phi_week_{j+1}"] = float(damped[g, j])
            row[f"selected_forecast_week_{j+1}"] = float(selected[g, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "final_group_trend_levels.csv", index=False)

    weekday_rows = []
    for g, label in enumerate(labels):
        for q, effect in enumerate(fit["weekday_effect"][g]):
            weekday_rows.append({"group": label, "weekday_index": q + 1, "weekday": weekdays[q],
                                 "rho_weighted_group_weekday_effect": float(effect)})
    weekday = pd.DataFrame(weekday_rows)
    weekday.to_csv(OUT / "final_group_weekday_effects.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    hist_x = np.arange(-7, 1)
    future_x = np.arange(1, 5)
    for g, ax in enumerate(axes.flat):
        ax.plot(hist_x, fit["week_means"][g], color="#555b66", marker="o", ms=3,
                linewidth=1.5, label="Observed weekly mean")
        constant_label = "Constant · SELECTED" if final_model == "constant" else "Constant"
        linear_label = "Linear · SELECTED" if final_model == "linear" else "Linear"
        damped_label = (f"Damped · SELECTED (final phi={final_phi:g})" if final_model == "damped_selected"
                        else f"Damped · dev phi={chosen_phi:g}")
        ax.plot(future_x, constant[g], color="#087f8c" if final_model == "constant" else "#667085",
                marker="o", linewidth=2.8 if final_model == "constant" else 1.6,
                label=constant_label)
        ax.plot(future_x, linear[g], color="#d17b0f", marker="o", linestyle="--",
                linewidth=2.8 if final_model == "linear" else 1.6, label=linear_label)
        ax.plot(future_x, damped[g], color="#7656a5", marker="o", linestyle=":",
                linewidth=2.8 if final_model == "damped_selected" else 1.6, label=damped_label)
        ax.axvline(0.5, color="#9aa0a6", linewidth=.8, alpha=.7)
        ax.set_title(f"{labels[g]}  (rho={rhos[g]:g})", fontsize=10)
        ax.grid(alpha=.22)
        ax.set_xticks([-7, -4, 0, 1, 2, 3, 4])
    for ax in axes[-1, :]:
        ax.set_xlabel("Week relative to origin (history ends at 0)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Group daily mean")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .965))
    fig.suptitle(f"d_1913 group trend decomposition · Final selection: {final_model}",
                 y=.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUT / "final_group_trend_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"group_parameter_file": "final_group_trend_levels.csv",
            "weekday_effect_file": "final_group_weekday_effects.csv",
            "panel_figure": "final_group_trend_panels.png",
            "history_week_order": "wk_-7 is oldest and wk_0 is the latest complete week",
            "weekday_effect_sum_by_group": {
                label: float(fit["weekday_effect"][g].sum()) for g, label in enumerate(labels)}}


def scoring_context(metric, train, meta, calendar, prices, aggs, origin):
    actual = np.asarray(train[:, origin:origin + HORIZON], dtype=float)
    scales, weights, coverage = metric.scales_and_weights(train, meta, aggs, calendar, prices, origin)
    stats = []
    for agg in aggs:
        nodes = np.asarray(agg @ actual).reshape(agg.shape[0], 4, 7)
        stats.append(nodes)
    return {"scales": scales, "weights": weights, "stats": stats, "coverage": coverage}


def score(metric, context, forecast, aggs):
    # Full 12-level WRMSSE using the same per-origin zero-scale handling as rho selection.
    total = 0.0
    for lev, agg in enumerate(aggs):
        pred = np.asarray(agg @ forecast, dtype=float).reshape(agg.shape[0], 4, 7)
        actual = context["stats"][lev]
        mse = np.mean((pred - actual) ** 2, axis=(1, 2))
        scale, weight = context["scales"][lev], context["weights"][lev]
        valid = scale > 0
        total += float(np.sum(weight[valid] * np.sqrt(mse[valid] / scale[valid]))) / len(aggs)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-final-holdout", action="store_true",
                        help="stop after rolling selection/validation; do not score d_1914..d_1941")
    parser.add_argument("--reuse-rolling-scores", action="store_true",
                        help="reuse a complete rolling_candidate_scores.csv and rebuild final outputs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metric = load_module(Path(__file__).with_name("02_select_x_wrmsse.py"), "trend_metric")
    rho_table = pd.read_csv(RHO_DIR / "learned_rho_final.csv")
    if set(rho_table.partition) != {"state_category"} or len(rho_table) != 9:
        raise ValueError("Expected the nine frozen state × category rho values")
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    train = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if train.shape != (len(meta), 1913):
        raise ValueError("Expected training matrix d_1..d_1913 only")
    codes, labels, rhos = setup_groups(meta, rho_table)
    rho_leaf, _ = load_module(Path(__file__).with_name("04_select_rho_wrmsse.py"), "rho_impl"), None
    first_release = metric.first_price_days(meta,
        pd.read_csv(RAW / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"]),
        pd.read_csv(RAW / "calendar.csv", usecols=["d", "wm_yr_wk"]).assign(
            d_num=lambda x: x.d.str.removeprefix("d_").astype(int)))
    aggs = metric.hierarchy(meta)
    calendar = pd.read_csv(RAW / "calendar.csv", usecols=["d", "wm_yr_wk"])
    calendar["d_num"] = calendar.d.str.removeprefix("d_").astype(int)
    prices = pd.read_csv(RAW / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    score_path = OUT / "rolling_candidate_scores.csv"
    if args.reuse_rolling_scores:
        if not score_path.is_file():
            raise FileNotFoundError(f"Cannot reuse missing rolling scores: {score_path}")
        scores = pd.read_csv(score_path)
        expected = len(ORIGINS) * (2 + len(PHI_GRID))
        if (len(scores) != expected or set(scores.origin_d) != set(ORIGINS)
                or set(scores.model) != {"constant", "linear", "damped"}
                or scores.duplicated(["origin_d", "model", "phi"]).any()):
            raise ValueError("Existing rolling candidate scores are incomplete or inconsistent")
        print(f"reused {len(scores)} complete rolling candidate scores", flush=True)
    else:
        rows = []
        for origin in ORIGINS:
            ctx = scoring_context(metric, train, meta, calendar, prices, aggs, origin)
            release = np.where(first_release <= origin, first_release, 10**9)
            legacy = rho_leaf.candidate_profile({"history": rho_leaf.weekday_history(train, origin, release),
                                                 "cache_namespace": "trend"}, codes, rhos, {})
            leaf_base = legacy
            fit = estimate_group_trends(train, origin, codes, rhos, leaf_base)
            const = np.tile(leaf_base, (1, 4))
            linear_clip_count, _ = clipped_group_weeks(fit, 1.0)
            for model, pred in [("constant", const), ("linear", trend_forecast(leaf_base, codes, fit, 1.0))]:
                rows.append({"origin_d": origin, "model": model, "phi": 1.0 if model == "linear" else np.nan,
                             "clipped_group_weeks": linear_clip_count if model == "linear" else 0,
                             "wrmsse": score(metric, ctx, pred, aggs)})
            for phi in PHI_GRID:
                pred = trend_forecast(leaf_base, codes, fit, phi)
                clip_count, _ = clipped_group_weeks(fit, phi)
                rows.append({"origin_d": origin, "model": "damped", "phi": phi,
                             "clipped_group_weeks": clip_count, "wrmsse": score(metric, ctx, pred, aggs)})
            print(f"scored d_{origin}", flush=True)
            del ctx, leaf_base, fit, const, legacy
        scores = pd.DataFrame(rows)
        scores.to_csv(score_path, index=False)
    # Select phi only from development; compare three fixed candidates on confirmation.
    dev = scores[scores.origin_d.isin(DEV)]
    phi_means = dev[dev.model == "damped"].groupby("phi").wrmsse.mean()
    chosen_phi = float(phi_means.idxmin())
    summary_rows = []
    for split, origins in (("development", DEV), ("confirmation", CONFIRM), ("audit", AUDIT)):
        part = scores[scores.origin_d.isin(origins)]
        for model, phi in (("constant", None), ("linear", 1.0), ("damped_selected", chosen_phi)):
            sel = part[part.model == ("damped" if model == "damped_selected" else model)]
            if phi is not None and model == "damped_selected":
                sel = sel[sel.phi == phi]
            summary_rows.append({"split": split, "model": model, "phi": phi,
                                 "mean_wrmsse": float(sel.wrmsse.mean())})
    summary = pd.DataFrame(summary_rows)
    confirmation = summary[summary.split == "confirmation"].set_index("model").mean_wrmsse
    final_model = str(confirmation.idxmin())
    if final_model == "damped_selected":
        final_phi = float(scores[scores.model == "damped"].query("origin_d in @DEV or origin_d in @CONFIRM")
                          .groupby("phi").wrmsse.mean().idxmin())
    elif final_model == "linear":
        final_phi = 1.0
    else:
        final_phi = None
    summary.loc[len(summary)] = {"split": "confirmation_selection", "model": final_model,
                                 "phi": final_phi if final_phi is not None else np.nan,
                                 "mean_wrmsse": float(confirmation[final_model])}
    summary.to_csv(OUT / "split_summary.csv", index=False)
    # Plot full rolling history and development phi grid.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, phi, label, color in [("constant", None, "Constant rho baseline", "#333333"),
                                    ("linear", None, "Linear", "#d16b35"),
                                    ("damped", chosen_phi, f"Damped (phi={chosen_phi:g})", "#267d80")]:
        part = scores[scores.model == model]
        if phi is not None:
            part = part[part.phi == phi]
        ax.plot(part.origin_d, part.wrmsse, marker=".", label=label, color=color)
    ax.axvline(ORIGINS[32] - 14, color="#888", linestyle="--", linewidth=1)
    ax.axvline(ORIGINS[48] - 14, color="#888", linestyle=":", linewidth=1)
    ax.set(xlabel="Rolling forecast origin d_", ylabel="12-level WRMSSE",
           title="Grouped trend comparison by rolling origin")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "rolling_trend_comparison.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(phi_means.index, phi_means.values, marker="o", color="#457b9d")
    ax.axvline(chosen_phi, color="#d16b35", linestyle="--", label=f"selected phi={chosen_phi:g}")
    ax.set(xlabel="Damping factor phi", ylabel="Development mean WRMSSE",
           title="Damping grid (first 32 rolling origins)")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "damping_grid.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Freeze the d_1913 forecast after confirmation-based model choice.
    final_release = np.where(first_release <= 1913, first_release, 10**9)
    final_leaf = rho_leaf.candidate_profile({"history": rho_leaf.weekday_history(train, 1913, final_release),
                                             "cache_namespace": "final_trend"}, codes, rhos, {})
    final_fit = estimate_group_trends(train, 1913, codes, rhos, final_leaf)
    day_names = pd.read_csv(RAW / "calendar.csv", usecols=["d", "weekday"])
    day_names["d_num"] = day_names.d.str.removeprefix("d_").astype(int)
    weekdays = day_names.loc[day_names.d_num.between(1907, 1913)].sort_values("d_num").weekday.tolist()
    diagnostic_info = save_final_diagnostics(final_fit, labels, rhos, chosen_phi,
                                             final_model, final_phi, weekdays)
    final_clip_count, final_clip_groups = clipped_group_weeks(final_fit, final_phi) if final_phi is not None else (0, [])
    if final_model == "constant":
        final_forecast = np.tile(final_leaf, (1, 4))
    else:
        final_forecast = trend_forecast(final_leaf, codes, final_fit, final_phi)
    pred = pd.DataFrame(final_forecast, columns=[f"d_{d}" for d in range(1914, 1942)])
    pred.insert(0, "id", meta.item_id.astype(str) + "_" + meta.store_id.astype(str) + "_validation")
    pred.to_csv(OUT / "grouped_trend_forecast.csv", index=False)
    final_summary = {"grouping": "state × category", "rho_source": str(RHO_DIR / "learned_rho_final.csv"),
                     "phi_development": chosen_phi, "final_model_selected_on_confirmation": final_model,
                     "final_phi_refit_on_development_plus_confirmation": final_phi,
                     "development_origins": list(DEV), "confirmation_origins": list(CONFIRM),
                     "audit_origins": list(AUDIT), "constant_is_exact_existing_rho_profile": True,
                     "final_forecast_origin": 1913, "forecast_days": [1914, 1941],
                     "final_origin_diagnostics": diagnostic_info,
                     "negative_group_level_clipping": {
                         "applied_to_trend_levels_before_ratio": True,
                         "rolling_linear_group_weeks_clipped": int(scores[(scores.model == "linear") & scores.origin_d.isin(ORIGINS)].clipped_group_weeks.sum()),
                         "rolling_damped_group_weeks_clipped_by_phi": {
                             str(phi): int(part.clipped_group_weeks.sum())
                             for phi, part in scores[scores.model == "damped"].groupby("phi")},
                         "final_forecast_groups_clipped": [labels[g] for g in final_clip_groups],
                         "final_forecast_group_weeks_clipped": final_clip_count},
                     "heldout_used_for_selection": False,
                     "selection_rule": "Choose phi minimizing mean damped WRMSSE on the first 32 origins. Select constant, linear, or that damped candidate by confirmation mean WRMSSE; report audit scores using the development-selected phi without using audit to select. If damped is selected, refit phi on development plus confirmation origins for the final forecast; this final refitted phi is not separately validated on audit."}
    if not args.skip_final_holdout:
        official = load_module(Path(__file__).with_name("01_snaive_wrmsse.py"), "trend_official")
        actual = official.load_test_actuals(meta)
        cal_full = pd.read_csv(RAW / "calendar.csv", usecols=["d", "date", "wm_yr_wk"])
        cal_full["d_num"] = cal_full.d.str.removeprefix("d_").astype(int)
        cal_train = cal_full.loc[cal_full.d_num.between(1, 1913)].sort_values("d_num").reset_index(drop=True)
        cal_test = cal_full.loc[cal_full.d_num.between(1914, 1941)].sort_values("d_num").reset_index(drop=True)
        relevant = set(cal_train.loc[cal_train.d_num.between(1886, 1913), "wm_yr_wk"].astype(int))
        official_prices = official.load_weight_window_prices(relevant)
        final_score, levels = official.score_wrmsse(meta, train, actual, final_forecast,
                                                    cal_train, cal_test, official_prices)
        levels.to_csv(OUT / "final_wrmsse_by_level.csv", index=False)
        final_summary["strict_final_holdout_wrmsse"] = final_score
    (OUT / "experiment_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
