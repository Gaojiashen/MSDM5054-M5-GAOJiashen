"""Select grouped recency factors for an eight-week same-weekday forecast.

Uses rolling origins through d_1913 only. The four temporal folds each fit on
all earlier origins and validate on the next eight. A rho of zero keeps only
the latest eligible same-weekday observation; rho=1 gives equal x=8 weights.
"""
from __future__ import annotations

import importlib.util
import argparse
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
OUT = ROOT / "outputs" / "evaluation" / "rho_selection"
FIGURES = OUT / "figures"
RHO_VALUES = (0.0, 0.25, 0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
X = 8
HORIZON = 28
PARTITIONS = ("global", "state", "state_category", "store_category")
FOLDS = ((32, 8), (40, 8), (48, 8), (56, 8))
BOOTSTRAP_REPS = 2000
SEED = 5054


def load_metric():
    path = Path(__file__).with_name("02_select_x_wrmsse.py")
    spec = importlib.util.spec_from_file_location("m5_x_selection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load WRMSSE implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grouping(meta: pd.DataFrame, partition: str) -> tuple[np.ndarray, list[str]]:
    if partition == "global":
        return np.zeros(len(meta), dtype=np.int16), ["global"]
    fields = {"state": ["state_id"], "state_category": ["state_id", "cat_id"],
              "store_category": ["store_id", "cat_id"]}[partition]
    keys = pd.MultiIndex.from_frame(meta[fields])
    codes, uniques = pd.factorize(keys, sort=True)
    labels = [" × ".join(map(str, key if isinstance(key, tuple) else (key,))) for key in uniques]
    return codes.astype(np.int16), labels


def weekday_history(train: np.ndarray, origin: int, release: np.ndarray):
    """Return (n_series, <=8) values and availability for each weekday."""
    result = []
    for weekday in range(7):
        last_day = origin - 6 + weekday
        days = last_day - 7 * np.arange(X)
        days = days[days >= 1]
        # Preserve the compact source integer dtype; profile() casts during
        # the weighted sum. Advanced indexing makes a small, bounded copy.
        values = np.asarray(train[:, days - 1])
        available = days[None, :] >= release[:, None]
        result.append((values, available))
    return result


def profile(history, rho: float) -> np.ndarray:
    """Weighted seven-day mean; missing pre-release weeks do not count."""
    daily = np.zeros((history[0][0].shape[0], 7), dtype=np.float64)
    for weekday, (values, available) in enumerate(history):
        # k=0 is the newest observation; rho=0 is the seasonal-naive limit.
        w = np.power(np.float32(rho), np.arange(values.shape[1], dtype=np.float32))
        if rho == 0:
            w[1:] = 0
        usable = available * w[None, :]
        den = usable.sum(axis=1)
        numerator = (values.astype(np.float64, copy=False) * usable).sum(axis=1)
        daily[:, weekday] = np.divide(numerator, den,
                                      out=np.zeros(len(den), np.float64), where=den > 0)
    return daily


def origin_context(m, train, meta, calendar, prices, first_release, aggs, origin):
    """Cache all origin-specific metric arrays and compact seven-day truth stats."""
    actual = np.asarray(train[:, origin:origin + HORIZON], dtype=np.float64)
    scales, weights, coverage = m.scales_and_weights(train, meta, aggs, calendar, prices, origin)
    # The four forecast weeks repeat. Store per-weekday actual first and second
    # moments so scoring needs only seven sparse aggregate products per level.
    stats = []
    for agg in aggs:
        nodes = np.asarray(agg @ actual)
        reshaped = nodes.reshape(nodes.shape[0], 4, 7)
        stats.append((reshaped.mean(axis=1), (reshaped * reshaped).mean(axis=1)))
    release = np.where(first_release <= origin, first_release, 10**9)
    return {"origin": origin, "scales": scales, "weights": weights, "coverage": coverage,
            "stats": stats, "history": weekday_history(train, origin, release)}


def score_profile(context, leaf7: np.ndarray, aggs) -> float:
    score = 0.0
    for lev, agg in enumerate(aggs):
        pred = np.asarray(agg @ leaf7, dtype=np.float64)
        mean, mean_sq = context["stats"][lev]
        mse = np.maximum(0.0, np.mean(pred * pred - 2 * pred * mean + mean_sq, axis=1))
        scale, weight = context["scales"][lev], context["weights"][lev]
        valid = scale > 0
        rms = np.zeros_like(mse)
        rms[valid] = np.sqrt(mse[valid] / scale[valid])
        score += float(np.dot(weight, rms)) / len(aggs)
    return score


def hierarchy_leaf_codes(m, meta: pd.DataFrame) -> list[np.ndarray]:
    """Map every leaf to its node row at each hierarchy level."""
    result = []
    for lev, (_, keys) in enumerate(m.LEVELS):
        if lev == 11:
            result.append(np.arange(len(meta), dtype=np.int32))
        elif not keys:
            result.append(np.zeros(len(meta), dtype=np.int32))
        else:
            codes = pd.factorize(pd.MultiIndex.from_frame(meta[list(keys)]), sort=True)[0]
            result.append(codes.astype(np.int32))
    return result


def node_contributions(context, level: int, pred: np.ndarray, node_ids=None) -> np.ndarray:
    mean, mean_sq = context["stats"][level]
    scale, weight = context["scales"][level], context["weights"][level]
    if node_ids is not None:
        mean, mean_sq = mean[node_ids], mean_sq[node_ids]
        scale, weight = scale[node_ids], weight[node_ids]
    mse = np.maximum(0.0, np.mean(pred * pred - 2 * pred * mean + mean_sq, axis=1))
    valid = scale > 0
    contribution = np.zeros_like(mse)
    contribution[valid] = weight[valid] * np.sqrt(mse[valid] / scale[valid]) / len(context["stats"])
    return contribution


def initial_coordinate_state(context, groups, aggs):
    leaf = np.zeros((len(groups), 7), dtype=np.float64)
    for code in np.unique(groups):
        rows = np.flatnonzero(groups == code)
        hist = [(v[rows], a[rows]) for v, a in context["history"]]
        leaf[rows] = profile(hist, 1.0)
    node_pred = [np.asarray(agg @ leaf, dtype=np.float64) for agg in aggs]
    parts = [node_contributions(context, lev, pred) for lev, pred in enumerate(node_pred)]
    return {"leaf": leaf, "node_pred": node_pred, "parts": parts,
            "score": float(sum(part.sum() for part in parts))}


def score_coordinate_choice(context, state, rows, rho, history, groups, node_codes, aggs):
    """Score one coordinate update by touching only its affected hierarchy nodes."""
    leaf_new = profile([(v[rows], a[rows]) for v, a in history], rho)
    delta_leaf = leaf_new - state["leaf"][rows]
    value = state["score"]
    updates = []
    for lev, agg in enumerate(aggs):
        touched = np.unique(node_codes[lev][rows])
        delta_nodes = np.asarray(agg[touched][:, rows] @ delta_leaf, dtype=np.float64)
        new_pred = state["node_pred"][lev][touched] + delta_nodes
        new_parts = node_contributions(context, lev, new_pred, touched)
        old_parts = state["parts"][lev][touched]
        value += float(new_parts.sum() - old_parts.sum())
        updates.append((lev, touched, delta_nodes, new_parts))
    return value, leaf_new, updates


def apply_coordinate_choice(state, rows, leaf_new, updates, value):
    state["leaf"][rows] = leaf_new
    for lev, touched, delta_nodes, new_parts in updates:
        state["node_pred"][lev][touched] += delta_nodes
        state["parts"][lev][touched] = new_parts
    state["score"] = float(value)


def candidate_profile(context, groups: np.ndarray, vector: np.ndarray, cache: dict) -> np.ndarray:
    result = np.zeros((len(groups), 7), dtype=np.float64)
    for code, rho in enumerate(vector):
        rows = np.flatnonzero(groups == code)
        if not len(rows):
            continue
        key = (context.get("cache_namespace", "validation"), code, float(rho))
        if key not in cache:
            cache[key] = profile([(v[rows], a[rows]) for v, a in context["history"]], float(rho))
        result[rows] = cache[key]
    return result


def fit_partition(contexts, aggs, groups, group_labels, partition, node_codes):
    """Coordinate descent on full-hierarchy mean WRMSSE over training origins."""
    vector = np.ones(len(group_labels), dtype=np.float64)
    states = [initial_coordinate_state(context, groups, aggs) for context in contexts]
    group_rows = [np.flatnonzero(groups == code) for code in range(len(group_labels))]
    # One full sweep is a bounded coordinate-descent search. The equal-weight
    # start gives a stable anchor; repeating sweeps adds substantial runtime.
    for pass_no in range(2):
        changed = False
        for code in range(len(vector)):
            options = []
            for rho in RHO_VALUES:
                trial = vector.copy()
                trial[code] = rho
                values = []
                for context, state in zip(contexts, states):
                    val, _, _ = score_coordinate_choice(context, state, group_rows[code], rho,
                                                         context["history"], groups, node_codes, aggs)
                    values.append(val)
                options.append((float(np.mean(values)), abs(1.0 - rho), rho))
            winner = min(options)[2]
            changed |= winner != vector[code]
            if winner != vector[code]:
                for context, state in zip(contexts, states):
                    value, leaf_new, updates = score_coordinate_choice(
                        context, state, group_rows[code], float(winner), context["history"],
                        groups, node_codes, aggs)
                    apply_coordinate_choice(state, group_rows[code], leaf_new, updates, value)
            vector[code] = winner
            print(f"fit {partition}: pass {pass_no + 1}/2 group {code + 1}/{len(vector)} rho={winner:g}",
                  flush=True)
        print(f"fit {partition}: pass {pass_no + 1}/2, {len(vector)} groups", flush=True)
        if not changed:
            break
    return vector


def block_bootstrap(deltas: pd.DataFrame) -> dict:
    """Paired moving-block bootstrap; sample two contiguous 4-origin blocks per fold."""
    rng = np.random.default_rng(SEED)
    folds = sorted(deltas.fold.unique())
    samples = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    arrays = [deltas.loc[deltas.fold == f].sort_values("origin_d") for f in folds]
    delta_arrays = [part.delta.to_numpy() for part in arrays]
    score_arrays = [part.score.to_numpy() for part in arrays] if "score" in deltas else None
    score_samples = np.empty(BOOTSTRAP_REPS, dtype=np.float64) if score_arrays is not None else None
    for i in range(BOOTSTRAP_REPS):
        picked_delta, picked_score = [], []
        for ix, da in enumerate(delta_arrays):
            starts = rng.integers(0, len(da) - 3, size=2)
            indexes = np.concatenate([np.arange(start, start + 4) for start in starts])
            picked_delta.extend(da[indexes])
            if score_arrays is not None:
                picked_score.extend(score_arrays[ix][indexes])
        samples[i] = np.mean(picked_delta)
        if score_samples is not None:
            score_samples[i] = np.mean(picked_score)
    result = {"mean_delta": float(deltas.delta.mean()),
            "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)),
            "bootstrap_sd_mean_delta": float(samples.std(ddof=1)),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "block_length_origins": 4,
            "note": "Descriptive paired time-block interval; not a claim of statistical significance."}
    if score_samples is not None:
        result["bootstrap_sd_mean_wrmsse"] = float(score_samples.std(ddof=1))
    return result


def select_partition(comparisons: pd.DataFrame):
    """Apply a paired one-SE simplicity rule relative to the best mean model."""
    means = comparisons.groupby("partition", as_index=False).wrmsse.mean().rename(
        columns={"wrmsse": "validation_wrmsse"})
    means["complexity"] = means.partition.map({p: i for i, p in enumerate(PARTITIONS)})
    pivot = comparisons.pivot(index=["fold", "origin_d"], columns="partition", values="wrmsse")
    best_partition = means.loc[means.validation_wrmsse.idxmin(), "partition"]
    rows = []
    bootstrap_rows = []
    for comparator in ("best", "global"):
        baseline_partition = best_partition if comparator == "best" else "global"
        for partition in PARTITIONS:
            paired_rows = []
            for (fold, origin), row in pivot.iterrows():
                # Loss-vs-best is candidate minus best; gain-vs-global is
                # baseline minus candidate so positive values mean improvement.
                delta = (row[partition] - row[baseline_partition] if comparator == "best"
                         else row[baseline_partition] - row[partition])
                paired_rows.append({"fold": fold, "origin_d": origin,
                                    "delta": float(delta), "score": float(row[partition])})
            paired = pd.DataFrame(paired_rows)
            result = block_bootstrap(paired)
            bootstrap_rows.append({"partition": partition, "comparator": comparator, **result})
            if comparator == "best":
                rows.append({"partition": partition,
                             "paired_mean_loss_vs_best": result["mean_delta"],
                             "paired_se_loss_vs_best": result["bootstrap_sd_mean_delta"],
                             "one_se_eligible": result["mean_delta"] <= result["bootstrap_sd_mean_delta"]})
    paired_rule = pd.DataFrame(rows)
    means = means.merge(paired_rule, on="partition", validate="one_to_one")
    eligible = means[means.one_se_eligible]
    chosen = eligible.sort_values("complexity").iloc[0].partition
    means["selected_by_one_se"] = means.partition == chosen
    return means, pd.DataFrame(bootstrap_rows), best_partition, chosen


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def make_figures(search: pd.DataFrame, comparisons: pd.DataFrame, chosen: pd.DataFrame,
                 final_rho: pd.DataFrame, bootstrap: pd.DataFrame):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(search.rho, search.mean_wrmsse, marker="o", color="#386fa4", label="global rho")
    ax.set(xlabel="rho", ylabel="Mean training WRMSSE (lower is better)",
           title="Coordinate search: x fixed at 8")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); save(fig, "01_rho_search")

    ordered = chosen.sort_values("validation_wrmsse")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(ordered))
    colors = ["#e59f3a" if selected else "#457b9d" for selected in ordered.selected_by_one_se]
    ax.scatter(ordered.validation_wrmsse, y, c=colors, s=72, zorder=3)
    low, high = ordered.validation_wrmsse.min(), ordered.validation_wrmsse.max()
    pad = max((high - low) * .25, 0.00003)
    ax.set_xlim(low - pad, high + pad)
    ax.set_yticks(y, ordered.partition)
    for yi, value in zip(y, ordered.validation_wrmsse):
        ax.annotate(f"{value:.6f}", (value, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set(xlabel="Mean rolling validation WRMSSE (lower is better)",
           title="Partition scores; orange marks the paired one-SE choice")
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, "02_partition_comparison")

    forest = bootstrap[(bootstrap.comparator == "global") & (bootstrap.partition != "global")].copy()
    forest = forest.sort_values("mean_delta")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(forest))
    lower = forest.mean_delta - forest.ci95_low
    upper = forest.ci95_high - forest.mean_delta
    ax.errorbar(forest.mean_delta, y, xerr=np.vstack([lower, upper]), fmt="o",
                color="#457b9d", ecolor="#7895aa", capsize=3, markersize=6)
    ax.axvline(0, color="#34495e", linewidth=1)
    ax.set_yticks(y, forest.partition)
    ax.set(xlabel="Global WRMSSE minus candidate (positive favors candidate)",
           title="Paired validation gain vs global rho · 95% time-block intervals")
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, "03_paired_improvements")

    fig, ax = plt.subplots(figsize=(9, 5))
    k = np.arange(X)
    for rho in RHO_VALUES:
        w = np.power(rho, k); w[0] = 1
        if rho == 0: w[1:] = 0
        w /= w.sum()
        ax.plot(k + 1, w, marker="o", label=f"{rho:g}")
    ax.set(xlabel="Past same-weekday observation (1 = newest)", ylabel="Normalized weight",
           title="Candidate recency profiles (all eight weeks available)")
    ax.grid(alpha=.25); ax.legend(ncol=5, frameon=False); fig.tight_layout(); save(fig, "04_weight_profiles")

    order = final_rho.sort_values("rho", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.26 * len(order))))
    colors = ["#e59f3a" if x == 1 else "#24827a" if x == 0 else "#457b9d" for x in order.rho]
    ax.barh(order.group, order.rho, color=colors)
    ax.set(xlim=(0, 1), xlabel="Selected rho", title=f"Final learned rho by {order.partition.iloc[0]} group")
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, "05_final_group_rhos")


def plot_final_holdout(actual: np.ndarray, forecast: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(1, HORIZON + 1)
    ax.plot(x, actual.sum(axis=0), color="#d06a3b", marker="o", ms=3, label="Held-out actual")
    ax.plot(x, forecast.sum(axis=0), color="#187c80", marker="o", ms=3, label="Grouped rho forecast")
    ax.set(xlabel="Day in forecast horizon", ylabel="Total units",
           title="Frozen grouped-rho forecast and held-out total")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); save(fig, "06_final_holdout_total")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-selection", action="store_true",
                        help="reuse rolling_validation_scores.csv and refit only the selected final model")
    args = parser.parse_args()
    m = load_metric()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    train = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if train.shape != (len(meta), 1913):
        raise ValueError("Expected sales through d_1913; held-out d_1914..d_1941 are not used")
    calendar = pd.read_csv(RAW / "calendar.csv", usecols=["d", "wm_yr_wk"])
    calendar["d_num"] = calendar.d.str.removeprefix("d_").astype(int)
    prices = pd.read_csv(RAW / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    aggs = m.hierarchy(meta)
    node_codes = hierarchy_leaf_codes(m, meta)
    first_release = m.first_price_days(meta, prices, calendar)
    origins = tuple(m.ORIGINS)
    # Reuse immutable origin metric contexts across all four fits and validation.
    contexts = {}
    for origin in origins:
        contexts[origin] = origin_context(m, train, meta, calendar, prices, first_release, aggs, origin)
        print(f"prepared origin d_{origin}", flush=True)

    if args.resume_selection:
        score_path = OUT / "rolling_validation_scores.csv"
        if not score_path.is_file():
            raise FileNotFoundError(f"Cannot resume selection; missing {score_path}")
        comparisons = pd.read_csv(score_path)
        expected = len(FOLDS) * len(PARTITIONS) * 8
        if (len(comparisons) != expected or set(comparisons.partition) != set(PARTITIONS)
                or comparisons.duplicated(["fold", "partition", "origin_d"]).any()
                or set(comparisons.fold) != set(range(1, len(FOLDS) + 1))):
            raise ValueError("Existing rolling_validation_scores.csv is incomplete or has unexpected partitions")
        print(f"resume: loaded {len(comparisons)} validation scores; four fold fits skipped", flush=True)
    else:
        rows, rho_rows, coverage_rows = [], [], []
        for fold_no, (n_train, n_valid) in enumerate(FOLDS, start=1):
            train_origins = origins[:n_train]
            valid_origins = origins[n_train:n_train + n_valid]
            fitted = {}
            for partition in PARTITIONS:
                groups, labels = grouping(meta, partition)
                vector = fit_partition([contexts[o] for o in train_origins], aggs, groups, labels,
                                       partition, node_codes)
                fitted[partition] = (groups, labels, vector)
                for label, rho in zip(labels, vector):
                    rho_rows.append({"fold": fold_no, "partition": partition, "group": label, "rho": float(rho)})
            for partition, (groups, labels, vector) in fitted.items():
                for origin in valid_origins:
                    context = contexts[origin]
                    pred = candidate_profile(context, groups, vector, {})
                    score = score_profile(context, pred, aggs)
                    rows.append({"fold": fold_no, "partition": partition, "origin_d": origin,
                                 "wrmsse": score, "train_origins": n_train})
                    coverage_rows.extend({**r, "fold": fold_no} for r in context["coverage"])
            print(f"finished fold {fold_no}: train={n_train}, validation={len(valid_origins)}", flush=True)

        comparisons = pd.DataFrame(rows)
        comparisons.to_csv(OUT / "rolling_validation_scores.csv", index=False)
        pd.DataFrame(rho_rows).to_csv(OUT / "learned_rho_by_fold.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(OUT / "origin_metric_coverage.csv", index=False)

    means, bootstrap, best_partition, chosen_partition = select_partition(comparisons)
    one_se = float(means.loc[means.partition == chosen_partition, "paired_se_loss_vs_best"].iloc[0])
    means.to_csv(OUT / "partition_validation_summary.csv", index=False)
    bootstrap.to_csv(OUT / "paired_bootstrap_deltas.csv", index=False)

    # Refit the selected structure on all 64 rolling origins for deployment.
    groups, labels = grouping(meta, chosen_partition)
    final_rho = fit_partition([contexts[o] for o in origins], aggs, groups, labels,
                               chosen_partition, node_codes)
    final_table = pd.DataFrame({"partition": chosen_partition, "group": labels, "rho": final_rho})
    final_table.to_csv(OUT / "learned_rho_final.csv", index=False)
    # Produce a 28-day leaf forecast from the latest available origin, d_1913.
    final_release = np.where(first_release <= 1913, first_release, 10**9)
    final_context = {"history": weekday_history(train, 1913, final_release),
                     "profile_cache": {}, "cache_namespace": "final_forecast"}
    leaf7 = candidate_profile(final_context, groups, final_rho, {})
    forecast = np.tile(leaf7, (1, 4))
    pred_df = pd.DataFrame(forecast, columns=[f"d_{d}" for d in range(1914, 1942)])
    pred_df.insert(0, "id", meta.item_id.astype(str) + "_" + meta.store_id.astype(str) + "_validation")
    pred_df.to_csv(OUT / "rho_x8_forecast.csv", index=False)

    # The frozen model and forecast are now fixed. Only now read held-out labels
    # to report a final official WRMSSE comparison with the retained sNaive run.
    official = load_module(Path(__file__).with_name("01_snaive_wrmsse.py"), "m5_official_eval")
    actual = official.load_test_actuals(meta)
    cal_full = pd.read_csv(RAW / "calendar.csv", usecols=["d", "date", "wm_yr_wk"])
    cal_full["d_num"] = cal_full.d.str.removeprefix("d_").astype(int)
    cal_train = cal_full.loc[cal_full.d_num.between(1, 1913)].sort_values("d_num").reset_index(drop=True)
    cal_test = cal_full.loc[cal_full.d_num.between(1914, 1941)].sort_values("d_num").reset_index(drop=True)
    relevant_weeks = set(cal_train.loc[cal_train.d_num.between(1886, 1913), "wm_yr_wk"].astype(int))
    official_prices = official.load_weight_window_prices(relevant_weeks)
    final_score, final_levels = official.score_wrmsse(meta, train, actual, forecast,
                                                       cal_train, cal_test, official_prices)
    final_levels.to_csv(OUT / "final_wrmsse_by_level.csv", index=False)
    snaive_summary_path = OUT.parent / "snaive" / "snaive_wrmsse_summary.json"
    snaive_reference = float(json.loads(snaive_summary_path.read_text(encoding="utf-8"))["wrmsse"])
    final_summary = {"grouped_rho_wrmsse": final_score,
                     "snaive_reference_wrmsse": snaive_reference,
                     "difference_snaive_minus_grouped_rho": snaive_reference - final_score,
                     "metric": "01_snaive_wrmsse.py strict official 12-level WRMSSE",
                     "model_frozen_before_actuals": True}
    (OUT / "final_holdout_score.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    plot_final_holdout(actual, forecast)

    search_path = OUT / "rho_search_curve.csv"
    if args.resume_selection and search_path.is_file():
        search = pd.read_csv(search_path)
    else:
        # The descriptive global rho curve is independent of partition choice.
        curve_rows = []
        g, labels0 = grouping(meta, "global")
        for rho in RHO_VALUES:
            trial = np.full(len(labels0), rho)
            values = []
            for origin in origins[:32]:
                ctx = contexts[origin]
                leaf_curve = candidate_profile(ctx, g, trial, {})
                values.append(score_profile(ctx, leaf_curve, aggs))
            curve_rows.append({"partition": "global", "rho": rho,
                               "mean_wrmsse": float(np.mean(values))})
        print("plotted global-rho search curve", flush=True)
        search = pd.DataFrame(curve_rows)
        search.to_csv(search_path, index=False)
    for context in contexts.values():
        context.get("profile_cache", {}).clear()
    make_figures(search, comparisons, means, final_table, bootstrap)
    summary = {"x_fixed": X, "rho_candidates": RHO_VALUES,
        "weight_rule": "w_k proportional to rho^k, k=0 newest; normalize only over post-release available same-weekday values",
        "rho_zero": "sNaive: latest available same-weekday observation", "rho_one": "equal-weight x=8",
        "partition_order": PARTITIONS, "folds": [{"train_origins": n, "validation_origins": v} for n, v in FOLDS],
        "validation_mean_wrmsse": means.to_dict(orient="records"),
        "paired_bootstrap": bootstrap.to_dict(orient="records"), "one_se": one_se,
        "selected_partition": chosen_partition,
        "best_mean_partition": best_partition,
        "one_se_rule": "For each candidate, compute its paired validation loss minus the best-mean partition. Choose the simplest candidate whose mean paired loss is no greater than its own within-fold length-4 moving-block bootstrap SD.",
        "paired_selection_table": means[["partition", "validation_wrmsse", "paired_mean_loss_vs_best", "paired_se_loss_vs_best", "one_se_eligible", "selected_by_one_se"]].to_dict(orient="records"),
        "selection_note": "Paired bootstrap intervals use two contiguous 4-origin blocks within each 8-origin fold; descriptive uncertainty only, not statistical significance.",
        "refit_origins": len(origins), "heldout_actuals": "d_1914..d_1941 are read only after model selection, refit, and forecast save; they do not enter tuning.",
        "optimizer_note": "Two-pass coordinate descent over the rho grid; this is a local search and does not prove a global optimum.",
        "forecast_origin": 1913, "forecast_horizon": [1914, 1941],
        "forecast_note": "Forecast is generated from d_1..d_1913; held-out actuals are read only after fitting and saving the frozen forecast."}
    summary["final_holdout_evaluation"] = final_summary
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
