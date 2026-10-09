"""Select category-specific same-weekday averaging windows by rolling WRMSSE.

Uses only d_1..d_1913. Each origin predicts the next 28 days as one block;
actuals are used only after that origin's forecast has been constructed.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "evaluation" / "x_selection"
CATEGORIES = ("FOODS", "HOBBIES", "HOUSEHOLD")
X_VALUES = (1, 2, 3, 4, 5, 6, 8, 12, 16)
HORIZON = 28
ORIGINS = tuple(range(121, 1886, 28))  # 64 origins; last target day is d_1913.
DEV_ORIGINS = ORIGINS[:32]
CONFIRM_ORIGINS = ORIGINS[32:48]
AUDIT_ORIGINS = ORIGINS[48:]
LEVELS = [
    ("Total", ()), ("State", ("state_id",)), ("Store", ("store_id",)),
    ("Category", ("cat_id",)), ("Department", ("dept_id",)),
    ("State × Category", ("state_id", "cat_id")),
    ("State × Department", ("state_id", "dept_id")),
    ("Store × Category", ("store_id", "cat_id")),
    ("Store × Department", ("store_id", "dept_id")),
    ("Item", ("item_id",)), ("Item × State", ("item_id", "state_id")),
    ("Item × Store", ("item_id", "store_id")),
]
GLOBAL_LEVELS = {0, 1, 2}  # only these nodes combine categories


def hierarchy(meta: pd.DataFrame) -> list[sparse.csr_matrix]:
    result = []
    for i, (_, keys) in enumerate(LEVELS):
        if i == 11:
            result.append(sparse.identity(len(meta), format="csr", dtype=np.float64))
        else:
            codes = np.zeros(len(meta), dtype=np.int32) if not keys else pd.factorize(
                pd.MultiIndex.from_frame(meta[list(keys)]), sort=True
            )[0]
            n = 1 if not keys else int(codes.max()) + 1
            result.append(sparse.csr_matrix((np.ones(len(meta)), (codes, np.arange(len(meta)))), shape=(n, len(meta))))
    if sum(a.shape[0] for a in result) != 42840:
        raise ValueError("Expected the 42,840 nodes in the M5 hierarchy")
    return result


def category_layout(meta: pd.DataFrame, aggs):
    """Precompute category masks and sliced aggregation matrices once per run."""
    cat_labels = meta.cat_id.to_numpy()
    categories = {c: cat_labels == c for c in CATEGORIES}
    node_masks = {}
    for lev, (_, keys) in enumerate(LEVELS):
        if lev in GLOBAL_LEVELS:
            continue
        if lev == 11:  # Level 12 preserves metadata order because it is identity.
            node_cat = cat_labels
        else:
            codes = (np.zeros(len(meta), dtype=np.int32) if not keys else
                     pd.factorize(pd.MultiIndex.from_frame(meta[list(keys)]), sort=True)[0])
            first_leaf = np.full(aggs[lev].shape[0], len(meta), dtype=int)
            np.minimum.at(first_leaf, codes, np.arange(len(meta)))
            if np.any(first_leaf == len(meta)):
                raise ValueError(f"Could not map every node to a leaf at {LEVELS[lev][0]}")
            node_cat = cat_labels[first_leaf]
        node_masks[lev] = {c: node_cat == c for c in CATEGORIES}
    rows = {c: np.flatnonzero(categories[c]) for c in CATEGORIES}
    sliced_aggs = {c: [agg[:, rows[c]].tocsr() for agg in aggs] for c in CATEGORIES}
    return categories, node_masks, rows, sliced_aggs


def scales_and_weights(train: np.ndarray, meta: pd.DataFrame, aggs, calendar: pd.DataFrame,
                       prices: pd.DataFrame, origin: int):
    """Return node scales and comparable weights using data through origin.

    A newly active node can have positive revenue but no nonzero training
    difference yet, making its RMSSE scale zero. The official expression is
    undefined for that node. For rolling model selection only, exclude such
    nodes for every candidate at this origin and renormalize the remaining
    revenue weights *within the same hierarchy level*. Report the excluded
    original weight so the user can see how far a window is from exact WRMSSE.
    """
    start = max(1, origin - HORIZON + 1)
    recent = train[:, start - 1:origin].astype(np.float64)
    weeks = set(calendar.loc[calendar.d_num.between(start, origin), "wm_yr_wk"].astype(int))
    p = prices.loc[prices.wm_yr_wk.isin(weeks)]
    lookup = p.merge(calendar.loc[calendar.d_num.between(start, origin), ["wm_yr_wk", "d_num"]], on="wm_yr_wk")
    keys = meta[["item_id", "store_id"]].copy()
    keys["row"] = np.arange(len(meta))
    joined = keys.merge(lookup[["item_id", "store_id", "d_num", "sell_price"]], on=["item_id", "store_id"], how="left")
    pmat = np.zeros_like(recent)
    valid = joined.d_num.notna()
    rr = joined.loc[valid, "row"].to_numpy(int)
    cc = joined.loc[valid, "d_num"].to_numpy(int) - start
    pmat[rr, cc] = joined.loc[valid, "sell_price"].to_numpy(float)
    if ((pmat == 0) & (recent > 0)).any():
        raise ValueError(f"Positive sale lacks price in weight window ending d_{origin}")
    revenue_leaf = (recent * pmat).sum(axis=1)
    scales, weights, coverage_rows = [], [], []
    hist = train[:, :origin].astype(np.float64)
    for agg in aggs:
        scale = np.zeros(agg.shape[0], dtype=np.float64)
        for begin in range(0, agg.shape[0], 512):
            end = min(begin + 512, agg.shape[0])
            block = np.asarray(agg[begin:end] @ hist)
            nz = block != 0
            has = nz.any(axis=1)
            first_idx = np.argmax(nz, axis=1)
            diff = np.diff(block, axis=1)
            prefix = np.concatenate([np.zeros((len(block), 1)), np.cumsum(diff * diff, axis=1)], axis=1)
            lens = (origin - 1) - first_idx
            ss = prefix[:, -1] - prefix[np.arange(len(block)), first_idx]
            scale[begin:end] = np.divide(ss, lens, out=np.zeros_like(ss), where=has & (lens > 0))
        rev = np.asarray(agg @ revenue_leaf[:, None]).ravel()
        total_revenue = float(rev.sum())
        if total_revenue <= 0:
            raise ValueError(f"No revenue to weight level {len(scales) + 1} at d_{origin}")
        valid_scale = scale > 0
        valid_revenue = float(rev[valid_scale].sum())
        if valid_revenue <= 0:
            raise ValueError(f"No scorable revenue at level {len(scales) + 1}, d_{origin}")
        weight = np.zeros_like(rev)
        weight[valid_scale] = rev[valid_scale] / valid_revenue
        coverage_rows.append({
            "origin_d": origin,
            "level": LEVELS[len(scales)][0],
            "zero_scale_nodes": int((~valid_scale).sum()),
            "excluded_revenue_share": float(1 - valid_revenue / total_revenue),
        })
        scales.append(scale)
        weights.append(weight)
    return scales, weights, coverage_rows


def first_price_days(meta: pd.DataFrame, prices: pd.DataFrame,
                     calendar: pd.DataFrame) -> np.ndarray:
    """First listed day for each product-store series, in metadata row order.

    This is computed once for speed. At a rolling origin, a date later than
    that origin is masked out, exactly as an as-of price lookup would do.
    """
    first_week = prices.groupby(["item_id", "store_id"], sort=False).wm_yr_wk.min()
    week_start = calendar.groupby("wm_yr_wk").d_num.min()
    first_day = first_week.map(week_start)
    key = pd.MultiIndex.from_frame(meta[["item_id", "store_id"]])
    return first_day.reindex(key).fillna(10**9).to_numpy(dtype=np.int32)


def candidate_leaf(train: np.ndarray, rows: np.ndarray, origin: int, x: int,
                   release: np.ndarray) -> np.ndarray:
    """Forecast seven weekdays, then repeat them across the 28-day horizon.

    A pre-release historical day is not counted as a zero sale. Observed zeros
    after release remain in the average. All four future weeks use the same
    information available at the origin, so their weekday pattern repeats.
    """
    week = np.zeros((len(rows), 7), dtype=np.float64)
    for h in range(7):
        # Day origin-6+h is seven days before future day origin+1+h.
        last = origin - 6 + h
        days = last - 7 * np.arange(x)
        days = days[(days >= 1)]
        if not len(days):
            continue
        # A candidate historical week counts only after release. This includes
        # observed zero-sales weeks because zero remains an ordinary value.
        values = train[np.ix_(rows, days - 1)].astype(np.float64)
        available = (days[None, :] >= release[rows, None])
        counts = available.sum(axis=1)
        sums = np.where(available, values, 0.0).sum(axis=1)
        week[:, h] = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.tile(week, (1, HORIZON // 7))


def evaluate_origin(meta, train, calendar, prices, first_release, aggs, layout,
                    origin, vectors):
    """Score x-vectors at one origin, using identical scorable nodes for all."""
    if not np.isfinite(train[:, :origin]).all() or np.any(train[:, :origin] < 0):
        raise ValueError(f"Sales history through d_{origin} must be finite and nonnegative")
    target = np.arange(origin + 1, origin + HORIZON + 1)
    actual = train[:, target - 1].astype(np.float64)
    scales, weights, coverage_rows = scales_and_weights(
        train, meta, aggs, calendar, prices, origin
    )
    # Future first-price dates cannot be known at this origin. Masking them to
    # an unreachable day makes those product-store forecasts zero.
    release = np.where(first_release <= origin, first_release, 10**9)
    cats, node_masks, cat_rows, cat_aggs = layout
    # Store category-specific aggregates once per x. Most hierarchy nodes belong
    # to one category; only Total/State/Store need joint vector scoring.
    cat_nodes = {c: {} for c in CATEGORIES}
    global_nodes = {c: {} for c in CATEGORIES}
    actual_aggregates = [np.asarray(agg @ actual) for agg in aggs]
    # Development needs all nine x values per category. A later comparison of
    # only a few fixed vectors calculates only their requested x values.
    needed_x = {
        c: tuple(sorted({vector[ci] for vector in vectors}))
        for ci, c in enumerate(CATEGORIES)
    }
    for c in CATEGORIES:
        rows = cat_rows[c]
        for x in needed_x[c]:
            leaf = candidate_leaf(train, rows, origin, x, release)
            for lev, agg in enumerate(cat_aggs[c]):
                block = np.asarray(agg @ leaf)
                if lev in GLOBAL_LEVELS:
                    global_nodes[c][(lev, x)] = block
                else:
                    cat_nodes[c][(lev, x)] = block[node_masks[lev][c]]
    # Category-contained levels: precompute full level score for each category/x.
    cat_loss = {c: {} for c in CATEGORIES}
    for c in CATEGORIES:
        for x in needed_x[c]:
            loss = 0.0
            for lev, _ in enumerate(LEVELS):
                if lev in GLOBAL_LEVELS:
                    continue
                # actual and forecast rows corresponding to that category's nodes
                agg = aggs[lev]
                keep = node_masks[lev][c]
                pred = cat_nodes[c][(lev, x)]
                truth = actual_aggregates[lev][keep]
                mse = np.mean((pred - truth) ** 2, axis=1)
                valid = scales[lev][keep] > 0
                rms = np.zeros_like(mse)
                rms[valid] = np.sqrt(mse[valid] / scales[lev][keep][valid])
                loss += float((weights[lev][keep] * rms).sum()) / len(LEVELS)
            cat_loss[c][x] = loss
    joint = np.zeros(len(vectors))
    for lev in GLOBAL_LEVELS:
        truth = actual_aggregates[lev]
        for vi, vector in enumerate(vectors):
            pred = sum(global_nodes[c][(lev, x)] for c, x in zip(CATEGORIES, vector))
            mse = np.mean((pred - truth) ** 2, axis=1)
            valid = scales[lev] > 0
            rms = np.zeros_like(mse)
            rms[valid] = np.sqrt(mse[valid] / scales[lev][valid])
            joint[vi] += float((weights[lev] * rms).sum()) / len(LEVELS)
    total = joint.copy()
    for vi, vector in enumerate(vectors):
        for c, x in zip(CATEGORIES, vector):
            total[vi] += cat_loss[c][x]
    return total, coverage_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    train = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if train.shape != (len(meta), 1913):
        raise ValueError("Expected the d_1..d_1913 processed sales matrix")
    calendar = pd.read_csv(RAW / "calendar.csv", usecols=["d", "wm_yr_wk"])
    calendar["d_num"] = calendar.d.str.removeprefix("d_").astype(int)
    prices = pd.read_csv(RAW / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    aggs = hierarchy(meta)
    layout = category_layout(meta, aggs)
    first_release = first_price_days(meta, prices, calendar)
    vectors = list(itertools.product(X_VALUES, repeat=3))
    score_rows = []
    coverage_rows = []
    # Search all combinations on development origins only.
    for origin in DEV_ORIGINS:
        origin_scores, origin_coverage = evaluate_origin(
            meta, train, calendar, prices, first_release, aggs, layout, origin, vectors
        )
        coverage_rows.extend(origin_coverage)
        score_rows.extend((origin, *vector, float(score)) for vector, score in zip(vectors, origin_scores))
        print(f"finished origin d_{origin}")
    score_df = pd.DataFrame(score_rows, columns=["origin_d", *CATEGORIES, "wrmsse"])
    means = score_df[score_df.origin_d.isin(DEV_ORIGINS)].groupby(list(CATEGORIES), as_index=False).wrmsse.mean()
    means = means.sort_values(["wrmsse", *CATEGORIES]).reset_index(drop=True)
    best = means.iloc[0]
    chosen = tuple(int(best[c]) for c in CATEGORIES)
    diagonal = score_df[score_df.FOODS == score_df.HOBBIES].query("HOBBIES == HOUSEHOLD")
    diagonal_means = diagonal[diagonal.origin_d.isin(DEV_ORIGINS)].groupby("FOODS").wrmsse.mean()
    shared_x = int(diagonal_means.idxmin())
    shared_vector = (shared_x,) * 3
    naive_vector = (1, 1, 1)
    confirmation_vectors = list(dict.fromkeys((chosen, shared_vector, naive_vector)))
    confirmation_rows = []
    for origin in CONFIRM_ORIGINS:
        origin_scores, origin_coverage = evaluate_origin(
            meta, train, calendar, prices, first_release, aggs, layout,
            origin, confirmation_vectors
        )
        coverage_rows.extend(origin_coverage)
        confirmation_rows.extend((origin, *vector, float(score)) for vector, score in zip(confirmation_vectors, origin_scores))
        print(f"finished confirmation origin d_{origin}")
    confirmation_df = pd.DataFrame(confirmation_rows, columns=["origin_d", *CATEGORIES, "wrmsse"])
    confirmation_mean = confirmation_df.groupby(list(CATEGORIES)).wrmsse.mean().to_dict()
    joint_confirmation = float(confirmation_mean[chosen])
    shared_confirmation = float(confirmation_mean[shared_vector])
    naive_confirmation = float(confirmation_mean[naive_vector])
    # Lowest confirmation mean wins. Ties prefer x=1, then shared-x, then joint.
    gate_options = [
        (naive_confirmation, 0, "seasonal_naive_x1", naive_vector),
        (shared_confirmation, 1, "shared_x", shared_vector),
        (joint_confirmation, 2, "joint_vector", chosen),
    ]
    _, _, adopted_name, adopted = min(gate_options)

    audit_comparisons = list(dict.fromkeys((adopted, chosen, shared_vector, (1, 1, 1))))
    audit_rows = []
    for origin in AUDIT_ORIGINS:
        origin_scores, origin_coverage = evaluate_origin(
            meta, train, calendar, prices, first_release, aggs, layout,
            origin, audit_comparisons
        )
        coverage_rows.extend(origin_coverage)
        audit_rows.extend((origin, *vector, float(score)) for vector, score in zip(audit_comparisons, origin_scores))
        print(f"finished audit origin d_{origin}")
    audit_df = pd.DataFrame(audit_rows, columns=["origin_d", *CATEGORIES, "wrmsse"])
    score_df = pd.concat([score_df, confirmation_df, audit_df], ignore_index=True)
    score_df.to_csv(OUT / "rolling_origin_scores.csv", index=False)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(OUT / "origin_metric_coverage.csv", index=False)
    split_rows = []
    split_comparisons = {
        "development": (("development_selected_joint_vector", chosen),
                        ("development_selected_shared_x", shared_vector),
                        ("seasonal_naive_x1", (1, 1, 1))),
        "confirmation": (("development_selected_joint_vector", chosen),
                         ("development_selected_shared_x", shared_vector),
                         ("seasonal_naive_x1", naive_vector)),
        "audit": (("adopted_choice", adopted), ("development_selected_joint_vector", chosen),
                  ("development_selected_shared_x", shared_vector), ("seasonal_naive_x1", (1, 1, 1))),
    }
    for name, origins in (("development", DEV_ORIGINS), ("confirmation", CONFIRM_ORIGINS), ("audit", AUDIT_ORIGINS)):
        for method, vector in split_comparisons[name]:
            mask = (score_df.origin_d.isin(origins) &
                    (score_df[list(CATEGORIES)] == pd.Series(vector, index=CATEGORIES)).all(axis=1))
            part = score_df.loc[mask]
            split_rows.append({"split": name, "method": method, "origins": len(origins),
                               "FOODS_x": vector[0], "HOBBIES_x": vector[1], "HOUSEHOLD_x": vector[2],
                               "mean_wrmsse": float(part.wrmsse.mean())})
    summary = {"categories_order": CATEGORIES, "candidate_x": X_VALUES, "origin_days": ORIGINS,
               "split_origin_days": {"development": DEV_ORIGINS, "confirmation": CONFIRM_ORIGINS, "audit": AUDIT_ORIGINS},
               "selected_vector_on_development_only": dict(zip(CATEGORIES, chosen)),
               "development_joint_mean_wrmsse": float(best.wrmsse),
               "development_best_shared_x": shared_x,
               "development_best_shared_x_wrmsse": float(diagonal_means.min()),
        "confirmation_joint_mean_wrmsse": joint_confirmation,
        "confirmation_shared_x_mean_wrmsse": shared_confirmation,
        "confirmation_snaive_x1_mean_wrmsse": naive_confirmation,
        "confirmation_gate": "adopt the lowest confirmation mean among dev joint, dev shared-x, and x=1; ties prefer x=1, then shared-x, then joint",
               "adopted_choice": adopted_name,
               "adopted_vector": dict(zip(CATEGORIES, adopted)),
               "holdout_scores": split_rows,
               "rolling_score_note": "Zero-scale nodes are excluded at each origin; remaining dollar weights are renormalized per level. See origin_metric_coverage.csv. When excluded shares are zero, the score is exact WRMSSE.",
               "maximum_excluded_revenue_share": float(coverage_df.excluded_revenue_share.max()),
               "exclusion": "d_1914..d_1941 are not read or used"}
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    means.to_csv(OUT / "development_vector_means.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "selected_vector_split_scores.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
