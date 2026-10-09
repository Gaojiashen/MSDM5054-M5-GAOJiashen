"""Diagnose zero-day errors in the frozen calendar+SNAP rolling forecasts.

Uses each validation fold's saved rho and ``both`` coefficients, reconstructs
leaf forecasts with the same formula as 06_calendar_effect_wrmsse.py, and
compares them with the corresponding rho-only reference. No parameters are fit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
RHO_DIR = ROOT / "outputs" / "evaluation" / "rho_selection"
CAL_DIR = ROOT / "outputs" / "evaluation" / "calendar_effect"
OUT = ROOT / "outputs" / "evaluation" / "sparse_zero_diagnostic"
ORIGINS = tuple(range(121, 1886, 28))
FOLDS = ((32, 8), (40, 8), (48, 8), (56, 8))
GROUP_FIELDS = ("state_id", "cat_id")
EVENTS = ("Pesach End", "OrthodoxEaster", "Cinco De Mayo", "Mother's day")
HORIZON = 28
SPARSE_BANDS = ((0.0, 0.0, "0"), (0.0, .05, "(0,.05]"), (.05, .1, "(.05,.1]"),
                (.1, .2, "(.1,.2]"), (.2, .5, "(.2,.5]"), (.5, 1.0, "(.5,1]"),
                (1.0, np.inf, ">1"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_matrix(calendar: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        calendar.event_name_1.eq(name).to_numpy() | calendar.event_name_2.eq(name).to_numpy()
        for name in EVENTS
    ]).astype(np.float64)


def cohort_labels(origin: int, first_release: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparsity uses only the release-aware 56 calendar days ending at origin."""
    start = max(1, origin - 55)
    days = np.arange(start, origin + 1)
    released = first_release <= origin
    usable = days[None, :] >= first_release[:, None]
    counts = usable.sum(axis=1)
    hist = np.asarray(train[:, days - 1], dtype=np.float64)
    zeros = ((hist == 0) & usable).sum(axis=1)
    zero_rate = np.divide(zeros, counts, out=np.full(len(first_release), np.nan), where=counts > 0)
    labels = np.full(len(first_release), "no_history", dtype=object)
    labels[released & (counts < 28)] = "young_<28d"
    mature = released & (counts >= 28)
    labels[mature & (zero_rate < .5)] = "zero_rate_<50%"
    labels[mature & (zero_rate >= .5) & (zero_rate < .8)] = "zero_rate_50-80%"
    labels[mature & (zero_rate >= .8)] = "zero_rate_>=80%"
    return labels, zero_rate


def leaf_local(origin, rhos, beta, gamma, train, meta, first_release, cal, group_rows):
    """Reproduce 06's ``leaf_local`` for model=both and model=baseline."""
    state_by_group = [meta.iloc[rows[0]].state_id for rows in group_rows]
    state_snap_col = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    target_days = np.arange(origin + 1, origin + 29)
    target_events = event_matrix(cal.loc[target_days - 1])
    target_snap = np.stack([
        cal.loc[target_days - 1, state_snap_col[s]].to_numpy(float) for s in state_by_group
    ])
    result_both = np.zeros((len(meta), HORIZON), dtype=np.float64)
    result_base = np.zeros_like(result_both)
    wd = np.arange(HORIZON) % 7
    for g, rows in enumerate(group_rows):
        for weekday in range(7):
            last = origin - 6 + weekday
            days = last - 7 * np.arange(8)
            days = days[days >= 1]
            vals = np.asarray(train[np.ix_(rows, days - 1)], dtype=float)
            available = days[None, :] >= first_release[rows, None]
            snap_hist = cal.loc[days - 1, state_snap_col[state_by_group[g]]].to_numpy(float)
            holiday = cal.loc[days - 1, "holiday"].to_numpy(bool)
            weights = np.power(np.float32(rhos[g]), np.arange(len(days), dtype=np.float32))
            if rhos[g] == 0 and len(weights):
                weights[1:] = 0
            keep = ~holiday
            idx = np.flatnonzero(wd == weekday)
            for out, b, gam, skip in ((result_both, beta[g], gamma, True),
                                      (result_base, 0.0, np.zeros(4), False)):
                w = weights * keep if skip else weights
                aw = available * w[None, :]
                den = aw.sum(axis=1)
                corrected = vals * np.exp(-b * snap_hist)[None, :]
                profile = np.divide((corrected * aw).sum(axis=1), den,
                                    out=np.zeros(len(rows)), where=den > 0)
                factor = np.exp(b * target_snap[g, idx] + target_events[idx] @ gam)
                out[rows[:, None], idx[None, :]] = profile[:, None] * factor[None, :]
    return result_both, result_base


def metric_row(y: np.ndarray, p: np.ndarray, origin: int, cohort: str, model: str) -> dict:
    if len(y) == 0:
        raise ValueError("metric_row requires at least one released target cell")
    zero = y == 0; nz = ~zero; err2 = (p-y)**2
    return {"origin_d": origin, "cohort": cohort, "model": model,
            "observations": len(y), "actual_zero_rate": float(zero.mean()),
            "zero_day_observations": int(zero.sum()),
            "zero_day_pred_gt_0_count": int((p[zero] > 0).sum()),
            "zero_day_pred_gt_0_5_count": int((p[zero] > .5).sum()),
            "zero_day_pred_gt_1_count": int((p[zero] > 1).sum()),
            "zero_day_pred_gt_0_rate": float((p[zero] > 0).mean()) if zero.any() else np.nan,
            "zero_day_pred_gt_0_5_rate": float((p[zero] > .5).mean()) if zero.any() else np.nan,
            "zero_day_pred_gt_1_rate": float((p[zero] > 1).mean()) if zero.any() else np.nan,
            "zero_day_mean_squared_loss": float(err2[zero].mean()) if zero.any() else np.nan,
            "zero_day_squared_loss": float(err2[zero].sum()),
            "zero_day_squared_loss_share": float(err2[zero].sum()/err2.sum()) if err2.sum() else 0.0,
            "nonzero_day_observations": int(nz.sum()),
            "nonzero_day_pred_eq_0_count": int((p[nz] == 0).sum()),
            "nonzero_day_underpred_count": int((p[nz] < y[nz]).sum()),
            "nonzero_day_actual_sum": float(y[nz].sum()),
            "nonzero_day_pred_sum": float(p[nz].sum()),
            "nonzero_day_pred_eq_0_rate": float((p[nz] == 0).mean()) if nz.any() else np.nan,
            "nonzero_day_underpred_rate": float((p[nz] < y[nz]).mean()) if nz.any() else np.nan,
            "nonzero_day_mean_actual": float(y[nz].mean()) if nz.any() else np.nan,
            "nonzero_day_mean_forecast": float(p[nz].mean()) if nz.any() else np.nan,
            "nonzero_day_mean_squared_loss": float(err2[nz].mean()) if nz.any() else np.nan,
            "nonzero_day_squared_loss": float(err2[nz].sum()),
            "always_zero_squared_loss": float(np.square(y).sum()),
            "total_squared_loss": float(err2.sum()),
            "all_day_mean_squared_loss": float(err2.mean())}


def sparse_diagnostics(y: np.ndarray, p: np.ndarray, origin: int, model: str) -> tuple[list[dict], dict]:
    """Unconditional forecast calibration and high-sales detection for one origin."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    err2 = np.square(y - p); high = y >= 3
    rows = []
    for lo, hi, label in SPARSE_BANDS:
        mask = (p == 0) if label == "0" else ((p > lo) & (p <= hi))
        rows.append({"origin_d": origin, "model": model, "forecast_band": label,
                     "observations": int(mask.sum()),
                     "mean_actual": float(y[mask].mean()) if mask.any() else np.nan,
                     "mean_forecast": float(p[mask].mean()) if mask.any() else np.nan,
                     "rate_y_ge_1": float((y[mask] >= 1).mean()) if mask.any() else np.nan,
                     "rate_y_ge_3": float((y[mask] >= 3).mean()) if mask.any() else np.nan})
    threshold = float(np.quantile(p, .9)) if len(p) else np.nan
    top = p >= threshold
    summary = {"origin_d": origin, "model": model, "observations": len(y),
               "actual_sum_all": float(y.sum()), "forecast_sum_all": float(p.sum()),
               "mean_actual_all": float(y.mean()) if len(y) else np.nan,
               "mean_forecast_all": float(p.mean()) if len(p) else np.nan,
               "y_ge_1_count": int((y >= 1).sum()),
               "rate_y_ge_1_all": float((y >= 1).mean()) if len(y) else np.nan,
               "rate_y_ge_3_all": float(high.mean()) if len(y) else np.nan,
               "y_ge_3_count": int(high.sum()), "y_ge_3_rate": float(high.mean()) if len(y) else np.nan,
               "y_ge_3_sse": float(err2[high].sum()),
               "total_sse": float(err2.sum()),
               "y_ge_3_sse_share": float(err2[high].sum()/err2.sum()) if err2.sum() else 0.0,
               "mean_forecast_on_y_ge_3": float(p[high].mean()) if high.any() else np.nan,
               "forecast_sum_on_y_ge_3": float(p[high].sum()),
               "top_decile_threshold": threshold, "top_decile_count": int(top.sum()),
               "top_decile_y_ge_3_count": int((high & top).sum()),
               "top_decile_y_ge_3_rate": float(high[top].mean()) if top.any() else np.nan,
               "top_decile_mean_actual": float(y[top].mean()) if top.any() else np.nan,
               "top_decile_rate_lift_vs_all": (float(high[top].mean()/high.mean())
                                                if top.any() and high.any() else np.nan)}
    return rows, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    train = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if train.shape != (len(meta), 1913):
        raise ValueError("Expected sales through d_1913; this diagnostic must not read holdout actuals")
    cal = pd.read_csv(RAW / "calendar.csv")
    cal["d_num"] = cal.d.str.removeprefix("d_").astype(int)
    cal["holiday"] = ((cal.event_name_1.notna() & cal.event_name_1.ne("")) |
                      (cal.event_name_2.notna() & cal.event_name_2.ne("")))
    metric_x = load_module(Path(__file__).with_name("02_select_x_wrmsse.py"), "sparse_metric")
    prices = pd.read_csv(RAW / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    first_release = metric_x.first_price_days(meta, prices, cal)
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(meta[list(GROUP_FIELDS)]), sort=True)
    if len(uniques) != 9:
        raise ValueError(f"Expected nine state × category groups, got {len(uniques)}")
    group_rows = [np.flatnonzero(codes == i) for i in range(9)]
    rho_fold = pd.read_csv(RHO_DIR / "learned_rho_by_fold.csv")
    coef_fold = pd.read_csv(CAL_DIR / "calendar_coefficients_by_fold.csv")

    # Verify the copied leaf formula against already-saved final forecasts.
    rho_final = pd.read_csv(RHO_DIR / "learned_rho_final.csv")
    group_labels = [" × ".join(map(str, key)) for key in uniques]
    final_rhos = np.array([float(rho_final.set_index("group").loc[label, "rho"])
                           for label in group_labels])
    coef_final = pd.read_csv(CAL_DIR / "final_calendar_coefficients.csv")
    both_final = coef_final[coef_final.model == "both"].set_index("parameter")
    final_beta = np.array([float(both_final.loc[f"beta:{label}", "log_multiplier"])
                           for label in group_labels])
    final_gamma = np.array([float(both_final.loc[f"gamma:{event}", "log_multiplier"])
                            for event in EVENTS])
    final_both, final_base = leaf_local(1913, final_rhos, final_beta, final_gamma,
                                        train, meta, first_release, cal, group_rows)
    saved_both = pd.read_csv(CAL_DIR / "forecast_both.csv").iloc[:, 1:].to_numpy(float)
    saved_base = pd.read_csv(CAL_DIR / "forecast_baseline.csv").iloc[:, 1:].to_numpy(float)
    parity_both = float(np.max(np.abs(final_both - saved_both)))
    parity_base = float(np.max(np.abs(final_base - saved_base)))
    if parity_both > 1e-5 or parity_base > 1e-5:
        raise AssertionError(f"saved forecast parity failed: both={parity_both:g}, baseline={parity_base:g}")

    summaries = []
    sparse_band_rows = []; sparse_summary_rows = []; sparse_cell_data = []
    for fold_no, (ntrain, nvalid) in enumerate(FOLDS, 1):
        origins = ORIGINS[ntrain:ntrain+nvalid]
        rho_part = rho_fold[(rho_fold.fold == fold_no) & (rho_fold.partition == "state_category")]
        rhos = np.array([float(rho_part.set_index("group").loc[label, "rho"]) for label in group_labels])
        cp = coef_fold[(coef_fold.fold == fold_no) & (coef_fold.model == "both")].set_index("parameter")
        beta = np.array([float(cp.loc[f"beta:{label}", "log_multiplier"]) for label in group_labels])
        gamma = np.array([float(cp.loc[f"gamma:{event}", "log_multiplier"]) for event in EVENTS])
        for origin in origins:
            pred, base = leaf_local(origin, rhos, beta, gamma, train, meta, first_release, cal, group_rows)
            target = np.arange(origin + 1, origin + 29)
            actual = np.asarray(train[:, target - 1], dtype=float)
            cohorts, hist_zero_rate = cohort_labels(origin, first_release, train)
            target_released = first_release[:, None] <= target[None, :]
            for model, forecast in (("calendar_snap", pred), ("rho_baseline", base)):
                sparse_cells = target_released & (cohorts == "zero_rate_>=80%")[:, None]
                if sparse_cells.any():
                    bands, diagnostic = sparse_diagnostics(actual[sparse_cells], forecast[sparse_cells], origin, model)
                    diagnostic["fold"] = fold_no
                    sparse_band_rows.extend(bands); sparse_summary_rows.append(diagnostic)
                    sparse_cell_data.append({"fold": fold_no, "origin_d": origin, "model": model,
                                             "y": actual[sparse_cells], "p": forecast[sparse_cells]})
                for cohort in ["all", "no_history", "young_<28d", "zero_rate_<50%",
                               "zero_rate_50-80%", "zero_rate_>=80%"]:
                    series_mask = np.ones(len(meta), dtype=bool) if cohort == "all" else cohorts == cohort
                    cells = target_released & series_mask[:, None]
                    if not cells.any(): continue
                    summaries.append(metric_row(actual[cells], forecast[cells], origin, cohort, model))
    per_origin = pd.DataFrame(summaries)
    per_origin.to_csv(OUT / "per_origin_cohort.csv", index=False)
    pooled = per_origin.groupby(["cohort", "model"], as_index=False).agg(
        observations=("observations", "sum"), zero_day_observations=("zero_day_observations", "sum"),
        zero_day_pred_gt_0_count=("zero_day_pred_gt_0_count", "sum"),
        zero_day_pred_gt_0_5_count=("zero_day_pred_gt_0_5_count", "sum"),
        zero_day_pred_gt_1_count=("zero_day_pred_gt_1_count", "sum"),
        nonzero_day_observations=("nonzero_day_observations", "sum"),
        nonzero_day_pred_eq_0_count=("nonzero_day_pred_eq_0_count", "sum"),
        nonzero_day_underpred_count=("nonzero_day_underpred_count", "sum"),
        nonzero_day_actual_sum=("nonzero_day_actual_sum", "sum"),
        nonzero_day_pred_sum=("nonzero_day_pred_sum", "sum"),
        zero_day_squared_loss=("zero_day_squared_loss", "sum"),
        nonzero_day_squared_loss=("nonzero_day_squared_loss", "sum"),
        always_zero_squared_loss=("always_zero_squared_loss", "sum"),
        total_squared_loss=("total_squared_loss", "sum"))
    pooled["actual_zero_rate"] = pooled.zero_day_observations / pooled.observations
    for suffix, count_col in (("0", "zero_day_pred_gt_0_count"),
                              ("0_5", "zero_day_pred_gt_0_5_count"),
                              ("1", "zero_day_pred_gt_1_count")):
        pooled[f"zero_day_pred_gt_{suffix}_rate"] = pooled[count_col] / pooled.zero_day_observations
    pooled["zero_day_mean_squared_loss"] = pooled.zero_day_squared_loss / pooled.zero_day_observations
    pooled["nonzero_day_pred_eq_0_rate"] = pooled.nonzero_day_pred_eq_0_count / pooled.nonzero_day_observations
    pooled["nonzero_day_underpred_rate"] = pooled.nonzero_day_underpred_count / pooled.nonzero_day_observations
    pooled["nonzero_day_mean_actual"] = pooled.nonzero_day_actual_sum / pooled.nonzero_day_observations
    pooled["nonzero_day_mean_forecast"] = pooled.nonzero_day_pred_sum / pooled.nonzero_day_observations
    pooled["nonzero_day_mean_squared_loss"] = pooled.nonzero_day_squared_loss / pooled.nonzero_day_observations
    pooled["all_day_mean_squared_loss"] = pooled.total_squared_loss / pooled.observations
    pooled["zero_day_squared_loss_share"] = pooled.zero_day_squared_loss / (
        pooled.zero_day_squared_loss + pooled.nonzero_day_squared_loss).replace(0, np.nan)
    pooled.to_csv(OUT / "pooled_cohort.csv", index=False)
    overall_origin = per_origin[per_origin.cohort == "all"].copy()
    overall_origin.to_csv(OUT / "per_origin_overall.csv", index=False)
    overall = pooled[pooled.cohort == "all"].copy()
    overall.to_csv(OUT / "pooled_overall.csv", index=False)

    sparse_bands = pd.DataFrame(sparse_band_rows)
    sparse_bands.to_csv(OUT / "sparse_forecast_bands_per_origin.csv", index=False)
    sparse_summary = pd.DataFrame(sparse_summary_rows)
    # Rank forecasts within each saved validation fold and model; the cutoff uses predictions only.
    fold_thresholds = {}
    for model in ("calendar_snap", "rho_baseline"):
        for fold_no in range(1, len(FOLDS) + 1):
            parts = [r["p"] for r in sparse_cell_data if r["model"] == model and r["fold"] == fold_no]
            if parts:
                fold_thresholds[(fold_no, model)] = float(np.quantile(np.concatenate(parts), .9))
    for row in sparse_cell_data:
        threshold = fold_thresholds[(row["fold"], row["model"])]
        top = row["p"] >= threshold; high = row["y"] >= 3
        ix = (sparse_summary.origin_d == row["origin_d"]) & (sparse_summary.model == row["model"])
        sparse_summary.loc[ix, "top_decile_threshold"] = threshold
        sparse_summary.loc[ix, "top_decile_count"] = int(top.sum())
        sparse_summary.loc[ix, "top_decile_y_ge_3_count"] = int((top & high).sum())
        sparse_summary.loc[ix, "top_decile_y_ge_3_rate"] = float(high[top].mean()) if top.any() else np.nan
        sparse_summary.loc[ix, "top_decile_mean_actual"] = float(row["y"][top].mean()) if top.any() else np.nan
        sparse_summary.loc[ix, "top_decile_rate_lift_vs_all"] = (
            float(high[top].mean()/high.mean()) if top.any() and high.any() else np.nan)
    sparse_summary.to_csv(OUT / "sparse_unconditional_per_origin.csv", index=False)
    # Pool counts and sums across origins, preserving each fold's prediction-only cutoff.
    band_pooled = sparse_bands.groupby(["model", "forecast_band"], as_index=False).agg(
        observations=("observations", "sum"))
    # Reconstruct pooled band means from per-origin means weighted by their cell counts.
    for col in ("mean_actual", "mean_forecast", "rate_y_ge_1", "rate_y_ge_3"):
        values = sparse_bands.assign(_weighted=sparse_bands[col] * sparse_bands.observations)
        agg = values.groupby(["model", "forecast_band"])._weighted.sum(min_count=1)
        band_pooled[col] = [agg.loc[(r.model, r.forecast_band)] / r.observations
                            if r.observations else np.nan for r in band_pooled.itertuples()]
    band_pooled.to_csv(OUT / "sparse_forecast_bands_pooled.csv", index=False)
    # Pooled calibration by fixed forecast band for the calendar + SNAP model.
    calibration = (band_pooled[band_pooled.model == "calendar_snap"]
                   .set_index("forecast_band").reindex([b[2] for b in SPARSE_BANDS]))
    band_order = [b[2] for b in SPARSE_BANDS]
    cal = calibration.reindex(band_order)
    cal = cal[cal.observations.fillna(0) > 0]
    labels = {"0": "0", "(0,.05]": "(0, .05]", "(.05,.1]": "(.05, .1]",
              "(.1,.2]": "(.1, .2]", "(.2,.5]": "(.2, .5]", "(.5,1]": "(.5, 1]", ">1": ">1"}
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(cal))
    actual = cal.mean_actual.to_numpy(float)
    predicted = cal.mean_forecast.to_numpy(float)
    max_value = max(float(np.nanmax(actual)), float(np.nanmax(predicted)), .1)
    right = max_value * 1.32
    for yi, (px, axv) in enumerate(zip(predicted, actual)):
        ax.plot([px, axv], [yi, yi], color="#aeb8c0", linewidth=2, zorder=1)
    ax.scatter(predicted, y, s=58, color="#2878a5", edgecolor="white", linewidth=1,
               label="Mean forecast", zorder=3)
    ax.scatter(actual, y, s=58, color="#e28b37", edgecolor="white", linewidth=1,
               label="Mean actual", zorder=3)
    ax.set_yticks(y, [labels[b] for b in cal.index])
    ax.invert_yaxis()
    ax.set_xlim(0, right)
    ax.set_ylim(len(cal)-.45, -.55)
    ax.set_xlabel("Mean units per item-day")
    ax.set_ylabel("Forecast-value band")
    zero_actual = float(cal.loc["0", "mean_actual"]) if "0" in cal.index else np.nan
    ax.set_title("Sparse-item calibration by forecast band\n"
                 f"Calendar + SNAP · pooled rolling validation · zero forecast: actual {zero_actual:.2f}/day")
    ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    for yi, (_, row) in enumerate(cal.iterrows()):
        ax.text(right * .985, yi, f"n={int(row.observations):,}", ha="right", va="center",
                fontsize=8, color="#87919a")
    fig.tight_layout()
    fig.savefig(OUT / "sparse_forecast_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    pooled_sparse = sparse_summary.groupby("model", as_index=False).agg(
        origins=("origin_d", "nunique"), observations=("observations", "sum"),
        actual_sum_all=("actual_sum_all", "sum"), forecast_sum_all=("forecast_sum_all", "sum"),
        y_ge_1_count=("y_ge_1_count", "sum"), y_ge_3_count=("y_ge_3_count", "sum"),
        y_ge_3_sse=("y_ge_3_sse", "sum"),
        total_sse=("total_sse", "sum"), forecast_sum_on_y_ge_3=("forecast_sum_on_y_ge_3", "sum"),
        top_decile_count=("top_decile_count", "sum"),
        top_decile_y_ge_3_count=("top_decile_y_ge_3_count", "sum"))
    # Correct weighted rates and error shares from additive per-origin quantities.
    pooled_sparse["mean_actual_all"] = pooled_sparse.actual_sum_all / pooled_sparse.observations
    pooled_sparse["mean_forecast_all"] = pooled_sparse.forecast_sum_all / pooled_sparse.observations
    pooled_sparse["rate_y_ge_1_all"] = pooled_sparse.y_ge_1_count / pooled_sparse.observations
    pooled_sparse["rate_y_ge_3_all"] = pooled_sparse.y_ge_3_count / pooled_sparse.observations
    pooled_sparse["y_ge_3_rate"] = pooled_sparse.rate_y_ge_3_all
    pooled_sparse["y_ge_3_sse_share"] = pooled_sparse.y_ge_3_sse / pooled_sparse.total_sse.replace(0, np.nan)
    pooled_sparse["mean_forecast_on_y_ge_3"] = pooled_sparse.forecast_sum_on_y_ge_3 / pooled_sparse.y_ge_3_count.replace(0, np.nan)
    pooled_sparse["top_decile_y_ge_3_rate"] = pooled_sparse.top_decile_y_ge_3_count / pooled_sparse.top_decile_count
    pooled_sparse["top_decile_rate_lift_vs_all"] = pooled_sparse.top_decile_y_ge_3_rate / pooled_sparse.rate_y_ge_3_all
    pooled_sparse.to_csv(OUT / "sparse_unconditional_pooled.csv", index=False)

    # The right panel uses all eligible leaf/day squared loss as denominator,
    # so it shows each cohort's contribution to total forecast error.
    order = ["zero_rate_<50%", "zero_rate_50-80%", "zero_rate_>=80%", "young_<28d"]
    short_labels = ["Mature <50%", "Mature 50–80%", "Mature ≥80%", "Young <28d"]
    chosen = pooled[pooled.cohort.isin(order)]
    total_loss = pooled[pooled.cohort == "all"].set_index("model").total_squared_loss
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(order)); width = .35
    for j, model in enumerate(("calendar_snap", "rho_baseline")):
        part = chosen[chosen.model == model].set_index("cohort").reindex(order)
        axes[0].bar(x+(j-.5)*width, part.zero_day_pred_gt_0_5_rate*100, width,
                    label=model.replace("_", " "), color=("#2878a5" if j == 0 else "#9aa3aa"))
        axes[1].bar(x+(j-.5)*width, part.zero_day_squared_loss / float(total_loss.loc[model])*100, width,
                    label=model.replace("_", " "), color=("#2878a5" if j == 0 else "#9aa3aa"))
    axes[0].set_ylabel("Actual zero days forecast > 0.5 (%)")
    axes[0].set_title("Zero day overforecast frequency")
    axes[1].set_ylabel("Cohort zero-day squared loss / all eligible loss (%)")
    axes[1].set_title("Contribution to total squared error")
    for ax in axes:
        ax.set_xticks(x, short_labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen calendar + SNAP rolling forecasts vs rho baseline")
    fig.tight_layout()
    fig.savefig(OUT / "zero_day_diagnostic.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
