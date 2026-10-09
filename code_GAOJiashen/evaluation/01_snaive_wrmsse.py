"""M5 official-Accuracy-style 28-day sNaive evaluation on d_1914..d_1941.

All forecasts and WRMSSE denominators/weights use d_1..d_1913 only. The
evaluation actuals are read solely after forecasts, for scoring and figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "m5-forecasting-accuracy"
PROCESSED = DATA / "processed"
OUT = ROOT / "outputs" / "evaluation" / "snaive"
TRAIN_END = 1913
HORIZON = 28
WEIGHT_START = TRAIN_END - HORIZON + 1  # d_1886
LEVELS = [
    ("Level 1: Total", []),
    ("Level 2: State", ["state_id"]),
    ("Level 3: Store", ["store_id"]),
    ("Level 4: Category", ["cat_id"]),
    ("Level 5: Department", ["dept_id"]),
    ("Level 6: State × Category", ["state_id", "cat_id"]),
    ("Level 7: State × Department", ["state_id", "dept_id"]),
    ("Level 8: Store × Category", ["store_id", "cat_id"]),
    ("Level 9: Store × Department", ["store_id", "dept_id"]),
    ("Level 10: Item", ["item_id"]),
    ("Level 11: Item × State", ["item_id", "state_id"]),
    ("Level 12: Item × Store", ["item_id", "store_id"]),
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def load_data():
    """Load history as a memmap; validate ordering from raw metadata columns only."""
    meta = pd.read_csv(require_file(PROCESSED / "sales_metadata.csv"))
    raw_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    raw_meta = pd.read_csv(require_file(RAW / "sales_train_validation.csv"), usecols=raw_columns)
    if len(meta) != len(raw_meta) or not meta[raw_columns].equals(raw_meta[raw_columns]):
        raise ValueError("Processed metadata does not match sales_train_validation row order/keys")
    train = np.load(require_file(PROCESSED / "sales_matrix.npy"), mmap_mode="r")
    if train.shape != (len(meta), TRAIN_END) or not np.issubdtype(train.dtype, np.integer):
        raise ValueError(f"Unexpected processed sales matrix: {train.shape}, {train.dtype}")
    if meta.duplicated(["item_id", "store_id"]).any():
        raise ValueError("item_id + store_id must uniquely identify each series")

    calendar = pd.read_csv(require_file(RAW / "calendar.csv"), usecols=["d", "date", "wm_yr_wk"])
    calendar["d_num"] = calendar["d"].str.removeprefix("d_").astype(int)
    cal_train = calendar.loc[calendar["d_num"].between(1, TRAIN_END)].sort_values("d_num")
    cal_test = calendar.loc[calendar["d_num"].between(TRAIN_END + 1, TRAIN_END + HORIZON)].sort_values("d_num")
    if len(cal_train) != TRAIN_END or len(cal_test) != HORIZON:
        raise ValueError("Calendar does not contain the requested train/test dates")
    cal_train = cal_train.reset_index(drop=True)
    cal_test = cal_test.reset_index(drop=True)
    relevant_weeks = set(cal_train.loc[
        cal_train["d_num"].between(WEIGHT_START, TRAIN_END), "wm_yr_wk"
    ].astype(int))
    prices = load_weight_window_prices(relevant_weeks)
    return meta, train, cal_train, cal_test, prices


def load_weight_window_prices(relevant_weeks: set[int]) -> pd.DataFrame:
    """Read only weekly prices needed for the final 28 training days."""
    pieces = []
    for chunk in pd.read_csv(require_file(RAW / "sell_prices.csv"),
                             usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
                             chunksize=500_000):
        part = chunk.loc[chunk["wm_yr_wk"].isin(relevant_weeks)]
        if not part.empty:
            pieces.append(part)
    if not pieces:
        raise ValueError("No weekly prices found for the WRMSSE weight window")
    return pd.concat(pieces, ignore_index=True)


def load_test_actuals(meta: pd.DataFrame) -> np.ndarray:
    """Read and align held-out outcomes; call only after creating forecasts."""
    actual_cols = [f"d_{d}" for d in range(TRAIN_END + 1, TRAIN_END + HORIZON + 1)]
    usecols = ["id", "item_id", "store_id", *actual_cols]
    actual_df = pd.read_csv(require_file(RAW / "sales_train_evaluation.csv"), usecols=usecols)
    key = ["item_id", "store_id"]
    if actual_df.duplicated(key).any():
        raise ValueError("Evaluation item_id + store_id keys are not unique")
    aligned = meta[key + ["id"]].merge(
        actual_df[key + ["id"] + actual_cols], on=key, how="left", validate="one_to_one",
        suffixes=("_validation", "_evaluation"),
    )
    if aligned[actual_cols].isna().any().any() or len(aligned) != len(meta):
        raise ValueError("Evaluation rows do not align one-to-one with validation history")
    if not aligned["id_evaluation"].str.endswith("_evaluation").all():
        raise ValueError("Unexpected evaluation ID suffix")
    if not aligned["id_validation"].str.replace("_validation$", "_evaluation", regex=True).equals(
        aligned["id_evaluation"]
    ):
        raise ValueError("Validation/evaluation ID mapping differs after suffix normalization")
    return aligned[actual_cols].to_numpy(dtype=np.float64)


def make_snaive(train: np.ndarray) -> np.ndarray:
    """Repeat the last observed seven daily values four times per leaf series."""
    if train.shape[1] != TRAIN_END:
        raise ValueError("Forecast input must stop at d_1913")
    return np.tile(train[:, -7:], (1, 4)).astype(np.float64, copy=False)


def build_hierarchy_matrices(meta: pd.DataFrame) -> list[sparse.csr_matrix]:
    """Build reusable node-by-leaf CSR aggregation matrices for all 12 levels."""
    matrices = []
    n_leaf = len(meta)
    for level, (_, keys) in enumerate(LEVELS):
        if level == 11:  # M5 Level 12 is exactly the 30,490 leaf series.
            matrices.append(sparse.identity(n_leaf, format="csr", dtype=np.float64))
            continue
        if not keys:
            codes = np.zeros(n_leaf, dtype=np.int32)
            n_nodes = 1
        else:
            codes, unique = pd.factorize(pd.MultiIndex.from_frame(meta[keys]), sort=True)
            n_nodes = len(unique)
        matrices.append(sparse.csr_matrix(
            (np.ones(n_leaf, dtype=np.float64), (codes, np.arange(n_leaf))),
            shape=(n_nodes, n_leaf),
        ))
    if sum(matrix.shape[0] for matrix in matrices) != 42840:
        raise ValueError("Hierarchy matrices do not total 42,840 nodes")
    return matrices


def calculate_scale(aggregator: sparse.csr_matrix, train: np.ndarray,
                    chunk_size: int = 512) -> np.ndarray:
    """Calculate row scales in chunks without materializing a full diff matrix."""
    n_nodes = aggregator.shape[0]
    scale = np.zeros(n_nodes, dtype=np.float64)
    for start in range(0, n_nodes, chunk_size):
        stop = min(start + chunk_size, n_nodes)
        block = np.asarray(aggregator[start:stop] @ train, dtype=np.float64)
        nonzero = block != 0
        has_nonzero = nonzero.any(axis=1)
        first = np.argmax(nonzero, axis=1)
        # Prefix squared first differences let each row use its own first-nonzero
        # start, while the temporary difference matrix remains chunk-sized.
        diffs = np.diff(block, axis=1)
        prefix = np.concatenate(
            [np.zeros((len(block), 1), dtype=np.float64), np.cumsum(diffs * diffs, axis=1)],
            axis=1,
        )
        rows = np.arange(len(block))
        lengths = (TRAIN_END - 1) - first
        sums = prefix[:, -1] - prefix[rows, first]
        valid = has_nonzero & (lengths > 0)
        scale[start:stop][valid] = sums[valid] / lengths[valid]
    return scale


def score_wrmsse(meta, train, actual, forecast, cal_train, cal_test, prices):
    """Compute twelve equal-weight level scores and total official WRMSSE."""
    recent_sales = train[:, WEIGHT_START - 1:TRAIN_END]
    if prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).any():
        raise ValueError("Duplicate item-store-week prices would make WRMSSE weights ambiguous")
    price_lookup = prices.merge(
        cal_train.loc[cal_train["d_num"].between(WEIGHT_START, TRAIN_END), ["d_num", "wm_yr_wk"]],
        on="wm_yr_wk", how="inner",
    )
    price_lookup = price_lookup[["item_id", "store_id", "d_num", "sell_price"]]
    key_frame = meta[["item_id", "store_id"]].copy()
    key_frame["row"] = np.arange(len(meta))
    price_join = key_frame.merge(price_lookup, on=["item_id", "store_id"], how="left", validate="one_to_many")
    pmat = np.full((len(meta), HORIZON), np.nan, dtype=np.float64)
    has_day = price_join["d_num"].notna()
    rr = price_join.loc[has_day, "row"].to_numpy(dtype=int)
    cc = price_join.loc[has_day, "d_num"].to_numpy(dtype=int) - WEIGHT_START
    pmat[rr, cc] = price_join.loc[has_day, "sell_price"].to_numpy(dtype=float)
    missing_sold = np.isnan(pmat) & (recent_sales > 0)
    if missing_sold.any():
        row, col = np.argwhere(missing_sold)[0]
        raise ValueError(
            "Positive training sales have no weekly price in the weight window: "
            f"{meta.iloc[row]['id']}, d_{WEIGHT_START + col}"
        )
    pmat = np.nan_to_num(pmat, nan=0.0)
    leaf_revenue = (recent_sales * pmat).sum(axis=1)

    result_rows = []
    aggregators = build_hierarchy_matrices(meta)
    for (name, _), aggregator in zip(LEVELS, aggregators):
        y_pred = np.asarray(aggregator @ forecast, dtype=np.float64)
        y_true = np.asarray(aggregator @ actual, dtype=np.float64)
        revenue = np.asarray(aggregator @ leaf_revenue[:, None], dtype=np.float64)[:, 0]
        scale = calculate_scale(aggregator, train)
        denom = revenue
        weight = denom / denom.sum() if denom.sum() > 0 else np.zeros_like(denom)
        err = y_pred - y_true
        mse = np.mean(err ** 2, axis=1)
        zero_scale = scale == 0
        nonzero_error = mse > 0
        if np.any(zero_scale & nonzero_error & (weight > 0)):
            raise ValueError(f"Positive-weight zero-scale nodes with error at {name}")
        rmsse = np.zeros_like(scale)
        valid = ~zero_scale
        rmsse[valid] = np.sqrt(mse[valid] / scale[valid])
        node_wrmsse = weight * rmsse
        level_score = float(node_wrmsse.sum())
        result_rows.append({
            "level": name, "node_count": len(scale), "weight_sum": float(weight.sum()),
            "zero_scale_nodes": int(zero_scale.sum()),
            "positive_weight_zero_scale_nodes": int((zero_scale & (weight > 0)).sum()),
            "wrmsse": level_score,
        })
    scores = pd.DataFrame(result_rows)
    return float(scores["wrmsse"].mean()), scores


def draw_outputs(meta, train, actual, forecast, cal_train, cal_test, scores):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    dates_train = pd.to_datetime(cal_train["date"])
    dates_test = pd.to_datetime(cal_test["date"])
    total_comparison = pd.DataFrame({
        "date": dates_test,
        "snaive_forecast_units": forecast.sum(axis=0),
        "actual_units": actual.sum(axis=0),
    })
    total_comparison.to_csv(OUT / "daily_total_comparison.csv", index=False, date_format="%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(dates_train.iloc[-112:], train.sum(axis=0)[-112:], color="#536878", lw=1.2, label="Actual (last 112 train days)")
    ax.plot(dates_test, forecast.sum(axis=0), color="#187C80", lw=2, marker="o", ms=3, label="sNaive forecast")
    ax.plot(dates_test, actual.sum(axis=0), color="#D06A3B", lw=1.7, marker="o", ms=3, label="Actual test")
    ax.axvline(dates_test.iloc[0], color="#8A9298", ls="--", lw=1)
    ax.set(title="M5 sNaive: total daily sales", ylabel="Units across all series")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "01_total_forecast.png", dpi=180); fig.savefig(OUT / "01_total_forecast.svg"); plt.close(fig)

    selected_path = ROOT / "outputs" / "data_prep" / "tables" / "example_series_selection.csv"
    if selected_path.is_file():
        selected = pd.read_csv(selected_path)
        rows = []
        for cat in ("FOODS", "HOUSEHOLD", "HOBBIES"):
            chosen = selected.loc[selected["cat_id"] == cat].iloc[0]
            row = int(chosen["row_index"])
            if meta.iloc[row]["id"] != chosen["id"]:
                raise ValueError("示例商品选择表与当前销量元数据不一致")
            rows.append(row)
    else:
        # 数据准备阶段的示例表不是评分的必要输入；缺少时取各品类近一年
        # 日均销量最接近中位数的一条，仍保持三个品类都有代表。
        recent_mean = train[:, -365:].mean(axis=1)
        rows = []
        for cat in ("FOODS", "HOUSEHOLD", "HOBBIES"):
            candidates = np.flatnonzero(meta["cat_id"].to_numpy() == cat)
            midpoint = np.median(recent_mean[candidates])
            rows.append(int(candidates[np.argmin(np.abs(recent_mean[candidates] - midpoint))]))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True)
    comparison_rows = []
    for ax, row in zip(axes, rows):
        ax.plot(dates_train.iloc[-56:], train[row, -56:], color="#536878", lw=1.1,
                label="Observed training days")
        ax.axvspan(dates_test.iloc[0], dates_test.iloc[-1], color="#EEF3F4", zorder=0)
        ax.plot(dates_test, forecast[row], color="#187C80", lw=1.8, label="sNaive forecast")
        ax.plot(dates_test, actual[row], color="#D06A3B", lw=1.7, label="Observed test days")
        ax.axvline(dates_test.iloc[0], color="#8A9298", ls="--", lw=0.9)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Units / day")
        ax.text(0.01, 0.91, f"{meta.iloc[row]['cat_id']} · {meta.iloc[row]['item_id']} · {meta.iloc[row]['store_id']}",
                transform=ax.transAxes, va="top", fontsize=10, weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        comparison_rows.append(pd.DataFrame({
            "date": dates_test, "id": meta.iloc[row]["id"],
            "snaive_forecast_units": forecast[row], "actual_units": actual[row],
        }))
    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("M5 sNaive: three product-store forecast examples", x=0.08, ha="left", fontsize=15)
    fig.text(0.08, 0.02, "Grey = training history · Teal = forecast · Orange = held-out actual · Shaded = 28-day test window",
             fontsize=9, color="#536878")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUT / "02_example_series_forecast.png", dpi=180)
    fig.savefig(OUT / "02_example_series_forecast.svg")
    plt.close(fig)
    pd.concat(comparison_rows, ignore_index=True).to_csv(
        OUT / "example_series_comparison.csv", index=False, date_format="%Y-%m-%d"
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(scores["level"], scores["wrmsse"], color="#187C80")
    ax.invert_yaxis(); ax.set(xlabel="WRMSSE within level", title="sNaive WRMSSE by hierarchy level")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_xlim(0, max(scores["wrmsse"].max() * 1.18, 0.01)); fig.tight_layout()
    fig.savefig(OUT / "03_wrmsse_by_level.png", dpi=180); fig.savefig(OUT / "03_wrmsse_by_level.svg"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, train, cal_train, cal_test, prices = load_data()
    forecast = make_snaive(train)  # Construct before using actuals for scoring.
    actual = load_test_actuals(meta)
    score, levels = score_wrmsse(meta, train, actual, forecast, cal_train, cal_test, prices)
    pred = pd.DataFrame(forecast, columns=[f"d_{d}" for d in range(TRAIN_END + 1, TRAIN_END + HORIZON + 1)])
    # 这一预测窗口属于 M5 validation 阶段；即使测试真值从
    # sales_train_evaluation.csv 读取，预测文件仍须使用 _validation ID。
    pred.insert(0, "id", meta["id"])
    pred.to_csv(OUT / "snaive_predictions.csv", index=False)
    levels.to_csv(OUT / "snaive_wrmsse_by_level.csv", index=False)
    summary = {
        "method": "Seasonal Naive (repeat d_1907..d_1913 four times)",
        "train_days": [1, TRAIN_END], "test_days": [TRAIN_END + 1, TRAIN_END + HORIZON],
        "weight_days": [WEIGHT_START, TRAIN_END], "series_count": len(meta),
        "hierarchy_node_count": int(levels["node_count"].sum()), "level_count": len(levels),
        "equal_level_weight": 1 / len(levels), "wrmsse": score,
        "zero_scale_nodes_across_levels": int(levels["zero_scale_nodes"].sum()),
        "positive_weight_zero_scale_nodes_across_levels": int(levels["positive_weight_zero_scale_nodes"].sum()),
    }
    (OUT / "snaive_wrmsse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_outputs(meta, train, actual, forecast, cal_train, cal_test, levels)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
