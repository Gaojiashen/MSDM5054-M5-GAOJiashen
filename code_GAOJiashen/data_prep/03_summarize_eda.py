"""第 3 步：只计算 M5 探索性分析的表，不绘图。

这样修改配色和布局时只需重跑 04_plot_eda.py，不必每次重读
30,490 × 1,913 的销量矩阵。所有图表都可追溯到本脚本的 CSV 表。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "data_prep" / "tables"
REPORTS = ROOT / "outputs" / "data_prep" / "reports"


def load_inputs() -> tuple[np.memmap, pd.DataFrame, pd.DataFrame]:
    """检查矩阵行列与元数据、日历严格对应。"""
    paths = [PROCESSED / name for name in (
        "manifest.json", "sales_matrix.npy", "sales_metadata.csv", "calendar_clean.csv"
    )]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"请先运行 02_prepare_clean.py，缺少：{missing}")
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    matrix = np.load(paths[1], mmap_mode="r")
    metadata = pd.read_csv(paths[2])
    calendar = pd.read_csv(paths[3], parse_dates=["date"])
    calendar = calendar.loc[calendar["d_num"].between(1, 1913)].sort_values("d_num")
    if list(matrix.shape) != manifest["matrix_shape"] or len(metadata) != matrix.shape[0]:
        raise ValueError("销量矩阵与元数据、manifest 规模不一致")
    if len(calendar) != matrix.shape[1] or not np.array_equal(calendar["d_num"], np.arange(1, 1914)):
        raise ValueError("日历与销量矩阵的日期顺序不一致")
    return matrix, metadata, calendar.reset_index(drop=True)


def summarize_daily(matrix: np.memmap, calendar: pd.DataFrame) -> pd.DataFrame:
    """每日总销量和零值比例，用于看趋势、停业日和稀疏程度。"""
    daily = calendar[["d_num", "date", "wm_yr_wk", "has_event"]].copy()
    daily["total_units"] = matrix.sum(axis=0, dtype=np.int64)
    daily["zero_series_share"] = (matrix == 0).mean(axis=0)
    daily["rolling_28d_mean"] = daily["total_units"].rolling(28, min_periods=28).mean()
    daily.to_csv(TABLES / "daily_summary.csv", index=False, date_format="%Y-%m-%d")
    return daily


def summarize_weekday(daily: pd.DataFrame) -> pd.DataFrame:
    """从日期计算真正的星期；M5 原始 wday 的编号方式不同。"""
    work = daily.assign(weekday_num=daily["date"].dt.weekday)
    weekday = work.groupby("weekday_num", as_index=False).agg(
        mean_units=("total_units", "mean"),
        median_units=("total_units", "median"),
        observed_days=("total_units", "size"),
    )
    weekday["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday.to_csv(TABLES / "weekday_summary.csv", index=False)
    return weekday


def summarize_group(matrix: np.memmap, metadata: pd.DataFrame,
                    calendar: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """每组同时保存总量和每序列日均量，防止量级差掩盖形态差。"""
    rows = []
    all_days = []
    for label in sorted(metadata[column].dropna().unique()):
        selected = matrix[metadata[column].to_numpy() == label]
        n_series = selected.shape[0]
        day_units = selected.sum(axis=0, dtype=np.int64)
        rows.append({
            column: label,
            "series_count": n_series,
            "total_units": int(day_units.sum()),
            "mean_daily_units": float(day_units.mean()),
            "mean_daily_units_per_series": float(day_units.mean() / n_series),
            "zero_cell_share": float((selected == 0).mean()),
        })
        all_days.append(pd.DataFrame({
            "date": calendar["date"].to_numpy(),
            column: label,
            "daily_units": day_units,
            "units_per_series": day_units / n_series,
        }))
    summary = pd.DataFrame(rows).sort_values("total_units", ascending=False)
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False)
    pd.concat(all_days, ignore_index=True).to_csv(
        TABLES / f"{prefix}_daily.csv", index=False, date_format="%Y-%m-%d"
    )
    return summary


def summarize_series(matrix: np.memmap, metadata: pd.DataFrame,
                     calendar: pd.DataFrame) -> pd.DataFrame:
    """保存全历史及最近一年特征，并选出每类可复现的代表序列。"""
    series = metadata[["id", "item_id", "cat_id", "store_id"]].copy()
    series.insert(0, "row_index", np.arange(len(metadata), dtype=np.int32))
    series["mean_daily_units"] = matrix.mean(axis=1)
    series["zero_day_share"] = (matrix == 0).mean(axis=1)
    # 最近 365 个已观测日；前后 90 天销量用于查看序列变化。
    recent_start = matrix.shape[1] - 365
    recent = matrix[:, recent_start:]
    series["mean_daily_units_last365"] = recent.mean(axis=1)
    series["zero_day_share_last365"] = (recent == 0).mean(axis=1)
    series["first90_units_last365"] = recent[:, :90].sum(axis=1)
    series["last90_units_last365"] = recent[:, -90:].sum(axis=1)
    series.to_csv(TABLES / "series_summary.csv", index=False)

    # 要求最近一年每个自然月都有正销量，避免把季节性停卖或短历史序列
    # 当作常规代表；再选取最接近品类中位数的销量水平与零值率。
    recent_dates = calendar["date"].iloc[recent_start:].reset_index(drop=True)
    month_codes = recent_dates.dt.to_period("M").astype(str).to_numpy()
    if len(month_codes) != 365:
        raise ValueError("最近一年日历与销量矩阵长度不一致")
    candidate_rows = []
    for cat_id in sorted(series["cat_id"].dropna().unique()):
        group = series.loc[series["cat_id"] == cat_id]
        eligible = []
        for row in group.itertuples(index=False):
            values = recent[int(row.row_index)]
            if all(np.any(values[month_codes == month] > 0) for month in np.unique(month_codes)):
                eligible.append(int(row.row_index))
        if not eligible:
            raise ValueError(f"品类 {cat_id} 没有最近 365 天所覆盖的各自然月均有销量的序列")
        pool = group.loc[group["row_index"].isin(eligible)].copy()
        med_mean = float(pool["mean_daily_units_last365"].median())
        med_zero = float(pool["zero_day_share_last365"].median())
        # 标准化绝对距离，避免销量和比例的量纲影响；排序字段确保稳定打破并列。
        mean_scale = max(float(pool["mean_daily_units_last365"].std(ddof=0)), 1e-12)
        zero_scale = max(float(pool["zero_day_share_last365"].std(ddof=0)), 1e-12)
        pool["_distance"] = (
            (pool["mean_daily_units_last365"] - med_mean).abs() / mean_scale
            + (pool["zero_day_share_last365"] - med_zero).abs() / zero_scale
        )
        chosen = pool.sort_values(
            ["_distance", "item_id", "store_id", "id"], kind="mergesort"
        ).iloc[0]
        candidate_rows.append(chosen.drop(labels="_distance"))
    selected = pd.DataFrame(candidate_rows).reset_index(drop=True)
    selected.to_csv(TABLES / "example_series_selection.csv", index=False)
    daily_parts = []
    for row in selected.itertuples(index=False):
        values = np.asarray(matrix[int(row.row_index), recent_start:], dtype=np.int64)
        daily_parts.append(pd.DataFrame({
            "date": recent_dates, "category": row.cat_id, "id": row.id,
            "actual_units": values,
        }))
    pd.concat(daily_parts, ignore_index=True).to_csv(
        TABLES / "example_series_daily.csv", index=False, date_format="%Y-%m-%d"
    )
    return series


def choose_price_example(metadata: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    """从固定样本里找价格变化较多的商品，展示周价格的变化方式。

    示例是为了看清价格如何按周记录，不能当作总体价格变动的代表。
    """
    price_path = PROCESSED / "sell_prices_clean.csv.gz"
    if not price_path.is_file():
        raise FileNotFoundError(f"缺少处理后价格文件：{price_path}")
    positions = np.unique(np.linspace(0, len(metadata) - 1, 240, dtype=int))
    candidates = metadata.iloc[positions][["store_id", "item_id"]].drop_duplicates()
    candidate_items = set(candidates["item_id"])
    candidate_stores = set(candidates["store_id"])
    pieces = []
    for chunk in pd.read_csv(price_path, chunksize=200_000):
        subset = chunk.loc[
            chunk["item_id"].isin(candidate_items) & chunk["store_id"].isin(candidate_stores)
        ]
        if not subset.empty:
            matched = subset.merge(candidates, on=["store_id", "item_id"], how="inner")
            if not matched.empty:
                pieces.append(matched)
    if not pieces:
        raise ValueError("候选商品中没有找到价格记录")
    prices = pd.concat(pieces, ignore_index=True)
    ranking = prices.groupby(["store_id", "item_id"]).agg(
        distinct_prices=("sell_price", "nunique"), weeks=("sell_price", "size")
    ).sort_values(["distinct_prices", "weeks"], ascending=False)
    store_id, item_id = ranking.index[0]
    example = prices.loc[(prices["store_id"] == store_id) & (prices["item_id"] == item_id)].copy()
    week_dates = calendar.groupby("wm_yr_wk", as_index=False)["date"].min()
    example = example.merge(week_dates, on="wm_yr_wk", how="inner").sort_values("date")
    example.to_csv(TABLES / "example_price_series.csv", index=False, date_format="%Y-%m-%d")
    return {
        "store_id": store_id, "item_id": item_id,
        "observed_weeks": len(example),
        "distinct_prices": int(example["sell_price"].nunique()),
        "selection": "240 个等距候选序列中价格种类最多者；仅作展示",
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    matrix, metadata, calendar = load_inputs()
    print("1/5 汇总每日销量…")
    daily = summarize_daily(matrix, calendar)
    print("2/5 汇总星期效应…")
    summarize_weekday(daily)
    print("3/5 汇总品类与州…")
    categories = summarize_group(matrix, metadata, calendar, "cat_id", "category")
    states = summarize_group(matrix, metadata, calendar, "state_id", "state")
    print("4/5 汇总每条序列的稀疏程度…")
    series = summarize_series(matrix, metadata, calendar)
    print("5/5 选取价格轨迹示例…")
    price_example = choose_price_example(metadata, calendar)
    summary = {
        "source_days": [1, 1913],
        "series": len(metadata),
        "total_units": int(daily["total_units"].sum()),
        "overall_zero_cell_share": float(series["zero_day_share"].mean()),
        "categories": categories.to_dict(orient="records"),
        "states": states.to_dict(orient="records"),
        "price_example": price_example,
        "series_examples": {
            "count": 3,
            "selection": "每个品类在最近 365 天的每个自然月均至少有一天正销量；在合格序列中，按最近一年日均销量和零销量占比相对品类中位数的标准化绝对距离之和选最近者，完全并列时按 item_id、store_id、id 字典序确定。",
            "daily_table": "example_series_daily.csv（最近 365 天实际日销量；预测区间不填入虚构值）",
        },
        "caution": "图表可用完整历史探索；建模回测须先切分时间，再计算特征。",
    }
    (REPORTS / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"汇总表：{TABLES}")


if __name__ == "__main__":
    main()
