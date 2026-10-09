"""检查 M5 原始数据，并写出数据质量报告；不会修改原始 CSV。

运行顺序：第 1 个。若检查失败，先阅读 outputs/data_prep/reports/raw_audit.json。
此脚本只审计 sales_train_validation.csv，避免把后续发布的真实销量
sales_train_evaluation.csv 误用为早期预测的历史信息。
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
REPORT_DIR = ROOT / "outputs" / "data_prep" / "reports"
REPORT_PATH = REPORT_DIR / "raw_audit.json"

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
DAY_COLUMNS = [f"d_{day}" for day in range(1, 1914)]
PRICE_KEY = ["store_id", "item_id", "wm_yr_wk"]


def require_raw_files() -> dict[str, Path]:
    """在开始较慢的全表检查前，先确定原始文件都存在。"""
    names = [
        "sales_train_validation.csv",
        "sales_train_evaluation.csv",
        "calendar.csv",
        "sell_prices.csv",
        "sample_submission.csv",
    ]
    paths = {name: RAW / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少原始文件：{missing}；预期位置：{RAW}")
    return paths


def audit_sales(path: Path, issues: list[str]) -> dict:
    """分块读取宽表，避免为了审计而同时把 1,913 天放进内存。"""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    expected = ID_COLUMNS + DAY_COLUMNS
    if columns != expected:
        issues.append("销量文件列名或顺序与预期不一致")
        return {"columns_found": len(columns), "expected_columns": len(expected)}

    dtype = {name: "float32" for name in DAY_COLUMNS}
    row_count = missing_cells = infinite_cells = negative_cells = fractional_cells = 0
    zero_cells = 0
    minimum, maximum = float("inf"), float("-inf")
    seen_ids: set[str] = set()
    duplicate_ids = 0
    stores, categories, states = Counter(), Counter(), Counter()
    metadata_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, dtype=dtype, chunksize=256):
        row_count += len(chunk)
        ids = chunk["id"].astype(str).tolist()
        duplicate_ids += len(ids) - len(set(ids))
        duplicate_ids += len(set(ids) & seen_ids)
        seen_ids.update(ids)
        stores.update(chunk["store_id"].astype(str))
        categories.update(chunk["cat_id"].astype(str))
        states.update(chunk["state_id"].astype(str))
        metadata_parts.append(chunk[ID_COLUMNS].copy())

        values = chunk[DAY_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        missing_cells += int(np.isnan(values).sum())
        infinite_cells += int(np.isinf(values).sum())
        finite = values[np.isfinite(values)]
        if finite.size:
            negative_cells += int((finite < 0).sum())
            fractional_cells += int((finite != np.floor(finite)).sum())
            zero_cells += int((finite == 0).sum())
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))

    metadata = pd.concat(metadata_parts, ignore_index=True)
    item_to_dept_violations = int(
        (metadata.groupby("item_id")["dept_id"].nunique() > 1).sum()
    )
    dept_to_cat_violations = int(
        (metadata.groupby("dept_id")["cat_id"].nunique() > 1).sum()
    )
    store_to_state_violations = int(
        (metadata.groupby("store_id")["state_id"].nunique() > 1).sum()
    )

    if row_count != 30_490:
        issues.append(f"销量行数为 {row_count}，预期 30,490")
    if duplicate_ids:
        issues.append(f"发现 {duplicate_ids} 个重复的销量序列 ID")
    if missing_cells or infinite_cells or negative_cells or fractional_cells:
        issues.append("销量中存在缺失、无穷大、负数或非整数；请先查看报告")
    if item_to_dept_violations or dept_to_cat_violations or store_to_state_violations:
        issues.append("商品/部门/品类或门店/州的映射不是一对一")

    total_cells = row_count * len(DAY_COLUMNS)
    return {
        "rows": row_count,
        "columns": len(columns),
        "days": len(DAY_COLUMNS),
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "missing_sales_cells": missing_cells,
        "infinite_sales_cells": infinite_cells,
        "negative_sales_cells": negative_cells,
        "fractional_sales_cells": fractional_cells,
        "zero_sales_ratio": round(zero_cells / total_cells, 6),
        "minimum_sale": None if minimum == float("inf") else minimum,
        "maximum_sale": None if maximum == float("-inf") else maximum,
        "stores": dict(sorted(stores.items())),
        "categories": dict(sorted(categories.items())),
        "states": dict(sorted(states.items())),
        "item_to_dept_violations": item_to_dept_violations,
        "dept_to_cat_violations": dept_to_cat_violations,
        "store_to_state_violations": store_to_state_violations,
    }


def audit_calendar(path: Path, issues: list[str]) -> dict:
    """检查日期轴是否完整、唯一、连续；事件名为空是正常情况。"""
    calendar = pd.read_csv(path)
    expected = {"date", "d", "wm_yr_wk", "wday", "event_name_1", "snap_CA", "snap_TX", "snap_WI"}
    missing_columns = sorted(expected - set(calendar.columns))
    if missing_columns:
        issues.append(f"日历缺少列：{missing_columns}")
        return {"rows": len(calendar), "missing_columns": missing_columns}

    dates = pd.to_datetime(calendar["date"], errors="coerce")
    day_numbers = pd.to_numeric(calendar["d"].str.extract(r"^d_(\d+)$")[0], errors="coerce")
    ordered = pd.DataFrame({"day": day_numbers, "date": dates}).sort_values("day")
    missing_training_days = sorted(set(range(1, 1914)) - set(day_numbers.dropna().astype(int)))
    duplicate_days = int(calendar["d"].duplicated().sum())
    duplicate_dates = int(dates.duplicated().sum())
    discontinuities = int((ordered["date"].diff().dt.days.dropna() != 1).sum())
    if dates.isna().any() or day_numbers.isna().any() or missing_training_days:
        issues.append("日历日期或 d 编号有缺失/解析失败")
    if duplicate_days or duplicate_dates or discontinuities:
        issues.append("日历日期轴出现重复或不连续")

    return {
        "rows": len(calendar),
        "first_date": None if dates.isna().all() else dates.min().date().isoformat(),
        "last_date": None if dates.isna().all() else dates.max().date().isoformat(),
        "duplicate_days": duplicate_days,
        "duplicate_dates": duplicate_dates,
        "discontinuities": discontinuities,
        "missing_training_days_count": len(missing_training_days),
        "event_name_1_nonempty": int(calendar["event_name_1"].notna().sum()),
    }


def audit_prices(path: Path, issues: list[str]) -> dict:
    """售价表按商品×门店×周存储；缺失价格可能有业务含义。"""
    prices = pd.read_csv(
        path,
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    missing = {name: int(prices[name].isna().sum()) for name in PRICE_KEY + ["sell_price"]}
    duplicate_keys = int(prices.duplicated(PRICE_KEY).sum())
    nonpositive_prices = int((prices["sell_price"] <= 0).sum())
    if any(missing.values()) or duplicate_keys or nonpositive_prices:
        issues.append("价格表存在缺失、重复键或非正价格")
    return {
        "rows": len(prices),
        "unique_price_keys": len(prices) - duplicate_keys,
        "duplicate_price_keys": duplicate_keys,
        "missing": missing,
        "nonpositive_prices": nonpositive_prices,
        "unique_stores": int(prices["store_id"].nunique()),
        "unique_items": int(prices["item_id"].nunique()),
        "minimum_price": float(prices["sell_price"].min()),
        "maximum_price": float(prices["sell_price"].max()),
    }


def main() -> None:
    paths = require_raw_files()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    print("1/3 检查销量宽表…")
    sales = audit_sales(paths["sales_train_validation.csv"], issues)
    print("2/3 检查日历…")
    calendar = audit_calendar(paths["calendar.csv"], issues)
    print("3/3 检查售价表…")
    prices = audit_prices(paths["sell_prices.csv"], issues)
    report = {"source_directory": str(RAW), "sales": sales, "calendar": calendar,
              "prices": prices, "issues": issues}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入：{REPORT_PATH}")
    if issues:
        raise ValueError("原始数据检查未通过；请先阅读 raw_audit.json 的 issues")
    print("原始数据检查通过，可以运行 02_prepare_clean.py。")


if __name__ == "__main__":
    main()
