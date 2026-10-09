"""把通过审计的 M5 原始 CSV 转成便于分析的中间数据。

运行顺序：第 2 个。先运行 01_audit_raw.py，且 raw_audit.json 的 issues 为空。
原始 CSV 不会被改动。这里的“清洗”指类型转换、日期规范化和质量把关；
不会凭空填补价格，也不会删除销量为 0 的日期。销售矩阵的行与
sales_metadata.csv 严格同序，列依次为 d_1 ... d_1913。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "data_prep" / "reports"
AUDIT_PATH = REPORTS / "raw_audit.json"
SUMMARY_PATH = REPORTS / "cleaning_summary.json"
DAY_COLUMNS = [f"d_{day}" for day in range(1, 1914)]
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def check_audit() -> dict:
    """审计是本脚本的前置关卡；避免在不合格原始数据上继续处理。"""
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"请先运行 01_audit_raw.py：{AUDIT_PATH}")
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if report.get("issues"):
        raise ValueError(f"原始数据审计未通过：{report['issues']}")
    if report["sales"]["rows"] != 30_490 or report["sales"]["days"] != 1913:
        raise ValueError("审计报告中的销量规模与本项目预期不符")
    return report


def prepare_sales(audit: dict) -> dict:
    """分块读取宽表，写为 NumPy 矩阵，避免构造数千万行的长表。"""
    max_sale = audit["sales"]["maximum_sale"]
    if max_sale is None or max_sale > np.iinfo(np.uint32).max:
        raise ValueError("销量最大值超出当前整数存储范围")
    # M5 销量是非负整数。选能容纳最大值的类型以节约磁盘和内存。
    storage_dtype = np.uint16 if max_sale <= np.iinfo(np.uint16).max else np.uint32
    n_rows, n_days = 30_490, len(DAY_COLUMNS)
    matrix_path = PROCESSED / "sales_matrix.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=storage_dtype, shape=(n_rows, n_days)
    )
    metadata_parts: list[pd.DataFrame] = []
    row_start = 0
    read_dtype = {name: "float32" for name in DAY_COLUMNS}
    for chunk in pd.read_csv(
        RAW / "sales_train_validation.csv", dtype=read_dtype, chunksize=256
    ):
        values = chunk[DAY_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        # 再次防守性检查：原始文件可能在审计后被替换。
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("销量出现缺失、无穷大、负数或小数；请重新审计原始数据")
        if values.max() > np.iinfo(storage_dtype).max:
            raise ValueError("销量超过选定整数类型范围；请重新审计原始数据")
        row_end = row_start + len(chunk)
        matrix[row_start:row_end] = values.astype(storage_dtype)
        metadata_parts.append(chunk[ID_COLUMNS].copy())
        row_start = row_end
    matrix.flush()
    del matrix

    if row_start != n_rows:
        raise ValueError(f"销量行数发生变化：读到 {row_start}，预期 {n_rows}")
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if metadata["id"].duplicated().any():
        raise ValueError("销量序列 ID 重复；请重新审计原始数据")
    metadata.to_csv(PROCESSED / "sales_metadata.csv", index=False)
    return {"rows": row_start, "days": n_days, "matrix_dtype": np.dtype(storage_dtype).name}


def prepare_calendar() -> dict:
    """把 d 编号与真实日期明确对应，保留事件缺失作为“无事件”。"""
    calendar = pd.read_csv(RAW / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    calendar["d_num"] = calendar["d"].str.extract(r"^d_(\d+)$")[0].astype("int32")
    calendar = calendar.sort_values("d_num").reset_index(drop=True)
    if calendar["d_num"].duplicated().any() or calendar["date"].duplicated().any():
        raise ValueError("日历中有重复的 d 编号或日期")
    if not set(range(1, 1914)).issubset(set(calendar["d_num"])):
        raise ValueError("日历缺少销量对应的历史日期")
    # SNAP 标记在原始文件中已有；转换为小整数，不改变原意。
    for state in ("CA", "TX", "WI"):
        column = f"snap_{state}"
        calendar[column] = calendar[column].astype("int8")
    calendar["has_event"] = calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8")
    calendar.to_csv(PROCESSED / "calendar_clean.csv", index=False, date_format="%Y-%m-%d")
    return {
        "rows": len(calendar),
        "training_start": calendar.loc[calendar["d_num"] == 1, "date"].iloc[0].date().isoformat(),
        "training_end": calendar.loc[calendar["d_num"] == 1913, "date"].iloc[0].date().isoformat(),
        "days_with_event_in_training": int(calendar.loc[calendar["d_num"] <= 1913, "has_event"].sum()),
    }


def prepare_prices() -> dict:
    """价格按商品、门店、周保留原始粒度；没有记录的周不擅自插值。"""
    prices = pd.read_csv(
        RAW / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    key = ["store_id", "item_id", "wm_yr_wk"]
    if prices[key + ["sell_price"]].isna().any().any():
        raise ValueError("价格表出现缺失；请重新审计原始数据")
    if prices.duplicated(key).any() or (prices["sell_price"] <= 0).any():
        raise ValueError("价格表出现重复键或非正价格；请重新审计原始数据")
    prices.to_csv(PROCESSED / "sell_prices_clean.csv.gz", index=False, compression="gzip")
    return {"rows": len(prices), "minimum_price": float(prices["sell_price"].min()),
            "maximum_price": float(prices["sell_price"].max())}


def main() -> None:
    audit = check_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # 重新运行若中途失败，旧 manifest 不应让下一步误认半成品已完成。
    manifest_path = PROCESSED / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    print("1/3 整理销量矩阵与序列元数据…")
    sales = prepare_sales(audit)
    print("2/3 整理日期与事件信息…")
    calendar = prepare_calendar()
    print("3/3 整理每周售价…")
    prices = prepare_prices()
    summary = {
        "sales": sales, "calendar": calendar, "prices": prices,
        "decisions": [
            "销量 0 保持为 0；不删除零销量日",
            "日历事件空值表示无已记录事件；保留为空",
            "未出现的商品-门店-周售价不补值；将来建模时须单独判断原因",
            "只使用 sales_train_validation.csv 的 d_1 至 d_1913；不读取 evaluation 销量",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "source": "data/m5-forecasting-accuracy/sales_train_validation.csv",
        "sales_matrix": "data/processed/sales_matrix.npy",
        "sales_metadata": "data/processed/sales_metadata.csv",
        "calendar": "data/processed/calendar_clean.csv",
        "sell_prices": "data/processed/sell_prices_clean.csv.gz",
        "matrix_shape": [sales["rows"], sales["days"]],
        "matrix_dtype": sales["matrix_dtype"],
        "row_order": "sales_metadata.csv 的行顺序",
        "column_order": "d_1, d_2, ..., d_1913",
        "warning": "建模时需按时间划分训练/验证；不能把验证期销量用于特征计算",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入处理后数据：{PROCESSED}")
    print(f"已写入清洗说明：{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
