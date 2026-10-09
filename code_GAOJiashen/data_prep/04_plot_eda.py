"""第 4 步：只读取第 3 步生成的 CSV 表，制作报告可用的探索图。

重新调整配色或布局时只需运行此文件。运行前会清除本脚本旧版图，
防止目录里混有不同风格的图片；不会删除其他目录的数据或报告。
"""

from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from plot_style import (
    BLUE, COLORS, INK, MUTED, ORANGE, PALE, TEAL,
    add_context, apply_style, export, set_compact_y, style_axis,
)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "data_prep" / "tables"
FIGURES = ROOT / "outputs" / "data_prep" / "figures"
SOURCE = "Source: M5 Forecasting — Accuracy, sales_train_validation.csv (d_1–d_1913)."
FIGURE_STEMS = (
    "01_sales_overview", "02_weekday_pattern", "03_category_small_multiples",
    "04_state_comparison", "05_zero_sales_by_category", "06_price_example",
    "07_product_series_examples",
)
# 旧版脚本的六张图：仅清除已知文件名，不触碰用户自行放入的图片。
LEGACY_STEMS = (
    "01_daily_total_sales", "02_weekday_profile", "03_category_trends",
    "04_state_trends", "05_series_mean_distribution", "06_example_weekly_price",
)


def read_table(name: str, *, dates: bool = False) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}；请先运行 03_summarize_eda.py")
    return pd.read_csv(path, parse_dates=["date"] if dates else None)


def clear_previous_figures() -> None:
    """只删当前绘图脚本及其前身生成的已知文件，避免留下旧丑图。"""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for stem in (*FIGURE_STEMS, *LEGACY_STEMS):
        for extension in (".png", ".svg"):
            (FIGURES / f"{stem}{extension}").unlink(missing_ok=True)


def format_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)


def plot_overview() -> None:
    """弱化每日噪声，突出 28 日趋势和年底重复的销量低谷。"""
    daily = read_table("daily_summary.csv", dates=True)
    fig, ax = plt.subplots(figsize=(12, 5.3))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.16)
    ax.plot(daily["date"], daily["total_units"], color=PALE, linewidth=0.65,
            alpha=0.68, label="Daily units")
    ax.plot(daily["date"], daily["rolling_28d_mean"], color=BLUE,
            linewidth=2.2, label="28-day mean")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Units sold per day")
    set_compact_y(ax)
    format_dates(ax)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2, labelcolor=INK)
    zero_dates = daily.loc[daily["total_units"] == 0, "date"]
    if not zero_dates.empty:
        ax.scatter(zero_dates, [0] * len(zero_dates), color=ORANGE, s=20, zorder=5)
        ax.text(zero_dates.iloc[-1], daily["total_units"].max() * 0.13,
                "Zero-sales dates\n(marked in orange)", color=ORANGE,
                ha="right", va="bottom", fontsize=9)
    add_context(fig, "M5 sales trend and recurring zero-sales days",
                "30,490 product–store series · daily totals and 28-day rolling mean · 2011–2016",
                SOURCE)
    export(fig, FIGURES / "01_sales_overview")


def plot_weekday() -> None:
    """按自然星期排列，强调最高的一天，且柱形图从零开始。"""
    weekday = read_table("weekday_summary.csv").sort_values("weekday_num")
    peak = weekday.loc[weekday["mean_units"].idxmax()]
    colors = [ORANGE if row.weekday_num == peak.weekday_num else BLUE
              for row in weekday.itertuples()]
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.79, bottom=0.17)
    bars = ax.bar(weekday["weekday"], weekday["mean_units"], color=colors, width=0.63)
    ax.set_ylim(0, weekday["mean_units"].max() * 1.16)
    for bar, value in zip(bars, weekday["mean_units"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + weekday["mean_units"].max() * 0.025,
                f"{value / 1000:.1f}k", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Mean units sold per day")
    set_compact_y(ax)
    style_axis(ax)
    add_context(fig, f"{peak['weekday']} has the highest average sales",
                "Calendar weekday · mean of observed historical days · orange highlights the maximum",
                SOURCE)
    export(fig, FIGURES / "02_weekday_pattern")


def plot_category() -> None:
    """品类使用独立小图，纵轴统一为每条序列的日销量，便于比较形态。"""
    daily = read_table("category_daily.csv", dates=True)
    summary = read_table("category_summary.csv")
    labels = sorted(daily["cat_id"].unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 8.4), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.88, top=0.82, bottom=0.10, hspace=0.27)
    for ax, label, color in zip(np.atleast_1d(axes), labels, COLORS):
        subset = daily.loc[daily["cat_id"] == label].sort_values("date")
        smooth = subset["units_per_series"].rolling(28, min_periods=28).mean()
        ax.plot(subset["date"], smooth, color=color, linewidth=1.8)
        ax.set_ylim(bottom=0)
        style_axis(ax)
        n_series = int(summary.loc[summary["cat_id"] == label, "series_count"].iloc[0])
        ax.text(0.01, 0.90, f"{label}  ·  {n_series:,} series", transform=ax.transAxes,
                va="top", color=INK, weight="bold", fontsize=10)
        ax.text(subset["date"].iloc[-1] + pd.Timedelta(days=22), smooth.iloc[-1],
                f"{smooth.iloc[-1]:.2f}", color=color, va="center", fontsize=9)
    axes[-1].set_xlim(daily["date"].min(), daily["date"].max() + pd.Timedelta(days=100))
    format_dates(axes[-1])
    axes[len(labels) // 2].set_ylabel("Units per series per day")
    add_context(fig, "Category trajectories on a comparable scale",
                "28-day rolling mean · units per product–store series, rather than category totals",
                SOURCE)
    export(fig, FIGURES / "03_category_small_multiples")


def plot_states() -> None:
    """三州每序列日销量画在同一标尺上，并直接标注线末端。"""
    daily = read_table("state_daily.csv", dates=True)
    fig, ax = plt.subplots(figsize=(12, 5.3))
    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.16)
    endpoints = []
    for label, color in zip(sorted(daily["state_id"].unique()), (BLUE, ORANGE, TEAL)):
        subset = daily.loc[daily["state_id"] == label].sort_values("date")
        smooth = subset["units_per_series"].rolling(28, min_periods=28).mean()
        ax.plot(subset["date"], smooth, color=color, linewidth=1.8)
        endpoints.append((float(smooth.iloc[-1]), label, color, subset["date"].iloc[-1]))
    # 若三条线在末端接近，轻微错开文字，并用细引导线指回真实数值。
    ordered = sorted(endpoints)
    minimum_gap = max(value for value, *_ in endpoints) * 0.08
    label_y = []
    for value, *_ in ordered:
        label_y.append(max(value, label_y[-1] + minimum_gap) if label_y else value)
    for (value, label, color, end_date), text_y in zip(ordered, label_y):
        text_date = end_date + pd.Timedelta(days=22)
        ax.plot([end_date, text_date], [value, text_y], color=color, linewidth=0.9)
        ax.text(text_date, text_y, f"{label}  {value:.2f}", color=color,
                va="center", fontsize=9, weight="bold")
    ax.set_xlim(daily["date"].min(), daily["date"].max() + pd.Timedelta(days=140))
    ax.set_ylim(0, max(ax.get_ylim()[1], max(label_y) * 1.08))
    ax.set_ylabel("Units per series per day")
    format_dates(ax)
    style_axis(ax)
    add_context(fig, "State-level sales differ after adjusting for series count",
                "CA, TX and WI · 28-day rolling mean · one shared vertical scale",
                SOURCE)
    export(fig, FIGURES / "04_state_comparison")


def plot_zero_shares() -> None:
    """用箱线图看品类内的稀疏程度，而不只比较一个总体均值。"""
    series = read_table("series_summary.csv")
    groups = ["FOODS", "HOUSEHOLD", "HOBBIES"]
    groups = [group for group in groups if group in set(series["cat_id"])]
    values = [series.loc[series["cat_id"] == group, "zero_day_share"].to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(10, 5.7))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.80, bottom=0.16)
    boxes = ax.boxplot(values, orientation="horizontal", positions=np.arange(len(groups)),
                       widths=0.53, patch_artist=True, showfliers=False,
                       medianprops={"color": INK, "linewidth": 2},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.76)
        patch.set_edgecolor("none")
    ax.set_yticks(np.arange(len(groups)), groups)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Share of historical days with zero sales")
    ax.grid(axis="x", color="#DDE4EC", alpha=0.7)
    style_axis(ax, horizontal_grid=False)
    for position, value in enumerate(values):
        median = float(np.median(value))
        ax.text(min(median + 0.025, 0.95), position, f"median {median:.0%}",
                va="center", color=INK, fontsize=9)
    add_context(fig, "Zero-sales frequency varies across product categories",
                "Each box represents product–store series · whiskers show 1.5 × IQR · outliers hidden",
                SOURCE)
    export(fig, FIGURES / "05_zero_sales_by_category")


def plot_price() -> None:
    """选价格变化多的示例，明确按周记录且选择并非总体代表。"""
    prices = read_table("example_price_series.csv", dates=True).sort_values("date")
    if prices.empty:
        raise ValueError("价格示例表为空；请重新运行 03_summarize_eda.py")
    # 最近两年的变化更容易读；原始汇总表仍保留完整历史。
    cutoff = prices["date"].max() - pd.Timedelta(weeks=104)
    shown = prices.loc[prices["date"] >= cutoff]
    item_id, store_id = str(prices["item_id"].iloc[0]), str(prices["store_id"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 5.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.16)
    ax.step(shown["date"], shown["sell_price"], where="post", color=TEAL, linewidth=2)
    ax.scatter(shown["date"].iloc[-1], shown["sell_price"].iloc[-1], color=ORANGE, s=38, zorder=5)
    ax.annotate(f"Last ${shown['sell_price'].iloc[-1]:.2f}",
                xy=(shown["date"].iloc[-1], shown["sell_price"].iloc[-1]),
                xytext=(-70, 25), textcoords="offset points", color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED}, fontsize=9)
    ax.set_ylabel("Weekly listed price (USD)")
    ax.set_xlabel("Week within historical data")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    style_axis(ax)
    add_context(fig, f"An example of changing weekly prices: {item_id} · {store_id}",
                "Last 104 weeks shown · selected from 240 candidate series for price variation; not representative",
                "Source: M5 Forecasting — Accuracy, sell_prices.csv and calendar.csv.")
    export(fig, FIGURES / "06_price_example")


def plot_product_series_examples() -> None:
    """展示每类一条具体序列的真实日销量与短期平滑走势。"""
    daily = read_table("example_series_daily.csv", dates=True)
    selected = read_table("example_series_selection.csv")
    labels = [label for label in ("FOODS", "HOUSEHOLD", "HOBBIES")
              if label in set(daily["category"])]
    if len(labels) != 3 or len(selected) != 3:
        raise ValueError("商品序列示例表应包含三个品类各一条序列")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.12, hspace=0.36)
    last_date = daily["date"].max()
    forecast_start = last_date + pd.Timedelta(days=1)
    forecast_end = last_date + pd.Timedelta(days=28)
    for ax, label, color in zip(axes, labels, COLORS):
        subset = daily.loc[daily["category"] == label].sort_values("date")
        row = selected.loc[selected["cat_id"] == label].iloc[0]
        ax.axvspan(forecast_start, forecast_end, color=PALE, alpha=0.65, zorder=0)
        ax.plot(subset["date"], subset["actual_units"], color=color,
                alpha=0.34, linewidth=0.75, label="Actual daily units")
        smooth = subset["actual_units"].rolling(14, min_periods=1).mean()
        ax.plot(subset["date"], smooth, color=color, linewidth=1.9,
                label="14-day moving average")
        ax.axvline(last_date, color=MUTED, linewidth=0.9, linestyle="--")
        ax.set_ylim(bottom=0)
        style_axis(ax)
        ax.text(0.01, 0.94,
                f"{label}  ·  {row['item_id']} × {row['store_id']}  ·  {row['id']}\n"
                f"Last-year mean {row['mean_daily_units_last365']:.2f} units/day  ·  "
                f"zero-sales days {row['zero_day_share_last365']:.0%}",
                transform=ax.transAxes, va="top", color=INK, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3})
        ax.text(forecast_end - pd.Timedelta(days=1), 0.88, "28-day forecast\n(horizon only)",
                transform=ax.get_xaxis_transform(), ha="right", va="top",
                color=MUTED, fontsize=8)
    axes[-1].set_xlim(daily["date"].min(), forecast_end)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].set_ylabel("Units sold per day (independent scale)")
    add_context(fig, "Daily sales for one representative product in each category",
                "Last 365 observed days · pale line is actual sales, dark line is a 14-day mean · each panel has its own y-axis",
                SOURCE)
    export(fig, FIGURES / "07_product_series_examples")


def main() -> None:
    apply_style()
    # 全部必要汇总表存在后才清理旧图，避免缺表时白白删除可看的结果。
    required = (
        "daily_summary.csv", "weekday_summary.csv", "category_daily.csv",
        "category_summary.csv", "state_daily.csv", "series_summary.csv", "example_price_series.csv",
        "example_series_selection.csv", "example_series_daily.csv",
    )
    missing = [name for name in required if not (TABLES / name).is_file()]
    if missing:
        raise FileNotFoundError(f"请先运行 03_summarize_eda.py；缺少：{missing}")
    clear_previous_figures()
    plot_overview()
    plot_weekday()
    plot_category()
    plot_states()
    plot_zero_shares()
    plot_price()
    plot_product_series_examples()
    print(f"已生成 7 张新图（PNG + SVG）：{FIGURES}")


if __name__ == "__main__":
    main()
