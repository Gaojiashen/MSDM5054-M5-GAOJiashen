"""Visualize M5 calendar information without reading future sales.

Run this script after downloading the raw M5 files. It creates three PNG/SVG
figure pairs: a training-calendar overview, a representative 28-day training
example from the prior year, and the exact 28 days to be forecast.

This is descriptive visualization only. Calendar markers are not estimated
sales effects, and this file does not fit or evaluate a forecasting model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from plot_style import (
    BLUE, GRID, INK, MUTED, ORANGE, TEAL,
    add_context, apply_style, export, style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
CALENDAR_FILE = ROOT / "data" / "m5-forecasting-accuracy" / "calendar.csv"
OUT = ROOT / "outputs" / "data_prep" / "calendar" / "figures"
TRAIN_END = 1913
FORECAST_FIRST = 1914
FORECAST_LAST = 1941
EXAMPLE_FIRST = pd.Timestamp("2015-04-25")
EXAMPLE_LAST = pd.Timestamp("2015-05-22")
EVENT_NAMES = ("event_name_1", "event_name_2")
EVENT_TYPES = ("event_type_1", "event_type_2")
SNAP_COLUMNS = ("snap_CA", "snap_TX", "snap_WI")
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def read_calendar() -> pd.DataFrame:
    """Read and validate the date and marker columns used by the figures."""
    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(f"Missing M5 calendar: {CALENDAR_FILE}")
    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["date"])
    required = {"date", "d", "weekday", "wm_yr_wk", *EVENT_NAMES,
                *EVENT_TYPES, *SNAP_COLUMNS}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"Calendar is missing required columns: {sorted(missing)}")
    calendar["d_num"] = calendar["d"].str.removeprefix("d_").astype(int)
    calendar = calendar.sort_values("d_num").reset_index(drop=True)
    if calendar.d_num.duplicated().any() or calendar.date.duplicated().any():
        raise ValueError("Calendar has duplicate day IDs or dates")
    if calendar.loc[calendar.d_num == TRAIN_END].empty:
        raise ValueError("Calendar does not contain d_1913")
    for column in (*EVENT_NAMES, *EVENT_TYPES):
        calendar[column] = calendar[column].fillna("").astype(str).str.strip()
    for column in SNAP_COLUMNS:
        calendar[column] = pd.to_numeric(calendar[column], errors="raise")
        if not calendar[column].isin((0, 1)).all():
            raise ValueError(f"{column} must contain only 0 or 1")
    return calendar


def validate_28_days(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Guard against silently plotting an incomplete or out-of-order period."""
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) != 28:
        raise ValueError(f"{label}: expected 28 calendar days, found {len(frame)}")
    if not np.all(np.diff(frame.d_num.to_numpy()) == 1):
        raise ValueError(f"{label}: M5 day IDs are not consecutive")
    if not (frame.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError(f"{label}: calendar dates are not consecutive")
    return frame


def event_records(frame: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Return both event fields, retaining two events recorded on one day."""
    records = []
    for day_index, row in frame.iterrows():
        for name_col, type_col in zip(EVENT_NAMES, EVENT_TYPES):
            name = row[name_col]
            if name:
                records.append((day_index, name, row[type_col]))
    return records


def plot_training_overview(train: pd.DataFrame) -> None:
    """Show the variety and frequency of markers across the full training era."""
    has_event = train[list(EVENT_NAMES)].ne("").any(axis=1)
    month = train.date.dt.month
    monthly_event_rate = has_event.groupby(month).mean().reindex(range(1, 13))
    snap_by_month = train.groupby(month)[list(SNAP_COLUMNS)].mean().reindex(range(1, 13))

    types = pd.concat([train[column] for column in EVENT_TYPES], ignore_index=True)
    type_counts = types[types.ne("")].value_counts().sort_values()

    fig = plt.figure(figsize=(13.8, 8.3))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1),
                           left=0.08, right=0.95, top=0.81, bottom=0.10,
                           wspace=0.24, hspace=0.48)
    ax_month = fig.add_subplot(grid[0, 0])
    ax_types = fig.add_subplot(grid[0, 1])
    ax_snap = fig.add_subplot(grid[1, :])

    x = np.arange(12)
    ax_month.bar(x, monthly_event_rate.to_numpy(), color=BLUE, width=0.66)
    ax_month.set_xticks(x, MONTH_LABELS)
    ax_month.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_month.set_ylabel("Days with at least one event")
    ax_month.set_title("Event-day frequency by month", loc="left", pad=12,
                       fontsize=12, weight="bold")
    ax_month.set_ylim(0, max(0.16, float(monthly_event_rate.max()) * 1.18))
    style_axis(ax_month)

    ax_types.barh(type_counts.index, type_counts.to_numpy(), color=ORANGE, height=0.62)
    ax_types.set_title("Recorded event types", loc="left", pad=12,
                       fontsize=12, weight="bold")
    ax_types.set_xlabel("Number of event records")
    ax_types.set_xlim(0, max(1.0, float(type_counts.max()) * 1.18))
    for y, count in enumerate(type_counts.to_numpy()):
        ax_types.text(count + max(1, type_counts.max() * 0.02), y, str(count),
                      va="center", color=INK, fontsize=9)
    style_axis(ax_types, horizontal_grid=False)
    ax_types.grid(axis="x", color=GRID, linewidth=0.7)
    ax_types.set_axisbelow(True)

    # This heatmap is a share of calendar days, not a share of sales.
    matrix = snap_by_month.to_numpy(dtype=float).T
    heat = ax_snap.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.5)
    ax_snap.set_xticks(x, MONTH_LABELS)
    ax_snap.set_yticks(range(3), ("California", "Texas", "Wisconsin"))
    ax_snap.set_title("Share of dates marked SNAP, by state and month",
                      loc="left", pad=12, fontsize=12, weight="bold")
    for row in range(3):
        for col in range(12):
            value = matrix[row, col]
            ax_snap.text(col, row, f"{value:.0%}", ha="center", va="center",
                         fontsize=8.5, color="white" if value >= 0.34 else INK)
    ax_snap.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax_snap.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_snap.grid(which="minor", color="white", linewidth=1.5)
    ax_snap.tick_params(which="both", length=0)
    for spine in ax_snap.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=ax_snap, pad=0.015, fraction=0.022)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.set_label("Marked dates", color=MUTED)

    start = train.date.min().date().isoformat()
    end = train.date.max().date().isoformat()
    add_context(
        fig,
        "Training calendar | markers available before forecasting",
        f"d_1–d_{TRAIN_END} · {start} to {end} · descriptive frequencies, not sales effects",
        "Source: M5 calendar.csv. Monthly rates use observed dates in each month; event types count records.",
    )
    export(fig, OUT / "01_training_calendar_overview")


def choose_annotation_lanes(records: list[tuple[int, str, str]]) -> list[int]:
    """Separate labels for nearby events so their names remain readable."""
    last_at_lane = [-100] * 4
    lanes = []
    for day_index, _, _ in records:
        eligible = [lane for lane, last in enumerate(last_at_lane)
                    if day_index - last >= 7]
        lane = eligible[0] if eligible else int(np.argmin(last_at_lane))
        last_at_lane[lane] = day_index
        lanes.append(lane)
    return lanes


def plot_28_day_strip(frame: pd.DataFrame, *, title: str,
                      subtitle: str, filename: str) -> None:
    """Use identical axes for the training example and forecast calendar."""
    records = event_records(frame)
    lanes = choose_annotation_lanes(records)
    fig = plt.figure(figsize=(16.5, 6.8))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1),
                           left=0.075, right=0.97, top=0.77,
                           bottom=0.16, hspace=0.13)
    ax_events = fig.add_subplot(grid[0])
    ax_snap = fig.add_subplot(grid[1], sharex=ax_events)
    ax_events.axhline(0.10, color=GRID, linewidth=2)
    ax_events.set_xlim(-0.8, 27.8)
    ax_events.set_ylim(-0.12, 2.4)
    ax_events.set_yticks([])
    ax_events.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_events.spines.values():
        spine.set_visible(False)
    ax_events.text(-0.7, 2.20, "RECORDED EVENTS", color=MUTED,
                   fontsize=9, weight="bold")
    if not records:
        ax_events.text(13.5, 1.25, "No named events in this 28-day period",
                       ha="center", va="center", color=MUTED)
    for (day_index, name, event_type), lane in zip(records, lanes):
        label = name if not event_type else f"{name}\n{event_type}"
        label_x = day_index + (0.30 if day_index < 23 else -0.30)
        ax_events.scatter(day_index, 0.10, s=68, color=ORANGE,
                          edgecolor="white", linewidth=1.3, zorder=4)
        ax_events.annotate(
            label, xy=(day_index, 0.12), xytext=(label_x, 0.70 + lane * 0.43),
            ha="left" if day_index < 23 else "right", va="bottom",
            color=INK, fontsize=8.6, linespacing=1.15,
            bbox={"boxstyle": "round,pad=0.23", "facecolor": "white",
                  "edgecolor": "#D7E0E8", "linewidth": 0.7},
            arrowprops={"arrowstyle": "-", "color": ORANGE,
                        "linewidth": 0.9, "shrinkA": 2, "shrinkB": 4},
        )

    data = frame[list(SNAP_COLUMNS)].to_numpy(dtype=int).T
    cmap = ListedColormap(["#E9F0F5", TEAL])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax_snap.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax_snap.set_yticks(range(3), ("California", "Texas", "Wisconsin"))
    ax_snap.set_ylabel("SNAP flag", labelpad=16)
    ax_snap.set_xticks(np.arange(28))
    ax_snap.set_xticklabels(
        [f"{d.month}/{d.day}\n{d.strftime('%a')}" for d in frame.date],
        fontsize=7.2, rotation=45, ha="right", rotation_mode="anchor",
    )
    ax_snap.set_xticks(np.arange(-0.5, 28, 1), minor=True)
    ax_snap.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_snap.grid(which="minor", color="white", linewidth=1.2)
    ax_snap.tick_params(which="both", length=0)
    for spine in ax_snap.spines.values():
        spine.set_visible(False)
    for boundary in (6.5, 13.5, 20.5):
        ax_events.axvline(boundary, color=GRID, linewidth=1, linestyle="--", zorder=0)
        ax_snap.axvline(boundary, color="#9AAFC2", linewidth=1.5)
    fig.legend(handles=[Patch(facecolor=TEAL, label="SNAP = 1"),
                        Patch(facecolor="#E9F0F5", label="SNAP = 0")],
               loc="lower right", bbox_to_anchor=(0.97, 0.10),
               ncol=2, frameon=False, fontsize=9)
    add_context(
        fig, title, subtitle,
        "Source: M5 calendar.csv. Event and SNAP markers are known dates; colors do not show sales or estimated effects.",
    )
    export(fig, OUT / filename)


def main() -> None:
    apply_style()
    calendar = read_calendar()
    train = calendar.loc[calendar.d_num <= TRAIN_END].copy()
    if len(train) != TRAIN_END or not np.array_equal(
        train.d_num.to_numpy(), np.arange(1, TRAIN_END + 1)
    ):
        raise ValueError("Training calendar must contain exactly d_1..d_1913")
    example = validate_28_days(
        train.loc[train.date.between(EXAMPLE_FIRST, EXAMPLE_LAST)].copy(),
        "Training example",
    )
    future = validate_28_days(
        calendar.loc[calendar.d_num.between(FORECAST_FIRST, FORECAST_LAST)].copy(),
        "Forecast horizon",
    )
    plot_training_overview(train)
    plot_28_day_strip(
        example,
        title="Training example | 28 days in the prior spring",
        subtitle=f"{EXAMPLE_FIRST.date()} to {EXAMPLE_LAST.date()} · same season as the forecast, observed before d_1913",
        filename="02_training_28_day_example",
    )
    plot_28_day_strip(
        future,
        title="Forecast calendar | the exact 28 days to predict",
        subtitle=f"d_{FORECAST_FIRST}–d_{FORECAST_LAST} · {future.date.min().date()} to {future.date.max().date()} · calendar known before forecasting",
        filename="03_forecast_28_days",
    )
    print(f"Saved three PNG/SVG figure pairs to {OUT}")


if __name__ == "__main__":
    main()
