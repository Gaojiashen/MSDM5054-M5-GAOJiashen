"""Save figures for the complete M5 x-selection process.

Run this *after* 02_select_x_wrmsse.py. The first run additionally scores
shared x=5 on the 16 confirmation origins, because the original selection run
only scored x=5 on development origins. The report covers the shared-x curve,
all 729 category vectors, the confirmation gate, the audit, and an item example.
This script never changes selection_summary.json or evaluates d_1914..d_1941.

Outputs: outputs/evaluation/x_selection/figures/*.svg and *.png,
         outputs/evaluation/x_selection/shared_x5_confirmation_scores.csv.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save files without opening a GUI window.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "evaluation" / "x_selection"
FIGURES = OUT / "figures"
SCORES_FILE = OUT / "rolling_origin_scores.csv"
MEANS_FILE = OUT / "development_vector_means.csv"
STAGES_FILE = OUT / "selected_vector_split_scores.csv"
EXTRA_FILE = OUT / "shared_x5_confirmation_scores.csv"
CALENDAR_FILE = ROOT / "data" / "processed" / "calendar_clean.csv"
EXAMPLE_FILE = ROOT / "outputs" / "data_prep" / "tables" / "example_series_daily.csv"
EXAMPLE_SELECTION_FILE = ROOT / "outputs" / "data_prep" / "tables" / "example_series_selection.csv"
MODEL_FILE = EVALUATION / "02_select_x_wrmsse.py"

BLUE = "#236B8E"
ORANGE = "#D77A35"
INK = "#253544"
MUTED = "#70808C"
GRID = "#E3E9ED"
LIGHT = "#F4F7F9"


def require(path: Path) -> Path:
    """Give a useful message if an earlier project stage has not run."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def load_model_module():
    """Import the existing scorer without triggering its main() function."""
    spec = importlib.util.spec_from_file_location("m5_x_scorer", require(MODEL_FILE))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import scorer from {MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_missing_x5_confirmation() -> pd.DataFrame:
    """Score shared x=5 on the same confirmation origins as shared x=8.

    The x=5 calculations are cached in their own CSV. We do not alter the
    original gate decision: this is a follow-up comparison prompted by the
    closeness of the development scores, not an independent new holdout.
    """
    scorer = load_model_module()
    expected = list(scorer.CONFIRM_ORIGINS)
    if EXTRA_FILE.is_file():
        cached = pd.read_csv(EXTRA_FILE)
        if sorted(cached["origin_d"].tolist()) == expected:
            return cached
        raise ValueError(f"Cached confirmation origins differ from the scorer: {EXTRA_FILE}")

    meta = pd.read_csv(require(scorer.PROCESSED / "sales_metadata.csv"))
    train = np.load(require(scorer.PROCESSED / "sales_matrix.npy"), mmap_mode="r")
    calendar = pd.read_csv(require(scorer.RAW / "calendar.csv"), usecols=["d", "wm_yr_wk"])
    calendar["d_num"] = calendar["d"].str.removeprefix("d_").astype(int)
    prices = pd.read_csv(
        require(scorer.RAW / "sell_prices.csv"),
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    )
    aggs = scorer.hierarchy(meta)
    layout = scorer.category_layout(meta, aggs)
    first_release = scorer.first_price_days(meta, prices, calendar)
    vector = (5, 5, 5)
    rows = []
    for origin in expected:
        scores, _coverage = scorer.evaluate_origin(
            meta, train, calendar, prices, first_release, aggs, layout,
            origin, [vector],
        )
        rows.append({"origin_d": origin, "FOODS": 5, "HOBBIES": 5,
                     "HOUSEHOLD": 5, "wrmsse": float(scores[0])})
        print(f"scored x=5 confirmation origin d_{origin}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(EXTRA_FILE, index=False)
    return result


def shared_rows(scores: pd.DataFrame, x: int, origins: list[int]) -> pd.DataFrame:
    """Find exactly one score for shared x at every requested origin."""
    rows = scores.loc[
        scores["origin_d"].isin(origins)
        & scores["FOODS"].eq(x)
        & scores["HOBBIES"].eq(x)
        & scores["HOUSEHOLD"].eq(x),
        ["origin_d", "wrmsse"],
    ].sort_values("origin_d")
    if rows["origin_d"].tolist() != origins:
        raise ValueError(f"Scores for shared x={x} do not cover the requested origins once each")
    return rows.reset_index(drop=True)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shared_curve(means: pd.DataFrame) -> pd.DataFrame:
    """Show the full baseline gap and a separate zoom for close candidates."""
    shared = means.loc[
        means["FOODS"].eq(means["HOBBIES"])
        & means["HOBBIES"].eq(means["HOUSEHOLD"]),
        ["FOODS", "wrmsse"],
    ].rename(columns={"FOODS": "x"}).sort_values("x")
    if shared["x"].tolist() != [1, 2, 3, 4, 5, 6, 8, 12, 16]:
        raise ValueError("Development means do not contain the nine shared-x candidates")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    for ax in axes:
        style_axes(ax)
        ax.plot(shared["x"], shared["wrmsse"], color=MUTED, linewidth=1.8,
                marker="o", markersize=5, zorder=3)
        ax.scatter([5, 8], shared.set_index("x").loc[[5, 8], "wrmsse"],
                   s=85, color=[ORANGE, BLUE], edgecolor="white", linewidth=1.3, zorder=5)
        ax.set_xticks(shared["x"])
        ax.set_xlabel("Number of past same-weekday observations (x)")
        ax.set_ylabel("Mean WRMSSE · development windows")
    axes[0].set_title("Averaging several weeks beats x=1", loc="left", color=INK, weight="bold")
    axes[1].set_xlim(3.5, 16.5)
    close = shared.loc[shared["x"] >= 4, "wrmsse"]
    axes[1].set_ylim(close.min() - 0.003, close.max() + 0.003)
    axes[1].set_title("Among longer windows, x=5 and x=8 are close", loc="left", color=INK, weight="bold")
    for x, color in ((5, ORANGE), (8, BLUE)):
        y = float(shared.loc[shared["x"].eq(x), "wrmsse"].iloc[0])
        axes[1].annotate(f"x={x}: {y:.4f}", (x, y), xytext=(7, -19 if x == 5 else 11),
                         textcoords="offset points", color=color, fontsize=10, weight="bold")
    save_figure(fig, "01_shared_x_curve")
    return shared


def plot_joint_profiles(means: pd.DataFrame) -> None:
    """Show how each category's x behaves when the other two can vary.

    Each point is the best *complete three-category vector* at that category's
    fixed x. It is not a score obtained by changing one category in isolation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True,
                             constrained_layout=True)
    for ax, category in zip(axes, ("FOODS", "HOBBIES", "HOUSEHOLD")):
        profile = means.groupby(category, as_index=False)["wrmsse"].min()
        profile = profile.sort_values(category)
        best = profile.loc[profile["wrmsse"].idxmin()]
        style_axes(ax)
        ax.plot(profile[category], profile["wrmsse"], marker="o", color=MUTED,
                linewidth=1.7, markersize=5)
        ax.scatter([best[category]], [best["wrmsse"]], s=100, color=BLUE,
                   edgecolor="white", linewidth=1.3, zorder=5)
        ax.set_xticks(profile[category])
        ax.set_title(category, loc="left", color=INK, weight="bold")
        ax.set_xlabel("Fixed x for this category")
        ax.annotate(f"best x={int(best[category])}",
                    (best[category], best["wrmsse"]), xytext=(7, 11),
                    textcoords="offset points", fontsize=9, color=BLUE)
    axes[0].set_ylabel("Best complete-vector WRMSSE · development")
    fig.suptitle("Joint search: fix one category, optimize the other two",
                 x=0.01, ha="left", color=INK, weight="bold", fontsize=14)
    save_figure(fig, "02_joint_vector_profiles")


def plot_all_vectors(means: pd.DataFrame) -> None:
    """Show every three-category combination as nine comparable heatmaps."""
    xs = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    values = means["wrmsse"].to_numpy(dtype=float)
    low, high = np.quantile(values, [0.05, 0.95])
    if not low < high:
        raise ValueError("The joint-vector scores have no usable color range")
    lookup = means.set_index(["FOODS", "HOBBIES", "HOUSEHOLD"])["wrmsse"]
    if len(lookup) != 729 or not lookup.index.is_unique:
        raise ValueError("Expected all 729 unique joint x-vectors")
    winner = means.loc[means["wrmsse"].idxmin()]
    winner_vector = (int(winner["FOODS"]), int(winner["HOBBIES"]),
                     int(winner["HOUSEHOLD"]))
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    last_image = None
    for ax, household_x in zip(axes.flat, xs):
        grid = np.array([
            [lookup.loc[(food_x, hobby_x, household_x)] for hobby_x in xs]
            for food_x in xs
        ], dtype=float)
        last_image = ax.imshow(grid, cmap="cividis", vmin=low, vmax=high,
                               interpolation="nearest", aspect="equal")
        ax.set_title(f"HOUSEHOLD x={household_x}", loc="left", color=INK,
                     weight="bold", fontsize=11)
        ax.set_xticks(range(9), xs, fontsize=8)
        ax.set_yticks(range(9), xs, fontsize=8)
        ax.set_xlabel("HOBBIES x", fontsize=9)
        ax.set_ylabel("FOODS x", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if household_x == winner_vector[2]:
            ax.add_patch(Rectangle((xs.index(winner_vector[1]) - 0.5,
                                    xs.index(winner_vector[0]) - 0.5), 1, 1, fill=False,
                                   edgecolor=ORANGE, linewidth=2.5))
    fig.colorbar(last_image, ax=axes, shrink=0.72,
                 label="Mean development WRMSSE (colors clipped at 5th–95th percentile)")
    fig.suptitle(f"All 729 category vectors · orange outline = development winner {winner_vector}",
                 color=INK, weight="bold", fontsize=15)
    save_figure(fig, "06_all_729_category_vectors")


def plot_selection_stages(stages: pd.DataFrame) -> None:
    """Trace the fixed development winners through confirmation and audit."""
    choices = {
        "development_selected_joint_vector": ("Joint category vector", ORANGE, "-"),
        "development_selected_shared_x": ("Best shared x", BLUE, "-"),
        "seasonal_naive_x1": ("sNaive x=1", MUTED, "--"),
    }
    order = ("development", "confirmation", "audit")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True,
                             constrained_layout=True)
    contender_values = []
    for ax, methods, title in (
        (axes[0], tuple(choices), "All three methods"),
        (axes[1], tuple(choices)[:2], "Zoom: category vector vs shared x"),
    ):
        style_axes(ax)
        for method in methods:
            label, color, line = choices[method]
            part = stages.loc[stages["method"].eq(method)].set_index("split")
            if not set(order).issubset(part.index):
                raise ValueError(f"Missing stage scores for {method}")
            values = part.loc[list(order), "mean_wrmsse"].to_numpy(dtype=float)
            ax.plot(range(3), values, color=color, linestyle=line, linewidth=2.2,
                    marker="o", markersize=7, label=label)
            if ax is axes[1]:
                contender_values.extend(values)
                for i, value in enumerate(values):
                    ax.annotate(f"{value:.3f}", (i, value), xytext=(5, 7),
                                textcoords="offset points", color=color, fontsize=9)
        ax.set_xticks(range(3), ["Development", "Confirmation", "Audit"])
        ax.set_ylabel("Mean WRMSSE across origins")
        ax.set_title(title, loc="left", color=INK, weight="bold")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    axes[1].set_ylim(min(contender_values) - 0.012, max(contender_values) + 0.012)
    save_figure(fig, "03_selection_stages")


def paired_table(scores: pd.DataFrame, origins: list[int], split: str,
                 date_lookup: pd.Series) -> pd.DataFrame:
    x5 = shared_rows(scores, 5, origins)
    x8 = shared_rows(scores, 8, origins)
    result = x5.rename(columns={"wrmsse": "wrmsse_x5"})
    result["wrmsse_x8"] = x8["wrmsse"].to_numpy()
    result["delta_x8_minus_x5"] = result["wrmsse_x8"] - result["wrmsse_x5"]
    result["split"] = split
    result["origin_date"] = pd.to_datetime(result["origin_d"].map(date_lookup))
    if result["origin_date"].isna().any():
        raise ValueError("A rolling origin has no calendar date")
    return result


def plot_paired(paired: pd.DataFrame) -> None:
    """Negative differences favor x=8; two panels keep time order visible."""
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.8), constrained_layout=True)
    for ax, split in zip(axes, ("development", "confirmation")):
        part = paired.loc[paired["split"].eq(split)].sort_values("origin_d")
        delta = part["delta_x8_minus_x5"].to_numpy()
        positions = np.arange(len(part))
        colors = np.where(delta < 0, BLUE, ORANGE)
        style_axes(ax)
        ax.bar(positions, delta, color=colors, width=0.75, zorder=3)
        ax.axhline(0, color=INK, linewidth=1)
        ax.axhline(delta.mean(), color=INK, linestyle="--", linewidth=1.3,
                   label=f"Mean x8 - x5 = {delta.mean():+.4f}")
        ax.set_title(f"{split.title()}: x=8 wins {int((delta < 0).sum())}/{len(delta)} windows",
                     loc="left", color=INK, weight="bold")
        ax.set_ylabel("WRMSSE(x=8) − WRMSSE(x=5)")
        step = 4 if split == "development" else 2
        ticks = positions[::step]
        ax.set_xticks(ticks, part["origin_date"].dt.strftime("%Y-%m").iloc[::step])
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Forecast origin (28 days between bars)")
    save_figure(fig, "04_x8_minus_x5_by_origin")


def plot_example(calendar: pd.DataFrame) -> None:
    """Illustrate one preselected HOBBIES item; this is not model selection."""
    selection = pd.read_csv(require(EXAMPLE_SELECTION_FILE))
    item = selection.loc[selection["cat_id"].eq("HOBBIES"), "id"]
    if len(item) != 1:
        raise ValueError("Expected exactly one preselected HOBBIES example series")
    item_id = str(item.iloc[0])
    daily = pd.read_csv(require(EXAMPLE_FILE), parse_dates=["date"])
    daily = daily.loc[daily["id"].eq(item_id)].set_index("date").sort_index()
    origin_date = pd.to_datetime(calendar.loc[calendar["d_num"].eq(1885), "date"].iloc[0])
    future_monday = origin_date + pd.Timedelta(days=1)
    history_dates = pd.date_range(end=future_monday - pd.Timedelta(days=7), periods=8, freq="7D")
    future_dates = pd.date_range(start=future_monday, periods=4, freq="7D")
    needed = history_dates.append(future_dates)
    if not needed.isin(daily.index).all():
        raise ValueError("The preselected example does not cover all required Mondays")
    past = daily.loc[history_dates, "actual_units"].to_numpy(dtype=float)
    actual = daily.loc[future_dates, "actual_units"].to_numpy(dtype=float)
    pred5, pred8 = float(past[-5:].mean()), float(past.mean())
    sse5 = float(((actual - pred5) ** 2).sum())
    sse8 = float(((actual - pred8) ** 2).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True, constrained_layout=True)
    for ax in axes:
        style_axes(ax)
        ax.set_ylim(0, max(3.5, past.max(), actual.max()) + 0.5)
        ax.set_ylabel("Units sold on Monday")
    axes[0].bar(history_dates, past, width=4, color=MUTED, zorder=3)
    axes[0].axvspan(history_dates[3] - pd.Timedelta(days=3),
                    history_dates[-1] + pd.Timedelta(days=3), color=LIGHT, zorder=0)
    axes[0].set_title("Eight observed Mondays before the origin", loc="left", color=INK, weight="bold")
    axes[0].text(0.02, 0.94, "Shaded: the latest five Mondays", transform=axes[0].transAxes,
                 va="top", color=MUTED, fontsize=9)
    axes[1].bar(future_dates, actual, width=4, color=INK, alpha=0.82, label="Actual", zorder=3)
    axes[1].axhline(pred5, color=ORANGE, linestyle="--", linewidth=2,
                    label=f"x=5 forecast: {pred5:g}")
    axes[1].axhline(pred8, color=BLUE, linewidth=2,
                    label=f"x=8 forecast: {pred8:g}")
    axes[1].set_title("Four future Mondays (example only)", loc="left", color=INK, weight="bold")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"{item_id} · Monday squared error: x=5 {sse5:.1f}, x=8 {sse8:.1f}",
                 x=0.01, ha="left", color=INK, weight="bold", fontsize=13)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=25)
    save_figure(fig, "05_hobbies_monday_example")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(require(SCORES_FILE))
    means = pd.read_csv(require(MEANS_FILE))
    stages = pd.read_csv(require(STAGES_FILE))
    scorer = load_model_module()
    extra = score_missing_x5_confirmation()
    combined = pd.concat([scores, extra], ignore_index=True)
    # If a previous run already included x=5, keep one record per origin/vector.
    combined = combined.drop_duplicates(["origin_d", "FOODS", "HOBBIES", "HOUSEHOLD"], keep="first")
    calendar = pd.read_csv(require(CALENDAR_FILE), usecols=["d_num", "date"])
    date_lookup = calendar.set_index("d_num")["date"]
    plot_shared_curve(means)
    plot_joint_profiles(means)
    plot_all_vectors(means)
    plot_selection_stages(stages)
    paired = pd.concat([
        paired_table(combined, list(scorer.DEV_ORIGINS), "development", date_lookup),
        paired_table(combined, list(scorer.CONFIRM_ORIGINS), "confirmation", date_lookup),
    ], ignore_index=True)
    paired.to_csv(OUT / "paired_x5_x8_scores.csv", index=False, date_format="%Y-%m-%d")
    plot_paired(paired)
    plot_example(calendar)
    print(f"Figures: {FIGURES}")
    print(f"Supporting paired scores: {OUT / 'paired_x5_x8_scores.csv'}")


if __name__ == "__main__":
    main()
