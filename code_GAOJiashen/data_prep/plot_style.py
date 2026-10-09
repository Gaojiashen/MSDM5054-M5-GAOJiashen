"""M5 探索图共用的视觉规则：统一颜色、字体、留白和导出格式。

依据 GitHub anthropics/knowledge-work-plugins 的 data-visualization skill：
颜色表达分组或重点，尽量直接标注曲线，保留来源与时间范围，
避免 3D、双轴和无意义装饰。此模块无需独立运行。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 脚本可在无图形桌面的终端保存图片。
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


INK = "#1E293B"
MUTED = "#65758B"
GRID = "#DDE4EC"
BLUE = "#2463A6"
ORANGE = "#D97932"
TEAL = "#21867A"
PURPLE = "#7357A4"
PALE = "#B9C7D5"
COLORS = (BLUE, ORANGE, TEAL, PURPLE)


def apply_style() -> None:
    """只使用现有 Matplotlib 依赖，不要求用户额外安装绘图库。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",  # SVG 文本可编辑，适合后续写报告。
    })


def style_axis(ax, *, horizontal_grid: bool = True) -> None:
    """简化边框和网格，把视觉注意力留给数据。"""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=7)
    if horizontal_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)


def add_context(fig, title: str, subtitle: str, source: str) -> None:
    """标题说清图的主要信息；副标题说明口径；页脚标明数据来源。"""
    fig.text(0.07, 0.97, title, ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.07, 0.91, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    fig.text(0.07, 0.035, source, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def compact_number(value: float, _position: float | None = None) -> str:
    """销量轴显示 10k、20k，比长串数字更容易扫读。"""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def set_compact_y(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))


def export(fig, path_without_extension: Path) -> None:
    """PNG 便于直接查看，SVG 便于放大和编辑；两者内容一致。"""
    path_without_extension.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_extension.with_suffix(".png"), dpi=180)
    fig.savefig(path_without_extension.with_suffix(".svg"))
    plt.close(fig)
