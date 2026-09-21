"""
Shared matplotlib style for every figure in paper.tex: bigger, bolder text
and a consistent, high-contrast palette so figures are still readable at
double-column print size. Import this before creating any figure.
"""
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 17,
    "figure.titleweight": "bold",
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "lines.linewidth": 2.6,
    "lines.markersize": 9,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.9,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BLUE = "#1f4e9c"
ORANGE = "#c8551d"
GREEN = "#1f8a3d"
RED = "#b5223a"
PURPLE = "#6a3d9a"
GRAY = "#888888"
GOLD = "#c99a1a"
TEAL = "#1f9c96"

MODEL_PALETTE = [BLUE, ORANGE, GREEN, PURPLE, GOLD, TEAL, RED, "#5b4636",
                 "#3d7a9c", "#9c3d7a", "#7a9c3d", GRAY]

SAFE_COLOR = GREEN
UNSAFE_COLOR = RED
