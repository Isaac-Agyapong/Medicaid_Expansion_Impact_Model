"""Shared chart style for the model project: Swiss minimal (white page, black Arial headlines, a thick black rule
above every chart, one strong colour per meaning). The web app uses the same palette.

    BLUE       states that expanded Medicaid, and the estimated effect of expansion
    TANGERINE  states that had not expanded by 2023 (and what they would gain)
    INK        the highlighted number or finding
    GREY       context
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE, BLUE_L, TANGERINE, TANGERINE_L = "#2f5bea", "#a9bbf5", "#f08a24", "#f9c796"
INK, INK_2, GREY, GREY_L, PAPER, RULE = "#111827", "#4b5563", "#9ca3af", "#e5e7eb", "#ffffff", "#eceef1"
# names kept for the shared plotting code: expansion / not expanded / highlight / light versions
EXP, EXP_L, NONEXP, SUN, GREY_LIGHT = BLUE, BLUE_L, TANGERINE, INK, GREY_L
FONT = ["Arial", "Helvetica", "DejaVu Sans"]

IMAGE_DIR = Path(__file__).resolve().parents[1] / "Image"
IMAGE_DIR.mkdir(exist_ok=True)


def apply():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (9, 4.8),
        "font.family": FONT, "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": INK, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.9,
        "axes.axisbelow": True, "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 30,
        "legend.frameon": False, "lines.linewidth": 2.6,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def title(ax, text, sub=None):
    """Bold black headline with a thick rule above it (Swiss style), grey takeaway underneath."""
    ax.set_title(text, fontsize=15, fontweight="bold", loc="left", pad=30 if sub else 14, color=INK)
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                    color=INK_2, fontsize=10.5, va="bottom")
    ax.annotate("", xy=(1, 1), xycoords="axes fraction", xytext=(0, 1), textcoords="axes fraction")
    fig = ax.figure
    fig.add_artist(plt.Line2D([0.01, 0.99], [1.0, 1.0], transform=fig.transFigure, color=INK, linewidth=4))


def pct(ax, axis="y", decimals=0):
    fmt = FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def source(fig, text="Source: Census SAHIE 2008-2023; KFF expansion tracker. Adults 18-64 at or below 138% of poverty."):
    fig.text(0.01, -0.02, text, color=GREY, fontsize=8, ha="left", va="top")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
