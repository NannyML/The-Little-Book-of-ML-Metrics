"""
wMAPE vs MAPE — zero-handling demonstration.

The defining property of wMAPE: it stays finite when actual values include
zeros. MAPE divides each error by its own |Y|, so a single Y=0 row makes
MAPE undefined (or infinite). wMAPE divides the sum of errors by the sum
of |Y|, so one zero just drops out of the denominator and the metric keeps
working.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import NML_CYAN, NML_RED, save_figure


def main():
    # Five observations: same |error| pattern, but the third has Y = 0.
    y     = np.array([100.0, 200.0,   0.0, 50.0, 80.0])
    yhat  = np.array([110.0, 180.0,   5.0, 55.0, 75.0])
    abs_err = np.abs(y - yhat)

    # Per-item MAPE (undefined where y == 0)
    per_item_mape = np.where(y == 0, np.nan, abs_err / np.abs(y) * 100)

    # Aggregates
    mape_value  = np.nanmean(per_item_mape)  # mean ignoring NaN — common workaround
    wmape_value = abs_err.sum() / np.abs(y).sum() * 100

    fig, ax = plt.subplots(figsize=(12, 5.6))

    # Column positions
    cols = {
        "Y":         0.0,
        "Ŷ":         1.3,
        "|error|":   2.6,
        "MAPE":      4.5,
        "wMAPE\ncontribution": 6.4,
    }
    header_y = 5.6
    for name, x in cols.items():
        ax.text(x, header_y, name, ha="center", va="bottom",
                fontsize=12, color="black", fontweight="bold")

    # Underline header
    ax.plot([-0.7, 7.4], [header_y - 0.15, header_y - 0.15],
            color="lightgray", linewidth=0.8)

    n = len(y)
    for i in range(n):
        row_y = n - 1 - i + 0.0
        zero_row = y[i] == 0

        ax.text(cols["Y"], row_y, f"{y[i]:.0f}",
                ha="center", va="center", fontsize=14,
                color=NML_RED if zero_row else "black",
                fontweight="bold" if zero_row else "normal")
        ax.text(cols["Ŷ"], row_y, f"{yhat[i]:.0f}",
                ha="center", va="center", fontsize=14, color="black")
        ax.text(cols["|error|"], row_y, f"{abs_err[i]:.0f}",
                ha="center", va="center", fontsize=14, color="black")

        if zero_row:
            ax.text(cols["MAPE"], row_y, "undefined",
                    ha="center", va="center", fontsize=13,
                    color=NML_RED, style="italic")
        else:
            ax.text(cols["MAPE"], row_y, f"{per_item_mape[i]:.1f}%",
                    ha="center", va="center", fontsize=14, color=NML_RED)

        ax.text(cols["wMAPE\ncontribution"], row_y,
                f"{abs_err[i]:.0f} / {np.abs(y).sum():.0f}",
                ha="center", va="center", fontsize=13, color=NML_CYAN)

    # Divider above the totals
    ax.plot([-0.7, 7.4], [-0.55, -0.55],
            color="lightgray", linewidth=0.8)

    # Totals — stacked vertically, centered, with the asymmetry of behavior
    # made visible: MAPE has to throw a row away; wMAPE keeps it.
    center_x = 3.3
    ax.text(center_x, -1.1,
            f"MAPE = mean of column = {mape_value:.1f}%   (only by dropping the Y=0 row)",
            ha="center", va="center",
            fontsize=12, color=NML_RED)
    ax.text(center_x, -1.7,
            f"wMAPE = $\\sum$|error| / $\\sum$|Y| = {abs_err.sum():.0f} / {np.abs(y).sum():.0f} = {wmape_value:.1f}%   (Y=0 row contributes 0 to the denominator, finite result)",
            ha="center", va="center",
            fontsize=12, color=NML_CYAN)

    # Highlight the Y = 0 row
    zero_idx = int(np.argmin(y))
    zero_row_y = n - 1 - zero_idx
    ax.add_patch(Rectangle(
        (-0.7, zero_row_y - 0.45), 8.1, 0.9,
        facecolor=NML_RED, alpha=0.06, edgecolor="none",
    ))
    ax.text(-1.0, zero_row_y, "Y = 0:",
            ha="right", va="center",
            fontsize=12, color=NML_RED, fontweight="bold")
    ax.text(-1.0, zero_row_y - 0.45, "MAPE undefined,\nwMAPE keeps working",
            ha="right", va="top", fontsize=10,
            color=NML_RED, style="italic")

    ax.set_xlim(-3.8, 8.2)
    ax.set_ylim(-2.4, 6.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    save_figure(fig, "wMAPE_compare_MAPE")


if __name__ == "__main__":
    main()
