"""Regenerate D2_Log_Loss_curve.png using Tufte's principles.

Tufte moves applied:
- Direct labels along the trajectory of each line (no legend)
- Marker on each line at the D2=0 crossing (where L_model = L_null);
  the most important conceptual point in the plot
- L_null annotated next to each marker (staggered above/below to avoid crowding)
- Range-frame spines: only bottom + left, spanning the data range
- Lighter line weight than the brand default to avoid the 1+1=3 phantom
  effect between three roughly parallel lines
- Subtle horizontal baseline at y=0 (whisper, not shout)
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *


def baseline_log_loss(p):
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def d2(model_log_loss, null_log_loss):
    return 1 - model_log_loss / null_log_loss


# (p_positive, line_label, line_label_pos, lnull_label_pos, color)
imbalances = [
    (0.5,  '50/50 balanced',       (1.02, -0.44), (0.69,  0.12), start_color),
    (0.2,  '80/20 imbalanced',     (1.02, -1.00), (0.50, -0.20), middle_color),
    (0.05, '95/5 heavy imbalance', (0.55, -1.40), (0.20,  0.12), end_color),
]

model_ll = np.linspace(0.01, 1.0, 200)

fig, ax = create_line_plot()

# Subtle baseline at D2=0
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.0, alpha=0.45, zorder=0)

for p, label, (lx, ly), (nx, ny), color in imbalances:
    null_ll = baseline_log_loss(p)
    d2_vals = d2(model_ll, null_ll)

    # Line
    ax.plot(model_ll, d2_vals, c=color, linewidth=4, solid_capstyle='round',
            clip_on=True)

    # D2=0 crossing marker
    ax.plot(null_ll, 0, 'o', color=color, markersize=14,
            markerfacecolor='white', markeredgecolor=color,
            markeredgewidth=3, zorder=5)

    # Direct line label along trajectory
    ax.text(lx, ly, label, color=color, fontsize=18,
            va='center', ha='left', fontweight='normal')

    # L_null label, staggered above/below to avoid crowding
    va = 'bottom' if ny > 0 else 'top'
    ax.text(nx, ny, f'$L_{{\\mathrm{{null}}}}={null_ll:.2f}$',
            color=color, fontsize=13, ha='center', va=va, fontweight='normal')

# Range-frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(0, 1.0)
ax.spines['left'].set_bounds(-1.5, 1.0)

ax.set_xlabel('Model log loss', fontsize=18)
ax.set_ylabel(r'$D^2$ log loss score', fontsize=18)
ax.tick_params(axis='both', labelsize=14)

ax.set_xlim(-0.02, 1.65)
ax.set_ylim(-1.55, 1.15)

# Range-frame: drop ticks beyond the actual data range (data ends at L_model = 1.0)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0])

save_figure(fig, 'D2_Log_Loss_curve')
plt.close()
print('D2 Log Loss (Tufte v4): done')
