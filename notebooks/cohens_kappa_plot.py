"""Regenerate Cohens_Kappa_levels.png — replace the simple interpretation-band
bar chart with a plot showing what Cohen's Kappa actually does.

Setup: two annotators that label items completely at random, but each matches
the class distribution marginally (i.e., they have no skill, just guess with
the right base rate). As class imbalance grows, raw observed agreement climbs
toward 1 — looking like the annotators are doing well — but Cohen's Kappa
correctly stays at 0 across the full range.

Tufte style: direct labels at line ends, range-frame, no legend, lighter
line weight than brand default, subtle baselines.
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *


p_pos = np.linspace(0.50, 0.99, 200)

# Two random-but-marginal-matching annotators. Each labels positive with prob p.
# Raw observed agreement = both positive OR both negative = p^2 + (1-p)^2
raw_agreement = p_pos**2 + (1 - p_pos)**2

# Expected (chance) agreement equals raw agreement in this case --> kappa = 0
kappa = np.zeros_like(p_pos)

fig, ax = create_line_plot()

# Subtle reference at 0 and 1
ax.axhline(y=1, color='gray', linewidth=0.7, alpha=0.3, zorder=0)
ax.axhline(y=0, color='gray', linewidth=0.7, alpha=0.3, zorder=0)

# Raw agreement line
ax.plot(p_pos, raw_agreement, c=end_color, linewidth=4, solid_capstyle='round')
# Cohen's Kappa line (flat at 0)
ax.plot(p_pos, kappa, c=start_color, linewidth=4, solid_capstyle='round')

# Direct labels at the right end
ax.text(p_pos[-1] + 0.005, raw_agreement[-1], 'Raw agreement',
        color=end_color, fontsize=18, va='center', ha='left')
ax.text(p_pos[-1] + 0.005, 0, "Cohen's $\\kappa$",
        color=start_color, fontsize=18, va='center', ha='left')

# Annotate a specific point: at p=0.9, raw agreement is 0.82 but kappa is 0
hi_x = 0.90
hi_raw = hi_x**2 + (1 - hi_x)**2
ax.plot(hi_x, hi_raw, 'o', color=end_color, markersize=10,
        markerfacecolor='white', markeredgecolor=end_color,
        markeredgewidth=2.5, zorder=5)
ax.plot(hi_x, 0, 'o', color=start_color, markersize=10,
        markerfacecolor='white', markeredgecolor=start_color,
        markeredgewidth=2.5, zorder=5)
ax.annotate('', xy=(hi_x, hi_raw), xytext=(hi_x, 0),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(hi_x - 0.01, hi_raw / 2,
        f'at $p={hi_x:.2f}$:\nraw $\\approx {hi_raw:.2f}$\n$\\kappa = 0$',
        color='gray', fontsize=12, ha='right', va='center', style='italic')

# Range-frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(0.5, 1.0)
ax.spines['left'].set_bounds(0, 1.0)

ax.set_xlabel('Majority-class proportion', fontsize=18)
ax.set_ylabel('Agreement score', fontsize=18)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', labelsize=14)

ax.set_xlim(0.48, 1.18)
ax.set_ylim(-0.08, 1.1)

save_figure(fig, 'Cohens_Kappa_levels')
plt.close()
print("Cohen's Kappa (Tufte): done")
