"""Regenerate P4_vs_F1.png as a continuous F1 vs P4 sweep.

The previous plot showed four discrete bar scenarios. This version makes
the key point continuous: for a model with fixed positive-class performance
(precision = recall = 0.8), F1 stays constant at 0.80 regardless of how
well it handles negatives, but P4 drops as specificity drops. The gap
between the two lines = F1's blindness to true negatives.

Tufte style applied: direct labels, range-frame spines, no legend,
markers at the crossover point.
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *


# Fixed positive performance: TP=80, FP=20, FN=20 (so precision = recall = 0.8)
TP = 80.0
FP = 20.0
FN = 20.0

# Sweep specificity by varying TN: specificity = TN / (TN + FP) => TN = FP*s/(1-s)
specificity = np.linspace(0.02, 0.99, 300)
TN = FP * specificity / (1 - specificity)

F1_const = 2 * (TP / (TP + FP)) * (TP / (TP + FN)) \
           / ((TP / (TP + FP)) + (TP / (TP + FN)))
F1 = np.full_like(specificity, F1_const)
P4 = 4 * TP * TN / (4 * TP * TN + (TP + TN) * (FP + FN))

fig, ax = create_line_plot()

# The two lines
ax.plot(specificity, F1, c=end_color, linewidth=4, solid_capstyle='round')
ax.plot(specificity, P4, c=start_color, linewidth=4, solid_capstyle='round')

# Direct labels at the right end of each line
ax.text(1.01, F1[-1], 'F1', color=end_color, fontsize=20,
        va='center', ha='left')
ax.text(1.01, P4[-1], 'P4', color=start_color, fontsize=20,
        va='center', ha='left')

# Mark the crossover where F1 = P4 (this is at specificity = precision = 0.8)
crossing_idx = np.argmin(np.abs(P4 - F1))
crossing_x = specificity[crossing_idx]
ax.plot(crossing_x, F1_const, 'o', color='black', markersize=11,
        markerfacecolor='white', markeredgecolor='black',
        markeredgewidth=2.5, zorder=5)
ax.text(crossing_x, F1_const + 0.05,
        f'F1 = P4 at\nspecificity = {crossing_x:.2f}',
        color='black', fontsize=12, ha='center', va='bottom')

# Highlight the "F1 hides this" gap at low specificity
gap_x = 0.3
gap_F1 = F1_const
gap_P4 = np.interp(gap_x, specificity, P4)
ax.annotate('', xy=(gap_x, gap_F1), xytext=(gap_x, gap_P4),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(gap_x - 0.03, (gap_F1 + gap_P4) / 2,
        'gap = F1\'s blindness\nto true negatives',
        color='gray', fontsize=12, ha='right', va='center', style='italic')

# Range-frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_bounds(0, 1.0)
ax.spines['left'].set_bounds(0, 1.0)

ax.set_xlabel(r'Specificity $= TN/(TN+FP)$', fontsize=18)
ax.set_ylabel('Score', fontsize=18)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', labelsize=14)

ax.set_xlim(-0.02, 1.12)
ax.set_ylim(-0.02, 1.0)

save_figure(fig, 'P4_vs_F1')
plt.close()
print('P4 vs F1 (Tufte): done')
