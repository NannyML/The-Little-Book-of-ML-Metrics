"""Generate missing regression metric figures for the book."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

# ============================================================
# MDA: bar chart showing directional accuracy for different scenarios
# ============================================================
np.random.seed(42)
n = 100
t = np.arange(n)
y = np.cumsum(np.random.randn(n))  # random walk

# Good directional predictor
y_hat_good = y + np.random.randn(n) * 0.5
# Poor directional predictor (lagged)
y_hat_bad = np.roll(y, 5) + np.random.randn(n) * 2

# Compute directional accuracy
def mda(y_true, y_pred):
    d_true = np.diff(y_true)
    d_pred = np.diff(y_pred)
    return np.mean(np.sign(d_true) == np.sign(d_pred))

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Left: good model
axs[0].plot(t, y, color='black', linewidth=2, label='Actual')
axs[0].plot(t, y_hat_good, color=start_color, linewidth=2, alpha=0.8, label='Predicted')
axs[0].set_title(f'MDA = {mda(y, y_hat_good):.2f}', fontsize=20, color=start_color)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')
axs[0].legend(fontsize=14)

# Right: poor model
axs[1].plot(t, y, color='black', linewidth=2, label='Actual')
axs[1].plot(t, y_hat_bad, color=end_color, linewidth=2, alpha=0.8, label='Predicted')
axs[1].set_title(f'MDA = {mda(y, y_hat_bad):.2f}', fontsize=20, color=end_color)
axs[1].set_xlabel('Time')
axs[1].legend(fontsize=14)

fig.tight_layout()
save_figure(fig, 'MDA_comparison')
plt.close()
print("MDA: done")

# ============================================================
# MPD: cross-section showing deviance as function of prediction
# ============================================================
y_true = 10.0  # fixed actual count
y_pred = np.linspace(1, 30, 200)
mpd = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))

fig, ax = create_line_plot()
ax.plot(y_pred, mpd, c=start_color, **LINE_KW)
ax.axvline(x=y_true, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel(r'Predicted Value ($\hat{Y}$)')
ax.set_ylabel('Poisson Deviance')
ax.annotate(f'$Y = {y_true:.0f}$', xy=(y_true, 0), xytext=(y_true + 3, 3),
            fontsize=16, arrowprops=dict(arrowstyle='->', color='black'))
save_figure(fig, 'MPD_cross_section')
plt.close()
print("MPD: done")

# ============================================================
# MGD: cross-section showing deviance as function of prediction
# Formula matches the book's LaTeX:
#     d(y, y_hat) = 2 * ( log(y_hat / y) + y / y_hat - 1 )
# ============================================================
y_true = 10.0
y_pred = np.linspace(1, 30, 400)
mgd = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)

fig, ax = create_line_plot()
ax.plot(y_pred, mgd, c=start_color, **LINE_KW)
ax.axvline(x=y_true, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel(r'Predicted Value ($\hat{Y}$)')
ax.set_ylabel('Gamma Deviance')
ax.annotate(f'$Y = {y_true:.0f}$', xy=(y_true, 0), xytext=(y_true + 4, 1.5),
            fontsize=16, arrowprops=dict(arrowstyle='->', color='black'))
save_figure(fig, 'MGD_cross_section')
plt.close()
print("MGD: done")

# ============================================================
# D2 Absolute Score: comparison of D2 vs R-squared with outliers
# ============================================================
np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
y_true = 2 * x + 1 + np.random.randn(n) * 1.5
y_pred = 2 * x + 1 + np.random.randn(n) * 0.8

# Add outliers
y_true_outlier = y_true.copy()
y_true_outlier[45] = 50  # big outlier

# Compute metrics
def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

def d2_abs(y, yhat):
    abs_res = np.sum(np.abs(y - yhat))
    abs_tot = np.sum(np.abs(y - np.median(y)))
    return 1 - abs_res / abs_tot

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Without outliers
axs[0].scatter(y_true, y_pred, color=start_color, s=50, alpha=0.7)
axs[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', linewidth=1)
axs[0].set_title('Without Outliers', fontsize=20)
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].annotate(f'R² = {r2(y_true, y_pred):.3f}\nD² = {d2_abs(y_true, y_pred):.3f}',
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=16,
                color=middle_color)

# With outliers
colors = [end_color if i == 45 else start_color for i in range(n)]
axs[1].scatter(y_true_outlier, y_pred, c=colors, s=50, alpha=0.7)
axs[1].plot([y_true_outlier.min(), y_true_outlier.max()],
            [y_true_outlier.min(), y_true_outlier.max()], 'k--', linewidth=1)
axs[1].set_title('With Outlier', fontsize=20)
axs[1].set_xlabel('Actual')
axs[1].annotate(f'R² = {r2(y_true_outlier, y_pred):.3f}\nD² = {d2_abs(y_true_outlier, y_pred):.3f}',
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=16,
                color=middle_color)

fig.tight_layout()
save_figure(fig, 'D2_abs_comparison')
plt.close()
print("D2: done")

# ============================================================
# Explained Variance Score: comparison EVS vs R-squared with bias
# ============================================================
np.random.seed(42)
n = 100
y_true = np.random.randn(n) * 3 + 10

# Model A: good predictions
y_pred_a = y_true + np.random.randn(n) * 0.5

# Model B: perfect pattern but constant offset
y_pred_b = y_true + 5  # shifted by 5

def evs(y, yhat):
    return 1 - np.var(y - yhat) / np.var(y)

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

axs[0].scatter(y_true, y_pred_a, color=start_color, s=30, alpha=0.7)
axs[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', linewidth=1)
axs[0].set_title('Good Model', fontsize=20)
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].annotate(f'EVS = {evs(y_true, y_pred_a):.3f}\nR²  = {r2(y_true, y_pred_a):.3f}',
                xy=(0.05, 0.8), xycoords='axes fraction', fontsize=16, color=middle_color)

axs[1].scatter(y_true, y_pred_b, color=end_color, s=30, alpha=0.7)
lims = [min(y_true.min(), y_pred_b.min()), max(y_true.max(), y_pred_b.max())]
axs[1].plot(lims, lims, 'k--', linewidth=1)
axs[1].set_title('Biased Model (offset +5)', fontsize=20)
axs[1].set_xlabel('Actual')
axs[1].annotate(f'EVS = {evs(y_true, y_pred_b):.3f}\nR²  = {r2(y_true, y_pred_b):.3f}',
                xy=(0.05, 0.8), xycoords='axes fraction', fontsize=16, color=middle_color)

fig.tight_layout()
save_figure(fig, 'EVS_vs_R2_comparison')
plt.close()
print("EVS: done")

print("\nAll regression figures generated!")
