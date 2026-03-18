"""Insight/comparative plots for regression metrics — the diagnostic value-add beyond 3D surfaces."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

np.random.seed(42)

# ============================================================
# 1. MSLE/RMSLE — show proportional vs absolute error behavior
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

targets = np.array([10, 100, 1000, 10000])
# Same absolute error (10 units off)
abs_preds = targets + 10
# Same relative error (10% off)
rel_preds = targets * 1.1

mae_abs = np.abs(targets - abs_preds)
msle_abs = (np.log1p(targets) - np.log1p(abs_preds))**2
mae_rel = np.abs(targets - rel_preds)
msle_rel = (np.log1p(targets) - np.log1p(rel_preds))**2

x = np.arange(len(targets))
labels = ['10', '100', '1K', '10K']

axs[0].bar(x, mae_abs, color=start_color, edgecolor='black', linewidth=0.5, width=0.6)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].set_xlabel('Target Value')
axs[0].set_ylabel('MAE Contribution')
axs[0].set_title('Same Absolute Error (+10)', fontsize=16, color=start_color)
for i, v in enumerate(mae_abs):
    axs[0].text(i, v + 0.3, f'{v:.0f}', ha='center', fontsize=13)

axs[1].bar(x, msle_rel, color=middle_color, edgecolor='black', linewidth=0.5, width=0.6)
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].set_xlabel('Target Value')
axs[1].set_ylabel('MSLE Contribution')
axs[1].set_title('Same Relative Error (10%)', fontsize=16, color=middle_color)
for i, v in enumerate(msle_rel):
    axs[1].text(i, v + 0.0003, f'{v:.4f}', ha='center', fontsize=13)

fig.tight_layout()
save_figure(fig, 'MSLE_absolute_vs_relative')
plt.close()
print("1. MSLE insight plot: done")

# ============================================================
# 2. MAPE vs sMAPE vs wMAPE — head-to-head on same data with zeros
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

actuals = np.array([100, 50, 0.1, 200, 80, 0, 150, 30])
preds =   np.array([110, 45, 5,   180, 90, 5, 140, 35])

# Compute per-item
mape_items = np.abs(actuals - preds) / np.where(actuals == 0, np.nan, actuals) * 100
smape_items = np.abs(actuals - preds) / ((np.abs(actuals) + np.abs(preds)) / 2) * 100

x = np.arange(len(actuals))
width = 0.35

bars1 = ax.bar(x - width/2, np.nan_to_num(mape_items, nan=0), width, label='MAPE', color=end_color,
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, smape_items, width, label='sMAPE', color=start_color,
               edgecolor='black', linewidth=0.5)

# Mark the problematic items
for i in [2, 5]:  # near-zero and zero actuals
    ax.annotate('!', xy=(i, max(np.nan_to_num(mape_items[i], nan=200), smape_items[i]) + 5),
                fontsize=18, color=end_color, ha='center', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'Y={a}' for a in actuals], fontsize=10, rotation=30)
ax.set_ylabel('Percentage Error')
ax.set_ylim(0, 220)
ax.legend(fontsize=14)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'MAPE_vs_sMAPE_zeros')
plt.close()
print("2. MAPE vs sMAPE comparison: done")

# ============================================================
# 3. R-squared — what R² values actually look like
# ============================================================
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for ax, r2_target, title in [
    (axs[0], 0.95, 'R² ≈ 0.95'),
    (axs[1], 0.70, 'R² ≈ 0.70'),
    (axs[2], 0.30, 'R² ≈ 0.30'),
    (axs[3], -0.2, 'R² ≈ -0.20'),
]:
    x = np.linspace(0, 10, 50)
    y_true = 2 * x + 5
    if r2_target > 0:
        noise_std = np.sqrt(np.var(y_true) * (1 - r2_target) / r2_target)
    else:
        noise_std = np.sqrt(np.var(y_true) * 3)
    y_pred = y_true + np.random.randn(50) * noise_std
    if r2_target < 0:
        y_pred = np.mean(y_true) + np.random.randn(50) * np.std(y_true) * 1.5

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    actual_r2 = 1 - ss_res / ss_tot

    color = start_color if actual_r2 > 0.5 else (middle_color if actual_r2 > 0 else end_color)
    ax.scatter(y_true, y_pred, color=color, s=20, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_title(f'{title}', fontsize=14, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    if ax == axs[0]:
        ax.set_ylabel('Predicted', fontsize=12)
    ax.set_xlabel('Actual', fontsize=12)

fig.tight_layout()
save_figure(fig, 'R2_what_values_look_like')
plt.close()
print("3. R² visual guide: done")

# ============================================================
# 4. MDA — directional accuracy matters for trading
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

t = np.arange(20)
prices = 100 + np.cumsum(np.random.randn(20) * 2)
# Model A: good direction, bad magnitude
pred_a = prices + np.random.randn(20) * 5
# Ensure direction matches most of the time
for i in range(1, 20):
    if np.sign(prices[i] - prices[i-1]) != np.sign(pred_a[i] - prices[i-1]):
        if np.random.rand() < 0.8:
            pred_a[i] = prices[i-1] + np.sign(prices[i] - prices[i-1]) * abs(np.random.randn() * 3)

directions_correct = sum(np.sign(np.diff(prices)) == np.sign(np.diff(pred_a)))
mda = directions_correct / (len(prices) - 1)

ax.plot(t, prices, color='black', linewidth=2, marker='o', markersize=5, label='Actual Price')
ax.plot(t, pred_a, color=start_color, linewidth=2, marker='s', markersize=5, alpha=0.7, label=f'Forecast (MDA={mda:.2f})')

# Shade correct/incorrect direction predictions
for i in range(1, 20):
    correct = np.sign(prices[i] - prices[i-1]) == np.sign(pred_a[i] - prices[i-1])
    color = start_color if correct else end_color
    ax.axvspan(i-0.5, i+0.5, alpha=0.08, color=color)

ax.set_xlabel('Time Step')
ax.set_ylabel('Price')
ax.legend(fontsize=13)
fig.tight_layout()
save_figure(fig, 'MDA_trading_example')
plt.close()
print("4. MDA trading example: done")

# ============================================================
# 5. Pinball Loss — asymmetric penalty for different quantiles
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

y_diff = np.linspace(-10, 10, 1000)
for q, color, label in [
    (0.1, end_color, 'q=0.1 (penalize over-prediction)'),
    (0.5, middle_color, 'q=0.5 (MAE equivalent)'),
    (0.9, start_color, 'q=0.9 (penalize under-prediction)')
]:
    loss = np.where(y_diff >= 0, q * y_diff, (1 - q) * (-y_diff))
    ax.plot(y_diff, loss, color=color, linewidth=3, label=label)

ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel(r'$Y - \hat{Y}$ (actual − predicted)')
ax.set_ylabel('Pinball Loss')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
ax.annotate('under-prediction\n(actual > predicted)', xy=(5, 0.5), fontsize=11, color='gray', ha='center')
ax.annotate('over-prediction\n(actual < predicted)', xy=(-5, 0.5), fontsize=11, color='gray', ha='center')
fig.tight_layout()
save_figure(fig, 'Pinball_Loss_quantiles')
plt.close()
print("5. Pinball Loss quantiles: done")

# ============================================================
# 6. EVS vs R² — the bias trap
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(16, 5))

n = 80
x = np.linspace(0, 10, n)
y_true = 2 * x + np.random.randn(n) * 1.5

# Model A: good predictions
y_a = y_true + np.random.randn(n) * 0.8
# Model B: perfect correlation but +5 offset (biased)
y_b = y_true + 5
# Model C: noisy but unbiased
y_c = y_true + np.random.randn(n) * 3

def evs(y, yhat):
    return 1 - np.var(y - yhat) / np.var(y)
def r2(y, yhat):
    return 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)

for ax, y_pred, name, color in [
    (axs[0], y_a, 'Good Model', start_color),
    (axs[1], y_b, 'Biased Model (+5)', end_color),
    (axs[2], y_c, 'Noisy Model', middle_color)
]:
    ax.scatter(y_true, y_pred, color=color, s=20, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_title(name, fontsize=15, color=color)
    ax.set_xlabel('Actual')
    ax.annotate(f'EVS = {evs(y_true, y_pred):.2f}\nR²  = {r2(y_true, y_pred):.2f}',
                xy=(0.05, 0.82), xycoords='axes fraction', fontsize=14,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if ax == axs[0]:
        ax.set_ylabel('Predicted')

fig.tight_layout()
save_figure(fig, 'EVS_vs_R2_three_models')
plt.close()
print("6. EVS vs R² comparison: done")

print("\nAll regression insight plots generated!")
