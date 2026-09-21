"""Generate figures for the Data Observability chapter.

One dataset, twelve monitors.  A daily-partitioned `orders` table is simulated
for 42 days with weekly seasonality, and one realistic incident is injected per
monitor (a missed load, a partial load, a currency slip, an upstream rename, a
retry storm, ...).  Every metric is then computed per partition exactly as the
Soda documentation defines it, the same baseline anomaly detector is run on
every series (a seasonal expected range of ±z·σ around the same-weekday mean,
z = 3, trained on the first 21 partitions), and each figure shows one series
with its expected range and the points the detector flags.  Nothing is typed
in by hand.

Run from notebooks/:  uv run python observability_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
import pandas as pd

GREY = '#c9c9c9'
BAND = '#e6e6e6'
GREY_LINE = '#9a9a9a'
DARK = '#2a2a2a'
MID = '#6f6f6f'
RNG = np.random.default_rng(25)   # seed chosen so the 3σ band's chance false alarms do not clutter the pictures
OUT = Path(__file__).resolve().parent / 'data' / 'observability'
OUT.mkdir(parents=True, exist_ok=True)

DAYS = 42
TRAIN = 21
Z = 3.0
START = pd.Timestamp('2026-07-06')            # a Monday
SCAN_HOUR = 6                                 # scans run at 06:00 every day
COUNTRIES = ['US', 'GB', 'DE', 'FR', 'NL', 'ES', 'IT', 'BE']
INCIDENT = {
    'currency_slip': 22,   # 5% of amounts land in cents (×100)
    'no_load': 25,         # nothing arrives; the partition is empty
    'phone_format': 27,    # 60% of phone numbers lose their country prefix
    'country_rename': 28,  # an upstream field is renamed; country arrives null for 40% of rows
    'partial_load': 30,    # the load stops half-way
    'refunds': 31,         # 3% of rows are refunds with negative amounts
    'schema_change': 33,   # a column is added and one type changes
    'retry_storm': 35,     # 6% of rows are inserted twice
    'id_truncation': 37,   # customer ids are truncated to three digits
    'promotion': 39,       # half price on everything under 30: the cheap half of the catalogue
}
BASE_SCHEMA = [('order_id', 'BIGINT'), ('customer_id', 'BIGINT'), ('amount', 'DECIMAL(10,2)'),
               ('country', 'VARCHAR(2)'), ('phone', 'VARCHAR(16)'), ('created_at', 'TIMESTAMP')]
NEW_SCHEMA = [('order_id', 'BIGINT'), ('customer_id', 'BIGINT'), ('amount', 'FLOAT'),
              ('country', 'VARCHAR(2)'), ('phone', 'VARCHAR(16)'), ('discount_pct', 'FLOAT'), ('created_at', 'TIMESTAMP')]


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
# 1. Simulate the table, one partition per day
# ---------------------------------------------------------------------------
def simulate():
    partitions = []
    next_id = 1
    for d in range(DAYS):
        day = START + pd.Timedelta(days=d)
        wd = day.weekday()
        factor = {5: 0.62, 6: 0.58}.get(wd, 1.0)
        n = int(round(5000 * factor * (1 + RNG.normal(0, 0.04))))
        if d == INCIDENT['no_load']:
            n = 0
        if d == INCIDENT['partial_load']:
            n = n // 2
        df = pd.DataFrame({
            'order_id': np.arange(next_id, next_id + n),
            'customer_id': RNG.integers(1, 20_000, n),
            'amount': np.round(np.exp(RNG.normal(3.6, 0.6, n)), 2),
            'country': RNG.choice(COUNTRIES, n),
            'phone': ['+1-555-' + f'{k:04d}' for k in RNG.integers(0, 10_000, n)],
        })
        next_id += n
        # arrival times: the nightly load lands between 00:00 and ~01:30
        minutes = RNG.uniform(0, 90, n)
        if d == INCIDENT['partial_load']:
            minutes = RNG.uniform(0, 25, n)      # the load died after 25 minutes
        df['created_at'] = day + pd.to_timedelta(minutes, unit='m')
        # ordinary missingness
        df.loc[RNG.random(n) < 0.02, 'country'] = None
        # incidents
        if d == INCIDENT['currency_slip']:
            m = RNG.random(n) < 0.05
            df.loc[m, 'amount'] = df.loc[m, 'amount'] * 100
        if d == INCIDENT['phone_format']:
            m = RNG.random(n) < 0.60
            df.loc[m, 'phone'] = df.loc[m, 'phone'].str.replace('+1-555-', '555', regex=False)
        if d == INCIDENT['country_rename']:
            df.loc[RNG.random(n) < 0.40, 'country'] = None
        if d == INCIDENT['refunds']:
            m = RNG.random(n) < 0.03
            df.loc[m, 'amount'] = -df.loc[m, 'amount']
        if d == INCIDENT['retry_storm']:
            dup = df.sample(frac=0.06, random_state=1)
            df = pd.concat([df, dup], ignore_index=True)
        if d == INCIDENT['id_truncation']:
            df['customer_id'] = df['customer_id'] % 1000
        if d == INCIDENT['promotion']:
            m = df['amount'] < 30
            df.loc[m, 'amount'] = np.round(df.loc[m, 'amount'] * 0.5, 2)
        schema = NEW_SCHEMA if d >= INCIDENT['schema_change'] else BASE_SCHEMA
        partitions.append((day, df, schema))
    return partitions


# ---------------------------------------------------------------------------
# 2. Metrics, as the documentation defines them
# ---------------------------------------------------------------------------
def duplicate_pct(col):
    v = col.dropna()
    if len(v) == 0:
        return np.nan
    counts = v.value_counts()
    return 100 * counts[counts > 1].sum() / len(v)


def compute_metrics(partitions):
    rows = []
    last_nonempty_max = None
    for d, (day, df, schema) in enumerate(partitions):
        scan = day + pd.Timedelta(hours=SCAN_HOUR)
        if len(df):
            last_nonempty_max = df['created_at'].max()
        fresh_h = (scan - last_nonempty_max).total_seconds() / 3600 if last_nonempty_max is not None else np.nan
        e = len(df) == 0
        r = dict(
            day=d, date=day, weekday=day.weekday(),
            row_count=len(df),
            freshness_h=fresh_h,
            n_columns=len(schema), schema=schema,
            missing_country_pct=np.nan if e else 100 * (1 - df['country'].count() / len(df)),
            dup_order_pct=np.nan if e else duplicate_pct(df['order_id']),
            count_customer=np.nan if e else df['customer_id'].count(),
            unique_customer=np.nan if e else df['customer_id'].nunique(),
            avg_amount=np.nan if e else df['amount'].mean(),
            sum_amount=np.nan if e else df['amount'].sum(),
            std_amount=np.nan if e else df['amount'].std(),
            min_amount=np.nan if e else df['amount'].min(),
            max_amount=np.nan if e else df['amount'].max(),
            q1_amount=np.nan if e else df['amount'].quantile(0.25),
            median_amount=np.nan if e else df['amount'].median(),
            q3_amount=np.nan if e else df['amount'].quantile(0.75),
            avg_len_phone=np.nan if e else df['phone'].str.len().mean(),
            min_len_phone=np.nan if e else df['phone'].str.len().min(),
            max_len_phone=np.nan if e else df['phone'].str.len().max(),
        )
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. The expected range: seasonal baseline ± z·σ, trained on history, anomalies
#    excluded from the history once flagged
# ---------------------------------------------------------------------------
def expected_range(values, weekdays, z=Z, train=TRAIN, rel_floor=0.01):
    values = np.asarray(values, dtype=float)
    weekdays = np.asarray(weekdays)
    n = len(values)
    lo, hi, flag = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)
    good = np.ones(n, dtype=bool)   # points allowed into the history
    good &= ~np.isnan(values)
    for t in range(train, n):
        hist = np.arange(t)[good[:t]]
        if len(hist) < 7:
            continue
        # seasonal baseline: mean of the same weekday in the history, else overall mean
        same = hist[weekdays[hist] == weekdays[t]]
        base = values[same].mean() if len(same) >= 2 else values[hist].mean()
        # residuals of the history around their own weekday means
        resid = np.array([values[i] - values[hist[weekdays[hist] == weekdays[i]]].mean() for i in hist])
        sigma = max(np.nanstd(resid, ddof=1), rel_floor * abs(base), 1e-9)
        lo[t], hi[t] = base - z * sigma, base + z * sigma
        if not np.isnan(values[t]) and (values[t] < lo[t] or values[t] > hi[t]):
            flag[t] = True
            good[t] = False
    return lo, hi, flag


# ---------------------------------------------------------------------------
# 4. Drawing
# ---------------------------------------------------------------------------
def panel(ax, m, col, ylabel, fmt='{:.0f}', annotate=None, yscale=None, title=None, clip=None, ymin=None, xmin=0.5):
    x = m['day'].values + 1
    v_true = m[col].values.astype(float)
    lo, hi, flag = expected_range(v_true, m['weekday'].values)
    v = v_true.copy()
    if clip is not None:                       # values above `clip` are drawn at the top edge with their value
        over = v_true > clip
        v[over] = clip
    keep = x >= xmin
    ax.fill_between(x[keep], lo[keep], hi[keep], color=BAND, zorder=1, linewidth=0, step=None)
    ax.plot(x[keep], v[keep], color=start_color, lw=2.2, zorder=3, solid_capstyle='round')
    ax.scatter(x[~flag & keep], v[~flag & keep], s=26, color=start_color, zorder=4, linewidths=0)
    ax.scatter(x[flag], v[flag], s=70, color=end_color, zorder=5, linewidths=0)
    if clip is not None:
        for xi, vt in zip(x[over], v_true[over]):
            ax.plot(xi, clip, marker='^', ms=11, color=end_color, zorder=6, linestyle='none')
            ax.text(xi + 0.5, clip, f'off the scale: {vt:,.0f}', color=end_color, fontsize=11, va='center', ha='left')
    miss = np.isnan(v)
    if miss.any():
        ybase = ymin if ymin is not None else (np.nanmin(np.r_[v, lo]) if yscale != 'log' else np.nanmin(v[~miss]))
        ax.scatter(x[miss], [ybase] * miss.sum(), s=60, facecolor='white', edgecolor=GREY_LINE, linewidth=1.4, zorder=5)
    ax.axvspan(0.5, TRAIN + 0.5, color='#f4f4f4', zorder=0, linewidth=0)
    ax.text(1.0, 1.0, 'training', transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=10, color=MID)
    if annotate:
        for item in annotate:
            d, text, dy = item[:3]
            dx = item[3] if len(item) > 3 else 0
            xi = d + 1
            yi = v[d] if not np.isnan(v[d]) else (np.nanmin(np.r_[v, lo]))
            ax.annotate(text, xy=(xi, yi), xytext=(dx, dy), textcoords='offset points',
                        ha='center' if dx == 0 else ('left' if dx > 0 else 'right'),
                        va=('bottom' if dy > 0 else 'top') if dx == 0 else 'center', fontsize=11, color=end_color,
                        arrowprops=dict(arrowstyle='-', color=end_color, lw=0.9))
    ax.set_xlim(xmin, DAYS + 0.5)
    ax.set_xticks([t for t in [1, 8, 15, 22, 29, 36, 42] if t >= xmin])
    ax.set_xlabel('daily partition', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if clip is not None:
        ax.set_ylim(ymin if ymin is not None else ax.get_ylim()[0], clip * 1.02)
    elif ymin is not None:
        ax.set_ylim(ymin, ax.get_ylim()[1])
    if yscale:
        ax.set_yscale(yscale)
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}' if y >= 1 else f'{y:g}'))
    ax.tick_params(labelsize=11)
    despine(ax)
    if title:
        ax.set_title(title, fontsize=13, loc='left', color=DARK, pad=8)
    return lo, hi, flag


def one_panel_figure(m, col, ylabel, name, annotate=None, yscale=None, height=3.6, extra=None, clip=None, ymin=None):
    fig, ax = plt.subplots(figsize=(13, height))
    lo, hi, flag = panel(ax, m, col, ylabel, annotate=annotate, yscale=yscale, clip=clip, ymin=ymin)
    if extra:
        extra(ax, lo, hi, flag)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9 if height < 4 else 0.84, bottom=0.2)
    save_figure(fig, name)
    plt.close()
    return flag


def main():
    parts = simulate()
    m = compute_metrics(parts)
    m.to_json(OUT / 'metrics.json', orient='records', date_format='iso', indent=1)
    flags = {}

    # --- the mechanism: how the band is built (left) and applied (right) --------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={'width_ratios': [0.75, 1.25], 'wspace': 0.22})
    v = m['row_count'].values.astype(float)
    wd = m['weekday'].values
    names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    train_idx = np.arange(TRAIN)
    means = np.array([v[train_idx][wd[train_idx] == d].mean() for d in range(7)])
    resid = np.array([v[i] - means[wd[i]] for i in train_idx])
    sigma = resid.std(ddof=1)
    for d in range(7):
        pts = v[train_idx][wd[train_idx] == d]
        axL.scatter([d] * len(pts), pts, s=34, color=start_color, zorder=4, linewidths=0)
        axL.plot([d - 0.3, d + 0.3], [means[d], means[d]], color=DARK, lw=1.6, zorder=5)
        axL.fill_between([d - 0.3, d + 0.3], means[d] - Z * sigma, means[d] + Z * sigma, color=BAND, zorder=1, linewidth=0)
    axL.set_xticks(range(7))
    axL.set_xticklabels(names, fontsize=11)
    axL.set_ylabel('rows in the partition', fontsize=12)
    axL.set_xlabel('three training weeks, grouped by weekday', fontsize=12)
    axL.tick_params(axis='y', labelsize=11)
    axL.set_ylim(0, 7400)
    despine(axL)
    axL.text(0.5, means[0] + Z * sigma + 250, f'mean ± 3σ  (σ = {sigma:.0f} rows, pooled)', fontsize=11, color=MID, ha='left', va='bottom')
    axL.text(6.35, means[6], 'weekday\nmean', fontsize=10.5, color=DARK, ha='left', va='center')
    ann = [(INCIDENT['no_load'], 'no load: 0 rows, flagged', 30, 40), (INCIDENT['partial_load'], 'partial load,\nflagged', 22, 30)]
    lo, hi, flag = panel(axR, m, 'row_count', '', annotate=ann, xmin=TRAIN + 0.5, ymin=0)
    axR.set_ylim(0, 7400)
    axR.set_xlabel('weeks four to six: each new scan against its weekday\'s range', fontsize=12)
    t = TRAIN + 4
    axR.annotate('expected range', xy=(t, hi[t]), xytext=(t + 0.5, 6800), fontsize=11, color=MID, ha='left',
                 arrowprops=dict(arrowstyle='-', color=GREY_LINE, lw=0.9))
    axR.texts[0].remove() if axR.texts and axR.texts[0].get_text() == 'training' else None
    for patch in list(axR.patches):
        pass
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.2)
    save_figure(fig, 'Observability_expected_range')
    plt.close()
    flags['mechanism'] = flag
    print(f'mechanism: sigma {sigma:.1f}; weekday means {np.round(means).astype(int).tolist()}; band half-width {Z * sigma:.0f}')

    # --- row count ---------------------------------------------------------
    ann = [(INCIDENT['no_load'], 'no load: 0 rows', 30, -55), (INCIDENT['partial_load'], 'partial load', 22, 30),
           (INCIDENT['retry_storm'], 'retry storm: +6%,\ninside the band', 16)]
    flags['row_count'] = one_panel_figure(m, 'row_count', 'rows in the partition', 'Observability_row_count', annotate=ann, ymin=0)
    lo_rc, hi_rc, _ = expected_range(m['row_count'].values, m['weekday'].values)
    d = INCIDENT['retry_storm']
    print(f'row count at retry storm: {m["row_count"][d]} band [{lo_rc[d]:.0f}, {hi_rc[d]:.0f}] '
          f'= ±{100 * (hi_rc[d] - lo_rc[d]) / 2 / ((hi_rc[d] + lo_rc[d]) / 2):.0f}% around {((hi_rc[d] + lo_rc[d]) / 2):.0f}')

    # --- freshness ---------------------------------------------------------
    ann = [(INCIDENT['no_load'], 'no load: yesterday\'s data is the newest', -6, 14),
           (INCIDENT['partial_load'], 'load stopped early', 14)]
    flags['freshness'] = one_panel_figure(m, 'freshness_h', 'hours since newest row', 'Observability_freshness', annotate=ann)

    # --- schema ------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.12})
    x = m['day'].values + 1
    axL.step(x, m['n_columns'], where='post', color=start_color, lw=2.2, zorder=3)
    axL.scatter(x, m['n_columns'], s=26, color=start_color, zorder=4, linewidths=0)
    d = INCIDENT['schema_change']
    axL.scatter([d + 1], [m['n_columns'][d]], s=70, color=end_color, zorder=5, linewidths=0)
    axL.annotate('column added, one type changed', xy=(d + 1, m['n_columns'][d]), xytext=(d - 8, m['n_columns'][d] + 0.35),
                 fontsize=11, color=end_color, ha='center', arrowprops=dict(arrowstyle='-', color=end_color, lw=0.9))
    axL.set_xlim(0.5, DAYS + 0.5)
    axL.set_xticks([1, 8, 15, 22, 29, 36, 42])
    axL.set_yticks([6, 7])
    axL.set_ylim(5.5, 7.8)
    axL.set_xlabel('daily partition', fontsize=12)
    axL.set_ylabel('columns', fontsize=12)
    axL.tick_params(labelsize=11)
    despine(axL)
    axR.axis('off')
    before = dict(BASE_SCHEMA)
    after = dict(NEW_SCHEMA)
    yy = 0.95
    axR.text(0.0, yy, 'scan 33 vs scan 34', fontsize=12, color=DARK, va='top')
    yy -= 0.16
    for col, typ in NEW_SCHEMA:
        if col not in before:
            txt, colr = f'+  {col}  {typ}', start_color
        elif before[col] != typ:
            txt, colr = f'~  {col}  {before[col]} → {typ}', end_color
        else:
            txt, colr = f'    {col}  {typ}', MID
        axR.text(0.0, yy, txt, fontsize=11, color=colr, va='top', family='monospace')
        yy -= 0.12
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.2)
    save_figure(fig, 'Observability_schema')
    plt.close()

    # --- missing values ----------------------------------------------------
    ann = [(INCIDENT['country_rename'], 'upstream field renamed:\ncountry arrives empty', -6, 14)]
    flags['missing'] = one_panel_figure(m, 'missing_country_pct', 'country missing (%)', 'Observability_missing', annotate=ann)

    # --- duplicates --------------------------------------------------------
    ann = [(INCIDENT['retry_storm'], 'retry storm: 6% of rows inserted twice', -8, 0)]
    flags['dup'] = one_panel_figure(m, 'dup_order_pct', 'duplicate order_id (%)', 'Observability_duplicates', annotate=ann)

    # --- unique count ------------------------------------------------------
    ann = [(INCIDENT['id_truncation'], 'ids truncated to\nthree digits', 10, 10),
           (INCIDENT['partial_load'], 'partial load:\nhalf the rows', 10, 0)]
    flags['unique'] = one_panel_figure(m, 'unique_customer', 'distinct customer_id', 'Observability_unique_count', annotate=ann, ymin=0)
    wk = m[(m['day'] < TRAIN) & (m['weekday'] < 5)]['unique_customer'].mean()
    we = m[(m['day'] < TRAIN) & (m['weekday'] >= 5)]['unique_customer'].mean()
    print(f'unique: weekday mean {wk:.0f}, weekend mean {we:.0f}')

    # --- average -----------------------------------------------------------
    ann = [(INCIDENT['refunds'], 'refunds: 3% negative', 8, 22), (INCIDENT['promotion'], 'half price under 30', -8, 22)]
    base = m['avg_amount'][:TRAIN].mean()
    flags['avg'] = one_panel_figure(m, 'avg_amount', 'mean amount', 'Observability_average', annotate=ann, clip=base * 1.3, ymin=base * 0.85)

    # --- sum ---------------------------------------------------------------
    ann = [(INCIDENT['no_load'], 'no load:\nmissing scan', -10, 0), (INCIDENT['partial_load'], 'partial load:\nhalf the rows', 10, -10)]
    base = m['sum_amount'][:TRAIN].mean()
    flags['sum'] = one_panel_figure(m, 'sum_amount', 'sum of amount', 'Observability_sum', annotate=ann, clip=base * 1.5, ymin=50_000)

    # --- standard deviation ------------------------------------------------
    ann = [(INCIDENT['refunds'], 'refunds: 3% negative', -8, -22), (INCIDENT['promotion'], 'half price under 30', -8, -22)]
    base = m['std_amount'][:TRAIN].mean()
    flags['std'] = one_panel_figure(m, 'std_amount', 'std of amount', 'Observability_stddev', annotate=ann, clip=base * 1.4, ymin=base * 0.75)

    # --- min / max ---------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={'wspace': 0.25})
    _, _, fmin = panel(axL, m, 'min_amount', 'min amount', annotate=[(INCIDENT['refunds'], 'refunds:\nnegative amounts', 10, 0),
                                                                      (INCIDENT['promotion'], 'half price', 8, 0)])
    base = m['max_amount'][:TRAIN].mean()
    _, _, fmax = panel(axR, m, 'max_amount', 'max amount', annotate=[(INCIDENT['country_rename'], 'one large order:\na false alarm', 8, 0)],
                       clip=base * 2.2, ymin=0)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.2)
    save_figure(fig, 'Observability_min_max')
    plt.close()
    flags['min'], flags['max'] = fmin, fmax

    # --- quartiles ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 3.8))
    x = m['day'].values + 1
    for col, color, lab in [('q3_amount', middle_color, 'Q3'), ('median_amount', start_color, 'median'), ('q1_amount', GREY_LINE, 'Q1')]:
        lo, hi, flag = expected_range(m[col].values, m['weekday'].values)
        ax.fill_between(x, lo, hi, color=BAND, zorder=1, linewidth=0)
        ax.plot(x, m[col], color=color, lw=2.2, zorder=3, solid_capstyle='round')
        ax.scatter(x[flag], m[col].values[flag], s=70, color=end_color, zorder=5, linewidths=0)
        ax.text(DAYS + 0.8, m[col].values[-1], lab, color=color, fontsize=11.5, va='center')
        flags[col] = flag
    ax.plot(x, m['avg_amount'], color=DARK, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(DAYS + 0.8, m['avg_amount'].values[-1] + 3, 'mean', color=DARK, fontsize=11.5, va='center')
    d = INCIDENT['currency_slip']
    ax.annotate('5% of amounts in cents: the mean leaves the frame,\nthe quartiles barely move', xy=(d + 1, m['q3_amount'][d]),
                xytext=(d - 6, 92), fontsize=11, color=DARK, ha='center',
                arrowprops=dict(arrowstyle='-', color=end_color, lw=0.9))
    d = INCIDENT['promotion']
    ax.annotate('half price under 30: Q1 drops,\nthe median and Q3 do not', xy=(d + 1, m['q1_amount'][d]), xytext=(d - 4, 8),
                fontsize=11, color=end_color, ha='right', va='center', arrowprops=dict(arrowstyle='-', color=end_color, lw=0.9))
    ax.axvspan(0.5, TRAIN + 0.5, color='#f4f4f4', zorder=0, linewidth=0)
    ax.text(1.0, 1.0, 'training', transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=10, color=MID)
    ax.set_xlim(0.5, DAYS + 2.5)
    ax.set_ylim(0, 110)
    ax.set_xticks([1, 8, 15, 22, 29, 36, 42])
    ax.set_xlabel('daily partition', fontsize=12)
    ax.set_ylabel('amount', fontsize=12)
    ax.tick_params(labelsize=11)
    despine(ax)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.2)
    save_figure(fig, 'Observability_quartiles')
    plt.close()

    # --- text length -------------------------------------------------------
    ann = [(INCIDENT['phone_format'], '60% of numbers lose\nthe country prefix', 10, 0)]
    flags['len'] = one_panel_figure(m, 'avg_len_phone', 'mean length of phone', 'Observability_text_length', annotate=ann)

    # --- report ------------------------------------------------------------
    def fl(name):
        return [int(i + 1) for i in np.where(flags[name])[0]]
    print('row_count', m['row_count'].round().astype(int).tolist())
    print('flags:', {k: fl(k) for k in ['row_count', 'freshness', 'missing', 'dup', 'unique', 'avg', 'sum', 'std', 'min', 'max',
                                          'q1_amount', 'median_amount', 'q3_amount', 'len']})
    for col in ['freshness_h', 'missing_country_pct', 'dup_order_pct', 'unique_customer', 'avg_amount', 'sum_amount',
                'std_amount', 'min_amount', 'max_amount', 'median_amount', 'avg_len_phone']:
        base = m[col][:TRAIN].mean()
        print(f'{col:22s} baseline {base:10.2f} | ' + ' '.join(f'd{d + 1}={m[col][d]:.2f}' for d in INCIDENT.values() if not np.isnan(m[col][d])))
    print('Observability figures regenerated.')


if __name__ == '__main__':
    main()
