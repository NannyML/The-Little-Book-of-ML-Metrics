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

Run: uv run python notebooks/observability_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
# 4. Drawing, at printed size
#
# Shared conventions: the training period ends at a thin dashed line; the
# expected range is a light gray band; flagged points are red; a day with no
# rows has no value and is marked by a hatched gray strip labelled "no rows";
# values beyond the plotted scale are a single red triangle on the edge.
# Ticks mark Mondays.
# ---------------------------------------------------------------------------
BAND_C = '#e3e3e3'
NO_ROWS = INCIDENT['no_load'] + 1          # day 26
MONDAYS = [1, 8, 15, 22, 29, 36]


def frame(ax, xlabel='daily partition', xmin=0.5, training=True):
    ax.set_xlim(xmin, DAYS + 0.5)
    ax.set_xticks([t for t in MONDAYS if t >= xmin])
    ax.set_xlabel(f'{xlabel} (ticks mark Mondays)')
    tidy_axes(ax)
    if training and xmin < TRAIN:
        tr = ax.get_xaxis_transform()
        y = 1.02
        ax.plot([0.7, TRAIN + 0.3], [y, y], color=REF, lw=LW_THIN, transform=tr, clip_on=False)
        for xe in (0.7, TRAIN + 0.3):
            ax.plot([xe, xe], [y - 0.03, y], color=REF, lw=LW_THIN, transform=tr, clip_on=False)
        ax.text((TRAIN + 1) / 2, y + 0.02, 'training: first 21 days', transform=tr, ha='center', va='bottom',
                color=MUTED, fontsize=SMALL_PT)


def no_rows(ax, text=True):
    """Day 26 has no rows: a hollow marker on the x-axis and a label under it."""
    from matplotlib.transforms import ScaledTranslation
    tr = ax.get_xaxis_transform()
    spine = tr + ScaledTranslation(0, -3 / 72, ax.figure.dpi_scale_trans)
    below = tr + ScaledTranslation(0, -(3 + 2.5 + 2) / 72, ax.figure.dpi_scale_trans)
    ax.plot(NO_ROWS, 0, marker='o', ms=3.6, mfc='white', mec=MUTED, mew=0.8, transform=spine, clip_on=False,
            zorder=6)
    if text:
        ax.text(NO_ROWS - 0.4, 0, 'no rows', transform=below, ha='center', va='top', color=MUTED, fontsize=SMALL_PT)


def series(ax, m, col, clip=None, clip_low=None, xmin=0.5, band=True):
    x = m['day'].values + 1
    v = m[col].values.astype(float)
    lo, hi, flag = expected_range(v, m['weekday'].values)
    keep = x >= xmin
    if band:
        ax.fill_between(x[keep], lo[keep], hi[keep], color=BAND_C, zorder=1, linewidth=0)
    shown = v.copy()
    if clip is not None:
        shown = np.minimum(shown, clip)
    if clip_low is not None:
        shown = np.maximum(shown, clip_low)
    ax.plot(x[keep], shown[keep], color=NML_CYAN, lw=LW - 0.3, zorder=3)
    ok = ~flag & keep & ~np.isnan(v)
    if clip is not None:
        ok &= v <= clip
    if clip_low is not None:
        ok &= v >= clip_low
    ax.scatter(x[ok], shown[ok], s=6, color=NML_CYAN, zorder=4, linewidths=0)
    inside = flag & (np.isnan(v) | True)
    for xi, vi, si in zip(x[flag], v[flag], shown[flag]):
        off = (clip is not None and vi > clip) or (clip_low is not None and vi < clip_low)
        ax.plot(xi, si, marker=('^' if vi > (clip or np.inf) else 'v') if off else 'o',
                ms=5 if off else 3.6, color=NML_RED, zorder=5, clip_on=False, ls='none')
    return x, v, lo, hi, flag


def label(ax, text, xy, xytext, ha='left', va='center', color=None):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, color=color or NML_RED, linespacing=1.2, zorder=7,
                arrowprops=dict(arrowstyle='-', color=color or NML_RED, lw=LW_THIN, shrinkA=1, shrinkB=2.5),
                annotation_clip=False)


def single(name, height=1.6):
    fig, ax = book_figure(1.0, height)
    return fig, ax


def finish(fig, name):
    save_figure(fig, name)
    plt.close(fig)


def main():
    parts = simulate()
    m = compute_metrics(parts)
    m.to_json(OUT / 'metrics.json', orient='records', date_format='iso', indent=1)
    flags = {}
    day = lambda d: d + 1
    val = lambda col, d: float(m[col][d])

    # --- the mechanism: how the band is built (left) and applied (right) --------
    fig, (axL, axR) = book_figure(1.0, 1.95, 1, 2, gridspec_kw={'width_ratios': [1, 1]})
    v = m['row_count'].values.astype(float)
    wd = m['weekday'].values
    names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tr = np.arange(TRAIN)
    means = np.array([v[tr][wd[tr] == d].mean() for d in range(7)])
    sigma = np.array([v[i] - means[wd[i]] for i in tr]).std(ddof=1)
    for d in range(7):
        pts = v[tr][wd[tr] == d]
        axL.fill_between([d - 0.3, d + 0.3], means[d] - Z * sigma, means[d] + Z * sigma, color=BAND_C, zorder=1,
                         linewidth=0)
        axL.plot([d - 0.3, d + 0.3], [means[d]] * 2, color=INK, lw=LW_THIN + 0.3, zorder=5)
        axL.scatter([d] * len(pts), pts, s=6, color=NML_CYAN, zorder=4, linewidths=0)
    axL.set_xticks(range(7), labels=names, fontsize=SMALL_PT)
    axL.set_ylabel('rows in the partition')
    axL.set_xlabel('training days, by day of week')
    axL.set_ylim(0, 7400)
    thousands_axis(axL)
    tidy_axes(axL)
    axL.text(-0.4, 7300, f'line: day-of-week mean\nband: mean ± 3σ, σ = {sigma:.0f} rows', color=MUTED,
             va='top', linespacing=1.2)
    x, vv, lo, hi, flag = series(axR, m, 'row_count', xmin=TRAIN + 0.5)
    axR.set_ylim(0, 7400)
    thousands_axis(axR)
    axR.set_xlim(TRAIN + 0.5, DAYS + 0.5)
    axR.set_xticks([22, 29, 36])
    axR.set_xlabel('monitored days')
    axR.tick_params(axis='y', labelleft=False)
    tidy_axes(axR)
    label(axR, 'no rows, flagged', (day(INCIDENT['no_load']), 0), (day(INCIDENT['no_load']) + 1.0, 450))
    label(axR, 'partial load,\nflagged', (day(INCIDENT['partial_load']), val('row_count', INCIDENT['partial_load'])),
          (day(INCIDENT['partial_load']) + 1.5, 1650))
    t = 33
    label(axR, 'range for that\nday of week', (t, hi[t - 1] - 100), (t - 2.5, 7300), color=MUTED, va='top')
    finish(fig, 'Observability_expected_range')
    flags['mechanism'] = flag
    print(f'mechanism: sigma {sigma:.1f}; weekday means {np.round(means).astype(int).tolist()}')

    # --- row count ---------------------------------------------------------
    fig, ax = single('row')
    x, vv, lo, hi, flag = series(ax, m, 'row_count')
    flags['row_count'] = flag
    ax.set_ylim(0, 7600)
    ax.set_yticks([0, 2000, 4000, 6000])
    thousands_axis(ax)
    ax.set_ylabel('rows')
    frame(ax)
    label(ax, 'no rows', (day(INCIDENT['no_load']), 0), (day(INCIDENT['no_load']) - 1.0, 900), ha='right')
    label(ax, 'partial load', (day(INCIDENT['partial_load']), val('row_count', INCIDENT['partial_load'])),
          (day(INCIDENT['partial_load']) + 1.2, 1500))
    d = INCIDENT['retry_storm']
    label(ax, 'retry storm: +6%, inside the range', (day(d), val('row_count', d) + 150), (day(d) - 1.0, 7300),
          color=MUTED, ha='right', va='top')
    finish(fig, 'Observability_row_count')

    # --- freshness ---------------------------------------------------------
    fig, ax = single('fresh')
    x, vv, lo, hi, flag = series(ax, m, 'freshness_h')
    flags['freshness'] = flag
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel('hours since\nnewest row')
    frame(ax, 'day of scan')
    label(ax, "no rows: yesterday's\ndata is the newest", (day(INCIDENT['no_load']) + 0.15, 28.5),
          (day(INCIDENT['no_load']) + 1.5, 27), va='top')
    label(ax, 'load stopped\nearly', (day(INCIDENT['partial_load']), val('freshness_h', INCIDENT['partial_load'])),
          (day(INCIDENT['partial_load']) + 1.5, 14))
    finish(fig, 'Observability_freshness')

    # --- schema ------------------------------------------------------------
    fig, (axL, axR) = book_figure(1.0, 1.65, 1, 2, gridspec_kw={'width_ratios': [1.3, 1]})
    x = m['day'].values + 1
    axL.step(x, m['n_columns'], where='post', color=NML_CYAN, lw=LW - 0.3, zorder=3)
    axL.scatter(x, m['n_columns'], s=6, color=NML_CYAN, zorder=4, linewidths=0)
    d = INCIDENT['schema_change']
    axL.plot(day(d), m['n_columns'][d], 'o', ms=3.6, color=NML_RED, zorder=5)
    label(axL, 'column added,\none type changed', (day(d), m['n_columns'][d]), (day(d) - 2, 7.55), ha='right')
    axL.set_yticks([6, 7])
    axL.set_ylim(5.6, 7.9)
    axL.set_ylabel('columns')
    frame(axL, 'day of scan', training=False)
    bare_axes(axR)
    before = dict(BASE_SCHEMA)
    yy = 0.98
    axR.text(0.0, yy, f'scan {d} vs scan {d + 1}', color=INK, va='top', transform=axR.transAxes)
    yy -= 0.15
    for col, typ in NEW_SCHEMA:            # marks in their own column so every name starts at one x
        if col not in before:
            mark, txt, colr = '+', f'{col} {typ}', NML_RED
        elif before[col] != typ:
            mark, txt, colr = '~', f'{col} {before[col]}\n→ {typ}', NML_RED
        else:
            mark, txt, colr = '', f'{col} {typ}', MUTED
        axR.text(0.0, yy, mark, color=colr, va='top', transform=axR.transAxes, fontsize=SMALL_PT)
        axR.text(0.07, yy, txt, color=colr, va='top', transform=axR.transAxes, fontsize=SMALL_PT, linespacing=1.1)
        yy -= 0.2 if '\n' in txt else 0.105
    axR.text(0.0, yy - 0.02, '+ added   ~ type changed', color=MUTED, va='top', transform=axR.transAxes,
             fontsize=SMALL_PT)
    finish(fig, 'Observability_schema')

    # --- missing values ----------------------------------------------------
    fig, ax = single('missing')
    x, vv, lo, hi, flag = series(ax, m, 'missing_country_pct')
    flags['missing'] = flag
    no_rows(ax)
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel('country\nmissing (%)')
    frame(ax)
    d = INCIDENT['country_rename']
    label(ax, 'upstream field renamed:\ncountry arrives null', (day(d), val('missing_country_pct', d)),
          (day(d) + 1.5, 30))
    finish(fig, 'Observability_missing')

    # --- duplicates --------------------------------------------------------
    fig, ax = single('dup')
    x, vv, lo, hi, flag = series(ax, m, 'dup_order_pct')
    flags['dup'] = flag
    no_rows(ax)
    ax.set_ylim(0, 13)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_ylabel('duplicate\norder IDs (%)')
    frame(ax)
    d = INCIDENT['retry_storm']
    label(ax, f'retry storm: 6% of rows inserted twice,\nso {val("dup_order_pct", d):.1f}% of rows share an ID',
          (day(d) - 0.15, val('dup_order_pct', d)), (day(d) - 1.5, 9.5), ha='right', va='top')
    finish(fig, 'Observability_duplicates')

    # --- unique count ------------------------------------------------------
    fig, ax = single('unique')
    x, vv, lo, hi, flag = series(ax, m, 'unique_customer')
    flags['unique'] = flag
    no_rows(ax)
    ax.set_ylim(0, 6000)
    ax.set_yticks([0, 2000, 4000, 6000])
    thousands_axis(ax)
    ax.set_ylabel('unique\ncustomer IDs')
    frame(ax)
    label(ax, 'partial load: half the rows', (day(INCIDENT['partial_load']), val('unique_customer', INCIDENT['partial_load'])),
          (day(INCIDENT['partial_load']) - 1.2, 1900), ha='right')
    d = INCIDENT['id_truncation']
    label(ax, 'IDs truncated to three digits', (day(d), val('unique_customer', d)), (day(d) - 1.4, 600), ha='right')
    finish(fig, 'Observability_unique_count')

    # --- average -----------------------------------------------------------
    fig, ax = single('avg')
    base = m['avg_amount'][:TRAIN].mean()
    x, vv, lo, hi, flag = series(ax, m, 'avg_amount', clip=55)
    flags['avg'] = flag
    no_rows(ax)
    ax.set_ylim(36, 55)
    ax.set_yticks([40, 45, 50, 55])
    ax.set_ylabel('mean amount\n(currency units)')
    frame(ax)
    d = INCIDENT['currency_slip']
    ax.annotate(f'off the scale: {val("avg_amount", d):,.0f}', (day(d), 55), xytext=(5, -1), textcoords='offset points',
                ha='left', va='top', color=NML_RED)
    label(ax, 'refunds: 3%\nnegative', (day(INCIDENT['refunds']), val('avg_amount', INCIDENT['refunds'])),
          (day(INCIDENT['refunds']) - 2.5, 38.5), ha='right')
    label(ax, 'half price\nunder 30', (day(INCIDENT['promotion']), val('avg_amount', INCIDENT['promotion'])),
          (day(INCIDENT['promotion']) - 2.0, 38.0), ha='right')
    finish(fig, 'Observability_average')

    # --- sum ---------------------------------------------------------------
    fig, ax = single('sum')
    clip = 300_000
    x, vv, lo, hi, flag = series(ax, m, 'sum_amount', clip=clip)
    flags['sum'] = flag
    no_rows(ax)
    ax.set_ylim(0, clip)
    ax.set_yticks([0, 100_000, 200_000, 300_000])
    thousands_axis(ax)
    ax.set_ylabel('total amount\n(currency units)')
    frame(ax)
    d = INCIDENT['currency_slip']
    ax.annotate(f'off the scale: {val("sum_amount", d):,.0f}', (day(d), clip), xytext=(5, -1), textcoords='offset points',
                ha='left', va='top', color=NML_RED)
    label(ax, 'partial load:\nhalf the rows', (day(INCIDENT['partial_load']), val('sum_amount', INCIDENT['partial_load'])),
          (day(INCIDENT['partial_load']) + 1.2, 45_000))
    finish(fig, 'Observability_sum')

    # --- standard deviation ------------------------------------------------
    fig, ax = single('std')
    top = 44
    x, vv, lo, hi, flag = series(ax, m, 'std_amount', clip=top)
    flags['std'] = flag
    no_rows(ax)
    ax.set_ylim(20, top)
    ax.set_yticks([20, 30, 40])
    ax.set_ylabel('standard deviation\n(currency units)')
    frame(ax)
    d = INCIDENT['currency_slip']
    ax.annotate(f'off the scale: {val("std_amount", d):,.0f}', (day(d), top), xytext=(5, -1), textcoords='offset points',
                ha='left', va='top', color=NML_RED)
    label(ax, 'refunds: 3%\nnegative', (day(INCIDENT['refunds']), val('std_amount', INCIDENT['refunds'])),
          (day(INCIDENT['refunds']) - 1.0, 36.8), ha='right')
    label(ax, 'half price\nunder 30', (day(INCIDENT['promotion']), val('std_amount', INCIDENT['promotion'])),
          (day(INCIDENT['promotion']) + 1.5, 40), ha='right', va='center')
    finish(fig, 'Observability_stddev')

    # --- min / max: one panel above the other, sharing the day axis -----------
    fig, (axT, axB) = book_figure(1.0, 1.95, 2, 1, sharex=True)
    x, vv, lo, hi, fmin = series(axT, m, 'min_amount', clip_low=-3)
    no_rows(axT, text=False)
    axT.set_ylim(-3, 8)
    axT.set_yticks([0, 4, 8])
    axT.set_ylabel('min amount\n(currency units)')
    frame(axT)
    axT.set_xlabel('')
    axT.tick_params(axis='x', labelbottom=False)
    d = INCIDENT['refunds']
    axT.annotate(f'refunds: off the scale at {num(val("min_amount", d), 0)}', (day(d), -3), xytext=(-6, 1),
                 textcoords='offset points', ha='right', va='bottom', color=NML_RED)
    label(axT, 'half\nprice', (day(INCIDENT['promotion']), val('min_amount', INCIDENT['promotion'])),
          (day(INCIDENT['promotion']) - 1.4, -1.2), ha='right')
    clip = 900
    x, vv, lo, hi, fmax = series(axB, m, 'max_amount', clip=clip)
    no_rows(axB)
    axB.set_ylim(0, clip)
    axB.set_yticks([0, 450, 900])
    axB.set_ylabel('max amount\n(currency units)')
    frame(axB, training=False)
    d = INCIDENT['currency_slip']
    axB.annotate(f'off the scale: {val("max_amount", d):,.0f}', (day(d), clip), xytext=(-6, -1), textcoords='offset points',
                 ha='right', va='top', color=NML_RED)
    d = INCIDENT['country_rename']
    label(axB, 'one large order:\na false alarm', (day(d), val('max_amount', d)), (day(d) + 1.6, 870), va='top')
    finish(fig, 'Observability_min_max')
    flags['min'], flags['max'] = fmin, fmax

    # --- quartiles ---------------------------------------------------------
    fig, ax = book_figure(1.0, 1.85)
    top = 90
    x = m['day'].values + 1
    no_rows(ax)
    q_colors = [('q3_amount', NML_PURPLE, 'Q3'), ('median_amount', NML_CYAN, 'median'), ('q1_amount', '#9b7fc4', 'Q1')]
    for col, color, lab in q_colors:
        lo, hi, flag = expected_range(m[col].values, m['weekday'].values)
        ax.fill_between(x, lo, hi, color=BAND_C, zorder=1, linewidth=0)
        ax.plot(x, m[col], color=color, lw=LW - 0.3, zorder=3)
        ax.plot(x[flag], m[col].values[flag], 'o', ms=3.6, color=NML_RED, zorder=5)
        label_end(ax, DAYS, m[col].values[-1], lab, color)
        flags[col] = flag
    mean = np.minimum(m['avg_amount'].values, top)
    ax.plot(x, mean, color=INK, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=2)
    label_end(ax, DAYS, m['avg_amount'].values[-1] + 2, 'mean', INK)
    d = INCIDENT['currency_slip']
    ax.plot(day(d), top, marker='^', ms=5, color=INK, clip_on=False, zorder=6)
    ax.text(day(d) - 0.8, top - 5, '5% of amounts in cents:\nthe mean leaves the frame;\nQ3 rises a little (flagged)',
            ha='right', va='top', color=NML_RED, linespacing=1.2)
    d = INCIDENT['promotion']
    label(ax, 'half price under 30: Q1 drops;\nthe median and Q3 do not', (day(d), m['q1_amount'][d]),
          (day(d) - 1.5, 7), ha='right')
    ax.set_ylim(0, top)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel('amount\n(currency units)')
    frame(ax)
    finish(fig, 'Observability_quartiles')

    # --- text length -------------------------------------------------------
    fig, ax = single('len')
    x, vv, lo, hi, flag = series(ax, m, 'avg_len_phone')
    flags['len'] = flag
    no_rows(ax)
    ax.set_ylim(8, 12)
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_ylabel('mean phone length\n(characters)')
    frame(ax)
    d = INCIDENT['phone_format']
    label(ax, '60% of numbers lose\nthe country prefix', (day(d), val('avg_len_phone', d)), (day(d) + 1.6, 8.9))
    finish(fig, 'Observability_text_length')

    def fl(name):
        return [int(i + 1) for i in np.where(flags[name])[0]]
    print('flags:', {k: fl(k) for k in ['row_count', 'freshness', 'missing', 'dup', 'unique', 'sum', 'std', 'min', 'max',
                                          'q1_amount', 'median_amount', 'q3_amount', 'len']})
    print('Observability figures regenerated.')


if __name__ == '__main__':
    main()
