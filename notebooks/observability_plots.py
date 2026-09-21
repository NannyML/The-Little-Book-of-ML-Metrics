"""Generate figures for the Data Observability chapter.

One dataset, twelve monitors.  A daily-partitioned `orders` table is simulated
for 42 days with weekly seasonality, and one realistic incident is injected per
monitor (a missed load, a partial load, a currency slip, an upstream rename, a
retry storm, ...).  Metrics are computed per partition using the conventions documented in the
chapter (which can differ between tools). The same illustrative detector runs on
every series (a seasonal expected range of ±z·σ around the same-weekday mean,
z = 3, trained on the first 21 partitions), and each figure shows one series
with its expected range and the points the detector flags.  Nothing is typed
in by hand.

Run from notebooks/:  uv run python observability_plots.py
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
SEED = 25  # fixed illustrative dataset, including one valid large order flagged by Max
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
    RNG = np.random.default_rng(SEED)
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
# 4. Print-sized figures: shared encoding, short annotations, explicit day numbers
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter


def figure_key(fig, missing=True):
    handles = [Patch(facecolor=BAND, label='Expected range'),
               Line2D([], [], marker='o', color=end_color, ls='', label='Flagged')]
    if missing:
        handles.append(Line2D([], [], marker='o', markerfacecolor='white',
                              markeredgecolor=MID, ls='', label='No measurement'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 1),
               ncol=len(handles), frameon=False, fontsize=14, handlelength=1,
               columnspacing=1.2, handletextpad=0.4)


def chart(m, col, ylabel, *, ax=None, ylim=None, events=(), clip=None, xmin=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.25))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.77)
        figure_key(fig, missing=m[col].isna().any())
    else:
        fig = ax.figure
    x = m.day.to_numpy() + 1
    raw = m[col].to_numpy(dtype=float)
    lo, hi, flags = expected_range(raw, m.weekday.to_numpy())
    shown = np.minimum(raw, clip) if clip is not None else raw.copy()
    ax.fill_between(x, lo, hi, color=BAND, lw=0, zorder=1)
    ax.axvspan(0.5, TRAIN + 0.5, color='#f5f5f5', lw=0, zorder=0)
    ax.plot(x, shown, color=start_color, lw=1.8, zorder=3)
    ax.scatter(x[flags], shown[flags], color=end_color, s=34, zorder=4)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.margins(y=0.25)
    # Missing values sit on a dedicated row below the axis, never at a numeric zero.
    missing = np.isnan(raw)
    ax.scatter(x[missing], [-0.07] * missing.sum(), transform=ax.get_xaxis_transform(),
               facecolors='white', edgecolors=MID, s=30, clip_on=False, zorder=5)
    for day, label, xytext in events:
        value = shown[day - 1]
        ax.annotate(label, (day, value), xytext=xytext, textcoords='axes fraction',
                    ha='center', va='center', fontsize=14, color=DARK,
                    arrowprops=dict(arrowstyle='-', color=MID, lw=0.8),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1))
    if clip is not None:
        for i in np.flatnonzero(raw > clip):
            ax.plot(x[i], clip, '^', color=end_color, ms=8, zorder=5)
            ax.annotate(f'Day {x[i]}: {raw[i]:,.0f} ↑', (x[i], clip),
                        xytext=(0, 8), textcoords='offset points', ha='center',
                        fontsize=14, color=end_color, annotation_clip=False)
    ax.set_xlim(xmin - 0.5, DAYS + 0.5)
    ax.set_xticks([d for d in [1, 8, 15, 22, 29, 36, 42] if d >= xmin])
    ax.set_xlabel('Day', fontsize=14, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=5)
    ax.tick_params(labelsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}' if abs(y) >= 1000 else f'{y:g}'))
    if xmin == 1:
        ax.text(0.02, 0.98, 'Baseline: days 1–21', transform=ax.transAxes,
                fontsize=14, va='top', color=MID)
    despine(ax)
    return fig, ax, flags


def finish(fig, name):
    save_figure(fig, 'Observability_' + name)
    plt.close(fig)


def unique_count_figure(m):
    """Compare volume and cardinality on three selected days, on identical scales."""
    days = [30, 31, 38]
    labels = ['Normal load', 'Incomplete load', 'Shortened IDs']
    fig = plt.figure(figsize=(8, 3.55))
    axes = [fig.add_axes([.29, .10, .28, .69]),
            fig.add_axes([.68, .10, .28, .69])]
    for ax, col, title, color in zip(axes,
            ['row_count', 'unique_customer'], ['Orders', 'Unique customer IDs'],
            [GREY_LINE, start_color]):
        values = m.set_index('day').loc[np.array(days)-1, col].to_numpy()
        ax.barh([2,1,0], values, height=.12, color=color, zorder=3)
        for y, value in zip([2,1,0], values):
            ax.text(0, y+.17, f'{value:,.0f}', fontsize=15, color=DARK,
                    ha='left', va='bottom')
        ax.set_xlim(0, 5500)
        ax.set_ylim(-.35, 2.55)
        ax.axis('off')
        ax.set_title(title, loc='left', fontsize=15, color=color, pad=16)
    for y, day, label in zip([2,1,0], days, labels):
        fy = .10 + .69 * (y+.35)/2.90
        fig.text(.015, fy+.025, label, fontsize=15, color=DARK, va='bottom')
        fig.text(.015, fy-.015, f'Day {day}', fontsize=14, color=MID, va='top')
    finish(fig, 'unique_count')


def main():
    m = compute_metrics(simulate())
    m.to_json(OUT / 'metrics.json', orient='records', date_format='iso', indent=1)
    # Training grouped by weekday, then the rolling detector on subsequent days.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), gridspec_kw={'wspace': .35})
    fig.subplots_adjust(left=.1, right=.98, bottom=.23, top=.77)
    figure_key(fig, missing=False)
    v = m.row_count.to_numpy(float)
    wd = m.weekday.to_numpy()
    means = np.array([v[:TRAIN][wd[:TRAIN] == i].mean() for i in range(7)])
    sigma = np.std(v[:TRAIN] - means[wd[:TRAIN]], ddof=1)
    a = axes[0]
    for i in range(7):
        points = v[:TRAIN][wd[:TRAIN] == i]
        a.vlines(i, means[i] - Z*sigma, means[i] + Z*sigma, color=BAND, lw=17)
        a.scatter([i] * len(points), points, color=start_color, s=20, zorder=3)
        a.plot([i-.2, i+.2], [means[i]]*2, color=DARK, lw=1.5, zorder=4)
    a.set_xticks(range(7), ['M','T','W','T','F','S','S'])
    a.set_ylabel('Rows', fontsize=14)
    a.set_xlabel('Weekday · first 21 days', fontsize=14)
    a.set_ylim(0, 7000)
    a.tick_params(labelsize=14)
    a.yaxis.set_major_locator(MaxNLocator(4))
    a.set_title(f'Mean ± 3σ; σ ≈ {sigma:.0f}', loc='left', fontsize=14)
    despine(a)
    chart(m, 'row_count', '', ax=axes[1], xmin=22, ylim=(0,7000))
    axes[1].set_title('New daily counts', loc='left', fontsize=14)
    finish(fig, 'expected_range')

    specs = [
        ('row_count','Rows','row_count',(0,7300),None,[(26,'No load',(.56,.24)),(31,'Partial load',(.76,.18)),(36,'Repeated rows',(.8,.92))]),
        ('freshness_h','Hours since latest arrival','freshness',(0,38),None,[(26,'No load',(.48,.77)),(31,'Stopped early',(.83,.42))]),
        ('missing_country_pct','Null country values (%)','missing',(0,55),None,[(29,'Source field renamed',(.58,.9))]),
        ('dup_order_pct','Duplicate order IDs (%)','duplicates',(-.5,16),None,[(36,'Repeated batch',(.72,.87))]),
        ('avg_amount','Mean amount','average',(36,58),55,[(32,'Refunds',(.68,.44)),(40,'Discounts',(.88,.19))]),
        ('sum_amount','Total amount','sum',(50000,330000),310000,[(31,'Partial load',(.75,.1))]),
        ('std_amount','Standard deviation','stddev',(20,44),41,[(32,'Refunds',(.68,.39)),(40,'Discounts',(.88,.61))]),
        ('avg_len_phone','Mean characters','text_length',(8,12.5),None,[(28,'Shorter format',(.70,.25))]),
    ]
    for col, ylabel, name, limits, clip, events in specs:
        fig, _, _ = chart(m,col,ylabel,ylim=limits,clip=clip,events=events)
        finish(fig,name)

    unique_count_figure(m)

    # Show only changed fields, large enough to read at the book's print size.
    fig, (a,b) = plt.subplots(1,2,figsize=(8,3.0),gridspec_kw={'width_ratios':[1,1.15],'wspace':.3})
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.9)
    a.step(m.day+1,m.n_columns,where='post',color=start_color,lw=2)
    day=INCIDENT['schema_change']+1
    a.scatter([day],[m.n_columns.iloc[day-1]],color=end_color,s=45,zorder=3)
    a.set(xlim=(.5,42.5),ylim=(5.7,7.5),xticks=[1,14,28,42],yticks=[6,7])
    a.set_xlabel('Day',fontsize=14); a.set_ylabel('Columns',fontsize=14)
    a.tick_params(labelsize=14); despine(a)
    b.axis('off')
    before,after=dict(BASE_SCHEMA),dict(NEW_SCHEMA)
    changed=[(k,t) for k,t in after.items() if before.get(k)!=t]
    lines=[f'Day {day}: schema changes']
    for name,typ in changed:
        if name not in before:
            lines += ['',f'Added: {name}',typ]
        else:
            lines += ['',f'Type changed: {name}',f'{before[name]} → {typ}']
    b.text(0,1,'\n'.join(lines),va='top',fontsize=14,linespacing=1.3,color=DARK)
    finish(fig,'schema')

    # Separate scales preserve detail in the minimum while marking the clipped maximum.
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),gridspec_kw={'wspace':.4})
    fig.subplots_adjust(left=.1,right=.98,bottom=.23,top=.75)
    figure_key(fig)
    chart(m,'min_amount','Minimum',ax=axes[0],ylim=(-150,40))
    chart(m,'max_amount','Maximum',ax=axes[1],ylim=(0,900),clip=660)
    for a in axes:
        for text in a.texts:
            if text.get_text() == 'Baseline: days 1–21':
                text.set_text('Baseline 1–21')
                text.set_position((0.02, 1.10))
    finish(fig,'min_max')

    fig,ax=plt.subplots(figsize=(8,3.5))
    fig.subplots_adjust(left=.1,right=.89,bottom=.21,top=.76)
    figure_key(fig)
    x=m.day.to_numpy()+1
    for col,color,label in [('q1_amount',MID,'Q1'),('median_amount',start_color,'Median'),('q3_amount',middle_color,'Q3')]:
        lo,hi,flag=expected_range(m[col],m.weekday)
        ax.fill_between(x,lo,hi,color=BAND,lw=0)
        ax.plot(x,m[col],color=color,lw=1.8)
        ax.scatter(x[flag],m[col].to_numpy()[flag],color=end_color,s=32,zorder=4)
        ax.text(43,m[col].iloc[-1],label,fontsize=14,color=color,va='center')
    limit=75
    avg=np.minimum(m.avg_amount.to_numpy(),limit)
    ax.plot(x,avg,color=DARK,lw=1.4,ls='--')
    ax.text(43,avg[-1]+2,'Mean',fontsize=14,color=DARK,va='center')
    day=INCIDENT['currency_slip']+1
    ax.plot(day,limit,'^',color=DARK,ms=8)
    ax.annotate(f'Day {day}: mean {m.avg_amount.iloc[day-1]:.0f} ↑',(day,limit),xytext=(0,8),textcoords='offset points',ha='center',fontsize=14)
    ax.axvspan(.5,TRAIN+.5,color='#f5f5f5',lw=0,zorder=0)
    ax.text(.02,.98,'Baseline: days 1–21',transform=ax.transAxes,fontsize=14,color=MID,va='top')
    ax.scatter([INCIDENT['no_load']+1],[-.07],transform=ax.get_xaxis_transform(),facecolors='white',edgecolors=MID,s=30,clip_on=False)
    ax.set(xlim=(.5,42.5),ylim=(0,82),xticks=[1,8,15,22,29,36,42])
    ax.set_xlabel('Day',fontsize=14); ax.set_ylabel('Amount',fontsize=14)
    ax.tick_params(labelsize=14); despine(ax)
    finish(fig,'quartiles')

    for col in ['row_count','freshness_h','missing_country_pct','dup_order_pct','unique_customer','avg_amount','sum_amount','std_amount','min_amount','max_amount','q1_amount','median_amount','q3_amount','avg_len_phone']:
        _,_,flag=expected_range(m[col],m.weekday)
        print(col, 'flagged days:', (np.flatnonzero(flag)+1).tolist())


if __name__ == '__main__':
    main()
