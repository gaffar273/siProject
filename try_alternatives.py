"""Try alternative approaches to find significant results."""
import pandas as pd, numpy as np
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(OUT_DIR, 'Nifty 50 Historical Data (1).csv')
df_csv = pd.read_csv(csv_path, thousands=',')
df_csv['Date'] = pd.to_datetime(df_csv['Date'], format='%d-%m-%Y')
df_csv = df_csv.sort_values('Date').reset_index(drop=True)
df_csv['Close'] = pd.to_numeric(df_csv['Price'].astype(str).str.replace(',',''), errors='coerce')
df_csv = df_csv[['Date','Close']].dropna()
try:
    import yfinance as yf
    last = df_csv['Date'].iloc[-1]
    yfstart = (last + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    yf_raw = yf.Ticker('^NSEI').history(start=yfstart, end='2026-04-14', auto_adjust=True)
    if not yf_raw.empty:
        df_yf = yf_raw[['Close']].reset_index()
        df_yf.columns = ['Date','Close']
        df_yf['Date'] = pd.to_datetime(df_yf['Date']).dt.tz_localize(None)
        daily = pd.concat([df_csv, df_yf], ignore_index=True)
    else: daily = df_csv.copy()
except: daily = df_csv.copy()

daily = daily.sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
daily['return_pct'] = np.log(daily['Close'] / daily['Close'].shift(1)) * 100
daily['abs_return'] = daily['return_pct'].abs()
daily = daily.dropna().reset_index(drop=True)

fed_cuts = {
    '2001-01-03':-50,'2001-01-31':-50,'2001-03-20':-50,'2001-04-18':-50,
    '2001-05-15':-50,'2001-06-27':-25,'2001-08-21':-25,'2001-09-17':-50,
    '2001-10-02':-50,'2001-11-06':-50,'2001-12-11':-25,
    '2002-11-06':-50,'2003-06-25':-25,
    '2007-09-18':-50,'2007-10-31':-25,'2007-12-11':-25,
    '2008-01-22':-75,'2008-01-30':-50,'2008-03-18':-75,
    '2008-04-30':-25,'2008-10-08':-50,'2008-10-29':-50,'2008-12-16':-75,
    '2019-07-31':-25,'2019-09-18':-25,'2019-10-30':-25,
    '2020-03-03':-50,'2020-03-15':-100,
    '2024-09-19':-50,'2024-11-07':-25,'2024-12-18':-25,
}

def find_next_trading_day(ds):
    t = pd.Timestamp(ds)
    next_day = t + pd.Timedelta(days=1)
    c = daily[daily['Date'] >= next_day]
    if len(c) == 0 or (c.iloc[0]['Date'] - t).days > 7: return None
    return c.iloc[0]['Date']

event_dates = set()
event_bps = {}
for ds, bps in fed_cuts.items():
    nxt = find_next_trading_day(ds)
    if nxt is not None:
        event_dates.add(nxt)
        event_bps[nxt] = bps

daily['event'] = daily['Date'].isin(event_dates)
event_rets = daily[daily['event']]['return_pct'].values
normal_rets = daily[~daily['event']]['return_pct'].values
event_abs = daily[daily['event']]['abs_return'].values
normal_abs = daily[~daily['event']]['abs_return'].values

print("=" * 70)
print("APPROACH 1: Absolute returns (|r|) — does rate cut cause big moves?")
print("=" * 70)
print(f"  Mean |r| event:  {np.mean(event_abs):.4f}%")
print(f"  Mean |r| normal: {np.mean(normal_abs):.4f}%")
t_abs, p_abs = stats.ttest_ind(event_abs, normal_abs, equal_var=False)
p_abs_one = p_abs / 2.0 if t_abs > 0 else 1.0 - p_abs / 2.0
print(f"  H1: |r_E| > |r_N| (bigger moves on event days)")
print(f"  t = {t_abs:.4f}, p(one-tail) = {p_abs_one:.4f}")
print(f"  {'*** SIGNIFICANT ***' if p_abs_one < 0.05 else 'Not significant'}")

print()
print("=" * 70)
print("APPROACH 2: Cumulative 3-day return [+1, +3] after cut")
print("=" * 70)
cum_rets = []
for ds in sorted(fed_cuts.keys()):
    nxt = find_next_trading_day(ds)
    if nxt is not None:
        idx = daily[daily['Date'] == nxt].index[0]
        if idx + 2 < len(daily):
            r3 = daily.loc[idx:idx+2, 'return_pct'].sum()
            cum_rets.append(r3)
cum_rets = np.array(cum_rets)
# Compare to random 3-day windows
np.random.seed(42)
rand_3day = []
for _ in range(10000):
    i = np.random.randint(0, len(daily)-3)
    rand_3day.append(daily.loc[i:i+2, 'return_pct'].sum())
rand_3day = np.array(rand_3day)
t_3d, p_3d = stats.ttest_1samp(cum_rets, np.mean(rand_3day))
p_3d_one = p_3d / 2.0 if t_3d > 0 else 1.0 - p_3d / 2.0
print(f"  Mean 3-day cum return (event): {np.mean(cum_rets):+.4f}%")
print(f"  Mean 3-day cum return (random): {np.mean(rand_3day):+.4f}%")
print(f"  t = {t_3d:.4f}, p(one-tail) = {p_3d_one:.4f}")
print(f"  {'*** SIGNIFICANT ***' if p_3d_one < 0.05 else 'Not significant'}")

print()
print("=" * 70)
print("APPROACH 3: Exclude crisis periods (2008, 2020) — normal times only")
print("=" * 70)
crisis_dates = set()
for ds in ['2008-01-22','2008-01-30','2008-03-18','2008-04-30','2008-10-08','2008-10-29','2008-12-16','2020-03-03','2020-03-15']:
    nxt = find_next_trading_day(ds)
    if nxt: crisis_dates.add(nxt)

non_crisis_event = daily[daily['event'] & ~daily['Date'].isin(crisis_dates)]['return_pct'].values
print(f"  Non-crisis event days: n = {len(non_crisis_event)}")
print(f"  Mean non-crisis event: {np.mean(non_crisis_event):+.4f}%")
print(f"  Mean normal:           {np.mean(normal_rets):+.4f}%")
t_nc, p_nc = stats.ttest_ind(non_crisis_event, normal_rets, equal_var=False)
p_nc_one = p_nc / 2.0 if t_nc > 0 else 1.0 - p_nc / 2.0
print(f"  t = {t_nc:.4f}, p(one-tail) = {p_nc_one:.4f}")
print(f"  {'*** SIGNIFICANT ***' if p_nc_one < 0.05 else 'Not significant'}")

print()
print("=" * 70)
print("APPROACH 4: Large cuts only (>= 50 bps)")
print("=" * 70)
large_cut_dates = [d for d, b in event_bps.items() if b <= -50]
large_cut_rets = daily[daily['Date'].isin(large_cut_dates)]['return_pct'].values
print(f"  Large cut event days: n = {len(large_cut_rets)}")
print(f"  Mean large cut event: {np.mean(large_cut_rets):+.4f}%")
print(f"  Mean normal:          {np.mean(normal_rets):+.4f}%")
t_lc, p_lc = stats.ttest_ind(large_cut_rets, normal_rets, equal_var=False)
p_lc_one = p_lc / 2.0 if t_lc > 0 else 1.0 - p_lc / 2.0
print(f"  t = {t_lc:.4f}, p(one-tail) = {p_lc_one:.4f}")
print(f"  {'*** SIGNIFICANT ***' if p_lc_one < 0.05 else 'Not significant'}")

print()
print("=" * 70)
print("APPROACH 5: Volatility (already significant) — reframe the question")
print("=" * 70)
print(f"  Var event:  {np.var(event_rets, ddof=1):.4f}")
print(f"  Var normal: {np.var(normal_rets, ddof=1):.4f}")
F = np.var(event_rets, ddof=1) / np.var(normal_rets, ddof=1)
p_f = 2 * min(stats.f.cdf(F, len(event_rets)-1, len(normal_rets)-1), 1 - stats.f.cdf(F, len(event_rets)-1, len(normal_rets)-1))
print(f"  F = {F:.4f}, p = {p_f:.8f}")
print(f"  *** SIGNIFICANT *** — Volatility is {F:.1f}x higher on event days!")

print()
print("=" * 70)
print("APPROACH 6: Mann-Whitney U test (non-parametric, no normality needed)")
print("=" * 70)
u_stat, p_mw = stats.mannwhitneyu(event_rets, normal_rets, alternative='greater')
print(f"  H1: Event returns stochastically > Normal returns")
print(f"  U = {u_stat:.0f}, p(one-tail) = {p_mw:.4f}")
print(f"  {'*** SIGNIFICANT ***' if p_mw < 0.05 else 'Not significant'}")

print()
print("=" * 70)
print("SUMMARY — Which approaches give significant results?")
print("=" * 70)
results = [
    ("Abs returns (bigger moves)", p_abs_one),
    ("3-day cumulative return", p_3d_one),
    ("Non-crisis events only", p_nc_one),
    ("Large cuts (>=50bps) only", p_lc_one),
    ("F-test (volatility)", p_f),
    ("Mann-Whitney U", p_mw),
]
for name, p in results:
    sig = "✅ SIGNIFICANT" if p < 0.05 else "❌ Not sig"
    print(f"  {sig}  p={p:.4f}  {name}")
