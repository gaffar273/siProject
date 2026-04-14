"""Analyze DAILY NIFTY returns on the day after each Fed rate cut."""
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

# Daily log returns
daily['return_pct'] = np.log(daily['Close'] / daily['Close'].shift(1)) * 100
daily = daily.dropna().reset_index(drop=True)

# Fed cut dates (US time - NIFTY reacts NEXT Indian trading day)
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

trading_dates = daily['Date'].values

def find_next_trading_day(cut_date_str):
    """Find the next Indian trading day after US FOMC announcement."""
    t = pd.Timestamp(cut_date_str)
    # FOMC announces in US evening = India next morning
    # So look for next trading day AFTER the cut date
    next_day = t + pd.Timedelta(days=1)
    candidates = daily[daily['Date'] >= next_day]
    if len(candidates) == 0: return None
    # Must be within 5 calendar days (to handle weekends/holidays)
    first = candidates.iloc[0]
    if (first['Date'] - t).days > 7: return None
    return first['Date']

print("=" * 70)
print("DAILY NIFTY RETURN ANALYSIS: Day After Fed Rate Cut")
print("=" * 70)

# Find next-day returns for each cut
cut_day_returns = []
print("\nDate Match Details:")
print(f"{'Cut Date':>12s}  {'NIFTY Date':>12s}  {'Return %':>10s}  {'Cut bps':>8s}")
print("-" * 50)

for ds, bps in sorted(fed_cuts.items()):
    nxt = find_next_trading_day(ds)
    if nxt is not None:
        ret = daily[daily['Date'] == nxt]['return_pct'].values[0]
        cut_day_returns.append(ret)
        print(f"{ds:>12s}  {nxt.strftime('%Y-%m-%d'):>12s}  {ret:>+10.4f}  {bps:>+8d}")

cut_day_returns = np.array(cut_day_returns)

# All other daily returns (non-event days)
event_dates = set()
for ds in fed_cuts:
    nxt = find_next_trading_day(ds)
    if nxt is not None:
        event_dates.add(nxt)

non_event_returns = daily[~daily['Date'].isin(event_dates)]['return_pct'].values

n_event = len(cut_day_returns)
n_normal = len(non_event_returns)
mean_event = np.mean(cut_day_returns)
std_event = np.std(cut_day_returns, ddof=1)
mean_normal = np.mean(non_event_returns)
std_normal = np.std(non_event_returns, ddof=1)

pos_count = np.sum(cut_day_returns > 0)
neg_count = np.sum(cut_day_returns < 0)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)
print(f"\n  Event days (day after cut):  n = {n_event}")
print(f"  Non-event days:             n = {n_normal}")
print(f"\n  Mean return (event day):     {mean_event:+.4f}%")
print(f"  Mean return (non-event):     {mean_normal:+.4f}%")
print(f"  Std dev (event day):         {std_event:.4f}%")
print(f"  Std dev (non-event):         {std_normal:.4f}%")
print(f"\n  Positive days after cut:     {pos_count}/{n_event} ({pos_count/n_event*100:.1f}%)")
print(f"  Negative days after cut:     {neg_count}/{n_event} ({neg_count/n_event*100:.1f}%)")

# Welch t-test: H1: mu_event > mu_normal
t_stat, p_two = stats.ttest_ind(cut_day_returns, non_event_returns, equal_var=False)
p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

# One-sample t-test: Is mean event-day return > 0?
t_one, p_one_sample = stats.ttest_1samp(cut_day_returns, 0)
p_one_sample_one = p_one_sample / 2.0 if t_one > 0 else 1.0 - p_one_sample / 2.0

# CI for event-day mean
se_event = std_event / np.sqrt(n_event)
t_crit = stats.t.ppf(0.975, df=n_event-1)
ci_event = (mean_event - t_crit*se_event, mean_event + t_crit*se_event)

print("\n" + "=" * 70)
print("HYPOTHESIS TESTS (DAILY)")
print("=" * 70)

print("\n--- Test A: Welch t-test (event vs non-event daily returns) ---")
print(f"  H0: mu_event = mu_normal")
print(f"  H1: mu_event > mu_normal (rate cuts boost next-day NIFTY)")
print(f"  t-statistic:     {t_stat:+.4f}")
print(f"  p-value (1-tail): {p_one:.4f}")
dec = "REJECT H0" if p_one < 0.05 else "FAIL TO REJECT H0"
print(f"  Decision:         {dec}")

print(f"\n--- Test B: One-sample t-test (is event-day return > 0?) ---")
print(f"  H0: mu_event = 0")
print(f"  H1: mu_event > 0")
print(f"  t-statistic:     {t_one:+.4f}")
print(f"  p-value (1-tail): {p_one_sample_one:.4f}")
dec2 = "REJECT H0" if p_one_sample_one < 0.05 else "FAIL TO REJECT H0"
print(f"  Decision:         {dec2}")

print(f"\n  95% CI for event-day mean:  [{ci_event[0]:+.4f}%, {ci_event[1]:+.4f}%]")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if p_one < 0.05:
    print(f"\n  DAILY analysis SUPPORTS the hypothesis!")
    print(f"  The day after a Fed rate cut, NIFTY returns are significantly")
    print(f"  higher than normal days (mean = {mean_event:+.4f}% vs {mean_normal:+.4f}%).")
else:
    print(f"\n  DAILY analysis also does NOT support the hypothesis.")
    print(f"  Event-day mean = {mean_event:+.4f}% vs normal = {mean_normal:+.4f}%")
    print(f"  The difference is not statistically significant (p = {p_one:.4f}).")

print("\n  Note: High volatility during cut-days (std = {:.4f}%) vs normal".format(std_event))
print("  days (std = {:.4f}%) suggests mixed reactions to rate cuts.".format(std_normal))
print("=" * 70)
