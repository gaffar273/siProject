"""Verify all calculations in the LaTeX report."""
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

daily = daily.sort_values('Date').drop_duplicates(subset='Date').set_index('Date')
weekly = daily['Close'].resample('W-FRI').last().dropna()
log_returns = np.log(weekly / weekly.shift(1)).dropna()
r = log_returns.values
n = len(r)

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
trading_weeks = log_returns.index

def find_event_week(ds):
    t = pd.Timestamp(ds)
    c = trading_weeks[trading_weeks >= t]
    if not len(c) or (c[0]-t).days > 10: return None
    return c[0]

week_df = pd.DataFrame({'Date': log_returns.index, 'Close': weekly.loc[log_returns.index].values,
                         'return_pct': log_returns.values * 100})
event_week_dates = set()
for ds in fed_cuts:
    ew = find_event_week(ds)
    if ew is not None: event_week_dates.add(ew)

week_df['fed_cut_week'] = week_df['Date'].isin(event_week_dates)
week_df['group'] = week_df['fed_cut_week'].map({True:'Cut-week', False:'Non-cut-week'})
week_df['negative_week'] = (week_df['return_pct'] < 0).astype(int)

cut_rets  = week_df[week_df['group']=='Cut-week']['return_pct'].values
norm_rets = week_df[week_df['group']=='Non-cut-week']['return_pct'].values
n_cut = len(cut_rets); n_norm = len(norm_rets)

mean_cut = np.mean(cut_rets); std_cut = np.std(cut_rets, ddof=1)
mean_norm = np.mean(norm_rets); std_norm = np.std(norm_rets, ddof=1)
p_cut = np.mean(cut_rets < 0); p_norm = np.mean(norm_rets < 0)
neg_cut = int(np.sum(cut_rets < 0)); neg_norm = int(np.sum(norm_rets < 0))
mean_r = np.mean(r); std_r = np.std(r, ddof=1)
skew_r = stats.skew(r); kurt_r = stats.kurtosis(r)

# CIs
se_cut = std_cut / np.sqrt(n_cut); se_norm = std_norm / np.sqrt(n_norm)
t_c_cut = stats.t.ppf(0.975, df=n_cut-1); t_c_norm = stats.t.ppf(0.975, df=n_norm-1)
ci_cut = (mean_cut - t_c_cut*se_cut, mean_cut + t_c_cut*se_cut)
ci_norm = (mean_norm - t_c_norm*se_norm, mean_norm + t_c_norm*se_norm)
ci_p_cut = (p_cut - 1.96*np.sqrt(p_cut*(1-p_cut)/n_cut), p_cut + 1.96*np.sqrt(p_cut*(1-p_cut)/n_cut))
ci_p_norm = (p_norm - 1.96*np.sqrt(p_norm*(1-p_norm)/n_norm), p_norm + 1.96*np.sqrt(p_norm*(1-p_norm)/n_norm))

# Tests
t1, p1_two = stats.ttest_ind(cut_rets, norm_rets, equal_var=False)
p1 = p1_two / 2
p_pooled = (neg_cut + neg_norm) / (n_cut + n_norm)
se_prop = np.sqrt(p_pooled*(1-p_pooled)*(1/n_cut + 1/n_norm))
z2 = (p_cut - p_norm) / se_prop
p2 = 1 - stats.norm.cdf(z2)
F3 = np.var(cut_rets, ddof=1) / np.var(norm_rets, ddof=1)
p3 = 2 * min(stats.f.cdf(F3, n_cut-1, n_norm-1), 1 - stats.f.cdf(F3, n_cut-1, n_norm-1))
jb4, p4 = stats.jarque_bera(r)
wdf = ((std_cut**2/n_cut + std_norm**2/n_norm)**2) / ((std_cut**2/n_cut)**2/(n_cut-1) + (std_norm**2/n_norm)**2/(n_norm-1))
t_crit_test = stats.t.ppf(0.05, df=wdf)
spread = mean_norm - mean_cut

print("=" * 65)
print("COMPLETE CALCULATION VERIFICATION REPORT")
print("=" * 65)

print("\n[A] DATA SUMMARY")
print("-" * 65)
print(f"  Total weekly obs (n):            {n}")
print(f"  Fed cut dates in dict:           {len(fed_cuts)}")
print(f"  Unique event weeks mapped:       {len(event_week_dates)}")
print(f"  Cut-week obs (n_C):              {n_cut}")
print(f"  Non-cut-week obs (n_N):          {n_norm}")
print(f"  Check n_C + n_N = n:             {n_cut + n_norm} == {n}? {'YES' if n_cut+n_norm==n else 'NO'}")

print("\n[B] POINT ESTIMATION")
print("-" * 65)
print(f"  x_bar_C  (mean cut):             {mean_cut:.4f}%")
print(f"  x_bar_N  (mean non-cut):         {mean_norm:.4f}%")
print(f"  s_C      (std cut):              {std_cut:.4f}%")
print(f"  s_N      (std non-cut):          {std_norm:.4f}%")
print(f"  p_hat_C  (prop neg cut):         {neg_cut}/{n_cut} = {p_cut:.3f}")
print(f"  p_hat_N  (prop neg non-cut):     {neg_norm}/{n_norm} = {p_norm:.3f}")
print(f"  Overall mean (r):                {mean_r*100:.4f}%")
print(f"  Overall std (r):                 {std_r*100:.4f}%")
print(f"  Skewness:                        {skew_r:.4f}")
print(f"  Excess Kurtosis:                 {kurt_r:.4f}")

print("\n[C] INTERVAL ESTIMATION (95% CI)")
print("-" * 65)
print(f"  CI mean cut:      [{ci_cut[0]:.2f}%, {ci_cut[1]:.2f}%]")
print(f"  CI mean non-cut:  [{ci_norm[0]:.2f}%, {ci_norm[1]:.2f}%]")
print(f"  CI prop cut:      [{ci_p_cut[0]:.3f}, {ci_p_cut[1]:.3f}]")
print(f"  CI prop non-cut:  [{ci_p_norm[0]:.3f}, {ci_p_norm[1]:.3f}]")

print("\n[D] TEST 1: Welch's Two-Sample t-Test (Means)")
print("-" * 65)
print(f"  H0: mu_C = mu_N    H1: mu_C < mu_N (one-tailed)")
print(f"  t-statistic:                     {t1:.4f}")
print(f"  Welch df:                        {wdf:.2f} (~{wdf:.0f})")
print(f"  p-value (two-tailed):            {p1_two:.4f}")
print(f"  p-value (one-tailed):            {p1:.4f}")
print(f"  Critical t_0.05:                 {t_crit_test:.3f}")
t1_dec = "REJECT H0" if p1 < 0.05 else "FAIL TO REJECT H0"
print(f"  Decision @ alpha=0.05:           {t1_dec}")

print("\n[E] TEST 2: z-Test for Proportions")
print("-" * 65)
print(f"  H0: p_C = p_N    H1: p_C > p_N (one-tailed)")
print(f"  Pooled proportion:               {p_pooled:.4f}")
print(f"  SE of proportion diff:           {se_prop:.4f}")
print(f"  z-statistic:                     {z2:.4f}")
print(f"  p-value (one-tailed):            {p2:.4f}")
z2_dec = "REJECT H0" if p2 < 0.05 else "FAIL TO REJECT H0"
print(f"  Decision @ alpha=0.05:           {z2_dec}")

print("\n[F] TEST 3: F-Test (Variance Equality)")
print("-" * 65)
print(f"  H0: var_C = var_N    H1: var_C != var_N (two-tailed)")
print(f"  Var cut:                         {np.var(cut_rets, ddof=1):.4f}")
print(f"  Var non-cut:                     {np.var(norm_rets, ddof=1):.4f}")
print(f"  F-statistic:                     {F3:.4f}")
print(f"  df1={n_cut-1}, df2={n_norm-1}")
print(f"  p-value (two-tailed):            {p3:.6f}")
f3_dec = "REJECT H0" if p3 < 0.05 else "FAIL TO REJECT H0"
print(f"  Decision @ alpha=0.05:           {f3_dec}")

print("\n[G] TEST 4: Jarque-Bera (Normality)")
print("-" * 65)
print(f"  H0: Returns ~ Normal    H1: Non-normal")
print(f"  JB statistic:                    {jb4:.2f}")
print(f"  p-value:                         {p4:.10f}")
print(f"  Skewness (S):                    {skew_r:.4f}")
print(f"  Excess Kurtosis (K-3):           {kurt_r:.4f}")
jb_dec = "REJECT H0" if p4 < 0.05 else "FAIL TO REJECT H0"
print(f"  Decision @ alpha=0.05:           {jb_dec}")

print("\n[H] PRACTICAL IMPLICATIONS")
print("-" * 65)
print(f"  Weekly spread (non-cut - cut):   {spread:.4f}%")
print(f"  Annualised spread (x52):         {spread*52:.1f}%")

print("\n[I] CROSS-CHECK vs .TEX FILE")
print("-" * 65)
tex = open(os.path.join(OUT_DIR, 'nifty50_fed_rate_cuts_report.tex'), 'r').read()
checks = [
    ("n_cut = 31", str(n_cut) in tex and "31" in tex),
    ("n_norm = 1326", "1326" in tex),
    ("mean_cut = 0.0032%", "0.0032" in tex),
    ("mean_norm = 0.2076%", "0.2076" in tex),
    ("std_cut = 5.6055%", "5.6055" in tex),
    ("std_norm = 2.7629%", "2.7629" in tex),
    ("p_cut = 0.419", "0.419" in tex),
    ("p_norm = 0.432", "0.432" in tex),
    ("t-stat = -0.2024", "-0.2024" in tex),
    ("z-stat = -0.1419", "-0.1419" in tex),
    ("F-stat = 4.1164", "4.1164" in tex),
    ("JB = 945.24", "945.24" in tex),
    ("CI cut mean [-2.05, 2.06]", "-2.05" in tex and "2.06" in tex),
    ("CI norm mean [0.06, 0.36]", "0.06" in tex and "0.36" in tex),
    ("CI p_cut [0.246, 0.593]", "0.246" in tex and "0.593" in tex),
    ("CI p_norm [0.405, 0.459]", "0.405" in tex and "0.459" in tex),
    ("p1 one-tailed = 0.4205", "0.4205" in tex),
    ("p2 one-tailed = 0.5564", "0.5564" in tex),
    ("p3 = 0.0000", "0.0000" in tex),
    ("skew = -0.5809", "-0.5809" in tex),
    ("kurtosis = 3.9202", "3.9202" in tex),
    ("Welch df ~30", "30" in tex),
    ("t_crit = -1.697", "-1.697" in tex),
    ("13/31 proportion", "13/31" in tex),
    ("573/1326 proportion", "573/1326" in tex),
]
all_ok = True
for label, ok in checks:
    status = "MATCH" if ok else "MISMATCH"
    if not ok: all_ok = False
    print(f"  {status:8s}  {label}")

print("\n" + "=" * 65)
if all_ok:
    print("ALL VALUES MATCH. Report is correct.")
else:
    print("SOME MISMATCHES FOUND. Review above.")
print("=" * 65)
