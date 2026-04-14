"""
NIFTY 50 Weekly Analysis: Impact of US Fed Rate Decisions (2000–2026)
=====================================================================
Uses:
  1) Local CSV: "Nifty 50 Historical Data (1).csv"  (daily, Apr 2000 – May 2020)
  2) Yahoo Finance fill-in for Jun 2020 – Apr 2026

Converts daily → weekly (Friday close), then runs the full statistical pipeline.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings, os
warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════
# 1.  LOAD & MERGE DATA
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 1 — Loading Data")
print("=" * 70)

# ── 1a. CSV data (Apr 2000 – May 2020) ───────────────────────────────
csv_path = os.path.join(OUT_DIR, "Nifty 50 Historical Data (1).csv")
df_csv = pd.read_csv(csv_path, thousands=",")
df_csv["Date"] = pd.to_datetime(df_csv["Date"], format="%d-%m-%Y")
df_csv = df_csv.sort_values("Date").reset_index(drop=True)
df_csv = df_csv.rename(columns={"Price": "Close"})
df_csv["Close"] = pd.to_numeric(df_csv["Close"].astype(str).str.replace(",", ""), errors="coerce")
df_csv = df_csv[["Date", "Close"]].dropna()
print(f"  CSV  : {len(df_csv)} daily obs  ({df_csv['Date'].iloc[0].date()} → {df_csv['Date'].iloc[-1].date()})")

# ── 1b. Yahoo Finance data (Jun 2020 – Apr 2026) ────────────────────
try:
    import yfinance as yf
    last_csv = df_csv["Date"].iloc[-1]
    yfstart  = (last_csv + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tk = yf.Ticker("^NSEI")
    yf_raw = tk.history(start=yfstart, end="2026-04-14", auto_adjust=True)
    if not yf_raw.empty:
        df_yf = yf_raw[["Close"]].reset_index()
        df_yf.columns = ["Date", "Close"]
        df_yf["Date"] = pd.to_datetime(df_yf["Date"]).dt.tz_localize(None)
        print(f"  YF   : {len(df_yf)} daily obs  ({df_yf['Date'].iloc[0].date()} → {df_yf['Date'].iloc[-1].date()})")
        daily = pd.concat([df_csv, df_yf], ignore_index=True)
    else:
        daily = df_csv.copy()
except Exception as e:
    print(f"  [YF unavailable: {e}] – using CSV only")
    daily = df_csv.copy()

daily = daily.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
daily = daily.set_index("Date")
print(f"  TOTAL: {len(daily)} daily obs  ({daily.index[0].date()} → {daily.index[-1].date()})")

# ── 1c. Resample to WEEKLY (Friday close) ────────────────────────────
weekly = daily["Close"].resample("W-FRI").last().dropna()
print(f"  WEEKLY: {len(weekly)} weekly obs  ({weekly.index[0].date()} → {weekly.index[-1].date()})")

# ═══════════════════════════════════════════════════════════════════════
# 2.  WEEKLY LOG RETURNS
# ═══════════════════════════════════════════════════════════════════════
#   r_t = ln(P_t / P_{t-1})
log_returns = np.log(weekly / weekly.shift(1)).dropna()
trading_weeks = log_returns.index
r = log_returns.values
n = len(r)

print(f"\n  Weekly log returns: N = {n}")

# ═══════════════════════════════════════════════════════════════════════
# 3.  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2 — Descriptive Statistics of Weekly Log Returns")
print("=" * 70)

mean_r   = np.mean(r)
std_r    = np.std(r, ddof=1)
skew_r   = stats.skew(r)
kurt_r   = stats.kurtosis(r)   # excess kurtosis
median_r = np.median(r)
min_r    = np.min(r)
max_r    = np.max(r)
min_idx  = log_returns.idxmin()
max_idx  = log_returns.idxmax()

ann_ret  = mean_r * 52
ann_vol  = std_r * np.sqrt(52)
sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0

print(f"  N (weeks)            : {n}")
print(f"  Mean weekly return   : {mean_r*100:.4f}%")
print(f"  Std deviation        : {std_r*100:.4f}%")
print(f"  Median               : {median_r*100:.4f}%")
print(f"  Min (worst week)     : {min_r*100:.4f}%  ({min_idx.date()})")
print(f"  Max (best week)      : {max_r*100:.4f}%  ({max_idx.date()})")
print(f"  Skewness             : {skew_r:.4f}")
print(f"  Excess kurtosis      : {kurt_r:.4f}")
print(f"  Annualised return    : {ann_ret*100:.2f}%")
print(f"  Annualised volatility: {ann_vol*100:.2f}%")
print(f"  Sharpe ratio (rf=0)  : {sharpe:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 4.  MOM & MLE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3 — Parameter Estimation (MOM & MLE)")
print("=" * 70)
print()
print("  --- Method of Moments (MOM) ---")
print("  mu_MOM  = (1/n) * sum(r_t)  =  sample mean")
print("  sig_MOM = sqrt[ 1/(n-1) * sum((r_t - r_bar)^2) ]  (unbiased)")
mu_mom  = np.mean(r)
sig_mom = np.std(r, ddof=1)
print(f"  mu_MOM  = {mu_mom*100:.5f}%")
print(f"  sig_MOM = {sig_mom*100:.5f}%")
print()
print("  --- Maximum Likelihood (MLE, Normal assumption) ---")
print("  mu_MLE  = r_bar  (same as MOM)")
print("  sig_MLE = sqrt[ 1/n * sum((r_t - r_bar)^2) ]  (biased)")
mu_mle   = np.mean(r)
sig_mle  = np.std(r, ddof=0)
var_mle  = sig_mle ** 2
print(f"  mu_MLE  = {mu_mle*100:.5f}%")
print(f"  sig_MLE = {sig_mle*100:.5f}%")
print(f"  sig2_MLE= {var_mle*10000:.6f}  (×10⁻⁴)")
print()
print("  --- Student-t MLE ---")
df_t, loc_t, scale_t = stats.t.fit(r)
print(f"  nu (degrees of freedom) = {df_t:.3f}")
print(f"  loc  = {loc_t*100:.5f}%")
print(f"  scale= {scale_t*100:.5f}%")

# ═══════════════════════════════════════════════════════════════════════
# 5.  CONFIDENCE INTERVALS  (95%)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 4 — 95% Confidence Intervals")
print("=" * 70)

alpha = 0.05

# CI for mu (t-distribution)
se_mu = sig_mom / np.sqrt(n)
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
ci_mu = (mu_mle - t_crit * se_mu, mu_mle + t_crit * se_mu)
print(f"\n  CI for mu (weekly return):")
print(f"    r_bar ± t_(a/2, n-1) * s/sqrt(n)")
print(f"    = {mu_mle*100:.5f}% ± {t_crit:.4f} * {se_mu*100:.5f}%")
print(f"    = [{ci_mu[0]*100:.5f}%,  {ci_mu[1]*100:.5f}%]")

# CI for variance (chi-squared)
chi_lo = stats.chi2.ppf(alpha/2, df=n-1)
chi_hi = stats.chi2.ppf(1 - alpha/2, df=n-1)
ci_var = ((n-1)*sig_mom**2/chi_hi, (n-1)*sig_mom**2/chi_lo)
print(f"\n  CI for sigma^2:")
print(f"    [(n-1)s^2 / chi2_(a/2), (n-1)s^2 / chi2_(1-a/2)]")
print(f"    = [{ci_var[0]*10000:.5f},  {ci_var[1]*10000:.5f}]  (×10⁻⁴)")

# ═══════════════════════════════════════════════════════════════════════
# 6.  FED RATE CUT EVENTS  (complete list 2001–2024)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5 — Fed Rate Events (Rate CUTS only for event study)")
print("=" * 70)

# All major Fed rate cuts 2001–2024
fed_cuts = {
    # Dot-com bust
    "2001-01-03": -50,  "2001-01-31": -50,  "2001-03-20": -50,
    "2001-04-18": -50,  "2001-05-15": -50,  "2001-06-27": -25,
    "2001-08-21": -25,  "2001-09-17": -50,  "2001-10-02": -50,
    "2001-11-06": -50,  "2001-12-11": -25,
    "2002-11-06": -50,
    "2003-06-25": -25,
    # GFC
    "2007-09-18": -50,  "2007-10-31": -25,  "2007-12-11": -25,
    "2008-01-22": -75,  "2008-01-30": -50,  "2008-03-18": -75,
    "2008-04-30": -25,  "2008-10-08": -50,  "2008-10-29": -50,
    "2008-12-16": -75,
    # 2019 insurance cuts
    "2019-07-31": -25,  "2019-09-18": -25,  "2019-10-30": -25,
    # COVID emergency
    "2020-03-03": -50,  "2020-03-15": -100,
    # 2024 cuts
    "2024-09-19": -50,  "2024-11-07": -25,  "2024-12-18": -25,
}

# Also keep hikes for H2 comparison
fed_hikes = {
    "2004-06-30": +25,  "2004-08-10": +25,  "2004-09-21": +25,
    "2004-11-10": +25,  "2004-12-14": +25,
    "2005-02-02": +25,  "2005-03-22": +25,  "2005-05-03": +25,
    "2005-06-30": +25,  "2005-08-09": +25,  "2005-09-20": +25,
    "2005-11-01": +25,  "2005-12-13": +25,
    "2006-01-31": +25,  "2006-03-28": +25,  "2006-05-10": +25,
    "2006-06-29": +25,
    "2015-12-16": +25,  "2016-12-14": +25,
    "2017-03-15": +25,  "2017-06-14": +25,  "2017-12-13": +25,
    "2018-03-21": +25,  "2018-06-13": +25,  "2018-09-26": +25,
    "2018-12-19": +25,
    "2022-03-17": +25,  "2022-05-05": +50,  "2022-06-16": +75,
    "2022-07-28": +75,  "2022-09-22": +75,  "2022-11-03": +75,
    "2022-12-15": +50,
    "2023-02-02": +25,  "2023-03-23": +25,  "2023-05-04": +25,
}

def find_event_week(date_str):
    """Find the weekly index that contains or follows the event date."""
    target = pd.Timestamp(date_str)
    candidates = trading_weeks[trading_weeks >= target]
    if len(candidates) == 0:
        return None
    nearest = candidates[0]
    # Allow up to 10 days gap (weekends + holidays)
    if (nearest - target).days > 10:
        return None
    return nearest

# ═══════════════════════════════════════════════════════════════════════
# 7.  EVENT STUDY — CAR  [-1, 0, +1] weeks
# ═══════════════════════════════════════════════════════════════════════
#   Estimation window: 60 weeks before event (weeks -70 to -11)
#   Event window: [-1, 0, +1] weeks around event
#
#   mu_hat = mean of estimation window returns
#   AR_t   = r_t - mu_hat
#   CAR_i  = AR_{-1} + AR_0 + AR_{+1}

print("\n" + "=" * 70)
print("  STEP 6 — Event Study: Cumulative Abnormal Returns (Weekly)")
print("  Estimation window: 60 weeks  |  Event window: [-1,0,+1] weeks")
print("=" * 70)

EST_WINDOW = 60
BUFFER     = 10  # gap between estimation and event window

def compute_cars(events_dict, event_type):
    """Compute CARs for a dict of {date_str: bps} events."""
    results = []
    for date_str, bps in events_dict.items():
        ew = find_event_week(date_str)
        if ew is None:
            continue
        idx = trading_weeks.get_loc(ew)
        # Need: idx-EST_WINDOW-BUFFER >= 0 and idx+1 < len
        if idx < (EST_WINDOW + BUFFER + 1) or idx + 1 >= len(trading_weeks):
            continue
        # Estimation window
        est_start = idx - EST_WINDOW - BUFFER
        est_end   = idx - BUFFER
        mu_est    = np.mean(log_returns.iloc[est_start:est_end].values)
        # Abnormal returns
        ar_m1 = float(log_returns.iloc[idx - 1]) - mu_est
        ar_0  = float(log_returns.iloc[idx])     - mu_est
        ar_p1 = float(log_returns.iloc[idx + 1]) - mu_est
        car   = ar_m1 + ar_0 + ar_p1
        results.append({
            "date": date_str, "type": event_type, "bps": bps,
            "mu_est": mu_est, "ar_m1": ar_m1, "ar_0": ar_0, "ar_p1": ar_p1, "car": car,
            "week": ew.date(),
        })
    return results

cut_results  = compute_cars(fed_cuts, "Cut")
hike_results = compute_cars(fed_hikes, "Hike")
all_results  = cut_results + hike_results
all_results.sort(key=lambda x: x["date"])

# Print cuts
print(f"\n  {'Date':<13}{'bps':>5}  {'Week':>12}  {'mu_est':>8}  {'AR(-1)':>8}  {'AR(0)':>8}  {'AR(+1)':>8}  {'CAR':>8}")
print("  " + "-" * 74)
for r_item in cut_results:
    print(f"  {r_item['date']:<13}{r_item['bps']:>5}  {str(r_item['week']):>12}  "
          f"{r_item['mu_est']*100:>7.3f}%  {r_item['ar_m1']*100:>7.3f}%  "
          f"{r_item['ar_0']*100:>7.3f}%  {r_item['ar_p1']*100:>7.3f}%  {r_item['car']*100:>7.3f}%")

df_all  = pd.DataFrame(all_results)
df_cuts = pd.DataFrame(cut_results)
df_hikes= pd.DataFrame(hike_results)

cars_all  = df_all["car"].values if len(df_all) else np.array([])
cars_cut  = df_cuts["car"].values if len(df_cuts) else np.array([])
cars_hike = df_hikes["car"].values if len(df_hikes) else np.array([])

print("  " + "-" * 74)
print(f"  CUTS  → Mean CAR = {np.mean(cars_cut)*100:.4f}%   (N={len(cars_cut)})")
if len(cars_hike):
    print(f"  HIKES → Mean CAR = {np.mean(cars_hike)*100:.4f}%   (N={len(cars_hike)})")
print(f"  ALL   → Mean CAR = {np.mean(cars_all)*100:.4f}%   (N={len(cars_all)})")

# Split cuts into large (≥50 bps) vs small (<50 bps) for H2
if len(df_cuts):
    cars_large = df_cuts[df_cuts["bps"].abs() >= 50]["car"].values
    cars_small = df_cuts[df_cuts["bps"].abs() < 50]["car"].values
    print(f"\n  LARGE cuts (≥50bps): Mean CAR = {np.mean(cars_large)*100:.4f}%  (N={len(cars_large)})")
    print(f"  SMALL cuts (<50bps): Mean CAR = {np.mean(cars_small)*100:.4f}%  (N={len(cars_small)})")

# CI for mean CAR (cuts)
m_cut    = len(cars_cut)
se_car   = np.std(cars_cut, ddof=1) / np.sqrt(m_cut)
t_c_car  = stats.t.ppf(1 - alpha/2, df=m_cut - 1)
ci_car   = (np.mean(cars_cut) - t_c_car * se_car, np.mean(cars_cut) + t_c_car * se_car)

print(f"\n  95% CI for mean CAR (cuts):")
print(f"    CAR_bar ± t_(a/2, m-1) * s_CAR / sqrt(m)")
print(f"    = [{ci_car[0]*100:.4f}%,  {ci_car[1]*100:.4f}%]")
if not (ci_car[0] < 0 < ci_car[1]):
    print("    >> CI excludes 0 → STATISTICALLY SIGNIFICANT")
else:
    print("    >> CI includes 0 → not significant at 95%")

# ═══════════════════════════════════════════════════════════════════════
# 8.  HYPOTHESIS TESTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7 — Hypothesis Tests")
print("=" * 70)

# H1: Mean CAR (cuts) = 0
print("\n  H1: Mean CAR = 0  (one-sample t-test)")
print("    H0: mu_CAR = 0   vs   H1: mu_CAR ≠ 0")
print("    t = CAR_bar / (s_CAR / sqrt(m))")
t1, p1 = stats.ttest_1samp(cars_cut, popmean=0)
print(f"    t = {t1:.4f},  p = {p1:.4f}")
print(f"    → {'** REJECT H0 **' if p1 < 0.05 else 'Fail to Reject H0'} at α=0.05")

# H2: Large cuts CAR = Small cuts CAR (Welch t-test)
print("\n  H2: CAR(large cuts) = CAR(small cuts)  (Welch t-test)")
print("    H0: mu_large = mu_small   vs   H1: mu_large ≠ mu_small")
if len(cars_large) >= 2 and len(cars_small) >= 2:
    t2, p2 = stats.ttest_ind(cars_large, cars_small, equal_var=False)
    # Welch-Satterthwaite df
    s1, n1 = np.std(cars_large, ddof=1), len(cars_large)
    s2, n2 = np.std(cars_small, ddof=1), len(cars_small)
    nu_ws = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    print(f"    t = {t2:.4f},  p = {p2:.4f},  Welch df = {nu_ws:.2f}")
    print(f"    → {'** REJECT H0 **' if p2 < 0.05 else 'Fail to Reject H0'} at α=0.05")
else:
    t2, p2 = np.nan, np.nan
    print("    [Insufficient data in one group]")

# H3: Variance during event weeks vs normal weeks (F-test)
print("\n  H3: Var(event weeks) = Var(normal weeks)  (F-test)")
print("    H0: sigma^2_event = sigma^2_normal")
event_week_set = set()
for date_str in list(fed_cuts.keys()) + list(fed_hikes.keys()):
    ew = find_event_week(date_str)
    if ew is not None:
        idx = trading_weeks.get_loc(ew)
        for offset in [-1, 0, 1]:
            if 0 <= idx + offset < len(trading_weeks):
                event_week_set.add(trading_weeks[idx + offset])

event_rets  = log_returns[log_returns.index.isin(event_week_set)].values
normal_rets = log_returns[~log_returns.index.isin(event_week_set)].values

if len(event_rets) >= 2 and len(normal_rets) >= 2:
    var_ev = np.var(event_rets, ddof=1)
    var_nm = np.var(normal_rets, ddof=1)
    F_stat = var_ev / var_nm
    df1, df2 = len(event_rets) - 1, len(normal_rets) - 1
    p3 = 2 * min(stats.f.cdf(F_stat, df1, df2), 1 - stats.f.cdf(F_stat, df1, df2))
    print(f"    F = {F_stat:.4f}  (df1={df1}, df2={df2})")
    print(f"    p = {p3:.4f}")
    print(f"    → {'** REJECT H0 **' if p3 < 0.05 else 'Fail to Reject H0'} at α=0.05")
else:
    F_stat, p3 = np.nan, np.nan

# H4: Normality (Jarque-Bera)
print("\n  H4: Returns are normally distributed  (Jarque-Bera)")
print("    H0: S=0 and K=3 (normal)  vs  H1: non-normal")
print("    JB = (n/6) * [S^2 + (K-3)^2 / 4]")
jb_stat, p4 = stats.jarque_bera(r)
print(f"    JB = {jb_stat:.4f},  p = {p4:.6f}")
print(f"    → {'** REJECT H0 **' if p4 < 0.05 else 'Fail to Reject H0'} at α=0.05")

# ═══════════════════════════════════════════════════════════════════════
# 9.  ERA-WISE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 8 — Era-Wise Performance")
print("=" * 70)
eras = {
    "Dot-com bust   (2000-02)": ("2000-01-01", "2002-12-31"),
    "Recovery        (2003-06)": ("2003-01-01", "2006-12-31"),
    "GFC             (2007-09)": ("2007-01-01", "2009-12-31"),
    "Post-GFC rally  (2010-19)": ("2010-01-01", "2019-12-31"),
    "COVID crash     (2020)   ": ("2020-01-01", "2020-12-31"),
    "Post-COVID bull (2021)   ": ("2021-01-01", "2021-12-31"),
    "Hike cycle      (2022-23)": ("2022-01-01", "2023-12-31"),
    "Cut cycle       (2024-26)": ("2024-01-01", "2026-04-14"),
}
print(f"  {'Era':<32}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'Sharpe':>7}  {'Skew':>7}")
print("  " + "-" * 68)
era_data = []
for label, (s, e) in eras.items():
    era_r = log_returns.loc[s:e].values
    if len(era_r) < 10:
        continue
    ar = np.mean(era_r) * 52
    av = np.std(era_r, ddof=1) * np.sqrt(52)
    sh = ar / av if av > 0 else 0
    sk = stats.skew(era_r) if len(era_r) > 2 else 0
    era_data.append({"label": label, "ann_ret": ar, "ann_vol": av, "sharpe": sh, "skew": sk})
    print(f"  {label:<32}  {ar*100:>7.2f}%  {av*100:>7.2f}%  {sh:>7.3f}  {sk:>7.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 10.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ALL KEY NUMBERS (copy into report)")
print("=" * 70)
print(f"  N (weekly obs)        = {n}")
print(f"  Mean weekly return    = {mean_r*100:.4f}%")
print(f"  Std dev (weekly)      = {std_r*100:.4f}%")
print(f"  Skewness              = {skew_r:.4f}")
print(f"  Excess kurtosis       = {kurt_r:.4f}")
print(f"  Annualised return     = {ann_ret*100:.2f}%")
print(f"  Annualised volatility = {ann_vol*100:.2f}%")
print(f"  Sharpe (rf=0)         = {sharpe:.4f}")
print(f"  MOM mu                = {mu_mom*100:.5f}%")
print(f"  MOM sigma             = {sig_mom*100:.5f}%")
print(f"  MLE sigma             = {sig_mle*100:.5f}%")
print(f"  t-dist nu             = {df_t:.3f}")
print(f"  95% CI mu             = [{ci_mu[0]*100:.5f}%, {ci_mu[1]*100:.5f}%]")
print(f"  95% CI sigma^2 (x1e4) = [{ci_var[0]*10000:.5f}, {ci_var[1]*10000:.5f}]")
print(f"  95% CI CAR (cuts)     = [{ci_car[0]*100:.4f}%, {ci_car[1]*100:.4f}%]")
print(f"  Mean CAR (cuts)       = {np.mean(cars_cut)*100:.4f}%")
print(f"  Mean CAR (large cuts) = {np.mean(cars_large)*100:.4f}%")
print(f"  Mean CAR (small cuts) = {np.mean(cars_small)*100:.4f}%")
if len(cars_hike):
    print(f"  Mean CAR (hikes)      = {np.mean(cars_hike)*100:.4f}%")
print(f"  H1: t={t1:.4f}, p={p1:.4f}")
if not np.isnan(t2):
    print(f"  H2: t={t2:.4f}, p={p2:.4f}")
if not np.isnan(F_stat):
    print(f"  H3: F={F_stat:.4f}, p={p3:.4f}")
print(f"  H4: JB={jb_stat:.2f}, p={p4:.6f}")
print(f"  N events (cuts)       = {len(cars_cut)}")
print(f"  N events (hikes)      = {len(cars_hike)}")
print(f"  N events (all)        = {len(cars_all)}")

# ═══════════════════════════════════════════════════════════════════════
# 11.  PLOTS
# ═══════════════════════════════════════════════════════════════════════
print("\n  Generating plots...")

BG   = "#0f1117";  FG = "#e8eaf0";  GREY = "#2a2d3a"
ACC1 = "#4fc3f7";  ACC2 = "#ef5350";  ACC3 = "#66bb6a";  ACC4 = "#ffa726"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GREY, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GREY,
    "grid.linestyle": "--", "grid.alpha": 0.5,
})

fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.suptitle("NIFTY 50  ·  Impact of US Fed Rate Cuts  (Weekly, 2000–2026)",
             fontsize=18, fontweight="bold", color=FG, y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32)

# ── Plot 1: Weekly price + events ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(weekly.index, weekly.values, color=ACC1, linewidth=1, label="NIFTY 50 Weekly")
ax1.fill_between(weekly.index, weekly.values, alpha=0.06, color=ACC1)
for ds, bps in fed_cuts.items():
    ew = find_event_week(ds)
    if ew is not None:
        ax1.axvline(ew, color=ACC3, alpha=0.4, linewidth=0.7)
for ds, bps in fed_hikes.items():
    ew = find_event_week(ds)
    if ew is not None:
        ax1.axvline(ew, color=ACC2, alpha=0.25, linewidth=0.5)
ax1.legend(handles=[
    Line2D([0],[0], color=ACC1, lw=2, label="NIFTY 50"),
    Line2D([0],[0], color=ACC3, lw=1.5, alpha=0.8, label="Fed Cut"),
    Line2D([0],[0], color=ACC2, lw=1.5, alpha=0.8, label="Fed Hike"),
], loc="upper left", facecolor=GREY, edgecolor="none", labelcolor=FG)
ax1.set_title("NIFTY 50 Weekly Price with Fed Rate Decision Events", color=FG, fontsize=13)
ax1.set_ylabel("Index Level", color=FG)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.grid(True)

# ── Plot 2: CAR bar chart (cuts only) ────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
if len(df_cuts):
    colors_bar = [ACC3 if abs(b) >= 50 else ACC4 for b in df_cuts["bps"]]
    ax2.bar(range(len(df_cuts)), df_cuts["car"].values * 100, color=colors_bar, alpha=0.85, edgecolor="none")
    ax2.axhline(0, color=FG, lw=0.8)
    ax2.axhline(np.mean(cars_cut)*100, color=ACC2, lw=1.5, ls="--")
    ax2.set_xticks(range(len(df_cuts)))
    ax2.set_xticklabels([f"{r['date']}\n{r['bps']}bps" for _, r in df_cuts.iterrows()],
                        rotation=45, ha="right", fontsize=6.5)
    ax2.legend(handles=[
        Patch(color=ACC3, label=f"Large cut ≥50bps (mean={np.mean(cars_large)*100:.2f}%)"),
        Patch(color=ACC4, label=f"Small cut <50bps (mean={np.mean(cars_small)*100:.2f}%)"),
        Line2D([0],[0], color=ACC2, ls="--", label=f"Overall mean={np.mean(cars_cut)*100:.2f}%"),
    ], facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax2.set_ylabel("CAR (%)", color=FG)
ax2.set_title("Cumulative Abnormal Returns per Fed Rate Cut  [week −1, 0, +1]", color=FG, fontsize=13)
ax2.grid(True, axis="y")

# ── Plot 3: Return distribution + fits ────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
x_vals = np.linspace(r.min(), r.max(), 400)
ax3.hist(r*100, bins=80, density=True, color=ACC1, alpha=0.5, edgecolor="none", label="Actual")
norm_pdf = stats.norm.pdf(x_vals, mean_r, std_r)
ax3.plot(x_vals*100, norm_pdf/100, color=ACC4, lw=1.8, label="Normal fit")
t_pdf = stats.t.pdf(x_vals, df_t, loc_t, scale_t)
ax3.plot(x_vals*100, t_pdf/100, color=ACC2, lw=1.8, label=f"t-dist (ν={df_t:.1f})")
ax3.set_xlim(-15, 15)
ax3.set_xlabel("Weekly Log Return (%)")
ax3.set_ylabel("Density")
ax3.set_title("Return Distribution", color=FG, fontsize=12)
ax3.legend(facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax3.grid(True)
box_txt = (f"μ={mean_r*100:.3f}%\nσ={std_r*100:.3f}%\n"
           f"Skew={skew_r:.3f}\nKurt={kurt_r:.1f}")
ax3.text(0.97, 0.97, box_txt, transform=ax3.transAxes, va="top", ha="right",
         fontsize=8.5, color=FG, bbox=dict(fc=GREY, ec="none", pad=4))

# ── Plot 4: Q-Q plot ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
ax4.scatter(osm, osr * std_r + mean_r, color=ACC1, alpha=0.3, s=4, label="Empirical")
lx = np.array([osm[0], osm[-1]])
ax4.plot(lx, slope*lx + intercept, color=ACC4, lw=2, label="Normal line")
ax4.set_xlabel("Theoretical Quantiles"); ax4.set_ylabel("Sample Quantiles")
ax4.set_title("Q-Q Plot (Normal)", color=FG, fontsize=12)
ax4.legend(facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax4.grid(True)

# ── Plot 5: Large vs Small cuts box ──────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
if len(cars_large) and len(cars_small):
    bp = ax5.boxplot([cars_large*100, cars_small*100],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color=FG, lw=2))
    bp["boxes"][0].set_facecolor(ACC3); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(ACC4); bp["boxes"][1].set_alpha(0.7)
    for w in bp["whiskers"]+bp["caps"]+bp["fliers"]:
        w.set(color=FG, lw=1, markersize=4)
    ax5.axhline(0, color=FG, lw=0.8, ls="--")
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels([f"LARGE ≥50bps\n(N={len(cars_large)})",
                         f"SMALL <50bps\n(N={len(cars_small)})"], color=FG)
    if not np.isnan(t2):
        ax5.text(0.5, 0.03, f"Welch t={t2:.3f}, p={p2:.3f}", transform=ax5.transAxes,
                 ha="center", fontsize=9, color=ACC4, bbox=dict(fc=GREY, ec="none", pad=3))
ax5.set_ylabel("CAR (%)")
ax5.set_title("CAR: Large vs Small Fed Cuts", color=FG, fontsize=12)
ax5.grid(True, axis="y")

# ── Plot 6: Era returns ──────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
if era_data:
    e_names = [d["label"].strip() for d in era_data]
    e_rets  = [d["ann_ret"]*100 for d in era_data]
    e_vols  = [d["ann_vol"]*100 for d in era_data]
    e_cols  = [ACC2 if v > 30 else ACC3 if v < 20 else ACC4 for v in e_vols]
    bars6 = ax6.bar(range(len(e_names)), e_rets, color=e_cols, alpha=0.8)
    ax6.plot(range(len(e_names)), e_vols, color=ACC1, marker="o", lw=2, ms=6, label="Volatility")
    for i, (bar, ret) in enumerate(zip(bars6, e_rets)):
        ax6.text(i, ret + (1 if ret >= 0 else -2), f"{ret:.1f}%", ha="center", fontsize=7.5, color=FG)
    ax6.set_xticks(range(len(e_names)))
    ax6.set_xticklabels(e_names, rotation=30, ha="right", fontsize=7)
    ax6.legend(facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax6.set_ylabel("Return / Volatility (%)")
ax6.set_title("Era-Wise Returns (bars) & Volatility (line)", color=FG, fontsize=12)
ax6.grid(True, axis="y")

out_path = os.path.join(OUT_DIR, "nifty_weekly_fed_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"\n  Saved: {out_path}")
plt.close()

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
