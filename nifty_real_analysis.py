import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DOWNLOAD REAL NIFTY 50 DATA (2000-2026)
# ============================================================
print("=" * 65)
print("DOWNLOADING REAL NIFTY 50 DATA  (Jan 2000 - Apr 2026)")
print("Source: Yahoo Finance  (^NSEI)")
print("=" * 65)

ticker = yf.Ticker("^NSEI")
raw_data = ticker.history(start="2000-01-01", end="2026-04-14", auto_adjust=True)

if raw_data.empty:
    raise RuntimeError("Download failed – check internet connection or ticker symbol.")

prices_raw = raw_data["Close"].dropna()
prices_raw.index = prices_raw.index.tz_localize(None)  # strip timezone

print(f"Downloaded {len(prices_raw)} trading days")
print(f"Date range  : {prices_raw.index[0].date()} to {prices_raw.index[-1].date()}")
print(f"Start price : {prices_raw.iloc[0]:,.2f}")
print(f"End price   : {prices_raw.iloc[-1]:,.2f}")
print(f"Total gain  : {(prices_raw.iloc[-1]/prices_raw.iloc[0]-1)*100:.1f}%")

# ============================================================
# 2. LOG RETURNS
# ============================================================
log_returns = np.log(prices_raw / prices_raw.shift(1)).dropna()
trading_days = log_returns.index
r  = log_returns.values
n  = len(r)

# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 65)
print("DESCRIPTIVE STATISTICS OF LOG RETURNS (Full Sample)")
print("=" * 65)

mean_r   = np.mean(r)
std_r    = np.std(r, ddof=1)
skew_r   = stats.skew(r)
kurt_r   = stats.kurtosis(r)
median_r = np.median(r)
min_r    = np.min(r)
max_r    = np.max(r)

print(f"N (trading days)      : {n}")
print(f"Mean daily return     : {mean_r*100:.4f}%")
print(f"Std deviation         : {std_r*100:.4f}%")
print(f"Median                : {median_r*100:.4f}%")
print(f"Min (worst day)       : {min_r*100:.4f}%  ({log_returns.idxmin().date()})")
print(f"Max (best day)        : {max_r*100:.4f}%  ({log_returns.idxmax().date()})")
print(f"Skewness              : {skew_r:.4f}")
print(f"Excess Kurtosis       : {kurt_r:.4f}")
print(f"Annualised return     : {mean_r*252*100:.2f}%")
print(f"Annualised volatility : {std_r*np.sqrt(252)*100:.2f}%")
print(f"Sharpe (rf=0)         : {(mean_r*252)/(std_r*np.sqrt(252)):.4f}")

# ============================================================
# 4. FULL HISTORICAL FED RATE EVENTS (2000-2026)
#    Format: date -> (type, bps_change)
# ============================================================
fed_events = {
    # --- Dot-com bust cuts ---
    "2001-01-03": ("Cut",  -50),
    "2001-01-31": ("Cut",  -50),
    "2001-03-20": ("Cut",  -50),
    "2001-04-18": ("Cut",  -50),
    "2001-05-15": ("Cut",  -50),
    "2001-06-27": ("Cut",  -25),
    "2001-08-21": ("Cut",  -25),
    "2001-09-17": ("Cut",  -50),  # post 9/11 emergency
    "2001-10-02": ("Cut",  -50),
    "2001-11-06": ("Cut",  -50),
    "2001-12-11": ("Cut",  -25),
    "2002-11-06": ("Cut",  -50),
    "2003-06-25": ("Cut",  -25),
    # --- Recovery hikes ---
    "2004-06-30": ("Hike", +25),
    "2004-08-10": ("Hike", +25),
    "2004-09-21": ("Hike", +25),
    "2004-11-10": ("Hike", +25),
    "2004-12-14": ("Hike", +25),
    "2005-02-02": ("Hike", +25),
    "2005-03-22": ("Hike", +25),
    "2005-05-03": ("Hike", +25),
    "2005-06-30": ("Hike", +25),
    "2005-08-09": ("Hike", +25),
    "2005-09-20": ("Hike", +25),
    "2005-11-01": ("Hike", +25),
    "2005-12-13": ("Hike", +25),
    "2006-01-31": ("Hike", +25),
    "2006-03-28": ("Hike", +25),
    "2006-05-10": ("Hike", +25),
    "2006-06-29": ("Hike", +25),
    # --- GFC cuts ---
    "2007-09-18": ("Cut",  -50),
    "2007-10-31": ("Cut",  -25),
    "2007-12-11": ("Cut",  -25),
    "2008-01-22": ("Cut",  -75),  # emergency
    "2008-01-30": ("Cut",  -50),
    "2008-03-18": ("Cut",  -75),
    "2008-04-30": ("Cut",  -25),
    "2008-10-08": ("Cut",  -50),  # coordinated global cut
    "2008-10-29": ("Cut",  -50),
    "2008-12-16": ("Cut",  -75),
    # --- Liftoff & taper ---
    "2015-12-16": ("Hike", +25),
    "2016-12-14": ("Hike", +25),
    "2017-03-15": ("Hike", +25),
    "2017-06-14": ("Hike", +25),
    "2017-12-13": ("Hike", +25),
    "2018-03-21": ("Hike", +25),
    "2018-06-13": ("Hike", +25),
    "2018-09-26": ("Hike", +25),
    "2018-12-19": ("Hike", +25),
    # --- 2019 insurance cuts ---
    "2019-07-31": ("Cut",  -25),
    "2019-09-18": ("Cut",  -25),
    "2019-10-30": ("Cut",  -25),
    # --- COVID emergency cuts ---
    "2020-03-03": ("Cut",  -50),
    "2020-03-15": ("Cut", -100),
    # --- Post-COVID hikes (2022-2023) ---
    "2022-03-17": ("Hike", +25),
    "2022-05-05": ("Hike", +50),
    "2022-06-16": ("Hike", +75),
    "2022-07-28": ("Hike", +75),
    "2022-09-22": ("Hike", +75),
    "2022-11-03": ("Hike", +75),
    "2022-12-15": ("Hike", +50),
    "2023-02-02": ("Hike", +25),
    "2023-03-23": ("Hike", +25),
    "2023-05-04": ("Hike", +25),
    # --- 2024 cuts ---
    "2024-09-19": ("Cut",  -50),
    "2024-11-07": ("Cut",  -25),
    "2024-12-18": ("Cut",  -25),
}

def nearest_trading_day(date_str):
    target     = pd.Timestamp(date_str)
    candidates = trading_days[trading_days >= target]
    if len(candidates) == 0:
        return None
    # Skip if > 5 business days away (weekend / holiday gap)
    if (candidates[0] - target).days > 7:
        return None
    return candidates[0]

# ============================================================
# 5. EVENT STUDY  –  CAR [-1, 0, +1]
# ============================================================
print("\n" + "=" * 65)
print("EVENT STUDY: Cumulative Abnormal Returns")
print("Estimation window: [t-120, t-10]  |  Event window: [t-1, t+1]")
print("=" * 65)
print(f"{'Date':<13}{'Type':<6}{'bps':>5}  {'mu_est':>8}  {'AR(-1)':>8}  {'AR(0)':>8}  {'AR(+1)':>8}  {'CAR':>8}")
print("-" * 68)

results = []
skipped = []

for date_str, (etype, bps) in fed_events.items():
    event_day = nearest_trading_day(date_str)
    if event_day is None:
        skipped.append(date_str)
        continue
    idx = trading_days.get_loc(event_day)
    if idx < 121 or idx + 1 >= len(trading_days):
        skipped.append(date_str)
        continue

    mu_est = np.mean(log_returns.iloc[idx-120:idx-10].values)
    ar_m1  = float(log_returns.iloc[idx-1]) - mu_est
    ar_0   = float(log_returns.iloc[idx])   - mu_est
    ar_p1  = float(log_returns.iloc[idx+1]) - mu_est
    car    = ar_m1 + ar_0 + ar_p1

    results.append({
        "date": date_str, "type": etype, "bps": bps,
        "mu_est": mu_est, "ar_m1": ar_m1, "ar_0": ar_0, "ar_p1": ar_p1, "car": car
    })
    print(f"{date_str:<13}{etype:<6}{bps:>5}  {mu_est*100:>7.3f}%  "
          f"{ar_m1*100:>7.3f}%  {ar_0*100:>7.3f}%  {ar_p1*100:>7.3f}%  {car*100:>7.3f}%")

if skipped:
    print(f"\n  [Skipped {len(skipped)} events – outside data range or >7d gap]")

df        = pd.DataFrame(results)
cars      = df["car"].values
cars_hike = df[df["type"] == "Hike"]["car"].values
cars_cut  = df[df["type"] == "Cut"]["car"].values

print("-" * 68)
print(f"  ALL  -> Mean CAR = {np.mean(cars)*100:.4f}%  (N={len(cars)})")
print(f"  HIKE -> Mean CAR = {np.mean(cars_hike)*100:.4f}%  (N={len(cars_hike)})")
print(f"  CUT  -> Mean CAR = {np.mean(cars_cut)*100:.4f}%   (N={len(cars_cut)})")

# ============================================================
# 6. MLE / MOM ESTIMATION
# ============================================================
print("\n" + "=" * 65)
print("PARAMETER ESTIMATION (MLE / MOM)")
print("=" * 65)
mu_mle  = np.mean(r)
sig_mle = np.std(r, ddof=0)
sig_mom = np.std(r, ddof=1)
print(f"MOM  mu  = {mu_mle*100:.5f}%  (same as MLE for normal)")
print(f"MOM  sig = {sig_mom*100:.5f}%  (unbiased, ddof=1)")
print(f"MLE  sig = {sig_mle*100:.5f}%  (biased,   ddof=0)")
print(f"MLE  sig2 = {sig_mle**2 * 10000:.6f}  (x10^-4)")
df_t, loc_t, scale_t = stats.t.fit(r)
print(f"t-dist fit: nu={df_t:.3f}, loc={loc_t*100:.5f}%, scale={scale_t*100:.5f}%")

# ============================================================
# 7. CONFIDENCE INTERVALS (95%)
# ============================================================
print("\n" + "=" * 65)
print("CONFIDENCE INTERVALS (95%,  alpha=0.05)")
print("=" * 65)
alpha  = 0.05
se_r   = sig_mom / np.sqrt(n)
t_c    = stats.t.ppf(1 - alpha/2, df=n-1)
ci_mu  = ((mu_mle - t_c*se_r)*100, (mu_mle + t_c*se_r)*100)

chi_lo = stats.chi2.ppf(alpha/2, df=n-1)
chi_hi = stats.chi2.ppf(1-alpha/2, df=n-1)
ci_var = ((n-1)*sig_mom**2/chi_hi*10000, (n-1)*sig_mom**2/chi_lo*10000)

n_e    = len(cars)
se_car = np.std(cars, ddof=1) / np.sqrt(n_e)
t_c2   = stats.t.ppf(1-alpha/2, df=n_e-1)
ci_car = ((np.mean(cars) - t_c2*se_car)*100, (np.mean(cars) + t_c2*se_car)*100)

print(f"CI for mu (daily return) : [{ci_mu[0]:.5f}%,  {ci_mu[1]:.5f}%]")
print(f"CI for sig2 (x10^-4)     : [{ci_var[0]:.5f},  {ci_var[1]:.5f}]")
print(f"CI for mean CAR          : [{ci_car[0]:.4f}%,  {ci_car[1]:.4f}%]")
if not (ci_car[0] < 0 < ci_car[1]):
    print("  >> CI excludes 0 -> STATISTICALLY SIGNIFICANT market reaction")
else:
    print("  >> CI includes 0 -> reaction not statistically significant at 95%")

# ============================================================
# 8. HYPOTHESIS TESTS
# ============================================================
print("\n" + "=" * 65)
print("HYPOTHESIS TESTS")
print("=" * 65)

# H1: Mean CAR = 0  (one-sample t-test)
t1, p1 = stats.ttest_1samp(cars, popmean=0)
# H2: Hike CAR = Cut CAR  (Welch t)
t2, p2 = stats.ttest_ind(cars_hike, cars_cut, equal_var=False)
# H3: Variance during events vs non-events  (F-test)
event_idx_list = [trading_days.get_loc(nearest_trading_day(d))
                  for d in fed_events
                  if nearest_trading_day(d) is not None and
                     120 < trading_days.get_loc(nearest_trading_day(d)) < len(trading_days)-1]
ev_ret = []
for idx in event_idx_list:
    ev_ret.extend([float(log_returns.iloc[idx-1]),
                   float(log_returns.iloc[idx]),
                   float(log_returns.iloc[idx+1])])
normal_ret = log_returns[~log_returns.index.isin(
    [nearest_trading_day(d) for d in fed_events if nearest_trading_day(d) is not None])].values

F_stat = np.var(ev_ret, ddof=1) / np.var(normal_ret, ddof=1)
p3     = 1 - stats.f.cdf(F_stat, len(ev_ret)-1, len(normal_ret)-1)
# H4: Normality (Jarque-Bera)
jb_stat, p4 = stats.jarque_bera(r)

print(f"\nH1  Mean CAR = 0       (1-sample t): t={t1:>8.4f}, p={p1:.4f}  -> {'** REJECT H0 **' if p1<0.05 else 'Fail to Reject H0'}")
print(f"H2  Hike-CAR = Cut-CAR (Welch t)  : t={t2:>8.4f}, p={p2:.4f}  -> {'** REJECT H0 **' if p2<0.05 else 'Fail to Reject H0'}")
print(f"H3  Var(event)=Var(non) (F-test)  : F={F_stat:>8.4f}, p={p3:.4f}  -> {'** REJECT H0 **' if p3<0.05 else 'Fail to Reject H0'}")
print(f"H4  Normality           (JB-test) : JB={jb_stat:>7.2f}, p={p4:.6f} -> {'** REJECT H0 **' if p4<0.05 else 'Fail to Reject H0'}")

# ============================================================
# 9. ERA-WISE BREAKDOWN
# ============================================================
print("\n" + "=" * 65)
print("ERA-WISE NIFTY PERFORMANCE")
print("=" * 65)
eras = {
    "Dot-com bust   (2000-2002)": ("2000-01-01", "2002-12-31"),
    "Recovery       (2003-2006)": ("2003-01-01", "2006-12-31"),
    "GFC            (2007-2009)": ("2007-01-01", "2009-12-31"),
    "Post-GFC rally (2010-2019)": ("2010-01-01", "2019-12-31"),
    "COVID crash    (2020)     ": ("2020-01-01", "2020-12-31"),
    "Post-COVID     (2021-2021)": ("2021-01-01", "2021-12-31"),
    "Hike cycle     (2022-2023)": ("2022-01-01", "2023-12-31"),
    "Cut cycle      (2024-2026)": ("2024-01-01", "2026-04-14"),
}
print(f"{'Era':<32}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'Sharpe':>7}  {'Skew':>7}")
print("-" * 70)
for label, (s, e) in eras.items():
    era_r = log_returns.loc[s:e].values
    if len(era_r) < 20:
        continue
    ann_ret = np.mean(era_r) * 252
    ann_vol = np.std(era_r, ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    sk      = stats.skew(era_r)
    print(f"{label:<32}  {ann_ret*100:>7.2f}%  {ann_vol*100:>7.2f}%  {sharpe:>7.3f}  {sk:>7.4f}")

# ============================================================
# 10. ALL KEY NUMBERS (copy into report)
# ============================================================
print("\n" + "=" * 65)
print("ALL KEY NUMBERS (copy into report)")
print("=" * 65)
print(f"N total obs           = {n}")
print(f"Mean daily return     = {mean_r*100:.4f}%")
print(f"Std dev (daily)       = {std_r*100:.4f}%")
print(f"Skewness              = {skew_r:.4f}")
print(f"Excess kurtosis       = {kurt_r:.4f}")
print(f"Annual return         = {mean_r*252*100:.2f}%")
print(f"Annual volatility     = {std_r*np.sqrt(252)*100:.2f}%")
print(f"Sharpe ratio          = {(mean_r*252)/(std_r*np.sqrt(252)):.4f}")
print(f"MLE mu                = {mu_mle*100:.5f}%")
print(f"MLE sig               = {sig_mle*100:.5f}%")
print(f"t-dist df (nu)        = {df_t:.3f}")
print(f"95% CI mu             = [{ci_mu[0]:.5f}%, {ci_mu[1]:.5f}%]")
print(f"95% CI sig2 (x10^-4)  = [{ci_var[0]:.5f}, {ci_var[1]:.5f}]")
print(f"95% CI CAR            = [{ci_car[0]:.4f}%, {ci_car[1]:.4f}%]")
print(f"Mean CAR (all)        = {np.mean(cars)*100:.4f}%")
print(f"Mean CAR (hike)       = {np.mean(cars_hike)*100:.4f}%")
print(f"Mean CAR (cut)        = {np.mean(cars_cut)*100:.4f}%")
print(f"H1: t={t1:.4f}, p={p1:.4f}")
print(f"H2: t={t2:.4f}, p={p2:.4f}")
print(f"H3: F={F_stat:.4f}, p={p3:.4f}")
print(f"H4: JB={jb_stat:.2f}, p={p4:.6f}")
print(f"\nN events analysed     = {len(cars)}")
print(f"  Hike events         = {len(cars_hike)}")
print(f"  Cut  events         = {len(cars_cut)}")
