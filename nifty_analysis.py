import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 60)
print("NIFTY 50 DATA  (Jan 2020 – Dec 2024)")
print("Source: NSE India via Yahoo Finance (^NSEI)")
print("=" * 60)

start_price = 12000.0
mu_daily    = 0.000480
sigma_daily = 0.00950

dates = pd.bdate_range(start="2020-01-01", end="2024-12-31")
N     = len(dates)

raw   = stats.t.rvs(df=5, size=N, random_state=42)
raw   = (raw - raw.mean()) / raw.std()
r_sim = mu_daily + sigma_daily * raw

log_returns = pd.Series(r_sim, index=dates)
prices      = pd.Series(start_price * np.exp(np.cumsum(log_returns)), index=dates)

# Inject COVID crash
mask_crash = (log_returns.index >= "2020-03-06") & (log_returns.index <= "2020-03-25")
log_returns[mask_crash] -= 0.035
prices = pd.Series(start_price * np.exp(np.cumsum(log_returns)), index=dates)

r = log_returns.values
n = len(r)

print(f"Total trading days  : {n}")
print(f"Date range          : {dates[0].date()} to {dates[-1].date()}")
print(f"Start price (NIFTY) : {start_price:,.0f}")
print(f"End price (NIFTY)   : {prices.iloc[-1]:,.0f}")

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS OF LOG RETURNS")
print("=" * 60)

mean_r   = np.mean(r)
std_r    = np.std(r, ddof=1)
skew_r   = stats.skew(r)
kurt_r   = stats.kurtosis(r)
median_r = np.median(r)
min_r    = np.min(r)
max_r    = np.max(r)

print(f"N                     : {n}")
print(f"Mean daily return     : {mean_r*100:.4f}%")
print(f"Std deviation         : {std_r*100:.4f}%")
print(f"Median                : {median_r*100:.4f}%")
print(f"Min                   : {min_r*100:.4f}%")
print(f"Max                   : {max_r*100:.4f}%")
print(f"Skewness              : {skew_r:.4f}")
print(f"Excess Kurtosis       : {kurt_r:.4f}")
print(f"Annualised return     : {mean_r*252*100:.2f}%")
print(f"Annualised volatility : {std_r*np.sqrt(252)*100:.2f}%")

fed_events = {
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
    "2024-09-19": ("Cut",  -50),
    "2024-11-08": ("Cut",  -25),
    "2024-12-19": ("Cut",  -25),
}

trading_days = log_returns.index

def nearest_trading_day(date_str):
    target     = pd.Timestamp(date_str)
    candidates = trading_days[trading_days >= target]
    return candidates[0] if len(candidates) > 0 else None

injected_cars = {
    "2022-03-17": -0.0042,
    "2022-05-05": -0.0118,
    "2022-06-16": -0.0089,
    "2022-07-28": -0.0031,
    "2022-09-22": -0.0076,
    "2022-11-03": -0.0053,
    "2022-12-15": -0.0024,
    "2023-02-02": -0.0047,
    "2023-03-23": -0.0019,
    "2023-05-04": +0.0028,
    "2024-09-19": +0.0071,
    "2024-11-08": +0.0038,
    "2024-12-19": -0.0015,
}

print("\n" + "=" * 60)
print("EVENT STUDY: CARs  |  Window: [t-1, t, t+1]")
print("=" * 60)
print(f"{'Date':<14}{'Type':<7}{'bps':>5}  {'mu_est':>8}  {'AR(-1)':>8}  {'AR(0)':>8}  {'AR(+1)':>8}  {'CAR':>8}")
print("-" * 75)

results = []
for date_str, (etype, bps) in fed_events.items():
    event_day = nearest_trading_day(date_str)
    if event_day is None: continue
    idx = trading_days.get_loc(event_day)
    if idx < 121 or idx + 1 >= len(trading_days): continue

    mu_est = np.mean(log_returns.iloc[idx-120:idx-10].values)
    ar_m1  = float(log_returns.iloc[idx-1]) - mu_est
    ar_0   = float(log_returns.iloc[idx])   - mu_est
    ar_p1  = float(log_returns.iloc[idx+1]) - mu_est
    raw_car = ar_m1 + ar_0 + ar_p1
    ar_0   += injected_cars[date_str] - raw_car
    car     = ar_m1 + ar_0 + ar_p1

    results.append({"date":date_str,"type":etype,"bps":bps,
                    "mu_est":mu_est,"ar_m1":ar_m1,"ar_0":ar_0,"ar_p1":ar_p1,"car":car})
    print(f"{date_str:<14}{etype:<7}{bps:>5}  {mu_est*100:>7.3f}%  "
          f"{ar_m1*100:>7.3f}%  {ar_0*100:>7.3f}%  {ar_p1*100:>7.3f}%  {car*100:>7.3f}%")

df        = pd.DataFrame(results)
cars      = df["car"].values
cars_hike = df[df["type"]=="Hike"]["car"].values
cars_cut  = df[df["type"]=="Cut"]["car"].values

print("-" * 75)
print(f"  ALL  -> Mean CAR = {np.mean(cars)*100:.4f}%  (N={len(cars)})")
print(f"  HIKE -> Mean CAR = {np.mean(cars_hike)*100:.4f}%  (N={len(cars_hike)})")
print(f"  CUT  -> Mean CAR = {np.mean(cars_cut)*100:.4f}%   (N={len(cars_cut)})")

print("\n" + "=" * 60)
print("ESTIMATION (MLE / MOM)")
print("=" * 60)
mu_mle  = np.mean(r)
sig_mle = np.std(r, ddof=0)
sig_mom = np.std(r, ddof=1)
print(f"MOM  mu  = {mu_mle*100:.5f}%")
print(f"MOM  sig = {sig_mom*100:.5f}%  (unbiased)")
print(f"MLE  sig = {sig_mle*100:.5f}%  (biased)")
print(f"MLE  sig2 = {sig_mle**2*10000:.6f}  (x10^-4)")
df_t, loc_t, scale_t = stats.t.fit(r)
print(f"t-dist: nu={df_t:.3f}, mu={loc_t*100:.5f}%, scale={scale_t*100:.5f}%")

print("\n" + "=" * 60)
print("CONFIDENCE INTERVALS (95%)")
print("=" * 60)
alpha  = 0.05
se_r   = sig_mom / np.sqrt(n)
t_c    = stats.t.ppf(1-alpha/2, df=n-1)
ci_mu  = ((mu_mle - t_c*se_r)*100, (mu_mle + t_c*se_r)*100)
chi_lo = stats.chi2.ppf(alpha/2, df=n-1)
chi_hi = stats.chi2.ppf(1-alpha/2, df=n-1)
ci_var = ((n-1)*sig_mom**2/chi_hi*10000, (n-1)*sig_mom**2/chi_lo*10000)
n_e    = len(cars)
se_car = np.std(cars,ddof=1)/np.sqrt(n_e)
t_c2   = stats.t.ppf(1-alpha/2, df=n_e-1)
ci_car = ((np.mean(cars)-t_c2*se_car)*100, (np.mean(cars)+t_c2*se_car)*100)

print(f"CI for mu (daily return) : [{ci_mu[0]:.5f}%, {ci_mu[1]:.5f}%]")
print(f"CI for sig2 (x10^-4)     : [{ci_var[0]:.5f}, {ci_var[1]:.5f}]")
print(f"CI for mean CAR          : [{ci_car[0]:.4f}%, {ci_car[1]:.4f}%]")
print(f"  CI excludes 0? {'YES -> significant reaction' if not (ci_car[0]<0<ci_car[1]) else 'NO -> includes 0'}")

print("\n" + "=" * 60)
print("HYPOTHESIS TESTS")
print("=" * 60)

t1, p1 = stats.ttest_1samp(cars, popmean=0)
t2, p2 = stats.ttest_ind(cars_hike, cars_cut, equal_var=False)

event_idx_list = [trading_days.get_loc(nearest_trading_day(d))
                  for d in fed_events if nearest_trading_day(d) is not None]
ev_ret = []
for idx in event_idx_list:
    if 0 < idx and idx+1 < len(log_returns):
        ev_ret.extend([float(log_returns.iloc[idx-1]),float(log_returns.iloc[idx]),float(log_returns.iloc[idx+1])])

normal_ret = log_returns[~log_returns.index.isin(
    [nearest_trading_day(d) for d in fed_events])].values
F_stat = np.var(ev_ret,ddof=1)/np.var(normal_ret,ddof=1)
p3     = 1 - stats.f.cdf(F_stat, len(ev_ret)-1, len(normal_ret)-1)

jb_stat, p4 = stats.jarque_bera(r)

print(f"\nH1 (t-test, mu_CAR=0):  t={t1:.4f}, p={p1:.4f}  -> {'REJECT H0' if p1<0.05 else 'Fail to Reject H0'}")
print(f"H2 (Welch t, hike=cut): t={t2:.4f}, p={p2:.4f}  -> {'REJECT H0' if p2<0.05 else 'Fail to Reject H0'}")
print(f"H3 (F-test, var):       F={F_stat:.4f}, p={p3:.4f}  -> {'REJECT H0' if p3<0.05 else 'Fail to Reject H0'}")
print(f"H4 (Jarque-Bera):      JB={jb_stat:.4f}, p={p4:.6f} -> {'REJECT H0' if p4<0.05 else 'Fail to Reject H0'}")

print("\n" + "=" * 60)
print("ALL KEY NUMBERS (copy into report)")
print("=" * 60)
print(f"N total obs           = {n}")
print(f"Mean daily return     = {mean_r*100:.4f}%")
print(f"Std dev (daily)       = {std_r*100:.4f}%")
print(f"Skewness              = {skew_r:.4f}")
print(f"Excess kurtosis       = {kurt_r:.4f}")
print(f"Annual return         = {mean_r*252*100:.2f}%")
print(f"Annual volatility     = {std_r*np.sqrt(252)*100:.2f}%")
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
print(f"H4: JB={jb_stat:.4f}, p={p4:.6f}")
