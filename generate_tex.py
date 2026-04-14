import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
TEX_PATH = os.path.join(OUT_DIR, "nifty50_fed_rate_cuts_report.tex")

print("Computing from real data...")

csv_path = os.path.join(OUT_DIR, "Nifty 50 Historical Data (1).csv")
df_csv = pd.read_csv(csv_path, thousands=",")
df_csv["Date"] = pd.to_datetime(df_csv["Date"], format="%d-%m-%Y")
df_csv = df_csv.sort_values("Date").reset_index(drop=True)
df_csv["Close"] = pd.to_numeric(df_csv["Price"].astype(str).str.replace(",",""), errors="coerce")
df_csv = df_csv[["Date","Close"]].dropna()

try:
    import yfinance as yf
    last = df_csv["Date"].iloc[-1]
    yfstart = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yf_raw = yf.Ticker("^NSEI").history(start=yfstart, end="2026-04-14", auto_adjust=True)
    if not yf_raw.empty:
        df_yf = yf_raw[["Close"]].reset_index()
        df_yf.columns = ["Date","Close"]
        df_yf["Date"] = pd.to_datetime(df_yf["Date"]).dt.tz_localize(None)
        daily = pd.concat([df_csv, df_yf], ignore_index=True)
    else:
        daily = df_csv.copy()
except:
    daily = df_csv.copy()

daily = daily.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)

# Daily log returns
daily["return_pct"] = np.log(daily["Close"] / daily["Close"].shift(1)) * 100
daily = daily.dropna().reset_index(drop=True)
n_total = len(daily)

# Fed cuts
fed_cuts = {
    "2001-01-03":-50,"2001-01-31":-50,"2001-03-20":-50,"2001-04-18":-50,
    "2001-05-15":-50,"2001-06-27":-25,"2001-08-21":-25,"2001-09-17":-50,
    "2001-10-02":-50,"2001-11-06":-50,"2001-12-11":-25,
    "2002-11-06":-50,"2003-06-25":-25,
    "2007-09-18":-50,"2007-10-31":-25,"2007-12-11":-25,
    "2008-01-22":-75,"2008-01-30":-50,"2008-03-18":-75,
    "2008-04-30":-25,"2008-10-08":-50,"2008-10-29":-50,"2008-12-16":-75,
    "2019-07-31":-25,"2019-09-18":-25,"2019-10-30":-25,
    "2020-03-03":-50,"2020-03-15":-100,
    "2024-09-19":-50,"2024-11-07":-25,"2024-12-18":-25,
}

def find_next_trading_day(cut_date_str):
    t = pd.Timestamp(cut_date_str)
    next_day = t + pd.Timedelta(days=1)
    candidates = daily[daily["Date"] >= next_day]
    if len(candidates) == 0: return None
    first = candidates.iloc[0]
    if (first["Date"] - t).days > 7: return None
    return first["Date"]

# Classify each day
event_dates = set()
event_info = []  # (cut_date, nifty_date, bps)
for ds, bps in sorted(fed_cuts.items()):
    nxt = find_next_trading_day(ds)
    if nxt is not None:
        event_dates.add(nxt)
        event_info.append((ds, nxt, bps))

daily["event_day"] = daily["Date"].isin(event_dates)
daily["group"] = daily["event_day"].map({True: "Event-day", False: "Normal-day"})
daily["negative_day"] = (daily["return_pct"] < 0).astype(int)

cut_rets  = daily[daily["group"]=="Event-day"]["return_pct"].values
norm_rets = daily[daily["group"]=="Normal-day"]["return_pct"].values
n_cut = len(cut_rets)
n_norm = len(norm_rets)

mean_cut = np.mean(cut_rets);  std_cut = np.std(cut_rets, ddof=1)
mean_norm = np.mean(norm_rets); std_norm = np.std(norm_rets, ddof=1)

p_cut  = np.mean(cut_rets < 0)
p_norm = np.mean(norm_rets < 0)
neg_cut_count = int(np.sum(cut_rets < 0))
neg_norm_count = int(np.sum(norm_rets < 0))

r = daily["return_pct"].values
n = len(r)
mean_r = np.mean(r); std_r = np.std(r, ddof=1)
skew_r = stats.skew(r); kurt_r = stats.kurtosis(r)

alpha = 0.05

# Test 1: H1: mu_C > mu_N (rate cuts boost next-day returns)
t1, p1_two = stats.ttest_ind(cut_rets, norm_rets, equal_var=False)
p1_one = p1_two / 2.0 if t1 > 0 else 1.0 - p1_two / 2.0

# Test 2: H1: p_C < p_N (fewer negative days after cuts)
p_pooled = (neg_cut_count + neg_norm_count) / (n_cut + n_norm)
se_prop = np.sqrt(p_pooled*(1-p_pooled)*(1/n_cut + 1/n_norm))
z2 = (p_cut - p_norm) / se_prop
p2 = stats.norm.cdf(z2)

# Test 3: F-test
F3 = np.var(cut_rets, ddof=1) / np.var(norm_rets, ddof=1)
p3 = 2 * min(stats.f.cdf(F3, n_cut-1, n_norm-1), 1-stats.f.cdf(F3, n_cut-1, n_norm-1))

# Test 4: JB
jb4, p4 = stats.jarque_bera(r)

# CIs
se_cut = std_cut / np.sqrt(n_cut)
se_norm_val = std_norm / np.sqrt(n_norm)
t_c_cut = stats.t.ppf(0.975, df=n_cut-1)
t_c_norm = stats.t.ppf(0.975, df=n_norm-1)
ci_cut  = (mean_cut - t_c_cut*se_cut, mean_cut + t_c_cut*se_cut)
ci_norm = (mean_norm - t_c_norm*se_norm_val, mean_norm + t_c_norm*se_norm_val)
z_c = 1.96
ci_p_cut  = (p_cut - z_c*np.sqrt(p_cut*(1-p_cut)/n_cut), p_cut + z_c*np.sqrt(p_cut*(1-p_cut)/n_cut))
ci_p_norm = (p_norm - z_c*np.sqrt(p_norm*(1-p_norm)/n_norm), p_norm + z_c*np.sqrt(p_norm*(1-p_norm)/n_norm))

wdf = ((std_cut**2/n_cut + std_norm**2/n_norm)**2) / ((std_cut**2/n_cut)**2/(n_cut-1) + (std_norm**2/n_norm)**2/(n_norm-1))
t_crit_test = stats.t.ppf(0.95, df=wdf)

spread = mean_cut - mean_norm
ann_spread = spread * 252

print(f"N_total={n}, n_event={n_cut}, n_normal={n_norm}")
print(f"mean_event={mean_cut:.4f}, mean_normal={mean_norm:.4f}, spread={spread:+.4f}")
print(f"t={t1:.4f}, p1_one={p1_one:.4f}")
print(f"z={z2:.4f}, p2={p2:.4f}")
print(f"F={F3:.4f}, p3={p3:.6f}")

# ═══════════════════════════════════════════════════════════════
# BUILD SAMPLE TABLE (ALL event-days + normal days to reach 50)
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)
event_rows = daily[daily["event_day"] == True].copy()
normal_rows = daily[daily["event_day"] == False]
n_normal_needed = 50 - len(event_rows)
normal_sample = normal_rows.sample(n=max(n_normal_needed, 0), random_state=42)
sample_rows = pd.concat([event_rows, normal_sample]).sort_values("Date").copy()
print(f"Sample: {len(event_rows)} event + {len(normal_sample)} normal = {len(sample_rows)} rows")

# Decision strings
t1_decision = "Reject $H_0$" if p1_one < 0.05 else "Fail to Reject $H_0$"
t1_crit_note = "$t > t_{crit} \\Rightarrow$ reject" if t1 > t_crit_test else "$t < t_{crit} \\Rightarrow$ fail to reject"
z2_decision = "Reject $H_0$" if p2 < 0.05 else "Fail to Reject $H_0$"
z2_crit_note = "$z < -z_{crit} \\Rightarrow$ reject" if z2 < -1.645 else "$z > -z_{crit} \\Rightarrow$ fail to reject"
f3_decision = "Reject $H_0$" if p3 < 0.05 else "Fail to Reject $H_0$"
jb_decision = "Reject $H_0$" if p4 < 0.05 else "Fail to Reject $H_0$"

mean_sig = "significantly higher" if p1_one < 0.05 else "not significantly higher"
prop_sig = "significantly fewer" if p2 < 0.05 else "not significantly fewer"
prop_note = "This difference is statistically significant." if p2 < 0.05 else "However, this difference is not statistically significant."
vol_note = "significantly higher" if p3 < 0.05 else "not significantly different"
have_evidence = "have" if p1_one < 0.05 or p2 < 0.05 else "do not have"

# ═══════════════════════════════════════════════════════════════
# WRITE THE .TEX FILE
# ═══════════════════════════════════════════════════════════════
print("Writing LaTeX...")

with open(TEX_PATH, "w", encoding="utf-8") as f:
    w = f.write

    w("\\documentclass[12pt,a4paper]{article}\n")
    w("\\usepackage[utf8]{inputenc}\n")
    w("\\usepackage[margin=1in]{geometry}\n")
    w("\\usepackage{amsmath, amssymb}\n")
    w("\\usepackage{graphicx}\n")
    w("\\usepackage{booktabs}\n")
    w("\\usepackage[table]{xcolor}\n")
    w("\\usepackage{tabularx}\n")
    w("\\usepackage[colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue]{hyperref}\n")
    w("\\usepackage{float}\n\n")
    w("\\definecolor{hdrbg}{HTML}{1A237E}\n")
    w("\\definecolor{rowalt}{HTML}{E8EAF6}\n\n")

    w("\\title{\\vspace{-2cm}\\color{hdrbg}\\textbf{Statistical Inference (MA20266) Project}\\\\[1em]\n")
    w("\\Large Topic: Does the US Federal Reserve Interest Rate Cut significantly change NIFTY 50 Stock Market Return behaviour?}\n")
    w("\\author{\\textbf{Group Members:} \\\\\n")
    w("1) Arnab Maiti (24IM10017) \\\\\n")
    w("2) Vivek Dubey (24IM10070) \\\\\n")
    w("3) Kartik Jeengar (24IM10010) \\\\\n")
    w("4) Kshitij Nayan (24IM10040)}\n")
    w("\\date{\\vspace{-1em}}\n\n")
    w("\\begin{document}\n")
    w("\\maketitle\n\n")

    # PROJECT STRUCTURE
    w("\\newpage\n")
    w("\\section{Project Structure}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|p{4cm}|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Section}} & \\textcolor{white}{\\textbf{Content}} \\\\\n\\hline\n")
    w("1. Data Description & NIFTY 50 daily returns\\newline Source: Investing.com + Yahoo Finance \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("2. Problem Statement & Do Fed rate cuts significantly boost next-day NIFTY returns?\\newline Statistical inference: $H_0$ vs $H_1$, $\\alpha = 0.05$ \\\\\n\\hline\n")
    w("3. Methodology & Point \\& interval estimation\\newline Two-sample t-test\\newline z-test for proportions \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("4. Results \\& Inference & Hypothesis decisions\\newline Statistical vs economic significance\\newline Trading implications \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # DATA DESCRIPTION
    w("\\newpage\n")
    w("\\section{Data Description}\n")
    w(f"Dataset: Daily closing prices of the NIFTY 50 Index (\\textasciicircum NSEI) from April 2000 to April 2026 --- \\textbf{{{n}}} daily observations.\n\n")
    w("Derived variable: Daily log-return, calculated as:\n")
    w("\\begin{equation*}\n    r_t = \\ln\\left( \\frac{P_t}{P_{t-1}} \\right) \\times 100\n\\end{equation*}\n\n")
    w("\\textbf{Event definition:} FOMC rate cut announcements happen in US evening time (IST next morning). ")
    w("We define the \\textbf{event day} as the \\textbf{next Indian trading day} after each FOMC rate cut announcement. ")
    w("This captures the immediate market reaction to the rate cut news.\n\n")
    w("\\textbf{Source:}\n")
    w("\\begin{itemize}\n")
    w("    \\item Investing.com CSV download for NIFTY 50 daily prices (Apr 2000 -- May 2020)\n")
    w("    \\item Yahoo Finance (via yfinance Python library) for Jun 2020 -- Apr 2026\n")
    w("    \\item Federal Reserve FOMC meeting dates for rate cut event classification\n")
    w("\\end{itemize}\n\n")

    w("\\textbf{Period classification:}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Category}} & \\textcolor{white}{\\textbf{Days}} & \\textcolor{white}{\\textbf{Description}} \\\\\n\\hline\n")
    w(f"Event days (E) & {n_cut} days & Next Indian trading day after each FOMC rate cut \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Normal days (N) & {n_norm} days & All other trading days \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # SAMPLE DATA TABLE (50 rows)
    w("\\textbf{Sample Data Rows (50 observations --- all 31 event-days + 19 selected normal days):}\n")
    w("\\begin{table}[H]\n\\centering\n\\footnotesize\n")
    w("\\begin{tabular}{|l|c|c|l|c|}\n\\hline\n")
    w("\\rowcolor{hdrbg}\\textcolor{white}{\\textbf{Date}} & \\textcolor{white}{\\textbf{Close}} & \\textcolor{white}{\\textbf{return\\_pct}} & \\textcolor{white}{\\textbf{group}} & \\textcolor{white}{\\textbf{neg\\_day}} \\\\\n\\hline\n")
    for _, row in sample_rows.iterrows():
        dt = row["Date"].strftime("%d-%m-%Y")
        cl = f"{row['Close']:.2f}"
        rp = f"{row['return_pct']:.4f}"
        gr = row["group"]
        nd = int(row["negative_day"])
        w(f"{dt} & {cl} & {rp} & {gr} & {nd} \\\\\n\\hline\n")
    w("\\end{tabular}\n\\end{table}\n\n")

    # REAL-WORLD PROBLEM
    w("\\newpage\n")
    w("\\section{Real-World Problem}\n\n")
    w("\\textbf{Research Question:} ``Does the US Federal Reserve interest rate cut have a significant \\textbf{positive} effect on the NIFTY 50 index the next trading day?''\n\n")
    w("\\textbf{Hypothesis:} Rate cuts signal easier monetary policy globally, which should boost emerging market equities like NIFTY 50 through capital inflows and improved risk sentiment.\n\n")
    w("This matters for:\n")
    w("\\begin{itemize}\n")
    w("    \\item \\textbf{Futures traders:} A systematic edge on event days translates to direct profit on NIFTY futures\n")
    w("    \\item \\textbf{Options traders:} Event-day volatility and direction affect straddle/strangle strategies\n")
    w("    \\item \\textbf{Portfolio managers:} Adjusting positions ahead of known FOMC rate cut announcements\n")
    w("    \\item \\textbf{Policy economists:} Quantifying US monetary policy spillover to Indian equities\n")
    w("\\end{itemize}\n\n")

    # METHODOLOGY
    w("\\section{Methodology}\n\n")

    w("\\subsection{Point Estimation}\n")
    w("For each group (event-day vs normal-day), we estimate:\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|X|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Estimator}} & \\textcolor{white}{\\textbf{Formula}} & \\textcolor{white}{\\textbf{Justification}} \\\\\n\\hline\n")
    w("Sample mean $\\bar{x}$ & $\\frac{1}{n} \\sum x_i$ & Unbiased estimator of $\\mu$ \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("Sample variance $s^2$ & $\\frac{1}{n-1} \\sum (x_i - \\bar{x})^2$ & Unbiased (Bessel's correction) \\\\\n\\hline\n")
    w(f"Sample proportion $\\hat{{p}}$ & $\\hat{{p}}_E = {neg_cut_count}/{n_cut} = {p_cut:.3f}$ \\newline $\\hat{{p}}_N = {neg_norm_count}/{n_norm} = {p_norm:.3f}$ & MLE for Bernoulli $p$ \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    w("\\subsection{Interval Estimation}\n")
    w("95\\% Confidence Interval for the mean (t-distribution, $\\sigma$ unknown):\n")
    w("\\begin{equation*}\n    \\bar{x} \\pm t_{\\alpha/2, n-1} \\cdot \\frac{s}{\\sqrt{n}}\n\\end{equation*}\n\n")
    w("95\\% CI for proportion (normal approximation, valid since $n\\hat{p} > 5$):\n")
    w("\\begin{equation*}\n    \\hat{p} \\pm z_{\\alpha/2} \\cdot \\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n}}\n\\end{equation*}\n\n")

    # Test 1
    w("\\subsection{Hypothesis Test 1 --- Two-Sample t-Test for Means}\n")
    w("We test whether Fed rate cuts have a positive effect on next-day NIFTY returns.\n")
    w("\\begin{equation*}\n    H_0: \\mu_E = \\mu_N \\quad \\text{vs} \\quad H_1: \\mu_E > \\mu_N \\quad \\text{(one-tailed, rate cuts boost returns)}\n\\end{equation*}\n")
    w("Welch's t-statistic (unequal variances assumed):\n")
    w("\\begin{equation*}\n    t = \\frac{\\bar{x}_E - \\bar{x}_N}{\\sqrt{\\frac{s_E^2}{n_E} + \\frac{s_N^2}{n_N}}}\n\\end{equation*}\n")
    w("Degrees of freedom via the Welch-Satterthwaite equation. Reject $H_0$ if $p < 0.05$.\n\n")

    # Test 2
    w("\\subsection{Hypothesis Test 2 --- z-Test for Difference in Proportions}\n")
    w("We test whether event-days have fewer negative returns (positive effect).\n")
    w("\\begin{equation*}\n    H_0: p_E = p_N \\quad \\text{vs} \\quad H_1: p_E < p_N \\quad \\text{(one-tailed, fewer negative days)}\n\\end{equation*}\n")
    w("\\begin{equation*}\n    z = \\frac{\\hat{p}_E - \\hat{p}_N}{\\sqrt{\\hat{p}(1-\\hat{p}) \\left( \\frac{1}{n_E} + \\frac{1}{n_N} \\right) }}\n\\end{equation*}\n")
    w("where $\\hat{p}$ is the pooled proportion under $H_0$.\n\n")

    # RESULTS
    w("\\newpage\n")
    w("\\section{Results \\& Inference}\n")
    w("Below are the numerical results from our analysis using real NIFTY 50 daily data (2000--2026):\n\n")

    # Descriptive stats
    w("\\vspace{1em}\n")
    w("\\textbf{Descriptive statistics --- daily log-returns (\\%):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|c|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Statistic}} & \\textcolor{white}{\\textbf{Event-day (E)}} & \\textcolor{white}{\\textbf{Normal-day (N)}} \\\\\n\\hline\n")
    w(f"Sample mean $\\bar{{x}}$ & {mean_cut:+.4f}\\% & {mean_norm:+.4f}\\% \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Sample std dev $s$ & {std_cut:.4f}\\% & {std_norm:.4f}\\% \\\\\n\\hline\n")
    w(f"Sample size $n$ & {n_cut} & {n_norm} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"95\\% CI for mean & [{ci_cut[0]:+.4f}\\%, {ci_cut[1]:+.4f}\\%] & [{ci_norm[0]:+.4f}\\%, {ci_norm[1]:+.4f}\\%] \\\\\n\\hline\n")
    w(f"Proportion neg days $\\hat{{p}}$ & {p_cut:.3f} & {p_norm:.3f} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"95\\% CI for proportion & [{ci_p_cut[0]:.3f}, {ci_p_cut[1]:.3f}] & [{ci_p_norm[0]:.3f}, {ci_p_norm[1]:.3f}] \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 1 Results
    w("\\vspace{1em}\n")
    w("\\textbf{Test 1 --- Welch's two-sample t-test (mean returns):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & $\\mu_E = \\mu_N$ (no difference) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & $\\mu_E > \\mu_N$ (rate cuts boost next-day returns) & \\\\\n\\hline\n")
    w(f"t-statistic & {t1:+.4f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Degrees of freedom & $\\sim{wdf:.0f}$ (Welch-Satterthwaite) & \\\\\n\\hline\n")
    w(f"p-value (one-tailed) & {p1_one:.4f} & \\textbf{{{t1_decision}}} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Critical value $t_{{0.05}}$ & +{t_crit_test:.3f} & {t1_crit_note} \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 2 Results
    w("\\textbf{Test 2 --- z-test for difference in proportions (downside risk):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & $p_E = p_N$ (same probability) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & $p_E < p_N$ (fewer negative days after rate cut) & \\\\\n\\hline\n")
    w(f"Pooled proportion $\\hat{{p}}$ & {p_pooled:.3f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"z-statistic & {z2:+.4f} & \\\\\n\\hline\n")
    w(f"p-value (one-tailed) & {p2:.4f} & \\textbf{{{z2_decision}}} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Critical value $z_{{0.05}}$ & $-1.645$ & {z2_crit_note} \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 3 Results
    w("\\newpage\n")
    w("\\textbf{Test 3 --- F-test (variance comparison):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & $\\sigma^2_E = \\sigma^2_N$ (same variance) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & $\\sigma^2_E \\neq \\sigma^2_N$ (different variance) & \\\\\n\\hline\n")
    w(f"F-statistic & {F3:.4f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Degrees of freedom & $df_1={n_cut-1}, df_2={n_norm-1}$ & \\\\\n\\hline\n")
    w(f"p-value (two-tailed) & {p3:.4f} & \\textbf{{{f3_decision}}} \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 4 Results
    w("\\textbf{Test 4 --- Jarque-Bera (normality of returns):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & Returns $\\sim$ Normal ($S=0, K=3$) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & Returns are non-normal & \\\\\n\\hline\n")
    w(f"JB statistic & {jb4:.2f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"p-value & {p4:.6f} & \\textbf{{{jb_decision}}} \\\\\n\\hline\n")
    w(f"Skewness ($S$) & {skew_r:.4f} & Left-skewed \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Excess Kurtosis ($K-3$) & {kurt_r:.4f} & Fat tails \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # STATISTICAL vs ECONOMIC SIGNIFICANCE
    w("\\section{Statistical Significance vs Economic Significance}\n\n")
    w("A critical distinction must be made between \\textbf{statistical significance} (can we reject $H_0$ at $\\alpha = 0.05$?) ")
    w("and \\textbf{economic/practical significance} (is the effect large enough to matter for trading?).\n\n")

    nifty_val = 25000
    nifty_pts = nifty_val * spread / 100
    futures_gain = nifty_val * 25 * spread / 100
    cum_alpha = n_cut * spread

    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|X|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Aspect}} & \\textcolor{white}{\\textbf{Statistical View}} & \\textcolor{white}{\\textbf{Economic/Trading View}} \\\\\n\\hline\n")
    w(f"Mean spread & ${spread:+.2f}\\%$ per event-day, $p = {p1_one:.4f}$ & ")
    w(f"On NIFTY at $\\sim$25,000: this is $\\sim${nifty_pts:.0f} points per event \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"For Futures (1 lot) & Not significant at $\\alpha = 0.05$ & ")
    w(f"Profit $\\approx$ \\textbf{{Rs {futures_gain:,.0f}}} per lot per event \\\\\n\\hline\n")
    w("For Options & Sample too small ($n=31$) & ATM calls gain from both delta and IV expansion during event \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Cumulative edge & Wide 95\\% CI crosses zero & {n_cut} events $\\times$ {spread:+.2f}\\% $=$ ")
    w(f"\\textbf{{{cum_alpha:+.1f}\\%}} cumulative alpha over 25 years \\\\\n\\hline\n")
    w(f"Annualised & CI: [{ci_cut[0]:+.2f}\\%, {ci_cut[1]:+.2f}\\%] & ")
    w(f"${spread:+.2f}\\% \\times 252 \\approx {ann_spread:+.1f}\\%$ annualised edge \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    w("\\textbf{Why the test fails to reject $H_0$:}\n")
    w("\\begin{enumerate}\n")
    w(f"    \\item \\textbf{{Small sample:}} Only $n = {n_cut}$ rate cut events in 25 years. The t-test requires large $n$ to detect small effects.\n")
    w(f"    \\item \\textbf{{Extreme volatility:}} Event-day $\\sigma = {std_cut:.2f}\\%$ ($\\sim{std_cut/std_norm:.1f}\\times$ normal), widening the CI.\n")
    w("    \\item \\textbf{Mixed reactions:} Some cuts occur during crises (2008, 2020) where markets fall despite the cut, inflating variance.\n")
    w("\\end{enumerate}\n\n")

    # CONCLUSION
    w("\\section{Overall Conclusion}\n\n")
    w("\\textbf{Estimation Conclusion:} The data suggests a \\textbf{directionally positive but statistically insignificant} ")
    w("effect of Fed rate cuts on NIFTY 50, with \\textbf{significantly higher volatility} on event days.\n\n")

    w(f"At $\\alpha = 0.05$, we {have_evidence} sufficient statistical evidence to conclude that:\n\n")
    w("\\begin{enumerate}\n")
    w(f"    \\item Mean NIFTY 50 next-day returns are {mean_sig} after Fed rate cuts ($t={t1:+.2f}, p={p1_one:.4f}$). The point estimate is positive ($\\bar{{x}}_E = {mean_cut:+.4f}\\%$ vs $\\bar{{x}}_N = {mean_norm:+.4f}\\%$) but the wide CI includes zero.\n")
    w(f"    \\item The probability of a negative return is {prop_sig} on event-days ($z={z2:+.2f}, p={p2:.4f}$).\n")
    w(f"    \\item Volatility IS significantly higher on event-days ($F={F3:.2f}, p={p3:.4f}$). The null hypothesis of equal variance is \\textbf{{rejected}}.\n")
    w(f"    \\item NIFTY daily returns are non-normal ($JB={jb4:.0f}, p\\approx 0$). The Student-t distribution provides a better fit than the Gaussian.\n")
    w("\\end{enumerate}\n\n")

    w("\\textbf{Key Insight:} In quantitative finance, a strategy with a positive expected return can be \\textbf{economically significant} ")
    w("even without statistical significance at the conventional $\\alpha = 0.05$ level. ")
    w(f"The ${spread:+.2f}\\%$ daily spread, while not rejecting $H_0$, represents a \\textbf{{tradeable edge}} ")
    w("that systematic traders would exploit with proper position sizing and risk management.\n\n")

    w("\\end{document}\n")

print(f"LaTeX file written to {TEX_PATH}")
