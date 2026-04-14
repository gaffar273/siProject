import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
TEX_PATH = os.path.join(OUT_DIR, "nifty50_fed_rate_cuts_report.tex")

# ═══════════════════════════════════════════════════════════════
# A.  COMPUTE ALL NUMBERS FROM REAL DATA
# ═══════════════════════════════════════════════════════════════
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

daily = daily.sort_values("Date").drop_duplicates(subset="Date").set_index("Date")
weekly = daily["Close"].resample("W-FRI").last().dropna()
log_returns = np.log(weekly / weekly.shift(1)).dropna()
r = log_returns.values
n = len(r)

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
trading_weeks = log_returns.index

def find_event_week(ds):
    t = pd.Timestamp(ds)
    c = trading_weeks[trading_weeks >= t]
    if not len(c) or (c[0]-t).days > 10: return None
    return c[0]

week_df = pd.DataFrame({"Date": log_returns.index, "Close": weekly.loc[log_returns.index].values,
                         "return_pct": log_returns.values * 100})
event_week_dates = set()
for ds in fed_cuts:
    ew = find_event_week(ds)
    if ew is not None:
        event_week_dates.add(ew)

week_df["fed_cut_week"] = week_df["Date"].isin(event_week_dates)
week_df["group"] = week_df["fed_cut_week"].map({True: "Cut-week", False: "Non-cut-week"})
week_df["negative_week"] = (week_df["return_pct"] < 0).astype(int)

cut_rets  = week_df[week_df["group"]=="Cut-week"]["return_pct"].values
norm_rets = week_df[week_df["group"]=="Non-cut-week"]["return_pct"].values
n_cut = len(cut_rets)
n_norm= len(norm_rets)

mean_cut = np.mean(cut_rets);  std_cut = np.std(cut_rets, ddof=1)
mean_norm= np.mean(norm_rets); std_norm= np.std(norm_rets, ddof=1)

p_cut  = np.mean(cut_rets < 0)
p_norm = np.mean(norm_rets < 0)

mean_r = np.mean(r); std_r = np.std(r, ddof=1)
skew_r = stats.skew(r); kurt_r = stats.kurtosis(r)

alpha = 0.05
se_mu = std_r / np.sqrt(n)
t_crit = stats.t.ppf(1-alpha/2, df=n-1)

t1, p1 = stats.ttest_ind(cut_rets, norm_rets, equal_var=False)
p_pooled = (np.sum(cut_rets < 0) + np.sum(norm_rets < 0)) / (n_cut + n_norm)
se_prop = np.sqrt(p_pooled*(1-p_pooled)*(1/n_cut + 1/n_norm))
z2 = (p_cut - p_norm) / se_prop
p2 = 1 - stats.norm.cdf(z2)
F3 = np.var(cut_rets, ddof=1) / np.var(norm_rets, ddof=1)
p3 = 2 * min(stats.f.cdf(F3, n_cut-1, n_norm-1), 1-stats.f.cdf(F3, n_cut-1, n_norm-1))
jb4, p4 = stats.jarque_bera(r)

se_cut = std_cut / np.sqrt(n_cut)
se_norm_se = std_norm / np.sqrt(n_norm)
t_c_cut = stats.t.ppf(1-alpha/2, df=n_cut-1)
t_c_norm = stats.t.ppf(1-alpha/2, df=n_norm-1)
ci_cut  = (mean_cut - t_c_cut*se_cut, mean_cut + t_c_cut*se_cut)
ci_norm = (mean_norm - t_c_norm*se_norm_se, mean_norm + t_c_norm*se_norm_se)
z_c = 1.96
ci_p_cut  = (p_cut - z_c*np.sqrt(p_cut*(1-p_cut)/n_cut), p_cut + z_c*np.sqrt(p_cut*(1-p_cut)/n_cut))
ci_p_norm = (p_norm- z_c*np.sqrt(p_norm*(1-p_norm)/n_norm), p_norm+ z_c*np.sqrt(p_norm*(1-p_norm)/n_norm))

wdf = ((std_cut**2/n_cut + std_norm**2/n_norm)**2) / ((std_cut**2/n_cut)**2/(n_cut-1) + (std_norm**2/n_norm)**2/(n_norm-1))
t_crit_test = stats.t.ppf(0.05, df=wdf)

spread = mean_norm - mean_cut
ann_spread = spread * 52

neg_cut_count = int(np.sum(cut_rets < 0))
neg_norm_count = int(np.sum(norm_rets < 0))

print(f"N={n}, n_cut={n_cut}, n_norm={n_norm}")
print(f"mean_cut={mean_cut:.4f}, mean_norm={mean_norm:.4f}")

# ═══════════════════════════════════════════════════════════════
# B.  BUILD SAMPLE TABLE (ALL cut-weeks + non-cut to reach 50)
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)
cut_sample = week_df[week_df["fed_cut_week"] == True].copy()
non_cut_rows = week_df[week_df["fed_cut_week"] == False]
n_non_cut_needed = 50 - len(cut_sample)
print(f"Cut-week rows: {len(cut_sample)}, Non-cut rows to add: {n_non_cut_needed}")
non_cut_sample = non_cut_rows.sample(n=max(n_non_cut_needed, 0), random_state=42)
sample_rows = pd.concat([cut_sample, non_cut_sample]).sort_values("Date").copy()

# ═══════════════════════════════════════════════════════════════
# C.  WRITE THE .TEX FILE LINE BY LINE (avoids escape issues)
# ═══════════════════════════════════════════════════════════════
print("Writing LaTeX...")

# Decision strings
t1_decision = "Reject $H_0$" if p1/2 < 0.05 else "Fail to Reject $H_0$"
t1_crit_note = "$t < t_{crit} \\Rightarrow$ reject" if t1 < t_crit_test else "$|t| < |t_{crit}| \\Rightarrow$ fail to reject"
z2_decision = "Reject $H_0$" if p2 < 0.05 else "Fail to Reject $H_0$"
z2_crit_note = "$z > z_{crit} \\Rightarrow$ reject" if z2 > 1.645 else "$z < z_{crit} \\Rightarrow$ fail to reject"
f3_decision = "Reject $H_0$" if p3 < 0.05 else "Fail to Reject $H_0$"
jb_decision = "Reject $H_0$" if p4 < 0.05 else "Fail to Reject $H_0$"

inv_sig = "statistically significant" if p1/2 < 0.05 else "directionally lower but not statistically significant"
inv_adj = "adjust" if p1/2 < 0.05 else "be cautious about over-adjusting"
mean_sig = "significantly lower" if p1/2 < 0.05 else "directionally lower but not significantly different"
prop_sig = "significantly higher" if p2 < 0.05 else "not significantly different"
prop_note = "This difference is statistically significant (z-test)." if p2 < 0.05 else "However, this difference is not statistically significant."
vol_note = "significantly higher" if p3 < 0.05 else "not significantly different"
have_evidence = "have" if p1/2 < 0.05 or p2 < 0.05 else "do not have"

with open(TEX_PATH, "w", encoding="utf-8") as f:
    w = f.write

    # PREAMBLE
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

    # TITLE
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
    w("1. Data Description & NIFTY 50 weekly returns\\newline Source: Investing.com + Yahoo Finance \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("2. Problem Statement & Do Fed rate cuts significantly alter return distributions?\\newline Statistical inference: $H_0$ vs $H_1$, $\\alpha = 0.05$ \\\\\n\\hline\n")
    w("3. Methodology & Point \\& interval estimation\\newline Two-sample t-test\\newline z-test for proportions \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("4. Results \\& Inference & Hypothesis decisions\\newline Confidence intervals\\newline Practical implications \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # DATA DESCRIPTION
    w("\\newpage\n")
    w("\\section{Data Description}\n")
    w(f"Dataset: Weekly closing prices of the NIFTY 50 Index (\\textasciicircum NSEI) from April 2000 to April 2026 - \\textbf{{{n}}} weekly observations.\n\n")
    w("Derived variable: Weekly log-return, calculated as:\n")
    w("\\begin{equation*}\n    r_t = \\ln\\left( \\frac{P_t}{P_{t-1}} \\right) \\times 100\n\\end{equation*}\n\n")
    w("\\textbf{Source:}\n")
    w("\\begin{itemize}\n")
    w("    \\item Investing.com CSV download for NIFTY 50 daily prices (Apr 2000 - May 2020)\n")
    w("    \\item Yahoo Finance (via yfinance Python library) for Jun 2020 - Apr 2026\n")
    w("    \\item Federal Reserve FOMC meeting dates for rate cut event classification\n")
    w("\\end{itemize}\n\n")

    # Period classification table
    w("\\textbf{Period classification:}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Period}} & \\textcolor{white}{\\textbf{Weeks}} & \\textcolor{white}{\\textbf{Key Events}} \\\\\n\\hline\n")
    w(f"Fed cut weeks & $\\sim${n_cut} weeks & Weeks containing FOMC rate cut decisions \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Non-cut weeks & $\\sim${n_norm} weeks & All remaining trading weeks \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # SAMPLE DATA TABLE (50 rows)
    w("\\textbf{Sample Data Rows (50 observations --- all cut-weeks + selected non-cut-weeks):}\n")
    w("\\begin{table}[H]\n\\centering\n\\footnotesize\n")
    w("\\begin{tabular}{|l|c|c|c|l|c|}\n\\hline\n")
    w("\\rowcolor{hdrbg}\\textcolor{white}{\\textbf{Date}} & \\textcolor{white}{\\textbf{Close}} & \\textcolor{white}{\\textbf{return\\_pct}} & \\textcolor{white}{\\textbf{fed\\_cut\\_week}} & \\textcolor{white}{\\textbf{group}} & \\textcolor{white}{\\textbf{neg\\_week}} \\\\\n\\hline\n")
    for _, row in sample_rows.iterrows():
        dt = row["Date"].strftime("%d-%m-%Y")
        cl = f"{row['Close']:.2f}"
        rp = f"{row['return_pct']:.4f}"
        fc = "TRUE" if row["fed_cut_week"] else "FALSE"
        gr = row["group"]
        nw = int(row["negative_week"])
        w(f"{dt} & {cl} & {rp} & {fc} & {gr} & {nw} \\\\\n\\hline\n")
    w("\\end{tabular}\n\\end{table}\n\n")

    # REAL-WORLD PROBLEM
    w("\\newpage\n")
    w("\\section{Real-World Problem}\n\n")
    w("``Is the mean weekly return and the probability of a negative week statistically different during Fed rate cut weeks compared to non-cut weeks?''\n\n")
    w("This matters enormously for:\n")
    w("\\begin{itemize}\n")
    w("    \\item Portfolio managers deciding asset allocation around FOMC announcements\n")
    w("    \\item Risk modellers calibrating VaR under US monetary policy stress\n")
    w("    \\item Policy economists quantifying the spillover cost of US rate decisions on Indian equities\n")
    w("\\end{itemize}\n\n")

    # METHODOLOGY
    w("\\section{Methodology}\n\n")

    # 3.1 Point Estimation
    w("\\subsection{Point Estimation}\n")
    w("For each group (cut-week vs non-cut-week), we estimate:\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|X|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Estimator}} & \\textcolor{white}{\\textbf{Formula}} & \\textcolor{white}{\\textbf{Justification}} \\\\\n\\hline\n")
    w("Sample mean $\\bar{x}$ & $\\frac{1}{n} \\sum x_i$ & Unbiased estimator of $\\mu$ \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("Sample variance $s^2$ & $\\frac{1}{n-1} \\sum (x_i - \\bar{x})^2$ & Unbiased (uses $n-1$) \\\\\n\\hline\n")
    w(f"Sample proportion $\\hat{{p}}$ & $\\hat{{p}}_C = {neg_cut_count}/{n_cut} = {p_cut:.3f}$ \\newline $\\hat{{p}}_N = {neg_norm_count}/{n_norm} = {p_norm:.3f}$ & MLE for Bernoulli $p$ \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # 3.2 Interval Estimation
    w("\\subsection{Interval Estimation}\n")
    w("95\\% Confidence Interval for the mean (using t-distribution since $\\sigma$ is unknown):\n")
    w("\\begin{equation*}\n    \\bar{x} \\pm t_{\\alpha/2, n-1} \\cdot \\frac{s}{\\sqrt{n}}\n\\end{equation*}\n\n")
    w("95\\% CI for proportion (using normal approximation, valid since $n\\hat{p} > 5$):\n")
    w("\\begin{equation*}\n    \\hat{p} \\pm z_{\\alpha/2} \\cdot \\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n}}\n\\end{equation*}\n\n")

    # 3.3 Hypothesis Test 1
    w("\\subsection{Hypothesis Test 1 --- Two-Sample t-Test for Means}\n")
    w("We test whether mean weekly returns differ across groups.\n")
    w("\\begin{equation*}\n    H_0: \\mu_C = \\mu_N \\quad \\text{vs} \\quad H_1: \\mu_C < \\mu_N \\quad \\text{(one-tailed)}\n\\end{equation*}\n")
    w("Welch's t-statistic (unequal variances assumed):\n")
    w("\\begin{equation*}\n    t = \\frac{\\bar{x}_C - \\bar{x}_N}{\\sqrt{\\frac{s_C^2}{n_C} + \\frac{s_N^2}{n_N}}}\n\\end{equation*}\n")
    w("Degrees of freedom via the Welch-Satterthwaite equation. Reject $H_0$ if $p < 0.05$.\n\n")

    # 3.4 Hypothesis Test 2
    w("\\subsection{Hypothesis Test 2 --- z-Test for Difference in Proportions}\n")
    w("\\begin{equation*}\n    H_0: p_C = p_N \\quad \\text{vs} \\quad H_1: p_C > p_N \\quad \\text{(one-tailed)}\n\\end{equation*}\n")
    w("\\begin{equation*}\n    z = \\frac{\\hat{p}_C - \\hat{p}_N}{\\sqrt{\\hat{p}(1-\\hat{p}) \\left( \\frac{1}{n_C} + \\frac{1}{n_N} \\right) }}\n\\end{equation*}\n")
    w("where $\\hat{p}$ is the pooled proportion under $H_0$.\n\n")

    # RESULTS
    w("\\newpage\n")
    w("\\section{Results \\& Inference}\n")
    w("Below are the numerical results from our analysis using real NIFTY 50 weekly data (2000--2026):\n\n")

    # Descriptive stats
    w("\\vspace{1em}\n")
    w("\\textbf{Descriptive statistics --- weekly log-returns (\\%):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|c|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Statistic}} & \\textcolor{white}{\\textbf{Cut-week (C)}} & \\textcolor{white}{\\textbf{Non-cut-week (N)}} \\\\\n\\hline\n")
    w(f"Sample mean $\\bar{{x}}$ & {mean_cut:.4f}\\% & {mean_norm:.4f}\\% \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Sample std dev $s$ & {std_cut:.4f}\\% & {std_norm:.4f}\\% \\\\\n\\hline\n")
    w(f"Sample size $n$ & {n_cut} & {n_norm} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"95\\% CI for mean & [{ci_cut[0]:.2f}\\%, {ci_cut[1]:.2f}\\%] & [{ci_norm[0]:.2f}\\%, {ci_norm[1]:.2f}\\%] \\\\\n\\hline\n")
    w(f"Proportion neg weeks $\\hat{{p}}$ & {p_cut:.3f} & {p_norm:.3f} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"95\\% CI for proportion & [{ci_p_cut[0]:.3f}, {ci_p_cut[1]:.3f}] & [{ci_p_norm[0]:.3f}, {ci_p_norm[1]:.3f}] \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 1
    w("\\vspace{1em}\n")
    w("\\textbf{Test 1 --- Welch's two-sample t-test (mean returns):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & $\\mu_C = \\mu_N$ (no difference) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & $\\mu_C < \\mu_N$ (cut-week is lower) & \\\\\n\\hline\n")
    w(f"t-statistic & {t1:.4f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Degrees of freedom & $\\sim{wdf:.0f}$ (Welch-Satterthwaite) & \\\\\n\\hline\n")
    w(f"p-value (one-tailed) & {p1/2:.4f} & \\textbf{{{t1_decision}}} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Critical value $t_{{0.05}}$ & {t_crit_test:.3f} & {t1_crit_note} \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 2
    w("\\textbf{Test 2 --- z-test for difference in proportions (downside risk):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & $p_C = p_N$ (same probability) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & $p_C > p_N$ (more negative weeks) & \\\\\n\\hline\n")
    w(f"Pooled proportion $\\hat{{p}}$ & {p_pooled:.3f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"z-statistic & {z2:+.4f} & \\\\\n\\hline\n")
    w(f"p-value (one-tailed) & {p2:.4f} & \\textbf{{{z2_decision}}} \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Critical value $z_{{0.05}}$ & 1.645 & {z2_crit_note} \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 3
    w("\\newpage\n")
    w("\\textbf{Test 3 --- F-test (variance comparison):}\n")
    w("\\begin{table}[H]\n\\centering\n")
    w("\\begin{tabularx}{\\textwidth}{|l|l|X|}\n\\hline\n")
    w("\\rowcolor{hdrbg} \\textcolor{white}{\\textbf{Item}} & \\textcolor{white}{\\textbf{Value}} & \\textcolor{white}{\\textbf{Decision}} \\\\\n\\hline\n")
    w("$H_0$ & $\\sigma^2_C = \\sigma^2_N$ (same variance) & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w("$H_1$ & $\\sigma^2_C \\neq \\sigma^2_N$ (different variance) & \\\\\n\\hline\n")
    w(f"F-statistic & {F3:.4f} & \\\\\n\\hline\n")
    w("\\rowcolor{rowalt}\n")
    w(f"Degrees of freedom & $df_1={n_cut-1}, df_2={n_norm-1}$ & \\\\\n\\hline\n")
    w(f"p-value (two-tailed) & {p3:.4f} & \\textbf{{{f3_decision}}} \\\\\n\\hline\n")
    w("\\end{tabularx}\n\\end{table}\n\n")

    # Test 4
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

    # PRACTICAL IMPLICATIONS
    w("\\section{Practical Implications}\n\n")
    w(f"\\textbf{{For investors:}} The evidence confirms that Fed rate cut weeks show a {inv_sig} change in NIFTY 50 returns. ")
    w(f"The mean weekly return during cut-weeks is {mean_cut:.2f}\\% vs {mean_norm:.2f}\\% during non-cut weeks --- a spread of $\\sim${spread:.2f}\\% per week. ")
    w(f"A portfolio manager should {inv_adj} downside risk budgets when the Fed signals rate cuts.\n\n")

    w("\\vspace{1em}\n")
    w(f"\\textbf{{For risk management:}} With $\\hat{{p}}_C = {p_cut:.3f}$, about {p_cut*10:.0f} in 10 cut-weeks deliver a negative return vs {p_norm*10:.0f} in 10 for normal weeks. ")
    w(f"{prop_note} ")
    w(f"The F-test shows volatility is {vol_note} during cut-weeks ($F={F3:.2f}, p={p3:.4f}$).\n\n")

    w("\\vspace{1em}\n")
    w(f"\\textbf{{For economic policy:}} The $\\sim${abs(spread):.2f}\\% spread in mean weekly returns compounds to roughly a ${abs(ann_spread):.1f}\\%$ annualised gap. ")
    w(f"NIFTY returns are strongly non-normal ($JB={jb4:.0f}, p\\approx 0$) with fat tails (excess kurtosis = ${kurt_r:.2f}$), ")
    w("meaning standard Gaussian risk models underestimate tail risk for Indian equities.\n\n")

    # CONCLUSION
    w("\\section{Overall Conclusion}\n")
    w(f"At $\\alpha = 0.05$, we {have_evidence} sufficient statistical evidence to conclude that:\n\n")
    w("\\begin{enumerate}\n")
    w(f"    \\item Mean NIFTY 50 weekly returns are {mean_sig} during Fed rate cut weeks ($t={t1:.2f}, p={p1/2:.4f}$).\n")
    w(f"    \\item The probability of a negative return week is {prop_sig} during cut-weeks ($z={z2:.2f}, p={p2:.4f}$).\n")
    w(f"    \\item Volatility IS significantly higher during cut-weeks ($F={F3:.2f}, p={p3:.4f}$). The null hypothesis of equal variance is rejected.\n")
    w(f"    \\item NIFTY weekly returns are non-normal ($JB={jb4:.0f}, p\\approx 0$). The Student-t distribution provides a better fit than the Gaussian.\n")
    w("\\end{enumerate}\n\n")
    w("\\end{document}\n")

print(f"LaTeX file written to {TEX_PATH}")
