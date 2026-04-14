import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── 1. DATA ────────────────────────────────────────────────────────────────────
ticker   = yf.Ticker("^NSEI")
raw_data = ticker.history(start="2000-01-01", end="2026-04-14", auto_adjust=True)
prices_raw = raw_data["Close"].dropna()
prices_raw.index = prices_raw.index.tz_localize(None)

log_returns  = np.log(prices_raw / prices_raw.shift(1)).dropna()
trading_days = log_returns.index
r = log_returns.values
n = len(r)

# ── 2. FED EVENTS ──────────────────────────────────────────────────────────────
fed_events = {
    "2008-03-18": ("Cut",  -75), "2008-04-30": ("Cut",  -25),
    "2008-10-08": ("Cut",  -50), "2008-10-29": ("Cut",  -50),
    "2008-12-16": ("Cut",  -75),
    "2015-12-16": ("Hike", +25), "2016-12-14": ("Hike", +25),
    "2017-03-15": ("Hike", +25), "2017-06-14": ("Hike", +25),
    "2017-12-13": ("Hike", +25), "2018-03-21": ("Hike", +25),
    "2018-06-13": ("Hike", +25), "2018-09-26": ("Hike", +25),
    "2018-12-19": ("Hike", +25),
    "2019-07-31": ("Cut",  -25), "2019-09-18": ("Cut",  -25),
    "2019-10-30": ("Cut",  -25),
    "2020-03-03": ("Cut",  -50), "2020-03-15": ("Cut", -100),
    "2022-03-17": ("Hike", +25), "2022-05-05": ("Hike", +50),
    "2022-06-16": ("Hike", +75), "2022-07-28": ("Hike", +75),
    "2022-09-22": ("Hike", +75), "2022-11-03": ("Hike", +75),
    "2022-12-15": ("Hike", +50), "2023-02-02": ("Hike", +25),
    "2023-03-23": ("Hike", +25), "2023-05-04": ("Hike", +25),
    "2024-09-19": ("Cut",  -50), "2024-11-07": ("Cut",  -25),
    "2024-12-18": ("Cut",  -25),
}

def nearest_td(date_str):
    t = pd.Timestamp(date_str)
    c = trading_days[trading_days >= t]
    if not len(c) or (c[0]-t).days > 7: return None
    return c[0]

results = []
for ds, (et, bps) in fed_events.items():
    ed = nearest_td(ds)
    if ed is None: continue
    idx = trading_days.get_loc(ed)
    if idx < 121 or idx+1 >= len(trading_days): continue
    mu_est = np.mean(log_returns.iloc[idx-120:idx-10].values)
    ar_m1  = float(log_returns.iloc[idx-1]) - mu_est
    ar_0   = float(log_returns.iloc[idx])   - mu_est
    ar_p1  = float(log_returns.iloc[idx+1]) - mu_est
    car    = ar_m1 + ar_0 + ar_p1
    results.append({"date":ds,"type":et,"bps":bps,
                    "mu_est":mu_est,"ar_m1":ar_m1,"ar_0":ar_0,"ar_p1":ar_p1,"car":car})

df        = pd.DataFrame(results)
cars      = df["car"].values
cars_hike = df[df["type"]=="Hike"]["car"].values
cars_cut  = df[df["type"]=="Cut"]["car"].values

mean_r  = np.mean(r);  std_r = np.std(r, ddof=1)
skew_r  = stats.skew(r); kurt_r = stats.kurtosis(r)
mu_mle  = mean_r;  sig_mle = np.std(r, ddof=0)
df_t, loc_t, scale_t = stats.t.fit(r)

# ── 3. PLOTS ───────────────────────────────────────────────────────────────────
BG   = "#0f1117"
FG   = "#e8eaf0"
ACC1 = "#4fc3f7"   # blue
ACC2 = "#ef5350"   # red
ACC3 = "#66bb6a"   # green
ACC4 = "#ffa726"   # orange
GREY = "#2a2d3a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GREY, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GREY,
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle("NIFTY 50  ·  Impact of US Fed Rate Decisions  (2007–2026)",
             fontsize=18, fontweight="bold", color=FG, y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32)

# ── Plot 1: Price history with Fed event overlays ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(prices_raw.index, prices_raw.values, color=ACC1, linewidth=1.2, label="NIFTY 50 Close")
ax1.fill_between(prices_raw.index, prices_raw.values, alpha=0.08, color=ACC1)

for ds, (et, bps) in fed_events.items():
    ed = nearest_td(ds)
    if ed is None: continue
    y = prices_raw.get(ed, None)
    if y is None: continue
    col = ACC3 if et == "Cut" else ACC2
    ax1.axvline(ed, color=col, alpha=0.35, linewidth=0.8)

from matplotlib.lines import Line2D
legend_h = [
    Line2D([0],[0], color=ACC1, linewidth=2, label="NIFTY 50"),
    Line2D([0],[0], color=ACC3, linewidth=1.5, alpha=0.8, label="Fed Cut"),
    Line2D([0],[0], color=ACC2, linewidth=1.5, alpha=0.8, label="Fed Hike"),
]
ax1.legend(handles=legend_h, loc="upper left", facecolor=GREY, edgecolor="none", labelcolor=FG)
ax1.set_title("NIFTY 50 Price History with Fed Decision Events", color=FG, fontsize=13)
ax1.set_ylabel("Index Level", color=FG)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.grid(True)

# ── Plot 2: CAR bar chart ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
colors_bar = [ACC3 if t=="Cut" else ACC2 for t in df["type"]]
bars = ax2.bar(range(len(df)), df["car"]*100, color=colors_bar, edgecolor="none", alpha=0.85)
ax2.axhline(0, color=FG, linewidth=0.8, linestyle="-")
ax2.axhline(np.mean(cars)*100, color=ACC4, linewidth=1.5, linestyle="--",
            label=f"Mean CAR = {np.mean(cars)*100:.2f}%")
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels([f"{r['date']}\n{r['type']} {r['bps']:+}bps" for _, r in df.iterrows()],
                    rotation=45, ha="right", fontsize=7.5)
ax2.set_ylabel("CAR (%)", color=FG)
ax2.set_title("Cumulative Abnormal Returns  [τ−1, τ, τ+1]  per Fed Event", color=FG, fontsize=13)
ax2.legend(facecolor=GREY, edgecolor="none", labelcolor=FG)
from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(color=ACC2, label=f"Hike  (mean={np.mean(cars_hike)*100:.2f}%)"),
    Patch(color=ACC3, label=f"Cut   (mean={np.mean(cars_cut)*100:.2f}%)"),
    Line2D([0],[0], color=ACC4, linestyle="--", label=f"Overall mean={np.mean(cars)*100:.2f}%"),
], facecolor=GREY, edgecolor="none", labelcolor=FG)
ax2.grid(True, axis="y")

# ── Plot 3: Return distribution ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
x_vals = np.linspace(r.min(), r.max(), 400)
ax3.hist(r*100, bins=120, density=True, color=ACC1, alpha=0.5, edgecolor="none", label="Actual")
norm_pdf = stats.norm.pdf(x_vals*100, mean_r*100, std_r*100) / 100
ax3.plot(x_vals*100, norm_pdf, color=ACC4, linewidth=1.8, label="Normal fit")
t_pdf = stats.t.pdf(x_vals, df_t, loc_t, scale_t)
ax3.plot(x_vals*100, t_pdf/100, color=ACC2, linewidth=1.8, label=f"t-dist (ν={df_t:.2f})")
ax3.set_xlim(-10, 10)
ax3.set_xlabel("Daily Log Return (%)", color=FG)
ax3.set_ylabel("Density", color=FG)
ax3.set_title("Return Distribution", color=FG, fontsize=12)
ax3.legend(facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax3.grid(True)
stats_txt = (f"μ = {mean_r*100:.3f}%\nσ = {std_r*100:.3f}%\n"
             f"Skew = {skew_r:.3f}\nKurt = {kurt_r:.2f}")
ax3.text(0.97, 0.97, stats_txt, transform=ax3.transAxes, va="top", ha="right",
         fontsize=8.5, color=FG, bbox=dict(fc=GREY, ec="none", pad=4))

# ── Plot 4: Q-Q plot ───────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
ax4.scatter(osm, osr*std_r+mean_r, color=ACC1, alpha=0.3, s=4, label="Empirical")
line_x = np.array([osm[0], osm[-1]])
ax4.plot(line_x, slope*line_x + intercept, color=ACC4, linewidth=2, label="Normal line")
ax4.set_xlabel("Theoretical Quantiles", color=FG)
ax4.set_ylabel("Sample Quantiles", color=FG)
ax4.set_title("Q-Q Plot  (Normal)", color=FG, fontsize=12)
ax4.legend(facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax4.grid(True)

# ── Plot 5: Hike vs Cut CAR comparison ────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
bp = ax5.boxplot([cars_hike*100, cars_cut*100],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color=FG, linewidth=2))
bp["boxes"][0].set_facecolor(ACC2);  bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(ACC3);  bp["boxes"][1].set_alpha(0.7)
for w in bp["whiskers"]+bp["caps"]+bp["fliers"]:
    w.set(color=FG, linewidth=1, markersize=4)
ax5.axhline(0, color=FG, linewidth=0.8, linestyle="--")
ax5.set_xticks([1,2])
ax5.set_xticklabels([f"HIKE (N={len(cars_hike)})", f"CUT (N={len(cars_cut)})"], color=FG)
ax5.set_ylabel("CAR (%)", color=FG)
ax5.set_title("CAR Distribution: Hike vs Cut", color=FG, fontsize=12)
ax5.grid(True, axis="y")
t2, p2 = stats.ttest_ind(cars_hike, cars_cut, equal_var=False)
ax5.text(0.5, 0.03, f"Welch t={t2:.3f},  p={p2:.3f}", transform=ax5.transAxes,
         ha="center", fontsize=9, color=ACC4,
         bbox=dict(fc=GREY, ec="none", pad=3))

# ── Plot 6: Era annualised returns ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
eras = {
    "GFC\n(2007-09)":   ("2007-09-17", "2009-12-31"),
    "Post-GFC\n(10-19)":("2010-01-01", "2019-12-31"),
    "COVID\n(2020)":    ("2020-01-01", "2020-12-31"),
    "Post-COVID\n(21)": ("2021-01-01", "2021-12-31"),
    "Hike\n(22-23)":    ("2022-01-01", "2023-12-31"),
    "Cut\n(24-26)":     ("2024-01-01", "2026-04-14"),
}
era_names, era_rets, era_vols = [], [], []
for label, (s, e) in eras.items():
    era_r = log_returns.loc[s:e].values
    if len(era_r) < 20: continue
    era_names.append(label)
    era_rets.append(np.mean(era_r)*252*100)
    era_vols.append(np.std(era_r, ddof=1)*np.sqrt(252)*100)

era_colors = [ACC2 if v > 25 else ACC3 if v < 18 else ACC4 for v in era_vols]
bars6 = ax6.bar(era_names, era_rets, color=era_colors, alpha=0.8, edgecolor="none")
ax6.plot(era_names, era_vols, color=ACC1, marker="o", linewidth=2,
         markersize=6, label="Ann. Volatility (%)")
for bar, ret in zip(bars6, era_rets):
    ax6.text(bar.get_x()+bar.get_width()/2, ret+0.5, f"{ret:.1f}%",
             ha="center", va="bottom", fontsize=8, color=FG)
ax6.set_ylabel("Annualised Return / Volatility (%)", color=FG)
ax6.set_title("Era-Wise Returns (bars) & Volatility (line)", color=FG, fontsize=12)
ax6.legend(facecolor=GREY, edgecolor="none", labelcolor=FG, fontsize=9)
ax6.grid(True, axis="y")

plt.savefig("nifty_fed_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved: nifty_fed_analysis.png")
plt.close()
