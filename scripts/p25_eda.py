"""
=============================================================================
P25 — p25_eda.py   Exploratory Data Analysis
=============================================================================
Run BEFORE any model training. Checks every statistical assumption the
pipeline depends on and ends with a GO / NO-GO training verdict.

Usage (from project root, venv active):
    python scripts/p25_eda.py

Outputs  →  outputs/p25_outputs/
    p25_eda_01_missing_values.png
    p25_eda_02_y1_distribution.png
    p25_eda_03_y2_class_balance.png
    p25_eda_04_correlation_heatmap.png
    p25_eda_05_top_features_y1.png
    p25_eda_06_heteroscedasticity.png
    p25_eda_07_outliers_y1.png
    p25_eda_08_d2_features.png
    p25_eda_09_y1_by_y2_class.png
    p25_eda_summary.txt
=============================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings("ignore")

# ── resolve paths whether script is run from project root or scripts/ ──────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT, "data", "processed")
OUT_DIR    = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}")

# ── colours ─────────────────────────────────────────────────────────────────
CB = "#2E75B6"; CG = "#1E5C2E"; CO = "#C55A11"
CR = "#9B1C1C"; CP = "#5B2C8D"; CGR = "#595959"

# ── check tracker ───────────────────────────────────────────────────────────
CHECKS   = []   # (status, name, detail)
ISSUES   = []
WARNINGS = []

def ok(name, detail=""):
    CHECKS.append(("PASS", name, detail))
    print(f"  \033[92m[PASS]\033[0m  {name}")
    if detail: print(f"          → {detail}")

def warn(name, detail=""):
    CHECKS.append(("WARN", name, detail))
    WARNINGS.append((name, detail))
    print(f"  \033[93m[WARN]\033[0m  {name}")
    if detail: print(f"          → {detail}")

def fail(name, detail=""):
    CHECKS.append(("FAIL", name, detail))
    ISSUES.append((name, detail))
    print(f"  \033[91m[FAIL]\033[0m  {name}")
    if detail: print(f"          → {detail}")

# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  P25 — Exploratory Data Analysis")
print("="*65)

for path, label in [(os.path.join(DATA_DIR,"stage1_input.csv"), "stage1_input.csv"),
                    (os.path.join(DATA_DIR,"stage2_base.csv"),  "stage2_base.csv")]:
    if not os.path.exists(path):
        print(f"\n[ERROR] {label} not found at:\n  {path}")
        print("Run: python scripts/p25_data_pipeline.py  first")
        sys.exit(1)

df1 = pd.read_csv(os.path.join(DATA_DIR, "stage1_input.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "stage2_base.csv"))
X1  = df1.drop("ViolentCrimesPerPop", axis=1)
Y1  = df1["ViolentCrimesPerPop"]
X2  = df2.drop("EmergencyActivation", axis=1)
Y2  = df2["EmergencyActivation"]

print(f"\n  Stage 1 : {df1.shape}  ({X1.shape[1]} features + Y1)")
print(f"  Stage 2 : {df2.shape}  ({X2.shape[1]} features + Y2)")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 1 — MISSING VALUES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 1 — Missing Values")
print("─"*65)

m1, m2 = df1.isnull().sum().sum(), df2.isnull().sum().sum()
ok("Stage 1: zero missing") if m1 == 0 else fail(f"Stage 1: {m1} missing","Re-run p25_data_pipeline.py")
ok("Stage 2: zero missing") if m2 == 0 else fail(f"Stage 2: {m2} missing","Re-run p25_data_pipeline.py")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Check 1 — Missing Values (all should be 0)",
             fontsize=13, fontweight="bold")
for ax, df, title in [(axes[0], df1, "Stage 1 Input (1994×101)"),
                      (axes[1], df2, "Stage 2 Base  (1994×8)")]:
    miss = df.isnull().sum()
    nonz = miss[miss > 0]
    if len(nonz) == 0:
        ax.bar(["All columns"], [0], color=CG, alpha=0.8, width=0.3)
        ax.text(0, 0.2, "✓  Zero missing values\nin all columns",
                ha="center", fontsize=13, color=CG, fontweight="bold")
        ax.set_ylim(0, 1)
    else:
        nonz.plot(kind="bar", ax=ax, color=CR, alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Missing count")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, "p25_eda_01_missing_values.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Y1 TARGET DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 2 — Y1 Distribution (ViolentCrimesPerPop)")
print("─"*65)

y1_skew  = Y1.skew()
y1_kurt  = Y1.kurtosis()
y1_log   = np.log1p(Y1)
ls       = y1_log.skew()

print(f"  Range  : [{Y1.min():.4f}, {Y1.max():.4f}]")
print(f"  Mean   : {Y1.mean():.4f}   Median : {Y1.median():.4f}")
print(f"  Std    : {Y1.std():.4f}   Skew   : {y1_skew:.4f}")
print(f"  log1p skew: {ls:.4f}   Kurtosis: {y1_kurt:.4f}")

if abs(y1_skew) > 1.0:
    warn(f"Y1 right-skewed (skew={y1_skew:.2f})",
         f"log1p applied for LR/Ridge → skew improves to {ls:.2f}")
else:
    ok(f"Y1 distribution acceptable (skew={y1_skew:.2f})")

if abs(ls) < 0.8:
    ok(f"log1p(Y1) skew={ls:.2f} — linear models can train on this")
else:
    warn(f"log1p(Y1) still skewed ({ls:.2f}) — RF/GBM will handle better")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
fig.suptitle("Check 2 — Y1 Target Distribution Analysis",
             fontsize=13, fontweight="bold")

# Histogram raw
ax = fig.add_subplot(gs[0, 0])
ax.hist(Y1, bins=50, color=CB, alpha=0.8, edgecolor="white")
ax.axvline(Y1.mean(),   color="red",    lw=2, ls="--", label=f"Mean {Y1.mean():.3f}")
ax.axvline(Y1.median(), color="orange", lw=2, ls="--", label=f"Median {Y1.median():.3f}")
ax.set_title(f"Raw Y1  (skew={y1_skew:.2f})", fontweight="bold")
ax.set_xlabel("ViolentCrimesPerPop"); ax.set_ylabel("Count")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Histogram log1p
ax = fig.add_subplot(gs[0, 1])
ax.hist(y1_log, bins=50, color=CG, alpha=0.8, edgecolor="white")
ax.axvline(y1_log.mean(), color="red", lw=2, ls="--",
           label=f"Mean {y1_log.mean():.3f}")
ax.set_title(f"log1p(Y1)  (skew={ls:.2f})", fontweight="bold")
ax.set_xlabel("log1p(ViolentCrimesPerPop)"); ax.set_ylabel("Count")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# QQ raw
ax = fig.add_subplot(gs[0, 2])
stats.probplot(Y1, dist="norm", plot=ax)
ax.set_title("Q-Q Plot: Raw Y1", fontweight="bold"); ax.grid(True, alpha=0.3)

# QQ log
ax = fig.add_subplot(gs[1, 0])
stats.probplot(y1_log, dist="norm", plot=ax)
ax.set_title("Q-Q Plot: log1p(Y1)", fontweight="bold"); ax.grid(True, alpha=0.3)

# CDF with threshold
ax = fig.add_subplot(gs[1, 1])
s  = np.sort(Y1); cdf = np.arange(1, len(s)+1)/len(s)
ax.plot(s, cdf, color=CB, lw=2, label="CDF")
thr = float(Y1.quantile(0.85))
ax.axvline(thr, color=CR, lw=2, ls="--",
           label=f"Y2 threshold 85th pct = {thr:.3f}")
ax.fill_between(s, 0, cdf, where=s >= thr, alpha=0.2, color=CR,
                label="Emergency zone (top 15%)")
ax.set_title("CDF + Emergency Threshold", fontweight="bold")
ax.set_xlabel("ViolentCrimesPerPop"); ax.set_ylabel("CDF")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Boxplot
ax = fig.add_subplot(gs[1, 2])
ax.boxplot(Y1, vert=True, patch_artist=True,
           boxprops=dict(facecolor=CB, alpha=0.5),
           medianprops=dict(color="red", lw=2),
           flierprops=dict(marker="o", markerfacecolor=CR,
                           markersize=3, alpha=0.5))
ax.set_ylabel("ViolentCrimesPerPop")
ax.set_title("Boxplot — Outlier View", fontweight="bold")
ax.grid(True, alpha=0.3)
save(fig, "p25_eda_02_y1_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Y2 CLASS BALANCE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 3 — Y2 Class Balance (EmergencyActivation)")
print("─"*65)

vc       = Y2.value_counts()
pct_pos  = Y2.mean() * 100
imbal    = vc[0] / vc[1]

print(f"  Class 0 (no activation): {vc[0]}  ({100-pct_pos:.1f}%)")
print(f"  Class 1 (emergency)    : {vc[1]}  ({pct_pos:.1f}%)")
print(f"  Imbalance ratio        : {imbal:.1f}:1")

if pct_pos < 5:
    fail(f"Severely imbalanced ({pct_pos:.1f}%)",
         "class_weight='balanced' mandatory in all Stage 2 classifiers")
elif pct_pos < 20:
    warn(f"Imbalanced ({pct_pos:.1f}% positive, {imbal:.1f}:1)",
         "Use class_weight='balanced'. Primary metrics: AUC + Recall (not accuracy)")
else:
    ok(f"Class balance acceptable ({pct_pos:.1f}% positive)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Check 3 — Y2 Class Balance (EmergencyActivation)",
             fontsize=13, fontweight="bold")

bars = ax1.bar(["Class 0\n(No Emergency)", "Class 1\n(Emergency)"],
               [vc[0], vc[1]], color=[CB, CR], alpha=0.85,
               edgecolor="white", width=0.5)
for bar, v in zip(bars, [vc[0], vc[1]]):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 15,
             f"{v}\n({v/len(Y2)*100:.1f}%)",
             ha="center", fontsize=11, fontweight="bold")
ax1.set_ylabel("Count")
ax1.set_title(f"Class Counts  (imbalance {imbal:.1f}:1)", fontweight="bold")
ax1.grid(axis="y", alpha=0.3)
ax1.text(0.97, 0.97,
         "Use class_weight='balanced'\n"
         "in ALL Stage 2 classifiers.\n\n"
         "Primary metrics:\n  AUC  (imbalance-robust)\n"
         "  Recall (catches emergencies)",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=CR))

wedges, texts, auts = ax2.pie(
    [vc[0], vc[1]],
    labels=["No Emergency", "Emergency"],
    autopct="%1.1f%%", colors=[CB, CR],
    explode=[0, 0.08], startangle=90,
    textprops={"fontsize": 11})
for a in auts: a.set_fontweight("bold")
ax2.set_title("Class Proportion", fontweight="bold")
plt.tight_layout()
save(fig, "p25_eda_03_y2_class_balance.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 4 — CORRELATION HEATMAP (multicollinearity)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 4 — Feature Correlations (multicollinearity)")
print("─"*65)

corr_y1  = X1.corrwith(Y1).abs().sort_values(ascending=False)
top25    = corr_y1.head(25).index.tolist()
corr_sub = X1[top25].corr()

pairs = []
for i in range(len(top25)):
    for j in range(i+1, len(top25)):
        r = corr_sub.iloc[i, j]
        if abs(r) > 0.85:
            pairs.append((top25[i], top25[j], round(r,3)))

print(f"  Feature pairs with |r| > 0.85: {len(pairs)}")
for f1, f2, r in pairs[:8]:
    print(f"    {f1:<28} ↔ {f2:<28}  r={r}")
if len(pairs) > 8:
    print(f"    ... and {len(pairs)-8} more")

if len(pairs) > 20:
    warn(f"High multicollinearity ({len(pairs)} pairs |r|>0.85)",
         "Ridge L2 penalty directly addresses this. RF/GBM unaffected.")
elif len(pairs) > 5:
    warn(f"Moderate multicollinearity ({len(pairs)} pairs |r|>0.85)",
         "Ridge will likely outperform plain LinearRegression")
else:
    ok(f"Low multicollinearity ({len(pairs)} pairs |r|>0.85)")

fig, ax = plt.subplots(figsize=(16, 13))
im = ax.imshow(corr_sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(top25))); ax.set_yticks(range(len(top25)))
ax.set_xticklabels(top25, rotation=55, ha="right", fontsize=7)
ax.set_yticklabels(top25, fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
for i in range(len(top25)):
    for j in range(len(top25)):
        v = corr_sub.iloc[i, j]
        if i != j and abs(v) > 0.65:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if abs(v) > 0.85 else "black")
ax.set_title(
    f"Check 4 — Correlation Matrix: Top 25 Features by |r| with Y1\n"
    f"({len(pairs)} pairs with |r|>0.85 → validates Ridge in Stage 1 zoo)",
    fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "p25_eda_04_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 5 — TOP FEATURES CORRELATED WITH Y1
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 5 — Top Features Correlated with Y1")
print("─"*65)

corr_signed = X1.corrwith(Y1).sort_values()
top_all     = pd.concat([corr_signed.head(15), corr_signed.tail(15)])
max_r       = corr_y1.iloc[0]

print(f"  Strongest positive : {corr_signed.tail(1).index[0]}  "
      f"r={corr_signed.tail(1).iloc[0]:+.4f}")
print(f"  Strongest negative : {corr_signed.head(1).index[0]}  "
      f"r={corr_signed.head(1).iloc[0]:+.4f}")
print(f"  Max |r| with Y1    : {max_r:.4f}")

if max_r < 0.3:
    warn(f"Weak max correlation with Y1 ({max_r:.3f})",
         "RF/GBM will likely outperform linear models significantly")
elif max_r < 0.6:
    ok(f"Moderate correlations (max |r|={max_r:.3f}) — linear models are viable")
else:
    ok(f"Strong correlations (max |r|={max_r:.3f}) — linear models should perform well")

fig, ax = plt.subplots(figsize=(13, 10))
colors = [CR if r < 0 else CB for r in top_all.values]
bars   = ax.barh(range(len(top_all)), top_all.values,
                 color=colors, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_all)))
ax.set_yticklabels(top_all.index, fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson r with ViolentCrimesPerPop")
ax.set_title(
    "Check 5 — Top 30 Feature Correlations with Y1\n"
    "Blue = positive (more X → more crime)  |  Red = negative (more X → less crime)",
    fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, top_all.values):
    ax.text(val + (0.004 if val >= 0 else -0.004),
            bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=7)
plt.tight_layout()
save(fig, "p25_eda_05_top_features_y1.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 6 — HETEROSCEDASTICITY (Breusch-Pagan)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 6 — Heteroscedasticity — Breusch-Pagan Test")
print("─"*65)

top10 = corr_y1.head(10).index.tolist()
sc    = StandardScaler()
Xb    = sc.fit_transform(X1[top10])
Xbc   = sm.add_constant(Xb)

# Raw Y1
lr1 = LinearRegression().fit(Xb, Y1)
res = Y1 - lr1.predict(Xb)
r2_raw = lr1.score(Xb, Y1)
bp_stat, bp_p, _, _ = het_breuschpagan(res, Xbc)

# log1p Y1
lr2 = LinearRegression().fit(Xb, y1_log)
res_log   = y1_log - lr2.predict(Xb)
r2_log    = lr2.score(Xb, y1_log)
bp2, bp2p, _, _ = het_breuschpagan(res_log, Xbc)

print(f"  LR R² on raw Y1       : {r2_raw:.4f}")
print(f"  Breusch-Pagan p-value : {bp_p:.2e}  "
      f"→ {'HETEROSCEDASTIC' if bp_p < 0.05 else 'homoscedastic'}")
print(f"\n  LR R² on log1p(Y1)    : {r2_log:.4f}")
print(f"  BP p-value (log1p)    : {bp2p:.4f}  "
      f"→ {'still heteroscedastic' if bp2p < 0.05 else 'homoscedastic ✓'}")

if bp_p < 0.05:
    warn(f"Heteroscedasticity confirmed (BP p={bp_p:.2e})",
         "log1p(Y1) applied for LR + Ridge. RF/GBM train on raw Y1 — unaffected.")
else:
    ok(f"Homoscedastic residuals (BP p={bp_p:.4f} > 0.05)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Check 6 — Heteroscedasticity\n"
    f"Raw Y1: BP p={bp_p:.2e}  |  log1p(Y1): BP p={bp2p:.4f}",
    fontsize=13, fontweight="bold")

for row, (residuals, fitted, title, c, pv) in enumerate([
    (res,     lr1.predict(Xb), "Raw Y1",      CO, bp_p),
    (res_log, lr2.predict(Xb), "log1p(Y1)",   CG, bp2p),
]):
    # Residuals vs fitted
    ax = axes[row][0]
    ax.scatter(fitted, residuals, alpha=0.25, s=7,
               color=CR if pv < 0.05 else CG)
    ax.axhline(0, color="black", lw=1.5)
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    verdict = f"BP p={pv:.2e} — {'FAN SHAPE ↑' if pv < 0.05 else 'uniform ✓'}"
    ax.set_title(f"{title}: Residuals vs Fitted\n{verdict}",
                 fontweight="bold",
                 color=CR if pv < 0.05 else CG)
    ax.grid(True, alpha=0.3)

    # Residual distribution
    ax = axes[row][1]
    ax.hist(residuals, bins=45, alpha=0.8, edgecolor="white",
            color=CR if pv < 0.05 else CG)
    ax.axvline(0, color="black", lw=1.5)
    ax.set_xlabel("Residual value"); ax.set_ylabel("Count")
    ax.set_title(f"{title}: Residual Histogram  "
                 f"(skew={residuals.skew():.2f})",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "p25_eda_06_heteroscedasticity.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 7 — OUTLIERS IN Y1 (IQR method)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 7 — Outliers in Y1 (IQR method)")
print("─"*65)

Q1  = Y1.quantile(0.25); Q3 = Y1.quantile(0.75); IQR = Q3 - Q1
lo  = Q1 - 1.5*IQR;      hi = Q3 + 1.5*IQR
out = Y1[(Y1 < lo) | (Y1 > hi)]
pct = len(out)/len(Y1)*100

print(f"  Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}")
print(f"  Lower fence : {lo:.4f}   Upper fence : {hi:.4f}")
print(f"  Outliers    : {len(out)} rows  ({pct:.1f}%)")

if pct > 10:
    warn(f"High outlier rate ({pct:.1f}%)",
         "GBM handles outliers best. MSE loss pulls LR/Ridge toward them.")
elif pct > 5:
    warn(f"Moderate outlier rate ({pct:.1f}%)",
         "Ridge L2 mitigates their effect on linear models.")
else:
    ok(f"Low outlier rate ({pct:.1f}%,  {len(out)} rows)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Check 7 — Y1 Outlier Detection  "
             f"({len(out)} outliers = {pct:.1f}% of rows)",
             fontsize=13, fontweight="bold")

ax1.boxplot(Y1, vert=True, patch_artist=True,
            boxprops=dict(facecolor=CB, alpha=0.5),
            medianprops=dict(color="red", lw=2),
            flierprops=dict(marker="o", markerfacecolor=CR,
                            markersize=4, alpha=0.5))
ax1.set_ylabel("ViolentCrimesPerPop")
ax1.set_title("Boxplot (IQR method)", fontweight="bold")
ax1.grid(True, alpha=0.3)

normal = Y1[(Y1 >= lo) & (Y1 <= hi)]
ax2.hist(normal,  bins=40, color=CB, alpha=0.7, edgecolor="white",
         label=f"Normal ({len(normal)})")
ax2.hist(out,     bins=15, color=CR, alpha=0.85, edgecolor="white",
         label=f"Outliers ({len(out)})")
ax2.axvline(hi, color=CR, lw=2, ls="--", label=f"Upper fence={hi:.3f}")
ax2.set_xlabel("ViolentCrimesPerPop"); ax2.set_ylabel("Count")
ax2.set_title("Normal vs Outlier Distribution", fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, "p25_eda_07_outliers_y1.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 8 — D2 STATE FEATURE DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 8 — D2 State Feature Distributions")
print("─"*65)

d2_skews = {c: X2[c].skew() for c in X2.columns}
for col, sk in d2_skews.items():
    print(f"  {col:<30}  mean={X2[col].mean():.3f}  "
          f"std={X2[col].std():.3f}  skew={sk:+.3f}")

high_sk = [c for c, s in d2_skews.items() if abs(s) > 2.0]
dmin    = X2.min().min(); dmax = X2.max().max()

if high_sk:
    warn(f"D2 skewed features: {high_sk}",
         "GBM/RF handle this natively. LogReg may be marginally affected.")
else:
    ok("D2 features: skew within acceptable range")

if dmin >= -0.01 and dmax <= 1.01:
    ok(f"D2 features in [0,1] range (min={dmin:.4f}, max={dmax:.4f})")
else:
    warn(f"D2 range outside [0,1]: min={dmin:.4f}, max={dmax:.4f}",
         "Re-check min-max normalisation in p25_data_pipeline.py")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes_flat = axes.flatten()
fig.suptitle("Check 8 — D2 State Feature Distributions (1994 communities)",
             fontsize=13, fontweight="bold")

for i, col in enumerate(X2.columns):
    ax = axes_flat[i]
    ax.hist(X2[col], bins=30, color=CP, alpha=0.8, edgecolor="white")
    ax.axvline(X2[col].mean(), color="red", lw=1.5, ls="--",
               label=f"mean={X2[col].mean():.3f}")
    ax.set_title(f"{col}\nskew={X2[col].skew():+.2f}",
                 fontsize=8, fontweight="bold")
    ax.set_xlabel("Normalised [0,1]"); ax.set_ylabel("Count")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

for i in range(len(X2.columns), len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
save(fig, "p25_eda_08_d2_features.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 9 — Y1 BY Y2 CLASS (Stage 2 signal quality)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("  CHECK 9 — Y1 by Y2 Class (validates Stage 2 bridge)")
print("─"*65)

y1c0  = Y1[Y2 == 0]; y1c1 = Y1[Y2 == 1]
tdiff = y1c1.mean() - y1c0.mean()
tstat, tpval = stats.ttest_ind(y1c1, y1c0)
cohend = tdiff / Y1.std()
r_y1y2 = float(Y1.corr(Y2.astype(float)))

print(f"  Y1 mean — Class 0 : {y1c0.mean():.4f}")
print(f"  Y1 mean — Class 1 : {y1c1.mean():.4f}")
print(f"  Mean difference   : {tdiff:+.4f}")
print(f"  Cohen's d         : {cohend:.4f}")
print(f"  t-test p-value    : {tpval:.2e}")
print(f"  Pearson r(Y1,Y2)  : {r_y1y2:.4f}")

if tpval < 0.001 and cohend > 0.8:
    ok(f"Strong class separation (d={cohend:.2f}, p={tpval:.2e})",
       "Y1_hat will be a highly informative bridge feature for Stage 2")
elif tpval < 0.001 and cohend > 0.5:
    ok(f"Good class separation (d={cohend:.2f}, p={tpval:.2e})")
elif tpval < 0.05:
    ok(f"Significant class separation (p={tpval:.2e})")
else:
    warn(f"Weak class separation (d={cohend:.2f})",
         "Stage 2 must rely more heavily on D2 state features")

if abs(r_y1y2) > 0.5:
    ok(f"Strong Y1-Y2 correlation (r={r_y1y2:.3f}) — Y1_hat is a valid bridge")
elif abs(r_y1y2) > 0.3:
    ok(f"Moderate Y1-Y2 correlation (r={r_y1y2:.3f})")
else:
    warn(f"Low Y1-Y2 correlation (r={r_y1y2:.3f})",
         "D2 state features become more critical for Stage 2 quality")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    f"Check 9 — Y1 Distribution by Y2 Class  "
    f"(Cohen's d={cohend:.2f},  p={tpval:.2e},  r(Y1,Y2)={r_y1y2:.3f})",
    fontsize=12, fontweight="bold")

ax = axes[0]
ax.hist(y1c0, bins=40, alpha=0.6, color=CB, edgecolor="white",
        label=f"Class 0 no-emergency (n={len(y1c0)})")
ax.hist(y1c1, bins=40, alpha=0.75, color=CR, edgecolor="white",
        label=f"Class 1 emergency (n={len(y1c1)})")
ax.axvline(y1c0.mean(), color=CB, lw=2, ls="--")
ax.axvline(y1c1.mean(), color=CR, lw=2, ls="--")
ax.set_xlabel("ViolentCrimesPerPop (Y1)"); ax.set_ylabel("Count")
ax.set_title("Y1 by Y2 Class", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
bp_data = [y1c0.values, y1c1.values]
bplot   = ax.boxplot(bp_data, labels=["Class 0\nNo Emergency",
                                       "Class 1\nEmergency"],
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2))
bplot["boxes"][0].set_facecolor(CB); bplot["boxes"][0].set_alpha(0.6)
bplot["boxes"][1].set_facecolor(CR); bplot["boxes"][1].set_alpha(0.6)
ax.set_ylabel("ViolentCrimesPerPop")
ax.set_title("Boxplot — Class Separation", fontweight="bold")
ax.grid(True, alpha=0.3)

ax = axes[2]
d2c0 = X2.loc[Y2 == 0, "state_violent_rate"]
d2c1 = X2.loc[Y2 == 1, "state_violent_rate"]
ax.hist(d2c0, bins=20, alpha=0.6, color=CB, edgecolor="white", label="Class 0")
ax.hist(d2c1, bins=20, alpha=0.75, color=CR, edgecolor="white", label="Class 1")
ax.set_xlabel("state_violent_rate (D2 feature)"); ax.set_ylabel("Count")
ax.set_title("D2 Signal: state_violent_rate by Y2\n"
             "(confirms D2 adds value beyond Y1_hat alone)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, "p25_eda_09_y1_by_y2_class.png")


# ════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ════════════════════════════════════════════════════════════════════════════
n_pass = sum(1 for s,_,_ in CHECKS if s=="PASS")
n_warn = sum(1 for s,_,_ in CHECKS if s=="WARN")
n_fail = sum(1 for s,_,_ in CHECKS if s=="FAIL")

print("\n" + "="*65)
print("  EDA COMPLETE — FINAL VERDICT")
print("="*65)
print(f"\n  Checks run  : {len(CHECKS)}")
print(f"  \033[92mPASS\033[0m        : {n_pass}")
print(f"  \033[93mWARN\033[0m        : {n_warn}")
print(f"  \033[91mFAIL\033[0m        : {n_fail}")

if n_fail > 0:
    verdict = "BLOCKED"
    print(f"\n  \033[91m✗ TRAINING BLOCKED — fix these first:\033[0m")
    for name, detail in ISSUES:
        print(f"    ✗ {name}")
        if detail: print(f"      → {detail}")
else:
    verdict = "APPROVED"
    print(f"\n  \033[92m✓ TRAINING APPROVED\033[0m")
    if WARNINGS:
        print(f"\n  Non-blocking notes for the modelling script:")
        for name, detail in WARNINGS:
            print(f"    ⚠  {name}")
            if detail: print(f"       → {detail}")

# Write summary file
lines = [
    "P25 EDA SUMMARY REPORT",
    "="*60,
    f"Stage 1 input : {df1.shape[0]} rows x {df1.shape[1]} cols",
    f"Stage 2 base  : {df2.shape[0]} rows x {df2.shape[1]} cols",
    "",
    f"{'Status':<6}  Check",
    "-"*60,
]
for status, name, detail in CHECKS:
    lines.append(f"{status:<6}  {name}")
    if detail: lines.append(f"        -> {detail}")

lines += [
    "",
    "KEY FINDINGS:",
    f"  Y1 skew raw          : {y1_skew:.4f}",
    f"  Y1 skew log1p        : {ls:.4f}",
    f"  Heteroscedasticity   : BP p={bp_p:.2e} "
    f"({'confirmed' if bp_p < 0.05 else 'not confirmed'})",
    f"  Y2 class balance     : {pct_pos:.1f}% positive ({imbal:.1f}:1 imbalance)",
    f"  Multicollinear pairs : {len(pairs)} with |r| > 0.85",
    f"  Y1 outliers          : {len(out)} ({pct:.1f}%)",
    f"  r(Y1, Y2)            : {r_y1y2:.4f}",
    f"  Cohen's d (Y1 sep.)  : {cohend:.4f}",
    "",
    "MODELLING IMPLICATIONS:",
    f"  1. log1p(Y1) for LinearRegression + Ridge only",
    f"  2. class_weight='balanced' in ALL Stage 2 classifiers",
    f"  3. Ridge justified by {len(pairs)} multicollinear pairs",
    f"  4. RF/GBM justified by heteroscedasticity + outliers",
    f"  5. Primary metrics: AUC + Recall (NOT accuracy)",
    f"  6. Y1_hat is a valid bridge (r={r_y1y2:.3f}, d={cohend:.2f})",
    "",
    f"TRAINING VERDICT: {verdict}",
]
spath = os.path.join(OUT_DIR, "p25_eda_summary.txt")
with open(spath, "w") as f:
    f.write("\n".join(lines))
print(f"\n  [SAVED] p25_eda_summary.txt")
print(f"\n  All outputs → outputs/p25_outputs/")
print("="*65 + "\n")
