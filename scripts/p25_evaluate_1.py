"""
=============================================================================
P25 — p25_evaluate.py    Evaluation & Production Layers
=============================================================================
Loads the best Stage 2 model's predictions and runs:
  1. Threshold sweep (0.05 → 0.95) — find optimal F1 and Recall@0.90 thresholds
  2. Calibration reliability diagram — is P=0.7 really 70% accurate?
  3. Dispatch cost simulation — asymmetric FN/FP cost curve
  4. Final tiered dispatch output — Level-1 / Level-2 / Standard

Run AFTER p25_train2.py:
    python scripts/p25_evaluate.py

Saves to outputs/evaluation/ and outputs/reports/:
    p25_threshold_sweep.png
    p25_calibration.png
    p25_dispatch_cost.png
    p25_final_dispatch_tiers.csv
    p25_final_report.txt
=============================================================================
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score, accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight

# ── paths ───────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
EVAL_DIR  = os.path.join(ROOT, "outputs", "evaluation")
REPORT_DIR = os.path.join(ROOT, "outputs", "reports")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CB = "#2E75B6"; CG = "#1E5C2E"; CR = "#9B1C1C"; CO = "#C55A11"; CP = "#5B2C8D"

STATE_NAMES = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas",
    6: "California", 8: "Colorado", 9: "Connecticut", 10: "Delaware",
    11: "District of Columbia", 12: "Florida", 13: "Georgia", 15: "Hawaii",
    16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa",
    20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine",
    24: "Maryland", 25: "Massachusetts", 26: "Michigan", 27: "Minnesota",
    28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska",
    32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico",
    36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio",
    40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island",
    45: "South Carolina", 46: "South Dakota", 47: "Tennessee", 48: "Texas",
    49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
}

def load(name):
    path = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(path):
        print(f"[ERROR] {name}.npy not found. Run previous scripts first.")
        sys.exit(1)
    return np.load(path, allow_pickle=True)

def save_fig(fig, name):
    path = os.path.join(EVAL_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}")

def clean_community_name(name):
    import re
    name = str(name).replace("_", " ").strip()
    for suffix in [
        "city", "township", "town", "borough", "village",
        "municipality", "county", "urbancounty",
    ]:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name else "Unknown"

def load_test_metadata(n_rows):
    meta_path = os.path.join(DATA_PROC, "community_metadata_test.csv")
    if os.path.exists(meta_path):
        metadata = pd.read_csv(meta_path)
        if len(metadata) == n_rows:
            return metadata.reset_index(drop=True)

    raw_path = os.path.join(ROOT, "data", "raw", "communities.data")
    test_idx_path = os.path.join(DATA_PROC, "test_idx.npy")
    if not os.path.exists(raw_path) or not os.path.exists(test_idx_path):
        return pd.DataFrame({
            "row_id": np.arange(n_rows),
            "city": [f"test_row_{i}" for i in range(n_rows)],
            "state": ["Unknown"] * n_rows,
        })

    test_idx = np.load(test_idx_path, allow_pickle=True)
    raw_meta = pd.read_csv(
        raw_path,
        header=None,
        usecols=[0, 3],
        names=["state_code", "community_raw"],
        na_values=["?"],
    )
    metadata = pd.DataFrame({
        "row_id": np.arange(len(raw_meta)),
        "city": raw_meta["community_raw"].map(clean_community_name),
        "state_code": raw_meta["state_code"].astype("Int64"),
    })
    metadata["state"] = metadata["state_code"].map(STATE_NAMES).fillna(
        metadata["state_code"].astype(str)
    )
    metadata = metadata.iloc[test_idx].reset_index(drop=True)
    if len(metadata) != n_rows:
        return pd.DataFrame({
            "row_id": np.arange(n_rows),
            "city": [f"test_row_{i}" for i in range(n_rows)],
            "state": ["Unknown"] * n_rows,
        })
    return metadata

print("\n" + "="*60)
print("  P25 — Evaluation & Production Layers")
print("="*60)

# ════════════════════════════════════════════════════════════════════════════
# LOAD AND REBUILD BEST MODEL
# We retrain the best Stage 2 model to get predictions for evaluation
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading data and rebuilding best Stage 2 model...")

X2_train  = load("X2_train")
X2_test   = load("X2_test")
Y2_train  = load("Y2_train")
Y2_test   = load("Y2_test")
Y1_hat_tr = load("Y1_hat_train")
Y1_hat_te = load("Y1_hat_test")

S2_train = np.hstack([X2_train, Y1_hat_tr])
S2_test  = np.hstack([X2_test,  Y1_hat_te])

# Retrain all 4 classifiers to get all probability arrays
sample_w = compute_sample_weight("balanced", Y2_train)

clf_configs = [
    ("LogisticRegression",
     LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ("DecisionTree",
     DecisionTreeClassifier(class_weight="balanced", max_depth=6,
                             min_samples_split=10, random_state=42)),
    ("RandomForest",
     RandomForestClassifier(n_estimators=200, class_weight="balanced",
                             min_samples_split=5, n_jobs=-1, random_state=42)),
    ("GradientBoosting",
     GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, subsample=0.8, random_state=42)),
]

all_probas = {}
for name, clf in clf_configs:
    if "Gradient" in name:
        clf.fit(S2_train, Y2_train, sample_weight=sample_w)
    else:
        clf.fit(S2_train, Y2_train)
    all_probas[name] = clf.predict_proba(S2_test)[:, 1]
    print(f"  {name}: AUC={roc_auc_score(Y2_test, all_probas[name]):.4f}")

best_name  = max(all_probas, key=lambda k: roc_auc_score(Y2_test, all_probas[k]))
best_proba = all_probas[best_name]
print(f"\n  Best model for evaluation: {best_name}")

# Rebuild Stage 1 test performance to state clearly which Stage 1 model won.
X1_train     = load("X1_train")
X1_test      = load("X1_test")
Y1_train_raw = load("Y1_train_raw")
Y1_train_log = load("Y1_train_log")
Y1_test_raw  = load("Y1_test_raw")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

s1_configs = [
    ("LinearRegression", LinearRegression(), "log"),
    ("Ridge(alpha=1.0)", Ridge(alpha=1.0), "log"),
    ("RandomForest", RandomForestRegressor(
        n_estimators=200, min_samples_split=5, n_jobs=-1, random_state=42
    ), "raw"),
    ("GradientBoosting", GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    ), "raw"),
]
s1_scores = {}
for name, model, target in s1_configs:
    y_tr = Y1_train_log if target == "log" else Y1_train_raw
    model.fit(X1_train, y_tr)
    hat = model.predict(X1_test)
    if target == "log":
        hat = np.clip(np.expm1(hat), 0, 1)
    s1_scores[name] = r2_score(Y1_test_raw, hat)

best_stage1_name = max(s1_scores, key=s1_scores.get)
best_stage1_r2 = s1_scores[best_stage1_name]
print(f"  Best Stage 1 model      : {best_stage1_name} "
      f"(highest Test R² among 4 Stage 1 models)")

# ════════════════════════════════════════════════════════════════════════════
# THRESHOLD SWEEP
# Default threshold is 0.5 but this is not optimal for imbalanced data.
# Sweep 0.05 → 0.95, compute F1 and Recall at each point.
# Find: (a) threshold that maximises F1
#       (b) lowest threshold where Recall >= 0.90
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Threshold sweep...")

thresholds = np.arange(0.05, 0.96, 0.01)
f1s, recs, precs, accs = [], [], [], []

for t in thresholds:
    preds = (best_proba >= t).astype(int)
    f1s.append(f1_score(Y2_test, preds, zero_division=0))
    recs.append(recall_score(Y2_test, preds, zero_division=0))
    precs.append(precision_score(Y2_test, preds, zero_division=0))
    accs.append(accuracy_score(Y2_test, preds))

f1s   = np.array(f1s)
recs  = np.array(recs)
precs = np.array(precs)

# Best F1 threshold
best_f1_idx   = np.argmax(f1s)
best_f1_thr   = thresholds[best_f1_idx]
best_f1_val   = f1s[best_f1_idx]

# Recall >= 0.90 threshold (lowest threshold that catches >= 90% of emergencies)
rec90_idx     = np.where(recs >= 0.90)[0]
rec90_thr     = thresholds[rec90_idx[0]] if len(rec90_idx) > 0 else None
rec90_prec    = precs[rec90_idx[0]] if len(rec90_idx) > 0 else None

print(f"  Best F1 threshold    : {best_f1_thr:.2f}  "
      f"(F1={best_f1_val:.4f}, Recall={recs[best_f1_idx]:.4f})")
if rec90_thr is not None:
    print(f"  Recall≥0.90 threshold: {rec90_thr:.2f}  "
          f"(Recall={recs[rec90_idx[0]]:.4f}, Precision={rec90_prec:.4f})")

# Plot
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(thresholds, f1s,   color=CB, lw=2.5, label="F1 Score")
ax.plot(thresholds, recs,  color=CG, lw=2.5, label="Recall (catches emergencies)")
ax.plot(thresholds, precs, color=CO, lw=2,   label="Precision", alpha=0.8)

ax.axvline(best_f1_thr, color=CB, lw=1.5, ls="--",
           label=f"Best F1 threshold = {best_f1_thr:.2f}  (F1={best_f1_val:.3f})")
if rec90_thr is not None:
    ax.axvline(rec90_thr, color=CG, lw=1.5, ls="--",
               label=f"Recall≥0.90 threshold = {rec90_thr:.2f}")
ax.axhline(0.90, color="gray", lw=1, ls=":", alpha=0.6)

# Business tier thresholds
ax.axvline(0.80, color=CR, lw=1.2, ls=":", alpha=0.7)
ax.axvline(0.50, color=CR, lw=1.2, ls=":", alpha=0.7)
ax.text(0.81, 0.05, "P=0.80\n(L1→L2)", fontsize=8, color=CR)
ax.text(0.51, 0.05, "P=0.50\n(L2→Std)", fontsize=8, color=CR)

ax.set_xlabel("Classification Threshold P", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(
    f"Threshold Sweep — {best_name}\n"
    "Finding the optimal threshold for emergency dispatch",
    fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
save_fig(fig, "p25_threshold_sweep.png")

# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION RELIABILITY DIAGRAM
# Question: when the model says P=0.7, is the true positive rate ~70%?
# A well-calibrated model has a diagonal reliability curve.
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Calibration reliability diagram...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Calibration Reliability Diagrams\n"
             "Diagonal line = perfectly calibrated model",
             fontsize=13, fontweight="bold")

colors = [CB, CG, CO, CP]
for ax in axes:
    ax.plot([0,1],[0,1], color="black", lw=1.5, ls="--",
            label="Perfect calibration", alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

for (name, proba), c in zip(all_probas.items(), colors):
    frac_pos, mean_pred = calibration_curve(Y2_test, proba, n_bins=10)
    axes[0].plot(mean_pred, frac_pos, marker="o", lw=2, color=c,
                 markersize=5, label=name)
    axes[1].hist(proba, bins=30, color=c, alpha=0.35,
                 edgecolor="white", label=name)

axes[0].set_title("Reliability Curves — All 4 Classifiers", fontweight="bold")
axes[0].legend(fontsize=8, loc="upper left")
axes[1].set_title("Predicted Probability Distributions", fontweight="bold")
axes[1].set_xlabel("P(Emergency)")
axes[1].set_ylabel("Count")
axes[1].legend(fontsize=8)
axes[1].axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.6)

note = ("Reading guide:\n"
        "If the blue curve hugs the diagonal:\n"
        "  model is well calibrated\n"
        "If it curves below diagonal:\n"
        "  model is overconfident\n"
        "  (use Platt scaling or Isotonic)")
axes[0].text(0.97, 0.05, note, transform=axes[0].transAxes,
             ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))
save_fig(fig, "p25_calibration.png")

# ════════════════════════════════════════════════════════════════════════════
# DISPATCH COST SIMULATION
# In emergency dispatch, missing an emergency (FN) costs far more than
# a false alarm (FP). We model this asymmetry.
#
# Cost model:
#   FN cost = 10  (missed emergency → delayed response → harm)
#   FP cost = 1   (false alarm → wasted resources → inconvenience)
# Total cost = FN_count × 10 + FP_count × 1
# Find threshold that minimises total cost.
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Dispatch cost simulation...")

FN_COST = 10   # cost of missing one real emergency
FP_COST = 1    # cost of one false alarm

total_costs = []
fn_counts   = []
fp_counts   = []

for t in thresholds:
    preds  = (best_proba >= t).astype(int)
    fn     = int(((Y2_test == 1) & (preds == 0)).sum())  # missed emergencies
    fp     = int(((Y2_test == 0) & (preds == 1)).sum())  # false alarms
    cost   = fn * FN_COST + fp * FP_COST
    total_costs.append(cost)
    fn_counts.append(fn)
    fp_counts.append(fp)

total_costs = np.array(total_costs)
fn_counts   = np.array(fn_counts)
fp_counts   = np.array(fp_counts)

min_cost_idx = np.argmin(total_costs)
min_cost_thr = thresholds[min_cost_idx]
min_cost_val = total_costs[min_cost_idx]

print(f"  FN cost = {FN_COST}  FP cost = {FP_COST}")
print(f"  Min-cost threshold : {min_cost_thr:.2f}")
print(f"  Min total cost     : {min_cost_val}")
print(f"  At that threshold  : {fn_counts[min_cost_idx]} FN + "
      f"{fp_counts[min_cost_idx]} FP")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Dispatch Cost Simulation  (FN penalty = {FN_COST}×,  FP penalty = {FP_COST}×)\n"
    "Lower is better — find the threshold that minimises total dispatch cost",
    fontsize=13, fontweight="bold")

ax1.plot(thresholds, total_costs, color=CR, lw=2.5, label="Total cost")
ax1.plot(thresholds, fn_counts * FN_COST, color="#e74c3c", lw=1.5,
         ls="--", label=f"FN cost (missed emergencies ×{FN_COST})")
ax1.plot(thresholds, fp_counts * FP_COST, color=CB, lw=1.5,
         ls="--", label=f"FP cost (false alarms ×{FP_COST})")
ax1.axvline(min_cost_thr, color="black", lw=2, ls=":",
            label=f"Min-cost threshold = {min_cost_thr:.2f}")
ax1.scatter([min_cost_thr], [min_cost_val], color="gold", s=150,
            zorder=6, edgecolors="black", lw=1.5)
ax1.set_xlabel("Threshold"); ax1.set_ylabel("Cost")
ax1.set_title("Cost Curve vs Threshold", fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2.scatter(fp_counts, fn_counts, c=total_costs, cmap="RdYlGn_r",
            s=30, alpha=0.8)
ax2.scatter(fp_counts[min_cost_idx], fn_counts[min_cost_idx],
            color="gold", s=200, zorder=6, edgecolors="black", lw=1.5,
            label=f"Min cost (thr={min_cost_thr:.2f})")
for i in range(0, len(thresholds), 15):
    ax2.annotate(f"{thresholds[i]:.2f}",
                 (fp_counts[i], fn_counts[i]),
                 fontsize=7, alpha=0.7)
ax2.set_xlabel("False Alarms (FP)"); ax2.set_ylabel("Missed Emergencies (FN)")
ax2.set_title("FP vs FN Tradeoff Space\nColour = total cost", fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.colorbar(ax2.collections[0], ax=ax2, label="Total cost")
save_fig(fig, "p25_dispatch_cost.png")

# ════════════════════════════════════════════════════════════════════════════
# FINAL DISPATCH TIER OUTPUT
# Apply business threshold layer to produce 3-tier dispatch decisions
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Final dispatch tier output...")

TIER_L1  = 0.80   # Level-1 Full deployment
TIER_L2  = 0.50   # Level-2 Standby

dispatch_decisions = []
for p in best_proba:
    if   p >= TIER_L1: dispatch_decisions.append("Level-1: Full Deployment")
    elif p >= TIER_L2: dispatch_decisions.append("Level-2: Standby")
    else:              dispatch_decisions.append("Standard Patrol")

from collections import Counter
dispatch_counts = Counter(dispatch_decisions)
print(f"  Threshold L1 (P≥{TIER_L1}) : {dispatch_counts['Level-1: Full Deployment']:>4} communities")
print(f"  Threshold L2 (P≥{TIER_L2}) : {dispatch_counts['Level-2: Standby']:>4} communities")
print(f"  Standard     (P<{TIER_L2})  : {dispatch_counts['Standard Patrol']:>4} communities")

test_metadata = load_test_metadata(len(Y2_test))
dispatch_table = test_metadata[["city", "state"]].copy()
dispatch_table["p_emergency"] = best_proba
dispatch_table["dispatch_tier"] = dispatch_decisions
dispatch_table["actual_emergency"] = Y2_test
dispatch_table["tier_rank"] = dispatch_table["dispatch_tier"].map({
    "Level-1: Full Deployment": 1,
    "Level-2: Standby": 2,
    "Standard Patrol": 3,
})
dispatch_table = dispatch_table.sort_values(
    ["tier_rank", "p_emergency", "state", "city"],
    ascending=[True, False, True, True],
).drop(columns=["tier_rank"])

dispatch_csv = os.path.join(REPORT_DIR, "p25_final_dispatch_tiers.csv")
dispatch_table.to_csv(dispatch_csv, index=False)
print(f"  [SAVED] p25_final_dispatch_tiers.csv")

# How well does each tier match the actual emergency labels?
best_preds = (best_proba >= best_f1_thr).astype(int)
final_auc  = roc_auc_score(Y2_test, best_proba)
final_f1   = f1_score(Y2_test, best_preds)
final_rec  = recall_score(Y2_test, best_preds)

# ── Save final report ──────────────────────────────────────────────────────
lines = [
    "P25 FINAL EVALUATION REPORT",
    "="*60,
    "MODEL SELECTION USED FOR FINAL OUTPUT:",
    f"  Stage 1 winner     : {best_stage1_name}  "
    f"(highest Test R² = {best_stage1_r2:.4f} among 4 Stage 1 models)",
    f"  Stage 2 winner     : {best_name}  "
    f"(highest Test AUC on held-out test set among 4 Stage 2 models)",
    f"  Final dispatch tiers below are generated from the Stage 2 winner: {best_name}",
    "",
    f"Best Stage 2 model   : {best_name}",
    f"Final AUC            : {final_auc:.4f}",
    f"Final F1             : {final_f1:.4f}  (at threshold {best_f1_thr:.2f})",
    f"Final Recall         : {final_rec:.4f}  "
    f"(catching {final_rec*100:.1f}% of emergencies)",
    "",
    "THRESHOLD ANALYSIS:",
    f"  Best F1 threshold    : {best_f1_thr:.2f}  "
    f"(F1={best_f1_val:.4f}, Recall={recs[best_f1_idx]:.4f})",
]
if rec90_thr:
    lines.append(
        f"  Recall≥0.90 threshold: {rec90_thr:.2f}  "
        f"(Precision={rec90_prec:.4f})")
lines += [
    "",
    "DISPATCH COST MODEL:",
    f"  FN penalty (missed emergency) : {FN_COST}×",
    f"  FP penalty (false alarm)      : {FP_COST}×",
    f"  Min-cost threshold            : {min_cost_thr:.2f}",
    f"  Min total cost                : {min_cost_val}",
    f"  At min-cost: {fn_counts[min_cost_idx]} FN + "
    f"{fp_counts[min_cost_idx]} FP",
    "",
    "DISPATCH TIER DISTRIBUTION (test set, business thresholds 0.80/0.50):",
    f"  Level-1 Full Deployment : "
    f"{dispatch_counts['Level-1: Full Deployment']} communities",
    f"  Level-2 Standby         : "
    f"{dispatch_counts['Level-2: Standby']} communities",
    f"  Standard Patrol         : "
    f"{dispatch_counts['Standard Patrol']} communities",
    "",
    "DISPATCH TIER TABLE (test set):",
    dispatch_table.assign(
        p_emergency=dispatch_table["p_emergency"].map(lambda x: f"{x:.4f}")
    )[["city", "state", "dispatch_tier", "p_emergency",
       "actual_emergency"]].to_string(index=False),
    "",
    "OUTPUTS SAVED:",
    "  p25_threshold_sweep.png",
    "  p25_calibration.png",
    "  p25_dispatch_cost.png",
    "  p25_final_dispatch_tiers.csv",
    "  p25_final_report.txt",
]
report_path = os.path.join(REPORT_DIR, "p25_final_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"  [SAVED] p25_final_report.txt")

print("\n" + "="*60)
print("  Evaluation complete!")
print()
print(f"  Final AUC    : {final_auc:.4f}")
print(f"  Final Recall : {final_rec:.4f}")
print(f"  Best Stage 1 : {best_stage1_name}  R²={best_stage1_r2:.4f}")
print(f"  Best Stage 2 : {best_name}")
print()
print("  Evaluation plots → outputs/evaluation/")
print("  Final reports    → outputs/reports/")
print("="*60 + "\n")
