"""
P25 — p25_evaluate.py   Production Layers + Threshold CSV
Loads best Stage 2 model from stage2_test_metrics.csv (is_best=True),
runs threshold sweep, calibration, dispatch cost simulation.
Writes thresholds.csv → read by p25_inference.py
Run after p25_test.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score)
warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODELS_DIR= os.path.join(ROOT, "models")
OUT_DIR   = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CB="#2E75B6"; CG="#1E5C2E"; CO="#C55A11"; CR="#9B1C1C"; CP="#5B2C8D"; CGR="#7F7F7F"

def load_arr(name):
    p = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(p):
        print(f"[ERROR] {name}.npy missing. Run p25_test.py first.")
        sys.exit(1)
    return np.load(p, allow_pickle=True)

def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}")

print("\n" + "="*62)
print("  P25 — Production Layers + Threshold CSV")
print("="*62)

# ── Read best combination from stage2_test_metrics.csv ───────────────────────
print("\n[1/5] Reading best model from stage2_test_metrics.csv...")
s2_csv = os.path.join(OUT_DIR, "stage2_test_metrics.csv")
if not os.path.exists(s2_csv):
    print("[ERROR] stage2_test_metrics.csv not found. Run p25_test.py first.")
    sys.exit(1)

s2_df      = pd.read_csv(s2_csv)
best_row   = s2_df[s2_df["is_best"] == True].iloc[0]
best_name  = best_row["model_name"]
print(f"  Best combination: {best_name}  (AUC={best_row['test_auc']:.4f})")

# ── Load best model + test data ───────────────────────────────────────────────
best_model = joblib.load(os.path.join(MODELS_DIR, f"stage2_{best_name}.joblib"))
S2_test    = load_arr("stage2_final_test")
Y2_test    = load_arr("Y2_test")
best_proba = best_model.predict_proba(S2_test)[:, 1]

# All 4 Stage 2 models for calibration plot
S2_NAMES  = ["LogisticRegression","DecisionTree","RandomForest","GradientBoosting"]
S2_COLORS = [CB, CG, CO, CR]
all_probas = {}
for name in S2_NAMES:
    m = joblib.load(os.path.join(MODELS_DIR, f"stage2_{name}.joblib"))
    all_probas[name] = m.predict_proba(S2_test)[:, 1]

print(f"  P(Emergency) range: [{best_proba.min():.3f}, {best_proba.max():.3f}]")

# ════════════════════════════════════════════════════════════════════════════
# THRESHOLD SWEEP
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Threshold sweep (0.05 → 0.95)...")

thresholds = np.arange(0.05, 0.96, 0.01)
f1s, recs, precs = [], [], []
for t in thresholds:
    preds = (best_proba >= t).astype(int)
    f1s.append(f1_score(Y2_test, preds, zero_division=0))
    recs.append(recall_score(Y2_test, preds, zero_division=0))
    precs.append(precision_score(Y2_test, preds, zero_division=0))
f1s  = np.array(f1s)
recs = np.array(recs)
precs= np.array(precs)

best_f1_idx   = int(np.argmax(f1s))
best_f1_thr   = float(thresholds[best_f1_idx])
best_f1_val   = float(f1s[best_f1_idx])
best_f1_rec   = float(recs[best_f1_idx])
best_f1_prec  = float(precs[best_f1_idx])

rec90_idxs    = np.where(recs >= 0.90)[0]
if len(rec90_idxs) > 0:
    rec90_idx  = int(rec90_idxs[0])
    rec90_thr  = float(thresholds[rec90_idx])
    rec90_rec  = float(recs[rec90_idx])
    rec90_prec = float(precs[rec90_idx])
    rec90_f1   = float(f1s[rec90_idx])
else:
    rec90_thr = rec90_rec = rec90_prec = rec90_f1 = None

print(f"  Best F1 threshold    : {best_f1_thr:.2f}  "
      f"(F1={best_f1_val:.4f}, Recall={best_f1_rec:.4f})")
if rec90_thr:
    print(f"  Recall≥0.90 threshold: {rec90_thr:.2f}  "
          f"(Recall={rec90_rec:.4f}, Precision={rec90_prec:.4f})")

# Threshold sweep plot
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(thresholds, f1s,   color=CB, lw=2.5, label="F1 Score")
ax.plot(thresholds, recs,  color=CG, lw=2.5, label="Recall")
ax.plot(thresholds, precs, color=CO, lw=2,   label="Precision", alpha=0.8)
ax.axvline(best_f1_thr, color=CB, lw=1.5, ls="--",
           label=f"Best F1 threshold = {best_f1_thr:.2f}  (F1={best_f1_val:.3f})")
if rec90_thr:
    ax.axvline(rec90_thr, color=CG, lw=1.5, ls="--",
               label=f"Recall≥0.90 threshold = {rec90_thr:.2f}")
ax.axhline(0.90, color="gray", lw=1, ls=":", alpha=0.6)
ax.axvline(0.80, color=CR, lw=1.2, ls=":", alpha=0.6)
ax.axvline(0.50, color=CR, lw=1.2, ls=":", alpha=0.6)
ax.text(0.81, 0.05, "P=0.80\n(L1→L2)", fontsize=8, color=CR)
ax.text(0.51, 0.05, "P=0.50\n(L2→Std)", fontsize=8, color=CR)
ax.set_xlabel("Classification Threshold P", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(f"Threshold Sweep — {best_name}\n"
             "Finding the optimal threshold for emergency dispatch",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
savefig(fig, "p25_threshold_sweep.png")

# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION — fixed: histogram y-axis is COUNT not [0,1]
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Calibration reliability diagram...")

fig, (ax_rel, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Calibration Reliability Diagrams\n"
             "Diagonal = perfectly calibrated", fontsize=13, fontweight="bold")

# Left — reliability curve: both x and y axes are [0,1]
ax_rel.plot([0,1],[0,1], color="black", lw=1.5, ls="--",
            label="Perfect calibration", alpha=0.7)
ax_rel.set_xlabel("Mean Predicted Probability")
ax_rel.set_ylabel("Fraction of Positives")
ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
ax_rel.grid(True, alpha=0.3)

# Right — histogram: x is probability [0,1], y is COUNT (no ylim cap!)
ax_hist.set_xlabel("P(Emergency)")
ax_hist.set_ylabel("Count")
ax_hist.set_xlim(0, 1)   # only constrain x
ax_hist.grid(True, alpha=0.3)

for (name, proba), c in zip(all_probas.items(), S2_COLORS):
    frac_pos, mean_pred = calibration_curve(Y2_test, proba, n_bins=10)
    ax_rel.plot(mean_pred, frac_pos, marker="o", lw=2, color=c,
                markersize=5, label=name)
    ax_hist.hist(proba, bins=30, color=c, alpha=0.45, edgecolor="white", label=name)

ax_rel.set_title("Reliability Curves — All 4 Classifiers", fontweight="bold")
ax_rel.legend(fontsize=8, loc="upper left")
ax_hist.set_title("Predicted Probability Distributions", fontweight="bold")
ax_hist.legend(fontsize=8)
ax_hist.axvline(0.5, color="black", lw=1.5, ls="--", alpha=0.7)
savefig(fig, "p25_calibration.png")

# ════════════════════════════════════════════════════════════════════════════
# DISPATCH COST SIMULATION
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Dispatch cost simulation (FN×10 + FP×1)...")
FN_COST, FP_COST = 10, 1
costs, fns, fps = [], [], []
for t in thresholds:
    preds = (best_proba >= t).astype(int)
    fn = int(((Y2_test==1)&(preds==0)).sum())
    fp = int(((Y2_test==0)&(preds==1)).sum())
    costs.append(fn*FN_COST + fp*FP_COST)
    fns.append(fn); fps.append(fp)
costs = np.array(costs); fns = np.array(fns); fps = np.array(fps)
min_i   = int(np.argmin(costs))
min_thr = float(thresholds[min_i])
min_cost= int(costs[min_i])

print(f"  Min-cost threshold: {min_thr:.2f}  total cost={min_cost}  "
      f"({fns[min_i]} FN + {fps[min_i]} FP)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Dispatch Cost Simulation  (FN penalty = {FN_COST}×,  "
             f"FP penalty = {FP_COST}×)\nLower is better",
             fontsize=13, fontweight="bold")
ax1.plot(thresholds, costs, color=CR, lw=2.5, label="Total cost")
ax1.plot(thresholds, fns*FN_COST, color="#e74c3c", lw=1.5, ls="--",
         label=f"FN cost (×{FN_COST})")
ax1.plot(thresholds, fps*FP_COST, color=CB, lw=1.5, ls="--",
         label=f"FP cost (×{FP_COST})")
ax1.axvline(min_thr, color="black", lw=2, ls=":",
            label=f"Min-cost = {min_thr:.2f}")
ax1.scatter([min_thr],[min_cost], color="gold", s=150, zorder=6,
            edgecolors="black", lw=1.5)
ax1.set_xlabel("Threshold"); ax1.set_ylabel("Cost")
ax1.set_title("Cost Curve vs Threshold", fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

sc = ax2.scatter(fps, fns, c=costs, cmap="RdYlGn_r", s=30, alpha=0.8)
ax2.scatter(fps[min_i], fns[min_i], color="gold", s=200, zorder=6,
            edgecolors="black", lw=1.5, label=f"Min cost (thr={min_thr:.2f})")
for ii in range(0, len(thresholds), 15):
    ax2.annotate(f"{thresholds[ii]:.2f}", (fps[ii], fns[ii]), fontsize=7, alpha=0.7)
ax2.set_xlabel("False Alarms (FP)"); ax2.set_ylabel("Missed Emergencies (FN)")
ax2.set_title("FP vs FN Tradeoff Space\nColour = total cost", fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax2, label="Total cost")
savefig(fig, "p25_dispatch_cost.png")

# ════════════════════════════════════════════════════════════════════════════
# SAVE THRESHOLDS CSV — read by p25_inference.py
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Saving thresholds.csv...")

thr_records = [
    {
        "threshold_type": "optimal_f1",
        "value":          best_f1_thr,
        "f1":             round(best_f1_val,  4),
        "recall":         round(best_f1_rec,  4),
        "precision":      round(best_f1_prec, 4),
        "note":           "Maximises F1 — recommended default for inference",
    },
    {
        "threshold_type": "min_cost",
        "value":          min_thr,
        "f1":             round(float(f1s[min_i]),   4),
        "recall":         round(float(recs[min_i]),  4),
        "precision":      round(float(precs[min_i]), 4),
        "note":           f"Minimises FN×{FN_COST} + FP×{FP_COST} dispatch cost",
    },
]
if rec90_thr:
    thr_records.insert(1, {
        "threshold_type": "recall_90",
        "value":          rec90_thr,
        "f1":             round(rec90_f1,   4),
        "recall":         round(rec90_rec,  4),
        "precision":      round(rec90_prec, 4),
        "note":           "Catches ≥90% of emergencies — high-safety policy",
    })

thr_df = pd.DataFrame(thr_records)
thr_path = os.path.join(OUT_DIR, "thresholds.csv")
thr_df.to_csv(thr_path, index=False)

print(f"\n{thr_df[['threshold_type','value','f1','recall','precision']].to_string(index=False)}")
print(f"\n  Saved: thresholds.csv")
print(f"  p25_inference.py will use 'optimal_f1' row by default")

print("\n" + "="*62)
print("  Evaluate complete")
print("  Next: python scripts/p25_inference.py")
print("="*62 + "\n")
