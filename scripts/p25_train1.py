"""
=============================================================================
P25 — p25_train1.py    Stage 1 Model Zoo (Training Only)
=============================================================================
Trains 4 regression models on Stage 1 demographic features.
Evaluates only on TRAINING data via 5-fold cross-validation.
Saves Y1_hat for both train and test sets (predictions, not evaluation).

Test evaluation happens exclusively in p25_test.py.

Run AFTER p25_split.py:
    python scripts/p25_train1.py

Saves to data/processed/:
    Y1_hat_train.npy   shape (1595, 4) — Stage 2 training input
    Y1_hat_test.npy    shape (399,  4) — Stage 2 test input (for p25_test.py)

Saves to outputs/p25_outputs/:
    p25_stage1_results.txt
=============================================================================
"""

import os, sys, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
OUT_DIR   = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    path = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(path):
        print(f"[ERROR] {name}.npy not found. Run p25_split.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=True)

print("\n" + "="*60)
print("  P25 — Stage 1 Model Zoo (Training Only)")
print("="*60)

# ════════════════════════════════════════════════════════════════════════════
# LOAD TRAINING ARRAYS ONLY
# Y1_test_raw is NOT loaded here — test evaluation is in p25_test.py
# X1_test IS loaded — needed to generate Y1_hat_test for Stage 2 stacking
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Loading training arrays...")

X1_train     = load("X1_train")
X1_test      = load("X1_test")       # loaded for prediction only, not evaluation
Y1_train_raw = load("Y1_train_raw")  # raw Y1 — for RF and GBM
Y1_train_log = load("Y1_train_log")  # log1p(Y1) — for LinearRegression and Ridge

print(f"  X1_train : {X1_train.shape}  ({X1_train.shape[1]} features)")
print(f"  X1_test  : {X1_test.shape}   (loaded for Y1_hat generation only)")
print(f"  Y1 train raw : [{Y1_train_raw.min():.4f}, {Y1_train_raw.max():.4f}]  "
      f"skew={float(np.mean(Y1_train_raw)):.4f}")
print(f"  Y1 train log : [{Y1_train_log.min():.4f}, {Y1_train_log.max():.4f}]")
print(f"\n  NOTE: Y1_test_raw NOT loaded here.")
print(f"  Stage 1 test evaluation → p25_test.py")

# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# EDA decisions embedded in "target" field:
#   "log"  → model trains on log1p(Y1) — fixes skew=1.52 for linear models
#   "raw"  → model trains on raw Y1    — RF/GBM are immune, no transform needed
# After prediction, "log" models run expm1() to return to original [0,1] scale
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Defining models...")

MODELS = [
    {
        "name":   "LinearRegression",
        "short":  "LinReg",
        "model":  LinearRegression(),
        "target": "log",
        "note":   "Baseline. Sensitive to multicollinearity — Ridge expected to win.",
    },
    {
        "name":   "Ridge(alpha=1.0)",
        "short":  "Ridge",
        "model":  Ridge(alpha=1.0),
        "target": "log",
        "note":   "L2 penalty addresses 12 multicollinear pairs found in EDA.",
    },
    {
        "name":   "RandomForest(n=200)",
        "short":  "RF",
        "model":  RandomForestRegressor(
                    n_estimators=200,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                  ),
        "target": "raw",
        "note":   "Immune to skew and heteroscedasticity. Non-linear.",
    },
    {
        "name":   "GradientBoosting(n=200)",
        "short":  "GBM",
        "model":  GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                  ),
        "target": "raw",
        "note":   "Sequential error correction. subsample=0.8 mitigates outliers.",
    },
]

for cfg in MODELS:
    tgt = "log1p(Y1)" if cfg["target"] == "log" else "raw Y1"
    print(f"  {cfg['name']:<24} trains on: {tgt}")

# ════════════════════════════════════════════════════════════════════════════
# TRAIN — 5-fold CV on training set only
# No test set used here. CV gives an unbiased estimate of generalisation.
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Training (5-fold CV on train set only)...")
print()
print(f"  {'Model':<24} {'Target':>8} {'CV R²':>8} {'CV std':>7} {'Time':>6}")
print(f"  {'-'*57}")

results      = []
Y1_hat_train = np.zeros((len(X1_train), 4))
Y1_hat_test  = np.zeros((len(X1_test),  4))

for i, cfg in enumerate(MODELS):
    model  = cfg["model"]
    target = cfg["target"]
    Y_tr   = Y1_train_log if target == "log" else Y1_train_raw

    # 5-fold CV — training data only
    t0    = time.time()
    cv_r2 = cross_val_score(model, X1_train, Y_tr,
                             cv=5, scoring="r2", n_jobs=-1)
    elapsed = time.time() - t0

    # Fit on full training set
    model.fit(X1_train, Y_tr)

    # Generate predictions for BOTH train and test
    # (not for evaluation — for feeding into Stage 2 and p25_test.py)
    hat_tr = model.predict(X1_train)
    hat_te = model.predict(X1_test)

    # Reverse log1p transform for linear models → back to [0,1] scale
    if target == "log":
        hat_tr = np.clip(np.expm1(hat_tr), 0, 1)
        hat_te = np.clip(np.expm1(hat_te), 0, 1)

    Y1_hat_train[:, i] = hat_tr
    Y1_hat_test[:,  i] = hat_te

    tgt_label = "log1p(Y1)" if target == "log" else "raw Y1   "
    print(f"  {cfg['name']:<24} {tgt_label:>8} "
          f"{cv_r2.mean():>8.4f} {cv_r2.std():>7.4f} {elapsed:>5.1f}s")

    results.append({
        "name":    cfg["name"],
        "short":   cfg["short"],
        "target":  target,
        "cv_r2":   cv_r2.mean(),
        "cv_std":  cv_r2.std(),
        "note":    cfg["note"],
        "model":   model,
    })

best = max(results, key=lambda r: r["cv_r2"])
print(f"\n  Best by CV R²  : {best['name']}")
print(f"  Best CV R²     : {best['cv_r2']:.4f} ± {best['cv_std']:.4f}")
print(f"  Note: {best['note']}")

# ════════════════════════════════════════════════════════════════════════════
# SAVE Y1_HAT ARRAYS
# Y1_hat_train → used by p25_train2.py to build Stage 2 training matrix
# Y1_hat_test  → used by p25_test.py to build Stage 2 test matrix
# All 4 columns saved (Option B stacking)
# All values are in original [0,1] crime rate scale regardless of transform
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Saving Y1_hat arrays...")

np.save(os.path.join(DATA_PROC, "Y1_hat_train.npy"), Y1_hat_train)
np.save(os.path.join(DATA_PROC, "Y1_hat_test.npy"),  Y1_hat_test)

print(f"  Y1_hat_train.npy : {Y1_hat_train.shape}  → Stage 2 training input")
print(f"  Y1_hat_test.npy  : {Y1_hat_test.shape}   → Stage 2 test input (p25_test.py)")
print(f"  Column order     : LinReg | Ridge | RF | GBM")
print(f"\n  Y1_hat range check (all should be in [0,1]):")
col_names = ["LinReg", "Ridge ", "RF    ", "GBM   "]
for j, cn in enumerate(col_names):
    col = Y1_hat_test[:, j]
    print(f"    {cn}  min={col.min():.4f}  max={col.max():.4f}  "
          f"mean={col.mean():.4f}")

# Save training report
lines = [
    "P25 STAGE 1 TRAINING REPORT",
    "="*55,
    "Evaluation: 5-fold CV on TRAINING SET only",
    "Test evaluation: see p25_test_report.txt (run p25_test.py)",
    "",
    f"Train rows : {len(X1_train)}",
    f"Features   : {X1_train.shape[1]}",
    "",
    f"{'Model':<24} {'Target':>10} {'CV R²':>8} {'±std':>7}",
    "-"*52,
]
for r in results:
    tgt = "log1p(Y1)" if r["target"] == "log" else "raw Y1"
    lines.append(f"{r['name']:<24} {tgt:>10} "
                 f"{r['cv_r2']:>8.4f} {r['cv_std']:>7.4f}")

lines += [
    "",
    f"BEST MODEL : {best['name']}  CV R² = {best['cv_r2']:.4f}",
    "",
    "EDA DECISIONS APPLIED IN THIS SCRIPT:",
    "  LR + Ridge → log1p(Y1) target  (EDA: skew=1.52, BP p=7.71e-58)",
    "  RF + GBM   → raw Y1 target     (immune to skew, no transform needed)",
    "  log1p reversed via expm1() before saving — all Y1_hat in [0,1]",
    "  Ridge alpha=1.0 → L2 addresses 12 multicollinear pairs (EDA Check 4)",
    "  RF min_samples_split=5 → outlier protection",
    "  GBM subsample=0.8 → reduces outlier influence per tree",
]
rpt = os.path.join(OUT_DIR, "p25_stage1_results.txt")
with open(rpt, "w") as f:
    f.write("\n".join(lines))
print(f"\n  [SAVED] p25_stage1_results.txt")

print("\n" + "="*60)
print("  Stage 1 training complete.")
print()
print("  Next step:")
print("    python scripts/p25_train2.py")
print("="*60 + "\n")
