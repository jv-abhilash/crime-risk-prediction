"""
=============================================================================
P25 — p25_train2.py    Stage 2 Model Zoo (Training Only)
=============================================================================
Builds Stage 2 input matrix: [D2 features (7) | 4×Y1_hat] = 11 columns.
Trains 4 classifiers using training data only.
Cross-validates on training set to select best model.

Test evaluation happens exclusively in p25_test.py.

Run AFTER p25_train1.py:
    python scripts/p25_train2.py

Saves to data/processed/:
    stage2_final_train.npy   (1595, 12) — 11 features + Y2 label
    stage2_final_test.npy    (399, 12)  — built for p25_test.py

Saves to outputs/p25_outputs/:
    p25_stage2_results.txt
=============================================================================
"""

import os, sys, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
OUT_DIR   = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    path = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(path):
        print(f"[ERROR] {name}.npy not found. Run p25_train1.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=True)

print("\n" + "="*60)
print("  P25 — Stage 2 Model Zoo (Training Only)")
print("="*60)

# ════════════════════════════════════════════════════════════════════════════
# LOAD ARRAYS
# Y2_test is NOT loaded — test evaluation belongs in p25_test.py
# X2_test and Y1_hat_te ARE loaded — needed to build stage2_final_test.npy
# which p25_test.py will load for Stage 2 test evaluation
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading arrays...")

X2_train  = load("X2_train")
X2_test   = load("X2_test")      # loaded only to build stage2_final_test
Y2_train  = load("Y2_train")
Y1_hat_tr = load("Y1_hat_train")  # shape (1595, 4)
Y1_hat_te = load("Y1_hat_test")   # shape (399,  4)
feat_s2   = load("feature_names_S2")

col_names_y1hat = ["LinReg_Y1hat", "Ridge_Y1hat", "RF_Y1hat", "GBM_Y1hat"]

print(f"  X2_train  : {X2_train.shape}   (D2 features, train)")
print(f"  Y1_hat_tr : {Y1_hat_tr.shape}  (4 Stage 1 predictions, train)")
print(f"  Y2_train  : pos={Y2_train.sum()} ({Y2_train.mean()*100:.1f}%)")
print(f"\n  NOTE: Y2_test NOT loaded here.")
print(f"  Stage 2 test evaluation → p25_test.py")

# ════════════════════════════════════════════════════════════════════════════
# BUILD STAGE 2 MATRICES
# Training matrix: used to train the classifiers
# Test matrix: saved for p25_test.py — NOT evaluated here
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Building Stage 2 input matrices...")

# Stacking layer: append all 4 Y1_hat columns to D2 features (Option B)
S2_train = np.hstack([X2_train, Y1_hat_tr])   # (1595, 11)
S2_test  = np.hstack([X2_test,  Y1_hat_te])   # (399,  11)
all_feat = list(feat_s2) + col_names_y1hat

print(f"  Stage 2 train : {S2_train.shape}  "
      f"(7 D2 + 4 Y1_hat = 11 features)")
print(f"  Stage 2 test  : {S2_test.shape}   "
      f"(built for p25_test.py, not evaluated here)")
print(f"  Feature order : {all_feat}")

# Save final matrices (include Y2 label as last column for reference)
np.save(os.path.join(DATA_PROC, "stage2_final_train.npy"),
        np.hstack([S2_train, Y2_train.reshape(-1, 1)]))
np.save(os.path.join(DATA_PROC, "stage2_final_test.npy"),
        S2_test)   # no label here — p25_test.py loads Y2_test separately
print(f"  stage2_final_train.npy saved")
print(f"  stage2_final_test.npy  saved")

# ════════════════════════════════════════════════════════════════════════════
# CLASS IMBALANCE SETUP
# EDA Check 3: 5.5:1 imbalance (84.6% vs 15.4%)
# Without handling: classifier predicts No Emergency always → Recall ≈ 0%
# Fix:
#   LogReg / DTree / RF → class_weight="balanced" at model definition
#   GBM                 → sample_weight passed at fit() time
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Preparing class balance weights...")

imbalance = (Y2_train == 0).sum() / (Y2_train == 1).sum()
print(f"  Train imbalance  : {imbalance:.1f}:1")
print(f"  Minority (emerg) : {(Y2_train==1).sum()} rows")
print(f"  Majority (normal): {(Y2_train==0).sum()} rows")

sample_w = compute_sample_weight("balanced", Y2_train)
print(f"  Sample weights   : min={sample_w.min():.4f}  max={sample_w.max():.4f}")
print(f"  → Emergency rows get weight ≈{sample_w[Y2_train==1].mean():.2f}")
print(f"    Normal rows get weight    ≈{sample_w[Y2_train==0].mean():.2f}")

# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Training Stage 2 classifiers...")

S2_MODELS = [
    {
        "name":  "LogisticRegression",
        "short": "LogReg",
        "model": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                 ),
        "use_sample_w": False,
        "note": "Linear boundary. Baseline classifier for Stage 2.",
    },
    {
        "name":  "DecisionTree",
        "short": "DTree",
        "model": DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=6,
                    min_samples_split=10,
                    random_state=42
                 ),
        "use_sample_w": False,
        "note": "Produces IF-THEN dispatch rules. Most interpretable.",
    },
    {
        "name":  "RandomForest(n=200)",
        "short": "RF",
        "model": RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                 ),
        "use_sample_w": False,
        "note": "Bagging reduces variance. Robust to noisy D2 features.",
    },
    {
        "name":  "GradientBoosting(n=200)",
        "short": "GBM",
        "model": GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                 ),
        "use_sample_w": True,   # GBM has no class_weight param → use sample_weight
        "note": "Best for rare event detection. sample_weight at fit().",
    },
]

# Stratified CV — preserves class ratio in every fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print()
print(f"  {'Model':<24} {'CV AUC':>8} {'CV std':>7} {'Time':>6}")
print(f"  {'-'*49}")

results = []
for cfg in S2_MODELS:
    model = cfg["model"]
    t0    = time.time()

    # CV AUC on training set only
    if cfg["use_sample_w"]:
        # For GBM we cannot pass sample_weight to cross_val_score directly
        # Run manual CV
        cv_aucs = []
        from sklearn.metrics import roc_auc_score
        for tr_idx, va_idx in skf.split(S2_train, Y2_train):
            sw_fold = sample_w[tr_idx]
            model.fit(S2_train[tr_idx], Y2_train[tr_idx],
                      sample_weight=sw_fold)
            proba = model.predict_proba(S2_train[va_idx])[:, 1]
            cv_aucs.append(roc_auc_score(Y2_train[va_idx], proba))
        cv_auc = np.array(cv_aucs)
    else:
        cv_auc = cross_val_score(model, S2_train, Y2_train,
                                  cv=skf, scoring="roc_auc", n_jobs=-1)

    elapsed = time.time() - t0

    # Fit on full training set
    if cfg["use_sample_w"]:
        model.fit(S2_train, Y2_train, sample_weight=sample_w)
    else:
        model.fit(S2_train, Y2_train)

    print(f"  {cfg['name']:<24} {cv_auc.mean():>8.4f} "
          f"{cv_auc.std():>7.4f} {elapsed:>5.1f}s")

    results.append({
        "name":    cfg["name"],
        "short":   cfg["short"],
        "cv_auc":  cv_auc.mean(),
        "cv_std":  cv_auc.std(),
        "model":   model,
        "note":    cfg["note"],
    })

best = max(results, key=lambda r: r["cv_auc"])
print(f"\n  Best by CV AUC : {best['name']}")
print(f"  Best CV AUC    : {best['cv_auc']:.4f} ± {best['cv_std']:.4f}")
print(f"  Note: {best['note']}")
print(f"\n  Test AUC, F1, Recall, confusion matrix → run p25_test.py")

# ════════════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Saving training report...")

lines = [
    "P25 STAGE 2 TRAINING REPORT",
    "="*55,
    "Evaluation: 5-fold Stratified CV on TRAINING SET only",
    "Test evaluation: see p25_test_report.txt (run p25_test.py)",
    "",
    f"Train rows  : {len(S2_train)}",
    f"Features    : 7 D2 state features + 4 Y1_hat = 11 total",
    f"Y2 pos rate : {Y2_train.mean()*100:.1f}%  imbalance={imbalance:.1f}:1",
    "",
    "STAGE 2 FEATURE MATRIX COLUMNS:",
]
for i, fn in enumerate(all_feat):
    src = "D2 state feature" if i < len(feat_s2) else "Stage 1 Y1_hat bridge"
    lines.append(f"  {i+1:>2}. {fn:<25} ← {src}")

lines += [
    "",
    f"{'Model':<24} {'CV AUC':>8} {'±std':>7}",
    "-"*42,
]
for r in results:
    lines.append(f"{r['name']:<24} {r['cv_auc']:>8.4f} {r['cv_std']:>7.4f}")

lines += [
    "",
    f"BEST MODEL : {best['name']}  CV AUC = {best['cv_auc']:.4f}",
    "",
    "EDA DECISIONS APPLIED IN THIS SCRIPT:",
    "  class_weight='balanced' → LogReg, DTree, RF  "
    "(EDA: 5.5:1 imbalance, Recall would be ~0% without this)",
    "  sample_weight → GBM.fit()  "
    "(GBM has no class_weight param, same effect)",
    "  Stratified K-Fold → CV preserves 15.4% emergency rate in every fold",
    "  state_murder_rate log1p → applied in p25_split.py  (skew 2.77→fixed)",
    "  All 4 Y1_hat columns appended (Option B stacking)",
    "    Disagreement between models is informative signal for Stage 2",
]

rpt = os.path.join(OUT_DIR, "p25_stage2_results.txt")
with open(rpt, "w") as f:
    f.write("\n".join(lines))
print(f"  [SAVED] p25_stage2_results.txt")

print("\n" + "="*60)
print("  Stage 2 training complete.")
print()
print("  Next step:")
print("    python scripts/p25_test.py")
print("="*60 + "\n")
