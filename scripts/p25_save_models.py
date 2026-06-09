"""
=============================================================================
P25 — p25_save_models.py    Train and Save All Models to Disk
=============================================================================
Trains all 8 models (4 Stage 1 + 4 Stage 2) and saves them to
models/ using joblib. Also fixes the calibration plot rendering bug.

Run AFTER p25_split.py:
    python scripts/p25_save_models.py

Saves to models/:
    stage1_LinearRegression.joblib
    stage1_Ridge.joblib
    stage1_RandomForest.joblib
    stage1_GradientBoosting.joblib
    stage2_LogisticRegression.joblib
    stage2_DecisionTree.joblib
    stage2_RandomForest.joblib
    stage2_GradientBoosting.joblib
    model_metadata.txt    ← which model won each stage + all CV scores
=============================================================================
"""

import os, sys, time
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR    = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    path = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(path):
        print(f"[ERROR] {name}.npy not found. Run p25_split.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=True)

print("\n" + "="*60)
print("  P25 — Train and Save All Models")
print("="*60)

# ════════════════════════════════════════════════════════════════════════════
# LOAD ARRAYS
# ════════════════════════════════════════════════════════════════════════════
print("\n[LOAD] Reading arrays from data/processed/...")
X1_train     = load("X1_train")
X1_test      = load("X1_test")
Y1_train_raw = load("Y1_train_raw")
Y1_train_log = load("Y1_train_log")
X2_train     = load("X2_train")
X2_test      = load("X2_test")
Y2_train     = load("Y2_train")

sample_w     = compute_sample_weight("balanced", Y2_train)
skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — TRAIN AND SAVE 4 REGRESSORS
# ════════════════════════════════════════════════════════════════════════════
print("\n[STAGE 1] Training and saving 4 regressors...")

S1_CONFIGS = [
    ("LinearRegression", LinearRegression(),                          "log"),
    ("Ridge",            Ridge(alpha=1.0),                            "log"),
    ("RandomForest",     RandomForestRegressor(n_estimators=200,
                          min_samples_split=5, n_jobs=-1,
                          random_state=42),                           "raw"),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200,
                          learning_rate=0.05, max_depth=4,
                          subsample=0.8, random_state=42),            "raw"),
]

s1_results   = []
Y1_hat_train = np.zeros((len(X1_train), 4))
Y1_hat_test  = np.zeros((len(X1_test),  4))

print(f"\n  {'Model':<24} {'Target':>10} {'CV R²':>8} {'Time':>6}  Saved As")
print(f"  {'-'*68}")

for i, (name, model, target) in enumerate(S1_CONFIGS):
    Y_tr  = Y1_train_log if target == "log" else Y1_train_raw
    t0    = time.time()
    cv_r2 = cross_val_score(model, X1_train, Y_tr,
                             cv=5, scoring="r2", n_jobs=-1)
    model.fit(X1_train, Y_tr)
    elapsed = time.time() - t0

    hat_tr = model.predict(X1_train)
    hat_te = model.predict(X1_test)
    if target == "log":
        hat_tr = np.clip(np.expm1(hat_tr), 0, 1)
        hat_te = np.clip(np.expm1(hat_te), 0, 1)
    Y1_hat_train[:, i] = hat_tr
    Y1_hat_test[:,  i] = hat_te

    # Save model to disk
    save_path = os.path.join(MODELS_DIR, f"stage1_{name}.joblib")
    joblib.dump(model, save_path)

    tgt_label = "log1p(Y1)" if target == "log" else "raw Y1   "
    print(f"  {name:<24} {tgt_label:>10} "
          f"{cv_r2.mean():>8.4f} {elapsed:>5.1f}s  "
          f"stage1_{name}.joblib")

    s1_results.append({"name": name, "cv_r2": cv_r2.mean(),
                        "cv_std": cv_r2.std(), "target": target})

# Save updated Y1_hat arrays
np.save(os.path.join(DATA_PROC, "Y1_hat_train.npy"), Y1_hat_train)
np.save(os.path.join(DATA_PROC, "Y1_hat_test.npy"),  Y1_hat_test)

best_s1 = max(s1_results, key=lambda r: r["cv_r2"])
print(f"\n  Best Stage 1: {best_s1['name']}  CV R²={best_s1['cv_r2']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TRAIN AND SAVE 4 CLASSIFIERS
# ════════════════════════════════════════════════════════════════════════════
print("\n[STAGE 2] Building input matrix and training 4 classifiers...")

S2_train = np.hstack([X2_train, Y1_hat_train])
S2_test  = np.hstack([X2_test,  Y1_hat_test])
np.save(os.path.join(DATA_PROC, "stage2_final_test.npy"), S2_test)

S2_CONFIGS = [
    ("LogisticRegression",  LogisticRegression(class_weight="balanced",
                             max_iter=1000, random_state=42),         False),
    ("DecisionTree",        DecisionTreeClassifier(class_weight="balanced",
                             max_depth=6, min_samples_split=10,
                             random_state=42),                        False),
    ("RandomForest",        RandomForestClassifier(n_estimators=200,
                             class_weight="balanced", min_samples_split=5,
                             n_jobs=-1, random_state=42),             False),
    ("GradientBoosting",    GradientBoostingClassifier(n_estimators=200,
                             learning_rate=0.05, max_depth=4,
                             subsample=0.8, random_state=42),         True),
]

s2_results = []
from sklearn.metrics import roc_auc_score

print(f"\n  {'Model':<24} {'CV AUC':>8} {'Time':>6}  Saved As")
print(f"  {'-'*60}")

for name, model, use_sw in S2_CONFIGS:
    t0 = time.time()

    # CV
    if use_sw:
        aucs = []
        for tr_i, va_i in skf.split(S2_train, Y2_train):
            model.fit(S2_train[tr_i], Y2_train[tr_i],
                      sample_weight=sample_w[tr_i])
            aucs.append(roc_auc_score(Y2_train[va_i],
                        model.predict_proba(S2_train[va_i])[:,1]))
        cv_auc = np.array(aucs)
    else:
        cv_auc = cross_val_score(model, S2_train, Y2_train,
                                  cv=skf, scoring="roc_auc", n_jobs=-1)

    # Final fit
    if use_sw:
        model.fit(S2_train, Y2_train, sample_weight=sample_w)
    else:
        model.fit(S2_train, Y2_train)
    elapsed = time.time() - t0

    # Save model
    save_path = os.path.join(MODELS_DIR, f"stage2_{name}.joblib")
    joblib.dump(model, save_path)

    print(f"  {name:<24} {cv_auc.mean():>8.4f} {elapsed:>5.1f}s  "
          f"stage2_{name}.joblib")

    s2_results.append({"name": name, "cv_auc": cv_auc.mean(),
                        "cv_std": cv_auc.std()})

best_s2 = max(s2_results, key=lambda r: r["cv_auc"])
print(f"\n  Best Stage 2: {best_s2['name']}  CV AUC={best_s2['cv_auc']:.4f}")

# Save best model names for inference script
joblib.dump(best_s1["name"], os.path.join(MODELS_DIR, "best_stage1_name.joblib"))
joblib.dump(best_s2["name"], os.path.join(MODELS_DIR, "best_stage2_name.joblib"))

# ════════════════════════════════════════════════════════════════════════════
# SAVE METADATA
# ════════════════════════════════════════════════════════════════════════════
print("\n[SAVE] Writing model_metadata.txt...")

lines = [
    "P25 MODEL METADATA",
    "="*55,
    "Models saved to: models/",
    "",
    "STAGE 1 MODELS (4 regressors):",
    f"  {'Model':<24} {'Target':>10} {'CV R²':>8} {'±std':>7}",
    "-"*52,
]
for r in s1_results:
    tgt = "log1p(Y1)" if r["target"] == "log" else "raw Y1"
    marker = "  ← BEST" if r["name"] == best_s1["name"] else ""
    lines.append(f"  {r['name']:<24} {tgt:>10} "
                 f"{r['cv_r2']:>8.4f} {r['cv_std']:>7.4f}{marker}")

lines += [
    "",
    "STAGE 2 MODELS (4 classifiers):",
    f"  {'Model':<24} {'CV AUC':>8} {'±std':>7}",
    "-"*42,
]
for r in s2_results:
    marker = "  ← BEST" if r["name"] == best_s2["name"] else ""
    lines.append(f"  {r['name']:<24} {r['cv_auc']:>8.4f} "
                 f"{r['cv_std']:>7.4f}{marker}")

lines += [
    "",
    "HOW TO LOAD FOR INFERENCE:",
    "  import joblib",
    "  model = joblib.load('models/stage1_GradientBoosting.joblib')",
    "  y1_hat = model.predict(X_new)",
    "  # or use p25_inference.py for end-to-end prediction",
    "",
    "FILES SAVED:",
]
for name, _, _ in S1_CONFIGS:
    lines.append(f"  models/stage1_{name}.joblib")
for name, _, _ in S2_CONFIGS:
    lines.append(f"  models/stage2_{name}.joblib")
lines += [
    "  models/best_stage1_name.joblib",
    "  models/best_stage2_name.joblib",
]

meta_path = os.path.join(MODELS_DIR, "model_metadata.txt")
with open(meta_path, "w") as f:
    f.write("\n".join(lines))

print("\n  Saved files in models/:")
for f in sorted(os.listdir(MODELS_DIR)):
    sz = os.path.getsize(os.path.join(MODELS_DIR, f))
    print(f"    {f:<42} {sz/1024:>8.1f} KB")

print("\n" + "="*60)
print("  All 8 models saved!")
print()
print("  For inference on new data:")
print("    python scripts/p25_inference.py")
print("="*60 + "\n")
