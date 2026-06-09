"""
P25 — p25_train2.py   Stage 2 Training + Model Persistence
Trains 4 classifiers, saves models to models/, writes stage2_cv_metrics.csv
Run after p25_train1.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR    = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    p = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(p):
        print(f"[ERROR] {name}.npy not found. Run p25_train1.py first.")
        sys.exit(1)
    return np.load(p, allow_pickle=True)

print("\n" + "="*62)
print("  P25 — Stage 2: Training + Saving Models")
print("="*62)

# ── Load ─────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading arrays...")
X2_train  = load("X2_train")
X2_test   = load("X2_test")
Y2_train  = load("Y2_train")
Y1_hat_tr = load("Y1_hat_train")
Y1_hat_te = load("Y1_hat_test")

print(f"  X2_train: {X2_train.shape}  |  Y1_hat_tr: {Y1_hat_tr.shape}")
print(f"  Y2 pos rate: {Y2_train.mean()*100:.1f}%  imbalance: "
      f"{(Y2_train==0).sum()/(Y2_train==1).sum():.1f}:1")

# ── Build Stage 2 matrix (Option B — all 4 Y1_hat columns appended) ───────────
print("\n[2/5] Building Stage 2 input matrix (D2 + 4×Y1_hat = 11 features)...")
S2_train = np.hstack([X2_train, Y1_hat_tr])
S2_test  = np.hstack([X2_test,  Y1_hat_te])
print(f"  S2_train: {S2_train.shape}  |  S2_test: {S2_test.shape}")

# Save S2_test for test.py to use (no labels — test.py loads Y2_test separately)
np.save(os.path.join(DATA_PROC, "stage2_final_test.npy"), S2_test)
print(f"  stage2_final_test.npy saved → used by p25_test.py")

# ── Imbalance weights ─────────────────────────────────────────────────────────
sample_w = compute_sample_weight("balanced", Y2_train)
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Model configs ─────────────────────────────────────────────────────────────
# class_weight="balanced" mandatory — 5.5:1 imbalance found in EDA
# GBM: no class_weight param → use sample_weight at fit()
MODELS = [
    {"name": "LogisticRegression",
     "model": LogisticRegression(class_weight="balanced",
                                  max_iter=1000, random_state=42),
     "use_sw": False},
    {"name": "DecisionTree",
     "model": DecisionTreeClassifier(class_weight="balanced",
                                      max_depth=6, min_samples_split=10,
                                      random_state=42),
     "use_sw": False},
    {"name": "RandomForest",
     "model": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      min_samples_split=5, n_jobs=-1,
                                      random_state=42),
     "use_sw": False},
    {"name": "GradientBoosting",
     "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, subsample=0.8,
                                          random_state=42),
     "use_sw": True},
]

# ── Train, save ───────────────────────────────────────────────────────────────
print("\n[3/5] Training (5-fold Stratified CV on train set only)...")
print(f"\n  {'Model':<24} {'CV AUC':>8} {'±std':>7} {'Time':>6}")
print(f"  {'-'*50}")

records = []
for cfg in MODELS:
    t0 = time.time()

    # CV
    if cfg["use_sw"]:
        aucs = []
        for tr_i, va_i in skf.split(S2_train, Y2_train):
            cfg["model"].fit(S2_train[tr_i], Y2_train[tr_i],
                             sample_weight=sample_w[tr_i])
            p = cfg["model"].predict_proba(S2_train[va_i])[:, 1]
            aucs.append(roc_auc_score(Y2_train[va_i], p))
        cv = np.array(aucs)
    else:
        cv = cross_val_score(cfg["model"], S2_train, Y2_train,
                              cv=skf, scoring="roc_auc", n_jobs=-1)

    # Full fit
    if cfg["use_sw"]:
        cfg["model"].fit(S2_train, Y2_train, sample_weight=sample_w)
    else:
        cfg["model"].fit(S2_train, Y2_train)
    elapsed = time.time() - t0

    # Save model
    mp = os.path.join(MODELS_DIR, f"stage2_{cfg['name']}.joblib")
    joblib.dump(cfg["model"], mp)

    print(f"  {cfg['name']:<24} {cv.mean():>8.4f} {cv.std():>7.4f} "
          f"{elapsed:>5.1f}s  → saved")

    records.append({
        "model_name":     cfg["name"],
        "cv_auc_mean":    round(float(cv.mean()), 6),
        "cv_auc_std":     round(float(cv.std()),  6),
        "train_time_sec": round(elapsed, 2),
        "model_path":     f"models/stage2_{cfg['name']}.joblib",
    })

# ── Save CV metrics CSV ───────────────────────────────────────────────────────
print("\n[4/5] Saving stage2_cv_metrics.csv...")
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "stage2_cv_metrics.csv"), index=False)
print(f"\n{df[['model_name','cv_auc_mean','cv_auc_std']].to_string(index=False)}")
print(f"\n  Best by CV AUC: {df.loc[df.cv_auc_mean.idxmax(), 'model_name']}")
print(f"  Test AUC (on held-out set) computed by p25_test.py")

# ── List saved models ─────────────────────────────────────────────────────────
print("\n[5/5] Models in models/:")
for f in sorted(os.listdir(MODELS_DIR)):
    if f.endswith(".joblib"):
        sz = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
        print(f"  {f:<45} {sz:>8.0f} KB")

print("\n" + "="*62)
print("  Stage 2 done — models saved to models/")
print("  Next: python scripts/p25_test.py")
print("="*62 + "\n")
