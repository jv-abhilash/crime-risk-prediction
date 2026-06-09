"""
P25 — p25_train1.py   Stage 1 Training + Model Persistence
Trains 4 regressors, saves models to models/, writes stage1_cv_metrics.csv
Run after p25_split.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR    = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    p = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(p):
        print(f"[ERROR] {name}.npy not found. Run p25_split.py first.")
        sys.exit(1)
    return np.load(p, allow_pickle=True)

print("\n" + "="*62)
print("  P25 — Stage 1: Training + Saving Models")
print("="*62)

# ── Load ─────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading arrays...")
X1_train     = load("X1_train")
X1_test      = load("X1_test")
Y1_train_raw = load("Y1_train_raw")
Y1_train_log = load("Y1_train_log")
print(f"  X1_train: {X1_train.shape}  |  X1_test: {X1_test.shape}")

# ── Model configs ─────────────────────────────────────────────────────────────
# target="log" → train on log1p(Y1), reverse with expm1 before saving hat
# target="raw" → train on raw Y1, RF/GBM immune to skew
MODELS = [
    {"name": "LinearRegression", "model": LinearRegression(),           "target": "log"},
    {"name": "Ridge",            "model": Ridge(alpha=1.0),             "target": "log"},
    {"name": "RandomForest",     "model": RandomForestRegressor(
                                     n_estimators=200, min_samples_split=5,
                                     n_jobs=-1, random_state=42),       "target": "raw"},
    {"name": "GradientBoosting", "model": GradientBoostingRegressor(
                                     n_estimators=200, learning_rate=0.05,
                                     max_depth=4, subsample=0.8,
                                     random_state=42),                  "target": "raw"},
]

# ── Train, save, generate Y1_hat ──────────────────────────────────────────────
print("\n[2/4] Training (5-fold CV on train set only)...")
print(f"\n  {'Model':<22} {'Target':>10} {'CV R²':>8} {'±std':>7} {'Time':>6}")
print(f"  {'-'*57}")

records      = []
Y1_hat_train = np.zeros((len(X1_train), 4))
Y1_hat_test  = np.zeros((len(X1_test),  4))

for i, cfg in enumerate(MODELS):
    Y_tr  = Y1_train_log if cfg["target"] == "log" else Y1_train_raw
    t0    = time.time()
    cv    = cross_val_score(cfg["model"], X1_train, Y_tr,
                            cv=5, scoring="r2", n_jobs=-1)
    cfg["model"].fit(X1_train, Y_tr)
    elapsed = time.time() - t0

    hat_tr = cfg["model"].predict(X1_train)
    hat_te = cfg["model"].predict(X1_test)
    if cfg["target"] == "log":
        hat_tr = np.clip(np.expm1(hat_tr), 0, 1)
        hat_te = np.clip(np.expm1(hat_te), 0, 1)

    Y1_hat_train[:, i] = hat_tr
    Y1_hat_test[:,  i] = hat_te

    # Save model
    mp = os.path.join(MODELS_DIR, f"stage1_{cfg['name']}.joblib")
    joblib.dump(cfg["model"], mp)

    lbl = "log1p(Y1)" if cfg["target"] == "log" else "raw Y1   "
    print(f"  {cfg['name']:<22} {lbl:>10} {cv.mean():>8.4f} "
          f"{cv.std():>7.4f} {elapsed:>5.1f}s  → saved")

    records.append({
        "model_name":     cfg["name"],
        "target":         lbl.strip(),
        "cv_r2_mean":     round(float(cv.mean()), 6),
        "cv_r2_std":      round(float(cv.std()),  6),
        "train_time_sec": round(elapsed, 2),
        "model_path":     f"models/stage1_{cfg['name']}.joblib",
    })

# ── Save Y1_hat arrays ────────────────────────────────────────────────────────
print("\n[3/4] Saving Y1_hat arrays...")
np.save(os.path.join(DATA_PROC, "Y1_hat_train.npy"), Y1_hat_train)
np.save(os.path.join(DATA_PROC, "Y1_hat_test.npy"),  Y1_hat_test)
print(f"  Y1_hat_train: {Y1_hat_train.shape}  |  Y1_hat_test: {Y1_hat_test.shape}")
print(f"  Column order: LinearRegression | Ridge | RandomForest | GradientBoosting")

# ── Save CV metrics CSV ───────────────────────────────────────────────────────
print("\n[4/4] Saving stage1_cv_metrics.csv...")
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "stage1_cv_metrics.csv"), index=False)
print(f"\n{df[['model_name','target','cv_r2_mean','cv_r2_std']].to_string(index=False)}")
print(f"\n  Best by CV R²: {df.loc[df.cv_r2_mean.idxmax(), 'model_name']}")
print(f"  Test R² computed by p25_test.py after loading saved models")

print("\n" + "="*62)
print("  Stage 1 done — models saved to models/")
print("  Next: python scripts/p25_train2.py")
print("="*62 + "\n")
