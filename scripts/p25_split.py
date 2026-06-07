"""
=============================================================================
P25 — p25_split.py    Data Split & Preparation
=============================================================================
Run ONCE before any training script.
Generates the train/test split using the SAME indices for both S1 and S2,
applies all transforms dictated by the EDA, and saves everything to
data/processed/ so every downstream script loads consistent data.

Usage (from project root, venv active):
    python scripts/p25_split.py

Saves to data/processed/:
    train_idx.npy          row indices used for training  (1595 rows)
    test_idx.npy           row indices used for test      (399 rows)
    X1_train.npy           Stage 1 features — train
    X1_test.npy            Stage 1 features — test
    Y1_train_raw.npy       Y1 original [0,1] — for RF/GBM
    Y1_train_log.npy       log1p(Y1) — for LinearRegression/Ridge
    Y1_test_raw.npy        Y1 original — evaluation reference
    X2_train.npy           Stage 2 (D2) features — train
    X2_test.npy            Stage 2 (D2) features — test
    Y2_train.npy           binary emergency label — train
    Y2_test.npy            binary emergency label — test
    feature_names_S1.npy   column names for Stage 1
    feature_names_S2.npy   column names for Stage 2 base
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── paths ───────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
S1_PATH   = os.path.join(DATA_PROC, "stage1_input.csv")
S2_PATH   = os.path.join(DATA_PROC, "stage2_base.csv")

SEED      = 42
TEST_SIZE = 0.20   # 80/20 split

# ── check input files exist ─────────────────────────────────────────────────
for path, name in [(S1_PATH, "stage1_input.csv"), (S2_PATH, "stage2_base.csv")]:
    if not os.path.exists(path):
        print(f"[ERROR] {name} not found at:\n  {path}")
        print("Run p25_data_pipeline.py first.")
        sys.exit(1)

print("\n" + "="*60)
print("  P25 — Data Split & Preparation")
print("="*60)

# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading datasets...")
df1 = pd.read_csv(S1_PATH)
df2 = pd.read_csv(S2_PATH)

X1     = df1.drop("ViolentCrimesPerPop", axis=1)
Y1_raw = df1["ViolentCrimesPerPop"]
X2     = df2.drop("EmergencyActivation", axis=1)
Y2     = df2["EmergencyActivation"]

print(f"  Stage 1 : {df1.shape}  ({X1.shape[1]} features + Y1)")
print(f"  Stage 2 : {df2.shape}  ({X2.shape[1]} features + Y2)")
print(f"  Y1 range: [{Y1_raw.min():.4f}, {Y1_raw.max():.4f}]  skew={Y1_raw.skew():.4f}")
print(f"  Y2 pos  : {Y2.sum()} ({Y2.mean()*100:.1f}%)")

# Verify row alignment (both files must have same number of rows)
assert len(df1) == len(df2), (
    f"Row mismatch: stage1={len(df1)}, stage2={len(df2)}. "
    "Re-run p25_data_pipeline.py")
print(f"  Row alignment: ✓ both files have {len(df1)} rows")

# ════════════════════════════════════════════════════════════════════════════
# GENERATE INDICES — stratified on Y2 so both splits have ~15% emergency
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Generating train/test split...")

indices = np.arange(len(df1))
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=Y2   # ← critical: preserves 15.4% emergency rate in both splits
)

print(f"  Total rows   : {len(indices)}")
print(f"  Train rows   : {len(train_idx)}  ({len(train_idx)/len(indices)*100:.1f}%)")
print(f"  Test  rows   : {len(test_idx)}   ({len(test_idx)/len(indices)*100:.1f}%)")

# Verify class preservation after stratification
y2_train_pct = Y2.iloc[train_idx].mean() * 100
y2_test_pct  = Y2.iloc[test_idx].mean()  * 100
print(f"  Y2 pos in train: {y2_train_pct:.1f}%  (target: 15.4%)")
print(f"  Y2 pos in test : {y2_test_pct:.1f}%   (target: 15.4%)")
assert abs(y2_train_pct - 15.4) < 2.0, "Stratification failed — class ratio skewed"
print(f"  Stratification  : ✓ class balance preserved")

# ════════════════════════════════════════════════════════════════════════════
# APPLY TRANSFORMS
# EDA finding: Y1 skew=1.52. log1p reduces to 1.17.
# Apply log1p ONLY for LinearRegression and Ridge (not RF/GBM).
# Both raw and log versions are saved so training scripts choose correctly.
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Applying transforms...")

# log1p on Y1 — only used by LR and Ridge in Stage 1
Y1_log = np.log1p(Y1_raw)
print(f"  log1p(Y1): skew {Y1_raw.skew():.4f} → {Y1_log.skew():.4f}")
print(f"  RF/GBM will use raw Y1  (no transform needed)")
print(f"  LR/Ridge  will use log1p(Y1) then predict → expm1 to reverse")

# EDA finding: D2 state_murder_rate is heavily skewed (skew=+2.77)
# Apply log1p to murder rate for LogisticRegression benefit
# (RF/GBM handle this natively through tree splits)
X2_transformed = X2.copy()
X2_transformed["state_murder_rate"] = np.log1p(
    X2_transformed["state_murder_rate"] * 10   # scale first since values are in [0,0.3]
) / np.log1p(10)  # re-normalise back to approximate [0,1]
print(f"  state_murder_rate log1p: skew "
      f"{X2['state_murder_rate'].skew():.2f} → "
      f"{X2_transformed['state_murder_rate'].skew():.2f}")

# ════════════════════════════════════════════════════════════════════════════
# SPLIT INTO TRAIN / TEST ARRAYS
# Same indices applied to S1 and S2 — this is the critical constraint
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Splitting arrays using same indices...")

# Stage 1 arrays
X1_train = X1.iloc[train_idx].values.astype(np.float64)
X1_test  = X1.iloc[test_idx].values.astype(np.float64)

Y1_train_raw = Y1_raw.iloc[train_idx].values.astype(np.float64)
Y1_train_log = Y1_log.iloc[train_idx].values.astype(np.float64)
Y1_test_raw  = Y1_raw.iloc[test_idx].values.astype(np.float64)

# Stage 2 arrays (transformed)
X2_train = X2_transformed.iloc[train_idx].values.astype(np.float64)
X2_test  = X2_transformed.iloc[test_idx].values.astype(np.float64)

Y2_train = Y2.iloc[train_idx].values.astype(np.int32)
Y2_test  = Y2.iloc[test_idx].values.astype(np.int32)

# Feature name arrays (for reporting and plots)
feat_names_S1 = np.array(X1.columns.tolist())
feat_names_S2 = np.array(X2.columns.tolist())

print(f"  X1_train : {X1_train.shape}")
print(f"  X1_test  : {X1_test.shape}")
print(f"  X2_train : {X2_train.shape}")
print(f"  X2_test  : {X2_test.shape}")

# Verify no overlap between train and test indices
overlap = set(train_idx) & set(test_idx)
assert len(overlap) == 0, f"Index leak: {len(overlap)} rows appear in both sets"
print(f"  Index leak check: ✓ zero overlap between train and test")

# ════════════════════════════════════════════════════════════════════════════
# SAVE ALL ARRAYS TO DISK
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Saving arrays to data/processed/...")

saves = {
    "train_idx.npy":       train_idx,
    "test_idx.npy":        test_idx,
    "X1_train.npy":        X1_train,
    "X1_test.npy":         X1_test,
    "Y1_train_raw.npy":    Y1_train_raw,
    "Y1_train_log.npy":    Y1_train_log,
    "Y1_test_raw.npy":     Y1_test_raw,
    "X2_train.npy":        X2_train,
    "X2_test.npy":         X2_test,
    "Y2_train.npy":        Y2_train,
    "Y2_test.npy":         Y2_test,
    "feature_names_S1.npy": feat_names_S1,
    "feature_names_S2.npy": feat_names_S2,
}

for filename, array in saves.items():
    path = os.path.join(DATA_PROC, filename)
    np.save(path, array)
    print(f"  Saved: {filename:<30} shape={array.shape}")

print("\n" + "="*60)
print("  Split complete!")
print()
print("  Summary:")
print(f"    Train : {len(train_idx)} rows  ({y2_train_pct:.1f}% emergency)")
print(f"    Test  : {len(test_idx)} rows   ({y2_test_pct:.1f}% emergency)")
print(f"    S1 features: {X1_train.shape[1]}")
print(f"    S2 features: {X2_train.shape[1]}")
print()
print("  Next step:")
print("    python scripts/p25_train1.py")
print("="*60 + "\n")
