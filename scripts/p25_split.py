"""
P25 — p25_split.py   Data Split & Preparation
Splits both datasets with same indices, applies transforms,
saves all arrays to data/processed/, builds community_reference.csv.
Run ONCE before any training script.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
DATA_RAW  = os.path.join(ROOT, "data", "raw")
S1_PATH   = os.path.join(DATA_PROC, "stage1_input.csv")
S2_PATH   = os.path.join(DATA_PROC, "stage2_base.csv")

SEED      = 42
TEST_SIZE = 0.20

for path, name in [(S1_PATH,"stage1_input.csv"),(S2_PATH,"stage2_base.csv")]:
    if not os.path.exists(path):
        print(f"[ERROR] {name} not found. Run p25_data_pipeline.py first.")
        sys.exit(1)

print("\n" + "="*60)
print("  P25 — Data Split & Preparation")
print("="*60)

# ── Load ─────────────────────────────────────────────────────────────────────
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
assert len(df1) == len(df2), "Row mismatch!"
print(f"  Row alignment: ✓ both files have {len(df1)} rows")

# ── Split ─────────────────────────────────────────────────────────────────────
print("\n[2/5] Generating train/test split...")
indices = np.arange(len(df1))
train_idx, test_idx = train_test_split(
    indices, test_size=TEST_SIZE, random_state=SEED, stratify=Y2)

print(f"  Total rows   : {len(indices)}")
print(f"  Train rows   : {len(train_idx)}  ({len(train_idx)/len(indices)*100:.1f}%)")
print(f"  Test  rows   : {len(test_idx)}   ({len(test_idx)/len(indices)*100:.1f}%)")

y2_train_pct = Y2.iloc[train_idx].mean() * 100
y2_test_pct  = Y2.iloc[test_idx].mean()  * 100
print(f"  Y2 pos in train: {y2_train_pct:.1f}%  (target: 15.4%)")
print(f"  Y2 pos in test : {y2_test_pct:.1f}%   (target: 15.4%)")
assert abs(y2_train_pct - 15.4) < 2.0
print(f"  Stratification  : ✓ class balance preserved")

# ── Transforms ───────────────────────────────────────────────────────────────
print("\n[3/5] Applying transforms...")

Y1_log = np.log1p(Y1_raw)
print(f"  log1p(Y1): skew {Y1_raw.skew():.4f} → {Y1_log.skew():.4f}")
print(f"  RF/GBM will use raw Y1  (no transform needed)")
print(f"  LR/Ridge  will use log1p(Y1) then predict → expm1 to reverse")

X2_transformed = X2.copy()
X2_transformed["state_murder_rate"] = (
    np.log1p(X2_transformed["state_murder_rate"] * 10) / np.log1p(10))
print(f"  state_murder_rate log1p: skew "
      f"{X2['state_murder_rate'].skew():.2f} → "
      f"{X2_transformed['state_murder_rate'].skew():.2f}")

# ── Split arrays ─────────────────────────────────────────────────────────────
print("\n[4/5] Splitting arrays using same indices...")

X1_train = X1.iloc[train_idx].values.astype(np.float64)
X1_test  = X1.iloc[test_idx].values.astype(np.float64)
Y1_train_raw = Y1_raw.iloc[train_idx].values.astype(np.float64)
Y1_train_log = Y1_log.iloc[train_idx].values.astype(np.float64)
Y1_test_raw  = Y1_raw.iloc[test_idx].values.astype(np.float64)
X2_train = X2_transformed.iloc[train_idx].values.astype(np.float64)
X2_test  = X2_transformed.iloc[test_idx].values.astype(np.float64)
Y2_train = Y2.iloc[train_idx].values.astype(np.int32)
Y2_test  = Y2.iloc[test_idx].values.astype(np.int32)
feat_S1  = np.array(X1.columns.tolist())
feat_S2  = np.array(X2.columns.tolist())

print(f"  X1_train : {X1_train.shape}")
print(f"  X1_test  : {X1_test.shape}")
print(f"  X2_train : {X2_train.shape}")
print(f"  X2_test  : {X2_test.shape}")
assert len(set(train_idx) & set(test_idx)) == 0
print(f"  Index leak check: ✓ zero overlap between train and test")

# ── Save arrays ───────────────────────────────────────────────────────────────
print("\n[5/5] Saving arrays to data/processed/...")
saves = {
    "train_idx":       train_idx,
    "test_idx":        test_idx,
    "X1_train":        X1_train,
    "X1_test":         X1_test,
    "Y1_train_raw":    Y1_train_raw,
    "Y1_train_log":    Y1_train_log,
    "Y1_test_raw":     Y1_test_raw,
    "X2_train":        X2_train,
    "X2_test":         X2_test,
    "Y2_train":        Y2_train,
    "Y2_test":         Y2_test,
    "feature_names_S1": feat_S1,
    "feature_names_S2": feat_S2,
}
for name, arr in saves.items():
    np.save(os.path.join(DATA_PROC, f"{name}.npy"), arr)
    print(f"  Saved: {name+'.npy':<30} shape={np.array(arr).shape}")

print("\n" + "="*60)
print("  Split complete!")
print(f"\n  Summary:")
print(f"    Train : {len(train_idx)} rows  ({y2_train_pct:.1f}% emergency)")
print(f"    Test  : {len(test_idx)} rows   ({y2_test_pct:.1f}% emergency)")
print(f"    S1 features: {X1_train.shape[1]}")
print(f"    S2 features: {X2_train.shape[1]}")
print(f"\n  Next step:")
print(f"    python scripts/p25_train1.py")
print("="*60)

# ════════════════════════════════════════════════════════════════
# COMMUNITY REFERENCE CSV
# Adds community names back alongside features so inference and
# Streamlit can look up communities by name.
# ════════════════════════════════════════════════════════════════
print("\n[EXTRA] Building community_reference.csv...")

fips_map = {
    1:'Alabama',2:'Alaska',4:'Arizona',5:'Arkansas',6:'California',
    8:'Colorado',9:'Connecticut',10:'Delaware',11:'DC',12:'Florida',
    13:'Georgia',15:'Hawaii',16:'Idaho',17:'Illinois',18:'Indiana',
    19:'Iowa',20:'Kansas',21:'Kentucky',22:'Louisiana',23:'Maine',
    24:'Maryland',25:'Massachusetts',26:'Michigan',27:'Minnesota',
    28:'Mississippi',29:'Missouri',30:'Montana',31:'Nebraska',
    32:'Nevada',33:'New Hampshire',34:'New Jersey',35:'New Mexico',
    36:'New York',37:'North Carolina',38:'North Dakota',39:'Ohio',
    40:'Oklahoma',41:'Oregon',42:'Pennsylvania',44:'Rhode Island',
    45:'South Carolina',46:'South Dakota',47:'Tennessee',48:'Texas',
    49:'Utah',50:'Vermont',51:'Virginia',53:'Washington',
    54:'West Virginia',55:'Wisconsin',56:'Wyoming'
}

raw_path   = os.path.join(DATA_RAW, "communities.data")
names_path = os.path.join(DATA_RAW, "communities.names")

if not os.path.exists(raw_path) or not os.path.exists(names_path):
    print("  WARNING: raw data files not found — skipping community_reference.csv")
    print(f"  Expected: {raw_path}")
else:
    # Load raw data to get community names and state FIPS
    raw_df = pd.read_csv(raw_path, header=None)
    raw_cols = []
    with open(names_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("@attribute"):
                raw_cols.append(line.split()[1])
    raw_df.columns = raw_cols[:raw_df.shape[1]]

    # Build community labels
    community_name = (raw_df["communityname"]
                      .str.replace("city","", regex=False)
                      .str.replace("town","", regex=False)
                      .str.replace("township","", regex=False))
    state_name  = raw_df["state"].map(fips_map).fillna("Unknown")
    comm_label  = community_name + ", " + state_name

    # Assemble: label columns + all D1 features + Y1 + all D2 features + Y2
    ref = pd.concat([
        pd.DataFrame({
            "community_name":  community_name,
            "state_fips":      raw_df["state"],
            "state_name":      state_name,
            "community_label": comm_label,
        }),
        X1.reset_index(drop=True),
        pd.DataFrame({"ViolentCrimesPerPop": Y1_raw.values}),
        X2.reset_index(drop=True),         # original X2, not log-transformed
        pd.DataFrame({"EmergencyActivation": Y2.values}),
    ], axis=1)

    ref_path = os.path.join(DATA_PROC, "community_reference.csv")
    ref.to_csv(ref_path, index=False)

    print(f"  community_reference.csv saved — {ref.shape[0]} communities × {ref.shape[1]} columns")
    print(f"  Columns: community_label, state_name, 100 D1 features, Y1, 7 D2 features, Y2")
    print(f"  Sample entries:")
    for label in ref["community_label"].head(4).tolist():
        print(f"    {label}")

print("\n")
