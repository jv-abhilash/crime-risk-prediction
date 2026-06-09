"""
P25 — p25_inference.py   Production Inference
Reads stage2_test_metrics.csv → finds is_best=True → loads that Stage 2 model
Reads thresholds.csv → uses optimal_f1 threshold (or --policy arg)
Loads all 4 Stage 1 models → runs full two-stage pipeline on new data

Usage:
    python scripts/p25_inference.py                         # demo on test set
    python scripts/p25_inference.py --input data.csv        # new communities
    python scripts/p25_inference.py --policy recall_90      # high-safety mode
    python scripts/p25_inference.py --policy min_cost       # cost-optimal mode
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODELS_DIR= os.path.join(ROOT, "models")
OUT_DIR   = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TIER_L1 = 0.80
TIER_L2 = 0.50

S1_ORDER  = ["LinearRegression","Ridge","RandomForest","GradientBoosting"]
S1_TARGET = {"LinearRegression":"log","Ridge":"log",
             "RandomForest":"raw","GradientBoosting":"raw"}

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="P25 Inference")
parser.add_argument("--input",  type=str, default=None,
                    help="Input CSV (D1 + optional D2 columns)")
parser.add_argument("--policy", type=str, default="optimal_f1",
                    choices=["optimal_f1","recall_90","min_cost"],
                    help="Threshold policy from thresholds.csv (default: optimal_f1)")
parser.add_argument("--output", type=str,
                    default=os.path.join(OUT_DIR, "inference_results.csv"),
                    help="Output CSV path")
args = parser.parse_args()

print("\n" + "="*62)
print("  P25 — Production Inference")
print("="*62)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Read best combination from stage2_test_metrics.csv
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Reading best combination from stage2_test_metrics.csv...")
s2_csv = os.path.join(OUT_DIR, "stage2_test_metrics.csv")
if not os.path.exists(s2_csv):
    print("[ERROR] stage2_test_metrics.csv not found. Run p25_test.py first.")
    sys.exit(1)

s2_df      = pd.read_csv(s2_csv)
best_row   = s2_df[s2_df["is_best"] == True].iloc[0]
best_name  = best_row["model_name"]
best_auc   = best_row["test_auc"]
print(f"  Best Stage 2 model : {best_name}  (AUC={best_auc:.4f})")
print(f"  Stage 1 models     : all 4 (Option B stacking)")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Read threshold from thresholds.csv
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[2/5] Reading threshold (policy='{args.policy}') from thresholds.csv...")
thr_csv = os.path.join(OUT_DIR, "thresholds.csv")
if not os.path.exists(thr_csv):
    print("[ERROR] thresholds.csv not found. Run p25_evaluate.py first.")
    sys.exit(1)

thr_df    = pd.read_csv(thr_csv)
thr_row   = thr_df[thr_df["threshold_type"] == args.policy]
if len(thr_row) == 0:
    print(f"[ERROR] policy '{args.policy}' not found in thresholds.csv")
    print(f"  Available: {thr_df['threshold_type'].tolist()}")
    sys.exit(1)

threshold = float(thr_row.iloc[0]["value"])
print(f"  Threshold type  : {args.policy}")
print(f"  Threshold value : {threshold}")
print(f"  Note            : {thr_row.iloc[0]['note']}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load all 4 Stage 1 models + best Stage 2 model
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5] Loading models from models/...")
s1_models = {}
for name in S1_ORDER:
    mp = os.path.join(MODELS_DIR, f"stage1_{name}.joblib")
    if not os.path.exists(mp):
        print(f"[ERROR] {mp} not found. Run p25_train1.py first.")
        sys.exit(1)
    s1_models[name] = joblib.load(mp)
    print(f"  Loaded: stage1_{name}.joblib")

s2_mp = os.path.join(MODELS_DIR, f"stage2_{best_name}.joblib")
if not os.path.exists(s2_mp):
    print(f"[ERROR] {s2_mp} not found. Run p25_train2.py first.")
    sys.exit(1)
s2_model = joblib.load(s2_mp)
print(f"  Loaded: stage2_{best_name}.joblib")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Prepare input data
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[4/5] Preparing input data...")
feat_s1 = np.load(os.path.join(DATA_PROC, "feature_names_S1.npy"), allow_pickle=True)
feat_s2 = np.load(os.path.join(DATA_PROC, "feature_names_S2.npy"), allow_pickle=True)

if args.input and os.path.exists(args.input):
    df_new = pd.read_csv(args.input)
    missing = [c for c in feat_s1 if c not in df_new.columns]
    if missing:
        print(f"[ERROR] Missing {len(missing)} D1 columns: {missing[:3]} ...")
        sys.exit(1)
    X_d1 = df_new[feat_s1].values.astype(np.float64)
    if all(c in df_new.columns for c in feat_s2):
        X_d2_raw = df_new[list(feat_s2)].values.astype(np.float64)
        print(f"  D2 features found in input CSV")
    else:
        X2_train = np.load(os.path.join(DATA_PROC, "X2_train.npy"))
        X_d2_raw = np.tile(np.median(X2_train, axis=0), (len(df_new), 1))
        print(f"  D2 features missing — using training medians as fallback")
    # Apply log1p to state_murder_rate (same transform as p25_split.py)
    X_d2 = X_d2_raw.copy()
    murder_col = list(feat_s2).index("state_murder_rate")
    X_d2[:, murder_col] = (np.log1p(X_d2_raw[:, murder_col] * 10)
                            / np.log1p(10))
    community_ids = (df_new["community_id"].tolist()
                     if "community_id" in df_new.columns
                     else [f"Row_{i}" for i in range(len(df_new))])
    print(f"  Input: {len(X_d1)} communities from {args.input}")
    Y2_true = None

else:
    # Demo mode — run on held-out test set
    print(f"  No --input provided — demo on held-out test set")
    X_d1    = np.load(os.path.join(DATA_PROC, "X1_test.npy"))
    X_d2    = np.load(os.path.join(DATA_PROC, "X2_test.npy"))   # already transformed
    Y2_true = np.load(os.path.join(DATA_PROC, "Y2_test.npy"))
    community_ids = [f"TestComm_{i}" for i in range(len(X_d1))]
    print(f"  Demo: {len(X_d1)} test communities")
    print(f"  True emergency rate: {Y2_true.mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Run full two-stage pipeline
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[5/5] Running two-stage pipeline...")

# Stage 1: generate Y1_hat from all 4 models
Y1_hat = np.zeros((len(X_d1), 4))
for j, name in enumerate(S1_ORDER):
    hat = s1_models[name].predict(X_d1)
    if S1_TARGET[name] == "log":
        hat = np.clip(np.expm1(hat), 0, 1)
    Y1_hat[:, j] = hat
    print(f"  Stage 1 {name:<22} Y1_hat mean={hat.mean():.4f}")

# Stage 2: classify on D2 + all 4 Y1_hats
S2_input    = np.hstack([X_d2, Y1_hat])
P_emergency = s2_model.predict_proba(S2_input)[:, 1]

print(f"\n  Stage 2 {best_name:<22} P range=[{P_emergency.min():.3f}, "
      f"{P_emergency.max():.3f}]  mean={P_emergency.mean():.3f}")

# Apply business threshold layer
def tier(p):
    if   p >= TIER_L1:    return "Level-1: Full Deployment"
    elif p >= threshold:  return "Level-2: Standby"
    else:                 return "Standard Patrol"

tiers = [tier(p) for p in P_emergency]
from collections import Counter
counts = Counter(tiers)
print(f"\n  Dispatch decisions (threshold={threshold}):")
for t, n in sorted(counts.items()):
    print(f"    {t:<35} {n:>4} communities ({n/len(tiers)*100:.1f}%)")

# Build output DataFrame
result_df = pd.DataFrame({
    "community_id":     community_ids,
    "Y1_hat_LinReg":    Y1_hat[:, 0].round(4),
    "Y1_hat_Ridge":     Y1_hat[:, 1].round(4),
    "Y1_hat_RF":        Y1_hat[:, 2].round(4),
    "Y1_hat_GBM":       Y1_hat[:, 3].round(4),
    "P_emergency":      P_emergency.round(4),
    "dispatch_tier":    tiers,
})

# Demo: add true label and accuracy
if Y2_true is not None:
    from sklearn.metrics import roc_auc_score, recall_score, f1_score
    preds_binary = (P_emergency >= threshold).astype(int)
    result_df["true_emergency"] = Y2_true
    result_df["correct"]        = (preds_binary == Y2_true).astype(int)
    auc  = roc_auc_score(Y2_true, P_emergency)
    rec  = recall_score(Y2_true, preds_binary)
    f1v  = f1_score(Y2_true, preds_binary)
    acc  = result_df["correct"].mean()
    print(f"\n  Demo metrics (threshold={threshold}):")
    print(f"    AUC    : {auc:.4f}")
    print(f"    Recall : {rec:.4f}  (catching {rec*100:.1f}% of real emergencies)")
    print(f"    F1     : {f1v:.4f}")
    print(f"    Accuracy: {acc:.4f}")

# Save output CSV
result_df.to_csv(args.output, index=False)
print(f"\n  [SAVED] {args.output}  ({len(result_df)} rows)")
print(f"\n  Preview (first 5):")
print(result_df[["community_id","Y1_hat_GBM","P_emergency",
                  "dispatch_tier"]].head().to_string(index=False))

print("\n" + "="*62)
print("  Inference complete!")
print(f"  Best model     : {best_name}  (AUC={best_auc:.4f})")
print(f"  Threshold used : {threshold}  (policy={args.policy})")
print(f"  Output         : {args.output}")
print("="*62 + "\n")