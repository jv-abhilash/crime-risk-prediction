"""
=============================================================================
P25 — p25_inference.py    Production Inference on New Data
=============================================================================
Loads the 8 saved models from models/ and runs the full two-stage pipeline
on new community data.

Usage:
    python scripts/p25_inference.py --input new_communities.csv

Input CSV must have the same columns as stage1_input.csv (100 D1 features)
plus the 7 D2 columns (state_violent_rate, state_assault_rate, etc.).

If D2 columns are not in the CSV, the national median from training is used.

Output:
    outputs/p25_outputs/inference_results.csv
    Columns: community_id, Y1_hat_GBM, P_emergency, dispatch_tier
=============================================================================
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR    = os.path.join(ROOT, "outputs", "p25_outputs")

# ── check models exist ───────────────────────────────────────────────────────
required = [
    "stage1_LinearRegression.joblib",
    "stage1_Ridge.joblib",
    "stage1_RandomForest.joblib",
    "stage1_GradientBoosting.joblib",
    "stage2_LogisticRegression.joblib",
    "stage2_DecisionTree.joblib",
    "stage2_RandomForest.joblib",
    "stage2_GradientBoosting.joblib",
]
for f in required:
    if not os.path.exists(os.path.join(MODELS_DIR, f)):
        print(f"[ERROR] {f} not found in models/")
        print("  Run: python scripts/p25_save_models.py  first")
        sys.exit(1)

# ── business thresholds ──────────────────────────────────────────────────────
TIER_L1  = 0.80    # Level-1: Full deployment
TIER_L2  = 0.50    # Level-2: Standby
OPTIMAL_THRESHOLD = 0.61   # Best F1 threshold from evaluate script

def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, name))

def dispatch_tier(p, threshold=OPTIMAL_THRESHOLD):
    if   p >= TIER_L1:     return "Level-1: Full Deployment"
    elif p >= threshold:   return "Level-2: Standby"
    else:                  return "Standard Patrol"

def run_inference(X_d1, X_d2, community_ids=None):
    """
    Full two-stage inference pipeline.

    Args:
        X_d1  : np.ndarray (N, 100) — D1 demographic features
        X_d2  : np.ndarray (N, 7)   — D2 state crime rate features
        community_ids : list of N community identifiers (optional)

    Returns:
        pd.DataFrame with columns:
            community_id, Y1_hat_LinReg, Y1_hat_Ridge, Y1_hat_RF,
            Y1_hat_GBM, P_emergency, dispatch_tier
    """
    N = len(X_d1)
    if community_ids is None:
        community_ids = [f"Community_{i}" for i in range(N)]

    print(f"\n  Running inference on {N} communities...")

    # ── Stage 1: generate 4 Y1_hat predictions ──────────────────────────────
    print("  Stage 1: predicting crime intensity scores...")
    s1_models = {
        "LinReg": ("stage1_LinearRegression.joblib",  "log"),
        "Ridge":  ("stage1_Ridge.joblib",              "log"),
        "RF":     ("stage1_RandomForest.joblib",       "raw"),
        "GBM":    ("stage1_GradientBoosting.joblib",   "raw"),
    }

    Y1_hat = np.zeros((N, 4))
    col_names = []
    for j, (short, (fname, target)) in enumerate(s1_models.items()):
        model = load_model(fname)
        hat   = model.predict(X_d1)
        if target == "log":
            hat = np.clip(np.expm1(hat), 0, 1)
        Y1_hat[:, j] = hat
        col_names.append(f"Y1_hat_{short}")
        print(f"    {short:<8} range=[{hat.min():.3f}, {hat.max():.3f}]  "
              f"mean={hat.mean():.3f}")

    # ── Stacking: append Y1_hat to D2 ───────────────────────────────────────
    S2_input = np.hstack([X_d2, Y1_hat])

    # ── Stage 2: predict emergency probability ──────────────────────────────
    print("\n  Stage 2: predicting emergency activation probability...")
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_stage2_name.joblib"))
    best_model = load_model(f"stage2_{best_name}.joblib")

    P_emergency = best_model.predict_proba(S2_input)[:, 1]
    print(f"    Model   : {best_name}")
    print(f"    P range : [{P_emergency.min():.3f}, {P_emergency.max():.3f}]")
    print(f"    P mean  : {P_emergency.mean():.3f}")

    # ── Apply business threshold ─────────────────────────────────────────────
    tiers  = [dispatch_tier(p) for p in P_emergency]
    from collections import Counter
    tier_counts = Counter(tiers)
    print(f"\n  Dispatch tiers (threshold={OPTIMAL_THRESHOLD}):")
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier:<35} : {count} communities ({count/N*100:.1f}%)")

    # ── Build results DataFrame ──────────────────────────────────────────────
    results = pd.DataFrame({
        "community_id": community_ids,
        **{cn: Y1_hat[:, j] for j, cn in enumerate(col_names)},
        "P_emergency":  P_emergency,
        "dispatch_tier": tiers,
    })

    return results

# ════════════════════════════════════════════════════════════════════════════
# MAIN — demo mode using held-out test set if no input CSV provided
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P25 Inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV with D1 + D2 columns")
    parser.add_argument("--threshold", type=float,
                        default=OPTIMAL_THRESHOLD,
                        help=f"Classification threshold (default={OPTIMAL_THRESHOLD})")
    args = parser.parse_args()

    OPTIMAL_THRESHOLD = args.threshold

    print("\n" + "="*60)
    print("  P25 — Production Inference")
    print("="*60)

    if args.input and os.path.exists(args.input):
        # ── Real input mode ──────────────────────────────────────────────────
        print(f"\n  Input: {args.input}")
        df_new = pd.read_csv(args.input)

        # Load saved feature names to align columns
        feat_s1 = np.load(os.path.join(DATA_PROC, "feature_names_S1.npy"),
                           allow_pickle=True)
        feat_s2 = np.load(os.path.join(DATA_PROC, "feature_names_S2.npy"),
                           allow_pickle=True)

        # Check D1 columns
        missing_d1 = [c for c in feat_s1 if c not in df_new.columns]
        if missing_d1:
            print(f"[ERROR] Missing D1 columns: {missing_d1[:5]}...")
            sys.exit(1)

        X_d1 = df_new[feat_s1].values.astype(np.float64)

        # D2 columns — use training medians if absent
        d2_cols = list(feat_s2)
        if all(c in df_new.columns for c in d2_cols):
            X_d2 = df_new[d2_cols].values.astype(np.float64)
            print("  D2 features found in input CSV")
        else:
            # Load training data medians as fallback
            X2_train = np.load(os.path.join(DATA_PROC, "X2_train.npy"))
            d2_medians = np.median(X2_train, axis=0)
            X_d2 = np.tile(d2_medians, (len(df_new), 1))
            print(f"  D2 features NOT in CSV — using training medians as fallback")

        ids = df_new.index.tolist()

    else:
        # ── Demo mode: use held-out test set ────────────────────────────────
        print("\n  No input CSV provided — running on held-out test set (demo)")
        print("  Usage: python scripts/p25_inference.py --input your_data.csv")

        X_d1 = np.load(os.path.join(DATA_PROC, "X1_test.npy"))
        X_d2 = np.load(os.path.join(DATA_PROC, "X2_test.npy"))

        # Load true labels for demo comparison
        Y2_test = np.load(os.path.join(DATA_PROC, "Y2_test.npy"))
        ids = [f"TestCommunity_{i}" for i in range(len(X_d1))]

        print(f"\n  Demo: {len(X_d1)} held-out test communities")
        print(f"  True emergency rate: {Y2_test.mean()*100:.1f}%")

    # ── Run inference ─────────────────────────────────────────────────────────
    results = run_inference(X_d1, X_d2, community_ids=ids)

    # ── Demo mode: add true label and check accuracy ──────────────────────────
    if not args.input:
        results["true_label"] = Y2_test
        results["correct"]    = (
            (results["P_emergency"] >= OPTIMAL_THRESHOLD).astype(int)
            == Y2_test
        )
        n_correct = results["correct"].sum()
        print(f"\n  Demo accuracy: {n_correct}/{len(results)} "
              f"({n_correct/len(results)*100:.1f}%)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "inference_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  [SAVED] inference_results.csv  ({len(results)} rows)")
    print(f"\n  Preview (first 5):")
    print(results[["community_id", "Y1_hat_GBM", "P_emergency",
                    "dispatch_tier"]].head().to_string(index=False))

    print("\n" + "="*60)
    print("  Inference complete!")
    print(f"\n  Threshold used       : {OPTIMAL_THRESHOLD}")
    print(f"  Level-1 activations  : "
          f"{(results['dispatch_tier']=='Level-1: Full Deployment').sum()}")
    print(f"  Level-2 standby      : "
          f"{(results['dispatch_tier']=='Level-2: Standby').sum()}")
    print(f"  Standard patrol      : "
          f"{(results['dispatch_tier']=='Standard Patrol').sum()}")
    print("="*60 + "\n")
