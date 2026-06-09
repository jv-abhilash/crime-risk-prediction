"""
P25 — p25_inference.py   Production Inference (CLI + Streamlit-ready)
=============================================================================
HOW INPUT VALUES REACH STAGE 1 AND STAGE 2:

  For every community in community_reference.csv:
    Stage 1 INPUT  → 100 D1 demographic feature columns
                     (already normalised [0,1] from UCI dataset)
                     e.g. PctIlleg=0.241, PctKids2Par=0.563 ...
    Stage 1 OUTPUT → 4 Y1_hat values (computed, not user-provided)
                     LinReg=0.31, Ridge=0.29, RF=0.33, GBM=0.34

    Stage 2 INPUT  → 7 D2 state crime rate columns (from same CSV row)
                     + 4 Y1_hat values from Stage 1 (auto-appended)
                     Total: 11 features. User never provides these manually.
    Stage 2 OUTPUT → P(Emergency) → dispatch tier decision

  User only needs to SELECT a community (or upload their own CSV).
  Everything else is computed automatically.

=============================================================================
CLI usage:
    python scripts/p25_inference.py                           # all test communities
    python scripts/p25_inference.py --community "Selma, California"
    python scripts/p25_inference.py --input new_data.csv
    python scripts/p25_inference.py --policy recall_90

Streamlit usage (import this file):
    from scripts.p25_inference import P25Pipeline
    pipe = P25Pipeline()                 # loads models once
    result = pipe.predict_community("Selma, California")
    communities = pipe.get_community_list()
=============================================================================
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR    = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Business tiers — fixed, independent of classification threshold
TIER_L1 = 0.80   # P >= 0.80 → Level-1: Full Deployment
TIER_L2 = 0.50   # P >= 0.50 → Level-2: Standby

S1_TRANSFORM = {"LinearRegression": "log", "Ridge": "log",
                "RandomForest": "raw", "GradientBoosting": "raw"}


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS — importable by Streamlit
# ════════════════════════════════════════════════════════════════════════════
class P25Pipeline:
    """
    Loads all models and data once at startup.
    Call predict_community() or predict_dataframe() for inference.
    """

    def __init__(self, policy="optimal_f1"):
        self.policy    = policy
        self.threshold = None
        self.s1_models = {}
        self.s2_model  = None
        self.s2_name   = None
        self.feat_s1   = None
        self.feat_s2   = None
        self.ref_df    = None
        self._load_config()
        self._load_models()
        self._load_reference()

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    def _load_config(self):
        # stage1_test_metrics.csv → Stage 1 model paths
        s1_csv = os.path.join(OUT_DIR, "stage1_test_metrics.csv")
        if not os.path.exists(s1_csv):
            raise FileNotFoundError(
                "stage1_test_metrics.csv not found. Run p25_test.py first.")
        self.s1_df = pd.read_csv(s1_csv)

        # stage2_test_metrics.csv → best Stage 2 model path (is_best=True)
        s2_csv = os.path.join(OUT_DIR, "stage2_test_metrics.csv")
        if not os.path.exists(s2_csv):
            raise FileNotFoundError(
                "stage2_test_metrics.csv not found. Run p25_test.py first.")
        self.s2_df = pd.read_csv(s2_csv)
        best_row   = self.s2_df[self.s2_df["is_best"] == True].iloc[0]
        self.s2_name = best_row["model_name"]
        self.s2_auc  = best_row["test_auc"]

        # thresholds.csv → classification threshold for chosen policy
        thr_csv = os.path.join(OUT_DIR, "thresholds.csv")
        if not os.path.exists(thr_csv):
            raise FileNotFoundError(
                "thresholds.csv not found. Run p25_evaluate.py first.")
        thr_df   = pd.read_csv(thr_csv)
        row      = thr_df[thr_df["threshold_type"] == self.policy]
        if len(row) == 0:
            raise ValueError(f"Policy '{self.policy}' not in thresholds.csv")
        self.threshold     = float(row.iloc[0]["value"])
        self.threshold_f1  = float(row.iloc[0]["f1"])
        self.threshold_rec = float(row.iloc[0]["recall"])

    # ── Load models ──────────────────────────────────────────────────────────
    def _load_models(self):
        # Stage 1: load all 4 models using paths from stage1_test_metrics.csv
        for _, row in self.s1_df.iterrows():
            name  = row["model_name"]
            mpath = os.path.join(ROOT, row["model_path"])
            self.s1_models[name] = joblib.load(mpath)

        # Stage 2: load best model using path from stage2_test_metrics.csv
        best_row = self.s2_df[self.s2_df["is_best"] == True].iloc[0]
        mpath    = os.path.join(ROOT, best_row["model_path"])
        self.s2_model = joblib.load(mpath)

        # Feature name arrays
        self.feat_s1 = np.load(os.path.join(DATA_PROC, "feature_names_S1.npy"),
                                allow_pickle=True)
        self.feat_s2 = np.load(os.path.join(DATA_PROC, "feature_names_S2.npy"),
                                allow_pickle=True)

    # ── Load community reference data ─────────────────────────────────────────
    def _load_reference(self):
        ref_path = os.path.join(DATA_PROC, "community_reference.csv")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                "community_reference.csv not found. Run p25_split.py first.")
        self.ref_df = pd.read_csv(ref_path)

    # ── Community list for Streamlit dropdown ─────────────────────────────────
    def get_community_list(self):
        """Returns list of 'CommunityName, StateName' strings for dropdown."""
        return self.ref_df["community_label"].tolist()

    # ── Predict single community by name ──────────────────────────────────────
    def predict_community(self, community_label):
        """
        Takes a community_label string (e.g. 'Selma, California').
        Returns a dict with Y1_hat values, P_emergency, dispatch_tier,
        feature values, and the true label if available.
        """
        row = self.ref_df[self.ref_df["community_label"] == community_label]
        if len(row) == 0:
            raise ValueError(f"Community '{community_label}' not found.")
        row = row.iloc[0]
        return self._predict_row(row, community_label)

    # ── Predict on uploaded CSV ────────────────────────────────────────────────
    def predict_dataframe(self, df_input):
        """
        Takes a DataFrame that must contain the 100 D1 feature columns.
        D2 columns are optional (training medians used if absent).
        Returns a results DataFrame.
        """
        # Validate D1 features
        missing = [c for c in self.feat_s1 if c not in df_input.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} required D1 columns missing: {missing[:5]}...")

        X_d1 = df_input[list(self.feat_s1)].values.astype(np.float64)

        # D2 features
        if all(c in df_input.columns for c in self.feat_s2):
            X_d2 = df_input[list(self.feat_s2)].values.astype(np.float64)
        else:
            X2_train = np.load(os.path.join(DATA_PROC, "X2_train.npy"))
            X_d2     = np.tile(np.median(X2_train, axis=0),
                               (len(df_input), 1))

        # Apply log1p to state_murder_rate
        X_d2 = X_d2.copy()
        m_idx = list(self.feat_s2).index("state_murder_rate")
        X_d2[:, m_idx] = np.log1p(X_d2[:, m_idx] * 10) / np.log1p(10)

        Y1_hat, P_emerg = self._run_pipeline(X_d1, X_d2)

        ids = (df_input["community_id"].tolist()
               if "community_id" in df_input.columns
               else [f"Row_{i}" for i in range(len(df_input))])

        return pd.DataFrame({
            "community_id":   ids,
            "Y1_hat_LinReg":  Y1_hat[:, 0].round(4),
            "Y1_hat_Ridge":   Y1_hat[:, 1].round(4),
            "Y1_hat_RF":      Y1_hat[:, 2].round(4),
            "Y1_hat_GBM":     Y1_hat[:, 3].round(4),
            "P_emergency":    P_emerg.round(4),
            "dispatch_tier":  [self._tier(p) for p in P_emerg],
        })

    # ── Predict all test communities ──────────────────────────────────────────
    def predict_test_set(self):
        """Runs inference on the held-out test set. Returns DataFrame."""
        X_d1    = np.load(os.path.join(DATA_PROC, "X1_test.npy"))
        X_d2    = np.load(os.path.join(DATA_PROC, "X2_test.npy"))
        Y2_true = np.load(os.path.join(DATA_PROC, "Y2_test.npy"))
        test_idx= np.load(os.path.join(DATA_PROC, "test_idx.npy"))

        Y1_hat, P_emerg = self._run_pipeline(X_d1, X_d2)

        labels = self.ref_df.iloc[test_idx]["community_label"].tolist()
        return pd.DataFrame({
            "community_label":  labels,
            "Y1_hat_LinReg":    Y1_hat[:, 0].round(4),
            "Y1_hat_Ridge":     Y1_hat[:, 1].round(4),
            "Y1_hat_RF":        Y1_hat[:, 2].round(4),
            "Y1_hat_GBM":       Y1_hat[:, 3].round(4),
            "P_emergency":      P_emerg.round(4),
            "dispatch_tier":    [self._tier(p) for p in P_emerg],
            "true_emergency":   Y2_true,
        })

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _predict_row(self, row, label):
        """Run pipeline on one DataFrame row."""
        # D1 values for Stage 1
        X_d1 = row[list(self.feat_s1)].values.astype(np.float64).reshape(1, -1)

        # D2 values for Stage 2 (already in reference CSV, already normalised)
        X_d2 = row[list(self.feat_s2)].values.astype(np.float64).reshape(1, -1)
        # Apply log1p to murder rate (same as split script)
        X_d2 = X_d2.copy()
        m_idx = list(self.feat_s2).index("state_murder_rate")
        X_d2[0, m_idx] = np.log1p(X_d2[0, m_idx] * 10) / np.log1p(10)

        Y1_hat, P_emerg = self._run_pipeline(X_d1, X_d2)

        return {
            "community":      label,
            "state":          row.get("state_name", ""),
            # Stage 1 outputs (crime intensity scores)
            "Y1_hat_LinReg":  round(float(Y1_hat[0, 0]), 4),
            "Y1_hat_Ridge":   round(float(Y1_hat[0, 1]), 4),
            "Y1_hat_RF":      round(float(Y1_hat[0, 2]), 4),
            "Y1_hat_GBM":     round(float(Y1_hat[0, 3]), 4),
            # Stage 2 output (emergency probability)
            "P_emergency":    round(float(P_emerg[0]), 4),
            "dispatch_tier":  self._tier(float(P_emerg[0])),
            # Key feature values shown in Streamlit UI
            "top_features": {
                "PctIlleg":       round(float(row.get("PctIlleg", 0)), 4),
                "PctKids2Par":    round(float(row.get("PctKids2Par", 0)), 4),
                "medIncome":      round(float(row.get("medIncome", 0)), 4),
                "PctUnemployed":  round(float(row.get("PctUnemployed", 0)), 4),
                "racepctblack":   round(float(row.get("racepctblack", 0)), 4),
            },
            "d2_features": {c: round(float(row.get(c, 0)), 4)
                            for c in self.feat_s2},
            "true_emergency": int(row.get("EmergencyActivation", -1)),
            "true_y1":        round(float(row.get("ViolentCrimesPerPop", 0)), 4),
            "threshold":      self.threshold,
            "policy":         self.policy,
        }

    def _run_pipeline(self, X_d1, X_d2):
        """Core pipeline: D1 → Stage 1 → Y1_hat → append D2 → Stage 2 → P."""
        n = len(X_d1)
        Y1_hat = np.zeros((n, 4))

        for j, (_, row) in enumerate(self.s1_df.iterrows()):
            name   = row["model_name"]
            model  = self.s1_models[name]
            hat    = model.predict(X_d1)
            if S1_TRANSFORM.get(name, "raw") == "log":
                hat = np.clip(np.expm1(hat), 0, 1)
            Y1_hat[:, j] = hat

        S2_input    = np.hstack([X_d2, Y1_hat])
        P_emergency = self.s2_model.predict_proba(S2_input)[:, 1]
        return Y1_hat, P_emergency

    def _tier(self, p):
        if   p >= TIER_L1:    return "Level-1: Full Deployment"
        elif p >= self.threshold: return "Level-2: Standby"
        else:                 return "Standard Patrol"

    def info(self):
        print(f"\n  Best Stage 2 : {self.s2_name}  (AUC={self.s2_auc:.4f})")
        print(f"  Threshold    : {self.threshold}  (policy={self.policy})")
        print(f"  Communities  : {len(self.ref_df)}")
        print(f"  Stage 1 models:")
        for _, r in self.s1_df.iterrows():
            print(f"    {r['model_name']:<24} Test R²={r['test_r2']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# CLI — runs when called directly from terminal
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P25 Inference")
    parser.add_argument("--community", type=str, default=None,
                        help="Community label, e.g. 'Selma, California'")
    parser.add_argument("--input",     type=str, default=None,
                        help="CSV with D1 feature columns for new communities")
    parser.add_argument("--policy",    type=str, default="optimal_f1",
                        choices=["optimal_f1","recall_90","min_cost"])
    parser.add_argument("--output",    type=str,
                        default=os.path.join(OUT_DIR,"inference_results.csv"))
    parser.add_argument("--list",      action="store_true",
                        help="Print all community names and exit")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  P25 — Production Inference")
    print("="*65)

    print(f"\nLoading pipeline (policy={args.policy})...")
    pipe = P25Pipeline(policy=args.policy)
    pipe.info()

    if args.list:
        print(f"\nAll {len(pipe.get_community_list())} communities:")
        for c in pipe.get_community_list():
            print(f"  {c}")
        sys.exit(0)

    if args.community:
        # ── Single community mode ─────────────────────────────────────────
        print(f"\nPredicting: {args.community}")
        r = pipe.predict_community(args.community)
        print(f"\n{'='*55}")
        print(f"  Community     : {r['community']}")
        print(f"  State         : {r['state']}")
        print(f"\n  Stage 1 — Crime Intensity Scores (Y1_hat):")
        print(f"    LinearRegression : {r['Y1_hat_LinReg']}")
        print(f"    Ridge            : {r['Y1_hat_Ridge']}")
        print(f"    RandomForest     : {r['Y1_hat_RF']}")
        print(f"    GradientBoosting : {r['Y1_hat_GBM']}")
        print(f"\n  Stage 2 — Emergency Activation:")
        print(f"    P(Emergency)     : {r['P_emergency']}")
        print(f"    Threshold        : {r['threshold']}  (policy={r['policy']})")
        print(f"    ┌─────────────────────────────────────────────┐")
        print(f"    │  DISPATCH DECISION: {r['dispatch_tier']:<24} │")
        print(f"    └─────────────────────────────────────────────┘")
        if r['true_emergency'] >= 0:
            correct = (r['P_emergency'] >= r['threshold']) == bool(r['true_emergency'])
            print(f"\n  True label       : {'Emergency' if r['true_emergency'] else 'No Emergency'}")
            print(f"  True Y1          : {r['true_y1']}")
            print(f"  Prediction       : {'✓ Correct' if correct else '✗ Incorrect'}")
        print(f"\n  Key community features (D1):")
        for feat, val in r['top_features'].items():
            print(f"    {feat:<20} : {val}")
        print(f"\n  State context features (D2):")
        for feat, val in r['d2_features'].items():
            print(f"    {feat:<25} : {val}")
        print(f"{'='*55}\n")

    elif args.input:
        # ── CSV upload mode ───────────────────────────────────────────────
        print(f"\nLoading input: {args.input}")
        df_input = pd.read_csv(args.input)
        print(f"  Rows: {len(df_input)}")
        results = pipe.predict_dataframe(df_input)
        results.to_csv(args.output, index=False)
        print(f"\nResults saved: {args.output}")
        print(results[["community_id","Y1_hat_GBM",
                        "P_emergency","dispatch_tier"]].to_string(index=False))

    else:
        # ── Test set mode (default) ───────────────────────────────────────
        print(f"\nRunning on held-out test set...")
        results = pipe.predict_test_set()
        results.to_csv(args.output, index=False)

        from sklearn.metrics import roc_auc_score, recall_score, f1_score
        binary = (results["P_emergency"] >= pipe.threshold).astype(int)
        print(f"\n  Test set metrics (threshold={pipe.threshold}):")
        print(f"    AUC      : {roc_auc_score(results['true_emergency'], results['P_emergency']):.4f}")
        print(f"    Recall   : {recall_score(results['true_emergency'], binary):.4f}")
        print(f"    F1       : {f1_score(results['true_emergency'], binary):.4f}")

        from collections import Counter
        counts = Counter(results["dispatch_tier"])
        print(f"\n  Dispatch decisions:")
        for t in ["Level-1: Full Deployment","Level-2: Standby","Standard Patrol"]:
            n = counts.get(t, 0)
            print(f"    {t:<35} {n:>4} ({n/len(results)*100:.1f}%)")

        print(f"\n  Results saved: {args.output}")
        print(f"\n  Preview (first 5):")
        print(results[["community_label","Y1_hat_GBM",
                        "P_emergency","dispatch_tier"]].head().to_string(index=False))

    print("\n  Done.\n")
