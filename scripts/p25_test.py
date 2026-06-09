"""
P25 — p25_test.py   Comprehensive Test Evaluation
Loads all 8 saved models, evaluates on held-out test set,
writes stage1_test_metrics.csv + stage2_test_metrics.csv,
marks is_best=True on best Stage 2 model (by AUC) for inference to use.
Run after p25_train2.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score,
    accuracy_score, mean_squared_error, r2_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODELS_DIR= os.path.join(ROOT, "models")
OUT_DIR   = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CB="#2E75B6"; CG="#1E5C2E"; CO="#C55A11"; CR="#9B1C1C"; CP="#5B2C8D"; CGR="#7F7F7F"

def load_arr(name):
    p = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(p):
        print(f"[ERROR] {name}.npy missing. Run p25_train2.py first.")
        sys.exit(1)
    return np.load(p, allow_pickle=True)

def load_model(fname):
    p = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(p):
        print(f"[ERROR] {fname} missing. Run p25_train1.py / p25_train2.py first.")
        sys.exit(1)
    return joblib.load(p)

def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}")

print("\n" + "="*62)
print("  P25 — Test Evaluation (loading saved models)")
print("="*62)

# ── Load arrays ───────────────────────────────────────────────────────────────
print("\n[LOAD] Arrays from data/processed/...")
X1_train     = load_arr("X1_train")
X1_test      = load_arr("X1_test")
Y1_train_raw = load_arr("Y1_train_raw")
Y1_test_raw  = load_arr("Y1_test_raw")
X2_train     = load_arr("X2_train")
X2_test      = load_arr("X2_test")
Y2_train     = load_arr("Y2_train")
Y2_test      = load_arr("Y2_test")
Y1_hat_tr    = load_arr("Y1_hat_train")
Y1_hat_te    = load_arr("Y1_hat_test")
feat_s1      = load_arr("feature_names_S1")
feat_s2      = load_arr("feature_names_S2")

S2_test = load_arr("stage2_final_test")
col_names_y1hat = ["LinReg_Y1hat","Ridge_Y1hat","RF_Y1hat","GBM_Y1hat"]
all_s2_feat = list(feat_s2) + col_names_y1hat

print(f"  Test set: {len(Y2_test)} rows  |  Emergency rate: {Y2_test.mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 TEST EVALUATION — load saved models, predict on X1_test
# ════════════════════════════════════════════════════════════════════════════
print("\n[STAGE 1] Loading saved models and evaluating on test set...")

S1_NAMES   = ["LinearRegression","Ridge","RandomForest","GradientBoosting"]
S1_TARGETS = {"LinearRegression":"log","Ridge":"log",
              "RandomForest":"raw","GradientBoosting":"raw"}
S1_COLORS  = [CB, CG, CO, CP]

s1_records = []
s1_hats    = {}

print(f"\n  {'Model':<22} {'Test R²':>8} {'RMSE':>8}")
print(f"  {'-'*42}")

for name in S1_NAMES:
    model  = load_model(f"stage1_{name}.joblib")
    hat    = model.predict(X1_test)
    if S1_TARGETS[name] == "log":
        hat = np.clip(np.expm1(hat), 0, 1)
    test_r2   = r2_score(Y1_test_raw, hat)
    test_rmse = mean_squared_error(Y1_test_raw, hat) ** 0.5
    s1_hats[name] = hat

    print(f"  {name:<22} {test_r2:>8.4f} {test_rmse:>8.4f}")
    s1_records.append({
        "model_name":  name,
        "target":      S1_TARGETS[name],
        "test_r2":     round(test_r2, 6),
        "test_rmse":   round(test_rmse, 6),
        "model_path":  f"models/stage1_{name}.joblib",
    })

# Save Stage 1 test metrics CSV
s1_df = pd.DataFrame(s1_records)
s1_csv = os.path.join(OUT_DIR, "stage1_test_metrics.csv")
s1_df.to_csv(s1_csv, index=False)
print(f"\n  Saved: stage1_test_metrics.csv")

# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 TEST EVALUATION — load saved models, predict on S2_test
# ════════════════════════════════════════════════════════════════════════════
print("\n[STAGE 2] Loading saved models and evaluating on test set...")

S2_NAMES  = ["LogisticRegression","DecisionTree","RandomForest","GradientBoosting"]
S2_COLORS = [CB, CG, CO, CR]

s2_records = []
s2_probas  = {}

print(f"\n  {'Model':<24} {'AUC':>7} {'F1':>7} {'Recall':>8} {'Precis':>8} {'Acc':>7}")
print(f"  {'-'*63}")

for name in S2_NAMES:
    model = load_model(f"stage2_{name}.joblib")
    proba = model.predict_proba(S2_test)[:, 1]
    preds = model.predict(S2_test)
    auc   = roc_auc_score(Y2_test, proba)
    f1    = f1_score(Y2_test, preds)
    rec   = recall_score(Y2_test, preds)
    prec  = precision_score(Y2_test, preds, zero_division=0)
    acc   = accuracy_score(Y2_test, preds)
    s2_probas[name] = proba

    print(f"  {name:<24} {auc:>7.4f} {f1:>7.4f} {rec:>8.4f} "
          f"{prec:>8.4f} {acc:>7.4f}")
    s2_records.append({
        "model_name":  name,
        "test_auc":    round(auc,  6),
        "test_f1":     round(f1,   6),
        "test_recall": round(rec,  6),
        "test_precision": round(prec, 6),
        "test_accuracy":  round(acc,  6),
        "model_path":  f"models/stage2_{name}.joblib",
        "is_best":     False,
    })

# Mark best combination — best Stage 2 by AUC (Stage 1 is always all 4 models)
s2_df = pd.DataFrame(s2_records)
best_idx = s2_df["test_auc"].idxmax()
s2_df.loc[best_idx, "is_best"] = True
best_s2_name = s2_df.loc[best_idx, "model_name"]
best_s2_auc  = s2_df.loc[best_idx, "test_auc"]

# Save Stage 2 test metrics CSV
s2_csv = os.path.join(OUT_DIR, "stage2_test_metrics.csv")
s2_df.to_csv(s2_csv, index=False)
print(f"\n  Best combination: {best_s2_name}  (AUC={best_s2_auc:.4f})")
print(f"  is_best=True written → inference.py will load this model")
print(f"  Saved: stage2_test_metrics.csv")

# Baseline: single LogReg on raw D1 features
sample_w = compute_sample_weight("balanced", Y2_train)
bl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
bl.fit(X1_train, Y2_train)
bl_proba = bl.predict_proba(X1_test)[:, 1]
bl_preds = bl.predict(X1_test)
bl_auc   = roc_auc_score(Y2_test, bl_proba)
bl_rec   = recall_score(Y2_test, bl_preds)
print(f"\n  Baseline (D1 only): AUC={bl_auc:.4f}  Recall={bl_rec:.4f}")
print(f"  Two-stage lift    : AUC {best_s2_auc - bl_auc:+.4f}")

# Best model objects for plots
best_s2_model = load_model(f"stage2_{best_s2_name}.joblib")
best_proba    = s2_probas[best_s2_name]
best_preds    = best_s2_model.predict(S2_test)

# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════
print("\n[PLOTS] Generating test evaluation plots...")

# ── Plot 1: Stage 1 actual vs predicted ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Stage 1 — Actual vs Predicted Y1 (held-out test set, N=399)",
             fontsize=13, fontweight="bold")
for ax, name, c in zip(axes.flatten(), S1_NAMES, S1_COLORS):
    hat = s1_hats[name]
    r2  = r2_score(Y1_test_raw, hat)
    rmse= mean_squared_error(Y1_test_raw, hat)**0.5
    ax.scatter(Y1_test_raw, hat, alpha=0.35, s=12, color=c)
    ax.plot([0,1],[0,1], "k--", lw=1.5, label="Perfect")
    resid = Y1_test_raw - hat
    ax.text(0.02, 0.97,
            f"R²={r2:.4f}  RMSE={rmse:.4f}\n"
            f"Resid std={resid.std():.4f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax.set_title(f"{name}", fontweight="bold", color=c)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 1.1)
plt.tight_layout()
savefig(fig, "p25_test_stage1_scatter.png")

# ── Plot 2: Stage 1 feature importance (RF + GBM) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("Stage 1 — Top 20 Feature Importances",
             fontsize=13, fontweight="bold")
for ax, name, c in zip(axes, ["RandomForest","GradientBoosting"], [CO, CP]):
    model = load_model(f"stage1_{name}.joblib")
    imp   = model.feature_importances_
    idx   = np.argsort(imp)[::-1][:20]
    ax.barh(range(20), imp[idx][::-1], color=c, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(20))
    ax.set_yticklabels([feat_s1[i] for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"{name}  (R²={r2_score(Y1_test_raw, s1_hats[name]):.4f})",
                 fontweight="bold", color=c)
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
savefig(fig, "p25_test_stage1_importance.png")

# ── Plot 3: Stage 2 confusion matrices ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle("Stage 2 — Confusion Matrices (test set N=399, Emergency=15.4%)",
             fontsize=13, fontweight="bold")
for ax, name, c in zip(axes.flatten(), S2_NAMES, S2_COLORS):
    model = load_model(f"stage2_{name}.joblib")
    preds = model.predict(S2_test)
    cm    = confusion_matrix(Y2_test, preds)
    auc   = s2_df.loc[s2_df.model_name==name,"test_auc"].values[0]
    f1v   = s2_df.loc[s2_df.model_name==name,"test_f1"].values[0]
    recv  = s2_df.loc[s2_df.model_name==name,"test_recall"].values[0]
    best_mark = " ★" if name == best_s2_name else ""
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    labels = ["No Emerg.\n(Class 0)","Emergency\n(Class 1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i,j]
            ax.text(j, i, f"{v}\n({v/total*100:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"{name}{best_mark}\nAUC={auc:.3f}  F1={f1v:.3f}  Recall={recv:.3f}",
                 fontweight="bold", color=c)
plt.tight_layout()
savefig(fig, "p25_test_stage2_confusion.png")

# ── Plot 4: ROC + PR curves ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Stage 2 — ROC & Precision-Recall Curves", fontsize=13, fontweight="bold")
ax1.plot([0,1],[0,1], color="gray", lw=1, ls="--", label="Random (AUC=0.50)")
for name, c in zip(S2_NAMES, S2_COLORS):
    fpr, tpr, _ = roc_curve(Y2_test, s2_probas[name])
    auc = s2_df.loc[s2_df.model_name==name,"test_auc"].values[0]
    lw  = 3 if name == best_s2_name else 1.8
    ax1.plot(fpr, tpr, color=c, lw=lw,
             label=f"{name}  AUC={auc:.4f}"+ (" ★" if name==best_s2_name else ""))
fpr_b, tpr_b, _ = roc_curve(Y2_test, bl_proba)
ax1.plot(fpr_b, tpr_b, color=CGR, lw=1.5, ls=":", label=f"Baseline AUC={bl_auc:.4f}")
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curves", fontweight="bold")
ax1.legend(fontsize=8, loc="lower right"); ax1.grid(True, alpha=0.3)

ax2.axhline(Y2_test.mean(), color="gray", lw=1, ls="--",
            label=f"No-skill (P={Y2_test.mean():.2f})")
for name, c in zip(S2_NAMES, S2_COLORS):
    prec_c, rec_c, _ = precision_recall_curve(Y2_test, s2_probas[name])
    ap = average_precision_score(Y2_test, s2_probas[name])
    lw = 3 if name == best_s2_name else 1.8
    ax2.plot(rec_c, prec_c, color=c, lw=lw,
             label=f"{name}  AP={ap:.4f}"+ (" ★" if name==best_s2_name else ""))
prec_b, rec_b, _ = precision_recall_curve(Y2_test, bl_proba)
ap_b = average_precision_score(Y2_test, bl_proba)
ax2.plot(rec_b, prec_b, color=CGR, lw=1.5, ls=":", label=f"Baseline AP={ap_b:.4f}")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves", fontweight="bold")
ax2.legend(fontsize=8, loc="upper right"); ax2.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, "p25_test_stage2_roc.png")

# ── Plot 5: Pipeline comparison ───────────────────────────────────────────────
metrics   = ["AUC","F1","Recall","Precision","Accuracy"]
bl_vals   = [bl_auc,
             f1_score(Y2_test, bl_preds),
             recall_score(Y2_test, bl_preds),
             precision_score(Y2_test, bl_preds, zero_division=0),
             accuracy_score(Y2_test, bl_preds)]
best_vals = [best_s2_auc,
             s2_df.loc[best_idx,"test_f1"],
             s2_df.loc[best_idx,"test_recall"],
             s2_df.loc[best_idx,"test_precision"],
             s2_df.loc[best_idx,"test_accuracy"]]
x = np.arange(len(metrics)); w = 0.35
fig, ax = plt.subplots(figsize=(13, 7))
b1 = ax.bar(x-w/2, bl_vals,   w, color=CGR, alpha=0.85, edgecolor="white",
            label="Baseline: single LogReg on D1 only")
b2 = ax.bar(x+w/2, best_vals, w, color=CB,  alpha=0.85, edgecolor="white",
            label=f"Two-Stage Pipeline: {best_s2_name} ★")
for bar, v in zip(list(b1)+list(b2), bl_vals+list(best_vals)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
for i, (bv, tv) in enumerate(zip(bl_vals, best_vals)):
    lift  = float(tv) - bv
    color = CG if lift > 0 else CR
    ax.text(x[i]+w/2+0.02, max(bv, float(tv))+0.025,
            f"{lift:+.3f}", ha="left", fontsize=8, color=color, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
ax.set_title(f"Pipeline Comparison  |  AUC lift={best_s2_auc-bl_auc:+.4f}",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
savefig(fig, "p25_test_pipeline_comparison.png")

# ── Plot 6: Stage 2 feature importance ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
if hasattr(best_s2_model, "feature_importances_"):
    imp = best_s2_model.feature_importances_
else:
    imp = np.abs(best_s2_model.coef_[0])
idx    = np.argsort(imp)[::-1]
colors = [CR if "Y1hat" in all_s2_feat[i] else CB for i in idx]
ax.barh(range(len(all_s2_feat)), imp[idx][::-1],
        color=[colors[j] for j in range(len(all_s2_feat)-1,-1,-1)],
        alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(all_s2_feat)))
ax.set_yticklabels([all_s2_feat[i] for i in idx[::-1]], fontsize=9)
ax.set_xlabel("|Coefficient|" if hasattr(best_s2_model,"coef_") else "Feature Importance")
ax.set_title(f"Stage 2 Feature Importance — {best_s2_name} ★\n"
             "Red = Y1_hat (Stage 1 bridge)  |  Blue = D2 state features",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
import matplotlib.patches as mp2
ax.legend(handles=[mp2.Patch(color=CR, alpha=0.85, label="Y1_hat (Stage 1)"),
                   mp2.Patch(color=CB, alpha=0.85, label="D2 features")],
          fontsize=10)
plt.tight_layout()
savefig(fig, "p25_test_feature_importance_s2.png")

print(f"\n  ★ Best combination: {best_s2_name}  AUC={best_s2_auc:.4f}")
print(f"  stage2_test_metrics.csv → is_best=True row → read by p25_inference.py")
print("\n" + "="*62)
print("  Test evaluation complete")
print("  Next: python scripts/p25_evaluate.py")
print("="*62 + "\n")
