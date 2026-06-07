"""
=============================================================================
P25 — p25_test.py    Comprehensive Test Evaluation
=============================================================================
Loads saved arrays from p25_split.py and retrains all 8 models to generate
a complete test evaluation. This script produces ZERO training decisions —
it only evaluates performance on the held-out test set.

Run AFTER p25_train2.py:
    python scripts/p25_test.py

Outputs → outputs/p25_outputs/:
    p25_test_stage1_scatter.png        actual vs predicted for all 4 regressors
    p25_test_stage1_importance.png     top feature importances (RF/GBM)
    p25_test_stage2_confusion.png      confusion matrix for all 4 classifiers
    p25_test_stage2_roc.png            ROC curves all 4 classifiers on one plot
    p25_test_stage2_pr.png             Precision-Recall curves
    p25_test_pipeline_comparison.png   baseline LogReg vs full two-stage
    p25_test_feature_importance_s2.png which of the 11 Stage 2 features matter
    p25_test_report.txt                full numeric summary
=============================================================================
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score,
    accuracy_score, mean_squared_error, r2_score,
    classification_report
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
OUT_DIR   = os.path.join(ROOT, "outputs", "p25_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colours ─────────────────────────────────────────────────────────────────
CB = "#2E75B6"; CG = "#1E5C2E"; CO = "#C55A11"
CR = "#9B1C1C"; CP = "#5B2C8D"; CGR = "#7F7F7F"
S1_COLORS = [CB, CG, CO, CP]
S2_COLORS = [CB, CG, CO, CR]

def load(name):
    path = os.path.join(DATA_PROC, f"{name}.npy")
    if not os.path.exists(path):
        print(f"[ERROR] {name}.npy not found.")
        print("  Run scripts in order: split → train1 → train2 → test")
        sys.exit(1)
    return np.load(path, allow_pickle=True)

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}")

print("\n" + "="*65)
print("  P25 — Comprehensive Test Evaluation")
print("="*65)

# ════════════════════════════════════════════════════════════════════════════
# LOAD ALL ARRAYS
# ════════════════════════════════════════════════════════════════════════════
print("\n[LOAD] Reading saved arrays from data/processed/...")

X1_train     = load("X1_train")
X1_test      = load("X1_test")
Y1_train_raw = load("Y1_train_raw")
Y1_train_log = load("Y1_train_log")
Y1_test_raw  = load("Y1_test_raw")
X2_train     = load("X2_train")
X2_test      = load("X2_test")
Y2_train     = load("Y2_train")
Y2_test      = load("Y2_test")
feat_s1      = load("feature_names_S1")
feat_s2      = load("feature_names_S2")

Y1_hat_tr    = load("Y1_hat_train")   # (1595, 4) saved by train1.py
Y1_hat_te    = load("Y1_hat_test")    # (399, 4)  saved by train1.py

S2_train = np.hstack([X2_train, Y1_hat_tr])
S2_test  = np.hstack([X2_test,  Y1_hat_te])

col_names_y1hat = ["LinReg_Y1hat", "Ridge_Y1hat", "RF_Y1hat", "GBM_Y1hat"]
all_s2_features = list(feat_s2) + col_names_y1hat

sample_w = compute_sample_weight("balanced", Y2_train)

print(f"  X1 test  : {X1_test.shape}")
print(f"  X2 test  : {X2_test.shape}")
print(f"  S2 test  : {S2_test.shape}  (7 D2 + 4 Y1_hat)")
print(f"  Y2 test  : pos={Y2_test.sum()} ({Y2_test.mean()*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# REBUILD MODELS (train on train set, evaluate on test set)
# ════════════════════════════════════════════════════════════════════════════
print("\n[BUILD] Retraining all models for test evaluation...")

# Stage 1 models
S1_MODELS = [
    {"name": "LinearRegression", "short": "LinReg",
     "model": LinearRegression(), "target": "log"},
    {"name": "Ridge(α=1.0)", "short": "Ridge",
     "model": Ridge(alpha=1.0), "target": "log"},
    {"name": "RandomForest", "short": "RF",
     "model": RandomForestRegressor(n_estimators=200, min_samples_split=5,
                                     n_jobs=-1, random_state=42),
     "target": "raw"},
    {"name": "GradientBoosting", "short": "GBM",
     "model": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                         max_depth=4, subsample=0.8,
                                         random_state=42),
     "target": "raw"},
]
for cfg in S1_MODELS:
    Y_tr = Y1_train_log if cfg["target"] == "log" else Y1_train_raw
    cfg["model"].fit(X1_train, Y_tr)
    hat = cfg["model"].predict(X1_test)
    if cfg["target"] == "log":
        hat = np.clip(np.expm1(hat), 0, 1)
    cfg["hat_te"]  = hat
    cfg["test_r2"] = r2_score(Y1_test_raw, hat)
    cfg["test_rmse"] = mean_squared_error(Y1_test_raw, hat) ** 0.5
    print(f"  {cfg['name']:<22} Test R²={cfg['test_r2']:.4f}  "
          f"RMSE={cfg['test_rmse']:.4f}")

# Stage 2 models
S2_MODELS = [
    {"name": "LogisticRegression", "short": "LogReg",
     "model": LogisticRegression(class_weight="balanced",
                                  max_iter=1000, random_state=42)},
    {"name": "DecisionTree", "short": "DTree",
     "model": DecisionTreeClassifier(class_weight="balanced",
                                      max_depth=6, min_samples_split=10,
                                      random_state=42)},
    {"name": "RandomForest", "short": "RF",
     "model": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      min_samples_split=5, n_jobs=-1,
                                      random_state=42)},
    {"name": "GradientBoosting", "short": "GBM",
     "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, subsample=0.8,
                                          random_state=42)},
]
for cfg in S2_MODELS:
    if "Gradient" in cfg["name"]:
        cfg["model"].fit(S2_train, Y2_train, sample_weight=sample_w)
    else:
        cfg["model"].fit(S2_train, Y2_train)
    cfg["proba"] = cfg["model"].predict_proba(S2_test)[:, 1]
    cfg["preds"] = cfg["model"].predict(S2_test)
    cfg["auc"]   = roc_auc_score(Y2_test, cfg["proba"])
    cfg["f1"]    = f1_score(Y2_test, cfg["preds"])
    cfg["rec"]   = recall_score(Y2_test, cfg["preds"])
    cfg["prec"]  = precision_score(Y2_test, cfg["preds"], zero_division=0)
    cfg["acc"]   = accuracy_score(Y2_test, cfg["preds"])
    print(f"  {cfg['name']:<22} AUC={cfg['auc']:.4f}  "
          f"F1={cfg['f1']:.4f}  Recall={cfg['rec']:.4f}")

# Baseline: single LogReg on raw D1 features (no two-stage pipeline)
baseline = LogisticRegression(class_weight="balanced",
                               max_iter=1000, random_state=42)
baseline.fit(X1_train, Y2_train)
bl_proba = baseline.predict_proba(X1_test)[:, 1]
bl_preds = baseline.predict(X1_test)
bl_auc   = roc_auc_score(Y2_test, bl_proba)
bl_rec   = recall_score(Y2_test, bl_preds)

best_s2  = max(S2_MODELS, key=lambda r: r["auc"])
print(f"\n  Baseline (D1 only LogReg): AUC={bl_auc:.4f}  Recall={bl_rec:.4f}")
print(f"  Best two-stage model     : AUC={best_s2['auc']:.4f}  "
      f"Recall={best_s2['rec']:.4f}")
print(f"  AUC lift                 : {best_s2['auc']-bl_auc:+.4f}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Stage 1: Actual vs Predicted (all 4 regressors)
# ════════════════════════════════════════════════════════════════════════════
print("\n[PLOT 1] Stage 1 actual vs predicted...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Stage 1 Test Evaluation — Actual vs Predicted Y1\n"
             "(ViolentCrimesPerPop on held-out test set, N=399)",
             fontsize=13, fontweight="bold")

for ax, cfg, c in zip(axes.flatten(), S1_MODELS, S1_COLORS):
    ax.scatter(Y1_test_raw, cfg["hat_te"],
               alpha=0.35, s=12, color=c)
    lim = [0, 1]
    ax.plot(lim, lim, color="black", lw=1.5, ls="--",
            label="Perfect prediction")
    ax.set_xlabel("Actual ViolentCrimesPerPop")
    ax.set_ylabel("Predicted ViolentCrimesPerPop")
    ax.set_title(
        f"{cfg['name']}\nTest R²={cfg['test_r2']:.4f}  "
        f"RMSE={cfg['test_rmse']:.4f}",
        fontweight="bold", color=c)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 1.1)
    # Annotate residual range
    resid = Y1_test_raw - cfg["hat_te"]
    ax.text(0.02, 0.97,
            f"Residual std : {resid.std():.4f}\n"
            f"Max overpredict: {resid.min():.4f}\n"
            f"Max underpredict: {resid.max():.4f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

plt.tight_layout()
save_fig(fig, "p25_test_stage1_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Stage 1: Feature Importance (RF and GBM only)
# ════════════════════════════════════════════════════════════════════════════
print("[PLOT 2] Stage 1 feature importance...")

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("Stage 1 — Top 20 Feature Importances\n"
             "Which demographic features drive the crime intensity prediction?",
             fontsize=13, fontweight="bold")

for ax, cfg, c in zip(axes,
                       [m for m in S1_MODELS if m["short"] in ["RF", "GBM"]],
                       [CO, CP]):
    model = cfg["model"]
    imp   = model.feature_importances_
    idx   = np.argsort(imp)[::-1][:20]
    ax.barh(range(20), imp[idx][::-1],
            color=c, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(20))
    ax.set_yticklabels([feat_s1[i] for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"{cfg['name']}  (R²={cfg['test_r2']:.4f})",
                 fontweight="bold", color=c)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
save_fig(fig, "p25_test_stage1_importance.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Stage 2: Confusion Matrix (all 4 classifiers)
# ════════════════════════════════════════════════════════════════════════════
print("[PLOT 3] Stage 2 confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle("Stage 2 Test Evaluation — Confusion Matrices\n"
             "(Held-out test set, N=399,  Emergency=15.4%)",
             fontsize=13, fontweight="bold")

for ax, cfg, c in zip(axes.flatten(), S2_MODELS, S2_COLORS):
    cm = confusion_matrix(Y2_test, cfg["preds"])
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    labels = ["No Emergency\n(Class 0)", "Emergency\n(Class 1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    # Annotate cells
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            cell_val = cm[i, j]
            ax.text(j, i,
                    f"{cell_val}\n({cell_val/total*100:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    # Label the quadrants
    ax.text(0, -0.5, "TN", ha="center", fontsize=8, color=CG)
    ax.text(1, -0.5, "FP", ha="center", fontsize=8, color=CO)
    ax.text(0,  1.5, "FN", ha="center", fontsize=8, color=CR)
    ax.text(1,  1.5, "TP", ha="center", fontsize=8, color=CG)
    ax.set_title(
        f"{cfg['name']}\n"
        f"AUC={cfg['auc']:.3f}  F1={cfg['f1']:.3f}  "
        f"Recall={cfg['rec']:.3f}",
        fontweight="bold", color=c)

plt.tight_layout()
save_fig(fig, "p25_test_stage2_confusion.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Stage 2: ROC Curves (all 4 on one plot)
# ════════════════════════════════════════════════════════════════════════════
print("[PLOT 4] Stage 2 ROC curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Stage 2 Test Evaluation — ROC & Precision-Recall Curves",
             fontsize=13, fontweight="bold")

# ROC
ax1.plot([0,1],[0,1], color="gray", lw=1, ls="--",
         label="Random classifier (AUC=0.50)")
for cfg, c in zip(S2_MODELS, S2_COLORS):
    fpr, tpr, _ = roc_curve(Y2_test, cfg["proba"])
    ax1.plot(fpr, tpr, color=c, lw=2.5,
             label=f"{cfg['short']}  AUC={cfg['auc']:.4f}")
# Baseline
fpr_b, tpr_b, _ = roc_curve(Y2_test, bl_proba)
ax1.plot(fpr_b, tpr_b, color=CGR, lw=1.5, ls=":",
         label=f"Baseline (D1 only) AUC={bl_auc:.4f}")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves — All Stage 2 Models", fontweight="bold")
ax1.legend(fontsize=9, loc="lower right"); ax1.grid(True, alpha=0.3)

# Precision-Recall
no_skill = Y2_test.mean()
ax2.axhline(no_skill, color="gray", lw=1, ls="--",
            label=f"No-skill baseline (P={no_skill:.2f})")
for cfg, c in zip(S2_MODELS, S2_COLORS):
    prec, rec, _ = precision_recall_curve(Y2_test, cfg["proba"])
    ap = average_precision_score(Y2_test, cfg["proba"])
    ax2.plot(rec, prec, color=c, lw=2.5,
             label=f"{cfg['short']}  AP={ap:.4f}")
prec_b, rec_b, _ = precision_recall_curve(Y2_test, bl_proba)
ap_b = average_precision_score(Y2_test, bl_proba)
ax2.plot(rec_b, prec_b, color=CGR, lw=1.5, ls=":",
         label=f"Baseline  AP={ap_b:.4f}")
ax2.set_xlabel("Recall (Sensitivity)"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves\n"
              "(Better for imbalanced data than ROC)",
              fontweight="bold")
ax2.legend(fontsize=9, loc="upper right"); ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, "p25_test_stage2_roc.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Pipeline Comparison: Baseline vs Two-Stage
# ════════════════════════════════════════════════════════════════════════════
print("[PLOT 5] Pipeline comparison...")

metrics   = ["AUC", "F1", "Recall", "Precision", "Accuracy"]
bl_vals   = [bl_auc,
             f1_score(Y2_test, bl_preds),
             recall_score(Y2_test, bl_preds),
             precision_score(Y2_test, bl_preds, zero_division=0),
             accuracy_score(Y2_test, bl_preds)]
best_vals = [best_s2["auc"], best_s2["f1"],
             best_s2["rec"], best_s2["prec"], best_s2["acc"]]

x   = np.arange(len(metrics))
w   = 0.35
fig, ax = plt.subplots(figsize=(13, 7))
b1  = ax.bar(x - w/2, bl_vals,   w, color=CGR, alpha=0.85,
             edgecolor="white", label="Baseline: single LogReg on D1 only")
b2  = ax.bar(x + w/2, best_vals, w, color=CB,  alpha=0.85,
             edgecolor="white",
             label=f"Two-Stage Pipeline: {best_s2['name']}")

for bar, v in zip(list(b1)+list(b2), bl_vals+best_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.008,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title(
    "Pipeline Comparison: Baseline vs Full Two-Stage System\n"
    f"AUC lift = {best_s2['auc']-bl_auc:+.4f}  |  "
    f"Recall lift = {best_s2['rec']-bl_rec:+.4f}",
    fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

# Annotate lift
for i, (bv, tv) in enumerate(zip(bl_vals, best_vals)):
    lift = tv - bv
    color = CG if lift > 0 else CR
    ax.text(x[i]+w/2+0.02, max(bv, tv)+0.025,
            f"{lift:+.3f}", ha="left", fontsize=8,
            color=color, fontweight="bold")

plt.tight_layout()
save_fig(fig, "p25_test_pipeline_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Stage 2 Feature Importance (best model)
# ════════════════════════════════════════════════════════════════════════════
print("[PLOT 6] Stage 2 feature importance (best model)...")

# Use the best Stage 2 model — if it has feature_importances_
best_model = best_s2["model"]
if hasattr(best_model, "feature_importances_"):
    imp  = best_model.feature_importances_
    idx  = np.argsort(imp)[::-1]
    cols = all_s2_features

    fig, ax = plt.subplots(figsize=(13, 7))
    colors_imp = [CR if "Y1hat" in cols[i] else CB for i in idx]
    bars = ax.barh(range(len(cols)), imp[idx][::-1],
                   color=[colors_imp[j] for j in range(len(cols)-1,-1,-1)],
                   alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([cols[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(
        f"Stage 2 Feature Importance — {best_s2['name']}\n"
        f"Red = Y1_hat columns (Stage 1 bridge)  |  "
        f"Blue = D2 state features",
        fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    import matplotlib.patches as mpatches
    r_p = mpatches.Patch(color=CR, alpha=0.85, label="Y1_hat (Stage 1 output)")
    b_p = mpatches.Patch(color=CB, alpha=0.85, label="D2 state crime features")
    ax.legend(handles=[r_p, b_p], fontsize=10)
    plt.tight_layout()
    save_fig(fig, "p25_test_feature_importance_s2.png")
else:
    # LogReg: use absolute coefficients as proxy for importance
    coef = np.abs(best_model.coef_[0])
    idx  = np.argsort(coef)[::-1]
    cols = all_s2_features
    fig, ax = plt.subplots(figsize=(13, 6))
    colors_imp = [CR if "Y1hat" in cols[i] else CB for i in idx]
    ax.barh(range(len(cols)), coef[idx][::-1],
            color=[colors_imp[j] for j in range(len(cols)-1,-1,-1)],
            alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([cols[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("|Coefficient|")
    ax.set_title(
        f"Stage 2 Coefficient Magnitudes — {best_s2['name']}\n"
        "Red = Y1_hat (Stage 1 bridge)  |  Blue = D2 features",
        fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "p25_test_feature_importance_s2.png")


# ════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n[REPORT] Writing p25_test_report.txt...")

lines = [
    "P25 TEST EVALUATION REPORT",
    "="*65,
    f"Test set size : {len(Y2_test)} communities",
    f"Emergency pos : {Y2_test.sum()} ({Y2_test.mean()*100:.1f}%)",
    "",
    "STAGE 1 TEST RESULTS (Y1 = ViolentCrimesPerPop):",
    f"{'Model':<22} {'Target':>6} {'Test R²':>9} {'Test RMSE':>10}",
    "-"*50,
]
for cfg in S1_MODELS:
    lines.append(
        f"{cfg['name']:<22} {cfg['target']:>6} "
        f"{cfg['test_r2']:>9.4f} {cfg['test_rmse']:>10.4f}")

lines += [
    "",
    "STAGE 2 TEST RESULTS (Y2 = EmergencyActivation):",
    f"{'Model':<24} {'AUC':>7} {'F1':>7} {'Recall':>8} "
    f"{'Precis':>8} {'Acc':>7}",
    "-"*60,
]
for cfg in S2_MODELS:
    lines.append(
        f"{cfg['name']:<24} {cfg['auc']:>7.4f} {cfg['f1']:>7.4f} "
        f"{cfg['rec']:>8.4f} {cfg['prec']:>8.4f} {cfg['acc']:>7.4f}")

lines += [
    "",
    "PIPELINE COMPARISON:",
    f"  Baseline (D1 only LogReg) : AUC={bl_auc:.4f}  Recall={bl_rec:.4f}",
    f"  Best two-stage pipeline   : AUC={best_s2['auc']:.4f}  "
    f"Recall={best_s2['rec']:.4f}",
    f"  AUC lift                  : {best_s2['auc']-bl_auc:+.4f}",
    f"  Recall lift               : {best_s2['rec']-bl_rec:+.4f}",
    "",
    f"BEST STAGE 2 MODEL: {best_s2['name']}",
    "",
    "CLASSIFICATION REPORT (best model):",
    classification_report(Y2_test, best_s2["preds"],
                           target_names=["No Emergency", "Emergency"]),
    "",
    "EDA WARNINGS — STATUS AFTER TRAINING:",
    f"  Y1 skew=1.52    → log1p applied for LR/Ridge. "
    f"Best R²={max(S1_MODELS,key=lambda m:m['test_r2'])['test_r2']:.4f}",
    f"  5.5:1 imbalance → class_weight=balanced. "
    f"Best Recall={best_s2['rec']:.4f} "
    f"({'improved' if best_s2['rec'] > 0.5 else 'still limited'})",
    f"  Multicollinearity → Ridge competed. "
    f"Ridge R²={[m for m in S1_MODELS if 'Ridge' in m['name']][0]['test_r2']:.4f}",
    f"  Heteroscedasticity → GBM/RF unaffected.",
    f"  5.5% outliers → Ridge/RF/GBM mitigate.",
]

with open(os.path.join(OUT_DIR, "p25_test_report.txt"), "w") as f:
    f.write("\n".join(lines))
print("  [SAVED] p25_test_report.txt")

print("\n" + "="*65)
print("  Test evaluation complete!")
print()
print(f"  Best Stage 1 : "
      f"{max(S1_MODELS,key=lambda m:m['test_r2'])['name']}  "
      f"R²={max(S1_MODELS,key=lambda m:m['test_r2'])['test_r2']:.4f}")
print(f"  Best Stage 2 : {best_s2['name']}  AUC={best_s2['auc']:.4f}")
print(f"  AUC lift vs baseline : {best_s2['auc']-bl_auc:+.4f}")
print()
print("  Outputs → outputs/p25_outputs/")
print("="*65 + "\n")
