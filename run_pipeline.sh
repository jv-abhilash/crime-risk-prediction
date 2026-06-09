#!/bin/bash
# ==============================================================
# P25 — Interactive Pipeline Runner
# Choose individual step or full pipeline at startup.
# ==============================================================

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
RED='\033[0;31m';   BLUE='\033[0;34m';   BOLD='\033[1m'; NC='\033[0m'

ok()     { echo -e "${GREEN}  ✓  $1${NC}"; }
info()   { echo -e "${BLUE}  →  $1${NC}"; }
warn()   { echo -e "${YELLOW}  ⚠  $1${NC}"; }
err()    { echo -e "${RED}  ✗  $1${NC}"; }
header() { echo -e "\n${BOLD}${CYAN}━━━  $1  ━━━${NC}"; }
sep()    { echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"; }

press_enter() {
    echo ""
    echo -e "${YELLOW}  Press Enter to run  |  Ctrl+C to stop${NC}"
    read -r
}

run_script() {
    local script="$1"; shift
    local extra_args="$@"
    echo -e "\n${BOLD}  Running: python scripts/${script} ${extra_args}${NC}"
    sep
    eval "python scripts/${script} ${extra_args}"
    local code=$?
    sep
    if [ $code -eq 0 ]; then
        ok "Done"
    else
        err "Failed (exit code $code) — fix the error above and retry this step"
        exit $code
    fi
}

# ── Activate venv ──────────────────────────────────────────────
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source .venv/bin/activate 2>/dev/null || {
        err ".venv not found. Run ./setup.sh first."
        exit 1
    }
fi
ok "venv: $(which python3)"

# ── Banner ─────────────────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   P25 — Crime Risk & Emergency Activation Pipeline  ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ════════════════════════════════════════════════════════════════
# STARTING MENU
# ════════════════════════════════════════════════════════════════
header "WHAT DO YOU WANT TO RUN?"
echo ""
echo -e "  ${BOLD}1)${NC}  Full pipeline     — runs all 6 steps in order"
echo -e "  ${BOLD}2)${NC}  Individual step   — you pick which step to run"
echo ""
echo -e "${YELLOW}  Enter choice [1/2]: ${NC}"
read -r MAIN_CHOICE

if [ "$MAIN_CHOICE" = "2" ]; then
    # ── Individual step menu ──────────────────────────────────
    header "WHICH STEP?"
    echo ""
    echo -e "  ${BOLD}1)${NC}  Split       — split datasets, apply transforms, build community_reference.csv"
    echo -e "  ${BOLD}2)${NC}  Train 1     — train 4 Stage 1 regression models, save to models/"
    echo -e "  ${BOLD}3)${NC}  Train 2     — train 4 Stage 2 classifier models, save to models/"
    echo -e "  ${BOLD}4)${NC}  Test        — load saved models, evaluate on test set, write metrics CSVs"
    echo -e "  ${BOLD}5)${NC}  Evaluate    — threshold sweep, calibration, dispatch cost, write thresholds.csv"
    echo -e "  ${BOLD}6)${NC}  Inference   — load best models from CSVs, predict dispatch decisions"
    echo ""
    echo -e "${YELLOW}  Enter step number [1-6]: ${NC}"
    read -r STEP_CHOICE
else
    STEP_CHOICE="all"
fi

# ════════════════════════════════════════════════════════════════
# STEP DESCRIPTIONS
# ════════════════════════════════════════════════════════════════
show_step1_info() {
    header "STEP 1 — Data Split & Preparation"
    sep
    info "Splits both datasets using the same row indices (critical for alignment)."
    info "Applies log1p to Y1 (for LR/Ridge) and state_murder_rate (for D2)."
    info "Builds community_reference.csv — 1994 communities with names + features."
    echo ""
    info "Outputs → data/processed/:"
    echo "    X1_train/test.npy   Y1_train_raw/log.npy   Y1_test_raw.npy"
    echo "    X2_train/test.npy   Y2_train/test.npy"
    echo "    train_idx.npy   test_idx.npy"
    echo "    feature_names_S1.npy   feature_names_S2.npy"
    echo "    community_reference.csv"
}

show_step2_info() {
    header "STEP 2 — Train Stage 1 (4 Regression Models)"
    sep
    info "Trains LinearRegression, Ridge, RandomForest, GradientBoosting."
    info "LR + Ridge use log1p(Y1).  RF + GBM use raw Y1."
    info "5-fold CV on training set — no test data touched here."
    echo ""
    info "Outputs:"
    echo "    models/stage1_*.joblib          (4 saved models)"
    echo "    outputs/p25_outputs/stage1_cv_metrics.csv"
}

show_step3_info() {
    header "STEP 3 — Train Stage 2 (4 Classification Models)"
    sep
    info "Builds 11-feature matrix: 7 D2 state features + 4 Y1_hat columns."
    info "Trains LogisticRegression, DecisionTree, RandomForest, GradientBoosting."
    info "class_weight='balanced' applied — mandatory for 5.5:1 imbalance."
    info "Stratified 5-fold CV on training set."
    echo ""
    info "Outputs:"
    echo "    models/stage2_*.joblib          (4 saved models)"
    echo "    outputs/p25_outputs/stage2_cv_metrics.csv"
}

show_step4_info() {
    header "STEP 4 — Test Evaluation (Held-Out Test Set)"
    sep
    info "LOADS saved models from disk — does NOT retrain."
    info "Evaluates all 8 models on 399-row held-out test set."
    info "Marks is_best=True on best Stage 2 model by AUC — used by inference."
    echo ""
    info "Outputs:"
    echo "    outputs/p25_outputs/stage1_test_metrics.csv  (R², RMSE per model)"
    echo "    outputs/p25_outputs/stage2_test_metrics.csv  (AUC, F1, Recall + is_best)"
    echo "    outputs/p25_outputs/p25_test_*.png           (6 evaluation plots)"
}

show_step5_info() {
    header "STEP 5 — Production Layers (Threshold + Calibration + Cost)"
    sep
    info "Loads best Stage 2 model from stage2_test_metrics.csv."
    info "Sweeps threshold 0.05 → 0.95 to find optimal cut-off points."
    info "Generates calibration reliability diagram."
    info "Simulates dispatch cost with FN×10 + FP×1 penalty model."
    echo ""
    info "Outputs:"
    echo "    outputs/p25_outputs/thresholds.csv     ← READ by inference"
    echo "      optimal_f1  — maximises F1            (default)"
    echo "      recall_90   — catches ≥90% emergencies"
    echo "      min_cost    — minimises dispatch cost"
    echo "    outputs/p25_outputs/p25_threshold_sweep.png"
    echo "    outputs/p25_outputs/p25_calibration.png"
    echo "    outputs/p25_outputs/p25_dispatch_cost.png"
}

show_step6_info() {
    header "STEP 6 — Inference"
    sep
    info "Reads stage1_test_metrics.csv → loads all 4 Stage 1 model paths."
    info "Reads stage2_test_metrics.csv → loads best Stage 2 model (is_best=True)."
    info "Reads thresholds.csv → loads threshold for chosen policy."
    echo ""
    info "Input features automatically pulled from community_reference.csv:"
    echo "    Stage 1 receives  100 D1 demographic features per community"
    echo "    Stage 1 produces  4 Y1_hat crime intensity scores (computed)"
    echo "    Stage 2 receives  7 D2 state crime rates + 4 Y1_hat = 11 features"
    echo "    Stage 2 produces  P(Emergency) → dispatch tier decision"
    echo ""
    info "Output: outputs/p25_outputs/inference_results.csv"
}

# ════════════════════════════════════════════════════════════════
# INFERENCE ARGUMENT PROMPTS (reused for step 6 and full pipeline)
# ════════════════════════════════════════════════════════════════
prompt_inference_args() {
    header "INFERENCE — Choose Run Mode"
    echo ""
    echo -e "  ${BOLD}1)${NC}  All 399 test communities      (default demo — no input needed)"
    echo -e "  ${BOLD}2)${NC}  Single community by name      (you type the community name)"
    echo -e "  ${BOLD}3)${NC}  Custom CSV file               (you provide path to CSV)"
    echo ""
    echo -e "${YELLOW}  Enter choice [1/2/3] or press Enter for default: ${NC}"
    read -r MODE_CHOICE

    header "INFERENCE — Choose Threshold Policy"
    echo ""
    echo -e "  ${BOLD}1)${NC}  optimal_f1   best balance of precision and recall  [default]"
    echo -e "  ${BOLD}2)${NC}  recall_90    catches ≥90% of emergencies (more false alarms)"
    echo -e "  ${BOLD}3)${NC}  min_cost     minimises FN×10 + FP×1 dispatch cost"
    echo ""
    echo -e "${YELLOW}  Enter choice [1/2/3] or press Enter for default: ${NC}"
    read -r POLICY_CHOICE

    case "$POLICY_CHOICE" in
        2) POLICY="recall_90" ;;
        3) POLICY="min_cost" ;;
        *) POLICY="optimal_f1" ;;
    esac
    ok "Policy: $POLICY"

    case "$MODE_CHOICE" in
        2)
            echo ""
            info "Example community names:"
            echo "    Selma, California"
            echo "    Lakewood, Colorado"
            echo "    Tukwila, Washington"
            echo "    (any name from data/processed/community_reference.csv)"
            echo ""
            echo -e "${YELLOW}  Type community name exactly: ${NC}"
            read -r COMMUNITY_NAME
            if [ -z "$COMMUNITY_NAME" ]; then
                warn "No name entered — running on test set instead"
                INF_CMD="--policy $POLICY"
            else
                INF_CMD="--community \"$COMMUNITY_NAME\" --policy $POLICY"
            fi
            ;;
        3)
            echo ""
            info "CSV must contain the 100 D1 columns from stage1_input.csv."
            info "D2 columns optional. 'community_id' column recommended."
            echo ""
            echo -e "${YELLOW}  Enter path to CSV file: ${NC}"
            read -r CSV_PATH
            if [ -z "$CSV_PATH" ] || [ ! -f "$CSV_PATH" ]; then
                warn "File not found — running on test set instead"
                INF_CMD="--policy $POLICY"
            else
                INF_CMD="--input \"$CSV_PATH\" --policy $POLICY"
            fi
            ;;
        *)
            INF_CMD="--policy $POLICY"
            ;;
    esac
}

# ════════════════════════════════════════════════════════════════
# EXECUTION
# ════════════════════════════════════════════════════════════════

run_step1() {
    show_step1_info
    press_enter
    run_script "p25_split.py"
}

run_step2() {
    show_step2_info
    press_enter
    run_script "p25_train1.py"
}

run_step3() {
    show_step3_info
    press_enter
    run_script "p25_train2.py"
}

run_step4() {
    show_step4_info
    press_enter
    run_script "p25_test.py"
}

run_step5() {
    show_step5_info
    press_enter
    run_script "p25_evaluate.py"
}

run_step6() {
    show_step6_info
    prompt_inference_args
    echo ""
    info "Command: python scripts/p25_inference.py $INF_CMD"
    press_enter
    run_script "p25_inference.py" "$INF_CMD"
}

# ── Route to selected step(s) ───────────────────────────────────
case "$STEP_CHOICE" in
    1)   run_step1 ;;
    2)   run_step2 ;;
    3)   run_step3 ;;
    4)   run_step4 ;;
    5)   run_step5 ;;
    6)   run_step6 ;;
    all)
        run_step1
        run_step2
        run_step3
        run_step4
        run_step5
        run_step6
        ;;
    *)
        err "Invalid choice"
        exit 1
        ;;
esac

# ── Final summary ───────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   Done!                                              ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Key files for Streamlit:"
echo "    data/processed/community_reference.csv"
echo "    outputs/p25_outputs/stage2_test_metrics.csv"
echo "    outputs/p25_outputs/thresholds.csv"
echo "    models/*.joblib"
echo ""
echo "  Useful commands:"
echo -e "    ${YELLOW}python scripts/p25_inference.py --community \"Selma, California\"${NC}"
echo -e "    ${YELLOW}python scripts/p25_inference.py --list${NC}"
echo -e "    ${YELLOW}python scripts/p25_inference.py --policy recall_90${NC}"
echo ""
