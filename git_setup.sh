#!/bin/bash
# ==============================================================
# P25 — Git Setup Script
# Handles four scenarios interactively:
#
#   1) New machine  — clone existing repo + environment setup
#   2) New project  — init git + push to GitHub (what we already did)
#   3) Pull updates — sync local repo with remote
#   4) Config only  — set git username and email
#
# Usage:
#   chmod +x git_setup.sh
#   ./git_setup.sh
# ==============================================================

# ── colours ────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m';   BLUE='\033[0;34m'
CYAN='\033[0;36m';  NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
info() { echo -e "${BLUE}[INFO]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
ask()  { echo -e "${CYAN}[INPUT]${NC} $1"; }

# ── repo config — change these if the repo moves ─────────────
REPO_URL="https://github.com/jv-abhilash/crime-risk-prediction.git"
REPO_DIR="ml_capstone_project"

# ── shared function: configure local git identity ────────────
configure_identity() {
    local SCOPE="$1"   # --local or --global
    echo ""
    info "Configuring git identity ($SCOPE)..."

    CUR_NAME=$(git config $SCOPE user.name 2>/dev/null || echo "")
    CUR_EMAIL=$(git config $SCOPE user.email 2>/dev/null || echo "")

    # Try to fall back to global if local is empty
    if [ -z "$CUR_NAME" ]; then
        CUR_NAME=$(git config --global user.name 2>/dev/null || echo "not set")
    fi
    if [ -z "$CUR_EMAIL" ]; then
        CUR_EMAIL=$(git config --global user.email 2>/dev/null || echo "not set")
    fi

    info "Current name  : ${CUR_NAME}"
    info "Current email : ${CUR_EMAIL}"
    echo ""

    ask "Username [Enter to keep '${CUR_NAME}']: "
    read -r NEW_NAME
    if [ -n "$NEW_NAME" ]; then
        git config $SCOPE user.name "$NEW_NAME"
        ok "Name set: $NEW_NAME"
    else
        [ -n "$(git config --global user.name 2>/dev/null)" ] && ok "Name unchanged: ${CUR_NAME}"
    fi

    ask "Email [Enter to keep '${CUR_EMAIL}']: "
    read -r NEW_EMAIL
    if [ -n "$NEW_EMAIL" ]; then
        git config $SCOPE user.email "$NEW_EMAIL"
        ok "Email set: $NEW_EMAIL"
    else
        [ -n "$(git config --global user.email 2>/dev/null)" ] && ok "Email unchanged: ${CUR_EMAIL}"
    fi
    echo ""
}

# ── header ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  P25 — Git Setup"
echo "  Repo: jv-abhilash/crime-risk-prediction"
echo "============================================================"
echo ""

# Check git installed
if ! command -v git &>/dev/null; then
    err "git not found. Install: sudo apt install git"
fi
ok "git $(git --version | cut -d' ' -f3)"

# ── MENU ─────────────────────────────────────────────────────
echo ""
echo "  What do you want to do?"
echo ""
echo "  1)  New machine  — clone the repo and set up environment"
echo "  2)  New project  — initialise git and push to GitHub"
echo "  3)  Pull updates — fetch and merge latest from remote"
echo "  4)  Config only  — set git username and email"
echo ""
ask "Enter choice [1-4]: "
read -r CHOICE
echo ""

# ─────────────────────────────────────────────────────────────
# OPTION 1 — Clone on a new machine
# ─────────────────────────────────────────────────────────────
if [ "$CHOICE" = "1" ]; then
    echo "──────────────────────────────────────────────────────"
    echo "  OPTION 1 — Clone repo to this machine"
    echo "──────────────────────────────────────────────────────"
    echo ""

    ask "Clone into which folder? [default: ${REPO_DIR}]: "
    read -r CLONE_DIR
    CLONE_DIR="${CLONE_DIR:-$REPO_DIR}"

    if [ -d "$CLONE_DIR" ]; then
        warn "Directory '${CLONE_DIR}' already exists."
        ask "Delete it and reclone? [y/N]: "
        read -r CONFIRM
        if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
            rm -rf "$CLONE_DIR"
            ok "Removed existing directory"
        else
            info "Cancelled."
            exit 0
        fi
    fi

    info "Cloning from ${REPO_URL}..."
    git clone "$REPO_URL" "$CLONE_DIR"
    ok "Cloned into ./${CLONE_DIR}"

    cd "$CLONE_DIR"
    configure_identity "--local"

    echo ""
    info "Repository cloned. Now set up the Python environment:"
    echo ""
    echo -e "    ${YELLOW}cd ${CLONE_DIR}${NC}"
    echo -e "    ${YELLOW}chmod +x setup.sh${NC}"
    echo -e "    ${YELLOW}./setup.sh${NC}"
    echo ""
    ok "Clone complete!"

# ─────────────────────────────────────────────────────────────
# OPTION 2 — Init new project and push
# ─────────────────────────────────────────────────────────────
elif [ "$CHOICE" = "2" ]; then
    echo "──────────────────────────────────────────────────────"
    echo "  OPTION 2 — Initialise and push project"
    echo "──────────────────────────────────────────────────────"
    echo ""

    if [ ! -f "requirements.txt" ]; then
        err "requirements.txt not found. Run from the project root."
    fi

    # Init
    if [ -d ".git" ]; then
        warn ".git already exists — skipping git init"
    else
        git init
        ok "Git repository initialised"
    fi

    # Set branch to main
    git checkout -b main 2>/dev/null || git checkout main 2>/dev/null || true
    ok "Branch: main"

    # Configure identity
    configure_identity "--local"

    # Remote
    if git remote get-url origin &>/dev/null; then
        EXISTING=$(git remote get-url origin)
        warn "Remote 'origin' already set: ${EXISTING}"
        ask "Update remote to ${REPO_URL}? [y/N]: "
        read -r CONFIRM
        if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
            git remote set-url origin "$REPO_URL"
            ok "Remote updated"
        fi
    else
        ask "Remote URL [default: ${REPO_URL}]: "
        read -r CUSTOM_URL
        REMOTE="${CUSTOM_URL:-$REPO_URL}"
        git remote add origin "$REMOTE"
        ok "Remote set: $REMOTE"
    fi

    # Commit
    git add .
    STAGED=$(git status --short | wc -l)
    info "${STAGED} files staged"

    ask "Commit message [default: 'Initial commit: P25 crime risk pipeline']: "
    read -r MSG
    MSG="${MSG:-Initial commit: P25 crime risk pipeline}"
    git commit -m "$MSG"
    ok "Committed"

    # Push
    info "Pushing to remote..."
    git push -u origin main
    ok "Pushed to origin/main"

    echo ""
    ok "Project live at: https://github.com/jv-abhilash/crime-risk-prediction"

# ─────────────────────────────────────────────────────────────
# OPTION 3 — Pull latest updates
# ─────────────────────────────────────────────────────────────
elif [ "$CHOICE" = "3" ]; then
    echo "──────────────────────────────────────────────────────"
    echo "  OPTION 3 — Pull latest changes from remote"
    echo "──────────────────────────────────────────────────────"
    echo ""

    if [ ! -d ".git" ]; then
        err "Not a git repository. Run option 1 to clone first."
    fi

    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    info "Current branch: ${BRANCH}"
    info "Remote        : $(git remote get-url origin 2>/dev/null || echo 'not set')"
    info "Last commit   : $(git log --oneline -1)"
    echo ""

    # Check for uncommitted changes
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        warn "You have uncommitted local changes:"
        git status --short
        echo ""
        echo "  What do you want to do with them?"
        echo "  1) Stash them temporarily, pull, then restore"
        echo "  2) Commit them now, then pull"
        echo "  3) Cancel (do nothing)"
        echo ""
        ask "Choose [1-3]: "
        read -r PULL_OPT

        if [ "$PULL_OPT" = "1" ]; then
            STASH_MSG="auto-stash $(date '+%Y-%m-%d %H:%M')"
            git stash push -m "$STASH_MSG"
            ok "Changes stashed"
            git pull origin "$BRANCH"
            ok "Pull complete"
            git stash pop
            ok "Stashed changes restored"
        elif [ "$PULL_OPT" = "2" ]; then
            ask "Commit message: "
            read -r COMMIT_MSG
            git add .
            git commit -m "${COMMIT_MSG:-WIP before pull}"
            git pull origin "$BRANCH" --no-rebase
            ok "Committed and pulled"
        else
            info "Cancelled — nothing changed."
            exit 0
        fi
    else
        # Clean working tree
        git pull origin "$BRANCH"
        ok "Up to date with remote"
    fi

    info "Latest commit : $(git log --oneline -1)"

    # Sync packages if requirements changed
    if [ -d ".venv" ]; then
        echo ""
        ask "Sync pip packages in case requirements.txt changed? [Y/n]: "
        read -r SYNC
        if [ "$SYNC" != "n" ] && [ "$SYNC" != "N" ]; then
            source .venv/bin/activate
            pip install -r requirements.txt --quiet
            pip freeze > requirements_frozen.txt
            ok "Packages synced"
        fi
    else
        warn ".venv not found — run ./setup.sh to create the environment"
    fi

    echo ""
    ok "Repository updated!"

# ─────────────────────────────────────────────────────────────
# OPTION 4 — Configure git identity only
# ─────────────────────────────────────────────────────────────
elif [ "$CHOICE" = "4" ]; then
    echo "──────────────────────────────────────────────────────"
    echo "  OPTION 4 — Configure git identity"
    echo "──────────────────────────────────────────────────────"

    if [ -d ".git" ]; then
        echo ""
        echo "  Apply to:"
        echo "  1) Local (this repo only) — recommended"
        echo "  2) Global (all repos on this machine)"
        echo ""
        ask "Choose [1]: "
        read -r SCOPE_OPT
        if [ "$SCOPE_OPT" = "2" ]; then SCOPE="--global"; else SCOPE="--local"; fi
    else
        warn "Not inside a git repo — configuring globally"
        SCOPE="--global"
    fi

    configure_identity "$SCOPE"
    ok "Git identity configured"

else
    err "Invalid choice '${CHOICE}'. Enter 1, 2, 3, or 4."
fi

echo ""
echo "============================================================"
echo -e "  ${GREEN}Done!${NC}"
echo "============================================================"
echo ""