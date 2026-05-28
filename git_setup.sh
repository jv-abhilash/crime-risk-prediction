#!/bin/bash
# ============================================================
# P25 — Git Initialisation Script
# Run this ONCE after setup.sh:
#   chmod +x git_setup.sh
#   ./git_setup.sh
#
# Before running, fill in your details below:
# ============================================================

# ── FILL THESE IN BEFORE RUNNING ────────────────────────────
GIT_USERNAME="your_github_username"
GIT_EMAIL="your_email@example.com"
REMOTE_URL="https://github.com/your_username/p25-crime-risk-pipeline.git"
# ────────────────────────────────────────────────────────────

set -e

echo ""
echo "=================================================="
echo "  P25 — Git Setup"
echo "=================================================="

# Step 1: Configure git identity (local to this repo)
echo ""
echo "[1/6] Configuring git identity..."
git config --local user.name  "$GIT_USERNAME"
git config --local user.email "$GIT_EMAIL"
echo "      Name : $GIT_USERNAME"
echo "      Email: $GIT_EMAIL"

# Step 2: Initialise repository
echo ""
echo "[2/6] Initialising git repository..."
if [ -d ".git" ]; then
    echo "      .git already exists — skipping init"
else
    git init
    echo "      Git repository initialised"
fi

# Step 3: Set default branch to main
echo ""
echo "[3/6] Setting default branch to 'main'..."
git checkout -b main 2>/dev/null || git checkout main
echo "      Branch: main"

# Step 4: Stage all files for first commit
echo ""
echo "[4/6] Staging files..."
git add .
echo "      Staged files:"
git status --short

# Step 5: First commit
echo ""
echo "[5/6] Creating initial commit..."
git commit -m "Initial commit: P25 project structure, datasets, requirements

- Folder structure: data/raw, data/processed, scripts, outputs, docs
- Raw datasets: communities.data, communities.names, state_crime.csv
- requirements.txt with all dependencies
- .gitignore (excludes .venv, outputs, processed CSVs)
- Project reference document: P25_Project_Reference.docx
- setup.sh for environment creation
"
echo "      Initial commit created"

# Step 6: Connect to remote and push
echo ""
echo "[6/6] Connecting to remote repository..."
git remote add origin "$REMOTE_URL"
echo "      Remote set to: $REMOTE_URL"
echo ""
echo "      Pushing to remote..."
git push -u origin main
echo "      Push complete"

echo ""
echo "=================================================="
echo "  Git setup complete!"
echo ""
echo "  Future workflow:"
echo "    git add ."
echo "    git commit -m 'your message'"
echo "    git push"
echo "=================================================="
