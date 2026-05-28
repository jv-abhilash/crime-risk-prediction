#!/bin/bash
# ============================================================
# P25 — Environment Setup Script
# Run this ONCE after cloning the repo:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e   # exit immediately if any command fails

echo ""
echo "=================================================="
echo "  P25 — Two-Stage Crime Risk Pipeline"
echo "  Environment Setup"
echo "=================================================="

# ── Step 1: Check Python version ────────────────────────────
echo ""
echo "[1/5] Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)')
if [ "$PYTHON_VERSION" -lt 38 ]; then
    echo "ERROR: Python 3.8+ required. Please upgrade."
    exit 1
fi
echo "      Python version OK"

# ── Step 2: Create virtual environment ──────────────────────
echo ""
echo "[2/5] Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo "      .venv already exists — skipping creation"
else
    python3 -m venv .venv
    echo "      .venv created successfully"
fi

# ── Step 3: Activate and upgrade pip ────────────────────────
echo ""
echo "[3/5] Activating .venv and upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip --quiet
echo "      pip upgraded: $(pip --version)"

# ── Step 4: Install requirements ────────────────────────────
echo ""
echo "[4/5] Installing requirements..."
pip install -r requirements.txt --quiet
echo "      All packages installed successfully"

# ── Step 5: Freeze exact versions ───────────────────────────
echo ""
echo "[5/5] Freezing exact package versions → requirements_frozen.txt"
pip freeze > requirements_frozen.txt
echo "      requirements_frozen.txt created"
echo "      (This file is in .gitignore — not committed)"

# ── Done ────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup complete!"
echo ""
echo "  To activate the environment in future sessions:"
echo "    source .venv/bin/activate"
echo ""
echo "  To run the full pipeline:"
echo "    python scripts/p25_pipeline_runner.py"
echo ""
echo "  To deactivate:"
echo "    deactivate"
echo "=================================================="
