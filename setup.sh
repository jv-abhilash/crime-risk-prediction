#!/bin/bash
# ==============================================================
# P25 — Environment Setup Script
# Works on any Ubuntu/Linux machine.
# Safe to run multiple times — detects existing state.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ==============================================================

set -e   # exit immediately if any command fails

# ── colours ────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m';   BLUE='\033[0;34m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
info() { echo -e "${BLUE}[INFO]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "============================================================"
echo "  P25 — Two-Stage Crime Risk Pipeline"
echo "  Environment Setup"
echo "============================================================"
echo ""

# ── Step 1: Verify project directory ───────────────────────
info "Step 1/7 — Verifying project directory..."
if [ ! -f "requirements.txt" ]; then
    err "requirements.txt not found.
Run this script from the project root:
  cd ml_capstone_project && ./setup.sh"
fi
ok "Project root confirmed: $(pwd)"

# ── Step 2: Check Python version ───────────────────────────
info "Step 2/7 — Checking Python version..."
if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install:
  sudo apt update && sudo apt install python3 python3-venv python3-pip"
fi
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
PY_VER="${PY_MAJOR}.${PY_MINOR}"
if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 8 ]; then
    err "Python 3.8+ required. Found Python ${PY_VER}."
fi
ok "Python ${PY_VER} — OK"

# ── Step 3: Create or reuse virtual environment ─────────────
info "Step 3/7 — Virtual environment..."
if [ -d ".venv" ]; then
    warn ".venv already exists — reusing it"
    warn "To force clean rebuild: rm -rf .venv && ./setup.sh"
else
    python3 -m venv .venv
    ok ".venv created"
fi

# ── Step 4: Activate venv ──────────────────────────────────
info "Step 4/7 — Activating .venv..."
source .venv/bin/activate
ok "Activated: $(which python3)"

# ── Step 5: Upgrade pip ────────────────────────────────────
info "Step 5/7 — Upgrading pip..."
pip install --upgrade pip --quiet
ok "pip $(pip --version | cut -d' ' -f2)"

# ── Step 6: Install ALL requirements ───────────────────────
info "Step 6/7 — Installing requirements from requirements.txt..."
echo ""
pip install -r requirements.txt
echo ""

# Verify critical packages loaded correctly
python3 - << 'PYCHECK'
import importlib, sys
pkgs = {
    "pandas":      "pandas",
    "numpy":       "numpy",
    "sklearn":     "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib":  "matplotlib",
    "scipy":       "scipy",
}
fail = []
for mod, name in pkgs.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', 'unknown')
        print(f"  OK   {name:<18} {ver}")
    except ImportError:
        print(f"  FAIL {name}")
        fail.append(name)
if fail:
    print(f"\nMissing: {fail}")
    sys.exit(1)
else:
    print("\n  All packages verified.")
PYCHECK

ok "All packages installed and verified"

# ── Step 7: Freeze exact versions ──────────────────────────
info "Step 7/7 — Freezing versions → requirements_frozen.txt"
pip freeze > requirements_frozen.txt
COUNT=$(wc -l < requirements_frozen.txt)
ok "requirements_frozen.txt — ${COUNT} packages frozen"
warn "This file is gitignored — it stays local to this machine"

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo ""
echo "  Activate every new terminal session:"
echo -e "    ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo "  Run the full pipeline:"
echo -e "    ${YELLOW}python scripts/p25_pipeline_runner.py${NC}"
echo ""
echo "  Run a single script:"
echo -e "    ${YELLOW}python scripts/p25_data_pipeline.py${NC}"
echo ""
echo "  Deactivate when done:"
echo -e "    ${YELLOW}deactivate${NC}"
echo "============================================================"