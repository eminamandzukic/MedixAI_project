#!/usr/bin/env bash
# Creates a .venv/ in the project root and installs all dependencies.
# Run once, then use ./run_pipeline.sh to run everything.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${SCRIPT_DIR}/.venv"

# Find a suitable Python interpreter
PYTHON_BIN=""
for candidate in \
    "${HOME}/miniconda3/envs/uni/bin/python3" \
    "$(which python3.10 2>/dev/null)" \
    "$(which python3.11 2>/dev/null)" \
    "$(which python3.12 2>/dev/null)" \
    "$(which python3 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No suitable Python 3 interpreter found. Install Python 3.10+ and retry."
    exit 1
fi

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at .venv/ - skipping creation."
    echo "To rebuild: rm -rf .venv && ./setup_venv.sh"
else
    echo "Creating virtual environment at .venv/ ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "  Created."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
echo "Activated: $(python3 --version) at $(which python3)"

echo ""
echo "Upgrading pip ..."
pip install --upgrade pip --quiet

echo ""
echo "Installing dependencies from requirements.txt ..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "Verifying key packages:"
python3 -c "
import sklearn, numpy, pandas, matplotlib, seaborn
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  numpy        : {numpy.__version__}')
print(f'  pandas       : {pandas.__version__}')
print(f'  matplotlib   : {matplotlib.__version__}')
print(f'  seaborn      : {seaborn.__version__}')
"

echo ""
echo "Setup complete. Run the pipeline with: ./run_pipeline.sh"
