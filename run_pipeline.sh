#!/usr/bin/env bash
# Full pipeline: feature extraction -> experiments -> CSV export -> uncertainty -> visualizations.
# Runs for both Pima Indians and Breast Cancer datasets.
#
# Usage:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --skip-experiments
#   ./run_pipeline.sh --skip-feature-extraction

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'

header()  { echo -e "\n${BOLD}${CYAN}== $1 ==${RESET}"; }
step()    { echo -e "\n${BOLD}${GREEN}> $1${RESET}"; }
ok()      { echo -e "  ${GREEN}done: $1${RESET}"; }
warn()    { echo -e "  ${YELLOW}skipped: $1${RESET}"; }
elapsed() { echo -e "  ${YELLOW}elapsed: $(( SECONDS - $1 ))s${RESET}"; }

VENV_DIR="${SCRIPT_DIR}/.venv"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo -e "${RED}ERROR: .venv not found. Run ./setup_venv.sh first.${RESET}"
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
echo -e "  ${GREEN}Using: $(python3 --version) at $(which python3)${RESET}"

SKIP_EXPERIMENTS=false
SKIP_FEATURE_EXTRACTION=false
for arg in "$@"; do
  case $arg in
    --skip-experiments)        SKIP_EXPERIMENTS=true ;;
    --skip-feature-extraction) SKIP_FEATURE_EXTRACTION=true ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

DATASETS=("pima" "breast_cancer")
PIPELINE_START=$SECONDS

header "LfD Research Pipeline"
echo "  Datasets: ${DATASETS[*]}"
echo "  Skip feature extraction: $SKIP_FEATURE_EXTRACTION"
echo "  Skip experiments: $SKIP_EXPERIMENTS"

# Step 0: Feature subset generation
if [ "$SKIP_FEATURE_EXTRACTION" = false ]; then
  header "Step 0: Feature Subset Generation"
  step "Installing mrmr-selection"
  pip install --quiet "mrmr-selection>=0.2.6" && ok "mrmr-selection ready"
  step "Generating feature subsets"
  T=$SECONDS
  python3 feature_extraction/generate_feature_subsets.py --dataset both
  ok "Feature subsets generated"; elapsed $T
else
  warn "Step 0 (--skip-feature-extraction)"
fi

# Step 1: Hyperparameter search
if [ "$SKIP_EXPERIMENTS" = false ]; then
  header "Step 1: Experiments"
  for DS in "${DATASETS[@]}"; do
    step "Running experiments: $DS"
    T=$SECONDS
    python3 experiments/run_all_experiments.py --dataset "$DS"
    ok "$DS"; elapsed $T
  done
else
  warn "Step 1 (--skip-experiments)"
fi

# Step 2: Export results CSV
header "Step 2: Export Results CSV"
for DS in "${DATASETS[@]}"; do
  step "Exporting: $DS"
  T=$SECONDS
  python3 experiments/export_results_csv.py --dataset "$DS"
  ok "$DS"; elapsed $T
done

# Step 3: Bootstrap uncertainty decomposition
header "Step 3: Bootstrap Uncertainty (M=30)"
for DS in "${DATASETS[@]}"; do
  step "Bootstrap: $DS"
  T=$SECONDS
  python3 experiments/uncertainty/bootstrap_uncertainty.py --dataset "$DS"
  ok "$DS"; elapsed $T
done

# Step 4: Export uncertainty CSV
header "Step 4: Export Uncertainty CSV"
for DS in "${DATASETS[@]}"; do
  step "Exporting uncertainty CSV: $DS"
  T=$SECONDS
  python3 experiments/uncertainty/export_uncertainty_csv.py --dataset "$DS"
  ok "$DS"; elapsed $T
done

# Step 5: Visualizations
header "Step 5: Visualizations"
step "Running all visualization scripts"
T=$SECONDS
python3 visualization/run_all_visualizations.py
ok "All figures saved to visualization/figures/"; elapsed $T

header "PIPELINE COMPLETE"
echo -e "  ${GREEN}Total time: $(( SECONDS - PIPELINE_START ))s${RESET}"
echo ""
echo "  Key outputs:"
echo "    results/pima_indians/all_results.csv"
echo "    results/breast_cancer/all_results.csv"
echo "    results/pima_indians/uncertainty_decomposition/uncertainty_results.csv"
echo "    results/breast_cancer/uncertainty_decomposition/uncertainty_results.csv"
echo "    visualization/figures/"
echo ""
