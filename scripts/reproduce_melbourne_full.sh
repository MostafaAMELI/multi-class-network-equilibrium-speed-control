#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/Downloads/gurobi.lic}"

python_port/run_melbourne.sh \
  --all-scenarios \
  --od-mode aggregate \
  --top-od 30000 \
  --k-paths 2 \
  --bpr-piece-l 1 \
  --bpr-piece-r 2 \
  --em-piece 3 \
  --time-limit 1800 \
  --output python_port/output/melbourne/summary_melbourne_full_completed.json

python3 python_port/melbourne_comparison_tables.py \
  --summary python_port/output/melbourne/summary_melbourne_full_completed.json \
  --out-dir python_port/output/melbourne

python3 python_port/melbourne_scenario_characteristics.py \
  --summary python_port/output/melbourne/summary_melbourne_full_completed.json \
  --out-dir python_port/output/melbourne

echo "Melbourne reproducibility pipeline completed."
