#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/Downloads/gurobi.lic}"

python_port/run_melbourne.sh \
  --scenario UE \
  --od-mode single \
  --od-slice 800_815 \
  --top-od 300 \
  --k-paths 2 \
  --time-limit 120 \
  --output python_port/output/melbourne/summary_melbourne_smoke_ue.json

echo "Melbourne smoke run completed."
