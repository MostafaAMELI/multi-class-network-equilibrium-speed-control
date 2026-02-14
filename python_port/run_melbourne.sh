#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/Users/ameli/Downloads/gurobi.lic}"

cd "$PROJECT_ROOT"
python3 python_port/stm_mip_brue_melbourne.py "$@"
