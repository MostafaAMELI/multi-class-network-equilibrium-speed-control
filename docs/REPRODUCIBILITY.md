# Reproducibility Guide

This document provides exact commands to reproduce the main computational outputs.

## 1) Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set Gurobi license:

```bash
export GRB_LICENSE_FILE=/absolute/path/to/gurobi.lic
```

## 2) Melbourne full-scale benchmark (paper-ready)

```bash
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
```

Generate comparison tables:

```bash
python3 python_port/melbourne_comparison_tables.py \
  --summary python_port/output/melbourne/summary_melbourne_full_completed.json \
  --out-dir python_port/output/melbourne
```

Generate scenario characteristics:

```bash
python3 python_port/melbourne_scenario_characteristics.py \
  --summary python_port/output/melbourne/summary_melbourne_full_completed.json \
  --out-dir python_port/output/melbourne
```

## 3) Expected outputs

- `python_port/output/melbourne/summary_melbourne_full_completed.json`
- `python_port/output/melbourne/table_indicators_melbourne.csv`
- `python_port/output/melbourne/table_costs_melbourne.csv`
- `python_port/output/melbourne/table_melbourne_overall.md`
- `python_port/output/melbourne/comparison_tables_melbourne_full.tex`

## 4) Known runtime caveat (WLS users)

If using Gurobi WLS, occasional DNS errors can occur:

`Could not resolve host: token.gurobi.com`

This is a licensing connectivity issue; rerun after connectivity is restored.

## 5) Tilburg run (Python)

```bash
python3 python_port/stm_mip_brue_v2.py \
  --all-scenarios \
  --time-limit 1800 \
  --output python_port/output/summary_v2_all.json
```
