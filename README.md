# Multi-Class Network Equilibrium With Speed Control (Python)
Multi-class static traffic assignment under UE/BRUE with speed-control and emission-cost optimization.

## Included use cases

- Tilburg benchmark (text-format network/demand): `data/tilburg/`
- Melbourne GMNS benchmark: `testcases/4541153/`

## Main entry points

- Tilburg scenarios: `python_port/stm_mip_brue_v2.py`
- Melbourne scenarios: `python_port/stm_mip_brue_melbourne.py`
- Melbourne convenience runner: `python_port/run_melbourne.sh`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set Gurobi license:

```bash
export GRB_LICENSE_FILE=/absolute/path/to/gurobi.lic
```

## Reproduce Melbourne full run

```bash
./scripts/reproduce_melbourne_full.sh
```

Outputs are written under `python_port/output/melbourne/`.

## Reproduce Melbourne smoke run

```bash
./scripts/reproduce_melbourne_smoke.sh
```

## Reproduce Tilburg all scenarios

```bash
python3 python_port/stm_mip_brue_v2.py \
  --all-scenarios \
  --time-limit 1800 \
  --output python_port/output/summary_v2_all.json
```

## Tables and reporting helpers

```bash
python3 python_port/melbourne_comparison_tables.py \
  --summary python_port/output/melbourne/summary_melbourne_full_completed.json \
  --out-dir python_port/output/melbourne
```

```bash
python3 python_port/melbourne_scenario_characteristics.py \
  --summary python_port/output/melbourne/summary_melbourne_full_completed.json \
  --out-dir python_port/output/melbourne
```

## Project structure

- `python_port/` optimization + reporting code
- `scripts/` one-command reproducibility scripts
- `data/tilburg/` Tilburg inputs for Python workflows
- `testcases/4541153/` Melbourne GMNS inputs
- `docs/REPRODUCIBILITY.md` exact command log

## License

See `LICENSE`.
