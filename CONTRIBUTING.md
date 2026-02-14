# Contributing

Thank you for contributing.

## Development setup

1. Create a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Configure `GRB_LICENSE_FILE`.

## Coding guidelines

- Keep scripts deterministic and parameterized by CLI arguments.
- Prefer adding new utilities under `python_port/` with clear `--help`.
- Document any new experiment command in `docs/REPRODUCIBILITY.md`.

## Pull requests

- Include a short summary of scientific/technical impact.
- If outputs change, describe why and provide the exact command used.
