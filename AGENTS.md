# AGENTS.md

**Purpose**
EvalML runs evaluation pipelines for data-driven weather models (Anemoi). Features:
- **Experiments**: compare model performance via standard and diagnostic verification
- **Showcasing**: produce visual material for specific weather events
- **Sandboxing**: generate isolated inference development environments

The CLI `evalml` orchestrates Snakemake workflows in `workflow/` using YAML experiment configs.

**Repo Layout**
- `src/evalml/` — CLI (`cli.py`), config models (`config.py`), helpers
- `src/verification/` — metrics and verification logic (`spatial.py`)
- `src/data_input/` — data loading and ingestion
- `src/plotting/` — visualization and colormap handling
- `workflow/` — Snakemake pipeline (`Snakefile`, `rules/`, `scripts/`, `envs/`, `tools/`)
- `config/` — example experiment configs
- `tests/` — unit and integration tests
- `output/` — default workflow output location (often a symlink to scratch)

**Setup**
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Install dependencies (including dev tools): `uv sync --dev`
- Activate the venv: `source .venv/bin/activate`
- Install pre-commit hooks: `pre-commit install`
- Some experiments require credentials; coordinate with maintainers to obtain access.

**Common Commands**
- Run an experiment: `evalml experiment path/to/config.yaml --report`
- Validate configs against schema: use `workflow/tools/config.schema.json` in your YAML editor
- EvalML is a thin wrapper over Snakemake; pass Snakemake options after `--` (e.g. `evalml experiment config.yaml -- --dry-run -j 1`)

**Configuration**
Experiment YAML files are validated by Pydantic. Key fields:
- `dates` — date range or explicit list of reference times
- `runs` — ML model runs referenced by MLflow ID
- `baselines` — reference forecasts for comparison
- `truth` — ground truth dataset
- `locations` — output paths and MLflow URIs
- `profile` — executor config (e.g. SLURM)

**Testing**
- Run unit tests: `pytest tests/unit`
- Run integration tests: `pytest tests/integration`
- Skip long tests: `pytest -m "not longtest"`
- For full workflow tests, use a minimal config to keep runs fast:
  - Copy a sample config from `config/` (e.g. `config/minimal-test.yaml`)
  - Reduce `dates` to 1–2 reference times, `runs` to 1–2 models, and steps to a few lead times
  - Run the workflow with that minimal config

**Formatting and QA**
- If editing Snakemake files, run `snakefmt workflow`
- Run `pre-commit run --all-files` before large changes (checks ruff, snakefmt, schema validation)

**Data and Outputs**
- Workflow outputs default to `output/`. Avoid committing generated data.
- Prefer using a scratch-backed symlink for `output/` when running large jobs.
