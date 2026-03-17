# AGENTS.md

**Purpose**
EvalML runs evaluation pipelines for data-driven weather models (Anemoi). The CLI `evalml` orchestrates Snakemake workflows in `workflow/` using YAML experiment configs.

**Repo Layout**
- `src/` Python packages (`evalml`, `verification`, `data_input`).
- `workflow/` Snakemake pipeline (`Snakefile`, `rules/`, `scripts/`, `envs/`, `tools/`).
- `config/` Example experiment configs.
- `tests/` Unit and integration tests.
- `output/` Default workflow output location (often a symlink to scratch).

**Setup**
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Install dependencies (including dev tools like pre-commit) with `uv sync --dev`.
- Activate the venv with `source .venv/bin/activate`.
- Some experiments require credentials; coordinate with maintainers to obtain access.

**Common Commands**
- Run an experiment: `evalml experiment path/to/config.yaml --report`.
- Validate configs against schema: use `workflow/tools/config.schema.json` in your YAML editor.
- EvalML is a thin wrapper over Snakemake; pass Snakemake options after `--` (e.g. `evalml experiment config.yaml -- --dry-run -j 1`).

**Testing**
- Run unit tests: `pytest tests/unit`.
- Run integration tests: `pytest tests/integration`.
- Skip long tests: `pytest -m "not longtest"`.
- For full workflow tests, use a minimal config to keep runs fast:
- Copy a sample config from `config/` to a new file (e.g. `config/minimal-test.yaml`).
- Reduce `dates` to a few reference times (e.g. 1–2 days).
- Reduce `runs` to 1–2 models.
- Reduce the forecast range/steps to a few lead times.
- Run the workflow with that minimal config.

**Formatting and QA**
- If editing Snakemake files, run `snakefmt workflow`.
- If pre-commit hooks are installed, run `pre-commit run --all-files` before large changes.

**Data and Outputs**
- Workflow outputs default to `output/`. Avoid committing generated data.
- Prefer using a scratch-backed symlink for `output/` when running large jobs.
