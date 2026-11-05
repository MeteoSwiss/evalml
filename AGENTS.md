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
- Budget storage: ML inference writes ~5 GB of GRIB per forecast (LAM + global), i.e.
  ~850 GB per model for a year of forecasts every 54 h.
- Inference environments are built with uv; its package cache goes to `$SCRATCH/.cache/uv`
  when `$SCRATCH` is set. If the home quota fills up, check `du -sh ~/.cache/uv` and run
  `uv cache prune`.

**Running workflows on Balfrin**
- Launch long runs detached so they survive the session, e.g.
  `setsid nohup evalml experiment cfg.yaml > run.log 2>&1 &`.
- Keep config files on a shared filesystem (repo, `/scratch`), never node-local `/tmp`:
  SLURM jobs re-read the configfile.
- Run workflows in the same workdir one after the other, not at the same time.
- Before `evalml ... -- --unlock`, check that no snakemake of yours is still running
  (`pgrep -u $USER -af "snakemake --executor"`): a "stale" lock may belong to a live run.
- Dry-run (`-- --dry-run`) before (re)launching and check the job counts: an unexpected
  `inference_execute` count means inputs changed and forecasts would be recomputed.
- On failure, the end-of-run summary lists failed jobs and their logs. SLURM job logs are
  under `.snakemake/slurm_logs/rule_<rule>/`, rule logs under `output/logs/`; for SLURM
  states use `sacct -u $USER -S <start> -X --format=JobID,JobName,State,Elapsed`.
- Override resources of a rule with `-- --set-resources <rule>:runtime=<minutes>`.

**Evaluating new checkpoints**
- `runs[].forecaster.checkpoint` accepts a local path to `inference-last.ckpt`.
- Checkpoints trained from editable installs record `file://` sources: pin
  anemoi-inference, anemoi-datasets and anemoi-core (`#subdirectory=models` / `graphs`)
  in `extra_requirements` (take the commits from the training workspace's submodules) and
  check `output/data/runs/<run>/requirements.txt` before a large run.
- anemoi-core ≥ 0.19 checkpoints use the multidataset metadata layout; adapt an
  `sgm-multidataset-*` inference config and its patch in `resources/inference/metadata/`
  (e.g. to choose analysis vs reanalysis initial conditions).
- Smoke-test with one date before a full experiment.

**Reading outputs**
- `data/runs/<run>/verif_aggregated_*.nc`: the `source` coordinate holds internal IDs
  (forecast first, truth second), not labels.
- Statistics are averaged over reference times: `max`/`min` are the mean per-forecast
  domain extremes (not location-matched), `std` is the root of the mean spatial
  variance; contingency scores (FBI, ETS, ...) use counts summed over all forecasts.
- A scorecard's `baseline` can be a baseline or a forecaster label; a run is not
  scored against itself.
