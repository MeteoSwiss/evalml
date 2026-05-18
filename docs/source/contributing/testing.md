# Testing

EvalML uses `pytest`, with tests split into `tests/unit/` and
`tests/integration/`. The CI matrix covers Python 3.11–3.13 and a
"dev anemoi" leg that pulls `anemoi-inference` and `anemoi-datasets`
straight from `main`.

## Running tests

```bash
# Everything (the default).
uv run pytest tests/

# Unit tests only.
uv run pytest tests/unit

# Integration tests only.
uv run pytest tests/integration

# Skip slow tests.
uv run pytest -m "not longtest"
```

The `longtest` marker is configured in `pyproject.toml` for tests that
touch external services or take more than a few seconds.

## What's covered today

`tests/unit/` ships:

| File | Under test |
| --- | --- |
| `test_config.py` | Pydantic model loading of every example config; baseline ID derivation. |
| `test_colormaps.py` | NCL colormap parser, boundary norms, default colormap smoke tests. |
| `test_run_identity.py` | `env_id` / `run_id` hashing logic in `common.smk` (via the constants exported from `evalml.config`). |
| `test_spatial_mapping.py` | Spherical NN mapping, grid indexing, forecast-to-truth coordinate alignment. |

`tests/integration/test_experiment.py` is a placeholder for end-to-end
experiment validation — flesh it out as workflow logic stabilises.

## Fixtures

`tests/conftest.py` loads two example configs as fixtures so tests don't
need to repeatedly parse YAML:

- `forecaster_config_dict` — `config/forecasters-co2.yaml`
- `interpolator_config_dict` — `config/interpolators-co2.yaml`

Use these whenever a test needs a realistic config. They are loaded as
plain `dict`s so individual tests can mutate them before passing them to
`ConfigModel.model_validate(...)`.

## Adding a test

- Put it under `tests/unit/` if it can run in under a second without
  network or filesystem state outside the repo. Otherwise put it under
  `tests/integration/` and consider marking it `@pytest.mark.longtest`.
- Reuse the fixtures in `conftest.py` rather than re-loading example
  configs.
- For Snakemake-touching tests, prefer testing the underlying Python
  function rather than running Snakemake itself — most of the
  hash/registry logic in `common.smk` is small enough to test in
  isolation.
- For new metrics or loaders, add at least one test that exercises the
  end-to-end shape of the returned `xarray.Dataset`.

## What CI runs

[`.github/workflows/ci.yaml`](../../../.github/workflows/ci.yaml) has two
jobs:

- **`test`** — matrix over Python 3.11/3.12/3.13. The "anemoi-dev" leg is
  defined in the matrix but currently set to `false` only; switching it
  on in the matrix turns on a job that installs anemoi from `main`.
- **`lint`** — runs `pre-commit run --all-files --verbose`.

The docs build is its own workflow (see [CI](ci.md)).
