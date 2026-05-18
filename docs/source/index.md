# EvalML

EvalML runs evaluation pipelines for data-driven weather models built with
[Anemoi](https://anemoi.readthedocs.io/). It provides a small Click-based CLI
that drives a Snakemake workflow capable of:

- **Experiments** — compare model checkpoints against baselines and truth data
  through standard and diagnostic verification.
- **Showcases** — produce visual material (forecast animations, meteograms)
  for specific weather events.
- **Sandboxes** — package isolated inference environments for any Anemoi
  checkpoint, suitable for development or sharing with collaborators.

These docs are written for developers who need to understand, extend, or debug
the pipeline. If you only want to *run* an experiment, the
[Quickstart](getting_started/quickstart.md) is the fastest path; if you want to
understand how a rule works or how `run_id` is hashed, the
[Workflow](workflow/overview.md) section is for you.

## How the pieces fit together

```text
YAML config ─► evalml CLI ─► Snakemake ─► rules/*.smk ─► scripts/*.py ─► src/ ─► OUT_ROOT/
```

The Click CLI (`src/evalml/cli.py`) validates the YAML config against the
Pydantic models in `src/evalml/config.py`, builds a Snakemake invocation, and
launches the appropriate top-level target (`experiment_all`, `showcase_all`,
`sandbox_all`). Snakemake then resolves the dependency DAG defined in
`workflow/rules/*.smk`, executing rules whose scripts import the four
src-layout packages: `evalml`, `verification`, `data_input`, `plotting`.

```{toctree}
:caption: Getting started
:maxdepth: 2

getting_started/installation
getting_started/auth
getting_started/workspace
getting_started/quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 2

user_guide/cli
user_guide/configuration
user_guide/outputs
user_guide/examples
```

```{toctree}
:caption: Workflow
:maxdepth: 2

workflow/overview
workflow/data
workflow/inference
workflow/verification
workflow/plotting
workflow/reporting
workflow/conventions
```

```{toctree}
:caption: Python API
:maxdepth: 2

modules/index
modules/evalml
modules/verification
modules/data_input
modules/plotting
```

```{toctree}
:caption: Reference
:maxdepth: 2

reference/config_schema
reference/resources
reference/glossary
```

```{toctree}
:caption: Contributing
:maxdepth: 2

contributing/dev_setup
contributing/testing
contributing/ci
contributing/style
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
