# `evalml` package

The `evalml` package contains the user-facing CLI, the Pydantic
configuration models, and a small handful of helpers used by Snakemake
scripts.

## Click CLI

The CLI is auto-rendered from the source on the
[The evalml CLI](../user_guide/cli.md) page — go there for the full
command tree with narrative.

## CLI plumbing (`evalml.cli`)

The Click commands themselves are thin; the interesting bits are the
`workflow_options` decorator, `execute_workflow`, and the
`generate_graph` helper that talks to kroki.io.

```{eval-rst}
.. automodule:: evalml.cli
   :members:
   :exclude-members: cli, experiment, showcase, sandbox, make
```

## Configuration models (`evalml.config`)

Every YAML config is validated through `ConfigModel`. The hierarchy:

- `ConfigModel` — top-level container.
- `Dates` / `ExplicitDates` — date specification.
- `RunConfig` (abstract base) → `ForecasterConfig`, `InterpolatorConfig`.
- `BaselineConfig`, `TruthConfig`, `Stratification`, `Locations`.
- `Profile` → `GlobalResources`, `DefaultResources`.
- `InferenceResources` — optional per-run override.

```{eval-rst}
.. automodule:: evalml.config
   :members:
   :show-inheritance:
   :exclude-members: model_config, model_fields, model_computed_fields
```

## Helpers (`evalml.helpers`)

```{eval-rst}
.. automodule:: evalml.helpers
   :members:
```

`setup_logger` is the recommended way to configure logging inside Snakemake
scripts — call it once near the top of a script with the rule's `log[0]`
path so all script output ends up in the rule's log file.
