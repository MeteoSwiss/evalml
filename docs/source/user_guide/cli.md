# The `evalml` CLI

The CLI is a thin Click wrapper around Snakemake. Its job is to validate the
config, build a Snakemake command, and pass control to Snakemake. The
implementation lives in [src/evalml/cli.py](../../../src/evalml/cli.py).

## Auto-generated reference

The full command tree, including options and arguments, is rendered directly
from the Click definitions:

```{eval-rst}
.. click:: evalml.cli:cli
   :prog: evalml
   :nested: full
```

## How a command runs

All four subcommands (`experiment`, `showcase`, `sandbox`, `make`) call into
the same `execute_workflow` helper. The high-level flow is:

1. Parse the YAML file with `load_yaml`, then validate it via
   `ConfigModel.model_validate(...)`.
2. Build the base Snakemake command from `config.profile.parsable()` plus
   `--configfile` and `--cores`.
3. Append flags from CLI options:
   - `--dry-run` adds `--dry-run`.
   - `--unlock` adds `--unlock`.
   - `--report FILE` adds `--report-after-run --report FILE` (only if not a
     dry run).
4. Append the target rule name (e.g. `experiment_all`) and any extra Snakemake
   args passed after `--`.
5. Write the full command to `.evalml_snakemake_cmd.txt` so the Snakefile's
   `onstart:` hook can echo it back, then `subprocess.run(...)` it.

`--dag` and `--rulegraph` short-circuit step 4: instead of running the
workflow, they ask Snakemake for a Graphviz `dot` representation, send it to
[kroki.io](https://kroki.io) for rendering, and write `dag.svg` /
`rulegraph.svg` next to your config.

## Common option block

All four subcommands share the same option set, defined by the
`workflow_options` decorator in `cli.py`:

| Option | Default | Effect |
| --- | --- | --- |
| `--dry-run` / `-n` | off | Pass `--dry-run` to Snakemake; nothing is executed. |
| `--unlock` | off | Pass `--unlock` to release a stale Snakemake lock. |
| `--verbose` / `-v` | off | If unset, EvalML appends `--quiet rules` to dampen Snakemake output. |
| `--cores` / `-c` | `4` | Local cores Snakemake may use. SLURM jobs also respect `profile.jobs`. |
| `--report [FILE]` | none | Generate `<command>_report.html` (or the file you specify). |
| `--dag` | off | Render the full DAG via kroki.io. |
| `--rulegraph` | off | Render only the rule graph via kroki.io. |
| `-- EXTRA_SMK_ARGS` | none | Anything after `--` is forwarded to Snakemake verbatim. |

## Forcing re-runs

EvalML decides what to rebuild based on file timestamps and the hash
machinery in `common.smk`. When you need to override that — typically
because a checkpoint mutated in place, or because you want to test a
new metric on already-computed inputs — pass Snakemake's force flags
after `--`:

```bash
# Force EVERY rule to re-run, even if outputs exist.
evalml experiment config.yaml -- -F

# Force one specific rule (and its downstream dependents) to re-run.
evalml experiment config.yaml -- -R verification_metrics

# Force a rule for a specific wildcard combination via --until.
evalml experiment config.yaml -- --until verification_metrics
```

`-F` is `--forceall` (Snakemake), `-R` is `--forcerun`, and `--until`
stops execution after a named rule completes. These compose with the
other `-- …` forwards documented in the option table above.

The most common use is `evalml experiment config.yaml -- -F` after a
checkpoint or baseline archive has been mutated in place — see the
warning in [Configuration → run identity](configuration.md#what-is-not-hashed-checkpoint-contents).

## When to use which subcommand

- **`experiment`** — full evaluation: data prep + inference + verification +
  dashboard + plots.
- **`showcase`** — specific case-study visualisation: data prep + inference + visualisation (forecast animations, meteograms).
- **`sandbox`** — only build per-checkpoint inference sandboxes (squashfs +
  zip), useful when handing models to collaborators.
- **`make`** — drop down to a specific Snakemake target. Use this when
  debugging a single rule, e.g.:
  ```bash
  evalml make config.yaml inference_all
  ```
