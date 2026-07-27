# Inference-fixture replay for the showcase longtest

**Date:** 2026-07-27
**Branch:** `feat/test-inference-fixture` (off `testing-hackathon`)
**Status:** design — awaiting review

## Problem

The one implemented integration `longtest`, `test_showcase_meteogram`
(`tests/integration/test_meteogram_small.py`), runs the full `evalml showcase`
pipeline end to end. Its slow, fragile part is **inference**: it needs a GPU,
MLflow credentials (to pull the checkpoint), and it builds an ~8 GB isolated
sandbox venv which it then packs with `mksquashfs` — the single step that
dominated a ~50 min cold run. None of that is what the test is checking.

The test only asserts `evalml showcase` exits `0` (snakemake fails if the
meteograms are not produced). It does **not** compare metric values, so there is
no numeric baseline to keep in sync.

## Goal

Let the workflow **replay frozen inference output** instead of computing it, so
the longtest exercises the deterministic pipeline tail (baseline read, truth
retrieval, meteogram plotting) without a GPU, MLflow, or the sandbox build.

**In scope:** freeze the forecaster **inference GRIB** only.
**Out of scope:** baseline (`ICON-CH2-EPS`) keeps reading `/store_new` — a plain
filesystem read available on the Balfrin CI runner; truth keeps coming live from
the DWH (`jretrievedwh`). Production behaviour is unchanged when the feature is
off.

## Principle

Cut the pipeline at a **replay seam** immediately after inference. Freeze the
inference output once to a filesystem fixture; when the workflow is pointed at
that fixture, `inference_execute` becomes a lightweight "stage the frozen GRIB"
step and the entire upstream sandbox-build chain drops out of the DAG.

## Fixture layout & storage

The fixture mirrors the `output/` tree so staging is a 1:1 path map. It lives on
the filesystem (not in the repo), at a configurable path:

```
<fixture_root>/                                            # /store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small/
  data/runs/<run_id>/<init_time>/grib/*.grib               # frozen forecaster inference (GBs)
  data/runs/<run_id>/<init_time>/config.yaml               # the anemoi run config produced alongside
  MANIFEST.yaml                                            # checkpoint id, config_label, capture date, run_ids
```

`MANIFEST.yaml` records what was frozen (checkpoint URL, config label, capture
date) so a stale fixture is diagnosable. No metric values are stored — the test
asserts only exit status.

## Config surface (opt-in)

A single top-level key in the experiment config, absent by default:

```yaml
fixture_root: /store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small   # set → replay inference from here
```

Global (not per-run): the `run_id`/`env_id`/`init_time` already namespace every
path underneath, so one root is sufficient and simpler. Added as an optional
field on the top-level config model in `src/evalml/config.py`. When unset,
every rule behaves exactly as today.

## Workflow changes

All confined to `workflow/rules/inference.smk`, gated on `fixture_root` being set.

1. **`inference_execute` input becomes conditional.** When `fixture_root` is
   set, the rule no longer takes the `venv.squashfs` image (nor the routing
   okfile) as input — it takes the frozen GRIB directory under `<fixture_root>`.
   This is the same conditional-input technique used elsewhere for optional
   behaviour; introduced here fresh since this base has no such flag yet.
   Because nothing downstream requests the squashfs, the whole sandbox chain
   (`inference_extract_requirements → prepare_env → create_sandbox →
   prepare_forecaster → make_squashfs`) drops out of the DAG automatically.

2. **`inference_execute` recipe becomes conditional.** When `fixture_root` is
   set, instead of `squashfs-mount … anemoi-inference`, it **symlinks** the
   frozen `grib/` into the run workdir and `touch`es the okfile. Symlink (not
   copy) so replaying gigabytes is instant and uses no extra disk.

3. Everything downstream (`data_extract_baseline`, `verification_metrics`,
   `plot_meteogram`, …) is unchanged — it finds its inputs already present.

## Capture command

A helper to produce/refresh a fixture from a real run:

```
evalml capture-fixture <config.yaml> <fixture_root>
```

After a normal (GPU) run has populated `output/`, it copies each run's
`grib/` + `config.yaml` into `<fixture_root>` following the layout above and
writes `MANIFEST.yaml`. Implemented as an `evalml` CLI subcommand
(`src/evalml/cli.py`) reusing the existing config loader to enumerate `run_id`s.
Copy (not move) so the source `output/` is left intact.

## Test integration

`tests/integration/configs/meteogram_small.yaml` gains
`fixture_root: <path>`. `test_showcase_meteogram` is otherwise unchanged — same
`evalml showcase` invocation, same `returncode == 0` assertion. With the fixture
present it needs only `/store_new` (baseline) + DWH (truth); no GPU, no MLflow,
no sandbox build. It stays marked `longtest` (still not hermetic — DWH), but the
runtime collapses from ~50 min to the CPU tail (minutes).

## Failure modes

- **Fixture missing / path wrong.** `inference_execute` should fail with a clear
  "fixture GRIB not found at <path> — run `evalml capture-fixture`" message
  rather than a bare snakemake missing-input error.
- **Stale fixture** (checkpoint/config changed since capture). Not auto-detected
  in v1; `MANIFEST.yaml` makes it diagnosable by hand. Auto-validation is a
  possible later addition, deliberately deferred (YAGNI).

## Verifying the mechanism

A fast unit/integration check that does not need a GPU: point a tiny config at a
small committed-or-generated fixture, run `evalml showcase`, assert the sandbox
rules did **not** run (e.g. no `venv.squashfs` produced) and the meteograms
were. This keeps the replay path from silently rotting.

## Out of scope / deferred

- Freezing baseline or truth (kept live per decision).
- Applying this to the heavytest / metric-comparison tests (would need
  `EXPECTED` re-recording; not part of this change).
- Automatic staleness detection of the fixture.
