# Plan: decouple "baseline" role from model/data-source type

Status: **not started** — design only, no code changes made yet.

## Problem

Today, `runs:` entries in a config are one of two disjoint kinds:

- A **run** (`forecaster` / `temporal_downscaler` / `spatial_downscaler`): the
  model(s) under evaluation. Staged/produced by the Snakemake `inference.smk`
  pipeline into `data/runs/{run_id}/{init_time}/grib`, then scored by
  `verification_metrics`.
- A **baseline** (`baseline:`, backed by `BaselineConfig`): the reference
  model(s) everything else is compared against in scorecards. Never staged;
  read live from an ICON/COSMO operational archive or an INCA NetCDF tree via
  hardcoded path-sniffing in `load_forecast_data`, and scored by the parallel
  `verification_metrics_baseline` rule.

This conflates two independent things:

1. **Data-source type** — *how* you obtain a model's GRIB (run
   anemoi-inference, stage pre-generated flat GRIB, or read a live ICON/INCA
   operational archive with a `member` selector).
2. **Role** — whether a model is the thing under evaluation or the reference
   other models are scored against.

Because role is currently baked into the type (`BaselineConfig` is the only
type usable as a baseline, and it's ICON/INCA-archive-shaped only), a
pre-generated-GRIB model (e.g. HiRAD, `spatial_downscaler`) can never be used
as a baseline, and an ICON/INCA archive model can never be evaluated as "the
experiment." See prior investigation in this thread for the concrete failure
modes (schema rejection on `member` under `spatial_downscaler:`, GRIB staging
mismatch, hardcoded `"ICON-CH1-EPS"`/`"ICON-CH2-EPS"` path checks in
`_collect_icon_archive_files`).

## Goal

Any model — regardless of how its GRIB is obtained — can be marked as either
the experiment (default) or the baseline, via a config field rather than
which YAML key wraps it.

## Decisions already made

- **Output namespace**: keep `data/runs/{run_id}/...` and
  `data/baselines/{baseline_id}/...` as two separate namespaces. Role
  continues to decide which namespace an entry's outputs land in; it does not
  collapse into one tree. This keeps the existing duplicated-rule-family
  structure in `verification.smk`/`plot.smk` largely intact, at the cost of
  each rule family needing to internally branch on data-source type (see
  below) instead of getting that for free from which rule family runs.
- **Migration**: add a temporary backward-compatibility shim so existing
  configs using the legacy `baseline:` key keep parsing unchanged; migrate
  configs off it opportunistically, then remove the shim in a follow-up once
  nothing uses it.

## Design

### 1. `src/evalml/config.py`

- Add `baseline: bool = False` to `RunConfig` (the common base class), so
  `CheckpointRunConfig`/`GRIBRunConfig` and everything under them inherit it.
  Any `forecaster:`/`temporal_downscaler:`/`spatial_downscaler:` entry can set
  `baseline: true`.
- Promote the ICON/INCA-archive-reading logic currently hidden inside
  `BaselineConfig` (`root` + `member`, path-sniffed in `load_forecast_data`)
  into a real `GRIBRunConfig`-family subclass, e.g. `ArchiveRunConfig(root,
  member)`, wrapped as a new `ArchiveItem` (`archive_model:` key). This is
  the type ICON-CH1-CTRL would use if configured as an experiment run.
- `ConfigModel.runs` union grows to include `ArchiveItem`, alongside
  `ForecasterItem | TemporalDownscalerItem | SpatialDownscalerItem`.
- Keep `BaselineConfig`/`BaselineItem` around, marked deprecated, as the input
  shape the compat shim below still accepts.
- Add a `model_validator(mode="before")` (or equivalent pre-parse step) that
  rewrites any legacy `{"baseline": {...}}` list entry into
  `{"archive_model": {..., "baseline": True}}` before the rest of validation
  runs. Log/warn when it fires so migration pressure stays visible.
- Decide (open question, see below) whether `member` stays exclusive to
  `ArchiveRunConfig` or becomes a no-op-if-absent field elsewhere.
- Regenerate `workflow/tools/config.schema.json` from the updated model.

### 2. `workflow/rules/common.smk` — collection

- `collect_all_runs()` / `collect_all_baselines()`
  ([common.smk:264-317](../../workflow/rules/common.smk#L264-L317)) stay as
  two functions producing two dicts (`RUN_CONFIGS` / `BASELINE_CONFIGS`), but
  the selector changes from `model_type == "baseline"` /
  `"baseline" in run_entry` to `run_config.get("baseline", False)` —
  role-driven, not key-driven.
- `collect_all_baselines()` currently bypasses `register_run()` entirely and
  hashes inline via `baseline_hash()`
  ([common.smk:306-317](../../workflow/rules/common.smk#L306-L317)) — safe
  today only because a baseline is always archive-shaped. Once a
  `spatial_downscaler`-type entry can be a baseline, it needs the same
  `env_id`/`run_id` bookkeeping a run gets, so `collect_all_baselines()`
  starts calling `register_run()` like `collect_all_runs()` does.
- `register_run()`'s type dispatch
  ([common.smk:204-261](../../workflow/rules/common.smk#L204-L261)), keyed
  today on `model_type in GRIB_MODEL_TYPES`, extends with the new archive
  bucket (hashed on `root` + `member` + `steps`, no checkpoint). This part is
  already type-driven rather than role-driven, so it changes shape the least.

### 3. `workflow/rules/inference.smk` — staging

- Baselines never touch this file today — they're skipped outright at
  [common.smk:269-270](../../workflow/rules/common.smk#L269-L270). Once a
  staged-GRIB or inference-based entry can be a baseline, it needs staging
  *into the baselines namespace*: a `grib_model_stage_baseline` /
  `inference_execute_baseline` sibling via Snakemake's
  `use rule ... as ... with:` (the same pattern already used for
  `verification_metrics_aggregation_baseline`,
  [verification.smk:143-156](../../workflow/rules/verification.smk#L143-L156)),
  targeting `data/baselines/{baseline_id}/{init_time}/grib`. Live-archive
  types still need no staging in either namespace.
- `_okfile_for` / `_okfile_template`
  ([inference.smk:424-437](../../workflow/rules/inference.smk#L424-L437))
  generalize from a 2-way dispatch (staged-GRIB vs. inference) to a
  (data-source type × namespace) dispatch.

### 4. `workflow/rules/verification.smk`

- `verification_metrics` / `verification_metrics_baseline` (plus their
  `_aggregation` / `_scoremaps` siblings) currently assume "non-baseline ⇒
  staged, no member" and "baseline ⇒ live-archive, has member." That
  assumption breaks once role and data-source type are independent — each
  rule's `--forecast` / `--member` param computation needs to branch on the
  entry's `model_type`, not on which rule family (`_baseline` suffix or not)
  it happens to run under.
  - Staged-GRIB-type entries (either family): `--forecast` = the staged
    per-init-time grib dir (`data/{runs,baselines}/{id}/{init_time}/grib`).
  - Live-archive-type entries (either family): `--forecast` = the
    unresolved configured `root`, `--member` = configured member; let
    `load_forecast_data`'s internal `FCST{yy}`/reftime-dir resolution handle
    the rest, same as today's baseline path.

### 5. `workflow/rules/plot.smk`, `workflow/rules/report.smk`, `workflow/Snakefile`

- These already key off `BASELINE_CONFIGS` / `baseline_id` /
  `resolve_baseline_id()` — they mostly keep working unchanged, since the
  dict just now can contain any data-source type, not only archive-style
  baselines. Spot-check `_get_available_baselines()`
  ([plot.smk:12-25](../../workflow/rules/plot.smk#L12-L25)), which reads
  `root`/`steps` directly without staging — currently fine only for
  archive/flat-GRIB-at-root layouts; verify it still makes sense once a
  staged-GRIB baseline (needing the resolved per-init-time dir, not the bare
  configured `root`) is possible.

### 6. Migration of existing configs

9 configs currently use the legacy `baseline:` key: `hirad.yaml`,
`forecasters-ich1.yaml`, `forecasters-ich1-oper.yaml`,
`forecasters-ich1-oper-fixed.yaml`, `forecasters-ich1_mec_ffv2.yaml`,
`varda-single-1.0.yaml`, `scoremaps_small.yaml`, `dashboard_small.yaml`,
`meteogram_small.yaml` (all under `tests/integration/configs/`). The compat
shim (§1) keeps them parsing as-is; migrate each to
`archive_model: {..., baseline: true}` opportunistically as touched, then
drop the shim once none remain on the old key.

## Open questions to resolve before implementation

- Exact naming: `archive_model` vs. some other key for the promoted
  ICON/INCA-archive type; `baseline: bool` vs. a `role:` enum (bool is
  simpler and matches every consumption site being a boolean check today).
- Should `resolve_baseline_id()`
  ([common.smk:320-334](../../workflow/rules/common.smk#L320-L334)) validate
  that the matched label's entry actually has `baseline: true` (catches a
  scorecard `baseline:` reference pointing at a non-baseline label), or keep
  matching by label alone?
- `steps` validation: `BaselineConfig.steps` uses a looser pattern
  (`^\d*/\d*/\d*$`, allows empty segments) than `RunConfig`'s
  `_validate_steps_range` (requires valid positive ints, start ≤ end). Decide
  whether `ArchiveRunConfig` should adopt the stricter shared validator or
  keep the looser one, and why the looser one existed in the first place.
- Whether `_get_available_baselines()` in `plot.smk` needs updating for
  staged-GRIB baselines (see §5), or whether that's out of scope for a first
  pass and staged-GRIB baselines simply aren't supported by showcase
  plotting initially.

## Suggested implementation order

1. `config.py` schema changes + compat shim + regenerated JSON schema, with
   unit tests for both new-style and legacy-style configs.
2. `common.smk` collection/hashing changes (§2) — verify `RUN_CONFIGS` /
   `BASELINE_CONFIGS` populate correctly for a config exercising all four
   data-source types in both roles.
3. `inference.smk` staging siblings for baseline-namespace staged GRIB (§3).
4. `verification.smk` branching (§4) — this is the piece with the most
   behavioral risk since it changes what each existing rule does, not just
   what feeds it.
5. `plot.smk` / `report.smk` spot-checks (§5).
6. Migrate the 9 existing test configs off the legacy `baseline:` key one at
   a time, confirming each still produces identical output before/after.
7. Remove the compat shim and `BaselineConfig`/`BaselineItem` once no config
   uses them.
