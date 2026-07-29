## Inference fixture for `test_showcase_meteogram`

The showcase longtest replays a frozen copy of the forecaster inference GRIB
instead of running inference, so it needs no GPU/MLflow/sandbox build (it still
needs DWH access for truth). `meteogram_small.yaml` **always** replays: it sets
`fixture_root`, so any `evalml showcase` of this config uses the fixture. The
longtest reads that path from the config and is skipped if the fixture is not
populated (no `MANIFEST.yaml`). When `fixture_root` is active, the workflow's
start banner prints `Inference: REPLAYED FROM FIXTURE <path>`.

**Scope:** replay covers the showcase path only. The MEC verification path
(`--mec`, `verif_obs.smk`) depends on GRIB at *derived* source init times
(`init_time − lead`), which a showcase fixture does not contain, so with `--mec`
enabled fixture mode would still trigger a real inference (checkpoint download).
Replaying MEC/FFV2 runs is out of scope for this fixture.

**Create/refresh the fixture** (needed once, or whenever the checkpoint/config
changes — requires a GPU node):

1. Temporarily comment out `fixture_root` in `meteogram_small.yaml` (otherwise
   the config forces replay and won't run real inference), then run a real
   showcase: `evalml showcase tests/integration/configs/meteogram_small.yaml`
2. Capture it (leave `fixture_root` commented for this step, or pass the config
   as-is — capture ignores `fixture_root`):
   `evalml capture-fixture tests/integration/configs/meteogram_small.yaml <FIXTURE_ROOT>`
3. Restore (uncomment) `fixture_root` in `meteogram_small.yaml`.
4. Clear the real inference output the capture run produced, so replay does not
   collide with it: `rm -rf output/data/runs/<forecaster-run> output/logs/inference_execute/<forecaster-run>`
   (or just start replay from a clean `output/`). Replay deliberately refuses to
   overwrite a real, Snakemake-owned `grib/` directory.

Subsequent longtest runs replay from the fixture automatically (CI checks out a
clean tree, so this only matters when capturing and replaying in the same
`output/`).

`capture-fixture` only snapshots GRIB dirs whose init time matches the config's
`dates`, so an unrelated experiment sharing the same `output/` tree is not swept
in. `MANIFEST.yaml` records the checkpoint(s), captured dates, capture time, the
`evalml` commit, and a **SHA-256 per GRIB dir**. At replay the workflow
re-checks each fixture GRIB against its recorded checksum and fails loudly if it
has drifted (corrupted, partial, or hand-edited); fixtures captured before
checksums existed simply skip the check.

## Versioning and multiple fixtures

A fixture is **per-config**, not global: each fixture-backed test reads
`fixture_root` from its own config (there is no shared fixture constant). To add
another fixture-backed test, add a config with its own `fixture_root` and a test
that reads it, reusing the skip-if-no-`MANIFEST.yaml` pattern.

**Layout** — one directory per test-config under a shared parent, with each
capture in a dated subfolder:

```
/store_new/mch/msopr/cmerker/evalml_test_fixtures/
  meteogram-small/
    v_20240801_b30a/   # MANIFEST.yaml + data/runs/<run_id>/<init_time>/grib
    v_20260729_b30a/
  <other-test>/
    v_<date>_<ckpt>/
```

Each config's `fixture_root` points at its own `<test>/<version>` directory.

Versioning needs **no code change**: `fixture_root` is a self-contained,
relocatable bundle — `MANIFEST.yaml`, the `data/` tree, and the per-GRIB
checksums are all anchored relative to `fixture_root`. To refresh, capture into a
new `v_<date>` directory and bump the `fixture_root` line; the old version stays
for rollback. The pointer lives in git (the config), the data lives in
`/store_new` — so the commit that bumps `fixture_root` is tied to the
code/checkpoint change that required re-capturing.

The version tag is a **human pointer only**; a fixture's true identity
(checkpoint, dates, `evalml` commit) is recorded in `MANIFEST.yaml`. Date-only
(`v_20260729`) is fine for infrequent captures — append the checkpoint
short-hash (`v_20260729_b30a`) if you capture more than once a day or want the
reason visible at a glance.

Notes as this grows:

- **Isolation.** `run_id` is a hash of the forecaster config, so tests with
  different checkpoints/configs get different `run_id`s and never collide.
  Prefer one bundle per test — re-capturing a shared bundle would silently
  affect every test pointing at it.
- **Capture scope.** `capture-fixture` only snapshots GRIB dirs whose init time
  matches the config's `dates`, so capturing one fixture from a shared `output/`
  tree will not sweep in another config's runs. Refreshing all fixtures means
  running `capture-fixture` once per config.
- **Storage.** N tests x M versions of GRIB adds up on `/store_new`; prune stale
  versions. The git history of the `fixture_root` bumps shows which versions
  were ever active.
