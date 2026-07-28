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
in. `MANIFEST.yaml` in the fixture records the checkpoint(s), the captured
dates, and the capture time.
