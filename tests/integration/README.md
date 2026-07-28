## Inference fixture for `test_showcase_meteogram`

The showcase longtest replays a frozen copy of the forecaster inference GRIB
instead of running inference, so it needs no GPU/MLflow/sandbox build (it still
needs DWH access for truth). Replay is a **test-only switch**: the test passes
`--config fixture_root=<path>` at invocation (see `test_meteogram_small.py`), so
`meteogram_small.yaml` itself stays a real-run config — no editing needed to
capture. The fixture path is the `FIXTURE_ROOT` constant in
`test_meteogram_small.py`; the longtest is skipped if that directory is absent.
When `fixture_root` is active, the workflow's start banner prints
`Inference: REPLAYED FROM FIXTURE <path>`.

**Scope:** replay covers the showcase path only. The MEC verification path
(`--mec`, `verif_obs.smk`) depends on GRIB at *derived* source init times
(`init_time − lead`), which a showcase fixture does not contain, so with `--mec`
enabled fixture mode would still trigger a real inference (checkpoint download).
Replaying MEC/FFV2 runs is out of scope for this fixture.

**Create/refresh the fixture** (needed once, or whenever the checkpoint/config
changes — requires a GPU node):

1. Run a real showcase (no replay, so no config edits):
   `evalml showcase tests/integration/configs/meteogram_small.yaml`
2. Capture it:
   `evalml capture-fixture tests/integration/configs/meteogram_small.yaml <FIXTURE_ROOT>`
3. Clear the real inference output the capture run produced, so replay does not
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
