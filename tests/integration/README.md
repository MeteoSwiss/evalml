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

**Create/refresh the fixture** (needed once, or whenever the checkpoint/config
changes — requires a GPU node):

1. Run a real showcase (no replay, so no config edits):
   `evalml showcase tests/integration/configs/meteogram_small.yaml`
2. Capture it:
   `evalml capture-fixture tests/integration/configs/meteogram_small.yaml <FIXTURE_ROOT>`

Subsequent longtest runs replay from the fixture automatically.

`capture-fixture` only snapshots GRIB dirs whose init time matches the config's
`dates`, so an unrelated experiment sharing the same `output/` tree is not swept
in. `MANIFEST.yaml` in the fixture records the checkpoint(s), the captured
dates, and the capture time.
