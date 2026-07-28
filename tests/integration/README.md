## Inference fixture for `test_showcase_meteogram`

The showcase longtest replays a frozen copy of the forecaster inference GRIB
instead of running inference, so it needs no GPU/MLflow/sandbox build (it still
needs DWH access for truth). The fixture lives at the `fixture_root` set in
`configs/meteogram_small.yaml`.

**Create/refresh the fixture** (needed once, or whenever the checkpoint/config
changes — requires a GPU node):

1. Temporarily remove/comment `fixture_root` in `meteogram_small.yaml`.
2. Run a real showcase: `evalml showcase tests/integration/configs/meteogram_small.yaml`
3. Capture it:
   `evalml capture-fixture tests/integration/configs/meteogram_small.yaml <fixture_root>`
4. Restore `fixture_root`. Subsequent longtest runs replay from the fixture.

`capture-fixture` only snapshots GRIB dirs whose init time matches the config's
`dates`, so an unrelated experiment sharing the same `output/` tree is not swept
in. `MANIFEST.yaml` in the fixture records the checkpoint(s), the captured
dates, and the capture time.
