# Inference

`workflow/rules/inference.smk` is the most involved module. It covers four
distinct concerns, in this order:

1. **Resolve** the checkpoint by URI type (MLflow / Hugging Face / local).
2. **Build** a relocatable `uv` virtual environment from the checkpoint's
   metadata and any `extra_requirements`, then snapshot it into a squashfs
   image.
3. **Package** the venv + checkpoint + config + Jinja2 README into a
   shareable sandbox zip.
4. **Execute** inference per (run_id, init_time) by mounting the squashfs
   image with `squashfs-mount` and running `anemoi-inference run` over
   `srun`.

## Rule reference

### `inference_get_checkpoint`

| Property | Value |
| --- | --- |
| Local rule | yes |
| Outputs | `data/runs/{env_id}/inference-last.ckpt`, `data/runs/{env_id}/anemoi.json` |
| Log | `logs/inference_prepare_checkpoint/{env_id}.log` |

The shell block branches on the checkpoint URI type:

- **MLflow** (`mlflow.ecmwf.int`, `service.meteoswiss.ch`,
  `servicedepl.meteoswiss.ch`) — symlinks the file resolved by
  `scripts/inference_get_checkpoint_mlflow.py`.
- **Hugging Face** (`huggingface.co/...ckpt`) — extracts `repo_id` and
  `file_path` from the URL with regex, then `cp $(uvx hf download ...)`.
- **Local path** — symlinks the path directly.

After the checkpoint is in place, `anemoi-utils metadata --dump --json`
extracts the metadata blob into `anemoi.json`.

### `inference_extract_requirements`

Reads `anemoi.json` to recover the Python version and dependency list the
checkpoint was trained with, merges the user's `extra_requirements`, and
writes a `requirements.txt`. The merging logic lives in
`scripts/inference_extract_requirements.py`.

### `inference_create_venv`

Creates a relocatable virtualenv with `uv`:

```bash
uv venv --managed-python --python $PYTHON_VERSION --relocatable --link-mode=copy {output.venv}
uv pip install -r {input.requirements}
python -m compileall -j 8 -o 0 -o 1 -o 2 .venv/lib/python*/site-packages
```

The compile pass produces `.pyc` files at all three optimisation levels so
import time stays low when the venv is mounted via squashfs. A final `import
eccodes` smoke-test ensures the GRIB stack is wired up before the venv is
snapshotted.

The output is `temp(directory(...))`, so Snakemake removes the venv after
the squashfs is created.

### `inference_make_squashfs_image`

Wraps the venv into `data/runs/{env_id}/venv.squashfs`:

```bash
mksquashfs $(realpath {input.venv}) {output.image} -no-recovery -noappend -Xcompression-level 3
```

This is the artefact that `inference_execute` later mounts on the compute
nodes.

### `inference_create_sandbox`

Bundles the checkpoint, requirements, inference config, and a rendered
README (from `resources/inference/sandbox/readme.md.jinja2`) into a single
zip. The result is suitable for handing a checkpoint plus reproducible
runtime to an external collaborator.

### `inference_prepare_forecaster` / `inference_prepare_interpolator`

Both rules call `scripts/inference_prepare.py`. They produce the per-init
`config.yaml`, a `resources/` directory of GRIB templates, and a `grib/`
output directory. The interpolator variant additionally:

- Depends on the upstream forecaster's `inference_execute` `.ok` file, when
  a `forecaster:` block is present.
- Symlinks the forecaster's output directory into `forecaster/` so the
  interpolator inference run can find it.
- Sets `params.forecaster_run_id`, used by the prepare script to wire the
  upstream input.

### `inference_execute`

The actual run. It depends on the appropriate `inference_prepare_*` `.ok`
file (selected by `_inference_routing_fn`) and the squashfs image, and
launches:

```bash
squashfs-mount {image}:/user-environment -- bash -c '
  source /user-environment/bin/activate
  if [ "{disable_local_definitions}" = "False" ]; then
    export ECCODES_DEFINITION_PATH=/user-environment/share/eccodes-cosmo-resources/definitions
  fi
  srun --partition=... --cpus-per-task=... ... anemoi-inference run config.yaml
'
```

Key points:

- Resources (partition, CPUs, runtime, GPUs) are resolved per run via
  `get_resource(wc, field, default)`, which falls back to sensible
  defaults (`short-shared`, 24 CPUs, 8 GB/CPU, 40 min, 1 GPU).
- For multi-GPU runs (`gpus > 1`), `runner.parallel.cluster=slurm` is
  appended to the inference command so anemoi-inference uses SLURM for
  distributed coordination.
- Output is signalled by touching
  `logs/inference_execute/{run_id}-{init_time}.ok`. Downstream rules depend
  on the `.ok` rather than on the GRIB directory itself, which keeps DAG
  invalidation predictable.

## Identity contract

`env_id` and `run_id` are computed in
[common.smk](../../../workflow/rules/common.smk) by `register_run`,
`env_entry_hash`, and `run_specific_hash`. The fields that go into each
hash are not free-form — they are explicitly listed in `RunConfig.ENV_FIELDS`
and consumed in `common.smk`:

- `env_entry_hash` hashes only `ENV_FIELDS`. For interpolators, it also
  appends the upstream forecaster's `env_id` so a different upstream model
  forces a new venv.
- `run_specific_hash` hashes `steps` plus the contents of the inference
  config YAML. For interpolators, it appends the forecaster's `run_id` so
  the interpolator's outputs reflect *which* forecaster run it consumed.

These hashes drive every output path under `data/runs/`, so a violation of
the contract (e.g. forgetting to include a new field in `ENV_FIELDS`) shows
up as silently reused environments. Tests in `tests/unit/test_run_identity.py`
guard the contract.
