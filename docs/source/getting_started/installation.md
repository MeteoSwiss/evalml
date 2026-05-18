# Installation

EvalML is distributed as a `uv`-managed Python project.

## Prerequisites

- A Linux environment; the workflow has been tested on CSCS Balfrin.
- [`uv`](https://github.com/astral-sh/uv) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A working `git`, `mksquashfs`, and `squashfs-mount` if you intend to run the
  inference rules on a SLURM cluster.

## Installing the project

Clone the repository and create the virtual environment with `uv`:

```bash
git clone https://github.com/MeteoSwiss/evalml.git
cd evalml
uv sync
source .venv/bin/activate
```

`uv sync` installs the runtime dependencies declared in `pyproject.toml`. Add
`--dev` to also install pre-commit and `snakefmt`, and `--group docs` to install
the documentation toolchain (Sphinx, theme, MyST). Both groups stack:

```bash
uv sync --dev --group docs
```

## Verifying the install

```bash
evalml --help
```

You should see the four top-level subcommands (`experiment`, `showcase`,
`sandbox`, `make`). If the entry point is missing, the project is not
installed in the active environment — re-run `uv sync` and ensure
`source .venv/bin/activate` was successful.

## Optional: Kerchunk extras

If you plan to read sharded Zarr datasets via Kerchunk references, install the
`kerchunk` optional group:

```bash
uv sync --extra kerchunk
```

This pulls `kerchunk`, `zarr<3.0.0`, `fastparquet`, and `ujson`. It is *not*
required for standard experiments.
