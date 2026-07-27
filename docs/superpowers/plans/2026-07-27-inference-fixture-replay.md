# Inference-fixture Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the showcase longtest replay frozen inference GRIB from a filesystem fixture instead of running inference, removing the GPU, MLflow, and sandbox-build cost.

**Architecture:** Add an opt-in top-level `fixture_root` config key. When set, the `inference_execute` snakemake rule symlinks pre-captured GRIB into the run workdir and touches its okfile instead of building a sandbox and running `anemoi-inference` — so the whole sandbox chain drops out of the DAG. A new `evalml.fixtures` module holds the (pure, unit-tested) path + copy logic, shared by a new `evalml capture-fixture` CLI command and by `common.smk`. The fixture mirrors the `output/data/runs/<run_id>/<init_time>/grib` layout so capture and replay agree by construction.

**Tech Stack:** Python 3.11+, pydantic v2 (config models), click (CLI), snakemake (workflow), pytest.

## Global Constraints

- Production behaviour must be unchanged when `fixture_root` is unset — every rule behaves exactly as today.
- Scope is **inference GRIB only**. Baseline keeps reading `/store_new`; truth keeps coming from the DWH (`jretrievedwh`).
- Fixture storage root (this test): `/store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small`.
- `ConfigModel` has `model_config = {"extra": "forbid"}` — any new config key MUST be added as a model field or validation fails.
- `run_id` contains a `/` (`<env_id>/<run_hash>`); treat it as a relative path, never split it.
- The evalml `src/` packages are importable inside the snakemake venv (`common.smk` already does `from data_input.jretrieve import ...`), so `common.smk` may `from evalml.fixtures import ...`.
- Do not push or open PRs. Commit locally per task.

---

### Task 1: Add `fixture_root` to the config model

**Files:**
- Modify: `src/evalml/config.py` (class `ConfigModel`, ~line 603)
- Modify (regenerate): `workflow/tools/config.schema.json`
- Test: `tests/unit/test_config_fixture_root.py`

**Interfaces:**
- Produces: `ConfigModel.fixture_root: Path | None` (default `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_fixture_root.py
from pathlib import Path
import pytest
from evalml.config import ConfigModel


def _minimal_config(**overrides):
    cfg = {
        "description": "t",
        "dates": ["2024-01-01T00:00"],
        "runs": [
            {
                "baseline": {
                    "label": "ICON-CH2-EPS",
                    "root": "/store_new/x",
                    "steps": "0/6/6",
                }
            }
        ],
        "truth": {"label": "SwissMetNet", "root": "jretrievedwh:1,2"},
        "experiment": {"params": ["T_2M"], "stratification": {"regions": []}},
        "locations": {"output_root": "output/"},
        "profile": {
            "executor": "slurm",
            "global_resources": {"gpus": 1},
            "default_resources": {"slurm_partition": "postproc"},
            "jobs": 1,
        },
    }
    cfg.update(overrides)
    return cfg


def test_fixture_root_defaults_to_none():
    model = ConfigModel(**_minimal_config())
    assert model.fixture_root is None


def test_fixture_root_is_parsed_as_path():
    model = ConfigModel(**_minimal_config(fixture_root="/store_new/fx/meteogram-small"))
    assert model.fixture_root == Path("/store_new/fx/meteogram-small")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config_fixture_root.py -v`
Expected: FAIL — `test_fixture_root_is_parsed_as_path` raises a pydantic `ValidationError` ("Extra inputs are not permitted: fixture_root") because the field does not exist yet. (If the minimal config itself is invalid, fix the fixture dict first until `test_fixture_root_defaults_to_none` passes and the second test fails only on the extra key.)

- [ ] **Step 3: Add the field**

In `src/evalml/config.py`, inside `class ConfigModel`, add after the `locations: Locations` line:

```python
    fixture_root: Path | None = Field(
        None,
        description=(
            "Opt-in test/dev setting. When set, inference is not run: the "
            "workflow replays frozen inference GRIB from this directory "
            "(layout: <fixture_root>/data/runs/<run_id>/<init_time>/grib). "
            "Populate it with `evalml capture-fixture`. Leave unset in production."
        ),
    )
```

`Path` and `Field` are already imported at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config_fixture_root.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Regenerate the JSON schema**

Run: `uv run python src/evalml/config.py workflow/tools/config.schema.json`
Then confirm the key is present:
Run: `grep -n "fixture_root" workflow/tools/config.schema.json`
Expected: a `"fixture_root"` property appears in the schema.

- [ ] **Step 6: Commit**

```bash
git add src/evalml/config.py workflow/tools/config.schema.json tests/unit/test_config_fixture_root.py
git commit -m "feat(config): add opt-in fixture_root for inference replay"
```

---

### Task 2: `fixture_grib_dir` path helper

**Files:**
- Create: `src/evalml/fixtures.py`
- Test: `tests/unit/test_fixtures.py`

**Interfaces:**
- Produces: `fixture_grib_dir(fixture_root: str | Path, run_id: str, init_time: str) -> Path` — the frozen GRIB directory for one run/init.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fixtures.py
from pathlib import Path
from evalml.fixtures import fixture_grib_dir


def test_fixture_grib_dir_mirrors_output_layout():
    got = fixture_grib_dir("/fx", "forecaster-abcd/6640", "202503010000")
    assert got == Path("/fx/data/runs/forecaster-abcd/6640/202503010000/grib")


def test_fixture_grib_dir_accepts_path_and_int_init():
    got = fixture_grib_dir(Path("/fx"), "temporal-x-on-forecaster-abcd/1a2b", 202503010000)
    assert got == Path("/fx/data/runs/temporal-x-on-forecaster-abcd/1a2b/202503010000/grib")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalml.fixtures'`.

- [ ] **Step 3: Create the module with the helper**

```python
# src/evalml/fixtures.py
"""Produce and consume frozen inference GRIB fixtures for tests/dev.

The fixture mirrors the pipeline's own output layout
``<root>/data/runs/<run_id>/<init_time>/grib`` so that capture (writing the
fixture from a real run) and replay (reading it back) agree by construction.
"""

from pathlib import Path


def fixture_grib_dir(fixture_root, run_id: str, init_time) -> Path:
    """Return the frozen GRIB directory for one run/init inside a fixture.

    ``run_id`` contains a '/' (``<env_id>/<run_hash>``) and is used verbatim.
    """
    return Path(fixture_root) / "data" / "runs" / run_id / str(init_time) / "grib"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_fixtures.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/evalml/fixtures.py tests/unit/test_fixtures.py
git commit -m "feat(fixtures): add fixture_grib_dir path helper"
```

---

### Task 3: `capture_fixture` + manifest, with capture→replay round-trip

**Files:**
- Modify: `src/evalml/fixtures.py`
- Test: `tests/unit/test_fixtures.py`

**Interfaces:**
- Consumes: `fixture_grib_dir` (Task 2).
- Produces:
  - `iter_grib_dirs(output_root: str | Path) -> list[Path]` — every `grib/` dir under `<output_root>/data/runs`.
  - `capture_fixture(output_root, fixture_root) -> list[Path]` — copies each into the fixture, returns destination dirs.
  - `write_manifest(fixture_root, *, config_label, checkpoints, captured_at, grib_dirs) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_fixtures.py
import yaml
from evalml.fixtures import (
    fixture_grib_dir,
    iter_grib_dirs,
    capture_fixture,
    write_manifest,
)


def _fake_run(output_root: Path, run_id: str, init_time: str):
    grib = output_root / "data" / "runs" / run_id / init_time / "grib"
    grib.mkdir(parents=True)
    (grib / "202503010_0.grib").write_bytes(b"GRIB-DATA")


def test_iter_grib_dirs_finds_all_runs(tmp_path):
    out = tmp_path / "output"
    _fake_run(out, "forecaster-abcd/6640", "202503010000")
    _fake_run(out, "temporal-x-on-forecaster-abcd/1a2b", "202503010000")
    found = iter_grib_dirs(out)
    assert len(found) == 2
    assert all(p.name == "grib" for p in found)


def test_capture_then_replay_paths_match(tmp_path):
    out = tmp_path / "output"
    fx = tmp_path / "fixture"
    _fake_run(out, "forecaster-abcd/6640", "202503010000")

    copied = capture_fixture(out, fx)

    # The consumer's expected path must be exactly what capture produced.
    expected = fixture_grib_dir(fx, "forecaster-abcd/6640", "202503010000")
    assert expected in copied
    assert (expected / "202503010_0.grib").read_bytes() == b"GRIB-DATA"


def test_capture_overwrites_existing(tmp_path):
    out = tmp_path / "output"
    fx = tmp_path / "fixture"
    _fake_run(out, "forecaster-abcd/6640", "202503010000")
    capture_fixture(out, fx)
    capture_fixture(out, fx)  # second run must not raise
    assert fixture_grib_dir(fx, "forecaster-abcd/6640", "202503010000").exists()


def test_write_manifest(tmp_path):
    path = write_manifest(
        tmp_path,
        config_label="meteogram-test",
        checkpoints=["https://.../runs/b30a"],
        captured_at="2026-07-27T10:00:00",
        grib_dirs=[tmp_path / "data/runs/forecaster-abcd/6640/202503010000/grib"],
    )
    data = yaml.safe_load(path.read_text())
    assert data["config_label"] == "meteogram-test"
    assert data["checkpoints"] == ["https://.../runs/b30a"]
    assert data["captured_at"] == "2026-07-27T10:00:00"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_fixtures.py -v`
Expected: FAIL — `ImportError: cannot import name 'iter_grib_dirs'`.

- [ ] **Step 3: Implement the functions**

Append to `src/evalml/fixtures.py`:

```python
import shutil

import yaml


def iter_grib_dirs(output_root) -> list[Path]:
    """Every ``grib/`` directory under ``<output_root>/data/runs``."""
    runs = Path(output_root) / "data" / "runs"
    if not runs.is_dir():
        return []
    return sorted(p for p in runs.rglob("grib") if p.is_dir())


def capture_fixture(output_root, fixture_root) -> list[Path]:
    """Copy every inference GRIB dir under ``output_root`` into ``fixture_root``.

    Preserves the relative path so the result is readable via
    :func:`fixture_grib_dir`. Overwrites any existing destination. Returns the
    list of destination directories.
    """
    output_root = Path(output_root)
    fixture_root = Path(fixture_root)
    copied: list[Path] = []
    for grib in iter_grib_dirs(output_root):
        dest = fixture_root / grib.relative_to(output_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(grib, dest)
        copied.append(dest)
    return copied


def write_manifest(
    fixture_root, *, config_label, checkpoints, captured_at, grib_dirs
) -> Path:
    """Write MANIFEST.yaml recording what was frozen (provenance only)."""
    manifest = {
        "config_label": config_label,
        "checkpoints": list(checkpoints),
        "captured_at": captured_at,
        "grib_dirs": [str(p) for p in grib_dirs],
    }
    path = Path(fixture_root) / "MANIFEST.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_fixtures.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/evalml/fixtures.py tests/unit/test_fixtures.py
git commit -m "feat(fixtures): add capture_fixture, iter_grib_dirs, write_manifest"
```

---

### Task 4: `evalml capture-fixture` CLI command

**Files:**
- Modify: `src/evalml/cli.py` (add a command to the `cli` group, ~after the `showcase` command at line 223)
- Test: `tests/unit/test_cli_capture_fixture.py`

**Interfaces:**
- Consumes: `evalml.fixtures.capture_fixture`, `write_manifest`.
- Produces: CLI command `evalml capture-fixture <configfile> <fixture_root>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli_capture_fixture.py
from pathlib import Path
import yaml
from click.testing import CliRunner
from evalml.cli import cli


def _write_config(path: Path, output_root: Path):
    path.write_text(
        yaml.safe_dump(
            {
                "config_label": "meteogram-test",
                "locations": {"output_root": str(output_root)},
                "runs": [
                    {"forecaster": {"checkpoint": "https://x/runs/b30a"}},
                    {"baseline": {"label": "ICON", "root": "/store_new/x"}},
                ],
            }
        )
    )


def test_capture_fixture_command(tmp_path):
    output_root = tmp_path / "output"
    grib = output_root / "data/runs/forecaster-abcd/6640/202503010000/grib"
    grib.mkdir(parents=True)
    (grib / "f.grib").write_bytes(b"G")
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, output_root)
    fixture_root = tmp_path / "fixture"

    result = CliRunner().invoke(
        cli, ["capture-fixture", str(cfg), str(fixture_root)]
    )

    assert result.exit_code == 0, result.output
    assert (fixture_root / "data/runs/forecaster-abcd/6640/202503010000/grib/f.grib").exists()
    manifest = yaml.safe_load((fixture_root / "MANIFEST.yaml").read_text())
    assert manifest["config_label"] == "meteogram-test"
    assert manifest["checkpoints"] == ["https://x/runs/b30a"]


def test_capture_fixture_errors_when_no_grib(tmp_path):
    output_root = tmp_path / "output"
    (output_root / "data/runs").mkdir(parents=True)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, output_root)

    result = CliRunner().invoke(
        cli, ["capture-fixture", str(cfg), str(tmp_path / "fixture")]
    )
    assert result.exit_code != 0
    assert "No inference GRIB" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_cli_capture_fixture.py -v`
Expected: FAIL — click reports "No such command 'capture-fixture'".

- [ ] **Step 3: Implement the command**

In `src/evalml/cli.py`, add after the `showcase` command:

```python
@cli.command(
    "capture-fixture",
    help="Snapshot inference GRIB output into a test fixture directory for replay.",
)
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("fixture_root", type=click.Path(file_okay=False, path_type=Path))
def capture_fixture_cmd(configfile: Path, fixture_root: Path) -> None:
    import datetime

    import yaml

    from evalml.fixtures import capture_fixture, write_manifest

    cfg = yaml.safe_load(configfile.read_text())
    output_root = Path(cfg.get("locations", {}).get("output_root", "output"))

    copied = capture_fixture(output_root, fixture_root)
    if not copied:
        raise click.ClickException(
            f"No inference GRIB dirs found under {output_root}/data/runs. "
            "Run the pipeline once (with a GPU) before capturing."
        )

    checkpoints = [
        next(iter(entry.values())).get("checkpoint")
        for entry in cfg.get("runs", [])
        if "baseline" not in entry
    ]
    write_manifest(
        fixture_root,
        config_label=cfg.get("config_label"),
        checkpoints=checkpoints,
        captured_at=datetime.datetime.now().isoformat(timespec="seconds"),
        grib_dirs=copied,
    )
    click.echo(f"Captured {len(copied)} GRIB dir(s) into {fixture_root}")
```

`click` and `Path` are already imported at the top of `cli.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_cli_capture_fixture.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Confirm the command is registered**

Run: `uv run evalml capture-fixture --help`
Expected: usage text for `capture-fixture` with `CONFIGFILE` and `FIXTURE_ROOT` args.

- [ ] **Step 6: Commit**

```bash
git add src/evalml/cli.py tests/unit/test_cli_capture_fixture.py
git commit -m "feat(cli): add evalml capture-fixture command"
```

---

### Task 5: Replay in the workflow (`common.smk` + `inference.smk`)

**Files:**
- Modify: `workflow/rules/common.smk` (near the constants block, ~line 10)
- Modify: `workflow/rules/inference.smk` (rule `inference_execute`, ~line 261)

**Interfaces:**
- Consumes: `evalml.fixtures.fixture_grib_dir`; `config.get("fixture_root")`.
- Produces: `FIXTURE_ROOT` global in the snakemake namespace; a conditional `inference_execute` rule.

- [ ] **Step 1: Add `FIXTURE_ROOT` + import to `common.smk`**

After `OUT_ROOT = Path(config["locations"]["output_root"])` (line 10) add:

```python
from evalml.fixtures import fixture_grib_dir

# Opt-in inference replay: when set, inference_execute stages frozen GRIB from
# here instead of running anemoi-inference (see inference.smk).
FIXTURE_ROOT = config.get("fixture_root")
```

- [ ] **Step 2: Make `inference_execute` conditional in `inference.smk`**

Replace the existing single `rule inference_execute:` block (from `rule inference_execute:` through the closing `# fmt: on`) with:

```python
if FIXTURE_ROOT:

    from snakemake.exceptions import WorkflowError

    def _fixture_grib(wc):
        p = fixture_grib_dir(FIXTURE_ROOT, wc.run_id, wc.init_time)
        if not p.exists():
            raise WorkflowError(
                f"Fixture GRIB not found at {p}. Capture it once from a real run "
                f"with: evalml capture-fixture <config> {FIXTURE_ROOT}"
            )
        return str(p)

    rule inference_execute:
        input:
            grib=_fixture_grib,
        output:
            okfile=OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.ok",
        log:
            OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.log",
        localrule: True
        params:
            workdir=lambda wc: (
                OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}"
            ).resolve(),
            fixture_grib=_fixture_grib,
        shell:
            """
            (
                set -euo pipefail
                mkdir -p {params.workdir}
                ln -sfn {params.fixture_grib} {params.workdir}/grib
            ) >{log} 2>&1
            touch {output.okfile}
            """

else:

    rule inference_execute:
        input:
            okfile=_inference_routing_fn,
            image=lambda wc: OUT_ROOT
            / f"data/runs/{RUN_CONFIGS[wc.run_id]['env_id']}/venv.squashfs",
        output:
            okfile=OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.ok",
        log:
            OUT_ROOT / "logs/inference_execute/{run_id}-{init_time}.log",
        localrule: True
        resources:
            slurm_partition=lambda wc: get_resource(wc, "slurm_partition", "short-shared"),
            cpus_per_task=lambda wc: get_resource(wc, "cpus_per_task", 24),
            mem_mb_per_cpu=lambda wc: get_resource(wc, "mem_mb_per_cpu", 8000),
            runtime=lambda wc: get_resource(wc, "runtime", "40m"),
            gres=lambda wc: f"gpu:{get_resource(wc, 'gpu',1)}",
            ntasks=lambda wc: get_resource(wc, "tasks", 1),
            gpus=lambda wc: get_resource(wc, "gpu", 1),
        params:
            env_path=lambda wc, input: f"{Path(input.image).resolve()}",
            workdir=lambda wc: (
                OUT_ROOT / f"data/runs/{wc.run_id}/{wc.init_time}"
            ).resolve(),
            disable_local_definitions=lambda wc: RUN_CONFIGS[wc.run_id].get(
                "disable_local_eccodes_definitions", False
            ),
        # fmt: off
        shell:
            """
            (
                set -euo pipefail

                cd {params.workdir}

                _run_inference() {{
                    local VENV=$1
                    source "$VENV/bin/activate"

                    if [ "{params.disable_local_definitions}" = "False" ]; then
                        export ECCODES_DEFINITION_PATH="$VENV/share/eccodes-cosmo-resources/definitions"
                    fi

                    CMD_ARGS=()

                    if [ {resources.gpus} -gt 1 ]; then
                        CMD_ARGS+=(runner.parallel.cluster=slurm)
                    fi

                    srun \
                        --unbuffered \
                        --partition={resources.slurm_partition} \
                        --cpus-per-task={resources.cpus_per_task} \
                        --mem-per-cpu={resources.mem_mb_per_cpu} \
                        --time={resources.runtime} \
                        --gres={resources.gres} \
                        --ntasks={resources.ntasks} \
                        anemoi-inference run config.yaml "${{CMD_ARGS[@]}}"
                }}
                export -f _run_inference

                squashfs-mount {params.env_path}:/user-environment -- bash -c '_run_inference /user-environment'
            ) >{log} 2>&1
            touch {output.okfile}
            """
        # fmt: on
```

> Note: the `else` branch is the original rule verbatim — do not change its behaviour. Copy it from git if unsure: `git show HEAD:workflow/rules/inference.smk`.

- [ ] **Step 3: Verify the workflow parses in production mode (fixture unset)**

The default `meteogram_small.yaml` has no `fixture_root` yet, so this exercises the `else` branch and confirms nothing regressed structurally.

Run: `uv run snakemake -s workflow/Snakefile --configfile tests/integration/configs/meteogram_small.yaml -n showcase_all 2>&1 | tail -30`
Expected: snakemake builds the DAG and lists jobs **including** `inference_make_squashfs_image` / `inference_execute` (needs DWH creds for the truth prerequisite check; if creds are unavailable this step is done manually on the CI runner — note that in the commit message).

- [ ] **Step 4: Verify replay mode skips the sandbox chain**

Create a throwaway fixture with a dummy GRIB dir for one of the config's run_ids (get the real `run_id`/`init_time` from a prior `output/data/runs` tree or the dry-run paths), then dry-run with `fixture_root` set via `--config`:

```bash
RID="forecaster-<hash>/<runhash>"; INIT="<init_time>"
mkdir -p /tmp/fx/data/runs/$RID/$INIT/grib && touch /tmp/fx/data/runs/$RID/$INIT/grib/x.grib
uv run snakemake -s workflow/Snakefile \
  --configfile tests/integration/configs/meteogram_small.yaml \
  --config fixture_root=/tmp/fx -n showcase_all 2>&1 | tail -40
```
Expected: the job list contains `inference_execute` but **not** `inference_make_squashfs_image`, `inference_create_sandbox`, or `inference_prepare_env`. (This is the mechanism check; if DWH creds are unavailable locally, record that this was verified on the CI runner.)

- [ ] **Step 5: Commit**

```bash
git add workflow/rules/common.smk workflow/rules/inference.smk
git commit -m "feat(workflow): replay inference from fixture_root when set"
```

---

### Task 6: Wire the fixture into the longtest + document capture

**Files:**
- Modify: `tests/integration/configs/meteogram_small.yaml`
- Modify/Create: `tests/integration/README.md` (create if absent)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Point the longtest config at the fixture**

Add to `tests/integration/configs/meteogram_small.yaml` (top level, e.g. above `locations:`):

```yaml
# Replay frozen inference GRIB instead of running it (no GPU / MLflow / sandbox
# build). Populate once with:
#   evalml capture-fixture tests/integration/configs/meteogram_small.yaml \
#     /store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small
# after a real (GPU) showcase run. Truth still comes live from the DWH.
fixture_root: /store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small
```

- [ ] **Step 2: Document the one-time capture workflow**

Add a section to `tests/integration/README.md`:

```markdown
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

`MANIFEST.yaml` in the fixture records the checkpoint(s) and capture date.
```

- [ ] **Step 3: Sanity-check the config still validates**

Run: `uv run python -c "import yaml; from evalml.config import ConfigModel; ConfigModel(**yaml.safe_load(open('tests/integration/configs/meteogram_small.yaml')))" && echo OK`
Expected: `OK` (the `fixture_root` key is accepted; no `extra_forbidden` error).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/configs/meteogram_small.yaml tests/integration/README.md
git commit -m "test(integration): replay showcase longtest from inference fixture"
```

---

## Manual acceptance (run once, on a GPU node with DWH access)

Not automatable in CI without a GPU, so verify by hand after Task 6:

1. With `fixture_root` commented out, run `evalml showcase tests/integration/configs/meteogram_small.yaml` to produce a real `output/`.
2. `evalml capture-fixture tests/integration/configs/meteogram_small.yaml /store_new/mch/msopr/cmerker/evalml_test_fixtures/meteogram-small`.
3. Restore `fixture_root`, delete `output/`, and run `pytest tests/integration -m longtest`.
4. Confirm: no `mksquashfs`/GPU SLURM jobs appear, the meteogram PNGs are produced, and the test passes — in a few minutes rather than ~50.

## Self-Review

- **Spec coverage:** fixture layout (Task 3 round-trip + `fixture_grib_dir`), opt-in `fixture_root` config (Task 1), staging rule + sandbox chain dropping out (Task 5), `evalml capture-fixture` (Task 4), test wiring + `returncode==0` unchanged (Task 6), mechanism verification (Task 5 Step 4 dry-run + manual acceptance). Truth/baseline left live — no task touches them. ✓
- **Placeholder scan:** the only literal placeholders are the real run_id/init_time hashes in Task 5 Step 4 (unknowable until a run exists) and `<fixture_root>` in docs — both are runtime values, not plan gaps. ✓
- **Type consistency:** `fixture_grib_dir(fixture_root, run_id, init_time)` signature identical in Tasks 2, 3, 5; `capture_fixture(output_root, fixture_root)` identical in Tasks 3, 4; `write_manifest(...)` keyword args identical in Tasks 3, 4. ✓
