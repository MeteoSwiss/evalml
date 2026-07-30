"""Hermetic `snakemake -n` coverage for inference-fixture replay.

Unlike the unit round-trip test (which checks ``capture_fixture`` against
``fixture_grib_dir`` over a layout it builds itself), these tests drive the
*real* workflow: they assert that replay is wired into the DAG the way the
rules actually construct paths (``inference.smk`` / ``verification.smk`` /
``plot.smk``), and that the missing-fixture guard fires.

Hermetic and GPU/DWH-free:
* truth is pointed at a local path so ``common.smk`` skips the jretrieve
  prerequisite check;
* the mlflow checkpoint URL is only *parsed* (no network) to derive the run id
  during DAG building;
* ``snakemake -n`` is invoked directly with the default (local) executor, so no
  SLURM plugin is needed and nothing is executed.

Marked ``longtest`` so it runs on PRs (ci/longtest.yml); it needs no GPU/DWH.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    PROJECT_ROOT / "tests" / "integration" / "configs" / "meteogram_small.yaml"
)

_ENV_BUILD_RULES = (
    "inference_make_squashfs_image",
    "inference_create_sandbox",
    "inference_prepare_env",
)


def _hermetic_config(tmp_path: Path) -> Path:
    """meteogram_small config with a local truth + tmp output, written to tmp."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    truth = tmp_path / "truth.zarr"
    truth.mkdir()
    cfg["truth"] = {"label": "local", "root": str(truth)}
    cfg["locations"]["output_root"] = str(tmp_path / "output")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _dry_run(config: Path, fixture_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "-s",
            "workflow/Snakefile",
            "--configfile",
            str(config),
            "--config",
            f"fixture_root={fixture_root}",
            "-n",
            "showcase_all",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.longtest
def test_replay_missing_fixture_errors(tmp_path):
    """An empty fixture makes DAG building fail with the guard's message."""
    result = _dry_run(_hermetic_config(tmp_path), tmp_path / "empty")
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-2000:]
    assert "Fixture GRIB not found" in combined, combined[-2000:]


@pytest.mark.longtest
def test_replay_uses_fixture_and_skips_env_build(tmp_path):
    """With the fixture present, replay is selected and the sandbox-build chain
    drops out of the DAG. The expected GRIB path is taken from the workflow's own
    error, so this pins fixture_grib_dir to the rules' real run_id/init layout."""
    config = _hermetic_config(tmp_path)
    fixture_root = tmp_path / "fixture"

    # First pass reveals exactly where the workflow looks for the fixture GRIB.
    probe = _dry_run(config, fixture_root)
    match = re.search(
        r"Fixture GRIB not found at (\S+/grib)", probe.stdout + probe.stderr
    )
    assert match, (probe.stdout + probe.stderr)[-2000:]

    grib = Path(match.group(1))
    grib.mkdir(parents=True)
    (grib / "dummy.grib").touch()

    result = _dry_run(config, fixture_root)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[-2000:]
    assert "inference_execute" in combined, combined[-2000:]
    for rule in _ENV_BUILD_RULES:
        assert rule not in combined, (
            f"{rule} must not run in replay mode\n{combined[-2000:]}"
        )


@pytest.mark.longtest
def test_replay_rejects_checksum_mismatch(tmp_path):
    """A manifest recording a checksum that no longer matches the fixture GRIB
    fails DAG building — proving verify_fixture fires through _fixture_grib."""
    config = _hermetic_config(tmp_path)
    fixture_root = tmp_path / "fixture"

    probe = _dry_run(config, fixture_root)
    match = re.search(
        r"Fixture GRIB not found at (\S+/grib)", probe.stdout + probe.stderr
    )
    assert match, (probe.stdout + probe.stderr)[-2000:]

    grib = Path(match.group(1))
    grib.mkdir(parents=True)
    (grib / "dummy.grib").touch()
    rel = grib.resolve().relative_to(fixture_root.resolve()).as_posix()
    (fixture_root / "MANIFEST.yaml").write_text(
        yaml.safe_dump({"grib_checksums": {rel: "deadbeef-not-the-real-checksum"}})
    )

    result = _dry_run(config, fixture_root)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-2000:]
    assert "does not match its recorded checksum" in combined, combined[-2000:]
