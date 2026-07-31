from pathlib import Path

import yaml
from click.testing import CliRunner

from evalml.cli import cli
from evalml.fixtures import expected_run_prefixes

# Capture now validates the config through ConfigModel (so the CLI and workflow
# agree), so the test config must be a real, complete config. Derive it from the
# committed showcase config and only redirect output_root into tmp. Its single
# reference date is 2024-08-01T00:00 -> init time 202408010000.
BASE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "integration"
    / "configs"
    / "meteogram_small.yaml"
)
INIT_TIME = "202408010000"


def _write_config(path: Path, output_root: Path) -> str:
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["locations"]["output_root"] = str(output_root)
    path.write_text(yaml.safe_dump(cfg))
    return cfg["runs"][0]["forecaster"]["checkpoint"]


def _run_prefix(cfg_path: Path) -> str:
    """The one env_id prefix `expected_run_prefixes` derives for BASE_CONFIG's
    single forecaster run, so tests don't hardcode a checkpoint-derived id."""
    cfg = yaml.safe_load(cfg_path.read_text())
    (prefix,) = expected_run_prefixes(cfg["runs"])
    return prefix


def test_capture_fixture_command(tmp_path):
    output_root = tmp_path / "output"
    cfg = tmp_path / "cfg.yaml"
    checkpoint = _write_config(cfg, output_root)
    env_id = _run_prefix(cfg) + "0000"
    grib = output_root / f"data/runs/{env_id}/6640/{INIT_TIME}/grib"
    grib.mkdir(parents=True)
    (grib / "f.grib").write_bytes(b"G")
    fixture_root = tmp_path / "fixture"

    result = CliRunner().invoke(cli, ["capture-fixture", str(cfg), str(fixture_root)])

    assert result.exit_code == 0, result.output
    assert (
        fixture_root / f"data/runs/{env_id}/6640/{INIT_TIME}/grib/f.grib"
    ).exists()
    manifest = yaml.safe_load((fixture_root / "MANIFEST.yaml").read_text())
    assert manifest["config_label"] == "meteogram-test"
    assert checkpoint in manifest["checkpoints"]
    assert manifest["dates"] == [INIT_TIME]
    # A checksum is recorded for the captured GRIB dir (validated at replay).
    rel = f"data/runs/{env_id}/6640/{INIT_TIME}/grib"
    assert rel in manifest["grib_checksums"]
    assert "evalml_commit" in manifest


def test_capture_fixture_excludes_other_run_sharing_same_date(tmp_path):
    """Regression: a leftover run dir from an unrelated config sharing the same
    output/ tree and init_time must not be swept into this config's fixture."""
    output_root = tmp_path / "output"
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, output_root)
    env_id = _run_prefix(cfg) + "0000"
    grib = output_root / f"data/runs/{env_id}/6640/{INIT_TIME}/grib"
    grib.mkdir(parents=True)
    (grib / "f.grib").write_bytes(b"G")
    # Leftover output from an unrelated config's forecaster, same output_root/date.
    other = output_root / f"data/runs/forecaster-unrelated-1234/1a2b/{INIT_TIME}/grib"
    other.mkdir(parents=True)
    (other / "g.grib").write_bytes(b"G")
    fixture_root = tmp_path / "fixture"

    result = CliRunner().invoke(cli, ["capture-fixture", str(cfg), str(fixture_root)])

    assert result.exit_code == 0, result.output
    assert (fixture_root / f"data/runs/{env_id}/6640/{INIT_TIME}/grib/f.grib").exists()
    assert not (
        fixture_root / f"data/runs/forecaster-unrelated-1234/1a2b/{INIT_TIME}/grib"
    ).exists()


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
