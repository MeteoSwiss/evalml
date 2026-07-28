from pathlib import Path
import yaml
from click.testing import CliRunner
from evalml.cli import cli


def _write_config(path: Path, output_root: Path):
    path.write_text(
        yaml.safe_dump(
            {
                "config_label": "meteogram-test",
                "dates": ["2025-03-01T00:00"],
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

    result = CliRunner().invoke(cli, ["capture-fixture", str(cfg), str(fixture_root)])

    assert result.exit_code == 0, result.output
    assert (
        fixture_root / "data/runs/forecaster-abcd/6640/202503010000/grib/f.grib"
    ).exists()
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
