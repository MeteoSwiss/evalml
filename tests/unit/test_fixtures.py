from pathlib import Path

import yaml

from evalml.fixtures import (
    capture_fixture,
    config_init_times,
    fixture_grib_dir,
    iter_grib_dirs,
    write_manifest,
)


def test_fixture_grib_dir_mirrors_output_layout():
    got = fixture_grib_dir("/fx", "forecaster-abcd/6640", "202503010000")
    assert got == Path("/fx/data/runs/forecaster-abcd/6640/202503010000/grib")


def test_fixture_grib_dir_accepts_path_and_int_init():
    got = fixture_grib_dir(
        Path("/fx"), "temporal-x-on-forecaster-abcd/1a2b", 202503010000
    )
    assert got == Path(
        "/fx/data/runs/temporal-x-on-forecaster-abcd/1a2b/202503010000/grib"
    )


def test_fixture_grib_dir_is_absolute_for_relative_root():
    # A relative fixture_root must still yield an absolute path, else the replay
    # rule's `ln -sfn` (run from a deep workdir) creates a dangling symlink.
    got = fixture_grib_dir("relfx", "forecaster-abcd/6640", "202503010000")
    assert got.is_absolute()
    assert str(got).endswith("relfx/data/runs/forecaster-abcd/6640/202503010000/grib")


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


def test_config_init_times_explicit_list():
    assert config_init_times(["2024-08-01T00:00", "2024-08-02T12:00"]) == {
        "202408010000",
        "202408021200",
    }


def test_config_init_times_range_with_blacklist():
    got = config_init_times(
        {
            "start": "2024-01-01T00:00",
            "end": "2024-01-03T00:00",
            "frequency": "1d",
            "blacklist": ["2024-01-02T00:00"],
        }
    )
    assert got == {"202401010000", "202401030000"}


def test_capture_scoped_by_init_times_excludes_other_dates(tmp_path):
    """Regression: capture must not sweep in an unrelated experiment that
    shares output/ but uses different dates."""
    out = tmp_path / "output"
    fx = tmp_path / "fixture"
    _fake_run(out, "forecaster-b30a/6640", "202408010000")  # the config's run/date
    _fake_run(out, "forecaster-sruc/94fd", "202503010000")  # unrelated experiment

    copied = capture_fixture(out, fx, init_times={"202408010000"})

    assert copied == [fixture_grib_dir(fx, "forecaster-b30a/6640", "202408010000")]
    assert not fixture_grib_dir(fx, "forecaster-sruc/94fd", "202503010000").exists()


def _replay_symlink(output_root: Path, run_id: str, init_time: str, target: Path):
    """Mimic replay: workdir/grib is a symlink into the fixture, not a real dir."""
    workdir = output_root / "data" / "runs" / run_id / init_time
    workdir.mkdir(parents=True)
    (workdir / "grib").symlink_to(target, target_is_directory=True)


def test_iter_grib_dirs_skips_replay_symlinks(tmp_path):
    out = tmp_path / "output"
    fx = tmp_path / "fixture"
    _fake_run(fx, "forecaster-abcd/6640", "202503010000")  # a fixture grib dir
    fixture_grib = fixture_grib_dir(fx, "forecaster-abcd/6640", "202503010000")
    _replay_symlink(out, "forecaster-abcd/6640", "202503010000", fixture_grib)
    # The only "grib" under output is a symlink into the fixture -> skipped.
    assert iter_grib_dirs(out) == []


def test_capture_after_replay_does_not_delete_fixture(tmp_path):
    """Regression: capturing while a replay symlink is in place must NOT
    resolve back onto and destroy the fixture it points at."""
    out = tmp_path / "output"
    fx = tmp_path / "fixture"
    # 1) a real captured fixture already exists
    _fake_run(fx, "forecaster-abcd/6640", "202503010000")
    fixture_grib = fixture_grib_dir(fx, "forecaster-abcd/6640", "202503010000")
    # 2) output/ holds only the replay symlink pointing into that fixture
    _replay_symlink(out, "forecaster-abcd/6640", "202503010000", fixture_grib)

    copied = capture_fixture(out, fx)  # must not raise, must not delete fixture

    assert copied == []  # nothing real to capture
    assert (fixture_grib / "202503010_0.grib").read_bytes() == b"GRIB-DATA"


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
