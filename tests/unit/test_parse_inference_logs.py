import pytest

from diagnostics import parse_logs, parse_single_log

LOG_CONTENT = """\
srun: job 4242140 queued and waiting for resources
srun: job 4242140 has been allocated resources
2026-05-01 11:10:19 INFO Loading multi-dataset metadata
2026-05-01 11:10:52 INFO Checkpoint size: 1.4 GiB
2026-05-01 11:10:52 INFO Lead time: 5 days, 0:00:00 Forecasting 3 steps through 3 autoregressive steps
2026-05-01 11:11:07 INFO Forecast. Model call 1: horizon 6:00:00, freq. 6:00:00 (2025-03-01 06:00:00): 6 seconds.
2026-05-01 11:11:12 INFO Forecast. Model call 2: horizon 12:00:00, freq. 6:00:00 (2025-03-01 12:00:00): 2 seconds.
2026-05-01 11:11:17 INFO Forecast. Model call 3: horizon 18:00:00, freq. 6:00:00 (2025-03-01 18:00:00): 2 seconds.
2026-05-01 11:11:20 INFO Done.
"""


def test_parse_single_log(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text(LOG_CONTENT)

    result = parse_single_log(str(log_file))

    assert result["job_id"] == "4242140"
    assert result["checkpoint_size_gib"] == 1.4
    assert result["n_steps"] == 3
    assert result["max_step_time_s"] == 6
    assert result["mean_step_time_s"] == pytest.approx(round((6 + 2 + 2) / 3, 2))
    # wall time: 11:11:20 - 11:10:19 = 61 seconds
    assert result["wall_time_s"] == pytest.approx(61.0)


def test_parse_logs_extracts_run_id_and_init_time(tmp_path):
    log_dir = tmp_path / "inference_execute" / "forecaster-c304-1e7e"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "253b-202503010000.log"
    log_file.write_text(LOG_CONTENT)

    label_map = {"forecaster-c304-1e7e/253b": "My Model"}
    gpu_map = {"forecaster-c304-1e7e/253b": 2}

    records = parse_logs(
        log_files=[str(log_file)],
        label_map=label_map,
        gpu_map=gpu_map,
        log_dir=str(tmp_path / "inference_execute"),
    )

    assert len(records) == 1
    r = records[0]
    assert r["source"] == "My Model"
    assert r["run_id"] == "forecaster-c304-1e7e/253b"
    assert r["init_time"] == "2025-03-01T00:00:00"
    assert r["n_gpu"] == 2
    assert r["gpu_hours"] == pytest.approx(61.0 / 3600 * 2, rel=1e-3)


def test_parse_logs_missing_file_is_skipped(tmp_path):
    records = parse_logs(
        log_files=[str(tmp_path / "does_not_exist.log")],
        label_map={},
        gpu_map={},
        log_dir=str(tmp_path),
    )
    assert records == []


def test_parse_logs_fallback_label_is_run_id(tmp_path):
    log_dir = tmp_path / "inference_execute" / "env-abc"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "1234-202503020000.log"
    log_file.write_text(LOG_CONTENT)

    records = parse_logs(
        log_files=[str(log_file)],
        label_map={},  # no label provided
        gpu_map={},
        log_dir=str(tmp_path / "inference_execute"),
    )

    assert records[0]["source"] == "env-abc/1234"
