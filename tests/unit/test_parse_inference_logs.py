import json

import pytest

from diagnostics import melt_for_dashboard, parse_logs
from diagnostics import parse_gpu_metrics_log, parse_sacct_log

SACCT_CONTENT = """\
JobID|JobName|Elapsed|CPUTime|MaxRSS|MaxVMSize|AveRSS|MaxDiskRead|MaxDiskWrite
12345678|run_with_metrics.sh|00:05:23|02:08:48|0|0|0|4320K|2100K
12345678.0|run_with_metrics.sh|00:05:23|02:08:48|2048K|5120K|1024K|4320K|2100K
12345678.batch|batch|00:00:01|00:00:24|512K|1024K|256K|0|0
"""

DMON_CONTENT = """\
# gpu   Date        Time        Idx   sm   mem   enc   dec   pwr  mclk  pclk pviol tviol    fb  bar1 sbecc dbecc   pci gtemp mtemp
# Devicems                           %     %     %     %    W   MHz   MHz      %     % MiB   MiB                       C     C
20250602 12:34:56    0   85    70     0     0   250  1215  1530     0     0  8192 10240     0     0     0    65    50
20250602 12:35:01    0   90    75     0     0   260  1215  1530     0     0  8300 10240     0     0     0    66    51
"""


def test_parse_sacct_wall_time(tmp_path):
    log_file = tmp_path / "slurm_metrics.log"
    log_file.write_text(SACCT_CONTENT)

    result = parse_sacct_log(str(log_file))

    assert result["wall_time_s"] == pytest.approx(5 * 60 + 23)


def test_parse_sacct_max_rss(tmp_path):
    log_file = tmp_path / "slurm_metrics.log"
    log_file.write_text(SACCT_CONTENT)

    result = parse_sacct_log(str(log_file))

    # 2048K = 2.0 MB; 512K = 0.5 MB; parent row is 0 → max is 2.0
    assert result["max_rss_mb"] == pytest.approx(2.0, rel=1e-3)


def test_parse_sacct_missing_file(tmp_path):
    result = parse_sacct_log(str(tmp_path / "does_not_exist.log"))
    assert result == {}


def test_parse_sacct_empty_file(tmp_path):
    f = tmp_path / "slurm_metrics.log"
    f.write_text("")
    assert parse_sacct_log(str(f)) == {}


def test_parse_gpu_metrics_utilisation(tmp_path):
    log_file = tmp_path / "gpu_metrics.log"
    log_file.write_text(DMON_CONTENT)

    result = parse_gpu_metrics_log(str(log_file))

    assert result["gpu_util_mean"] == pytest.approx((85 + 90) / 2, rel=1e-3)
    assert result["gpu_util_max"] == pytest.approx(90.0)


def test_parse_gpu_metrics_memory(tmp_path):
    log_file = tmp_path / "gpu_metrics.log"
    log_file.write_text(DMON_CONTENT)

    result = parse_gpu_metrics_log(str(log_file))

    assert result["gpu_mem_used_mean"] == pytest.approx((8192 + 8300) / 2, rel=1e-3)
    assert result["gpu_mem_used_max"] == pytest.approx(8300.0)


def test_parse_gpu_metrics_power(tmp_path):
    log_file = tmp_path / "gpu_metrics.log"
    log_file.write_text(DMON_CONTENT)

    result = parse_gpu_metrics_log(str(log_file))

    assert result["gpu_power_mean"] == pytest.approx((250 + 260) / 2, rel=1e-3)


def test_parse_gpu_metrics_missing_file(tmp_path):
    assert parse_gpu_metrics_log(str(tmp_path / "does_not_exist.log")) == {}


def test_parse_logs_reads_slurm_and_gpu_files(tmp_path):
    workdir = tmp_path / "data" / "runs" / "forecaster-abc" / "202503010000"
    workdir.mkdir(parents=True)
    (workdir / "slurm_metrics.log").write_text(SACCT_CONTENT)
    (workdir / "gpu_metrics.log").write_text(DMON_CONTENT)
    (workdir / "slurm_job_id").write_text("12345678\n")

    records = parse_logs(
        run_info=[
            {
                "workdir": str(workdir),
                "run_id": "forecaster-abc",
                "init_time": "202503010000",
            }
        ],
        label_map={"forecaster-abc": "My Model"},
        gpu_map={"forecaster-abc": 2},
    )

    assert len(records) == 1
    r = records[0]
    assert r["source"] == "My Model"
    assert r["run_id"] == "forecaster-abc"
    assert r["model_type"] == "forecaster"
    assert r["init_time"] == "2025-03-01T00:00:00"
    assert r["n_gpu"] == 2
    assert r["job_id"] == "12345678"
    assert r["wall_time_s"] == pytest.approx(323.0)
    assert r["gpu_hours"] == pytest.approx(323.0 / 3600 * 2, rel=1e-3)
    assert "gpu_util_mean" in r
    assert "max_rss_mb" in r


def test_parse_logs_missing_workdir_is_skipped(tmp_path):
    records = parse_logs(
        run_info=[
            {
                "workdir": str(tmp_path / "does_not_exist"),
                "run_id": "forecaster-abc",
                "init_time": "202503010000",
            }
        ],
        label_map={},
        gpu_map={},
    )
    assert records == []


def test_parse_logs_no_metrics_files_is_skipped(tmp_path):
    workdir = tmp_path / "empty_run"
    workdir.mkdir()
    records = parse_logs(
        run_info=[
            {"workdir": str(workdir), "run_id": "x", "init_time": "202503010000"}
        ],
        label_map={},
        gpu_map={},
    )
    assert records == []


def test_parse_logs_model_type_from_run_id_prefix(tmp_path):
    for prefix, expected_type in [
        ("forecaster-c304", "forecaster"),
        ("interpolator-tmp-d5aa", "interpolator"),
    ]:
        workdir = tmp_path / prefix / "202503010000"
        workdir.mkdir(parents=True)
        (workdir / "slurm_metrics.log").write_text(SACCT_CONTENT)

        records = parse_logs(
            run_info=[
                {"workdir": str(workdir), "run_id": prefix, "init_time": "202503010000"}
            ],
            label_map={},
            gpu_map={},
        )
        assert records[0]["model_type"] == expected_type


def test_parse_logs_fallback_label_is_run_id(tmp_path):
    workdir = tmp_path / "env-abc" / "202503020000"
    workdir.mkdir(parents=True)
    (workdir / "slurm_metrics.log").write_text(SACCT_CONTENT)

    records = parse_logs(
        run_info=[
            {"workdir": str(workdir), "run_id": "env-abc", "init_time": "202503020000"}
        ],
        label_map={},
        gpu_map={},
    )
    assert records[0]["source"] == "env-abc"


def test_melt_for_dashboard_exposes_model_type_and_distribution_metrics():
    records = [
        {
            "source": "ModelA",
            "model_type": "forecaster",
            "init_time": "2025-03-01T00:00:00",
            "n_gpu": 1,
            "job_id": "111",
            "wall_time_s": 60.0,
            "gpu_hours": 1 / 60,
        },
        {
            "source": "ModelA",
            "model_type": "interpolator",
            "init_time": "2025-03-02T00:00:00",
            "n_gpu": 1,
            "job_id": "222",
            "wall_time_s": 120.0,
            "gpu_hours": 2 / 60,
        },
    ]
    data_json, sources, model_types = melt_for_dashboard(records)
    rows = json.loads(data_json)

    assert sources == ["ModelA"]
    assert model_types == ["forecaster", "interpolator"]
    # only wall_time_s and gpu_hours are present in the test records
    metrics_present = {r["metric"] for r in rows}
    assert metrics_present == {"Wall Time (min)", "GPU Hours"}
    # model_type must be present in every row
    assert all("model_type" in r for r in rows)
