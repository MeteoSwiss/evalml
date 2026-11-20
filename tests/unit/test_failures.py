import os
import time

from evalml.failures import summarize_failures


def _write(path, text, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def test_summarize_failures(tmp_path):
    start = time.time() - 100
    slurm = tmp_path / "slurm_logs"
    logs = tmp_path / "logs"
    # failed SLURM job of this run
    _write(
        slurm / "rule_verification_scoremaps" / "run-a_TOT_PREC6_72" / "123.log",
        "running...\nslurmstepd: error: *** JOB 123 CANCELLED DUE TO TIME LIMIT ***\n",
        start + 50,
    )
    # SLURM log from an earlier run: ignored
    _write(slurm / "rule_plot_meteogram" / "x" / "99.log", "Error: old\n", start - 500)
    # rule log with a traceback from this run
    _write(
        logs / "showcase" / "run-a" / "plot_meteogram_TOT_PREC.log",
        "INFO ok\nTraceback (most recent call last):\n"
        "ValueError: No DWH parameter mapping for requested params: ['TOT_PREC']\n",
        start + 60,
    )
    # successful rule log from this run: ignored
    _write(logs / "verification_metrics" / "ok.log", "INFO all good\n", start + 10)

    failures = summarize_failures(start, slurm, logs)

    assert [f.kind for f in failures] == ["rule", "slurm"]  # newest first
    rule_failure, slurm_failure = failures
    assert rule_failure.what == "showcase/run-a/plot_meteogram_TOT_PREC.log"
    assert rule_failure.message.startswith("ValueError: No DWH parameter mapping")
    assert slurm_failure.what == "verification_scoremaps run-a_TOT_PREC6_72 (job 123)"
    assert "TIME LIMIT" in slurm_failure.message


def test_summarize_failures_missing_dirs(tmp_path):
    assert summarize_failures(0, tmp_path / "nope", tmp_path / "nada") == []
