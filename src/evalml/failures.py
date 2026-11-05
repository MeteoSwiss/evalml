"""Summarise failed jobs of a workflow run from the logs it left behind.

evalml runs Snakemake with ``--quiet rules``, which also removes the
"Error in rule ..." blocks from Snakemake's own log file, so a failed run does
not name the failing job. This module recovers that information from:

- SLURM job logs written during the run. The slurm executor deletes the logs of
  successful jobs by default, so the ones left are failed, cancelled or timed-out
  jobs. They live under ``rule_<rule>/<wildcards>/<jobid>.log``.
- Rule logs written during the run that contain an error, with their last
  error line.
"""

import re
from dataclasses import dataclass
from pathlib import Path

ERROR_PATTERN = re.compile(r"Traceback|Error|Exception|DUE TO TIME LIMIT|CANCELLED")


@dataclass(frozen=True)
class Failure:
    kind: str  # "slurm" or "rule"
    what: str  # "<rule> <wildcards> (job <id>)" for SLURM logs, log path for rule logs
    log: Path
    message: str  # last error line found in the log, if any


def _last_error_line(path: Path, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if ERROR_PATTERN.search(line):
            return line.strip()
    return ""


def _recent(paths, since: float):
    for path in paths:
        try:
            if path.is_file() and path.stat().st_mtime >= since:
                yield path
        except OSError:
            continue


def summarize_failures(
    since: float, slurm_logs: Path, rule_logs: Path
) -> list[Failure]:
    """Failures of the run that started at ``since`` (UNIX time), newest first."""
    failures = []
    slurm_logs, rule_logs = Path(slurm_logs), Path(rule_logs)
    if slurm_logs.is_dir():
        for path in _recent(slurm_logs.glob("rule_*/**/*.log"), since):
            parts = path.relative_to(slurm_logs).parts
            rule = parts[0].removeprefix("rule_")
            wildcards = "/".join(parts[1:-1])
            failures.append(
                Failure(
                    kind="slurm",
                    what=f"{rule} {wildcards} (job {path.stem})".replace("  ", " "),
                    log=path,
                    message=_last_error_line(path),
                )
            )
    if rule_logs.is_dir():
        for path in _recent(rule_logs.rglob("*.log"), since):
            message = _last_error_line(path)
            if not message:
                continue
            failures.append(
                Failure(
                    kind="rule",
                    what=str(path.relative_to(rule_logs)),
                    log=path,
                    message=message,
                )
            )
    failures.sort(key=lambda f: f.log.stat().st_mtime, reverse=True)
    return failures
