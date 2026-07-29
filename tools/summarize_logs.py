#!/usr/bin/env python3
"""
Collect CI log files and generate log/artifacts_summary.html.

Sources:
  .snakemake/log/         - Snakemake orchestration log (most recent)
  output/logs/            - Per-rule logs
  .snakemake/slurm_logs/  - SLURM job logs
"""

import shutil
from html import escape
from pathlib import Path

ROOT = Path.cwd()
DEST = ROOT / "tests" / "log"
DEST.mkdir(exist_ok=True)

ERROR_KEYWORDS = {"error", "exception", "traceback", "failed", "oom", "killed"}

# Lines that contain an ERROR_KEYWORDS substring but do not indicate a real
# failure. Each entry is a tuple of substrings that must ALL appear (lowercased)
# in a line for it to be skipped by has_error().
ERROR_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    # After downloading a grid to its cache, eckit removes the lockfile and logs
    # "Unlink failed <path>.ek.lock (No such file or directory)" when another
    # worker already removed it — a harmless cleanup race in eckit's own cache.
    ("unlink failed", "no such file or directory"),
)


def _is_benign(line: str) -> bool:
    """True if a log line matches a known-benign ERROR_ALLOWLIST pattern."""
    return any(all(sub in line for sub in entry) for entry in ERROR_ALLOWLIST)


def copy(src: Path, dest_root: Path) -> Path:
    rel = src.relative_to(ROOT)
    dst = dest_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def read(path: Path, max_lines: int = 300) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        if len(lines) > max_lines:
            lines = [f"... ({len(lines) - max_lines} lines omitted) ...", ""] + lines[
                -max_lines:
            ]
        return "\n".join(lines)
    except Exception as e:
        return f"(could not read: {e})"


def has_error(path: Path) -> bool:
    try:
        lines = path.read_text(errors="replace").lower().splitlines()
    except Exception:
        return False
    return any(
        kw in line
        for line in lines
        if not _is_benign(line)
        for kw in ERROR_KEYWORDS
    )


# --- collect ---

snakemake_logs = sorted(ROOT.glob(".snakemake/log/*.snakemake.log"))
latest_snakemake = [snakemake_logs[-1]] if snakemake_logs else []

rule_logs = [
    p
    for p in sorted(ROOT.glob("output/logs/**/*.log"), key=lambda p: p.stat().st_mtime)
    if p.suffix == ".log"
]

slurm_logs = sorted(ROOT.glob(".snakemake/slurm_logs/**/*.log"))

all_sources = latest_snakemake + rule_logs + slurm_logs
copied = [copy(src, DEST) for src in all_sources]
print(f"Collected {len(copied)} log files into {DEST}/")

# --- HTML ---

CSS = """
body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; margin: 1em 2em; }
h1 { color: #9cdcfe; }
h2 { color: #9cdcfe; border-bottom: 1px solid #444; padding-bottom: 0.2em; margin-top: 1.5em; }
details { margin: 0.4em 0; }
summary { cursor: pointer; padding: 0.3em 0.5em; border-radius: 3px; }
summary:hover { background: #2d2d2d; }
.err > summary { border-left: 4px solid #f44747; color: #f44747; }
.ok  > summary { border-left: 4px solid #4ec9b0; color: #4ec9b0; }
pre { background: #252526; padding: 1em; overflow: auto; white-space: pre-wrap;
      font-size: 0.85em; border: 1px solid #333; border-radius: 3px; }
"""


def details(label: str, path: Path, *, open_by_default: bool = False) -> str:
    err = has_error(path)
    css = "err" if err else "ok"
    icon = "❌" if err else "✓"
    tag = "<details open>" if open_by_default else "<details>"
    return (
        f'<div class="{css}">'
        f"{tag}<summary>{icon} {escape(label)}</summary>"
        f"<pre>{escape(read(path))}</pre>"
        f"</details></div>"
    )


parts = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>CI Log Summary</title>",
    f"<style>{CSS}</style>",
    "</head><body>",
    "<h1>CI Log Summary</h1>",
]

if latest_snakemake:
    parts.append("<h2>Snakemake Log</h2>")
    parts.append(
        details(
            ".snakemake/log/" + latest_snakemake[0].name,
            DEST / latest_snakemake[0].relative_to(ROOT),
            open_by_default=True,
        )
    )

if rule_logs:
    parts.append("<h2>Rule Logs</h2>")
    for src in rule_logs:
        dst = DEST / src.relative_to(ROOT)
        label = str(src.relative_to(ROOT / "output/logs"))
        parts.append(details(label, dst))

if slurm_logs:
    parts.append("<h2>SLURM Logs</h2>")
    for src in slurm_logs:
        dst = DEST / src.relative_to(ROOT)
        label = str(src.relative_to(ROOT / ".snakemake/slurm_logs"))
        parts.append(details(label, dst))

parts.append("</body></html>")

summary_path = DEST / "artifacts_summary.html"
summary_path.write_text("\n".join(parts))
print(f"Summary written to {summary_path}")
