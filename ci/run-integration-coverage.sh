#!/usr/bin/env bash
# Tier-2 coverage: run an integration suite and measure coverage across the full
# snakemake subprocess chain
#   pytest -> `evalml` -> snakemake -> workflow/scripts/*.py (incl. SLURM jobs).
#
# Usage: run-integration-coverage.sh <marker> [target...]
#   <marker>   pytest marker selecting the suite, e.g. longtest or heavytest.
#   [target]   test path(s) to run; defaults to the whole suite. The CSCS jobs
#              pass `tests/integration`.
#
# Requires whatever the tests themselves need (GPU, MLflow/DWH credentials,
# access to /store_new). Run from anywhere; it cd's to the repo root.
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: $(basename "$0") <marker> [target...]" >&2
    exit 2
fi
MARKER="$1"
shift

cd "$(dirname "$0")/.."
CONFIG="ci/coverage-integration.cfg"

# Point every child process (spawned by evalml/snakemake, including SLURM jobs,
# which inherit the environment via sbatch --export=ALL) at the parallel-mode
# config so the a1_coverage.pth startup hook activates coverage there too.
export COVERAGE_PROCESS_START="$PWD/$CONFIG"

# Remaining args are the test path(s); default to the whole suite.
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(tests/)

# `-o addopts=` clears the project default addopts ("-m 'not longtest'" plus the
# Tier-1 --cov flags) so this run uses ONLY the subprocess-aware Tier-2 config.
uv run pytest "${TARGETS[@]}" \
    -m "$MARKER" \
    -o addopts= \
    --cov \
    --cov-config="$CONFIG" \
    --cov-report=term-missing
