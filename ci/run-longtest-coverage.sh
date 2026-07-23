#!/usr/bin/env bash
# Tier-2 coverage: run the longtest (integration) suite and measure coverage
# across the full snakemake subprocess chain
#   pytest -> `evalml` -> snakemake -> workflow/scripts/*.py (incl. SLURM jobs).
#
# Requires whatever the longtests themselves need (GPU, MLflow/DWH credentials,
# access to /store_new). Run from anywhere; it cd's to the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="ci/coverage-longtest.cfg"

# Point every child process (spawned by evalml/snakemake, including SLURM jobs,
# which inherit the environment via sbatch --export=ALL) at the parallel-mode
# config so the a1_coverage.pth startup hook activates coverage there too.
export COVERAGE_PROCESS_START="$PWD/$CONFIG"

# Test path(s) to run; defaults to the whole suite, override e.g. with
# `tests/integration` (as the CSCS integration job does).
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(tests/)

# `-o addopts=` clears the project default addopts ("-m 'not longtest'" plus the
# Tier-1 --cov flags) so this run uses ONLY the subprocess-aware Tier-2 config.
uv run pytest "${TARGETS[@]}" \
    -m longtest \
    -o addopts= \
    --cov \
    --cov-config="$CONFIG" \
    --cov-report=term-missing
