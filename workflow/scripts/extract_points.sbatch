#!/bin/bash
# Slurm array job for stage 1 of the temporal-variability analysis: extract
# forecast series at the 145 SwissMetNet grid points, one array task per
# (season, run, init). See workflow/scripts/extract_points.py.
#
# Build the task list, then submit with as many tasks as it has lines:
#   uv run python workflow/scripts/extract_points_joblist.py > output/points/joblist.txt
#   sbatch --array=1-$(wc -l < output/points/joblist.txt)%20 \
#          workflow/scripts/extract_points.sbatch output/points/joblist.txt
#
# Concurrency: 50 parallel tasks starved each other on GRIB I/O (tasks that take
# 3.5 min alone took up to 29 min), so keep it near 20.
#
# Each task is independent and skips work already done, so a partially failed
# array can simply be resubmitted.
#SBATCH --job-name=extract_points
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --output=output/points/logs/%A_%a.log

set -euo pipefail

JOBLIST=${1:?usage: sbatch extract_points.sbatch <joblist.txt>}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
GRIB_DIR=$(echo "$LINE" | cut -d' ' -f1)
OUT=$(echo "$LINE" | cut -d' ' -f2)

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: $GRIB_DIR -> $OUT"
export PYTHONPATH=src
uv run python workflow/scripts/extract_points.py --grib-dir "$GRIB_DIR" --out "$OUT"
echo "[$(date)] done"
