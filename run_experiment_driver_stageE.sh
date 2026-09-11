#!/bin/bash
#SBATCH --job-name=evalml-driver-stageE
#SBATCH --partition=postproc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1800
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mch/huppd/varda/evalml/driver-stageE-%j.log

set -euo pipefail
cd /scratch/mch/huppd/varda/evalml
source .venv/bin/activate
evalml experiment config/forecasters-ich1-stageE.yaml --report
