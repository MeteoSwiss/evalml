#!/bin/bash
#SBATCH --job-name=anemoi-inference
#SBATCH --partition=debug  
#SBATCH --mem=0
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
##SBATCH --gres=gpu:4
#SBATCH --error=/users/omiralle/evalml/slogs/stage_C-run-%a.err
#SBATCH --output=/users/omiralle/evalml/slogs/stage_C-run-%a.out
#SBATCH --array=1-1%10

cd /users/omiralle/evalml/
source .venv/bin/activate
source ~/.bashrc
export ECCODES_DEFINITION_PATH="/users/omiralle/evalml/.venv/share/eccodes-cosmo-resources/definitions"
export TZ=UTC
evalml experiment /users/omiralle/evalml/config/myconfig.yaml --report
