#!/usr/bin/env bash
# Run the members of an ensemble forecaster for one init time (rule inference_execute).
#
# Usage: inference_execute_members.sh GPUS MEMBERS SEED LOGDIR
#
# Runs inside the srun step of inference_execute, in the init-time working directory
# (it holds config_001.yaml ... config_<MEMBERS>.yaml), and expects the exported
# function `_run_inference <config file>` to be defined.
#
# GPUS worker loops run concurrently, one per GPU of the step (pinned with CUDA_VISIBLE_DEVICES),
# and take the members round-robin: worker s runs members s+1, s+1+GPUS, ... Member m
# is seeded with SEED+m (EVALML_SEED). Output goes to LOGDIR/<NNN>.log. A member that
# succeeded leaves LOGDIR/<NNN>.ok and is skipped on a retry, unless its config
# was written again since. The script exits non-zero
# if any member failed; the remaining members still run, so a retry only redoes the
# failed ones.
set -euo pipefail

gpus=$1
members=$2
seed=$3
logdir=$4

mkdir -p "$logdir"

worker() {
    local slot=$1 m nnn failed=0
    for ((m = slot + 1; m <= members; m += gpus)); do
        nnn=$(printf '%03d' "$m")
        # an .ok older than the member config is from a previous prepare: rerun
        if [ -e "$logdir/$nnn.ok" ] && ! [ "config_$nnn.yaml" -nt "$logdir/$nnn.ok" ]; then
            echo "[$(date)] member $nnn: already done, skipping"
            continue
        fi
        echo "[$(date)] member $nnn: start on GPU ${devices[slot]} (seed $((seed + m)))"
        # env -u: ANEMOI_BASE_SEED would override the member seed in some versions
        if CUDA_VISIBLE_DEVICES=${devices[slot]} EVALML_SEED=$((seed + m)) \
            env -u ANEMOI_BASE_SEED bash -c '_run_inference "$1"' _ "config_$nnn.yaml" \
            >"$logdir/$nnn.log" 2>&1; then
            touch "$logdir/$nnn.ok"
            echo "[$(date)] member $nnn: done"
        else
            echo "[$(date)] member $nnn: FAILED, see $logdir/$nnn.log" >&2
            failed=1
        fi
    done
    return $failed
}

# Slurm does not confine the job to its GPUs on every partition: map the slots to the
# devices it was given rather than to 0..GPUS-1.
IFS=, read -ra devices <<<"${CUDA_VISIBLE_DEVICES:?srun did not set CUDA_VISIBLE_DEVICES}"
if [ "${#devices[@]}" -lt "$gpus" ]; then
    echo "[$(date)] expected $gpus GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

pids=()
for ((slot = 0; slot < gpus; slot++)); do
    worker "$slot" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done

if [ "$failed" -ne 0 ]; then
    echo "[$(date)] some members failed" >&2
    exit 1
fi
echo "[$(date)] all $members members done"
