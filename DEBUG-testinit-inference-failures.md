# Debug handover: test-init inference failures

Status as of 2026-09-10. Written for a separate debugging conversation.

## RESOLVED 2026-09-10

Two unrelated causes, not one:

1. **The balfrin -> linard move** broke how the venv squashfs gets mounted.
   Failures A and B are both this.
2. **The new configs were never known-good.** They omit the version pins that
   every tracked forecaster config carries. This is independent of linard and
   would have failed on balfrin too, which answers the open question at the
   bottom of this document: no, this config was never run successfully.

### Status per failure

- **Failure A**: real. Fixed by adopting **PR 260**, not by the syntax change
  suggested below. See "Applied fix".
- **Failure B**: not an independent failure. Snakemake SIGTERMed the still-running
  env build while shutting down after A. Nothing to fix.
- **Secondary resource suspicion**: does not apply. `get_resource` reads only the
  run config, and no run here sets `inference_resources`, so the srun uses the
  in-rule defaults (`short-shared`, 24 cpus, 8000 MB, `gpu:1`). The `profile`
  block never reaches it.
- **Failure C** (new, found after A and B were fixed): `KeyError: 'FI_100'`.
  See "Failure C" below.

### Relevant work by others

Colleagues hit the same linard breakage on 2026-09-09. Check these before
re-fixing anything:

- **PR 260** `fix/inference-srun-uenv-mount` (dnerini, open, based on
  `fix/linard-fixes`): replaces `squashfs-mount` with `srun --uenv`. This is the
  correct fix for A. `fix/linard-fixes` itself has no commits of its own, it is
  just an older `main` used as the PR base.
- **PR 262** `fix/inference-srun-account` (lclanzi, open, based on `main`):
  adds `--account=` to the inner srun. Independent real bug, see below.
- **PR 261** `feat/check-missing-steps`: closed unmerged, rejected in favour of
  issue #247. Not relevant here, but the scenario it describes (asking for
  `TOT_PREC1` from a forecaster coarser than 1 h, surfacing as an obscure pandas
  error) is worth remembering if verification fails oddly.

PRs 260 and 262 both edit the same srun call and conflict. Both are applied to
the working tree here, merged by hand.

Evidence for each is recorded in the sections below.

## What was being run and why

A single-init smoke test of the multistep vs Varda stage C comparison, config
[config/multistep-stage-C-testinit.yaml](config/multistep-stage-C-testinit.yaml),
one init at `2024-12-30T12:00`, two runs (Varda two-stage, multistep), no
baselines, no showcases.

It was meant to answer two questions before committing 82 inits x 2 models:

1. Whether a late-December init completes all 120 h even though the global
   initial-condition datasets end 2024-12-31. `2024-12-30T12` is the latest init
   in the planned winter sample and therefore the worst case.
2. Whether REA-L-CH1 has usable 2-6 h temporal spectral power at all, which the
   primary diagnostic in [exp_plan_01.md](exp_plan_01.md) depends on.

**Neither question was answered.** The workflow failed before any forecast data
was produced. Both failures are environment and infrastructure problems, not
data problems.

## Reproduce

```bash
uv run evalml experiment config/multistep-stage-C-testinit.yaml -n   # dry-run, resolves 22 jobs, OK
uv run evalml experiment config/multistep-stage-C-testinit.yaml -v   # fails after ~5m20s
```

Snakemake command that was actually issued (from the driver output):

```
snakemake --executor slurm --resources gpus=16 \
  --default-resources slurm_partition=postproc cpus_per_task=1 \
  mem_mb_per_cpu=1800 runtime=1h slurm_account=s83 gpus=0 \
  --jobs 50 --configfile config/multistep-stage-C-testinit.yaml --cores 4 experiment_all
```

Persistent logs (the driver log itself was in a session scratchpad and is gone):

- `output/testinit/logs/inference_execute/forecaster-0eb1-7ed2/1b33-202412301200.log`
- `output/testinit/logs/inference_prepare_env/*.log`
- `.snakemake/logs/`, `.snakemake/slurm_logs/`

10 of 22 steps completed before the failure.

## Failure A: squashfs-mount syntax incompatibility (root cause found, fix confirmed)

The multistep `inference_execute` job failed with:

```
error: unable to exec '/scratch/.../output/testinit/data/runs/forecaster-0eb1-7ed2/venv.squashfs:/user-environment': No such file or directory (errno=2)
```

The squashfs image does exist (4.7 GB), so this is not a missing file despite
what the message suggests.

The rule invokes, at the end of the shell block in
[workflow/rules/inference.smk](workflow/rules/inference.smk) (around line 322):

```bash
squashfs-mount {params.env_path}:/user-environment -- bash -c '_run_inference /user-environment'
```

That is the **old** squashfs-mount interface, where the first positional is an
`image:mountpoint` spec. The version installed here is different:

```
$ squashfs-mount --version
squashfs-mount 10.0.1

Usage: squashfs-mount [OPTIONS] [commands...]
Positionals:
  commands TEXT ...           the command to run, including with arguments
Options:
  -s,--sqfs TEXT              comma separated list of squashfs files to mount
```

Version 10.0.1 has no `image:mountpoint` positional at all. The image must be
passed via `-s`/`--sqfs`, and the only positional is the command to run. So
squashfs-mount is treating `<path>:/user-environment` as the command it should
exec, which is exactly the error observed.

### Applied fix

The suggestion above (`-s {params.env_path}`, image only) does **not** work.
10.0.1 still wants the full `image:mountpoint` spec, it just wants it behind
`-s`:

```
$ squashfs-mount -s /path/to/venv.squashfs -- bash -c 'ls /user-environment'
error: expected a ':' separating the squashfs image and mount path, found ''
```

So the mountpoint is still explicit and still `/user-environment`, and the
`$VENV` argument to `_run_inference` is unchanged. Verified by mounting the
existing 4.7 GB image and importing eccodes from it.

**But fixing the syntax is not enough, and is not the fix that was applied.**
With the corrected syntax the job got further and then failed with:

```
error: execve(): anemoi-inference: No such file or directory
srun: error: nid002832: task 0: Exited with exit code 2
```

`squashfs-mount` mounts the image in the *login node's* mount namespace, but the
work happens on a compute node via the inner `srun`, which never sees that mount.
Whatever made this work on balfrin does not on linard.

The applied fix is **PR 260**: drop `squashfs-mount` and let slurm do the mount
with `srun --uenv=<image>:/user-environment`, via linard's `slurm-uenv-mount`
SPANK plugin, which mounts inside the task's own namespace on the allocated node.
The rule's inner function is no longer passed a path, it hardcodes
`/user-environment`. PR 260 also sets `--job-name=anemoi-inference` so the job
does not appear as "bash" in the queue.

Verified directly on a compute node before rerunning:

```
$ srun --uenv=$IMG:/user-environment --partition=short-shared --gres=gpu:1 ... \
    bash -c 'source /user-environment/bin/activate && which anemoi-inference && \
             python -c "import torch;print(torch.cuda.is_available())"'
/user-environment/bin/anemoi-inference
cuda: True
```

**PR 262** is also applied. It is a genuine separate bug: `slurm_account` was
only wired into snakemake's `--default-resources` and never reached the inner
srun, so inference billed the cluster default account. `sacctmgr` confirms the
default for this user is `msclim`, not the `s83` the config declares. Confirmed
working in the rerun, where the job shows as `anemoi-inferen` under `s83`.

## Failure B: environment build killed with SIGTERM

`inference_prepare_env` for the temporal downscaler died with `SIGTERM` during
`uv pip install`:

```
/usr/local/bin/bash: line 1: 55190 Terminated   uv pip install -r .../requirements.txt
```

The log shows it got as far as building `anemoi-plugins-meteoswiss` and
preparing 6 packages, then was killed. The subsequent
`ERROR: eccodes is not installed correctly` line in the driver output is just
the rule's own check text being echoed in the failure report, not the actual
cause.

Note the two *forecaster* environments built fine in the same run, producing
4.4 and 4.7 GB squashfs images. Only the third one died.

### Resolution: falsified, it was Snakemake shutting down

The `/dev/shm` hypothesis below is wrong. Timeline from
`.snakemake/log/2026-09-10T103053.279810.snakemake.log`:

1. 10:33:31 `inference_execute` fails (failure A).
2. Snakemake logs `Will exit after finishing currently running jobs (scheduler).`
3. The downscaler `inference_prepare_env`, still running, gets SIGTERM.

That is Snakemake's own teardown, not a watchdog. Supporting evidence against
the RAM-disk story: linard's login node has 502 GB RAM with `/dev/shm` at 353 GB
and 316 GB free, no memory limit on `user.slice` or the per-user slice
(`memory.max` is `max`), no systemd-oomd, and the rule's `trap ... EXIT` left no
`evalml_*` directories behind. Three ~10 GB venvs were never close to a limit.

Fix failure A and this build simply runs. No concurrency limit needed.

Original hypothesis, kept for the record: `/dev/shm` contention. The rule builds
each venv in a RAM disk:

```bash
VENV_DIR=$(mktemp -d /dev/shm/evalml_XXXXXXXX)
```

`inference_prepare_env` is a `localrule`, so all three builds ran concurrently
on the login node with `--cores 4`. Three uncompressed venvs of roughly 10 GB
each in `/dev/shm` at once could exhaust the RAM disk or trip a login-node
memory watchdog, and SIGTERM (rather than SIGKILL from a cgroup OOM) points at a
watchdog or a `/dev/shm` limit rather than a plain out-of-memory kill.

Cheap things to try:

- Re-run with `-c 1` so the env builds serialise. If it then succeeds, this is
  confirmed and the real fix is a concurrency limit on that rule.
- Check `df -h /dev/shm` and the login-node limits while a build is running.
- The two forecaster squashfs images already exist in `output/testinit/data/runs/`,
  so a re-run only needs to build the downscaler environment. That makes the
  serialised retry cheap.

## Failure C: KeyError 'FI_100' (found after A and B were fixed)

With A and B fixed, both forecasters reach a GPU node, load their checkpoints
and start inference. Both then fail identically while writing the initial state
to GRIB:

```
File ".../anemoi/inference/outputs/grib.py", line 269, in write_initial_state
    variable = self.typed_variables[name]
KeyError: 'FI_100'
```

`typed_variables` is keyed by the checkpoint's own variable names, so a field
called `FI_100` is arriving in the state while the checkpoint knows it as
`z_100`. `FI` is the MeteoSwiss name for geopotential.

Ruled out, each by checking rather than reasoning:

- **Patch metadata**: the `variables_metadata` block in
  `sgm-multidataset-ich1-patch.yaml` is byte-identical to the working
  `sgm-multidataset-ich1-oper-patch.yaml`. Both declare `z_100` with
  `mars.param: FI`. The two files differ only in dataset paths and timesteps.
- **Datasets**: all four zarrs involved, old and new, name the variable `z_100`.
  Checked via `.zattrs`.
- **Inference configs**: `sgm-multidataset-forecaster-global-ich1.yaml` is
  identical to the tracked oper config except for the `patch_metadata:` line.
- **anemoi-inference version, for this code path**: `write_initial_state` in
  `grib.py` is unchanged between 0.11.1 and 0.12.0.

Remaining explanation, and the one being tested: **missing version pins.** The
two forecaster entries in the testinit config declared no `extra_requirements`,
so they built against anemoi-inference 0.12.0, earthkit-data 0.20.0 and
earthkit-utils 0.3.0. Every tracked forecaster config in `config/` pins:

```yaml
      extra_requirements:
        - earthkit-utils<0.2.0
        - earthkit-data<0.19.0
        - anemoi-inference==0.11.1
```

earthkit is the layer that names fields from GRIB/mars metadata, which is where
an `FI` + level name would be constructed. The downscaler entry was already
pinned; the nested `forecaster:` entry simply did not inherit it.

These pins are now added to both forecaster entries in the testinit config.
Note this changes the env hash, so all three environments rebuild.

Status: **hypothesis, not yet confirmed.** The rerun with pins is the test.

## Secondary suspicion: checked, does not apply

`get_resource` (defined at `workflow/rules/inference.smk:157`) reads exclusively
from `RUN_CONFIGS[run_id]["inference_resources"]`. Neither run in this config
sets that key, so every lookup falls through to the in-rule default. The srun
therefore asks for `short-shared`, 24 cpus, 8000 MB per cpu and `gpu:1`, which
is what is wanted.

The `profile.default_resources` block only feeds Snakemake's own slurm
executor, and `inference_execute` is a `localrule` that builds its own srun, so
the block cannot reach it. Snakemake's `--default-resources` also only fills
resources a rule does not define, and this rule defines all of them.

The `gpu` vs `gpus` naming noted below is real but harmless here for the same
reason: nothing in the config feeds those lookups. It is still a trap for anyone
who later adds an `inference_resources` block, so worth renaming upstream.

Original text:

### Secondary suspicion, not yet investigated

The `profile.default_resources` block in the config (inherited unchanged from
`config/multistep-stage-C-analysis_yaml.yaml`) sets `slurm_partition: postproc`,
`cpus_per_task: 1`, `mem_mb_per_cpu: 1800`, `gpus: 0`.

`inference_execute` builds an `srun` call from `resources.slurm_partition`,
`resources.cpus_per_task`, `resources.mem_mb_per_cpu` and
`resources.gres`. Its own in-rule defaults are `short-shared`, 24 cpus,
8000 MB per cpu and `gpu:1`. Note that `gres` is derived from a resource key
named `gpu` (singular), not `gpus`, so the config's `gpus: 0` does not switch the
GPU request off: the srun would still ask for `--gres=gpu:1`, but on the
`postproc` partition.

If the profile's defaults do propagate into that srun, inference would be asking
for a GPU on a partition that has none, with 1 cpu and 1.8 GB. This did not
surface as an error yet because failure A killed the job before srun was
reached. Worth settling by resolving what `get_resource` actually returns here.
It is defined somewhere other than `workflow/rules/common.smk`.

Related: commit 4641fe6 "Scale the number of physical cores requested with the
expected memory usage (#252)" suggests resource sizing in this pipeline is a
known live issue.

## What is known to work

- Config validation: all three new configs parse against `evalml.config.ConfigModel`.
- DAG resolution: dry-run produces 22 jobs with no errors.
- `inference_get_checkpoint` and `inference_extract_requirements` for all three
  environments.
- `inference_prepare_env` for both forecaster environments.
- Dataset coverage was audited separately and is fine. See the coverage table in
  [exp_plan_01.md](exp_plan_01.md).

## Was this config ever known-good?

**Answered: no.** The missing version pins behind failure C are not something a
successful run would have left behind, and they have nothing to do with linard.
So the config is a first-run draft, and failure C is a teething problem in it,
while failures A and B are genuine linard regressions affecting everyone.

Original text:

Unclear, and it matters. The `profile` block and the run definitions were copied
verbatim from `config/multistep-stage-C-analysis_yaml.yaml`, which is an
untracked draft. If that draft was never executed successfully, both failures
here are first-run teething problems on this checkout rather than regressions.

Note the same missing-pin problem very likely applies to the other untracked
draft configs copied from the same source: `multistep-stage-C-analysis_yaml.yaml`,
`multistep-stage-C-summer2024.yaml` and `multistep-stage-C-winter2024.yaml`.
Worth fixing there too once the pins are confirmed to work.
