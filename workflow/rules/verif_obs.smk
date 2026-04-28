from pathlib import Path
from datetime import datetime, timedelta


def _parse_timedelta(s: str) -> timedelta:
    """Parse a frequency string like '12h' or '1d' into a timedelta."""
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    raise ValueError(
        f"Unsupported frequency format: {s!r} (expected e.g. '12h' or '1d')"
    )


# TODO: merge _parse_steps from generate_mec_namelist.py and verif_single_init.py?
def _parse_steps(steps: str) -> list[int]:
    # check that steps is in the format "start/stop/step"
    if "/" not in steps:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    if len(steps.split("/")) != 3:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    start, end, step = map(int, steps.split("/"))
    return list(range(start, end + 1, step))


# TODO: merge with _ref_times from common.smk?
def _reftimes_mec():
    """
    Construct ref times for MEC. Needs to be max of all
    leadtimes shorter than ref times from the config.
    """
    cfg = config["dates"]
    if isinstance(cfg, list):
        return [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in cfg]
    start = datetime.strptime(cfg["start"], "%Y-%m-%dT%H:%M")
    leads = _parse_steps(config["runs"][0]["forecaster"]["steps"])
    start_mec = start + timedelta(hours=max(leads))
    end = datetime.strptime(cfg["end"], "%Y-%m-%dT%H:%M")
    freq = _parse_timedelta(cfg["frequency"])
    times = []
    t = start_mec
    while t <= end:
        times.append(t)
        t += freq
    return times


REFTIMES_MEC = _reftimes_mec()


def init_times_for_mec(wc):
    """
    Return list of init times (YYYYMMDDHHMM) from init_time - lead ... init_time
    stepping by configured frequency.
    """
    init = wc.init_time
    base = datetime.strptime(init, "%Y%m%d%H%M")

    lt = get_leadtime(wc)  # expects something like "48h"
    lead_h = int(str(lt).rstrip("h"))
    freq_td = _parse_timedelta(config["dates"]["frequency"])

    # iterate from base - lead to base stepping by the parsed timedelta
    t = base - timedelta(hours=lead_h)
    times = []

    while t <= base:
        times.append(t.strftime("%Y%m%d%H%M"))
        t += freq_td
    return times


# prepare_mec_input: setup run dir, gather observations and model data in the run dir for the actual init time
rule prepare_mec_input:
    input:
        inference_ok=lambda wc: expand(
            rules.inference_execute.output.okfile,
            run_id=wc.run_id,
            init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
        ),
    output:
        obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        ekf_file=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs/ekfSYNOP.nc",
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/prepare_mec_input.log",
    shell:
        """
        (
        set -euo pipefail
        shopt -s nullglob

        mkdir -p {output.obs}

        # extract YYYYMM from init_time (which is YYYYMMDDHHMM)
        init="{wildcards.init_time}"
        ym="${{init:0:6}}"
        echo "init time: ${{init}}"

        # collect observations (ekfSYNOP) and/or (monSYNOP from DWD; includes precip) files
        cp /store_new/mch/msopr/osm/KENDA-CH1/EKF/${{ym}}/ekfSYNOP_${{init}}00.nc {output.ekf_file}
        cp /scratch/mch/paa/mec/MEC_ML_input/monFiles2025/${{init:0:10}}/monSYNOP.nc {output.obs}/monSYNOP.nc
        ######cp /scratch/mch/paa/mec/MEC_ML_input/monFiles2020/hpc/uwork/swahl/temp/feedback/monSYNOP.${{init:0:10}} {output.obs}/monSYNOP.nc
        echo "Copied obs files to {output.obs}"

        ) > {log} 2>&1
        """


# link_mec_input: create the input_mod dir with symlinks to all fc files from all source inits
rule link_mec_input:
    input:
        # depend on ALL source grib dirs: for each lead l, source_init = init_time - l hours
        obs_file=rules.prepare_mec_input.output.ekf_file,
        src_dirs=lambda wc: expand(
            str(OUT_ROOT / "data/runs/{run_id}/{src_init}/grib"),
            run_id=wc.run_id,
            src_init=[
                (
                    datetime.strptime(wc.init_time, "%Y%m%d%H%M") - timedelta(hours=l)
                ).strftime("%Y%m%d%H%M")
                for l in _parse_steps(RUN_CONFIGS[wc.run_id]["steps"])
            ],
        ),
    output:
        # own the final input_mod directory for this init (and its contents)
        mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
    params:
        # generate a space-separated list of lead hours from the run config
        leads=lambda wc: " ".join(
            str(l) for l in _parse_steps(RUN_CONFIGS[wc.run_id]["steps"])
        ),
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/link_mec_input.log",
    shell:
        """
        (
        set -euo pipefail

        mkdir -p {output.mod}
        cd {output.mod}/../../..

        init="{wildcards.init_time}"
        echo "Creating input_mod links for init $init (leads: {params.leads})"

        # for each configured lead copy (and optionally merge) source files into input_mod
        for lead in {params.leads}; do
            lead3=$(printf "%03d" "$lead")
            # compute source init such that source_init + lead = ref(init)
            src_epoch=$(date -u -d "${{init:0:4}}-${{init:4:2}}-${{init:6:2}}T${{init:8:2}}:${{init:10:2}}:00Z" +%s)
            source_init=$(date -u -d "@$(( src_epoch - lead * 3600 ))" +"%Y%m%d%H%M")
            src_rel="$source_init/grib/${{source_init}}_${{lead3}}.grib"

            if [[ -e "$src_rel" ]]; then
                dest="{wildcards.init_time}/mec/input_mod/${{source_init}}.grib"
                if [[ "$lead" -eq 0 ]]; then
                    echo "Copying $src_rel -> $dest"
                    cp "$src_rel" "$dest"
                else
                    prev_lead3=$(printf "%03d" "$(( lead - 6 ))")
                    prev_rel="$source_init/grib/${{source_init}}_${{prev_lead3}}.grib"
                    if [[ -e "$prev_rel" ]]; then
                        echo "Merging $prev_rel + $src_rel -> $dest"
                        cat "$prev_rel" "$src_rel" > "$dest"
                    else
                        echo "WARNING: previous lead file $prev_rel not found, copying $src_rel only" >&2
                        cp "$src_rel" "$dest"
                    fi
                fi
            else
                echo "WARNING: source file $src_rel not found" >&2
            fi
        done
        ) > {log} 2>&1
        """


rule generate_mec_namelist:
    localrule: True
    input:
        script="workflow/scripts/generate_mec_namelist.py",
        template="resources/mec/namelist.jinja2",
        mod_dir=directory(rules.link_mec_input.output.mod),
    output:
        namelist=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/namelist",
    params:
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
    shell:
        """
        uv run {input.script} \
            --steps {params.steps} \
            --init_time {wildcards.init_time} \
            --template {input.template} \
            --namelist {output.namelist}
        """


rule run_mec:
    input:
        namelist=rules.generate_mec_namelist.output.namelist,
        prepare_obs=rules.prepare_mec_input.output.ekf_file,
        mod_dir=directory(rules.link_mec_input.output.mod),
    output:
        fdbk_file=OUT_ROOT / "data/runs/{run_id}/fdbk_files/verSYNOP_{init_time}00.nc",
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/run_mec.log",
    shell:
        """
        (
        set -euo pipefail

        # Run MEC inside sarus container
        # Note: pull command currently needed only once to download the container
        sarus pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        run_dir=$(dirname {input.namelist})
        abs_run_dir=$(realpath "$run_dir")
        abs_mod_root=$(realpath "$run_dir/../..")   # two levels up (so that all links are mounted to the container)

        # build mount options in a variable for readability
        MOUNTS="\
          --mount=type=bind,source=$abs_run_dir,destination=/src/bin2 \
          --mount=type=bind,source=$abs_mod_root,destination=$abs_mod_root,readonly \
          --mount=type=bind,source=/oprusers/osm/opr.inn/data/,destination=/oprusers/osm/opr.inn/data/ \
        "

        # run container (split over multiple lines for readability)
        sarus run $MOUNTS container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main

        # Run MEC using local executable (Alternative to sarus container)
        #cd "$run_dir"
        #export LM_HOST=balfrin-ln003
        #source /oprusers/osm/opr.inn/abs/mec.env
        #./mec > ./mec_out.log 2>&1

        # copy the output file to the final location for the Feedback files plus renaming to
        # match NWP naming conventions (verSYNOP_YYYYMMDDHHMMSS.nc)
        mkdir -p "$run_dir/../../fdbk_files"
        cp "$run_dir/verSYNOP.nc" "$run_dir/../../fdbk_files/verSYNOP_{wildcards.init_time}00.nc"
        ) > {log} 2>&1
        """
