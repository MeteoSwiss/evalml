from pathlib import Path
from datetime import datetime, timedelta


def init_times_for_mec(wc):
    """
    Return list of init times (YYYYMMDDHHMM) from init_time - lead ... init_time
    stepping by configured frequency.
    """
    init = wc.init_time
    lt = get_leadtime(wc)  # expects something like "48h"
    lead_h = int(str(lt).rstrip("h"))
    dates_cfg = config["dates"]

    # use the same parsing as in common.smk; support "Xh" and "Xd"
    freq_td = _parse_timedelta(dates_cfg["frequency"])

    base = datetime.strptime(init, "%Y%m%d%H%M")
    times = []

    # iterate from base - lead to base stepping by the parsed timedelta
    t = base - timedelta(hours=lead_h)
    while t <= base:
        times.append(t.strftime("%Y%m%d%H%M"))
        t += freq_td
    return times


# prepare_mec_input: setup run dir, gather observations and model data in the run dir for the actual init time
rule prepare_mec_input:
    input:
        src_dir=OUT_ROOT / "data/runs/{run_id}/{init_time}/grib",
    output:
        run=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec"),
        obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        ekf_file=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs/ekfSYNOP.nc",
        # prepare_mec_input no longer claims ownership of input_mod dir;
        # it should produce the fc file somewhere visible (or create a temp),
        # but we keep fc as a produced file returned here in the mec dir
        fc_file=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/fc_{init_time}",
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/prepare_mec_input.log",
    shell:
        """
        (
        #set -uo pipefail

        mkdir -p {output.run} {output.obs}
        src_dir="{input.src_dir}"
        fc_file="{output.fc_file}"

        # extract YYYYMM from init_time (which is YYYYMMDDHHMM)
        init="{wildcards.init_time}"
        ym="${{init:0:6}}"
        ymdh="${{init:0:10}}"
        echo "init time: ${{init}}"

        # concatenate all grib files in src_dir into a single file fc_file
        echo "grib files processed:"
        if ls "$src_dir"/20*.grib >/dev/null 2>&1; then
            # concatenate all matching files into the target file
            ls  "$src_dir"/20*.grib
            cat "$src_dir"/20*.grib > "$fc_file"
        else
            echo "WARNING: no grib files found in $src_dir" >&2
        fi

        # collect observations (ekfSYNOP) and/or (monSYNOP from DWD; includes precip) files
        cp /store_new/mch/msopr/osm/KENDA-1/EKF/${{ym}}/ekfSYNOP_${{init}}00.nc {output.ekf_file}
        cp /scratch/mch/paa/mec/MEC_ML_input/monFiles2020/hpc/uwork/swahl/temp/feedback/monSYNOP.${{init:0:10}} {output.obs}/monSYNOP.nc

        ) > {log} 2>&1
        """


# link_mec_input: create the input_mod dir with symlinks to all fc files from all source inits
rule link_mec_input:
    input:
        # list of source fc files produced by prepare_mec_input for each init in the window
        fc_files=lambda wc: [
            OUT_ROOT / f"data/runs/{wc.run_id}/{t}/mec/fc_{t}"
            for t in init_times_for_mec(wc)
        ],
    output:
        # own the final input_mod directory for this init (and its contents)
        mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/link_mec_input.log",
    shell:
        """
        (
        set -euo pipefail

        mkdir -p {output.mod}
        cd {output.mod}/../../..

        # create symlinks for each source init into this init's input_mod
        for src in {input.fc_files}; do
            src_basename=$(basename "$src")
            echo "Processing source fc file: $src_basename"
            one_init_time="${{src_basename: -12}}"
            realpath_src=$(realpath -m "$PWD/$one_init_time/mec/")

            echo "Linking $realpath_src/$src_basename to {wildcards.init_time}/mec/input_mod/$src_basename" 
            ln -s "$realpath_src/$src_basename" {wildcards.init_time}/mec/input_mod/"$src_basename"
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
        run_dir=directory(rules.prepare_mec_input.output.run),
        mod_dir=directory(rules.link_mec_input.output.mod),
    output:
        fdbk_file=OUT_ROOT / "data/runs/{run_id}/fdbk_files/verSYNOP_{init_time}.nc",
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/run_mec.log",
    shell:
        """
        (
        set -euo pipefail
 
        # Run MEC inside sarus container
        # Note: pull command currently needed only once to download the container
        sarus pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        abs_run_dir=$(realpath {input.run_dir})
        abs_mod_root=$(realpath {input.run_dir}/../..)   # two levels up (so that all links are mounted to the container)

        # build mount options in a variable for readability
        MOUNTS="\
          --mount=type=bind,source=$abs_run_dir,destination=/src/bin2 \
          --mount=type=bind,source=$abs_mod_root,destination=$abs_mod_root,readonly \
          --mount=type=bind,source=/oprusers/osm/opr.emme/data/,destination=/oprusers/osm/opr.emme/data/ \
        "

        # run container (split over multiple lines for readability)
        sarus run $MOUNTS container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
 
        # Run MEC using local executable (Alternative to sarus container)
        #cd {input.run_dir}
        #export LM_HOST=balfrin-ln003
        #source /oprusers/osm/opr.emme/abs/mec.env
        #./mec > ./mec_out.log 2>&1
 
        # copy the output file to the final location for the Feedback files
        mkdir -p {input.run_dir}/../../fdbk_files
        cp {input.run_dir}/verSYNOP.nc {input.run_dir}/../../fdbk_files/verSYNOP_{wildcards.init_time}.nc
        ) > {log} 2>&1
        """
