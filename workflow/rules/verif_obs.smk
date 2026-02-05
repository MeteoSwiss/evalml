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
    params:
        final_fdbk_file_dir=lambda wc: str(OUT_ROOT / f"data/runs/{wc.run_id}/fdbk_files"),
    resources:
        cpus_per_task=1,
        runtime="1h",
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

        # move the output file to the final location for the Feedback files
        mkdir -p {params.final_fdbk_file_dir}
        cp {input.run_dir}/verSYNOP.nc {params.final_fdbk_file_dir}/verSYNOP_{wildcards.init_time}.nc
        echo "...time at end of run_mec: $(date)"
        ) > {log} 2>&1
        """

rule generate_ffv2_namelist:
    localrule: True
    input:
        script="workflow/scripts/generate_ffv2_namelist.py",
        template="resources/ffv2/template_SYNOP_DET.nl.jinja2",
        # This will cause the namelist generation to block on MEC running.
        # Not strictly needed for namelist to be generated, but namelist 
        # generation script checks that the feedback dirs exist.
        # So blocking is desireable.
        # QUESTION: We may want more than one directory here, if we are comparing models.
        feedback_directory=rules.run_mec.params.final_fdbk_file_dir,
    output:
        # Question: Definitely want to aggregate over init time, but will we have 1 run_ffv2 per run_id, or 1 run of ffv2 for all run_ids?
        namelist=OUT_ROOT / "data/runs/{run_id}/SYNOP_DET.nl",
    params:
        # TODO: consider including run_ids here?
        experiment_ids="SrucMLModel,",
        # Keeping this as a param. We will create it in run_ffv2 rule.
        output_directory=lambda wc: str(OUT_ROOT / f"data/runs/{wc.run_id}/scores"),
        # TODO: update descriptions to something more fitting
        experiment_description="emulator_onPL_ALL_obs_2020",
        file_description="exp_ACOSMO-2-models_C-2E-CTRL_2020",
        domain_table="/users/paa/01_store/02_FFV2/data/7_ML_inner_polygon",
        blacklists="/users/paa/01_store/02_FFV2/data/blacklist",
    shell:
        """
        mkdir -p {params.output_directory}
        uv run {input.script} \
            --template {input.template} \
            --namelist {output.namelist} \
            --experiment_ids {params.experiment_ids} \
            --feedback_directories {input.feedback_directory} \
            --output_directory {params.output_directory} \
            --experiment_description {params.experiment_description} \
            --file_description {params.file_description} \
            --domain_table {params.domain_table} \
            --blacklists {params.blacklists}
        """

# Question: one run per run_id? (If not will need to change wildcards around)
rule run_ffv2:
    input:
        namelist=rules.generate_ffv2_namelist.output.namelist,
        # QUESTION: Will we want to compare with other models?
        # Need to specify this in order to mount it.
        feedback_directory=rules.generate_ffv2_namelist.input.feedback_directory,
    output:
        # DO NOT SUBMIT: Fix to use directly.
        scores=directory(OUT_ROOT / "data/runs/{run_id}/scores")
        #scores=directory(rules.generate_ffv2_namelist.params.output_directory),
    params:
        # domain_table and blacklists are locations on Balfrin, that will be
        # mounted into container (with the same filepaths)
        domain_table=rules.generate_ffv2_namelist.params.domain_table,
        blacklists=rules.generate_ffv2_namelist.params.blacklists,
    resources:
        cpus_per_task=1,
        runtime="1h",
    log:
        OUT_ROOT / "data/runs/{run_id}/run_ffv2.log",
    shell:
        """
        (
        set -euo pipefail
        echo "...time at start of run_ffv2: $(date)"

        # Create the output directory to hold scores, if it does not exist
        mkdir -p {output.scores}

        # Run FFV2 inside sarus container
        # Note: pull command currently needed only once to download the container
        # TODO(mmcglohon): Update from dev to main once things work
        sarus pull container-registry.meteoswiss.ch/ffv2ctr/ffv2-container:0.1.0-main
        namelist=$(realpath {input.namelist})
        domain_table={params.domain_table}
        blacklists={params.blacklists}
        # Mount needs to have source as absolute path
        feedback_dir_abs=$(realpath {input.feedback_directory})
        output_dir_abs=$(realpath {output.scores})
        # DO NOT SUBMIT: Need to mount feedback files, with absolute path
        sarus run \
        --mount=type=bind,source=$namelist,destination=/src/ffv2/SYNOP_DET.nl \
        --mount=type=bind,source=$domain_table,destination=$domain_table \
        --mount=type=bind,source=$blacklists,destination=$blacklists \
        --mount=type=bind,source=$feedback_dir_abs,destination=/src/ffv2/input \
        --mount=type=bind,source=$output_dir_abs,destination=/src/ffv2/output \
        container-registry.meteoswiss.ch/ffv2ctr/ffv2-container:0.1.0-main

        echo "...time at end of run_ffv2: $(date)"
        ) > {log} 2>&1
        """
