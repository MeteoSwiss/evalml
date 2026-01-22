from pathlib import Path
from datetime import datetime, timedelta


def get_init_times(wc):
    """
    Return list of init times (YYYYMMDDHHMM) from init_time - lead ... init_time
    stepping by configured frequency (default 12h).
    """
    init = wc.init_time
    lt = get_leadtime(wc)  # expects something like "48h"
    lead_h = int(str(lt).rstrip("h"))
    freq_cfg = RUN_CONFIGS[wc.run_id].get("frequency", "12h")
    freq_h = int(str(freq_cfg).rstrip("h"))
    base = datetime.strptime(init, "%Y%m%d%H%M")
    times = []
    for h in range(lead_h, -1, -freq_h):
        t = base - timedelta(hours=h)
        times.append(t.strftime("%Y%m%d%H%M"))
    return times


rule collect_mec_input:
    input:
        inference_dir=rules.prepare_inference_forecaster.output.grib_out_dir,
    output:
        run=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec"),
        obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
    params:
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        init_list_str=lambda wc: " ".join(get_init_times(wc)),
        run_root=lambda wc: str(OUT_ROOT / f"data/runs/{wc.run_id}"),
    log:
        OUT_ROOT
        / "data/runs/{run_id}/{init_time}/mec/{run_id}-{init_time}_collect_mec_input.log",
    shell:
        """
        (
        set -euo pipefail
        echo "...time at start of collect_mec_input: $(date)"

        # create the input_obs and input_mod dirs
        mkdir -p {output.run} {output.obs} {output.mod}

        # extract YYYYMM from init_time (which is YYYYMMDDHHMM) and use it in the paths
        init="{wildcards.init_time}"
        ym="${{init:0:6}}"
        ymdh="${{init:0:10}}"
        echo "init time: ${{init}}, ym: ${{ym}}"

        # collect observations (ekfSYNOP) and/or (monSYNOP from DWD; includes precip) files
        cp /store_new/mch/msopr/osm/KENDA-1/EKF/${{ym}}/ekfSYNOP_${{init}}00.nc {output.obs}/ekfSYNOP.nc
        cp /scratch/mch/paa/mec/MEC_ML_input/monFiles2020/hpc/uwork/swahl/temp/feedback/monSYNOP.${{init:0:10}} {output.obs}/monSYNOP.nc

        # For each source init (src_init) produce one output file named fc_<src_init>
        for src_init in {params.init_list_str}; do
            src_dir="{params.run_root}/$src_init/grib"
            out_file="{output.mod}/fc_$src_init"
            echo "creating $out_file from $src_dir"
            # create/truncate out_file
            : > "$out_file"
            # only concat if matching files exist
            if compgen -G "$src_dir/20*.grib" > /dev/null; then
                cat "$src_dir"/20*.grib >> "$out_file"
            else
                echo "WARNING: no grib files found in $src_dir" >&2
            fi
        done

        ls -l {output.mod} {output.obs}
        echo "...time at end of collect_mec_input: $(date)"
        ) > {log} 2>&1
        """


rule generate_mec_namelist:
    localrule: True
    input:
        script="workflow/scripts/generate_mec_namelist.py",
        template="resources/mec/namelist.jinja2",
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
        run_dir=directory(rules.collect_mec_input.output.run),
    output:
        fdbk_file=OUT_ROOT / "data/runs/{run_id}/fdbk_files/verSYNOP_{init_time}.nc",
    params:
        final_fdbk_file_dir=lambda wc: str(OUT_ROOT / f"data/runs/{wc.run_id}/fdbk_files"),
    resources:
        cpus_per_task=1,
        runtime="1h",
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/{run_id}-{init_time}_run_mec.log",
    shell:
        """
        (
        set -euo pipefail
        echo "...time at start of run_mec: $(date)"

        # Run MEC inside sarus container
        # Note: pull command currently needed only once to download the container
        #### sarus pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        abs_run_dir=$(realpath {input.run_dir})
        sarus run --mount=type=bind,source=$abs_run_dir,destination=/src/bin2 --mount=type=bind,source=/oprusers/osm/opr.emme/data/,destination=/oprusers/osm/opr.emme/data/ container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main

        # Run MEC using local executable (Alternative to sarus container)
        #cd {input.run_dir}
        #export LM_HOST=balfrin-ln002
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
        sarus pull container-registry.meteoswiss.ch/ffv2ctr/ffv2-container:0.1.0-dev
        namelist=$(realpath {input.namelist})
        domain_table=$(realpath {params.domain_table})
        blacklists=$(realpath {params.blacklists})
        # Mount needs to have source as absolute path
        feedback_dir_abs=$(realpath {input.feedback_directory})
        # DO NOT SUBMIT: Need to mount feedback files, with absolute path
        sarus run \
        --mount=type=bind,source=$namelist,destination=/src/ffv2/SYNOP_DET.nl \
        --mount=type=bind,source=$domain_table,destination=$domain_table \
        --mount=type=bind,source=$blacklists,destination=$blacklists \
        --mount=type=bind,source=$feedback_dir_abs,destination={input.feedback_directory} \
        container-registry.meteoswiss.ch/ffv2ctr/ffv2-container:0.1.0-dev

        echo "...time at end of run_ffv2: $(date)"
        ) > {log} 2>&1
        """