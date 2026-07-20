import json
from pathlib import Path
from datetime import timedelta


def _parse_steps(steps: str) -> list[int]:
    # check that steps is in the format "start/stop/step"
    if "/" not in steps:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    if len(steps.split("/")) != 3:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    start, end, step = map(int, steps.split("/"))
    return list(range(start, end + 1, step))


def _reftimes_mec():
    """
    Return the subset of REFTIMES that can serve as MEC valid times.

    A time t is eligible if it is at least max_lead hours after the first REFTIMES
    entry, ensuring that at least some source inference runs predate t.
    Source init times that are missing from REFTIMES are handled gracefully by
    link_mec_input (warning, no failure).
    """
    candidates = [cfg for cfg in RUN_CONFIGS.values() if cfg.get("_is_candidate")]
    max_lead = max(max(_parse_steps(run_cfg["steps"])) for run_cfg in candidates)
    return [t for t in REFTIMES if t >= REFTIMES[0] + timedelta(hours=max_lead)]


if config["mec"] is not None:
    REFTIMES_MEC = _reftimes_mec()
else:
    REFTIMES_MEC = []


if config["mec"] is not None:

    rule prepare_mec_input:
        """Collect EKF SYNOP, monSYNOP, and reference verSYNOP observation files into the MEC input_obs directory."""
        input:
            inference_ok=lambda wc: expand(
                rules.inference_execute.output.okfile,
                run_id=wc.run_id,
                init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
            ),
        output:
            obs=directory(OUT_ROOT / "data/runs/{run_id}/mec/{init_time}/input_obs"),
            ekf_file=OUT_ROOT
            / "data/runs/{run_id}/mec/{init_time}/input_obs/ekfSYNOP.nc",
            obs_file=OUT_ROOT
            / "data/runs/{run_id}/mec/{init_time}/input_obs/obsSYNOP.nc",
        log:
            OUT_ROOT / "logs/prepare_mec_input/{run_id}-{init_time}.log",
        params:
            ekf_root=config["mec"]["ekf_root"],
            mon_synop_root=config["mec"]["mon_synop_root"],
            ver_synop_root=config["mec"]["ver_synop_root"],
        shell:
            """
            (
                set -euo pipefail

                mkdir -p {output.obs}

                # extract YYYYMM from init_time (which is YYYYMMDDHHMM)
                init="{wildcards.init_time}"
                ym="${{init:0:6}}"
                echo "init time: ${{init}}"

                # collect observations (ekfSYNOP) and/or (monSYNOP from DWD; includes precip) files
                cp {params.ekf_root}/${{ym}}/ekfSYNOP_${{init}}00.nc {output.ekf_file}
                cp {params.mon_synop_root}/${{init:0:10}}/monSYNOP.nc {output.obs}/monSYNOP.nc
                cp {params.ver_synop_root}/verSYNOP_${{init}}00.nc {output.obs_file}
                echo "Copied obs files to {output.obs}"

            ) >{log} 2>&1
            """

    rule link_mec_input:
        """For each valid time, merge the two inference GRIB files 0h and - 6h and copy it
        from the source init time into input_mod/, named by source init time.
        This assembles the model input directory that MEC expects for a single valid time.
        """
        input:
            # depend on ALL source grib dirs: for each lead l, source_init = init_time - l hours
            obs_file=rules.prepare_mec_input.output.obs_file,
            src_dirs=lambda wc: expand(
                str(OUT_ROOT / "data/runs/{run_id}/{src_init}/grib"),
                run_id=wc.run_id,
                src_init=[
                    (
                        datetime.strptime(wc.init_time, "%Y%m%d%H%M")
                        - timedelta(hours=l)
                    ).strftime("%Y%m%d%H%M")
                    for l in _parse_steps(RUN_CONFIGS[wc.run_id]["steps"])
                    if datetime.strptime(wc.init_time, "%Y%m%d%H%M")
                    - timedelta(hours=l)
                    in set(REFTIMES)
                ],
            ),
        output:
            # own the final input_mod directory for this init (and its contents)
            mod=directory(OUT_ROOT / "data/runs/{run_id}/mec/{init_time}/input_mod"),
        log:
            OUT_ROOT / "logs/link_mec_input/{run_id}-{init_time}.log",
        params:
            # generate a space-separated list of lead hours from the run config
            leads=lambda wc: " ".join(
                str(l) for l in _parse_steps(RUN_CONFIGS[wc.run_id]["steps"])
            ),
        shell:
            """
            (
                set -euo pipefail

                mkdir -p {output.mod}
                cd {output.mod}/../../..

                init="{wildcards.init_time}"
                echo "Creating input_mod files for init $init (leads: {params.leads})"

                # for each configured lead copy (and optionally merge) source files into input_mod
                for lead in {params.leads}; do
                    # compute source init such that source_init + lead = ref(init)
                    src_epoch=$(date -u -d "${{init:0:4}}-${{init:4:2}}-${{init:6:2}}T${{init:8:2}}:${{init:10:2}}:00Z" +%s)
                    source_init=$(date -u -d "@$((src_epoch - lead * 3600))" +"%Y%m%d%H%M")
                    # anemoi-inference writes grib/{{date}}{{time_int}}_{{step_int}}.grib where
                    # time_int is HHMM stripped of leading zeros (e.g. "0000" -> "0", "1800" -> "1800")
                    src_date="${{source_init:0:8}}"
                    src_time_int=$((10#${{source_init:8:4}}))
                    src_rel="$source_init/grib/${{src_date}}${{src_time_int}}_${{lead}}.grib"

                    if [[ -e "$src_rel" ]]; then
                        dest="mec/{wildcards.init_time}/input_mod/${{source_init}}.grib"
                        if [[ "$lead" -eq 0 ]]; then
                            echo "Copying $src_rel -> $dest"
                            cp "$src_rel" "$dest"
                        else
                            prev_lead=$((lead - 6))
                            prev_rel="$source_init/grib/${{src_date}}${{src_time_int}}_${{prev_lead}}.grib"
                            if [[ -e "$prev_rel" ]]; then
                                echo "Merging $prev_rel + $src_rel -> $dest"
                                cat "$prev_rel" "$src_rel" >"$dest"
                            else
                                echo "WARNING: previous lead file $prev_rel not found, copying $src_rel only" >&2
                                cp "$src_rel" "$dest"
                            fi
                        fi
                    else
                        echo "WARNING: source file $src_rel not found" >&2
                    fi
                done
            ) >{log} 2>&1
            """

    rule generate_mec_namelist:
        input:
            script="workflow/scripts/generate_mec_namelist.py",
            template="resources/mec/namelist.jinja2",
            mod_dir=directory(rules.link_mec_input.output.mod),
        output:
            namelist=OUT_ROOT / "data/runs/{run_id}/mec/{init_time}/namelist",
        localrule: True
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

    rule sarus_pull_mec:
        """Pull the MEC sarus container image once before parallel MEC jobs."""
        output:
            touch(OUT_ROOT / "logs/sarus_pull_mec.ok"),
        localrule: True
        shell:
            "sarus pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main"

    rule run_mec:
        """Run the MEC container for one initialisation time, producing a verSYNOP feedback file in fdbk_files/.
        TODO: Support not only 6h intervals for time-range variables such as precipitation.
        Rather, make this dependent on the steps of the forecasts.
        """
        input:
            namelist=rules.generate_mec_namelist.output.namelist,
            prepare_obs=rules.prepare_mec_input.output.obs_file,
            mod_dir=directory(rules.link_mec_input.output.mod),
            pull_ok=rules.sarus_pull_mec.output,
        output:
            fdbk_file=OUT_ROOT
            / "data/runs/{run_id}/fdbk_files/verSYNOP_{init_time}00.nc",
        log:
            OUT_ROOT / "logs/run_mec/{run_id}-{init_time}.log",
        shell:
            """
            (
                set -euo pipefail

                run_dir=$(dirname {input.namelist})
                abs_run_dir=$(realpath "$run_dir")
                abs_mod_root=$(realpath "$run_dir/../..") # two levels up (so that all links are mounted to the container)

                # run container
                sarus run \
                    --mount=type=bind,source=$abs_run_dir,destination=/src/bin2 \
                    --mount=type=bind,source=$abs_mod_root,destination=$abs_mod_root,readonly \
                    --mount=type=bind,source=/oprusers/osm/opr.inn/data/,destination=/oprusers/osm/opr.inn/data/ \
                    container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main

                # Run MEC using local executable (Alternative to sarus container)
                #cd "$run_dir"
                #export LM_HOST=balfrinew-ln002
                #source /oprusers/osm/opr.inn/abs/mec.env
                #cp /oprusers/osm/opr.inn/abs/mec .
                #./mec > ./mec_out.log 2>&1
                #cd -

                # copy the output file to the final location for the Feedback files
                # and rename to match NWP conventions
                mkdir -p "$run_dir/../../fdbk_files"
                cp "$run_dir/verSYNOP.nc" "$run_dir/../../fdbk_files/verSYNOP_{wildcards.init_time}00.nc"
                echo "...time at end of run_mec: $(date)"
            ) >{log} 2>&1
            """

    rule generate_ffv2_namelist:
        input:
            script="workflow/scripts/generate_ffv2_namelist.py",
            template="resources/ffv2/template_SYNOP_DET.nl.jinja2",
            # Block on MEC running for all input times, since FFV2 is across feedback files.
            mec_ok=lambda wc: expand(
                rules.run_mec.output.fdbk_file,
                run_id=wc.run_id,
                init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES_MEC],
            ),
        output:
            # Question: Definitely want to aggregate over init time, but will we have 1 run_ffv2 per run_id, or 1 run of ffv2 for all run_ids?
            namelist=OUT_ROOT / "data/runs/{run_id}/SYNOP_DET.nl",
        log:
            OUT_ROOT / "logs/generate_ffv2_namelist/{run_id}.log",
        localrule: True
        params:
            # TODO: We may want more than one directory here, if we are comparing models.
            feedback_directory=lambda wc: str(
                OUT_ROOT / f"data/runs/{wc.run_id}/fdbk_files"
            ),
            # Keeping this as a param. We will create it in run_ffv2 rule.
            output_directory=lambda wc: str(OUT_ROOT / f"data/runs/{wc.run_id}/scores"),
            # TODO: consider including run_ids here?
            experiment_ids=config["ffv2"]["experiment_ids"],
            veri_ens_member=config["ffv2"]["veri_ens_member"],
            catthresholds=json.dumps(config["ffv2"]["catthresholds"]),
            pecthresholds=json.dumps(config["ffv2"]["pecthresholds"]),
            experiment_description=config["ffv2"]["experiment_description"],
            file_description=config["ffv2"]["file_description"],
            domain_table=config["ffv2"]["domain_table"],
            blacklists=config["ffv2"]["blacklists"],
        shell:
            """
            (
                set -euo pipefail
                mkdir -p {params.output_directory}
                uv run {input.script} \
                    --template {input.template} \
                    --namelist {output.namelist} \
                    --experiment_ids {params.experiment_ids} \
                    --veri_ens_member {params.veri_ens_member} \
                    --catthresholds '{params.catthresholds}' \
                    --pecthresholds '{params.pecthresholds}' \
                    --feedback_directories {params.feedback_directory} \
                    --output_directory {params.output_directory} \
                    --experiment_description {params.experiment_description} \
                    --file_description {params.file_description} \
                    --domain_table {params.domain_table} \
                    --blacklists {params.blacklists}
            ) >{log} 2>&1
            """

    rule sarus_pull_ffv2:
        """Pull the FFV2 sarus container image once before the FFV2 job."""
        output:
            touch(OUT_ROOT / "logs/sarus_pull_ffv2.ok"),
        localrule: True
        shell:
            "sarus pull container-registry.meteoswiss.ch/ffv2ctr/ffv2-container:0.1.0-main"

    rule run_ffv2:
        input:
            namelist=rules.generate_ffv2_namelist.output.namelist,
            pull_ok=rules.sarus_pull_ffv2.output,
            # Direct dependency on MEC outputs so that Snakemake re-evaluates this rule
            # when dates change and new fdbk_files are needed (even if scores/ and shiny/
            # already exist from a prior run with different dates).
            mec_files=lambda wc: expand(
                rules.run_mec.output.fdbk_file,
                run_id=wc.run_id,
                init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES_MEC],
            ),
        output:
            scores=directory(OUT_ROOT / "data/runs/{run_id}/scores"),
        log:
            OUT_ROOT / "logs/run_ffv2/{run_id}.log",
        params:
            # domain_table and blacklists are locations on Balfrin, that will be
            # mounted into container (with the same filepaths)
            domain_table=rules.generate_ffv2_namelist.params.domain_table,
            blacklists=rules.generate_ffv2_namelist.params.blacklists,
            # QUESTION: Will we want to compare with other models?
            # Need to specify this in order to mount it.
            # Because namelist is a blocking input, and namelist generation
            # blocks on the MEC run, this should be OK to just use as param.
            feedback_directory=rules.generate_ffv2_namelist.params.feedback_directory,
        shell:
            """
            (
                set -euo pipefail
                echo "...time at start of run_ffv2: $(date)"

                # Create the output directory to hold scores, if it does not exist
                mkdir -p {output.scores}

                namelist=$(realpath {input.namelist})
                domain_table={params.domain_table}
                blacklists={params.blacklists}
                # Mount needs to have source as absolute path
                feedback_dir_abs=$(realpath {params.feedback_directory})
                output_dir_abs=$(realpath {output.scores})
                sarus run \
                    --mount=type=bind,source=$namelist,destination=/src/ffv2/SYNOP_DET.nl \
                    --mount=type=bind,source=$domain_table,destination=$domain_table \
                    --mount=type=bind,source=$blacklists,destination=$blacklists \
                    --mount=type=bind,source=$feedback_dir_abs,destination=/src/ffv2/input \
                    --mount=type=bind,source=$output_dir_abs,destination=/src/ffv2/output \
                    container-registry.meteoswiss.ch/ffv2ctr/ffv2-container:0.1.0-main

                echo "...time at end of run_ffv2: $(date)"
            ) >{log} 2>&1
            """

    rule reorganize_ffv2_files:
        input:
            scores=rules.run_ffv2.output.scores,
        output:
            shiny_dir=directory(OUT_ROOT / "data/runs/{run_id}/shiny/"),
        log:
            OUT_ROOT / "logs/reorganize_ffv2_files/{run_id}.log",
        localrule: True
        shell:
            """
            (
                set -euo pipefail
                echo "...time at start of reorganize_ffv2_files: $(date)"

                input_dir_abs=$(realpath {input.scores})
                output_dir_abs=$(realpath {output.shiny_dir})

                # move score files into app-specific subdirectories, for the Shiny app
                # display.
                mkdir -p $output_dir_abs/fdbk_cont/data
                mkdir -p $output_dir_abs/fdbk_cont_bystat/data
                mkdir -p $output_dir_abs/fdbk_cont_ts/data
                mkdir -p $output_dir_abs/fdbk_synop_categ/data
                mkdir -p $output_dir_abs/fdbk_synop_categ_bystat/data
                mkdir -p $output_dir_abs/fdbk_synop_categ_ts/data

                # DET surface continuous scores
                cp $input_dir_abs/CONT_exp* $output_dir_abs/fdbk_cont/data/
                # DET surface continuous scores as time series
                cp $input_dir_abs/CONT_TS_exp* $output_dir_abs/fdbk_cont_ts/data/
                # DET surface continuous by stations
                cp $input_dir_abs/CONT_bs_exp* $output_dir_abs/fdbk_cont_bystat/data/

                # Categorical verification against SYNOP
                cp $input_dir_abs/CATEG_exp* $output_dir_abs/fdbk_synop_categ/data
                cp $input_dir_abs/PEC_exp* $output_dir_abs/fdbk_synop_categ/data

                # Categorical verification against SYNOP by station
                # This is not presently generated, so skip.
                #cp $input_dir_abs/CATEG_TS_exp* $output_dir_abs/fdbk_synop_categ_ts/data

                # Categorical verification against SYNOP as time series
                cp $input_dir_abs/CATEG_bs_exp* $output_dir_abs/fdbk_synop_categ_bystat/data

                echo "...time at end of reorganize_ffv2_files: $(date)"
            ) >{log} 2>&1
            """
