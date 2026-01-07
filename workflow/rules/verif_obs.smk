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
        obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
    params:
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        init_list_str=lambda wc: " ".join(get_init_times(wc)),
        run_root=lambda wc: str(OUT_ROOT / f"data/runs/{wc.run_id}"),
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/{run_id}-{init_time}_collect_mec_input.log",
    shell:
        """
        (
        set -euo pipefail
        echo "...time at start of collect_mec_input: $(date)"

        # create the input_obs and input_mod dirs
        mkdir -p {output.obs} {output.mod}

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
        run_dir=directory(rules.collect_mec_input.output.mod),
    output:
        fdbk_file=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/verSYNOP.nc",
    resources:
        cpus_per_task=1,
        runtime="1h",
    log:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/{run_id}-{init_time}_run_mec.log",
    shell:
        #TODO(mmcglohon): Replace podman with sarus if needed.
        """
        (
        set -euo pipefail
        echo "...time at start of run_mec: $(date)"
        # Note: pull command currently redundant; may not be the case with sarus.
        #podman pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        #srun --pty -N1 -c 1 -p postproc -t 2:00:00 podman run --mount=type=bind,source={{input.run_dir}},destination=/src/bin2 --mount=type=bind,source=/oprusers/osm/opr.emme/data/,destination=/oprusers/osm/opr.emme/data/ container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main

        # change to the MEC run directory, set env and run MEC
        cd {input.run_dir}/..
        export LM_HOST=balfrin-ln002
        source /oprusers/osm/opr.emme/abs/mec.env
        ./mec > ./mec_out.log 2>&1

        # move the output file to the expected location
        mkdir -p ../../fdbk_files
        cp verSYNOP.nc ../../fdbk_files/verSYNOP_{wildcards.init_time}.nc
        echo "...time at end of run_mec: $(date)"
        ) > {log} 2>&1
        """
