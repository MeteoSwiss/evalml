# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime


include: "common.smk"


def make_header_text():
    dates = config["dates"]
    truth = config["truth"]["label"]
    if isinstance(dates, list):
        return f"Explicit initializations from {len(dates)} runs have been used."
    text = f"Verification against {truth} with initializations from {dates.get('start')} to {dates.get('end')} by {dates.get('frequency')}"
    blacklist = dates.get("blacklist", [])
    if blacklist:
        text += f" (excluding {len(blacklist)} blacklisted date{'s' if len(blacklist)!=1 else ''})"
    return text


def _candidate_gpu(run_cfg: dict) -> int:
    """Return the GPU count for a run, defaulting to 1."""
    ir = run_cfg.get("inference_resources")
    if ir is None:
        return 1
    if isinstance(ir, dict):
        return int(ir.get("gpu", 1) or 1)
    return int(getattr(ir, "gpu", 1) or 1)


rule collect_system_metrics:
    input:
        okfiles=[
            OUT_ROOT / f"logs/inference_execute/{run_id}-{t.strftime('%Y%m%d%H%M')}.ok"
            for run_id in RUN_CONFIGS
            for t in REFTIMES
        ],
    output:
        OUT_ROOT / "results/{experiment}/system_metrics.json",
    localrule: True
    params:
        run_info=[
            {
                "workdir": str(
                    (
                        OUT_ROOT / f"data/runs/{run_id}/{t.strftime('%Y%m%d%H%M')}"
                    ).resolve()
                ),
                "run_id": run_id,
                "init_time": t.strftime("%Y%m%d%H%M"),
            }
            for run_id in RUN_CONFIGS
            for t in REFTIMES
        ],
        label_map={
            run_id: run_cfg.get("label", run_id)
            for run_id, run_cfg in RUN_CONFIGS.items()
        },
        gpu_map={
            run_id: _candidate_gpu(run_cfg) for run_id, run_cfg in RUN_CONFIGS.items()
        },
    run:
        import json
        from pathlib import Path
        from diagnostics import parse_logs

        records = parse_logs(
            run_info=params.run_info,
            label_map=params.label_map,
            gpu_map=params.gpu_map,
        )
        out_path = Path(str(output[0]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2))


rule report_experiment_dashboard:
    input:
        "src/verification/__init__.py",
        script="workflow/scripts/report_experiment_dashboard.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
        template="resources/report/dashboard/template.html.jinja2",
        js_script="resources/report/dashboard/script.js",
        configfile={workflow.configfiles[0]},
        sysmetrics=OUT_ROOT / "results/{experiment}/system_metrics.json",
    output:
        report(
            directory(OUT_ROOT / "results/{experiment}/dashboard"),
            htmlindex="dashboard.html",
        ),
    log:
        OUT_ROOT / "logs/report_experiment_dashboard/{experiment}.log",
    localrule: True
    params:
        sources=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
        header_text=make_header_text(),
        stratification=" ".join(config["experiment"]["dashboard"]["stratification"]),
    shell:
        """
        python {input.script} \
            --verif_files {input.verif} \
            --template {input.template} \
            --script {input.js_script} \
            --header_text "{params.header_text}" \
            --configfile "{input.configfile}" \
            --stratification {params.stratification} \
            --sysmetrics_file "{input.sysmetrics}" \
            --output {output} >{log} 2>&1
        """


rule report_scorecard:
    input:
        script="workflow/scripts/report_scorecard.py",
        verif_run=lambda wc: EXPERIMENT_PARTICIPANTS[f"{wc.env_id}/{wc.config_hash}"],
        verif_baseline=lambda wc: EXPERIMENT_PARTICIPANTS[
            SCORECARD_CONFIGS[wc.scorecard_name]["baseline"]
        ],
    output:
        report(
            OUT_ROOT
            / "results/{experiment}/scorecards/{scorecard_name}/scorecard_{scorecard_name}_{env_id}_{config_hash}.png",
        ),
    log:
        OUT_ROOT
        / "logs/report_scorecard/{experiment}/{scorecard_name}/{env_id}/{config_hash}.log",
    wildcard_constraints:
        env_id="[^/]+",  # no slashes
        config_hash="[^/]+",
        scorecard_name="[^/]+",
    localrule: True
    params:
        lead_times=lambda wc: SCORECARD_CONFIGS[wc.scorecard_name]["lead_times"],
        stratification=lambda wc: SCORECARD_CONFIGS[wc.scorecard_name]["stratification"],
        variables=lambda wc: SCORECARD_CONFIGS[wc.scorecard_name]["variables"],
        run_source=lambda wc: RUN_CONFIGS[f"{wc.env_id}/{wc.config_hash}"].get(
            "label", f"{wc.env_id}/{wc.config_hash}"
        ),
        baseline_source=lambda wc: BASELINE_CONFIGS[
            SCORECARD_CONFIGS[wc.scorecard_name]["baseline"]
        ].get("label", SCORECARD_CONFIGS[wc.scorecard_name]["baseline"]),
    shell:
        """
        VAR_ARGS=()
        for v in {params.variables:q}; do
            VAR_ARGS+=(--variable "$v")
        done

        python {input.script} \
            --verif_run {input.verif_run:q} \
            --verif_baseline {input.verif_baseline:q} \
            --run_source {params.run_source:q} \
            --baseline_source {params.baseline_source:q} \
            --lead_times {params.lead_times:q} \
            --stratification {params.stratification:q} \
            "${{VAR_ARGS[@]}}" \
            --output {output:q} >{log} 2>&1
        """
