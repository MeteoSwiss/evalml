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
    return f"Verification against {truth} with initializations from {dates.get('start')} to {dates.get('end')} by {dates.get('frequency')}"


def _candidate_gpu(run_cfg: dict) -> int:
    """Return the GPU count for a run, defaulting to 1."""
    ir = run_cfg.get("inference_resources")
    if ir is None:
        return 1
    if isinstance(ir, dict):
        return int(ir.get("gpu", 1) or 1)
    return int(getattr(ir, "gpu", 1) or 1)


rule collect_system_metrics:
    localrule: True
    input:
        okfiles=[
            OUT_ROOT / f"logs/inference_execute/{run_id}-{t.strftime('%Y%m%d%H%M')}.ok"
            for run_id, run_cfg in RUN_CONFIGS.items()
            if run_cfg.get("_is_candidate", False)
            for t in REFTIMES
        ],
    output:
        OUT_ROOT / "results/{experiment}/system_metrics.json",
    params:
        log_files=[
            str(
                OUT_ROOT
                / f"logs/inference_execute/{run_id}-{t.strftime('%Y%m%d%H%M')}.log"
            )
            for run_id, run_cfg in RUN_CONFIGS.items()
            if run_cfg.get("_is_candidate", False)
            for t in REFTIMES
        ],
        label_map={
            run_id: run_cfg.get("label", run_id)
            for run_id, run_cfg in RUN_CONFIGS.items()
            if run_cfg.get("_is_candidate", False)
        },
        gpu_map={
            run_id: _candidate_gpu(run_cfg)
            for run_id, run_cfg in RUN_CONFIGS.items()
            if run_cfg.get("_is_candidate", False)
        },
        log_dir=str(OUT_ROOT / "logs/inference_execute"),
    run:
        import json
        from pathlib import Path

        from diagnostics import parse_logs

        records = parse_logs(
            log_files=params.log_files,
            label_map=params.label_map,
            gpu_map=params.gpu_map,
            log_dir=params.log_dir,
        )
        out_path = Path(str(output[0]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2))


rule report_experiment_dashboard:
    localrule: True
    input:
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
    params:
        sources=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
        header_text=make_header_text(),
    log:
        OUT_ROOT / "logs/report_experiment_dashboard/{experiment}.log",
    shell:
        """
        python {input.script} \
            --verif_files {input.verif} \
            --template {input.template} \
            --script {input.js_script} \
            --header_text "{params.header_text}" \
            --configfile "{input.configfile}" \
            --sysmetrics_file "{input.sysmetrics}" \
            --output {output} > {log} 2>&1
        """
