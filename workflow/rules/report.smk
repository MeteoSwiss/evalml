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


rule report_experiment_dashboard:
    input:
        "src/verification/__init__.py",
        script="workflow/scripts/report_experiment_dashboard.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
        template="resources/report/dashboard/template.html.jinja2",
        js_script="resources/report/dashboard/script.js",
        configfile={workflow.configfiles[0]},
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
        label_map=",".join(
            "{}:{}".format(
                sid,
                (
                    BASELINE_CONFIGS[sid].get("label", sid)
                    if sid in BASELINE_CONFIGS
                    else RUN_CONFIGS[sid].get("label", sid)
                ),
            )
            for sid in EXPERIMENT_PARTICIPANTS
        )
        + ",truth_{}:{}".format(TRUTH_HASH, config["truth"]["label"]),
    shell:
        """
        python {input.script} \
            --verif_files {input.verif} \
            --template {input.template} \
            --script {input.js_script} \
            --header_text "{params.header_text}" \
            --configfile "{input.configfile}" \
            --stratification {params.stratification} \
            --labels "{params.label_map}" \
            --output {output} >{log} 2>&1
        """


rule report_scorecard:
    input:
        script="workflow/scripts/report_scorecard.py",
        verif_run=lambda wc: EXPERIMENT_PARTICIPANTS[f"{wc.env_id}/{wc.config_hash}"],
        verif_baseline=lambda wc: EXPERIMENT_PARTICIPANTS[
            resolve_baseline_id(SCORECARD_CONFIGS[wc.scorecard_name]["baseline"])
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
        run_source=lambda wc: f"{wc.env_id}/{wc.config_hash}",
        run_label=lambda wc: RUN_CONFIGS[f"{wc.env_id}/{wc.config_hash}"].get(
            "label", f"{wc.env_id}/{wc.config_hash}"
        ),
        baseline_source=lambda wc: resolve_baseline_id(
            SCORECARD_CONFIGS[wc.scorecard_name]["baseline"]
        ),
        baseline_label=lambda wc: SCORECARD_CONFIGS[wc.scorecard_name]["baseline"],
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
            --run_label {params.run_label:q} \
            --baseline_source {params.baseline_source:q} \
            --baseline_label {params.baseline_label:q} \
            --lead_times {params.lead_times:q} \
            --stratification {params.stratification:q} \
            "${{VAR_ARGS[@]}}" \
            --output {output:q} >{log} 2>&1
        """
