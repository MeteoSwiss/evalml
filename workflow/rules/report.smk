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


rule report_experiment_dashboard:
    localrule: True
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
    params:
        sources=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
        header_text=make_header_text(),
        stratification=" ".join(config["dashboard"]["stratification"]),
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
            --stratification {params.stratification} \
            --output {output} > {log} 2>&1
        """


rule report_scorecard:
    localrule: True
    wildcard_constraints:
        run_id="[^/]+/[^/]+",  # env_id/r_hash — exactly one slash
        baseline="[^/]+",
    input:
        script="workflow/scripts/report_scorecard.mo.py",
        verif_run=lambda wc: EXPERIMENT_PARTICIPANTS[wc.run_id],
        verif_baseline=lambda wc: EXPERIMENT_PARTICIPANTS[wc.baseline],
    output:
        report(
            OUT_ROOT
            / "results/{experiment}/scorecard_plots/{run_id}/scorecard_{baseline}.png",
        ),
    params:
        lead_times="6/33/6",
        regions=[
            "all",
            "mittelland",
            "voralpen",
            "alpennordhang",
            "innerealpentaeler",
            "alpensuedseite",
            "jura",
        ],
        variables=[
            "U_10M:RMSE,MAE,STDE,CORR,R2",
            "V_10M:RMSE,MAE,STDE,CORR,R2",
            "T_2M:RMSE,MAE,STDE,CORR,R2",
            "PMSL:RMSE,MAE,STDE,CORR,R2",
            "TD_2M:RMSE,MAE,STDE,CORR,R2",
            "TOT_PREC:RMSE,MAE,STDE,CORR,R2",
        ],
        run_source=lambda wc: RUN_CONFIGS[wc.run_id].get("label", wc.run_id),
        baseline_source=lambda wc: BASELINE_CONFIGS[wc.baseline].get(
            "label", wc.baseline
        ),
    log:
        OUT_ROOT
        / "logs/report_scorecard/{experiment}/{run_id}/scorecard_{baseline}.log",
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
            --regions {params.regions:q} \
            "${{VAR_ARGS[@]}}" \
            --output {output:q} > {log} 2>&1
        """
