# ----------------------------------------------------- #
# CREATE A SUMMARY                                      #
# ----------------------------------------------------- #


include: "common.smk"


from pprint import pprint


# Produce a human-readable summary of the configuration.
rule write_summary:
    output:
        OUT_ROOT / "data/runs/{run_id}/summary.md",
    localrule: True
    params:
        configfile=lambda wc: workflow.configfiles[0],
    run:
        import yaml
        from datetime import datetime

        cfg_path = params.configfile
        cfg = yaml.safe_load(open(cfg_path))
        with open(output[0], "w") as out:
            out.write(f"# Run Summary\n\n")
            out.write(f"- 🆔 **Run ID:** {wildcards.run_id}\n")
            out.write(f"- 📄 **Configuration file:** `{cfg_path}`\n")
            out.write(f"- 🕒 **Generated on:** {datetime.now().isoformat()}\n\n")
            # config block
            out.write("## ⚙️ Configuration\n")
            out.write("```yaml\n")
            yaml.dump(cfg["runs"], out, default_flow_style=False, sort_keys=False)
            out.write("```\n")
