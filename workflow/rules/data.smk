from pathlib import Path


include: "common.smk"


# Write experiment.station_holdout to a file, the single handoff of the holdout
# station set to verification and to inference runs that consume it (e.g. nudging).
rule station_holdout:
    output:
        HOLDOUT_FILE,
    localrule: True
    params:
        stations=HOLDOUT_STATIONS,
    run:
        import pandas as pd

        pd.DataFrame({"nat_abbr": params.stations}).to_csv(output[0], index=False)
