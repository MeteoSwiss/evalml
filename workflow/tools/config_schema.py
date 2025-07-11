from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, HttpUrl


class Dates(BaseModel):
    """Start/stop of the hindcast period and the launch frequency."""
    start: str = Field(..., description="First forecast initialisation as an ISO-8601 formatted string.",)
    end: str = Field(..., description="Last forecast initialisation as an ISO-8601 formatted string.",)
    frequency: str = Field(..., description="Time between initialisations. Must be a combination of a number and a time unit (h or d).",
                           pattern=r"^\d+[hd]$")


class RunConfig(BaseModel):
    """Single training run stored in MLflow."""
    run_id: str = Field(..., min_length=32, max_length=32,
                        description="The mlflow run ID, as a 32-character hexadecimal string.")
    label: str = Field(..., description="The label for the run that will be used in experiment results such as reports and figures.")


class Execution(BaseModel):
    """Configuration for the execution of the experiment."""
    run_group_size: int = Field(..., ge=1, description="Number of runs to execute in the same SLURM job.")


class Locations(BaseModel):
    """Locations of data and services used in the workflow."""
    output_root: Path = Field(..., description="Root directory for all output files.")
    mlflow_uri: List[HttpUrl] = Field(
        ..., description="MLflow tracking URI(s) for the experiment. Can be a list of URIs if using multiple tracking servers."
    )


class ExperimentConfig(BaseModel):
    """Top-level configuration."""
    description: str = Field(..., description="Description of the experiment, e.g. 'Hindcast of the 2023 season.'")
    dates: Dates
    lead_time: str = Field(..., description="Forecast length, e.g. '120h'", pattern=r"^\d+[hmd]$")
    runs: Dict[str, RunConfig]
    baseline: str = Field(..., description="The label of the NWP baseline run to compare against.")
    execution: Execution
    locations: Locations

    model_config = {
        "extra": "forbid",  # fail on misspelled keys
        "populate_by_name": True,
    }


def generate_config_schema() -> str:
    """Generate the JSON schema for the ExperimentConfig model."""
    return ExperimentConfig.model_json_schema()


if __name__ == "__main__":
    import json
    
    with open(Path(__file__).parent / "config.schema.json", "w") as f:
        json.dump(generate_config_schema(), f, indent=2)