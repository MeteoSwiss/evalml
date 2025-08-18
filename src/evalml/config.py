from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, RootModel, HttpUrl

PROJECT_ROOT = Path(__file__).parents[2]


class Dates(BaseModel):
    """Start/stop of the hindcast period and the launch frequency."""

    start: str = Field(
        ...,
        description="First forecast initialisation as an ISO-8601 formatted string.",
    )
    end: str = Field(
        ...,
        description="Last forecast initialisation as an ISO-8601 formatted string.",
    )
    frequency: str = Field(
        ...,
        description="Time between initialisations. Must be a combination of a number and a time unit (h or d).",
        pattern=r"^\d+[hd]$",
    )


class ExplicitDates(RootModel[List[str]]):
    """Explicit list of initialisation dates as ISO-8601 formatted strings."""


class AnemoiInferenceConfig(RootModel[Dict[str, Any]]):
    """Configuration for the Anemoi inference workflow."""


class RunConfig(BaseModel):
    run_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="The mlflow run ID, as a 32-character hexadecimal string.",
    )
    label: str | None = Field(
        None,
        description="The label for the run that will be used in experiment results such as reports and figures.",
    )
    extra_dependencies: List[str] = Field(
        default_factory=list,
        description="List of extra dependencies to install for this model. "
        "These will be added to the pyproject.toml file in the run directory.",
    )

    config: Dict[str, Any] | str


class ForecasterConfig(RunConfig):
    """Single training run stored in MLflow."""

    config: Dict[str, Any] | str = Field(
        default_factory=lambda _: str(
            PROJECT_ROOT / "resources" / "inference" / "configs" / "forecaster.yaml"
        ),
        description="Configuration for the forecaster run. Can be a dictionary of parameters or a path to a configuration file."
        "By default, it will point to resources/inference/configs/forecaster.yaml in the evalml repository.",
    )


class InterpolatorConfig(RunConfig):
    """Single training run stored in MLflow."""

    config: Dict[str, Any] | str = Field(
        default_factory=lambda _: str(
            PROJECT_ROOT / "resources" / "inference" / "configs" / "interpolator.yaml"
        ),
        description="Configuration for the interpolator run. Can be a dictionary of parameters or a path to a configuration file. "
        "By default, it will point to resources/inference/configs/interpolator.yaml in the evalml repository.",
    )

    forecaster: ForecasterConfig | None = Field(
        None,
        description="Configuration for the forecaster run that this interpolator is based on.",
    )


class ForecasterItem(BaseModel):
    forecaster: ForecasterConfig


class InterpolatorItem(BaseModel):
    interpolator: InterpolatorConfig


class VerifConfig(BaseModel):
    """Configuration for the verification of the experiment."""

    valid_every: Optional[int] = Field(
        ge=1,
        description="Hours between verification times starting from 00:00 UTC. If None, no filtering is applied.",
    )


class Execution(BaseModel):
    """Configuration for the execution of the experiment."""

    run_group_size: int = Field(
        ..., ge=1, description="Number of runs to execute in the same SLURM job."
    )


class Locations(BaseModel):
    """Locations of data and services used in the workflow."""

    output_root: Path = Field(..., description="Root directory for all output files.")
    mlflow_uri: List[HttpUrl] = Field(
        ...,
        description="MLflow tracking URI(s) for the experiment. Can be a list of URIs if using multiple tracking servers.",
    )


class DefaultResources(BaseModel):
    """Default resource settings for job execution."""

    slurm_partition: str = Field(..., description="SLURM partition to use.")
    cpus_per_task: int = Field(..., ge=1, description="Number of CPUs per task.")
    mem_mb_per_cpu: int = Field(..., ge=1, description="Memory per CPU in MB.")
    runtime: str = Field(..., description="Maximum runtime, e.g. '1h'.")

    def parsable(self) -> str:
        """Convert the default resources to a string of key=value pairs."""
        return [f"{key}={value}" for key, value in self.model_dump().items()]


class Profile(BaseModel):
    """Workflow execution profile."""

    executor: str = Field(..., description="Job executor, e.g. 'slurm'.")
    default_resources: DefaultResources
    jobs: int = Field(..., ge=1, description="Maximum number of parallel jobs.")

    def parsable(self) -> Dict[str, str]:
        """Convert the profile to a dictionary of command-line arguments."""
        out = []
        out += ["--executor", self.executor]
        out += ["--default-resources"] + self.default_resources.parsable()
        out += ["--jobs", str(self.jobs)]
        return out


class ConfigModel(BaseModel):
    """Top-level configuration."""

    description: str = Field(
        ...,
        description="Description of the experiment, e.g. 'Hindcast of the 2023 season.'",
    )
    dates: Dates | ExplicitDates
    lead_time: str = Field(
        ..., description="Forecast length, e.g. '120h'", pattern=r"^\d+[hmd]$"
    )
    runs: List[ForecasterItem | InterpolatorItem] = Field(
        ...,
        description="Dictionary of runs to execute, with run IDs as keys and configurations as values.",
    )
    baseline: str = Field(
        ..., description="The label of the NWP baseline run to compare against."
    )
    verification: Optional[VerifConfig] = None
    execution: Execution
    locations: Locations
    profile: Profile

    model_config = {
        "extra": "forbid",  # fail on misspelled keys
        "populate_by_name": True,
    }


def generate_config_schema() -> str:
    """Generate the JSON schema for the ConfigModel."""
    return ConfigModel.model_json_schema()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate the JSON schema for the evalml configuration"
    )
    parser.add_argument(
        "output", type=str, help="Path to save the generated JSON schema."
    )
    args = parser.parse_args()

    with open(args.output, "w") as f:
        json.dump(generate_config_schema(), f, indent=2)
        f.write("\n")
