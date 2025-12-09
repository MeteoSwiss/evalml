"""Update anemoi model dependencies based on MLflow run information.

Short summary
-------------
Update anemoi model dependencies based on MLflow run information.

Description
-----------
This module provides functionality to:
1. Fetch model version information from MLflow
2. Resolve git dependencies for anemoi components
3. Update pyproject.toml with correct dependency versions

The script maintains a backup of configuration files and restores them in case of failure.
"""

import logging
import shutil
import subprocess
from pathlib import Path
import argparse

import toml
from anemoi.utils.mlflow.auth import TokenAuth
from anemoi.utils.mlflow.client import AnemoiMlflowClient
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_mlflow_client_given_runid(
    mlflow_uri: str | list[str], run_id: str
) -> MlflowClient:
    """
    Get an MLflow client for a given run ID.

    Parameters
    ----------
    mlflow_uri : str | list[str]
        One or more MLflow tracking URIs to search.
    run_id : str
        The MLflow run ID to look up.

    Returns
    -------
    MlflowClient
        A client configured for the server where the run was found.

    Raises
    ------
    ValueError
        If the run ID is not found in any of the provided URIs.
    """
    uris = [mlflow_uri] if isinstance(mlflow_uri, str) else mlflow_uri
    for uri in uris:
        if "ecmwf.int" in uri:
            TokenAuth(url=uri).login()
            client = AnemoiMlflowClient(uri, authentication=True)
        else:
            client = MlflowClient(tracking_uri=uri)
        try:
            client.get_run(run_id)
            return client
        except RestException:
            continue
    raise ValueError(
        f"Run ID {run_id} not found in any of the provided MLflow URIs: {uris}"
    )


def get_python_version(client: MlflowClient, run_id: str) -> str | None:
    """
    Extract the Python version used in the MLflow run.

    Parameters
    ----------
    client : MlflowClient
        MLflow client instance.
    run_id : str
        ID of the MLflow run.

    Returns
    -------
    str | None
        Python version string, e.g. "3.10.14".

    Raises
    ------
    ValueError
        If no valid python version is found in the MLflow run.
    """
    run = client.get_run(run_id)
    python_version = run.data.params.get("metadata.provenance_training.python")

    if not python_version:
        raise ValueError("No valid python version found in MLflow run")

    return python_version


def get_path_to_checkpoint(client: MlflowClient, run_id: str) -> str:
    """
    Get the path to the checkpoints directory from an MLflow run.

    Parameters
    ----------
    client : MlflowClient
        MLflow client instance.
    run_id : str
        ID of the MLflow run.

    Returns
    -------
    str
        Path to the checkpoints directory, e.g. "/scratch/mch/user/output/checkpoint".

    Raises
    ------
    ValueError
        If no valid checkpoints path is found in the MLflow run.
    """
    run = client.get_run(run_id)
    path = run.data.params.get("config.hardware.paths.checkpoints")

    if not path:
        raise ValueError("No valid checkpoints path found in MLflow run")

    return str(Path(path) / "inference-last.ckpt")


def version_to_pep440_range(version: str) -> str:
    """
    Convert a Python version into a PEP 440 compatible range.

    Parameters
    ----------
    version : str
        A Python version string like '3.10.6'.

    Returns
    -------
    str
        A PEP 440 range string like '>=3.10,<3.11'.

    Raises
    ------
    ValueError
        If the version string does not contain at least major and minor parts.
    """
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError("Version must have at least major and minor parts")

    major, minor = int(parts[0]), int(parts[1])
    return f">={major}.{minor},<{major}.{minor + 1}"


def update_pyproject_toml(
    toml_path: Path,
    python_version: str,
    checkpoint_path: str,
    run_mlflow_link: str,
) -> None:
    """
    Update pyproject.toml [project.dependencies] with versions or VCS references.

    Parameters
    ----------
    toml_path : Path
        Path to pyproject.toml.
    python_version : str
        Python version string, e.g. "3.10.14".
    checkpoint_path : str
        Path to the checkpoints directory.
    run_mlflow_link : str
        Link to the MLflow run for metadata in the pyproject.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the provided toml_path does not exist.
    RuntimeError
        If the file cannot be read or written.
    ValueError
        If [project.dependencies] in the pyproject is not a list.
    """
    if not toml_path.exists():
        raise FileNotFoundError(f"{toml_path} not found")

    try:
        with open(toml_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
    except Exception as e:
        raise RuntimeError("Failed to read pyproject.toml") from e

    deps = config.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError("[project.dependencies] must be a list")

    config["project"]["requires-python"] = version_to_pep440_range(python_version)
    config.setdefault("tool", {})["anemoi"] = {"checkpoint_path": checkpoint_path}
    config["tool"]["anemoi"]["run_mlflow_link"] = run_mlflow_link

    try:
        with open(toml_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
    except Exception as e:
        raise RuntimeError("Failed to write pyproject.toml") from e


class ScriptConfig(argparse.Namespace):
    """Configuration for the script to update pyproject.toml."""

    mlflow_uri: str | list[str] = None
    mlflow_id: str = None
    pyproject_template: Path = None
    pyproject_out: Path = None
    extra_dependencies: list[str] = []


def program_summary_log(args: ScriptConfig) -> None:
    """Log a welcome message with the script information."""
    logger.info("=" * 80)
    logger.info("Updating pyproject.toml with MLflow model dependencies")
    logger.info("=" * 80)
    logger.info("MLflow URIs: %s", args.mlflow_uri)
    logger.info("MLflow run ID: %s", args.mlflow_id)
    logger.info("Pyproject template: %s", args.pyproject_template)
    logger.info("Pyproject output: %s", args.pyproject_out)
    logger.info("Extra dependencies: %s", args.extra_dependencies)
    logger.info("=" * 80)


def main(args) -> None:
    program_summary_log(args)

    mlflow_uri = args.mlflow_uri
    mlflow_id = args.mlflow_id
    pyproject_template = args.pyproject_template
    pyproject_out = args.pyproject_out

    # copy template to output location
    shutil.copy2(pyproject_template, pyproject_out)

    # extract information from MLFlow and update pyproject.toml
    client = get_mlflow_client_given_runid(mlflow_uri, mlflow_id)
    python_version = get_python_version(client, mlflow_id)
    checkpoint_path = get_path_to_checkpoint(client, mlflow_id)
    run_mlflow_link = client.tracking_uri + "/#/runs/" + mlflow_id
    update_pyproject_toml(
        pyproject_out, python_version, checkpoint_path, run_mlflow_link
    )

    # write minimal requirement.txt for the model
    requirements_path = str(pyproject_out.parent / "requirements.txt")
    command = ["anemoi-inference"]
    command += ["inspect", "--requirements", str(checkpoint_path)]
    command += ["|", "grep", "-E", "'anemoi|torch'"]
    command += [">", requirements_path]
    subprocess.run(" ".join(command), check=True, shell=True)
    logger.info("Writing temporary requirements to %s", requirements_path)

    # add dependencies from requirements.txt to pyproject.toml
    command = ["uv"]
    command += ["--project", str(pyproject_out.parent.resolve())]
    command += ["add", "--no-sync", "--verbose"]
    command += ["--requirements", requirements_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(
            "Failed to add dependencies from requirements.txt: %s\n%s",
            e,
            e.stderr or e.stdout,
        )
        raise RuntimeError("Failed to add dependencies from requirements.txt") from e
    logger.info("Added dependencies from requirements.txt")

    # add any extra dependencies specified by the user
    if args.extra_dependencies:
        command = ["uv"]
        command += ["--project", str(pyproject_out.parent.resolve())]
        command += ["add", "--no-sync", "--verbose"]
        command += args.extra_dependencies
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to add extra dependencies: %s\n%s",
                e,
                e.stderr or e.stdout,
            )
            raise RuntimeError("Failed to add extra dependencies") from e
        logger.info("Added extra dependencies: %s", args.extra_dependencies)

    logger.info("Successfully updated dependencies in %s", pyproject_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update pyproject.toml with MLflow model dependencies."
    )

    parser.add_argument(
        "--mlflow_uri",
        type=lambda x: x.split(","),
        required=True,
        help="A comma-separated list of MLflow tracking URIs to search.",
    )
    parser.add_argument(
        "--mlflow_id",
        type=str,
        required=True,
        help="MLflow run ID to extract model information from.",
    )
    parser.add_argument(
        "--pyproject_template",
        type=Path,
        required=True,
        help="Path to the template pyproject.toml file.",
    )
    parser.add_argument(
        "--pyproject_out",
        type=Path,
        required=True,
        help="Path to output the updated pyproject.toml file.",
    )
    parser.add_argument(
        "--extra_dependencies",
        type=lambda x: x.split(","),
        default=[],
        help="Additional dependencies to add to pyproject.toml, specified as a comma-separated list.",
    )

    args = parser.parse_args()

    main(args)
