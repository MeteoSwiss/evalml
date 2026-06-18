import argparse
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

from anemoi.utils.mlflow.auth import TokenAuth
from anemoi.utils.mlflow.client import AnemoiMlflowClient
from mlflow.tracking import MlflowClient

LOG = logging.getLogger(__name__)

KNOWN_MLFLOW_TRACKING_URI = [
    "mlflow.ecmwf.int",
    "service.meteoswiss.ch",
    "servicedepl.meteoswiss.ch",
]

CHECKPOINT_FILENAME = "inference-last.ckpt"


def _find_artifact_path(client, run_id, filename, path=""):
    """Recursively search a run's artifacts for a file by name, returning its artifact path."""
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            result = _find_artifact_path(client, run_id, filename, artifact.path)
            if result is not None:
                return result
        elif Path(artifact.path).name == filename:
            return artifact.path
    return None


def main(args):
    run_uri = args.run_uri
    parsed_url = urlparse(run_uri)
    if parsed_url.netloc in KNOWN_MLFLOW_TRACKING_URI:
        uri, fragment = run_uri.split("#")
        if parsed_url.netloc == "mlflow.ecmwf.int":
            TokenAuth(uri).login()
            client = AnemoiMlflowClient(uri, authentication=True)
        else:
            client = MlflowClient(tracking_uri=uri)
        if "/models/" in fragment:
            parts = fragment.strip("/").split("/")
            model_name = parts[1]
            if len(parts) >= 4 and parts[2] == "versions":
                model_version = client.get_model_version(model_name, parts[3])
            else:
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    raise ValueError(
                        f"No versions found for model '{model_name}' in the registry"
                    )
                model_version = max(versions, key=lambda v: int(v.version))
            LOG.info(
                "Found model version: %s (run ID: %s)",
                model_version.version,
                model_version.run_id,
            )
            output_path = Path(args.output)
            artifact_path = _find_artifact_path(
                client, model_version.run_id, CHECKPOINT_FILENAME
            )
            if artifact_path is None:
                raise FileNotFoundError(
                    f"Could not find '{CHECKPOINT_FILENAME}' in MLflow artifacts for run {model_version.run_id}"
                )
            local_path = Path(
                client.download_artifacts(
                    model_version.run_id, artifact_path, str(output_path.parent)
                )
            )
            if local_path != output_path:
                shutil.move(str(local_path), str(output_path))
            return
        else:
            run_id = fragment.split("/")[-1]
        run = client.get_run(run_id)
        path = run.data.params.get("config.hardware.paths.checkpoints")
        path = path or run.data.params.get("config.system.output.checkpoints.root")
        path = Path(path) / CHECKPOINT_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        print(str(path))
        return
    else:
        raise ValueError(f"Unsupported MLFlow tracking URI: {parsed_url.netloc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get local checkpoint location.")
    parser.add_argument("run_uri", type=str, help="MLFlow run URI")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for the downloaded checkpoint (required for model registry URLs).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
