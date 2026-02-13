import argparse
from pathlib import Path
from urllib.parse import urlparse
from anemoi.utils.mlflow.auth import TokenAuth
from anemoi.utils.mlflow.client import AnemoiMlflowClient
from mlflow.tracking import MlflowClient

KNOWN_MLFLOW_TRACKING_URI = [
    "mlflow.ecmwf.int",
    "service.meteoswiss.ch",
    "servicedepl.meteoswiss.ch",
]


def main(args):
    run_uri = args.run_uri
    parsed_url = urlparse(run_uri)
    if parsed_url.netloc in KNOWN_MLFLOW_TRACKING_URI:
        uri, fragment = run_uri.split("#")
        run_id = fragment.split("/")[-1]
        if parsed_url.netloc == "mlflow.ecmwf.int":
            TokenAuth(uri).login()
            client = AnemoiMlflowClient(uri, authentication=True)
        else:
            client = MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)
        path = run.data.params.get("config.hardware.paths.checkpoints")
        path = path or run.data.params.get("config.system.output.checkpoints.root")
        path = Path(path) / "inference-last.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        print(str(path))
        return
    else:
        raise ValueError(f"Unsupported MLFlow tracking URI: {parsed_url.netloc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get local checkpoint location.")
    parser.add_argument("run_uri", type=str, help="MLFlow run URI")
    args = parser.parse_args()
    main(args)
