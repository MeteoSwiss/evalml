# Credentials setup

Some experiments are stored on the ECMWF-hosted MLflow server at
[https://mlflow.ecmwf.int](https://mlflow.ecmwf.int). To access these runs in
the evaluation workflow, you need to authenticate using a valid token.

## One-time login

Run the following commands **once** to obtain a token:

```bash
uv pip install anemoi-training --no-deps
anemoi-training mlflow login --url https://mlflow.ecmwf.int
```

You will be prompted to paste a seed token obtained from
<https://mlflow.ecmwf.int/seed>.

## Token lifetime

After the first login, the token is stored locally and reused for subsequent
runs. Tokens are valid for **30 days**, but every successful training or
evaluation run within that window automatically extends the token by another
30 days.

It's good practice to run the login command before executing the workflow to
ensure the token is still valid — a stale token will surface as an MLflow
authentication error during `inference_get_checkpoint`.

## Other checkpoint sources

EvalML accepts checkpoints from three sources, classified by URL in
[common.smk](../workflow/inference.md):

- **MLflow** (`mlflow.ecmwf.int`, `service.meteoswiss.ch`,
  `servicedepl.meteoswiss.ch`).
- **Hugging Face** (`huggingface.co/<repo>/blob/<rev>/<path>.ckpt`) — uses
  `uvx hf download` and follows the standard Hugging Face authentication
  (set `HF_TOKEN` if the repo is gated).
- **Local paths** — a filesystem path that exists is treated as a local
  checkpoint.
