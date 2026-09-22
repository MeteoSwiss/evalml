import base64
import json
import os
import urllib.request
from pathlib import Path


def _read_dotenv(path):
    result = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return result


_client_id = os.environ.get("JRETRIEVE_CLIENT_ID")
_client_secret = os.environ.get("JRETRIEVE_CLIENT_SECRET")
if not _client_id or not _client_secret:
    _dotenv = _read_dotenv(Path(os.environ.get("JRETRIEVE_CONF_DIR", ".")) / ".env")
    _client_id = _client_id or _dotenv.get("JRETRIEVE_CLIENT_ID")
    _client_secret = _client_secret or _dotenv.get("JRETRIEVE_CLIENT_SECRET")
if not _client_id or not _client_secret:
    raise RuntimeError(
        "jretrieve credentials not found. Set JRETRIEVE_CLIENT_ID and "
        "JRETRIEVE_CLIENT_SECRET in the environment or in a .env file next "
        "to this script."
    )

jretrieve_url = "https://service.meteoswiss.ch/jretrieve/api/v1"
with urllib.request.urlopen(
    urllib.request.Request(
        method="POST",
        url="https://service.meteoswiss.ch/auth/realms/meteoswiss.ch/protocol/openid-connect/token",
        data=b"grant_type=client_credentials",
        headers={
            b"Authorization": b"Basic "
            + base64.b64encode(f"{_client_id}:{_client_secret}".encode())
        },
    )
) as f:
    auth_header = "Bearer " + json.loads(f.read().decode())["access_token"]
