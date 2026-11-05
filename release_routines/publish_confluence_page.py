# /// script
# requires-python = ">=3.10"
# ///
"""Create a new Confluence page next to an existing one (same parent, same space). Nothing existing is changed.

Usage: uv run publish_confluence_page.py <existing_page_url> <title> <body.xhtml> [--draft]
Body is Confluence storage format. Credentials are read from ~/.atlassian-token (either `email:token`, or a bare token
with the email in $ATLASSIAN_EMAIL) and never printed.
"""

import base64
import json
import os
import pathlib
import re
import sys
import urllib.request

SITE, EMAIL = "https://meteoswiss.atlassian.net", os.environ.get("ATLASSIAN_EMAIL")
page_url, title, body_file = sys.argv[1:4]
status = "draft" if "--draft" in sys.argv[4:] else "current"
# The token file holds either a bare token or `email:token`.
cred = pathlib.Path("~/.atlassian-token").expanduser().read_text().strip()
auth = (
    "Basic "
    + base64.b64encode((cred if ":" in cred else f"{EMAIL}:{cred}").encode()).decode()
)


def api(method, path, data=None):
    req = urllib.request.Request(
        SITE + "/wiki/api/v2" + path,
        method=method,
        data=data and json.dumps(data).encode(),
        headers={
            "Authorization": auth,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path}: HTTP {e.code}\n{e.read().decode()[:1000]}")


page_id = re.search(r"/pages/(\d+)", page_url).group(1)
existing = api("GET", f"/pages/{page_id}")
page = api(
    "POST",
    "/pages",
    {
        "spaceId": existing["spaceId"],
        "parentId": existing["parentId"],
        "status": status,
        "title": title,
        "body": {
            "representation": "storage",
            "value": pathlib.Path(body_file).read_text(),
        },
    },
)
links = page["_links"]
print(
    f"Page created ({status}), next to '{existing['title']}':",
    links["base"]
    + links.get("editui" if status == "draft" else "webui", links["webui"]),
)
