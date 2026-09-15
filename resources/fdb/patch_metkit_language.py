"""Patch metkitlib's bundled language.yaml to recognize VARDA-SINGLE's custom
`model` values for FDB *retrieve* requests.

`model` is not a standard MARS/metkit key -- it was added purely for FDB archival,
via the `marsModel` concepts in eccodes-cosmo-mars's local.215.def/local.98.def
(see resources/fdb/realtime-varda.schema and the VARDA-SINGLE FDB archival notes).
`fdb.list()` doesn't validate key values against any vocabulary, so archival and
completeness checks work fine without this patch. But `fdb.retrieve()` parses
requests through metkit's MarsLanguage, which validates enum-typed keys like
`model` against language.yaml's fixed value list -- and rejects anything not
already registered there, even though the value is a perfectly valid archived key.

Run once per venv build, after `metkitlib` is installed (it ships its own copy of
language.yaml under its package data; there's no config-based override for this,
unlike ECCODES_DEFINITION_PATH/FDB5_CONFIG_FILE, so the file has to be edited in place).
"""

import argparse
import re
import sys
from pathlib import Path

# Custom VARDA-SINGLE model names, per marsModel concepts in eccodes-cosmo-mars's
# local.215.def (ICON, GPI=180) / local.98.def (IFS, GPI=180). "varda-single-g" is
# the global-domain variant seen from the temporal downscaler's IFS-grid output.
_CUSTOM_MODELS = ["varda-single", "varda-single-g"]

# Structural anchor, not a specific value: metkitlib/share/metkit/language.yaml's
# top-level `  model:` key ends with the default (no-context) `type: enum` block,
# listing recognized model names -- its last entry (and thus the exact line to
# anchor on) shifts between metkitlib releases as upstream adds new models, so
# this matches the *end of the model: block* (a blank line before the next
# top-level `  key:`) instead of any one entry.
_BLOCK_END_RE = re.compile(r"(\n  model:\n(?:.*\n)*?)(\n  [a-zA-Z_]+:)")


def patch(language_yaml: Path) -> None:
    text = language_yaml.read_text()
    if all(f"- [{m}]" in text for m in _CUSTOM_MODELS):
        print(f"[patch_metkit_language] {language_yaml} already patched, skipping")
        return
    m = _BLOCK_END_RE.search(text)
    if m is None:
        raise RuntimeError(
            f"Could not locate the end of the 'model:' block in {language_yaml} -- "
            "metkitlib's language.yaml layout may have changed; update "
            "_BLOCK_END_RE in this script."
        )
    addition = "\n".join(f"          - [{model}]" for model in _CUSTOM_MODELS)
    text = text[: m.end(1)] + addition + text[m.end(1) :]
    language_yaml.write_text(text)
    print(f"[patch_metkit_language] added {_CUSTOM_MODELS} to {language_yaml}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "venv", type=Path, help="Path to the venv root (contains lib/python*/site-packages)"
    )
    args = parser.parse_args()

    matches = list(args.venv.glob("lib/python*/site-packages/metkitlib/share/metkit/language.yaml"))
    if not matches:
        raise RuntimeError(f"metkitlib language.yaml not found under {args.venv}")
    for language_yaml in matches:
        patch(language_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
