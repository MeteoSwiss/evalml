# The experiment store

Registering promotes a finished evaluation from the working directory into a shared,
immutable archive at `/store_new/mch/msopr/ml/experiments`, cross-referenced with
devml's model store at `/store_new/mch/msopr/ml/models`. The design mirrors devml's
model registration: one owned directory per registration, an append-only JSONL index,
tombstones that retire names forever, and a Confluence page that is a copy of the
index — never the truth.

## Commands

```
evalml register RESULTS_DIR [NAME]    # promote results/<experiment> into the store
evalml unregister NAME                # take a registration out (tombstones the name)
evalml list [--json] [--rebuild] [--columns ...]
evalml publish [--dry-run]            # write the index to the Confluence page
```

All accept `--store DIR` (and, where relevant, `--models-store DIR`) so tests run
against temp directories. Register, unregister, and publish have `--dry-run`;
`list --rebuild` does not — it rewrites the index and recreates missing model-store
symlinks whenever it runs.

`RESULTS_DIR` is the workflow's `<output_root>/results/{date}_{config-label}_{confighash}`.
It must contain `config.yaml` (the `experiment_config` rule copies it there); for older
results, pass `--config`. `NAME` is a slug (`^[a-z0-9][a-z0-9-]*$`), defaulting to
`{config-label-slug}-{date}`. It is the identity everywhere — directory, index key,
symlink, Confluence anchor — and is **never reused**: unregistering leaves an
`.orphaned-<name>` tombstone.

`list` and `publish` need no venv: `python src/evalml/store.py list` runs on a login
node's system python (3.6, stdlib only). Registering reads the config YAML and needs
the venv.

## Layout and atomicity

```
<store>/                    # 3775: anyone in the group adds, only the owner removes
  index.jsonl
  <experiment-name>/        # 0755, owned by whoever registered it
    experiment.json         # written once, then 0444
    results/                # copied from the WD, files read-only
```

Registration builds everything in `<store>/.tmp.<name>` and renames it into place; a
crash leaves a `.tmp.` directory to sweep, never a half-registered experiment.

## experiment.json — the contract

`name`, `models`, and `baselines` are the contract; the rest is provenance.

```json
{
  "name": "forecasters-ich1-oper-20260824",
  "registered": "2026-08-24T10:00:00+02:00",
  "by": "fzanetta",
  "description": "…the config's description…",
  "models": ["amber-ridge"],
  "baselines": ["ICON-CH1"],
  "evalml": {"version": "…", "commit": "…", "dirty": false, "config": "results/config.yaml"},
  "identity": "20260824_forecasters-ich1-oper_ab12"
}
```

- `models`: **registered model-store names only** (stable foreign keys), detected from
  the config's `checkpoint` entries — a bare name or a path under the model store both
  count; MLflow/HF URLs and plain paths are not cross-referenced. Each listed name must
  exist as `<models-store>/<name>/model.json`, or registration refuses.
- `baselines`: descriptive free text, no existence check.
- `identity`: the results directory's basename; what "already registered" is judged by,
  always against the store itself.
- `evalml.commit`/`dirty` describe the evalml tree **at registration time** (not
  necessarily the run's) — recorded, never gated on.

## Cross-references

The truth of the model↔experiment edge is `experiment.json`'s `models` list. The
reverse direction is materialized as relative symlinks, created at registration:

```
<models-store>/<model>/experiments/<name> -> ../../../experiments/<name>
```

Symlink failures warn and carry on (`list --rebuild` recreates missing ones); dangling
symlinks are history — reported, never deleted. Nothing else is ever written into the
model store. The Models page on Confluence renders these symlinks but is owned by
devml's publisher: when evalml changes a model's links it prints a reminder that the
page is stale (`just model-publish` in devml refreshes it).

## The index

`index.jsonl`: one JSON object per line, one line per register/unregister event,
append-only; replaying it yields the inventory. **The store is the truth; the index is
derived** — a failed append warns, and `list --rebuild` reconstructs it from the
experiment directories. `list` cross-checks store and index both ways and reports
disagreements.

## Unregistering

Only the owner can unregister. Destructive step last: rename to `.orphaned-<name>`,
write `NOTE.txt`, append the index line, remove this experiment's model-store symlinks,
and only then delete the copied results (`--keep-results` keeps them).

## Confluence

`evalml publish` renders the index as a table on the
[Experiments page](https://meteoswiss.atlassian.net/wiki/spaces/MR/pages/2142666865/Experiments)
(`EXPERIMENTS_PAGE` in `src/evalml/store.py`). The tool owns only its marker-delimited
section (`<!-- evalml:experiment-index:start/end -->`), located by its "Registered
experiments" heading if Confluence strips the comments, appended if neither is found.
Name cells carry anchor macros (bare `#<name>` fragments resolve — verified 2026-08-25
against the rendered HTML); the Models column links to devml's Models page (id
`2139488787`), whose `#<model-name>` fragments start landing once devml's publisher
adds anchors there.

Credentials, exactly like devml: a token in `~/.atlassian-token` (bare or
`email:token`), or `$ATLASSIAN_TOKEN`/`$ATLASSIAN_EMAIL`, with `git config user.email`
as the fallback address; `python src/evalml/store.py publish` also takes `--email`,
`--token-file`, and `--site`. The token is never printed or logged. Register and
unregister publish at the end; a publish failure only warns, and publish refuses to
touch the real page from any non-default `--store`.
