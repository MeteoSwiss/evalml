# The experiment store

Registering an experiment promotes a finished evaluation from the working directory into
a shared, immutable archive at `/store_new/mch/msopr/ml/experiments`, cross-referenced
with the model store devml maintains at `/store_new/mch/msopr/ml/models`. The design
mirrors devml's model registration: an owned directory per registration, an append-only
JSONL index, tombstones that retire names forever, and a Confluence page that is a copy
of the index — never the truth.

## Commands

```
evalml register RESULTS_DIR [NAME]    # promote results/<experiment> into the store
evalml unregister NAME                # take a registration back out (tombstones the name)
evalml list [--json] [--rebuild]      # what is in the store; --rebuild repairs the index
evalml publish [--dry-run]            # write the index to the Confluence page
```

All four accept `--store DIR` (and, where relevant, `--models-store DIR`) so tests can
run against temp directories; everything that changes anything has `--dry-run`.

`RESULTS_DIR` is the workflow's results directory,
`<output_root>/results/{date}_{config-label}_{confighash}`. It must contain
`config.yaml` — the workflow copies the config there since the `experiment_config` rule
was added; for older results, pass `--config` pointing at the YAML that produced them.

`NAME` is a slug (`^[a-z0-9][a-z0-9-]*$`). Omitted, it defaults to
`{config-label-slug}-{date}`, e.g. `forecasters-ich1-oper-20260824`. The name is the
identity everywhere — directory, index key, symlink, Confluence anchor — and is **never
reused**: unregistering leaves an `.orphaned-<name>` tombstone that retires it for good.

Listing (and publishing) need no evalml environment: `python src/evalml/store.py list`
runs on a login node's system python (3.6, stdlib only). Registering reads the config
YAML and therefore needs the venv.

## The store layout

```
/store_new/mch/msopr/ml/experiments/     # 3775: anyone in the group adds,
  index.jsonl                            #   only whoever added a thing removes it
  <experiment-name>/                     # 0755, owned by whoever registered it
    experiment.json                      # written once, then read-only (0444)
    results/                             # copied from the WD, files read-only
```

Registration is atomic: everything is built in `<store>/.tmp.<name>` beside the target
and renamed into place. A crash leaves a `.tmp.` directory to sweep, never a
half-registered experiment; `list` skips `.tmp.*` and `.orphaned-*`.

## experiment.json — the contract

Written once at registration, then read-only. `name`, `models`, and `baselines` are the
contract other tools read; the rest is provenance.

```json
{
  "name": "forecasters-ich1-oper-20260824",
  "registered": "2026-08-24T10:00:00+02:00",
  "by": "fzanetta",
  "description": "…the config's description — required, non-empty…",
  "models": ["amber-ridge"],
  "baselines": ["ICON-CH1"],
  "evalml": {"version": "…", "commit": "…", "dirty": false, "config": "results/config.yaml"},
  "identity": "20260824_forecasters-ich1-oper_ab12"
}
```

- `models` holds **registered model-store names only** — stable foreign keys, since the
  model store never reuses a name. They are detected from the config's `checkpoint`
  entries: a bare registered name (`amber-ridge`) or a path under the model store both
  count; MLflow/Hugging Face URLs and plain local paths have no stable identity and are
  not cross-referenced. Every listed name must exist as `<models-store>/<name>/model.json`
  at registration time, or registration refuses.
- `baselines` is descriptive free text (the baseline labels from the config). No
  existence check, no cross-reference.
- `identity` is the results directory's basename — the workflow's own name for one
  evaluation run — and is what "already registered" is judged by. The store is asked
  (the `experiment.json` files are scanned), never any local state.
- `evalml.commit`/`dirty` record which evalml produced the registration. Recorded, not
  gated on — but a reader can tell a dirty-tree run from a reproducible one.

## Cross-references

The truth of the model↔experiment edge lives in one place: `experiment.json`'s `models`
list. The reverse direction is materialized as convenience symlinks, one per referenced
model, created at registration:

```
/store_new/mch/msopr/ml/models/<model>/experiments/<experiment-name>
    -> ../../../experiments/<experiment-name>
```

Relative, exactly that shape, so both stores survive different mount prefixes. Rules:

- Symlink creation failure warns and carries on — the symlink is derived state, and
  `evalml list --rebuild` recreates missing ones.
- A dangling symlink (the model was unregistered later) is accurate history: reported,
  never deleted, never "fixed".
- Nothing else is ever written into the model store, and `model.json` is never touched.
- The Models page on Confluence renders these symlinks, but devml's publisher owns that
  page. When registering or unregistering changes a model's links, evalml prints a
  reminder that the page is stale; `just model-publish` in devml refreshes it.

## The index

`<store>/index.jsonl`: one JSON object per line, one line per event, append-only.
Registering appends a `register` line (the flattened `experiment.json` plus location and
size), unregistering an `unregister` line. Replaying top to bottom yields the current
inventory. **The store is the truth; the index is derived** — a failed append warns and
carries on, and `evalml list --rebuild` reconstructs the file from the experiment
directories (losing only the memory of past unregistrations, which the tombstones keep).

`evalml list` cross-checks as it reads: every indexed name must be in the store, every
store directory must be indexed, and disagreements are reported.

## Unregistering

Only whoever registered an experiment can unregister it (the store is sticky). Order
matters, destructive step last: the directory is renamed to `.orphaned-<name>`, a
`NOTE.txt` says what it was, the index line is appended, this experiment's symlinks are
removed from the model store (failures on links you don't own are reported and
tolerated), and only then are the copied results deleted (`--keep-results` keeps them).
Tombstones are permanent — they are what retires the name.

## Confluence

`evalml publish` renders the index as a table on the Experiments page. The tool owns a
marker-delimited section of the page (`<!-- evalml:experiment-index:start/end -->`), not
the page: only what is between the markers is replaced, the section is located by its
"Registered experiments" heading if Confluence has eaten the comments, and appended if
neither is found. Each row's Name cell carries an anchor macro named after the
experiment, so other pages can deep-link `#<experiment-name>`; the Models column links
each model to devml's Models page (id `2139488787`) by id-based URL plus `#<model-name>`
fragment. The Models page itself is never written to — its Experiments column is devml's
job.

Credentials, exactly like devml: an Atlassian API token in `~/.atlassian-token` (bare
token or `email:token`), or `$ATLASSIAN_TOKEN`/`$ATLASSIAN_EMAIL`, with
`git config user.email` as the last-resort address. The token is never printed or logged.

Registering calls publish at the end, and a publish failure only warns — the store, not
the page, is the registry. Publish refuses to touch the real page when `--store` is not
the default store.

The Experiments page is
[`2142666865`](https://meteoswiss.atlassian.net/wiki/spaces/MR/pages/2142666865/Experiments)
(`EXPERIMENTS_PAGE` in `src/evalml/store.py`).

### Anchor fragments (verified 2026-08-25)

Confluence Cloud renders an anchor macro with *both* ids, `#PageTitle-anchor` and the
bare `#anchor`, so the bare `#<name>` fragments this tool emits do land — verified
against the rendered (`export_view`) HTML of the Experiments page. The Models-column
links carry a `#<model-name>` fragment too, but devml's Models page has no anchors yet
(its Name cells are plain text), so those fragments are inert until devml's
`model_publish.py` adds an anchor macro to the Name cell. The fragment is kept anyway:
it is harmless today and starts working the day the anchors appear. An Experiments
column on the Models page (rendering `<model>/experiments/` symlinks) is likewise
devml's change to make.
