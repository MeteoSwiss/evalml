#!/usr/bin/env python3
"""The experiment store: promote finished evaluations into a shared, immutable archive.

    evalml register RESULTS_DIR [NAME]       (reads the config YAML — see evalml.registration)
    evalml list | unregister | publish       (thin wrappers around this module)

Layout, mirroring the model store devml maintains at /store_new/mch/msopr/ml/models:

    /store_new/mch/msopr/ml/experiments/
      index.jsonl                  # append-only event log — one JSON object per line
      <experiment-name>/           # 0755, owned by whoever registered it
        experiment.json            # written once, then read-only (0444)
        results/                   # copied from the working directory, files read-only

One rule holds throughout, same as the model store: anyone in the group may add, only
whoever added a thing may remove it. The store directory is 3775 (setgid + sticky +
group-writable); each registration is an owned 0755 directory, which — not the read-only
bit — is what stops deletion by others, since unlinking needs write on the parent.

**The store is the truth; everything else is derived.** The index, the symlinks under
each model's `experiments/`, and the Confluence page are all reconstructible from the
`experiment.json` files (`list --rebuild`), so none of them is ever allowed to fail a
registration — they warn and carry on.

Cross-references: `experiment.json`'s `models` list is the one source of truth for the
model↔experiment edge. The reverse direction is materialized as convenience symlinks
`<models-store>/<model>/experiments/<name> -> ../../../experiments/<name>` — relative,
exactly that shape, because both stores share /store_new/mch/msopr/ml/ and must survive
different mount prefixes. A dangling symlink (the model was unregistered later) is
accurate history: report it, never delete other people's links.

Names are the identity everywhere — directory, index key, symlink, Confluence anchor —
and are never reused: an unregistered experiment leaves an `.orphaned-<name>` tombstone
behind, and that name stays spent. Listing skips `.tmp.*` (a registration in progress)
and `.orphaned-*` (one taken back out).

This module owns the store itself — layout, index, symlinks, Confluence page.
Registering (reading the config YAML, collecting evalml's provenance) lives in
evalml.registration.
"""

import base64
import html
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from evalml.config import MODEL_REGISTRY_ROOT as MODELS_STORE  # devml's model store

STORE = Path("/store_new/mch/msopr/ml/experiments")

INDEX = "index.jsonl"
MANIFEST = "experiment.json"
RESULTS = "results"
ORPHAN_PREFIX = ".orphaned-"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# The reverse cross-reference, seen from <models-store>/<model>/experiments/. Relative,
# exactly this shape — see the module docstring.
LINK_TARGET = "../../../experiments/%s"

SITE = "https://meteoswiss.atlassian.net/wiki"
# The "Experiments" page in the MR space:
# https://meteoswiss.atlassian.net/wiki/spaces/MR/pages/2142666865/Experiments
EXPERIMENTS_PAGE = "2142666865"
MODELS_PAGE = "2139488787"  # devml's "Models" page — linked to, never written to
TOKEN_FILE = Path.home() / ".atlassian-token"
HEADING = "Registered experiments"
START = "<!-- evalml:experiment-index:start -->"
END = "<!-- evalml:experiment-index:end -->"


def log(message):
    sys.stderr.write("[experiment-store] %s\n" % message)


def fail(message):
    raise SystemExit("[experiment-store] error: %s" % message)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def owner_name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:  # a uid with no passwd entry — the number will do
        return str(uid)


def whoami():
    return owner_name(os.getuid())


def human(size):
    value = float(size or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return "%d %s" % (value, unit) if unit == "B" else "%.1f %s" % (value, unit)
        value /= 1024
    raise AssertionError  # unreachable: the TiB branch always returns


def measure(path):
    """Bytes under `path`. Symlinks count as links, not as what they point at."""
    total = 0
    for root, _, files in os.walk(str(path)):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass  # vanished mid-walk — not worth failing an inventory over
    return total


# ── the index ───────────────────────────────────────────────────────────────
# One JSON object per line, one line per event, append-only. Replaying it top to bottom
# yields the current inventory; nothing ever rewrites a line, so two concurrent
# registrations both survive as appends and a killed process costs one bad line.


def index_path(store):
    return Path(store) / INDEX


def _entry(event, record, **stamp):
    entry = {"event": event, "logged": now(), "by": whoami()}
    entry.update(stamp)
    entry.update(record)
    if entry.get("location"):
        entry["location"] = os.path.abspath(str(entry["location"]))
    return entry


def append(store, event, record):
    """Add one line to the log. A single write to a file opened for append: the kernel
    puts it at the end whatever else is writing."""
    entry = _entry(event, record)
    path = index_path(store)
    new = not path.exists()
    with open(str(path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if new:
        try:  # 0664, not the umask: whoever registers next must be able to append
            os.chmod(str(path), 0o664)
        except OSError:
            pass  # not ours to chmod; it is readable, which is what matters
    return entry


def rebuild(store, records):
    """Write the log again from scratch, one register line per experiment found. The one
    repair that does not append; it reconstructs the present — the tombstones in the
    store, not this file, are what retires a name. Written beside and renamed over, so a
    reader never sees a half-written index; if the file belongs to someone else (the
    store is sticky), its contents are rewritten in place instead."""
    path = index_path(store)
    staging_file = path.with_name(".tmp." + INDEX)
    with open(str(staging_file), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(_entry("register", record, rebuilt=True), ensure_ascii=False)
                + "\n"
            )
    os.chmod(str(staging_file), 0o664)
    try:
        os.rename(str(staging_file), str(path))
    except OSError:
        if not path.exists():
            os.unlink(str(staging_file))
            raise
        with open(str(staging_file), encoding="utf-8") as handle:
            content = handle.read()
        with open(str(path), "r+", encoding="utf-8") as handle:
            handle.write(content)
            handle.truncate()
        os.unlink(str(staging_file))
    return path


def load(store):
    """Replay the log: what is registered now, what was taken out, what would not parse.
    A broken line is reported rather than skipped — in an append-only file it is an
    interrupted write, and that is worth knowing."""
    result = {"experiments": [], "retired": [], "broken": []}
    path = index_path(store)
    if not path.is_file():
        return result
    current = {}
    with open(str(path), encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError as exc:
                result["broken"].append({"line": number, "problem": str(exc)})
                continue
            name = entry.get("name")
            if not name:
                result["broken"].append({"line": number, "problem": "no name"})
            elif entry.get("event") == "unregister":
                current.pop(name, None)
                result["retired"].append(entry)
            else:
                current[name] = entry
    result["experiments"] = list(current.values())
    return result


def experiment_dirs(store):
    """The store's experiment directories: the plain-named ones. `.tmp.<name>` and
    `.orphaned-<name>` are hidden precisely so a reader can skip them."""
    return sorted(
        p for p in Path(store).iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def read_manifest(path):
    return json.loads((Path(path) / MANIFEST).read_text(encoding="utf-8"))


def from_manifest(path):
    """The index record for one registered experiment, read from its own experiment.json.
    The single place that decides what a record holds: a registration writes one through
    here and a rebuild reconstructs one the same way. The nested `evalml` block is lifted
    to the top level, because a record is meant to be grepped across experiments."""
    path = Path(path)
    manifest = read_manifest(path)
    evalml = manifest.get("evalml") or {}
    record = {
        "name": manifest.get("name") or path.name,
        "description": " ".join((manifest.get("description") or "").split()),
        "models": manifest.get("models") or [],
        "baselines": manifest.get("baselines") or [],
        "identity": manifest.get("identity") or "",
        "location": os.path.abspath(str(path)),
        "registered": manifest.get("registered") or "",
        "by": manifest.get("by") or "",
        # As registered; a rebuild refreshes it.
        "size": measure(path),
        "evalml_version": evalml.get("version") or "",
        "evalml_commit": evalml.get("commit") or "",
        "evalml_dirty": bool(evalml.get("dirty")),
    }
    return record


def identities(store):
    """What each experiment in the store *is* — read from the experiment.json files, not
    from the index: the index is derived and may be stale, and a duplicate check has to
    ask the truth. A directory that cannot be read is passed over; saying so is `list`'s
    job, not a registration's."""
    found = []
    for path in experiment_dirs(store):
        try:
            manifest = read_manifest(path)
        except (OSError, ValueError):
            continue
        found.append(
            {
                "name": manifest.get("name") or path.name,
                "identity": manifest.get("identity") or "",
                "registered": manifest.get("registered") or "",
                "location": os.path.abspath(str(path)),
            }
        )
    return found


# ── names and layout ────────────────────────────────────────────────────────


def staging(store, name):
    return Path(store) / (".tmp.%s" % name)


def orphan(store, name):
    """Where unregistering leaves the tombstone. Hidden, so it cannot be mistaken for an
    experiment, and permanent: it is what keeps the name from ever being handed out again."""
    return Path(store) / (ORPHAN_PREFIX + name)


def check_name(store, name):
    """A name is a slug, and it is never reused — not while the directory exists, and not
    after it is gone: a tombstone retires it for good. That guarantee is what makes the
    cross-references (symlinks, index lines, Confluence anchors) safe to keep around."""
    if not NAME_PATTERN.match(name):
        fail(
            "%r is not a valid experiment name (lowercase letters, digits and dashes: %s)"
            % (name, NAME_PATTERN.pattern)
        )
    if (Path(store) / name).exists():
        fail(
            "%s already exists — a name is never reused; pick another"
            % (Path(store) / name)
        )
    if orphan(store, name).exists():
        fail(
            "the name %r was used and unregistered — a name is never reused; pick another"
            % name
        )
    if staging(store, name).exists():
        fail(
            "%s exists — another registration of %r is in progress, or a crashed one "
            "left it behind; sweep it by hand if it is stale"
            % (staging(store, name), name)
        )


def refuse_if_registered(store, identity):
    """Refuse to register the same evaluation twice. The identity is the results
    directory's basename — `{date}_{config-label}_{confighash}`, the workflow's own name
    for one evaluation run. The store is asked, never any local state: another clone's
    registrations are invisible from here. A check, not a lock — so it is made twice,
    before the copy and again in the instant before the rename."""
    if not identity:
        return
    for other in identities(store):
        if other["identity"] == identity:
            when = (
                ", registered %s" % other["registered"][:10]
                if other["registered"]
                else ""
            )
            fail(
                "%s is already a registration of %s%s — an evaluation is registered "
                "once.\n  To replace it, take that one out first: evalml unregister %s"
                % (other["location"], identity, when, other["name"])
            )


# ── cross-reference symlinks ────────────────────────────────────────────────


def model_link_path(models_store, model, name):
    return Path(models_store) / model / "experiments" / name


def check_models(models_store, models):
    """Every `models` entry must be a registered model-store name — they are the foreign
    keys the cross-references hang off, and the model store never reuses a name."""
    for model in models:
        manifest = Path(models_store) / model / "model.json"
        if not manifest.is_file():
            fail("%r is not a registered model (%s does not exist)" % (model, manifest))


def link_experiment(models_store, model, name):
    """Materialize the reverse edge for one model. Returns a problem string instead of
    raising: the symlink is derived state, reconstructible from experiment.json, and must
    never fail a registration — `<model>/experiments/` is sticky and group-writable by
    design, but the model may be gone or the filesystem unwilling."""
    link = model_link_path(models_store, model, name)
    try:
        if link.is_symlink() or link.exists():
            if os.readlink(str(link)) == LINK_TARGET % name:
                return None  # already there and correct — a --rebuild rerun
            return "%s exists and is not ours to replace" % link
        os.symlink(LINK_TARGET % name, str(link))
        return None
    except OSError as exc:
        return "cannot create %s: %s" % (link, exc)


def unlink_experiment(models_store, model, name):
    """Remove one reverse edge, tolerating everything: a link that is already gone, a
    model directory that is gone, a link we do not own (the store is sticky — report,
    continue, never delete other people's links)."""
    link = model_link_path(models_store, model, name)
    try:
        if link.is_symlink():
            os.unlink(str(link))
            return None
        return None  # nothing there — already gone is the desired state
    except OSError as exc:
        return "could not remove %s: %s" % (link, exc)


def models_page_hint(models_store, models):
    """The Models page on Confluence renders these symlinks, but devml's publisher owns
    that page and evalml never writes it — so when the symlinks change, say the page is
    stale instead of leaving it so silently. Only for the real store: a temp store's
    links are not what the page shows."""
    if not models:
        return
    if os.path.abspath(str(models_store)) != os.path.abspath(str(MODELS_STORE)):
        return
    log(
        "the Models page on Confluence is now stale for %s — refresh it with "
        "devml's `just model-publish`" % ", ".join(models)
    )


def repair_links(models_store, records):
    """Recreate missing symlinks and report dangling ones, for `list --rebuild`. A
    dangling link under a model that still exists points at an experiment that was
    unregistered — accurate history, reported and left alone."""
    problems = []
    for record in records:
        for model in record.get("models") or []:
            problem = link_experiment(models_store, model, record["name"])
            if problem:
                problems.append(
                    {
                        "path": str(
                            model_link_path(models_store, model, record["name"])
                        ),
                        "problem": problem,
                    }
                )
    for model_dir in (
        Path(models_store).iterdir() if Path(models_store).is_dir() else []
    ):
        experiments = model_dir / "experiments"
        if not experiments.is_dir():
            continue
        for link in experiments.iterdir():
            if link.is_symlink() and not link.exists():
                problems.append(
                    {
                        "path": str(link),
                        "problem": "dangling — the experiment it points at is gone (kept: it is history)",
                    }
                )
    return problems


# ── registration ────────────────────────────────────────────────────────────
# evalml.registration reads the config YAML and hands everything store-shaped down
# to here.


def describe_registration(results_dir, name, meta, store, models_store):
    lines = ["about to register %s as %r:" % (results_dir, name)]
    lines.append(
        "  copies      %s (%s) -> %s"
        % (results_dir, human(measure(results_dir)), Path(store) / name / RESULTS)
    )
    lines.append(
        "  writes      %s (read-only once placed)" % (Path(store) / name / MANIFEST)
    )
    lines.append("  identity    %s" % meta.get("identity"))
    for model in meta.get("models") or []:
        lines.append(
            "  symlinks    %s -> %s"
            % (model_link_path(models_store, model, name), LINK_TARGET % name)
        )
    if not meta.get("models"):
        lines.append("  symlinks    none — no registered models in the config")
    lines.append(
        "  baselines   %s" % (", ".join(meta.get("baselines") or []) or "none")
    )
    lines.append("  index       a line appended to %s" % index_path(store))
    return "\n".join(lines)


def ensure_store(store):
    """Create the store root on first registration: 3775 — setgid + sticky +
    group-writable — the same contract as the model store."""
    store = Path(store)
    if store.is_dir():
        return
    store.mkdir(parents=True)
    try:
        os.chmod(str(store), 0o3775)
    except OSError:
        pass


def harden(root):
    """Explicit modes, not the caller's umask: directories 0755 (ours alone — what stops
    deletion by others), files 0444 (what stops overwrites and makes `rm -r` stop and ask)."""
    os.chmod(str(root), 0o755)
    for parent, dirs, files in os.walk(str(root)):
        for d in dirs:
            os.chmod(os.path.join(parent, d), 0o755)
        for f in files:
            path = os.path.join(parent, f)
            if not os.path.islink(path):
                os.chmod(path, 0o444)


def register(
    results_dir,
    name,
    meta,
    store=STORE,
    models_store=MODELS_STORE,
    config_src=None,
    dry_run=False,
    no_publish=False,
):
    """Copy `results_dir` into the store as `<store>/<name>` and cross-reference it.

    `meta` is the experiment.json payload minus name/registered/by (description, models,
    baselines, identity, evalml provenance) — extracted by evalml.registration, which is
    the tier that can read YAML. `config_src` is a config file to copy in as
    `results/config.yaml` when the results directory itself does not carry one.

    Everything that can fail is checked before anything is copied; the copy lands in
    `.tmp.<name>` beside the target and is renamed into place, so `<store>/<name>` never
    exists half-written. Derived state (symlinks, index, Confluence) comes after the
    rename and warns instead of failing.
    """
    results_dir = Path(results_dir)
    results_dir.is_dir() or fail("%s is not a directory" % results_dir)
    (meta.get("description") or "").strip() or fail(
        "the experiment has no description — the config's `description` is what the store shows"
    )
    if not (results_dir / "config.yaml").is_file() and config_src is None:
        fail(
            "%s does not contain config.yaml — re-run the experiment with a current evalml, "
            "or pass --config pointing at the YAML that produced it" % results_dir
        )
    check_models(models_store, meta.get("models") or [])
    if Path(store).is_dir():
        refuse_if_registered(store, meta.get("identity"))
        check_name(store, name)
    else:
        NAME_PATTERN.match(name) or fail("%r is not a valid experiment name" % name)

    if dry_run:
        log(describe_registration(results_dir, name, meta, store, models_store))
        log("dry run — nothing copied, nothing written")
        return name

    ensure_store(store)
    check_name(store, name)  # again, now that the store certainly exists
    tmp = staging(store, name)
    try:
        tmp.mkdir()
    except FileExistsError:
        fail("%s exists — another registration of %r is in progress" % (tmp, name))

    try:
        log("copying %s (%s) ..." % (results_dir, human(measure(results_dir))))
        # Symlinked results are materialized: the store outlives the working directory
        # they would point into. A dangling one is skipped rather than fatal.
        shutil.copytree(
            str(results_dir), str(tmp / RESULTS), ignore_dangling_symlinks=True
        )
        if config_src is not None and not (tmp / RESULTS / "config.yaml").is_file():
            shutil.copy2(str(config_src), str(tmp / RESULTS / "config.yaml"))

        manifest = {"name": name, "registered": now(), "by": whoami()}
        manifest.update(meta)
        (tmp / MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        harden(tmp)

        # The last instant at which nothing has been published: the copy took a while and
        # the store may have gained a registration of this very evaluation meanwhile.
        refuse_if_registered(store, meta.get("identity"))
        target = Path(store) / name
        tmp.rename(target)
    except BaseException:
        # BaseException, not Exception: a refused duplicate (SystemExit) and a Ctrl-C
        # during the copy both must take the staging directory with them — a .tmp.<name>
        # left behind blocks that name until someone sweeps it.
        shutil.rmtree(str(tmp), ignore_errors=True)
        raise
    log("registered at %s" % target)

    # From here on everything is derived, and nothing may fail the registration.
    for model in meta.get("models") or []:
        problem = link_experiment(models_store, model, name)
        if problem:
            log(
                "WARNING: cross-reference not created — %s\n  the registration is fine; "
                "`evalml list --rebuild` recreates missing links" % problem
            )
        else:
            log("linked %s" % model_link_path(models_store, model, name))
    models_page_hint(models_store, meta.get("models") or [])
    try:
        append(store, "register", from_manifest(target))
    except (OSError, ValueError) as exc:
        log(
            "WARNING: %s was not added to %s: %s\n  the registration is fine — "
            "`evalml list --rebuild` writes the line later"
            % (name, index_path(store), exc)
        )
    if not no_publish:
        publish_quietly(store)
    return name


def publish_quietly(store):
    """Update the Confluence page, and never mind if it cannot: the store, not the page,
    is the registry. Refuses (quietly) to touch the real page from any non-default store."""
    if os.path.abspath(str(store)) != os.path.abspath(str(STORE)):
        log("%s is not the default store — leaving the Confluence page alone" % store)
        return
    try:
        cmd_publish(store=Path(store))
    except SystemExit as exc:
        log("WARNING: the Confluence page was not updated — %s" % str(exc).strip())
        log(
            "  nothing about the registration changes; run `evalml publish` to catch it up"
        )
    except Exception as exc:  # noqa: BLE001 — a wiki page is never worth a traceback here
        log("WARNING: the Confluence page was not updated — %r" % (exc,))
        log(
            "  nothing about the registration changes; run `evalml publish` to catch it up"
        )


# ── unregister ──────────────────────────────────────────────────────────────


def payload(target):
    """What the registration itself wrote — the only things we may delete."""
    found = []
    if (target / MANIFEST).is_file():
        found.append(target / MANIFEST)
    if (target / RESULTS).is_dir():
        found.append(target / RESULTS)
    return found


def check_store_copy(target, name):
    """Refuse to touch a directory that is not ours or not what it is named after."""
    owner = target.stat().st_uid
    if owner != os.getuid():
        fail(
            "%s belongs to %s (uid %d), not you — the store is sticky, so only whoever "
            "registered %s can take it back out"
            % (target, owner_name(owner), owner, name)
        )
    if not os.access(str(target.parent), os.W_OK):
        fail(
            "no write access to %s — the directory cannot be renamed out of the way"
            % target.parent
        )
    try:
        manifest = read_manifest(target)
    except (OSError, ValueError) as exc:
        fail(
            "%s has no readable %s (%s) — refusing to guess what it is"
            % (target, MANIFEST, exc)
        )
    if manifest.get("name") != name:
        fail(
            "%s calls itself %r, not %r"
            % (target / MANIFEST, manifest.get("name"), name)
        )
    return manifest


def write_note(tomb, name, manifest, reason, kept):
    """Leave the tombstone able to explain itself to whoever finds it."""
    kept_here = (
        "Its results are still here, but this is not a registered experiment any more."
        if kept
        else "Its results have been deleted."
    )
    paragraphs = [
        "Not a registered experiment.",
        '"%s" was a registration of %s, unregistered on %s.'
        % (
            name,
            manifest.get("identity") or "an evalml evaluation",
            datetime.now().date().isoformat(),
        ),
    ]
    if reason:
        paragraphs.append("Reason: %s" % reason)
    paragraphs.append(
        kept_here + " The name is retired: a re-registration gets a new one."
    )
    wrapped = (
        textwrap.fill(p, 80, break_on_hyphens=False, break_long_words=False)
        for p in paragraphs
    )
    (tomb / "NOTE.txt").write_text("\n\n".join(wrapped) + "\n", encoding="utf-8")


def confirm(question):
    if not sys.stdin.isatty():
        fail("not a terminal and --yes was not given — refusing to unregister unasked")
    sys.stderr.write("%s [y/N] " % question)
    sys.stderr.flush()
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_unregister(
    name,
    store=STORE,
    models_store=MODELS_STORE,
    reason="",
    dry_run=False,
    yes=False,
    keep_results=False,
    no_publish=False,
):
    """Order matters, in the opposite direction to registering: checks first, the rename
    is the pivot (reversible), derived state next, and the copied results are deleted
    last — only once everything agrees the experiment is gone. A failure halfway always
    fails with the data still there, a rename away from being back."""
    target = Path(store) / name
    tomb = orphan(store, name)
    if not target.exists():
        fail(
            "%s does not exist — nothing to unregister.\n  `evalml list` shows what is in %s"
            % (target, store)
        )
    manifest = check_store_copy(target, name)
    if tomb.exists():
        fail(
            "%s already exists — an earlier unregistration left it behind; clear it by hand first"
            % tomb
        )

    models = manifest.get("models") or []
    files = payload(target)
    size = measure(target)
    lines = ["about to unregister %s:" % name]
    if keep_results:
        lines.append(
            "  store       %s -> %s, contents kept (--keep-results)"
            % (target, tomb.name)
        )
    else:
        lines.append("  store       %s -> %s" % (target, tomb.name))
        lines.append(
            "  DELETES     %s — %s" % (human(size), ", ".join(f.name for f in files))
        )
    for model in models:
        lines.append("  removes     %s" % model_link_path(models_store, model, name))
    lines.append("  index       a line in %s takes it back out" % index_path(store))
    lines.append("  the name %s is retired for good" % name)
    log("\n".join(lines))
    if dry_run:
        log("dry run — nothing moved, nothing deleted, nothing written")
        return
    if not yes and not confirm("Unregister %s?" % name):
        fail("cancelled — nothing was touched")

    # 1. The pivot: puts the experiment out of reach of anything reading the store.
    target.rename(tomb)
    os.chmod(str(tomb), 0o755)
    write_note(tomb, name, manifest, reason, keep_results)
    log("moved to %s" % tomb)

    # 2. The index. Appended, never edited; derived, so it may not fail the unregistration.
    try:
        append(
            store,
            "unregister",
            {
                "name": name,
                "identity": manifest.get("identity") or "",
                "location": str(tomb),
                "reason": reason,
                "kept": keep_results,
            },
        )
    except OSError as exc:
        log(
            "WARNING: %s was not taken out of %s: %s\n  the unregistration is fine — "
            "`evalml list --rebuild` fixes the index" % (name, index_path(store), exc)
        )

    # 3. Our symlinks in the model store. Tolerate everything: report, carry on.
    for model in models:
        problem = unlink_experiment(models_store, model, name)
        if problem:
            log("WARNING: %s" % problem)
        else:
            log("removed %s" % model_link_path(models_store, model, name))
    models_page_hint(models_store, models)

    # 4. Last, and only now: the copied results.
    if not keep_results:
        for path in payload(tomb):
            if path.is_dir():
                _rmtree_readonly(path)
            else:
                os.chmod(str(path), 0o644)
                path.unlink()
        log("deleted the results under %s" % tomb)

    if not no_publish:
        publish_quietly(store)
    print(name)


def _rmtree_readonly(path):
    """rmtree over files we deliberately made 0444: writable-ize, then remove."""
    for parent, dirs, files in os.walk(str(path)):
        for f in files:
            try:
                os.chmod(os.path.join(parent, f), 0o644)
            except OSError:
                pass
    shutil.rmtree(str(path))


# ── list ────────────────────────────────────────────────────────────────────

COLUMNS = (
    "name",
    "models",
    "baselines",
    "registered",
    "by",
    "size",
    "description",
    "identity",
    "location",
    "evalml_commit",
)
DEFAULT_COLUMNS = "name,models,baselines,registered,by,size,description"


def scan(store):
    records, problems = [], []
    for path in experiment_dirs(store):
        try:
            records.append(from_manifest(path))
        except OSError as exc:
            problems.append(
                {"path": str(path), "problem": "no readable %s (%s)" % (MANIFEST, exc)}
            )
        except ValueError as exc:
            problems.append(
                {
                    "path": str(path),
                    "problem": "%s is not readable JSON (%s)" % (MANIFEST, exc),
                }
            )
    return records, problems


def verify(store, records):
    """Hold the index against the store: every indexed name must be there, every store
    directory must be indexed. Cheap on purpose, so it runs on every list."""
    problems = []
    indexed = set()
    for record in records:
        path = Path(record.get("location") or (Path(store) / record["name"]))
        indexed.add(path.name)
        if not path.is_dir():
            record["missing"] = True
            problems.append(
                {"path": str(path), "problem": "in the index, but not in the store"}
            )
    for path in experiment_dirs(store):
        if path.name not in indexed:
            problems.append(
                {
                    "path": str(path),
                    "problem": "in the store, but not in the index (--rebuild)",
                }
            )
    return problems


def clip(text, width):
    return text if len(text) <= width else text[: width - 1] + "…"


def cell(record, column):
    value = record.get(column)
    if column == "size":
        return human(value or 0)
    if column == "name" and record.get("missing"):
        return "%s (gone)" % value
    if column == "registered":
        return (value or "")[:10]
    if column == "evalml_commit":
        return ((value or "")[:9] + ("*" if record.get("evalml_dirty") else "")) or "-"
    if isinstance(value, list):
        return ", ".join(value) or "-"
    return value or "-"


def render_table(records, columns, width):
    """Only free text is ever shortened: a name or a path is worth nothing truncated, so
    the description column is the one that gives way, down to a floor of 20 characters."""
    rows = [[cell(record, column) for column in columns] for record in records]
    widths = [
        max([len(column)] + [len(row[i]) for row in rows])
        for i, column in enumerate(columns)
    ]
    slack = width - (sum(widths) + 2 * (len(columns) - 1))
    if slack < 0 and "description" in columns:
        index = columns.index("description")
        widths[index] = max(20, widths[index] + slack)
    lines = ["  ".join(c.upper().ljust(w) for c, w in zip(columns, widths)).rstrip()]
    for row in rows:
        lines.append(
            "  ".join(clip(c, w).ljust(w) for c, w in zip(row, widths)).rstrip()
        )
    return "\n".join(lines)


def cmd_list(
    store=STORE,
    models_store=MODELS_STORE,
    as_json=False,
    rebuild_index=False,
    columns=DEFAULT_COLUMNS,
):
    wanted = [column.strip() for column in columns.split(",") if column.strip()]
    unknown = [column for column in wanted if column not in COLUMNS]
    if unknown:
        fail(
            "no such column: %s (available: %s)"
            % (", ".join(unknown), ", ".join(COLUMNS))
        )
    if not Path(store).is_dir():
        fail("store directory %s does not exist" % store)
    store = Path(os.path.abspath(str(store)))

    retired, broken = [], []
    if rebuild_index:
        records, problems = scan(store)
        try:
            path = rebuild(store, records)
        except OSError as exc:
            fail("cannot write the index: %s" % exc)
        log("rebuilt %s from %d experiment directory(s)" % (path, len(records)))
        problems += repair_links(models_store, records)
    else:
        index = load(store)
        records, retired, broken = (
            index["experiments"],
            index["retired"],
            index["broken"],
        )
        problems = verify(store, records)

    records.sort(key=lambda r: r.get("registered") or "", reverse=True)

    if as_json:
        print(
            json.dumps(
                {
                    "store": str(store),
                    "generated": now(),
                    "experiments": records,
                    "retired": retired,
                    "problems": problems,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        log(
            "%d experiment(s) in %s, %s%s"
            % (
                len(records),
                store,
                human(sum(r.get("size") or 0 for r in records)),
                ", %d name(s) retired" % len(retired) if retired else "",
            )
        )
        if records:
            print(
                render_table(
                    records, wanted, shutil.get_terminal_size((120, 24)).columns
                )
            )
    for problem in problems:
        log("WARNING: %s: %s" % (problem["path"], problem["problem"]))
    for problem in broken:
        log(
            "WARNING: %s line %d: %s"
            % (index_path(store), problem["line"], problem["problem"])
        )


# ── Confluence ──────────────────────────────────────────────────────────────
# Mirrors devml's model_publish.py: the tool owns a marker-delimited section of the
# page, not the page; storage-format XHTML over the v2 REST API via urllib.


def credentials(email="", token_file=TOKEN_FILE):
    """The `email:token` pair Atlassian Cloud authenticates with. The token file may hold
    a bare token or `email:token`; `git config user.email` is the last resort. The token
    is never printed, logged, or passed on a command line."""
    token = os.environ.get("ATLASSIAN_TOKEN") or ""
    source = "--email"
    if not email:
        email, source = os.environ.get("ATLASSIAN_EMAIL") or "", "$ATLASSIAN_EMAIL"
    if not token:
        path = token_file
        if not path.is_file():
            fail("no API token: %s does not exist (or set $ATLASSIAN_TOKEN)" % path)
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            fail("cannot read %s: %s" % (path, exc))
        if ":" in raw and "@" in raw.split(":", 1)[0]:
            found, token = raw.split(":", 1)
            if not email:
                email, source = found, str(path)
        else:
            token = raw
    if not token:
        fail("the API token is empty")
    if not email:
        try:
            email = (
                subprocess.check_output(["git", "config", "user.email"])
                .decode()
                .strip()
            )
            source = "git config user.email"
        except (subprocess.CalledProcessError, OSError):
            email = ""
    if not email:
        fail(
            "no account email to authenticate with — Atlassian Cloud wants email:token.\n"
            "  Pass --email, set $ATLASSIAN_EMAIL, or put `email:token` in the token file"
        )
    return email, token, source


def page_id(reference):
    reference = (reference or "").strip()
    if reference.isdigit():
        return reference
    found = re.search(r"/pages/(\d+)", reference)
    if found:
        return found.group(1)
    fail(
        "cannot find a page id in %r — pass the id, or a .../pages/<id>/... URL"
        % reference
    )


class Confluence:
    def __init__(self, site, email, token):
        self.site = site.rstrip("/")
        self.auth = base64.b64encode(("%s:%s" % (email, token)).encode("utf-8")).decode(
            "ascii"
        )

    def _call(self, method, path, payload=None):
        import urllib.error
        import urllib.request

        url = self.site + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", "Basic " + self.auth)
        request.add_header("Accept", "application/json")
        if data:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code == 409:
                fail(
                    "the page was edited between reading it and writing it back — nothing "
                    "was published. Run it again.\n  %s" % detail
                )
            if exc.code in (401, 403):
                fail(
                    "Confluence refused the credentials (%s). Check that the token is current "
                    "and belongs to the email it is being sent with.\n  %s"
                    % (exc.code, detail)
                )
            if exc.code == 404:
                fail(
                    "Confluence returned 404 for that page. Either the page id is wrong, or "
                    "the account has no access to it, or the token is not valid — a stale "
                    "token looks exactly like a missing page from here.\n  %s" % detail
                )
            fail(
                "%s %s failed: %s %s\n  %s"
                % (method, url, exc.code, exc.reason, detail)
            )
        except urllib.error.URLError as exc:
            fail("cannot reach %s: %s" % (url, exc.reason))

    def get(self, page):
        return self._call("GET", "/api/v2/pages/%s?body-format=storage" % page)

    def update(self, page, title, body, version, message):
        return self._call(
            "PUT",
            "/api/v2/pages/%s" % page,
            {
                "id": str(page),
                "status": "current",
                "title": title,
                "body": {"representation": "storage", "value": body},
                "version": {"number": version, "message": message},
            },
        )


def escape(text):
    return html.escape(str(text))


def paragraphs(*blocks):
    return "".join("<p>%s</p>" % block for block in blocks if block) or "<p />"


def anchor(name):
    """The anchor macro that makes each row addressable as `#<name>` — the target end of
    the cross-reference devml's Models page (or anyone else) can link to."""
    return (
        '<ac:structured-macro ac:name="anchor"><ac:parameter ac:name="">%s'
        "</ac:parameter></ac:structured-macro>" % escape(name)
    )


def model_fragment(name):
    """The fragment that lands on a model's anchor on the Models page. Confluence Cloud
    has been inconsistent about `#anchor` vs `#PageTitle-anchor` — verify by hand which
    one resolves and adjust here; a link without the fragment is the acceptable fallback."""
    return "#%s" % name


def model_link(name):
    """One registered model, linked into devml's Models page by page id — never by title,
    which can change. The page itself is never written to; its Experiments column is
    devml's job."""
    url = "%s/pages/viewpage.action?pageId=%s%s" % (
        SITE,
        MODELS_PAGE,
        model_fragment(name),
    )
    return '<a href="%s"><code>%s</code></a>' % (escape(url), escape(name))


PAGE_COLUMNS = (
    ("Name", "name", 140),
    ("Description", "description", 320),
    ("Models", "models", 160),
    ("Baselines", "baselines", 130),
    ("Registered", "registered", 100),
    ("By", "by", 80),
    ("Size", "size", 80),
    ("Location", "location", 240),
)


def page_cell(record, key):
    if key == "name":
        return paragraphs(
            anchor(record.get("name") or "")
            + "<code>%s</code>" % escape(record.get("name") or "")
        )
    if key == "models":
        links = [model_link(m) for m in record.get("models") or []]
        return paragraphs(", ".join(links)) if links else paragraphs()
    if key == "baselines":
        return paragraphs(escape(", ".join(record.get("baselines") or [])))
    if key == "size":
        return paragraphs(escape(human(record.get("size"))))
    if key == "registered":
        return paragraphs(escape((record.get("registered") or "")[:10]))
    if key == "location":
        return paragraphs("<code>%s</code>" % escape(record.get("location") or ""))
    return paragraphs(escape(record.get(key) or ""))


def page_table(records):
    widths = "".join(
        '<col style="width: %d.0px;" />' % width for _, _, width in PAGE_COLUMNS
    )
    rows = [
        "<tr>%s</tr>"
        % "".join(
            "<th><p><strong>%s</strong></p></th>" % escape(h)
            for h, _, _ in PAGE_COLUMNS
        )
    ]
    for record in records:
        rows.append(
            "<tr>%s</tr>"
            % "".join(
                "<td>%s</td>" % page_cell(record, key) for _, key, _ in PAGE_COLUMNS
            )
        )
    return (
        '<table data-layout="full-width"><colgroup>%s</colgroup><tbody>%s</tbody></table>'
        % (widths, "".join(rows))
    )


def section(records, store, retired):
    """The whole of what this tool owns on the page, markers included. The line above the
    table says what it is a copy of, when it was taken and how to refresh it — otherwise
    it gets believed six months from now."""
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    counts = [
        "<strong>%d experiment%s · %s</strong>"
        % (
            len(records),
            "" if len(records) == 1 else "s",
            escape(human(sum(r.get("size") or 0 for r in records))),
        )
    ]
    if retired:
        counts.append(
            "%d name%s retired" % (len(retired), "" if len(retired) == 1 else "s")
        )
    body = [
        START,
        "<h2>%s</h2>" % escape(HEADING),
        "<p>%s</p>" % " · ".join(counts),
        "<p><em>From <code>%s</code>, generated %s by <code>evalml publish</code>. The store is "
        "the source of truth; this section is a copy of its index and is replaced whole, so "
        "anything edited here is lost.</em></p>" % (escape(store), escape(when)),
    ]
    if records:
        body.append(page_table(records))
    else:
        body.append("<p><em>Nothing is registered in the store yet.</em></p>")
    body.append(END)
    return "".join(body)


def locate(body):
    """Where on the page the generated section is: between the markers; else from our
    heading to the next heading of the same level or above (this Confluence strips HTML
    comments, so the heading is what makes the next publish a replacement rather than a
    second table); else the empty span at the end — an addition."""
    start, end = body.find(START), body.find(END)
    if start != -1 and end > start:
        return start, end + len(END), "markers"
    heading = re.search(r"<h2[^>]*>\s*%s\s*</h2>" % re.escape(HEADING), body)
    if heading:
        following = re.search(r"<h[12][^>]*>", body[heading.end() :])
        stop = heading.end() + following.start() if following else len(body)
        return heading.start(), stop, "heading"
    return len(body), len(body), "appended"


def same_table(existing, replacement):
    """Whether the section already on the page says what the new one would. Only the
    tables are compared — the line above them carries the generation time, which is no
    reason to notify everyone watching the page. Confluence stamps
    ac:macro-id/ac:schema-version onto every structured macro it stores (the anchors
    here), which would make the stored table never equal a freshly generated one —
    those attributes are stripped before comparing."""
    old = re.search(r"<table.*?</table>", existing, re.S)
    new = re.search(r"<table.*?</table>", replacement, re.S)
    if not old or not new:
        return False

    def normal(table):
        table = re.sub(r'\s+ac:(?:macro-id|schema-version)="[^"]*"', "", table)
        return html.unescape(table)

    return normal(old.group(0)) == normal(new.group(0))


def cmd_publish(
    store=STORE,
    page=EXPERIMENTS_PAGE,
    site=SITE,
    email="",
    token_file=TOKEN_FILE,
    dry_run=False,
):
    # The page describes *the* store; a test store has no business rewriting it.
    if os.path.abspath(str(store)) != os.path.abspath(str(STORE)) and not dry_run:
        fail(
            "%s is not the default store (%s) — refusing to rewrite the shared page from it"
            % (store, STORE)
        )
    index = index_path(store)
    if not index.is_file():
        fail(
            "%s does not exist — there is nothing to publish, and publishing an empty table "
            "over a good one would be worse than doing nothing.\n"
            "  Build it first: evalml list --rebuild" % index
        )
    loaded = load(store)
    records = sorted(
        loaded["experiments"], key=lambda r: r.get("registered") or "", reverse=True
    )
    for problem in loaded["broken"]:
        log("WARNING: %s line %d: %s" % (index, problem["line"], problem["problem"]))

    # The index can name an experiment the store no longer has; the store gets the final say.
    missing = [r for r in records if not Path(r.get("location") or "").is_dir()]
    for record in missing:
        log(
            "WARNING: %s is in the index but not in the store — not publishing it "
            "(--rebuild fixes the index)" % record.get("name")
        )
    records = [r for r in records if r not in missing]

    replacement = section(records, store, loaded["retired"])
    if dry_run:
        print(replacement)
        log("dry run — %d experiment(s), nothing sent to %s" % (len(records), site))
        return

    if not (page or "").strip():
        fail(
            "the Experiments page has not been created yet — create it in Confluence, then "
            "set EXPERIMENTS_PAGE in %s (or pass --page)" % __file__
        )

    email, token, source = credentials(email, token_file)
    log("authenticating as %s%s" % (email, " (%s)" % source if source else ""))
    confluence = Confluence(site, email, token)

    page = page_id(page)
    current = confluence.get(page)
    title = current["title"]
    version = current["version"]["number"]
    body = ((current.get("body") or {}).get("storage") or {}).get("value") or ""

    start, end, how = locate(body)
    if how == "appended":
        log(
            "no generated section on '%s' yet — adding one, keeping what is already there"
            % title
        )
    elif same_table(body[start:end], replacement):
        log(
            "'%s' already lists exactly these %d experiment(s) — not publishing (version %d)"
            % (title, len(records), version)
        )
        return
    updated = body[:start] + replacement + body[end:]

    result = confluence.update(
        page,
        title,
        updated,
        version + 1,
        "evalml: %d registered experiment(s)" % len(records),
    )
    published = result["version"]["number"]
    where = ((result.get("_links") or {}).get("webui") or "").strip()
    log(
        "published %d experiment(s) to '%s' (version %d): %s"
        % (
            len(records),
            title,
            published,
            site + where if where else "%s/pages/%s" % (site, page),
        )
    )

    # Read it back: Confluence rewrites what it is given, and the day the section cannot
    # be found again is the day the next publish appends a second table.
    stored = ((confluence.get(page).get("body") or {}).get("storage") or {}).get(
        "value"
    ) or ""
    _, _, found = locate(stored)
    if found == "appended":
        fail(
            "published, but the section cannot be found again on the page — neither the "
            "markers nor the '%s' heading survived. Publishing again would add a second "
            "table instead of replacing this one; fix that before running it again."
            % HEADING
        )
