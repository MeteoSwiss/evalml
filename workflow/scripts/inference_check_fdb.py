"""Check whether inference output for a given init_time already exists in FDB.

Each FDB root passed on the command line is assumed dedicated to a single
checkpoint (the caller is responsible for pointing --fdb-root at an
env_id-keyed directory, e.g. '{fdb_root_global}/{env_id}' or
'{local_base}/{env_id}'), so no cross-checkpoint MARS-key disambiguation
(class/expver/model) is needed here.

A single init_time's output can still be split into several distinct
*blocks* within that one root -- e.g. an ICON-grid target and an IFS-grid
target from the same run's `output.tee`, which go through different
post-processing and so don't necessarily share the same param set. A block
is identified by its combined key minus {step, param, levelist, timespan}
(i.e. by whatever class/stream/type/etc. distinguishes it). A "hit" requires:

  - at least `expected_blocks` distinct, complete blocks are present;
  - each block holds every requested lead time (from --steps);
  - each block's params are consistent across steps: every step other than
    the first has exactly the same param set (the union across those steps),
    and the first step's params are a subset of it. This tolerates fields
    that are legitimately absent only at the initial step (e.g. accumulated
    params with no meaningful value at lead time 0) without needing to name
    them, while still catching a run that crashed partway through writing a
    step's fields (which would otherwise still show up as "present").

Roots are checked in the order given; the first one holding a complete set
wins. Writes "hit\t<root>" or "miss\t" to the output file.

Intended to run inside a squashfs-mounted venv that provides pyfdb and the
native FDB C library (libfdb5.so). The squashfs is the same one built by
inference_create_venv / inference_make_squashfs_image.
"""
import argparse
import sys
from pathlib import Path


def _make_fdb_config(fdb_root: str, fdb_schema: str) -> dict:
    return {
        "type": "local",
        "engine": "toc",
        "schema": fdb_schema,
        "spaces": [{"handler": "Default", "roots": [{"path": fdb_root}]}],
    }


def _parse_steps(steps_spec: str) -> set[int]:
    """Parse a 'start/end/step' lead-time spec (hours) into the expected step set."""
    start, end, step = map(int, steps_spec.split("/"))
    return set(range(start, end + 1, step))


def _group_by_block(entries) -> dict[tuple, dict[int, set]]:
    """Group list() entries into {block_key: {step: {(levtype, param), ...}}}.

    block_key is the combined key with step/param/levelist/timespan/levtype
    removed -- i.e. whatever identifies a distinct *output.tee target*
    (class/stream/type/...) sharing this date+time. levtype is deliberately
    NOT part of block_key: one target (e.g. an ICON-grid `fdb:` entry) writes
    both 'sfc' and 'pl' levtypes, and those must be checked together as a
    single block -- otherwise a run that crashed after writing only 'sfc'
    would still count as its own separate, "complete" block, undermining the
    --expected-blocks check. levtype is instead folded into each step's
    param-identity tuple so sfc/pl params are never conflated with each other.
    """
    blocks: dict[tuple, dict[int, set]] = {}
    for el in entries:
        key = el.combined_key()
        if "step" not in key or "param" not in key:
            continue
        block_key = tuple(
            sorted(
                (k, v)
                for k, v in key.items()
                if k not in ("step", "param", "levelist", "timespan", "levtype")
            )
        )
        item = (key.get("levtype"), key["param"])
        blocks.setdefault(block_key, {}).setdefault(int(key["step"]), set()).add(item)
    return blocks


def _block_is_complete(step_params: dict[int, set], expected_steps: set[int]) -> bool:
    found_steps = set(step_params)
    if expected_steps - found_steps:
        return False

    first_step = min(expected_steps)
    other_steps = expected_steps - {first_step}
    if not other_steps:
        return True  # single-step run: nothing to cross-check against

    reference = set()
    for s in other_steps:
        reference |= step_params[s]
    if not reference:
        return False  # no params at all on the non-first steps

    for s in other_steps:
        if step_params[s] != reference:
            return False
    return step_params[first_step] <= reference


def check_fdb(
    fdb_root: str,
    fdb_schema: str,
    date: str,
    time: str,
    steps_spec: str,
    expected_blocks: int,
) -> bool:
    """Return True if `fdb_root` holds a complete output for this init time."""
    import pyfdb

    if not Path(fdb_root).is_dir():
        return False

    expected_steps = _parse_steps(steps_spec)

    # Pass the config as a dict directly. pyfdb.FDB(...) treats a `str` argument as
    # inline YAML *content* to parse, not a file path (only a `Path` object is read
    # as a file) -- passing str(config_path) here silently mis-parses and falls back
    # to FDB5's built-in default config, which doesn't exist in this venv and fails
    # with "Cannot open .../fdb5lib/etc/fdb/schema". A dict sidesteps the ambiguity
    # entirely and needs no temp file.
    with pyfdb.FDB(_make_fdb_config(fdb_root, fdb_schema)) as fdb:
        # level=3 reaches the index level, where 'step'/'param' are part of the
        # combined key (schema: [date,time,...,type[levtype,number?
        # [step,param,levelist?,timespan?]]]).
        entries = fdb.list({"date": date, "time": time}, level=3)
        blocks = _group_by_block(entries)

    complete_blocks = [
        step_params
        for step_params in blocks.values()
        if _block_is_complete(step_params, expected_steps)
    ]
    if len(complete_blocks) < expected_blocks:
        print(
            f"FDB check: {fdb_root} has {len(complete_blocks)}/{expected_blocks} "
            f"complete block(s) for date={date} time={time}"
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check if FDB already holds a complete output for a given init_time."
    )
    parser.add_argument(
        "--fdb-root",
        required=True,
        action="append",
        dest="fdb_roots",
        help="Root path of an FDB store to check. Repeatable; checked in order, "
        "first complete match wins.",
    )
    parser.add_argument("--fdb-schema", required=True, help="Path to the FDB5 schema file.")
    parser.add_argument("--date", required=True, help="Initialisation date (YYYYMMDD).")
    parser.add_argument("--time", required=True, help="Initialisation time (HHMM).")
    parser.add_argument(
        "--steps", required=True, help="Expected lead times as 'start/end/step' (hours)."
    )
    parser.add_argument(
        "--expected-blocks",
        required=True,
        type=int,
        help="Number of distinct fdb: output targets this run's config declares.",
    )
    parser.add_argument("--output", required=True, help="File to write the result to.")
    args = parser.parse_args()

    hit_root = ""
    for root in args.fdb_roots:
        if check_fdb(
            root, args.fdb_schema, args.date, args.time, args.steps, args.expected_blocks
        ):
            hit_root = root
            break
    status = "hit" if hit_root else "miss"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{status}\t{hit_root}")

    print(
        f"FDB check: date={args.date} time={args.time} steps={args.steps} "
        f"-> {status}" + (f" ({hit_root})" if hit_root else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
