import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Compare GRIB output from two evalml runs")
    parser.add_argument("path_a", nargs="?", default=None, help="First GRIB directory")
    parser.add_argument("path_b", nargs="?", default=None, help="Second GRIB directory")
    args, _ = parser.parse_known_args()

    path_a_input = mo.ui.text(
        value=args.path_a if args.path_a else "",
        label="Path A",
        full_width=True,
    )
    path_b_input = mo.ui.text(
        value=args.path_b if args.path_b else "",
        label="Path B",
        full_width=True,
    )
    mo.vstack([path_a_input, path_b_input])
    return args, path_a_input, path_b_input, parser, sys


@app.cell
def _(mo, path_a_input, path_b_input):
    from pathlib import Path

    path_a = Path(path_a_input.value)
    path_b = Path(path_b_input.value)

    if not path_a_input.value or not path_b_input.value:
        mo.stop(True, mo.callout(mo.md("**Enter both directory paths above.**"), kind="warn"))
    if not path_a.is_dir():
        mo.stop(True, mo.callout(mo.md(f"**Path A does not exist:** `{path_a}`"), kind="danger"))
    if not path_b.is_dir():
        mo.stop(True, mo.callout(mo.md(f"**Path B does not exist:** `{path_b}`"), kind="danger"))

    return Path, path_a, path_b


@app.cell
def _(Path, mo, path_a, path_b):
    import eccodes
    import numpy as np

    TOLERANCE = 1e-6

    def read_grib(path: Path) -> dict:
        """Return {(shortName, stepRange, level): np.ndarray} for all messages in a GRIB file."""
        messages = {}
        with open(path, "rb") as f:
            while True:
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    break
                key = (
                    eccodes.codes_get(msg, "shortName", ktype=str),
                    eccodes.codes_get(msg, "stepRange", ktype=str),
                    eccodes.codes_get(msg, "level"),
                )
                messages[key] = eccodes.codes_get_values(msg).copy()
                eccodes.codes_release(msg)
        return messages

    files_a = {f.name for f in path_a.glob("*.grib")}
    files_b = {f.name for f in path_b.glob("*.grib")}
    common = sorted(files_a & files_b)
    only_a = sorted(files_a - files_b)
    only_b = sorted(files_b - files_a)

    rows = []
    for fname in common:
        msgs_a = read_grib(path_a / fname)
        msgs_b = read_grib(path_b / fname)
        all_keys = sorted(set(msgs_a) | set(msgs_b))
        for key in all_keys:
            short_name, step, level = key
            if key not in msgs_a:
                rows.append({"file": fname, "field": short_name, "step": step, "level": level,
                             "max_diff": None, "mean_diff": None, "n_exceed": None, "status": "⚠ only in B"})
                continue
            if key not in msgs_b:
                rows.append({"file": fname, "field": short_name, "step": step, "level": level,
                             "max_diff": None, "mean_diff": None, "n_exceed": None, "status": "⚠ only in A"})
                continue
            diff = np.abs(msgs_a[key] - msgs_b[key])
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            n_exceed = int((diff > TOLERANCE).sum())
            status = "✓ pass" if max_diff <= TOLERANCE else f"✗ fail ({n_exceed} pts)"
            rows.append({"file": fname, "field": short_name, "step": step, "level": level,
                         "max_diff": max_diff, "mean_diff": mean_diff, "n_exceed": n_exceed, "status": status})

    n_pass = sum(1 for r in rows if r["status"].startswith("✓"))
    n_fail = sum(1 for r in rows if r["status"].startswith("✗"))
    n_warn = sum(1 for r in rows if r["status"].startswith("⚠"))

    return (
        TOLERANCE,
        common,
        eccodes,
        fname,
        files_a,
        files_b,
        msgs_a,
        msgs_b,
        n_fail,
        n_pass,
        n_warn,
        np,
        only_a,
        only_b,
        read_grib,
        rows,
    )


@app.cell
def _(mo, n_fail, n_pass, n_warn, only_a, only_b, path_a, path_b):
    summary_kind = "success" if n_fail == 0 else "danger"
    summary_lines = [
        f"**Path A:** `{path_a}`",
        f"**Path B:** `{path_b}`",
        f"**Fields compared:** {n_pass + n_fail + n_warn} &nbsp;|&nbsp; "
        f"✓ pass: {n_pass} &nbsp;|&nbsp; ✗ fail: {n_fail}",
    ]
    if n_warn:
        summary_lines.append(f"⚠ fields present in only one path: {n_warn}")
    if only_a:
        summary_lines.append(f"Files only in A: {', '.join(only_a)}")
    if only_b:
        summary_lines.append(f"Files only in B: {', '.join(only_b)}")

    mo.callout(mo.md("\n\n".join(summary_lines)), kind=summary_kind)
    return (summary_kind, summary_lines)


@app.cell
def _(mo, rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    if not df.empty:
        df["max_diff"] = df["max_diff"].map(lambda x: f"{x:.3e}" if x is not None else "—")
        df["mean_diff"] = df["mean_diff"].map(lambda x: f"{x:.3e}" if x is not None else "—")
        df["n_exceed"] = df["n_exceed"].map(lambda x: str(x) if x is not None else "—")

    mo.ui.table(df, label="All fields", selection=None)
    return (df, pd)


@app.cell
def _(mo, n_fail, pd, rows):
    failures = [r for r in rows if r["status"].startswith("✗")]
    if not failures:
        mo.stop(True, mo.callout(mo.md("All fields pass — no failures to show."), kind="success"))

    _df_fail = pd.DataFrame(failures)
    _df_fail["max_diff"] = _df_fail["max_diff"].map(lambda x: f"{x:.6e}")
    _df_fail["mean_diff"] = _df_fail["mean_diff"].map(lambda x: f"{x:.6e}")

    mo.vstack([
        mo.callout(mo.md(f"**{n_fail} field(s) exceed tolerance of 1e-6:**"), kind="danger"),
        mo.ui.table(_df_fail, label="Failures", selection=None),
    ])
    return (failures,)


if __name__ == "__main__":
    app.run()
