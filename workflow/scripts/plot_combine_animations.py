"""Combine two GIF animations side by side, synced in simulated time.

Each GIF may have a different time step (e.g. 6h vs 1h). The output plays at
the finest resolution, holding frames from the coarser GIF steady while the
finer one advances. Both panels stay in sync with respect to simulated time.

Usage
-----
    python plot_combine_animations.py \\
        --left  left.gif  --left_step  6 \\
        --right right.gif --right_step 1 \\
        --output comparison.gif \\
        [--speed 6]          # simulated hours per second (default: 6)
        [--total_hours 120]  # total simulated hours covered (default: 120)
        [--start_hour 1]     # first simulated hour in the GIFs (default: min step)
"""

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


def load_frames(path: str) -> list[Image.Image]:
    im = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(im):
        frames.append(frame.convert("RGBA"))
    return frames


def frame_for_hour(frames: list[Image.Image], step: int, hour: int) -> Image.Image:
    """Return the frame that is valid at the given simulated hour.

    Frames are assumed to start at ``step`` (i.e. frames[0] covers hour=step,
    frames[1] covers hour=2*step, etc.).  Hours before the first frame return
    frames[0]; hours beyond the last frame return the last frame.
    """
    idx = max(0, min(len(frames) - 1, (hour - 1) // step))
    return frames[idx]


def combine(
    left_path: str,
    right_path: str,
    out_path: str,
    left_step: int,
    right_step: int,
    speed: float = 6.0,
    total_hours: int = 120,
    start_hour: int | None = None,
) -> None:
    left_frames = load_frames(left_path)
    right_frames = load_frames(right_path)

    out_step = min(left_step, right_step)
    if start_hour is None:
        start_hour = out_step

    sim_hours = list(range(start_hour, total_hours + 1, out_step))
    delay_ms = round(out_step / speed * 1000)

    w = left_frames[0].width + right_frames[0].width
    h = max(left_frames[0].height, right_frames[0].height)

    out_frames = []
    for sh in sim_hours:
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        lf = frame_for_hour(left_frames, left_step, sh)
        rf = frame_for_hour(right_frames, right_step, sh)
        canvas.paste(lf, (0, 0))
        canvas.paste(rf, (left_frames[0].width, 0))
        out_frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))

    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        loop=0,
        duration=delay_ms,
        optimize=False,
    )
    print(
        f"Written {len(out_frames)} frames "
        f"({out_step}h steps, {delay_ms}ms/frame, "
        f"{speed} sim-h/s) → {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine two GIF animations side by side, synced in simulated time."
    )
    parser.add_argument("--left", required=True, help="Path to the left GIF")
    parser.add_argument("--right", required=True, help="Path to the right GIF")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument(
        "--left_step", type=int, required=True, help="Time step of the left GIF (h)"
    )
    parser.add_argument(
        "--right_step", type=int, required=True, help="Time step of the right GIF (h)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=6.0,
        help="Animation speed in simulated hours per second (default: 6)",
    )
    parser.add_argument(
        "--total_hours",
        type=int,
        default=120,
        help="Total simulated hours covered by the GIFs (default: 120)",
    )
    parser.add_argument(
        "--start_hour",
        type=int,
        default=None,
        help="First simulated hour in the GIFs (default: min step size)",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combine(
        left_path=args.left,
        right_path=args.right,
        out_path=args.output,
        left_step=args.left_step,
        right_step=args.right_step,
        speed=args.speed,
        total_hours=args.total_hours,
        start_hour=args.start_hour,
    )


if __name__ == "__main__":
    main()
