"""Convert a video file to an animated GIF.

Uses ffmpeg's two-pass palette pipeline for high-quality output (generates a
256-color palette tuned to the input, then maps frames against it). Requires
``ffmpeg`` on PATH (``brew install ffmpeg`` on macOS).

Examples
--------
    # Basic: defaults to 15 fps, 480px wide, full duration
    python -m scripts.video_to_gif input.mp4 output.gif

    # Trim a clip from 2s for 4s, 640px wide, 24 fps
    python -m scripts.video_to_gif input.mp4 output.gif \\
        --start 2 --duration 4 --fps 24 --width 640

    # Keep native size and frame rate
    python -m scripts.video_to_gif input.mp4 output.gif --fps 0 --width 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    """Run an ffmpeg command and surface failures clearly."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {' '.join(cmd)}")


def _build_filter(fps: int, width: int) -> str:
    """Compose the fps/scale filter chain (skips zero-valued stages)."""
    parts: list[str] = []
    if fps > 0:
        parts.append(f"fps={fps}")
    if width > 0:
        # -1 preserves aspect ratio; lanczos gives crisp downscales
        parts.append(f"scale={width}:-1:flags=lanczos")
    return ",".join(parts) if parts else "null"


def video_to_gif(
    src: Path,
    dst: Path,
    fps: int = 15,
    width: int = 480,
    start: float | None = None,
    duration: float | None = None,
    loop: int = 0,
) -> Path:
    """Convert ``src`` video to ``dst`` GIF.

    Parameters
    ----------
    src, dst : Path
        Input video and output gif paths.
    fps : int
        Output frame rate. ``0`` keeps the source rate.
    width : int
        Output width in pixels (height auto). ``0`` keeps source size.
    start, duration : float | None
        Optional trim window, in seconds.
    loop : int
        ``0`` = loop forever (default), ``-1`` = play once.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`).")
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    flt = _build_filter(fps, width)
    trim: list[str] = []
    if start is not None:
        trim += ["-ss", str(start)]
    if duration is not None:
        trim += ["-t", str(duration)]

    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"

        # Pass 1: derive an optimised 256-colour palette from the input frames.
        _run([
            "ffmpeg", "-y",
            *trim,
            "-i", str(src),
            "-vf", f"{flt},palettegen=stats_mode=diff",
            str(palette),
        ])

        # Pass 2: encode the gif using that palette + Floyd-Steinberg dithering.
        _run([
            "ffmpeg", "-y",
            *trim,
            "-i", str(src),
            "-i", str(palette),
            "-lavfi", f"{flt} [x]; [x][1:v] paletteuse=dither=floyd_steinberg",
            "-loop", str(loop),
            str(dst),
        ])

    return dst


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a video file to an animated GIF.")
    p.add_argument("input", type=Path, help="Input video (mp4, mov, webm, ...).")
    p.add_argument("output", type=Path, help="Output gif path.")
    p.add_argument("--fps", type=int, default=15, help="Output frame rate (0 = keep source). Default: 15")
    p.add_argument("--width", type=int, default=480, help="Output width in px (0 = keep source). Default: 480")
    p.add_argument("--start", type=float, default=None, help="Trim start time (seconds).")
    p.add_argument("--duration", type=float, default=None, help="Trim duration (seconds).")
    p.add_argument("--loop", type=int, default=0, help="0 = infinite (default); -1 = play once.")
    args = p.parse_args()

    out = video_to_gif(
        src=args.input,
        dst=args.output,
        fps=args.fps,
        width=args.width,
        start=args.start,
        duration=args.duration,
        loop=args.loop,
    )
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
