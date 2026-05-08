"""Wrappers around `just video-probe` / `just video-frames`."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from strands import tool
from ._common import just_run, ok, err


@tool
def video_probe(video_path: str) -> dict:
    """Get video metadata via `just video-probe` (ffprobe JSON).

    Args:
        video_path: Path to video file.
    """
    p = Path(video_path).expanduser()
    if not p.exists():
        return err(f"video not found: {p}")

    proc = just_run("video-probe", str(p), timeout_s=30)
    if not proc.get("ok"):
        return err(f"ffprobe failed: {proc.get('stderr', '')[:200]}")

    try:
        raw = proc.get("stdout", "")
        json_start = raw.find("{")
        if json_start >= 0:
            data = json.loads(raw[json_start:])
        else:
            data = {}
        vstream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        rate = vstream.get("r_frame_rate", "0/1")
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except Exception:
            fps = 0.0
        summary = {
            "duration": float(data.get("format", {}).get("duration", 0) or 0),
            "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
            "codec": vstream.get("codec_name"),
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "fps": round(fps, 2),
            "pix_fmt": vstream.get("pix_fmt"),
            "nb_frames": vstream.get("nb_frames"),
        }
        return ok(
            f"📹 {p.name}: {summary['width']}×{summary['height']} @ "
            f"{summary['fps']}fps, {summary['duration']:.1f}s, {summary['codec']}",
            data={"summary": summary},
        )
    except Exception as e:
        return err(f"probe parse failed: {e}")


@tool
def video_extract_frames(
    video_path: str,
    output_dir: str = "",
    fps: float = 1.0,
    max_frames: int = 0,
    return_first: bool = True,
) -> dict:
    """Extract frames from video via `just video-frames`.

    Args:
        video_path: Path to input video.
        output_dir: Output dir (default: temp).
        fps: Frames/sec to extract (1.0 = every second).
        max_frames: Stop after N frames (0 = unlimited).
        return_first: Embed the first frame in the response.
    """
    p = Path(video_path).expanduser()
    if not p.exists():
        return err(f"video not found: {p}")

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="cosmos_frames_")
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    proc = just_run("video-frames", str(p), str(outp), str(fps), str(max_frames),
                    timeout_s=60 * 30)
    if not proc.get("ok"):
        return err(f"frame extraction failed: {proc.get('stderr', '')[:200]}")

    frames = sorted(outp.glob("frame_*.jpg"))
    if not frames:
        return err("no frames extracted", data={"output_dir": str(outp)})

    first_bytes = frames[0].read_bytes() if return_first else None
    return ok(
        text=f"📼 extracted {len(frames)} frame(s) → {outp}",
        data={
            "output_dir": str(outp),
            "frame_count": len(frames),
            "first_frame": str(frames[0]),
            "last_frame": str(frames[-1]),
        },
        image_bytes=first_bytes,
        image_format="jpeg",
    )
