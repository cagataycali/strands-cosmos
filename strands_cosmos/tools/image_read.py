"""Load an image file and embed it in the response (agent can see it)."""
from __future__ import annotations

from pathlib import Path

from strands import tool
from ._common import ok, err


@tool
def image_read(image_path: str) -> dict:
    """Read an image file and embed it in the response so the agent can see it.

    Supports PNG, JPEG/JPG, GIF, WebP.

    Args:
        image_path: Path to image on disk.
    """
    p = Path(image_path).expanduser()
    if not p.exists():
        return err(f"image not found: {p}")

    fmt = p.suffix.lower().lstrip(".") or "png"
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "jpeg", "gif", "webp"):
        fmt = "png"

    data = p.read_bytes()
    return ok(
        text=f"📷 loaded {p.name} ({len(data)} bytes, {fmt})",
        data={"path": str(p), "size": len(data), "format": fmt},
        image_bytes=data,
        image_format=fmt,
    )
