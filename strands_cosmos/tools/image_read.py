"""Load an image file and embed it in the response (agent can see it).

SECURITY: ``image_path`` is LLM-controlled. It is confined to the workspace
allow-list (CWE-22) before being read, so the agent cannot exfiltrate
arbitrary files (e.g. ~/.ssh/id_rsa) into the conversation.
"""
from __future__ import annotations

from strands import tool

from ._common import err, ok
from ._security import SecurityError, resolve_in_workspace


@tool
def image_read(image_path: str) -> dict:
    """Read an image file and embed it in the response so the agent can see it.

    Supports PNG, JPEG/JPG, GIF, WebP. The path must be inside the workspace.

    Args:
        image_path: Path to image on disk (inside the workspace).
    """
    try:
        p = resolve_in_workspace(image_path, must_exist=True)
    except SecurityError as e:
        return err(str(e))

    fmt = p.suffix.lower().lstrip(".") or "png"
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "jpeg", "gif", "webp"):
        return err(f"unsupported image format: {p.suffix!r} (allowed: png/jpg/jpeg/gif/webp)")

    data = p.read_bytes()
    return ok(
        text=f"\U0001F4F7 loaded {p.name} ({len(data)} bytes, {fmt})",
        data={"path": str(p), "size": len(data), "format": fmt},
        image_bytes=data,
        image_format=fmt,
    )
