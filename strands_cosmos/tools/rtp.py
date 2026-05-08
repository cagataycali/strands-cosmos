"""Wrapper around `just rtp-capture` — GStreamer RTP/H.264 frame capture."""
from __future__ import annotations

import tempfile
from pathlib import Path

from strands import tool
from ._common import just_run, ok, err


@tool
def rtp_capture_frame(
    bind_ip: str = "0.0.0.0",
    port: int = 5600,
    width: int = 800,
    height: int = 600,
    timeout_s: int = 5,
    output_path: str = "",
    return_image: bool = True,
) -> dict:
    """Capture one JPEG from an RTP/H.264 stream via `just rtp-capture`.

    The recipe tries Jetson HW decode (nvv4l2decoder/nvjpegenc),
    falling back to software decode automatically.

    Args:
        bind_ip: UDP bind IP.
        port: RTP UDP port.
        width / height: expected frame size.
        timeout_s: give up after N seconds.
        output_path: where to save JPEG (default: temp file).
        return_image: if True, embed JPEG bytes in the response.
    """
    if not output_path:
        output_path = tempfile.mktemp(suffix=".jpg", prefix="cosmos_rtp_")

    proc = just_run(
        "rtp-capture",
        str(port), output_path, str(width), str(height), str(timeout_s),
        timeout_s=timeout_s + 10,
        extra_env={"RTP_BIND": bind_ip},
    )

    p = Path(output_path)
    captured = p.exists() and p.stat().st_size > 0

    if not captured:
        return err(
            "no frame captured",
            data={"stderr": proc.get("stderr", "")[-600:], "cmd": proc.get("cmd")},
        )

    image_bytes = p.read_bytes() if return_image else None
    return ok(
        text=f"📸 captured {p.stat().st_size} bytes → {output_path}",
        data={"image_path": str(p), "size": p.stat().st_size,
              "width": width, "height": height},
        image_bytes=image_bytes,
        image_format="jpeg",
    )
