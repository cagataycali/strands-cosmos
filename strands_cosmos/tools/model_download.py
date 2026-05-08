"""Wrapper around `just download` / `just download-dataset`."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err


@tool
def cosmos_model_download(
    name: str,
    local_dir: str = "",
    kind: str = "model",
) -> dict:
    """Download a Cosmos model or dataset from HuggingFace via `just`.

    Known model shortcuts: reason2-2b, reason2-7b, reason1-7b-reward,
                           predict2.5-2b, predict2.5-14b,
                           transfer2.5-2b/edge/depth/seg
    Known dataset shortcuts: gr1, gr1-100, gr00t-eval, safe-unsafe
    Or pass a full HF repo id.

    Args:
        name: Shortcut name OR full HF repo id.
        local_dir: Where to store files (optional).
        kind: "model" | "dataset".
    """
    if kind == "model":
        proc = just_run("download", name, local_dir, timeout_s=60 * 60 * 4)
    elif kind == "dataset":
        proc = just_run("download-dataset", name, local_dir, timeout_s=60 * 60 * 4)
    else:
        return err(f"kind must be 'model' or 'dataset', got {kind!r}")

    return proc_result(
        proc,
        success_text=f"✅ downloaded {kind}={name}" + (f" → {local_dir}" if local_dir else ""),
        fail_text=f"download failed: {proc.get('stderr', '')[:200]}",
    )
