"""Wrapper around `just export-llm` / `just export-visual` (x86)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result, err


@tool
def cosmos_export_onnx(
    model_dir: str,
    output_dir: str,
    which_part: str = "llm",
    dtype: str = "fp16",
    quantization: str = "",
) -> dict:
    """Export a Cosmos model component to ONNX via just recipes.

    Args:
        model_dir: Path to model (quantized for llm, HF original for visual).
        output_dir: Destination for .onnx files.
        which_part: "llm" | "visual".
        dtype: Base dtype (visual only).
        quantization: e.g. "fp8" (visual only).
    """
    if which_part == "llm":
        proc = just_run("export-llm", model_dir, output_dir, timeout_s=60 * 60 * 2)
    elif which_part == "visual":
        proc = just_run("export-visual", model_dir, output_dir, dtype, quantization,
                        timeout_s=60 * 60 * 2)
    else:
        return err(f"which_part must be 'llm' or 'visual', got {which_part!r}")

    return proc_result(
        proc,
        success_text=f"✅ ONNX ({which_part}) exported → {output_dir}",
        fail_text=f"ONNX export failed: {proc.get('stderr', '')[:200]}",
    )
