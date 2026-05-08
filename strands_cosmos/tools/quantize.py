"""Wrapper around `just quantize` — FP8/INT8/INT4 quantization (x86)."""
from __future__ import annotations

from strands import tool
from ._common import just_run, proc_result


@tool
def cosmos_quantize(
    model_dir: str = "nvidia/Cosmos-Reason2-2B",
    output_dir: str = "./quantized/Cosmos-Reason2-2B-fp8",
    dtype: str = "fp16",
    quantization: str = "fp8",
) -> dict:
    """Quantize a Cosmos VLM/LLM via `just quantize` (x86 GPU host only).

    Args:
        model_dir: HF model id or local path.
        output_dir: Where to write quantized weights.
        dtype: Base precision (fp16 | bf16).
        quantization: Target quantization (fp8 | int8 | int4).
    """
    proc = just_run("quantize", model_dir, output_dir, dtype, quantization, timeout_s=60 * 60 * 3)
    return proc_result(
        proc,
        success_text=f"✅ quantized {model_dir} ({dtype}→{quantization}) → {output_dir}",
        fail_text=f"quantization failed: {proc.get('stderr', '')[:200]}",
    )
