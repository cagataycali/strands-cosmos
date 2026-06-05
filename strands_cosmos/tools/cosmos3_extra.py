"""Cosmos 3 extra tools — prompt upsampling, batch captioning, VideoPhy2 eval.

Thin wrappers over justfile recipes (c3-upsample / c3-caption-batch /
c3-eval-videophy2), themselves backed by cosmos_framework scripts. These close
the upstream parity gaps for generator prompt-upsampling and task-specific
evaluation.

  - cosmos3_upsample_prompt : short scene desc -> dense structured prompt
  - cosmos3_caption_batch   : batch video captioning (VLM, reasoner-backed)
  - cosmos3_eval_videophy2  : VideoPhy-2 physical-plausibility benchmark
"""
from __future__ import annotations

from strands import tool

from ._common import just_run, proc_result

_UPSAMPLE_TIMEOUT = 60 * 10     # 10m: single LLM call (max_tokens=20000)
_CAPTION_TIMEOUT = 60 * 60      # 1h: batch over a directory of videos
_EVAL_TIMEOUT = 60 * 60 * 4     # 4h: run+eval over a val manifest


@tool
def cosmos3_upsample_prompt(
    description: str,
    task: str = "t2v",
    port: int = 8000,
    aspect: str = "16,9",
    width: int = 832,
    height: int = 480,
    fps: int = 24,
    duration: int = 8,
    image: str = "",
) -> dict:
    """Cosmos 3 prompt upsampling: expand a short scene description into a dense,
    structured generator prompt (the recommended Generator input path).

    Uses the canonical v4.2 upsampler template + Cosmos 3 sampling defaults
    (max_tokens=20000, temperature=0.7, top_p=0.8, top_k=20, presence=1.5,
    seed=3407) and queries a running reasoner server (`just c3-serve-reason`).

    Args:
        description: Short source scene description to upsample.
        task: Generation task the prompt is for — "t2v" | "t2i" | "i2v".
        port: Reasoner vLLM server port (default 8000).
        aspect: Aspect ratio in comma form (e.g. "16,9", "1,1", "9,16").
        width: Output frame width in pixels (480p default: 832).
        height: Output frame height in pixels (480p default: 480).
        fps: Target FPS (t2v/i2v only).
        duration: Clip duration in whole seconds (t2v/i2v only).
        image: Conditioning image path/URL for i2v (optional).
    """
    if task not in ("t2v", "t2i", "i2v"):
        from ._common import err
        return err("task must be one of: t2v, t2i, i2v")
    proc = just_run(
        "c3-upsample", description, task, str(port), aspect, str(width),
        str(height), str(fps), str(duration), image,
        timeout_s=_UPSAMPLE_TIMEOUT,
    )
    return proc_result(proc, "cosmos3 upsampled prompt:", "c3-upsample failed")


@tool
def cosmos3_caption_batch(
    video: str,
    out: str = "/tmp/c3_captions",
    port: int = 8000,
    workers: int = 16,
    template: str = "",
) -> dict:
    """Cosmos 3 batch video captioning via the framework VLM script.

    Captions a single video or every video in a directory, writing one caption
    file per input under `out`. Useful for SFT dataset preparation. Needs a
    running reasoner server (`just c3-serve-reason`).

    Args:
        video: Path to a single video file OR a directory of videos.
        out: Output directory for generated captions.
        port: Reasoner vLLM server port (default 8000).
        workers: Max concurrent requests to the server.
        template: Optional custom prompt-template path. Empty => auto-resolve the
            built-in video_captioner.txt (handles the upstream default-path bug).
    """
    proc = just_run(
        "c3-caption-batch", video, out, str(port), str(workers), template,
        timeout_s=_CAPTION_TIMEOUT,
    )
    return proc_result(proc, "cosmos3 batch captions -> " + out, "c3-caption-batch failed")


@tool
def cosmos3_eval_videophy2(
    results_dir: str,
    hf_ckpt: str = "",
    val_root: str = "",
    batch_size: int = 1,
    max_new_tokens: int = 256,
    nproc: int = 1,
) -> dict:
    """Cosmos 3 VideoPhy-2 evaluation (task-specific physical-plausibility benchmark).

    Two modes (mirrors upstream eval_videophy2):
      - run+eval : provide `hf_ckpt` + `val_root` — loads the HF safetensors
        export, runs batched generation over the val manifest, writes per-sample
        JSON + summary.json into `results_dir`.
      - eval-only: provide only `results_dir` (already filled by a prior run) —
        re-scores and rewrites summary.json.

    Multi-GPU data-parallel via torchrun when `nproc` > 1.

    Args:
        results_dir: Output/scan directory for per-sample JSON + summary.json.
        hf_ckpt: HF safetensors checkpoint dir (run+eval mode).
        val_root: Prepared VideoPhy-2 val manifest root (run+eval mode).
        batch_size: Per-rank generation batch size.
        max_new_tokens: Max new tokens per generation.
        nproc: GPUs for torchrun (1 = single-process on cuda:0).
    """
    proc = just_run(
        "c3-eval-videophy2", results_dir, hf_ckpt, val_root,
        str(batch_size), str(max_new_tokens), str(nproc),
        timeout_s=_EVAL_TIMEOUT,
    )
    return proc_result(
        proc, "cosmos3 videophy2 eval -> " + results_dir + "/summary.json",
        "c3-eval-videophy2 failed",
    )
