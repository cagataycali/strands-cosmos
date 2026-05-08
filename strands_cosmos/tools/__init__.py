"""Strands Cosmos tools — full Cosmos ecosystem coverage via justfile recipes.

Tools are thin Python wrappers over `just <recipe>` commands.
The justfile is the single source of truth for all pipeline commands.

Tool families:
  - Reason2 (VLM): cosmos_inference, cosmos_reason_hf, cosmos_serve
  - Predict2.5 (world model): cosmos_predict_generate
  - Transfer2.5 (ControlNet): cosmos_transfer_generate
  - Model lifecycle: cosmos_model_download, cosmos_quantize, cosmos_export_onnx, cosmos_build_engine
  - Training: cosmos_post_train, cosmos_distill
  - Data: cosmos_curate
  - Evaluation: cosmos_evaluate
  - I/O: rtp_capture_frame, nats_publish, video_probe, video_extract_frames, image_read
  - System: cosmos_sysinfo
  - Legacy (direct HF): cosmos_invoke, cosmos_vision_invoke
"""

# ── Reason2 VLM ──────────────────────────────────────────────────────────
from strands_cosmos.tools.inference import cosmos_inference
from strands_cosmos.tools.reason_hf import cosmos_reason_hf
from strands_cosmos.tools.serve import cosmos_serve

# ── Predict2.5 (world model) ─────────────────────────────────────────────
from strands_cosmos.tools.predict_generate import cosmos_predict_generate

# ── Transfer2.5 (ControlNet) ─────────────────────────────────────────────
from strands_cosmos.tools.transfer_generate import cosmos_transfer_generate

# ── Model lifecycle ───────────────────────────────────────────────────────
from strands_cosmos.tools.model_download import cosmos_model_download
from strands_cosmos.tools.quantize import cosmos_quantize
from strands_cosmos.tools.export_onnx import cosmos_export_onnx
from strands_cosmos.tools.build_engine import cosmos_build_engine

# ── Training ──────────────────────────────────────────────────────────────
from strands_cosmos.tools.post_train import cosmos_post_train
from strands_cosmos.tools.distill import cosmos_distill

# ── Data curation ─────────────────────────────────────────────────────────
from strands_cosmos.tools.curate import cosmos_curate

# ── Evaluation ────────────────────────────────────────────────────────────
from strands_cosmos.tools.evaluate import cosmos_evaluate

# ── I/O + utilities ───────────────────────────────────────────────────────
from strands_cosmos.tools.rtp import rtp_capture_frame
from strands_cosmos.tools.nats_pub import nats_publish
from strands_cosmos.tools.video_utils import video_probe, video_extract_frames
from strands_cosmos.tools.image_read import image_read

# ── System ────────────────────────────────────────────────────────────────
from strands_cosmos.tools.sysinfo import cosmos_sysinfo

# ── Legacy (direct HF inference, kept for backward compat) ───────────────
from strands_cosmos.tools.cosmos_invoke import cosmos_invoke
from strands_cosmos.tools.cosmos_vision_invoke import cosmos_vision_invoke

__all__ = [
    # Reason2 VLM
    "cosmos_inference",
    "cosmos_reason_hf",
    "cosmos_serve",
    # Predict2.5
    "cosmos_predict_generate",
    # Transfer2.5
    "cosmos_transfer_generate",
    # Model lifecycle
    "cosmos_model_download",
    "cosmos_quantize",
    "cosmos_export_onnx",
    "cosmos_build_engine",
    # Training
    "cosmos_post_train",
    "cosmos_distill",
    # Data
    "cosmos_curate",
    # Evaluation
    "cosmos_evaluate",
    # I/O
    "rtp_capture_frame",
    "nats_publish",
    "video_probe",
    "video_extract_frames",
    "image_read",
    # System
    "cosmos_sysinfo",
    # Legacy
    "cosmos_invoke",
    "cosmos_vision_invoke",
]
