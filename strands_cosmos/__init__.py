"""Strands Cosmos - NVIDIA Cosmos ecosystem provider for Strands Agents.

Full-lifecycle support: Reason2 VLM, Predict2.5, Transfer2.5, Xenna curation,
quantization, edge deployment, and evaluation.

Model Providers (use as Agent model):
  - CosmosModel: Text-only Reason2 inference via HF Transformers
  - CosmosVisionModel: Video + image + text via HF Transformers

Tools (use inside any Agent):
  - 20 tools covering the full Cosmos pipeline via justfile recipes
  - See strands_cosmos.tools for the complete list
"""

from strands_cosmos.cosmos_model import CosmosModel
from strands_cosmos.cosmos_vision_model import CosmosVisionModel

# Export all tools for convenient access
from strands_cosmos.tools import (
    # Reason2 VLM
    cosmos_inference,
    cosmos_reason_hf,
    cosmos_serve,
    # Predict2.5
    cosmos_predict_generate,
    # Transfer2.5
    cosmos_transfer_generate,
    # Model lifecycle
    cosmos_model_download,
    cosmos_quantize,
    cosmos_export_onnx,
    cosmos_build_engine,
    # Training
    cosmos_post_train,
    cosmos_distill,
    # Data
    cosmos_curate,
    # Evaluation
    cosmos_evaluate,
    # I/O
    rtp_capture_frame,
    nats_publish,
    video_probe,
    video_extract_frames,
    image_read,
    # System
    cosmos_sysinfo,
    # Legacy
    cosmos_invoke,
    cosmos_vision_invoke,
)

__all__ = [
    # Model providers
    "CosmosModel",
    "CosmosVisionModel",
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
    "cosmos_invoke",
    "cosmos_vision_invoke",
]
