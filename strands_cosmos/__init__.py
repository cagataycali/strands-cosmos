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
from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel
from strands_cosmos.cosmos3_generator_model import Cosmos3GeneratorModel

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
    # Cosmos 3
    cosmos3_reason,
    cosmos3_caption,
    cosmos3_temporal,
    cosmos3_embodied,
    cosmos3_ground,
    cosmos3_plausibility,
    cosmos3_situation,
    cosmos3_action_cot,
    cosmos3_text2image,
    cosmos3_text2video,
    cosmos3_image2video,
    cosmos3_text2video_sound,
    cosmos3_forward_dynamics,
    cosmos3_inverse_dynamics,
    cosmos3_policy,
    cosmos3_serve,
)

__all__ = [
    # Model providers
    "CosmosModel",
    "CosmosVisionModel",
    "Cosmos3ReasonerModel",
    "Cosmos3GeneratorModel",
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
    # Cosmos 3
    "cosmos3_reason",
    "cosmos3_caption",
    "cosmos3_temporal",
    "cosmos3_embodied",
    "cosmos3_ground",
    "cosmos3_plausibility",
    "cosmos3_situation",
    "cosmos3_action_cot",
    "cosmos3_text2image",
    "cosmos3_text2video",
    "cosmos3_image2video",
    "cosmos3_text2video_sound",
    "cosmos3_forward_dynamics",
    "cosmos3_inverse_dynamics",
    "cosmos3_policy",
    "cosmos3_serve",
]
