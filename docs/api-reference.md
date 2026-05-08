# API Reference

## Models

### `CosmosVisionModel`

The primary model class — supports video, image, and text input.

```python
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id: str = "nvidia/Cosmos-Reason2-2B",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    reasoning: bool = False,
    fps: int = 4,
    min_vision_tokens: int = 256,
    max_vision_tokens: int = 8192,
    params: dict = {},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `nvidia/Cosmos-Reason2-2B` | HuggingFace model ID |
| `device_map` | `str` | `auto` | GPU device placement |
| `torch_dtype` | `str` | `auto` | Tensor dtype (float16/bfloat16) |
| `reasoning` | `bool` | `False` | Enable chain-of-thought `<think>` reasoning |
| `fps` | `int` | `4` | Video frame sampling rate |
| `min_vision_tokens` | `int` | `256` | Minimum visual tokens per frame |
| `max_vision_tokens` | `int` | `8192` | Maximum visual tokens per frame |
| `params` | `dict` | `{}` | Generation params: `max_tokens`, `temperature`, `top_p` |

### `CosmosModel`

Text-only model — same interface but no vision capabilities.

```python
from strands_cosmos import CosmosModel

model = CosmosModel(model_id="nvidia/Cosmos-Reason2-2B")
```

---

## Tools

All tools are `@tool`-decorated functions compatible with any Strands Agent.

### Reason2 VLM

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_inference` | `prompt`, `image_path?`, `video_path?`, `server_url?` | Query TRT-Edge-LLM inference server |
| `cosmos_reason_hf` | `prompt`, `image_path?`, `video_path?`, `max_new_tokens?`, `model_id?` | Direct HF Transformers inference (no server needed) |
| `cosmos_serve` | `action` (`start`/`stop`/`status`) | Manage TRT-Edge-LLM server lifecycle |

### World Models

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_predict_generate` | `config_path` | Generate future video frames with Predict2.5 |
| `cosmos_transfer_generate` | `config_path` | Video-to-video with Transfer2.5 (ControlNet) |

### Model Lifecycle

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_model_download` | `name`, `local_dir?`, `kind?` | Download model from HuggingFace |
| `cosmos_quantize` | `model_dir`, `output_dir?`, `precision?` | FP8/INT8 quantization |
| `cosmos_export_onnx` | `model_dir`, `output_dir?` | Export to ONNX format |
| `cosmos_build_engine` | `onnx_dir`, `output_dir?`, `component?` | Build TRT engine (LLM or visual) |

### Training

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_post_train` | `config_path`, `method?` | Post-training (SFT, LoRA, full) |
| `cosmos_distill` | `config_path` | Knowledge distillation (8B→2B) |

### Data & Evaluation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_curate` | `config_path` | Run Xenna data curation pipeline |
| `cosmos_evaluate` | `config_path`, `metrics?` | Evaluate with FID/FVD/CSE/CLIP |

### I/O & Media

| Tool | Parameters | Description |
|------|-----------|-------------|
| `rtp_capture_frame` | `port?`, `output_path?` | Capture single frame from RTP/GStreamer stream |
| `nats_publish` | `subject`, `payload` | Publish JSON to NATS subject |
| `video_probe` | `video_path` | Get video metadata (resolution, fps, duration, codec) |
| `video_extract_frames` | `video_path`, `output_dir`, `fps?`, `max_frames?` | Extract frames as JPEGs |
| `image_read` | `image_path` | Read image as base64 string |

### System

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_sysinfo` | — | GPU info, platform, memory, CUDA version |

### Legacy (backward-compatible)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `cosmos_invoke` | `prompt`, `model_id?` | Text-only inference tool |
| `cosmos_vision_invoke` | `prompt`, `media_path?`, `model_id?` | Vision inference tool |

---

## Task Prompts

Pre-defined prompts optimized for specific tasks:

```python
from strands_cosmos.cosmos_vision_model import TASK_PROMPTS
```

| Key | Use Case |
|-----|----------|
| `caption` | Detailed video/image captioning |
| `embodied_reasoning` | Robot workspace analysis |
| `driving` | Dashcam driving safety |
| `causal` | Physical cause-and-effect |
| `temporal_localization` | Event timestamps in video |
| `2d_grounding` | Bounding box coordinates |
| `robot_cot` | Step-by-step robot planning |
| `describe_anything` | General scene description |
| `mvp_bench` | MVP benchmark evaluation |

---

## CLI

### `strands-cosmos-fix-cublas`

Fix CUBLAS compatibility on NVIDIA Jetson devices.

```bash
strands-cosmos-fix-cublas           # Auto-detect and fix
strands-cosmos-fix-cublas --check   # Check status only
strands-cosmos-fix-cublas --revert  # Restore original
```

---

## Justfile Recipes

Run `just --list` for all available recipes. Key ones:

```bash
just setup          # Clone all Cosmos ecosystem repos
just setup-full     # Full setup (apt + pip + repos + doctor)
just doctor         # Diagnose platform, tools, GPU
just install-trt-edge-llm  # Build TRT-Edge-LLM from source

just serve-start    # Start TRT inference server
just serve-stop     # Stop server
just predict-generate config.json
just transfer-generate config.json
just evaluate config.json
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COSMOS_MODEL_ID` | Default HF model | `nvidia/Cosmos-Reason2-2B` |
| `COSMOS_SERVER_URL` | TRT server endpoint | `http://127.0.0.1:8080` |
| `NATS_URL` | NATS server URL | `nats://127.0.0.1:4222` |
| `RTP_PORT` | RTP receive port | `5600` |
| `HF_TOKEN` | HuggingFace token for gated models | — |
| `COSMOS_PREDICT_REPO` | Path to cosmos-predict2.5 clone | `../cosmos-predict2.5` |
| `COSMOS_TRANSFER_REPO` | Path to cosmos-transfer2.5 clone | `../cosmos-transfer2.5` |
| `COSMOS_REASON_REPO` | Path to cosmos-reason2 clone | `../cosmos-reason2` |
| `COSMOS_XENNA_REPO` | Path to cosmos-xenna clone | `../cosmos-xenna` |
| `COSMOS_COOKBOOK_REPO` | Path to cosmos-cookbook clone | `../cosmos-cookbook` |
