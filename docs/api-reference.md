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

### `cosmos_vision_invoke`

Strands tool for vision inference (video + image + text).

```python
from strands_cosmos import cosmos_vision_invoke
```

**Tool Parameters (called by the agent):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | `str` | ✅ | Question about the media |
| `media_path` | `str` | ❌ | Path to video or image file |
| `model_id` | `str` | ❌ | Override model ID |

### `cosmos_invoke`

Strands tool for text-only inference.

```python
from strands_cosmos import cosmos_invoke
```

**Tool Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | `str` | ✅ | Text prompt for reasoning |
| `model_id` | `str` | ❌ | Override model ID |

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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SAMPLE_VIDEO` | Default video path for examples | `sample.mp4` |
| `SAMPLE_IMAGE` | Default image path for examples | `sample.png` |
| `HF_TOKEN` | HuggingFace token for gated models | — |
