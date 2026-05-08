# Installation

## Requirements

- Python ≥ 3.10
- NVIDIA GPU (24 GB+ for 2B model, 32 GB+ for 8B model)
- CUDA 12.x

## Install

```bash
pip install strands-cosmos
```

!!! tip "zsh users"
    If you see `zsh: no matches found`, quote the package name: `pip install "strands-cosmos"`

## Platform Compatibility

| Platform | GPU | Status |
|----------|-----|--------|
| Desktop Linux x86_64 | A100 / H100 / RTX 4090 | ✅ |
| Jetson AGX Thor | Thor 132 GB | ✅ (with CUBLAS fix) |
| Jetson Orin | Orin 32/64 GB | ✅ (may need CUBLAS fix) |
| macOS (Apple Silicon) | ❌ | No CUDA — use [strands-mlx](https://github.com/cagataycali/strands-mlx) |

## Jetson Setup

On NVIDIA Jetson devices, PyTorch's pip-bundled CUBLAS may not support the GPU architecture. Run the included fix after install:

```bash
# Fix CUBLAS (auto-detects if needed, safe on any platform)
strands-cosmos-fix-cublas

# Or check without fixing:
strands-cosmos-fix-cublas --check

# Revert if needed:
strands-cosmos-fix-cublas --revert
```

→ See [Jetson Deployment Guide](../guide/jetson.md) for details.

## Verify

```python
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
print("✅ Model loaded successfully")
```

!!! note "First run"
    The first run downloads the model from HuggingFace (~5 GB for 2B). Subsequent runs load from cache.

## What Gets Installed

```mermaid
graph LR
    COSMOS["strands-cosmos"] --> SA["strands-agents<br/><i>Agent framework</i>"]
    COSMOS --> HF["transformers<br/><i>Model loading</i>"]
    COSMOS --> TORCH["torch<br/><i>GPU inference</i>"]
    COSMOS --> TV["torchvision<br/><i>Video decoding</i>"]

    style COSMOS fill:#76b900,color:#fff
    style SA fill:#264653,color:#fff
```

## What's Next

- [**Quickstart**](quickstart.md) — Your first Cosmos agent in 5 lines
- [**Video Understanding**](../guide/video-understanding.md) — Process dashcam, robot, and scene videos
- [**Jetson Deployment**](../guide/jetson.md) — Run on edge hardware
