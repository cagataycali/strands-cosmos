# Jetson Deployment

Run Cosmos-Reason2 on NVIDIA Jetson edge devices (AGX Thor, Orin).

---

## Supported Jetson Devices

| Device | GPU Memory | Model | Status |
|--------|-----------|-------|--------|
| Jetson AGX Thor | 132 GB | 2B + 8B | ✅ |
| Jetson AGX Orin 64 | 64 GB | 2B + 8B | ✅ |
| Jetson AGX Orin 32 | 32 GB | 2B | ✅ |
| Jetson Orin NX 16 | 16 GB | ❌ | Not enough memory |

## Setup

```bash
# 1. Install
pip install strands-cosmos strands-agents

# 2. Fix CUBLAS (required for Jetson)
strands-cosmos-fix-cublas
```

## The CUBLAS Problem

PyTorch wheels bundle their own `libcublas.so` which doesn't support Jetson GPU architectures:

- **Thor:** SM 11.0 — not in pip torch's CUBLAS
- **Orin:** SM 8.7 — may not be in pip torch's CUBLAS

**Symptom:** `CUBLAS_STATUS_INVALID_VALUE` on any matrix operation.

```mermaid
graph LR
    A["pip install torch"] --> B["Bundled CUBLAS<br/>SM 7.0–9.0 only"]
    B -->|"Jetson SM 11.0"| C["❌ CUBLAS_STATUS_INVALID_VALUE"]
    D["strands-cosmos-fix-cublas"] --> E["System CUBLAS<br/>from JetPack"]
    E -->|"Jetson SM 11.0"| F["✅ Works"]

    style C fill:#9b2226,color:#fff
    style F fill:#2d6a4f,color:#fff
```

## Fix Commands

```bash
# Auto-detect and fix
strands-cosmos-fix-cublas

# Check status without fixing
strands-cosmos-fix-cublas --check

# Revert to original
strands-cosmos-fix-cublas --revert
```

### What the Fix Does

1. Backs up torch's bundled `libcublas.so` and `libcublasLt.so`
2. Copies system CUBLAS from JetPack (`/usr/local/cuda/targets/*/lib/`)
3. Verifies with a quick `torch.mm` test

!!! warning "Run after every torch upgrade"
    If you upgrade PyTorch, re-run `strands-cosmos-fix-cublas` — the new torch will overwrite the fix.

## Performance on Jetson AGX Thor

Tested benchmarks (Cosmos-Reason2-2B):

| Task | Time |
|------|------|
| Text-only physics | ~11s |
| Video caption (10s @ 4fps) | ~15s |
| Driving analysis + CoT | ~16s |
| Embodied reasoning + CoT | ~43s |
| Tool invocation | ~9s |

## Troubleshooting

```mermaid
flowchart TD
    ERR["Error on Jetson?"] --> E1{"Error message?"}
    E1 -->|"CUBLAS_STATUS_INVALID_VALUE"| FIX1["strands-cosmos-fix-cublas"]
    E1 -->|"Out of memory"| FIX2["Use 2B model<br/>Reduce max_vision_tokens"]
    E1 -->|"Model download fails"| FIX3["Check HF token:<br/>huggingface-cli login"]
    E1 -->|"Slow inference"| FIX4["Ensure GPU is in MAX power mode:<br/>sudo nvpmodel -m 0"]

    style ERR fill:#9b2226,color:#fff
    style FIX1 fill:#2d6a4f,color:#fff
    style FIX2 fill:#2d6a4f,color:#fff
    style FIX3 fill:#2d6a4f,color:#fff
    style FIX4 fill:#2d6a4f,color:#fff
```

---

## What's Next

- [**Quickstart**](../getting-started/quickstart.md) — Run your first agent
- [**Video Understanding**](video-understanding.md) — Process video on Jetson
