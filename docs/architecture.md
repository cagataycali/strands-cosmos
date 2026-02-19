# Architecture

How strands-cosmos is structured internally.

---

## Package Structure

```
strands_cosmos/
├── __init__.py                  # Exports: CosmosModel, CosmosVisionModel, tools
├── cosmos_model.py              # Text-only model (Strands Model interface)
├── cosmos_vision_model.py       # Vision model (video + image + text)
├── fix_cublas.py                # Jetson CUBLAS compatibility fix
└── tools/
    ├── __init__.py              # Tool exports
    ├── cosmos_invoke.py         # Text inference tool (@tool decorated)
    └── cosmos_vision_invoke.py  # Vision inference tool (@tool decorated)
```

## Model Hierarchy

```mermaid
graph TD
    SM["strands.models.Model<br/><i>Abstract base class</i>"] --> CM["CosmosModel<br/><i>Text-only</i>"]
    SM --> CVM["CosmosVisionModel<br/><i>Video + Image + Text</i>"]

    CVM --> Q["Qwen3VLForConditionalGeneration<br/><i>HuggingFace Transformers</i>"]
    CM --> Q

    Q --> GPU["🖥️ NVIDIA GPU<br/>CUDA inference"]

    style SM fill:#264653,color:#fff
    style CVM fill:#76b900,color:#fff
    style CM fill:#76b900,color:#fff
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as Strands Agent
    participant Model as CosmosVisionModel
    participant HF as Transformers
    participant GPU as CUDA

    User->>Agent: agent("caption: <video>file.mp4</video>")
    Agent->>Model: format_request(messages)
    Model->>Model: Parse <video>/<image> tags
    Model->>HF: processor(text, images, videos)
    HF->>GPU: input_ids + pixel_values
    GPU->>HF: logits (autoregressive)
    HF->>Model: generated tokens
    Model->>Agent: format_response(stream_events)
    Agent->>User: Result text
```

## Two Usage Modes

```mermaid
graph TD
    subgraph "Mode 1: As the Model"
        A1["Agent(model=CosmosVisionModel())"] --> B1["Cosmos IS the agent's brain"]
    end

    subgraph "Mode 2: As a Tool"
        A2["Agent(tools=[cosmos_vision_invoke])"] --> B2["Cosmos is a tool<br/>called by another model"]
    end

    style B1 fill:#76b900,color:#fff
    style B2 fill:#264653,color:#fff
```

## Strands Model Interface

`CosmosVisionModel` implements the full [Strands Model interface](https://strandsagents.com):

| Method | Purpose |
|--------|---------|
| `update_config()` | Merge user config |
| `get_config()` | Return current config |
| `format_request()` | Convert messages → HF inputs |
| `format_chunk()` | Stream tokens → StreamEvents |
| `format_response()` | Finalize response metadata |

## Configuration

```python
CosmosVisionModel(
    # Model selection
    model_id="nvidia/Cosmos-Reason2-2B",  # HuggingFace ID

    # GPU settings
    device_map="auto",        # GPU placement
    torch_dtype="auto",       # float16 / bfloat16

    # Vision settings
    fps=4,                    # Video frame sampling rate
    min_vision_tokens=256,    # Min visual tokens per frame
    max_vision_tokens=8192,   # Max visual tokens per frame

    # Reasoning
    reasoning=True,           # Enable <think> CoT

    # Generation
    params={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
    },
)
```
