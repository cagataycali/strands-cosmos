# Examples

Runnable examples tested on NVIDIA Jetson AGX Thor (132GB unified memory).

---

## Demo Video

<a href="https://github.com/cagataycali/strands-cosmos/releases/download/v0.1.1/strands-cosmos-demo.mp4">
  <img src="/strands-cosmos/strands-cosmos-demo-preview.gif" alt="Demo — Driving analysis on Jetson AGX Thor" width="100%">
</a>

> *Click to watch the full demo video*

---

## All Examples

<div class="grid cards" markdown>

- **01 — Basic Text (Physics Reasoning)**

    <img src="/strands-cosmos/assets/videos/01_basic_text.gif" alt="Basic text inference" width="100%">

    Text-only physics reasoning — no video or image needed. ~11s on Thor.

    → [Full example + code](basic-text.md)

- **02 — Video Captioning**

    <img src="/strands-cosmos/assets/videos/02_video_caption.gif" alt="Video captioning" width="100%">

    Detailed temporal-spatial descriptions from video. ~15s on Thor.

    → [Full example + code](video-caption.md)

- **03 — Driving Analysis (CoT)**

    <img src="/strands-cosmos/assets/videos/03_driving_analysis.gif" alt="Driving analysis" width="100%">

    Dashcam safety analysis with chain-of-thought reasoning. ~16s on Thor.

    → [Full example + code](driving.md)

- **04 — Embodied Reasoning**

    <img src="/strands-cosmos/assets/videos/04_embodied_reasoning.gif" alt="Embodied reasoning" width="100%">

    Robot next-action prediction from workspace images. ~43s on Thor.

    → [Full example + code](embodied.md)

- **05 — Tool Usage**

    <img src="/strands-cosmos/assets/videos/05_tool_usage.gif" alt="Tool usage" width="100%">

    Cosmos as a callable tool inside any Strands agent. ~9s on Thor.

    → [Full example + code](tool-usage.md)

</div>

---

## Quick Reference

| # | Example | Time (Thor) | Recording |
|---|---------|-------------|-----------|
| 1 | [Basic Text](basic-text.md) | ~11s | [:material-play: cast](../assets/casts/01_basic_text.cast) |
| 2 | [Video Caption](video-caption.md) | ~15s | [:material-play: cast](../assets/casts/02_video_caption.cast) |
| 3 | [Driving Analysis](driving.md) | ~16s | [:material-play: cast](../assets/casts/03_driving_analysis.cast) |
| 4 | [Embodied Reasoning](embodied.md) | ~43s | [:material-play: cast](../assets/casts/04_embodied_reasoning.cast) |
| 5 | [Tool Usage](tool-usage.md) | ~9s | [:material-play: cast](../assets/casts/05_tool_usage.cast) |

---

## Running Locally

```bash
git clone https://github.com/cagataycali/strands-cosmos.git
cd strands-cosmos
pip install -e .

# Jetson devices: fix CUBLAS first
strands-cosmos-fix-cublas

# Run any example
python examples/01_basic_text.py
python examples/02_video_caption.py
python examples/03_driving_analysis.py
python examples/04_embodied_reasoning.py
python examples/05_tool_usage.py
```

!!! note "Sample media"
    Examples 02–05 need a `sample.mp4` (video) and/or `sample.png` (image) in the project root. Set paths via environment variables:
    ```bash
    export SAMPLE_VIDEO=/path/to/your/video.mp4
    export SAMPLE_IMAGE=/path/to/your/image.png
    ```

## Playing Terminal Recordings

All examples have asciinema `.cast` recordings:

```bash
pip install asciinema

# Play any recording
asciinema play docs/assets/casts/01_basic_text.cast
asciinema play docs/assets/casts/03_driving_analysis.cast
```

---

## Execution Flow

```mermaid
graph TD
    START["Run Example"] --> MODEL["Load Model<br/>~3s (cached)"]
    MODEL --> MEDIA{"Has media?"}
    MEDIA -->|"Video"| DECODE["Decode frames<br/>@ configured FPS"]
    MEDIA -->|"Image"| PROCESS["Process image<br/>visual tokens"]
    MEDIA -->|"Text only"| TOKENIZE["Tokenize text"]
    DECODE --> INFER["GPU Inference<br/>token-by-token streaming"]
    PROCESS --> INFER
    TOKENIZE --> INFER
    INFER --> OUTPUT["Stream output<br/>to terminal"]
    OUTPUT --> DONE["✅ PASS"]

    style MODEL fill:#264653,color:#fff
    style INFER fill:#76b900,color:#fff
    style DONE fill:#2d6a4f,color:#fff
```
