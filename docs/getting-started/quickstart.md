# Quickstart

Get from zero to a running Cosmos agent in under 2 minutes.

---

## The Journey

```mermaid
graph LR
    A["1️⃣ Install"] --> B["2️⃣ Create Model"]
    B --> C["3️⃣ Create Agent"]
    C --> D["4️⃣ Ask Anything"]

    style A fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style B fill:#76b900,stroke:#76b900,color:#fff
    style C fill:#4a1d96,stroke:#a78bfa,color:#fff
    style D fill:#92400e,stroke:#fbbf24,color:#fff
```

## Install

```bash
pip install strands-cosmos strands-agents
```

## 1. Text-Only Physics Reasoning

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

result = agent("What happens when you push a ball off the edge of a table?")
```

## 2. Video Understanding

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    fps=4,
    params={"max_tokens": 4096},
)
agent = Agent(model=model)

# Inline video reference
agent("Caption this video in detail: <video>dashcam.mp4</video>")
```

### How It Works

```mermaid
sequenceDiagram
    participant You as 🧑 You
    participant Agent as 🤖 Strands Agent
    participant Cosmos as 🌌 Cosmos-Reason2

    You->>Agent: "Caption: <video>dashcam.mp4</video>"
    Agent->>Cosmos: Extract frames @ 4 FPS
    Cosmos->>Cosmos: Visual tokens + text tokens
    Cosmos->>Cosmos: Autoregressive generation
    Cosmos-->>Agent: Detailed scene description
    Agent-->>You: Response text
```

## 3. Image Reasoning

```python
agent("<image>robot_workspace.jpg</image> What is the robot grasping?")
```

## 4. Chain-of-Thought Reasoning

```python
model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,  # Enables <think>...</think>
)
agent = Agent(model=model)

# The model reasons step-by-step before answering
agent("<video>intersection.mp4</video> Analyze the safety situation.")
```

## 5. As a Tool (Inside Another Agent)

```python
from strands import Agent
from strands_cosmos import cosmos_vision_invoke

# Cosmos becomes a tool inside a Bedrock / OpenAI / Ollama agent
agent = Agent(tools=[cosmos_vision_invoke])
agent("Analyze this dashcam video for safety hazards: /path/to/video.mp4")
```

!!! info "Tool Usage"
    When used as a tool, Cosmos runs locally on GPU while the orchestrating agent can be any provider (Bedrock, Anthropic, OpenAI, etc.). See [Tool Usage Guide](../guide/tool-usage.md).

---

## What's Next

```mermaid
graph LR
    QS["✅ You are here:<br/>Quickstart"] --> V["🎬 Video<br/>Understanding"]
    QS --> C["🧠 Chain-of-<br/>Thought"]
    QS --> T["🔧 Tool<br/>Usage"]
    QS --> J["🔲 Jetson<br/>Deployment"]

    style QS fill:#76b900,stroke:#76b900,color:#fff
    style V fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style C fill:#4a1d96,stroke:#a78bfa,color:#fff
    style T fill:#92400e,stroke:#fbbf24,color:#fff
    style J fill:#831843,stroke:#f472b6,color:#fff
```

- [**Video Understanding**](../guide/video-understanding.md) — Process dashcam, robot, and scene videos
- [**Chain-of-Thought**](../guide/chain-of-thought.md) — Enable step-by-step reasoning
- [**Tool Usage**](../guide/tool-usage.md) — Use Cosmos inside any agent
- [**Jetson Deployment**](../guide/jetson.md) — Run on NVIDIA Jetson edge devices
