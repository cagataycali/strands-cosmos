# Examples

Runnable examples tested on Jetson AGX Thor.

---

## Overview

| # | Example | Description | Time (Thor) |
|---|---------|-------------|-------------|
| 1 | [Basic Text](https://github.com/cagataycali/strands-cosmos/blob/main/examples/01_basic_text.py) | Text-only physics reasoning | ~11s |
| 2 | [Video Caption](https://github.com/cagataycali/strands-cosmos/blob/main/examples/02_video_caption.py) | Detailed video captioning | ~15s |
| 3 | [Driving Analysis](https://github.com/cagataycali/strands-cosmos/blob/main/examples/03_driving_analysis.py) | Dashcam safety analysis with CoT | ~16s |
| 4 | [Embodied Reasoning](https://github.com/cagataycali/strands-cosmos/blob/main/examples/04_embodied_reasoning.py) | Robot next-action prediction | ~43s |
| 5 | [Tool Usage](https://github.com/cagataycali/strands-cosmos/blob/main/examples/05_tool_usage.py) | Cosmos as a tool in another agent | ~9s |

## 1. Basic Text — Physics Reasoning

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

result = agent("What happens when a ball rolls off the edge of a table?")
print(result)
```

## 2. Video Caption

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    fps=4,
    params={"max_tokens": 4096},
)
agent = Agent(model=model)

result = agent("Caption this video in detail: <video>sample.mp4</video>")
print(result)
```

## 3. Driving Analysis with Chain-of-Thought

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    fps=4,
    params={"max_tokens": 4096, "temperature": 0.6},
)
agent = Agent(model=model)

result = agent("""<video>dashcam.mp4</video>
You are an expert driving assistant. Analyze the driving scene:
1. Describe the environment
2. Identify potential hazards  
3. Recommend actions for the driver""")
print(result)
```

## 4. Embodied Reasoning (Robot Vision)

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    params={"max_tokens": 4096, "temperature": 0.6},
)
agent = Agent(model=model)

result = agent("""<image>robot_workspace.jpg</image>
This image shows a bimanual robot workspace. 
What is the immediate next action the robot should take?""")
print(result)
```

## 5. Tool Usage (Cosmos Inside Another Agent)

```python
from strands import Agent
from strands_cosmos import cosmos_vision_invoke

agent = Agent(tools=[cosmos_vision_invoke])
result = agent("Analyze this video for activity: sample.mp4")
print(result)
```

---

## Running the Examples

```bash
git clone https://github.com/cagataycali/strands-cosmos.git
cd strands-cosmos
pip install -e .

# Set sample media paths (or use defaults)
export SAMPLE_VIDEO=sample.mp4
export SAMPLE_IMAGE=sample.png

# Run any example
python examples/01_basic_text.py
python examples/02_video_caption.py
python examples/03_driving_analysis.py
python examples/04_embodied_reasoning.py
python examples/05_tool_usage.py
```
