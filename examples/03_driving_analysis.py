"""Autonomous driving analysis with chain-of-thought reasoning."""
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    fps=4,
)
agent = Agent(model=model)

agent("<video>dashcam.mp4</video> Identify safety hazards and recommend actions.")
