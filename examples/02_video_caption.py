"""Video captioning with Cosmos-Reason2."""
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

agent("Caption in detail: <video>sample.mp4</video>")
