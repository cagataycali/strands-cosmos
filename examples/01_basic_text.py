"""Basic text inference with Cosmos-Reason2."""
from strands import Agent
from strands_cosmos import CosmosModel

model = CosmosModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

agent("Explain the physics of a ball rolling down a ramp.")
