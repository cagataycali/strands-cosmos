"""Robot embodied reasoning - next action prediction."""
from strands import Agent
from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
)
agent = Agent(model=model)

agent("<image>robot_view.jpg</image> What can be the next immediate action?")
