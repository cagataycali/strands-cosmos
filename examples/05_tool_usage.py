"""Using Cosmos vision as a tool within another agent."""
from strands import Agent
from strands_cosmos import cosmos_vision_invoke

agent = Agent(tools=[cosmos_vision_invoke])

agent("Analyze this dashcam video for safety hazards: /path/to/dashcam.mp4")
