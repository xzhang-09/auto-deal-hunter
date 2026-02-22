from agents.agent import Agent
from agents.frontier_agent import FrontierAgent
from agents.specialist_agent import SpecialistAgent


class EnsembleAgent(Agent):
    """Combines Frontier (RAG) and Specialist (fine-tuned) for price estimation."""

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        self.log("Initializing")
        self.frontier = FrontierAgent(collection)
        self.specialist = SpecialistAgent()
        self.log("Ready")

    def price(self, description: str) -> float:
        self.log("Running ensemble")
        frontier = self.frontier.price(description)
        specialist = self.specialist.price(description)
        combined = frontier * 0.8 + specialist * 0.2
        self.log(f"Ensemble: ${combined:.2f} (F:{frontier:.0f} S:{specialist:.0f})")
        return combined
