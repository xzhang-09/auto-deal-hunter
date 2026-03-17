import os
import modal
from agents.agent import Agent


class SpecialistAgent(Agent):
    """Fine-tuned Qwen 2.5 on Modal for price prediction."""

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.log("Connecting to Modal pricer-service")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        self.log("Calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Predicted ${result:.2f}")
        return result
