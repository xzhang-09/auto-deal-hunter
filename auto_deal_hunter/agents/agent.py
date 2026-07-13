import logging

from auto_deal_hunter.infra.log_utils import BG_BLACK, BLUE, CYAN, RESET, WHITE


class Agent:
    # ANSI codes are defined once in infra.log_utils (which also maps each to a hex color for
    # the Gradio log panel, so the two can never drift apart). Re-exported here as class
    # attributes for subclasses to select via ``color = Agent.CYAN`` etc.
    BLUE = BLUE
    CYAN = CYAN
    WHITE = WHITE
    BG_BLACK = BG_BLACK
    RESET = RESET

    name: str = ""
    color: str = WHITE

    def log(self, message: str):
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)
