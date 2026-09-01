
from __future__ import annotations

from agents.foresight import ForesightMixin
from agents.react_agent import ReactAgent
from common.registry import registry


@registry.register_agent("WMReactAgent")
class WMReactAgent(ForesightMixin, ReactAgent):


    PREDICTION_TAG: str = "Foresight"


    FORESIGHT_INTRO: str = (
        "When 'Foresight' appears in your memory, it contains "
        "world model predictions of one or more future observations. Use it to guide "
        "your next action."
    )
