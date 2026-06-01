from .react_agent import ReactAgent
from .wm_react_agent import WMReactAgent
from .wm_reflact_agent import WMReflactAgent
from common.registry import registry

__all__ = [
    "ReactAgent",
    "WMReactAgent", "WMReflactAgent",
]


def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    return agent
