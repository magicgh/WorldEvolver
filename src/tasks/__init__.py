from .alfworld import Evalalfworld
from .new.alfworld_s1 import AlfWorldS1
from .new.alfworld_s3 import AlfWorldS3
from .new.scienceworld_s1 import ScienceworldS1
from .new.scienceworld_s3 import ScienceworldS3
from .new.word2world_s2 import AlfWorldS2, ScienceworldS2
from .scienceworld import EvalScienceworld
from common.registry import registry

__all__ = [
    "Evalalfworld",
    "EvalScienceworld",
    "AlfWorldS1",
    "ScienceworldS1",
    "AlfWorldS2",
    "ScienceworldS2",
    "AlfWorldS3",
    "ScienceworldS3",
]


def load_task(name, run_config, llm_config, agent_config, env_config, llm=None):
    cls = registry.get_task_class(name)
    if cls is None:
        raise ImportError(
            f"Task {name!r} is not registered. "
            f"Registered: {sorted((registry.mapping.get('task_name_mapping') or {}).keys())!r}."
        )
    return cls.from_config(run_config, llm_config, agent_config, env_config, llm=llm)
