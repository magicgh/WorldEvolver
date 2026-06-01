
from __future__ import annotations

from common.registry import registry

from ..scienceworld import EvalScienceworld
from .word2world_s1 import Stage1Word2WorldTask


@registry.register_task("stage1_scienceworld")
@registry.register_task("stage1_scienceworld_none")
@registry.register_task("stage1_scienceworld_noisy")
@registry.register_task("stage1_scienceworld_perfect")
class ScienceworldS1(Stage1Word2WorldTask, EvalScienceworld):


    ENV_NAME = "scienceworld"
