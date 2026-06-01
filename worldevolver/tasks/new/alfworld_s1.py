
from __future__ import annotations

from common.registry import registry

from ..alfworld import Evalalfworld
from .word2world_s1 import Stage1Word2WorldTask


@registry.register_task("stage1_alfworld")
@registry.register_task("stage1_alfworld_none")
@registry.register_task("stage1_alfworld_noisy")
@registry.register_task("stage1_alfworld_perfect")
class AlfWorldS1(Stage1Word2WorldTask, Evalalfworld):


    ENV_NAME = "alfworld"
