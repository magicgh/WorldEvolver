
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from agents.foresight import ForesightMixin
from agents.react_agent import ReactAgent
from common.registry import registry


_ACTION_RE = re.compile(r"(?:^|\n|\.[ \t]+)Action: ")


REFLACT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "ReflActAgent"
REFLACT_PROMPT_PATH_BY_ENV = {
    "alfworld": REFLACT_PROMPT_DIR / "alfworld_reflact.json",
    "scienceworld": REFLACT_PROMPT_DIR / "scienceworld_reflact.json",
}


def _normalize_env(env_name: str) -> str:
    key = (env_name or "").lower()
    if key in {"sciworld", "science_world"}:
        key = "scienceworld"
    if key not in REFLACT_PROMPT_PATH_BY_ENV:
        raise KeyError(
            f"No ReflAct prompt for env_name={env_name!r}. "
            f"Known envs: {sorted(REFLACT_PROMPT_PATH_BY_ENV)!r}"
        )
    return key


def reflact_prompt_path(env_name: str) -> Path:

    return REFLACT_PROMPT_PATH_BY_ENV[_normalize_env(env_name)]


def load_reflact_prompt(env_name: str) -> Dict[str, object]:

    path = reflact_prompt_path(env_name)
    with path.open("r", encoding="utf-8") as f:
        prompt = json.load(f)
    if "system_msg" not in prompt:
        raise ValueError(f"{path} is missing system_msg")
    examples = prompt.get("examples")
    if not isinstance(examples, (dict, str)):
        raise ValueError(f"{path} examples must be a dict or string")
    example_blocks = examples.values() if isinstance(examples, dict) else [examples]
    for block in example_blocks:
        text = "".join(block) if isinstance(block, list) else str(block)
        if "Reflection:" not in text or "Action:" not in text:
            raise ValueError(f"{path} examples must contain Reflection and Action tags")
        if text.count("Reflection:") != text.count("Action:"):
            raise ValueError(
                f"{path} examples must insert one Reflection before each Action"
            )
    return prompt


@registry.register_agent("WMReflactAgent")
class WMReflactAgent(ForesightMixin, ReactAgent):


    PREDICTION_TAG: str = "Foresight"
    REASONING_TAG: str = "Reflection"
    FORESIGHT_INTRO: str = (
        "When 'Foresight' appears in your memory, it contains "
        "world model predictions of one or more future observations. Use it to guide "
        "your next action."
    )

    def __init__(
        self,
        llm_model,
        memory_size: int = 100,
        examples=None,
        instruction: str = "",
        init_prompt_path: Optional[str] = None,
        system_message: str = "You are a helpful assistant.",
        need_goal: bool = False,
        check_actions=None,
        check_inventory=None,
        use_parser: bool = True,
        max_think_iters: int = 3,
    ) -> None:
        super().__init__(
            llm_model,
            memory_size,
            examples or [],
            instruction,
            init_prompt_path,
            system_message,
            need_goal,
            check_actions,
            check_inventory,
            use_parser,
            max_think_iters,
        )


    def extract_reflect(self, response: str) -> str:

        if response.startswith("Reflection:"):
            text = response.split("Reflection:", 1)[-1].strip()
            action_match = _ACTION_RE.search(text)
            if action_match is not None:
                text = text[:action_match.start()].strip()
            return text.split("\n")[0]
        return response


    def run(self, init_prompt_dict=None):

        if init_prompt_dict is not None:
            self.init_prompt_dict = init_prompt_dict
            self.instruction = init_prompt_dict["instruction"]
            self.examples = init_prompt_dict["examples"]
        system_message = self.init_prompt_dict["system_msg"]

        flag = False
        while not flag:
            input_prompt = self.make_prompt(
                need_goal=self.need_goal,
                check_actions=self.check_actions,
                check_inventory=self.check_inventory,
                system_message=system_message,
            )
            success, response = self.llm_model.generate(system_message, input_prompt)


            if not success or response is None:
                flag = True
                response = "inventory"
                self.think_count = 0
                self.force_action = False
                continue

            if response.startswith("Action"):
                flag = True
                self.think_count = 0
                self.force_action = False
                response = self.extract_action(response)
                if self.use_parser:
                    response = self.action_parser_for_special_llms(response)
            elif response.startswith("Reflection:"):


                self.force_action = False
                reflection = self.extract_reflect(response)
                self.memory.append((self.REASONING_TAG, reflection))
                action_match = _ACTION_RE.search(response)
                if action_match is not None:


                    flag = True
                    self.think_count = 0


                    action_response = response[action_match.start():].lstrip(" \t\n.")
                    response = self.extract_action(action_response)
                    if self.use_parser:
                        response = self.action_parser_for_special_llms(response)
                else:
                    flag = False
                    self.think_count += 1
                    response = reflection


                    if self.think_count > self.max_think_iters:
                        flag = True
                        response = "inventory"
                        self.think_count = 0
                        self.force_action = False
            else:


                if self.force_action:
                    flag = True
                    self.think_count = 0
                    self.force_action = False


                    if self.use_parser:
                        response = self.action_parser_for_special_llms(response)
                else:
                    flag = False
                    self.think_count = 0
                    self.force_action = True

        return success, response
