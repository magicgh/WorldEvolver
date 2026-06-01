
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from common.registry import registry as ab_registry

from .base import WorldModelBackend, llm_generate_with_logprobs
from .modules.delta_card import DeltaCard
from .modules.memory_mixins import (
    EpisodicMemoryMixin,
    prompt_with_blocks,
)
from .base import _build_user_msg, _load_base_prompt


_DEFAULT_TOP_K = int(os.environ.get("WORLDEVOLVER_WM_TOP_K", "3"))


def _cards_in_user_prompt() -> bool:

    return os.environ.get(
        "WORLDEVOLVER_EPISODIC_CARD_PLACEMENT", "system"
    ).strip().lower() == "user"


@ab_registry.register_wm("wm-episodic")
class WMEpisodic(EpisodicMemoryMixin, WorldModelBackend):


    name = "wm-episodic"

    def __init__(
        self,
        llm: Any,
        env_name: str,
        top_k: int = _DEFAULT_TOP_K,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, env_name=env_name, **kwargs)
        self._init_episodic_memory(
            llm=llm,
            top_k=top_k,
        )
        self._base_prompt = _load_base_prompt(self.env_name, "episodic")


    def _system_prompt(self, cards: List[DeltaCard]) -> str:
        return prompt_with_blocks(self._base_prompt, self._render_episodic_cards(cards))

    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        query_triples, cards = self._episodic_context(state, action)


        if _cards_in_user_prompt():
            cards_block = self._render_episodic_cards(cards)
            system = self._base_prompt
            user = _build_user_msg(state, action, goal)
            if cards_block:
                user = f"{cards_block}\n\n{user}"
        else:
            system = self._system_prompt(cards)
            user = _build_user_msg(state, action, goal)
        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            system,
            user,
            max_tokens=self.wm_max_tokens,
        )
        trace = {**self._episodic_trace(cards, query_triples), "n_rules": 0}
        if not ok or text is None:
            return {
                "prediction": None,
                "mean_logprob": mean_lp,
                "system": system,
                **trace,
                "error": "llm_failed",
            }
        return {
            "prediction": text.strip(),
            "mean_logprob": mean_lp,
            "system": system,
            **trace,
        }


    def update(
        self,
        state: str,
        action: str,
        prediction: Optional[str],
        gold_next_state: str,
        info: Optional[dict] = None,
    ) -> None:
        self._append_episodic_transition(
            state=state,
            action=action,
            gold_next_state=gold_next_state,
            info=info,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_name": self.env_name,
            **self._episodic_state_dict(include_embedder_failed=True),
        }
