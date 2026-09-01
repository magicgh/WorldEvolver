
from __future__ import annotations

from typing import Any, Dict, Optional

from common.registry import registry as ab_registry

from .base import WorldModelBackend, llm_generate_with_logprobs, prompt_telemetry
from .modules.delta_card import DeltaCard
from .modules.memory_mixins import (
    EpisodicMemoryMixin,
    prompt_with_blocks,
)
from .base import _build_user_msg, _load_base_prompt

_DEFAULT_TOP_K = 3


@ab_registry.register_wm("wm-episodic")
class WMEpisodic(EpisodicMemoryMixin, WorldModelBackend):


    name = "wm-episodic"

    def __init__(
        self,
        llm: Any,
        env_name: str,
        top_k: int = _DEFAULT_TOP_K,
        episodic_retriever: str = "jaccard_topk",
        episodic_card_placement: str = "system",
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, env_name=env_name, **kwargs)
        placement = str(episodic_card_placement).strip().lower()
        if placement not in {"system", "user"}:
            raise ValueError(
                "episodic_card_placement must be 'system' or 'user'; "
                f"got {episodic_card_placement!r}"
            )
        self.episodic_card_placement = placement
        self._init_episodic_memory(
            llm=llm,
            top_k=top_k,
            episodic_retriever=episodic_retriever,
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
        cards_block = self._render_episodic_cards(cards)

        if self.episodic_card_placement == "user":
            system = self._base_prompt
            user = _build_user_msg(state, action, goal)
            if cards_block:
                user = f"{cards_block}\n\n{user}"
        else:
            system = prompt_with_blocks(self._base_prompt, cards_block)
            user = _build_user_msg(state, action, goal)
        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            system,
            user,
            max_tokens=self.wm_max_tokens,
        )
        trace = {**self._episodic_trace(cards, query_triples), "n_rules": 0}
        telemetry = prompt_telemetry(
            self.llm,
            system,
            user,
            text if ok else None,
            episodic_block=cards_block,
        )
        if not ok or text is None:
            return {
                "prediction": None,
                "mean_logprob": mean_lp,
                "system": system,
                **trace,
                **telemetry,
                "error": "llm_failed",
            }
        return {
            "prediction": text.strip(),
            "mean_logprob": mean_lp,
            "system": system,
            **trace,
            **telemetry,
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
            "reset_scope": self.reset_scope,
            "episodic_card_placement": self.episodic_card_placement,
            **self._episodic_state_dict(include_embedder_failed=True),
        }

    def clear_memory(self) -> None:
        self._clear_episodic_memory()

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            **super().checkpoint_state(),
            "episodic": self._episodic_checkpoint_state(),
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        super().load_checkpoint_state(state)
        self._load_episodic_checkpoint_state(state.get("episodic"))
