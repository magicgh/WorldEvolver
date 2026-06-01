
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from common.registry import registry as ab_registry

from .base import WorldModelBackend, llm_generate_with_logprobs
from .modules.delta_card import DeltaCard
from .modules.memory_mixins import (
    DEFAULT_TOKEN_BUDGET,
    EpisodicMemoryMixin,
    SemanticMemoryMixin,
    prompt_with_blocks,
)
from .modules.selective_foresight import SelectiveForesightGate
from .base import _build_user_msg, _load_base_prompt


_DEFAULT_TOP_K = int(os.environ.get("WORLDEVOLVER_WM_TOP_K", "3"))
_DEFAULT_BATCH_K = int(os.environ.get("WORLDEVOLVER_WM_BATCH_K", "3"))
_DEFAULT_TOKEN_BUDGET = DEFAULT_TOKEN_BUDGET

_DEFAULT_SF_CONFIDENCE_PCT = float(os.environ.get(
    "WORLDEVOLVER_WM_SF_CONFIDENCE_PCT", "0",
))


@ab_registry.register_wm("wm-episodic-semantic")
class WMEpisodicSemantic(
    EpisodicMemoryMixin,
    SemanticMemoryMixin,
    WorldModelBackend,
):


    name = "wm-episodic-semantic"

    def __init__(
        self,
        llm: Any,
        env_name: str,
        top_k: int = _DEFAULT_TOP_K,
        batch_k: int = _DEFAULT_BATCH_K,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        filter_enabled: bool = True,
        registry_path: Optional[str] = None,
        sf_confidence_pct: float = _DEFAULT_SF_CONFIDENCE_PCT,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, env_name=env_name, **kwargs)
        self._init_episodic_memory(
            llm=llm,
            top_k=top_k,
        )
        self._init_semantic_memory(
            llm=llm,
            batch_k=batch_k,
            token_budget=token_budget,
            filter_enabled=filter_enabled,
            registry_path=registry_path,
        )
        self._base_prompt = _load_base_prompt(self.env_name, "episodic_semantic")
        self.sf_gate = SelectiveForesightGate(sf_confidence_pct, cell_name=self.name)


    def _system_prompt(self, cards: List[DeltaCard]) -> str:
        return prompt_with_blocks(
            self._base_prompt,
            self._render_semantic_rules(),
            self._render_episodic_cards(cards),
        )

    def _trace(
        self,
        cards: List[DeltaCard],
        query_triples: List[List[str]],
    ) -> Dict[str, Any]:
        return {
            **self._episodic_trace(cards, query_triples),
            **self._semantic_trace(),
        }

    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        query_triples, cards = self._episodic_context(state, action)
        system = self._system_prompt(cards)
        user = _build_user_msg(state, action, goal)
        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            system,
            user,
            max_tokens=self.wm_max_tokens,
        )
        trace = self._trace(cards, query_triples)
        if not ok or text is None:
            return {
                "prediction": None,
                "mean_logprob": mean_lp,
                "system": system,
                **trace,
                "error": "llm_failed",
            }
        prediction = text.strip()
        out: Dict[str, Any] = {


            "prediction": prediction,
            "foresight": prediction,
            "mean_logprob": mean_lp,
            "system": system,
            **trace,
        }
        return self.sf_gate.apply(out, mean_lp)


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

        self._update_semantic_memory(
            state=state,
            action=action,
            prediction=prediction,
            gold_next_state=gold_next_state,
            info=info,
        )

    def reset(self, env: Any = None) -> None:


        super().reset(env)
        self.sf_gate.reset()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_name": self.env_name,
            **self._episodic_state_dict(include_embedder_failed=True),
            **self._semantic_state_dict(),
            "sf_confidence_pct": self.sf_gate.confidence_pct,
            "sf_scored_count": self.sf_gate.count,
        }
