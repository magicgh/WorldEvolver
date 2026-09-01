
from __future__ import annotations

from typing import Any, Dict, Optional

from common.registry import registry as ab_registry

from .base import WorldModelBackend, llm_generate_with_logprobs, prompt_telemetry
from .base import _build_user_msg, _load_base_prompt
from .modules.memory_mixins import (
    DEFAULT_TOKEN_BUDGET,
    SemanticMemoryMixin,
    prompt_with_blocks,
)

_DEFAULT_BATCH_K = 3
_DEFAULT_TOKEN_BUDGET = DEFAULT_TOKEN_BUDGET


@ab_registry.register_wm("wm-semantic")
class WMSemantic(SemanticMemoryMixin, WorldModelBackend):


    name = "wm-semantic"

    def __init__(
        self,
        llm: Any,
        env_name: str,
        batch_k: int = _DEFAULT_BATCH_K,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        filter_enabled: bool = True,
        registry_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, env_name=env_name, **kwargs)
        self._init_semantic_memory(
            llm=llm,
            batch_k=batch_k,
            token_budget=token_budget,
            filter_enabled=filter_enabled,
            registry_path=registry_path,
        )
        self._base_prompt = _load_base_prompt(self.env_name, "semantic")


    def _system_prompt(self) -> str:
        return prompt_with_blocks(self._base_prompt, self._render_semantic_rules())

    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        rules_block = self._render_semantic_rules()
        system = prompt_with_blocks(self._base_prompt, rules_block)
        user = _build_user_msg(state, action, goal)
        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            system,
            user,
            max_tokens=self.wm_max_tokens,
        )
        trace = {
            **self._semantic_trace(),
            "n_retrieved": 0,
            "embedder_failed": False,
        }
        telemetry = prompt_telemetry(
            self.llm,
            system,
            user,
            text if ok else None,
            semantic_block=rules_block,
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
        self._update_semantic_memory(
            state=state,
            action=action,
            prediction=prediction,
            gold_next_state=gold_next_state,
            info=info,
        )

    def reset(self, env: Any = None) -> None:


        return None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_name": self.env_name,
            "reset_scope": self.reset_scope,
            **self._semantic_state_dict(),
        }

    def clear_memory(self) -> None:
        self._clear_semantic_memory()

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            **super().checkpoint_state(),
            "semantic": self._semantic_checkpoint_state(),
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        super().load_checkpoint_state(state)
        self._load_semantic_checkpoint_state(state.get("semantic"))
