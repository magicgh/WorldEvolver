
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .delta_card import DeltaCard, _to_triple_set, render_delta_cards
from .llm_factorizer import LLMFactorizer
from .metrics import normalize
from .state_library import EpisodicLibrary
from .typed_registry import (
    MismatchSample,
    TypedRegistry,
    extract_entries,
)


DEFAULT_TOKEN_BUDGET = int(os.environ.get("WORLDEVOLVER_WM_TOKEN_BUDGET", "2048"))


def prompt_with_blocks(base_prompt: str, *blocks: str) -> str:

    return "\n\n".join([b for b in (*blocks, base_prompt) if b])


class EpisodicMemoryMixin:


    def _init_episodic_memory(
        self,
        *,
        llm: Any,
        top_k: int,
    ) -> None:


        self.top_k = max(0, int(top_k))


        self.library = EpisodicLibrary(self.env_name)

    def _retrieve(
        self,
        state: str,
        action: str,
    ) -> List[DeltaCard]:
        return self.library.retrieve_top_k(
            state=state,
            action=action,
            k=self.top_k,
        )

    def _render_episodic_cards(self, cards: Sequence[DeltaCard]) -> str:
        return render_delta_cards(cards) if cards else ""

    def _episodic_context(
        self,
        state: str,
        action: str,
    ) -> tuple[List[List[str]], List[DeltaCard]]:


        return [], self._retrieve(state, action)

    def _episodic_trace(
        self,
        cards: Sequence[DeltaCard],
        query_triples: Sequence[Sequence[str]],
    ) -> Dict[str, Any]:
        return {
            "n_retrieved": len(cards),
            "embedder_failed": self.library.embedder_failed,
        }

    def _append_episodic_transition(
        self,
        *,
        state: str,
        action: str,
        gold_next_state: str,
        info: Optional[dict] = None,
    ) -> None:
        task = (info or {}).get("task", "")


        self.library.append(
            task=str(task or ""),
            state=str(state or ""),
            action=str(action or ""),
            gold_next_state=str(gold_next_state or ""),
        )

    def _episodic_state_dict(self, *, include_embedder_failed: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "top_k": self.top_k,
            "n_records": len(self.library.records),
        }

        if hasattr(self, "factorizer"):
            out["factorizer"] = self.factorizer.state_dict()
        if include_embedder_failed:
            out["embedder_failed"] = self.library.embedder_failed
        return out


class SemanticMemoryMixin:


    def _init_semantic_memory(
        self,
        *,
        llm: Any,
        batch_k: int,
        token_budget: int,
        filter_enabled: bool = True,
        registry_path: Optional[str] = None,
    ) -> None:
        if not hasattr(self, "factorizer"):
            self.factorizer = LLMFactorizer(
                llm,
                self.env_name,
                max_tokens=getattr(self, "wm_max_tokens", None),
            )
        self.batch_k = max(1, int(batch_k))
        self.token_budget = max(1, int(token_budget))
        if registry_path is not None and os.path.isfile(registry_path):
            self.registry = TypedRegistry.load(registry_path)
            self.registry.filter_enabled = filter_enabled
        else:
            self.registry = TypedRegistry(filter_enabled=filter_enabled)
        self._pending: List[MismatchSample] = []
        self._displayed_rule_keys = set()

    def _render_semantic_rules(self) -> str:
        rendered, keys = self.registry.compile_with_keys(
            token_budget=self.token_budget
        )
        self._displayed_rule_keys = keys
        return rendered

    def _semantic_trace(self) -> Dict[str, Any]:
        return {"n_rules": len(self.registry)}

    def _semantic_observation_tuples(self, observation: str) -> tuple[set, bool]:
        triples, ok = self.factorizer.factorize_checked(observation)
        return set(_to_triple_set(triples)), ok

    def _semantic_is_mismatch(
        self,
        prediction: str,
        gold_next_state: str,
    ) -> bool:
        pred_tuples, pred_ok = self._semantic_observation_tuples(prediction)
        gold_tuples, gold_ok = self._semantic_observation_tuples(gold_next_state)
        if not pred_ok or not gold_ok:
            return normalize(prediction) != normalize(gold_next_state)
        if not pred_tuples and not gold_tuples:
            return False
        return pred_tuples != gold_tuples

    def _update_semantic_memory(
        self,
        *,
        state: str,
        action: str,
        prediction: Optional[str],
        gold_next_state: str,
        info: Optional[dict] = None,
    ) -> None:


        if prediction is None:
            return
        if not self._semantic_is_mismatch(prediction, gold_next_state):
            self.registry.credit_match(self._displayed_rule_keys)
            return
        self.registry.credit_mismatch(self._displayed_rule_keys)
        task = (info or {}).get("task", "")
        self._pending.append(MismatchSample(
            task=str(task or ""),
            state_obs=str(state or ""),
            action=str(action or ""),
            prediction=str(prediction or ""),
            gold_next_obs=str(gold_next_state or ""),
        ))
        if len(self._pending) >= self.batch_k:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        batch = self._pending[: self.batch_k]
        self._pending = self._pending[self.batch_k :]
        new_entries = extract_entries(
            self.llm,
            batch,
            env_name=self.env_name,
            max_tokens=getattr(self, "wm_max_tokens", None),
        )
        if new_entries:
            self.registry.merge(new_entries)

    def _semantic_state_dict(self) -> Dict[str, Any]:
        return {
            "batch_k": self.batch_k,
            "token_budget": self.token_budget,
            "n_rules": len(self.registry),
            "pending": len(self._pending),
            "filter_enabled": self.registry.filter_enabled,
        }

    def save_registry(self, path: str) -> None:
        self.registry.save(path)
