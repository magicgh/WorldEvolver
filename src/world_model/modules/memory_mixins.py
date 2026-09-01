
from __future__ import annotations

import time
import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from .delta_card import DeltaCard, _to_triple_set, render_delta_cards
from .llm_factorizer import LLMFactorizer
from .metrics import normalize
from .state_library import EPISODIC_RETRIEVERS, EpisodicLibrary
from .typed_registry import (
    MismatchSample,
    TypedRegistry,
    extract_entries,
)

DEFAULT_TOKEN_BUDGET = 2048


def prompt_with_blocks(base_prompt: str, *blocks: str) -> str:

    return "\n\n".join([b for b in (*blocks, base_prompt) if b])


class EpisodicMemoryMixin:


    def _init_episodic_memory(
        self,
        *,
        llm: Any,
        top_k: int,
        episodic_retriever: str = "jaccard_topk",
    ) -> None:
        self.top_k = max(0, int(top_k))
        self.episodic_retriever = str(episodic_retriever).strip().lower()
        if self.episodic_retriever not in EPISODIC_RETRIEVERS:
            raise ValueError(
                "episodic_retriever must be one of "
                f"{sorted(EPISODIC_RETRIEVERS)!r}; "
                f"got {self.episodic_retriever!r}"
            )
        self._last_retrieval_latency_ms: Optional[float] = None
        self.library = EpisodicLibrary(self.env_name)

    def _retrieve(
        self,
        state: str,
        action: str,
    ) -> List[DeltaCard]:
        started = time.perf_counter()
        try:
            kwargs = {
                "state": state,
                "action": action,
                "k": self.top_k,
            }
            if self.episodic_retriever != "jaccard_topk":
                kwargs["retriever"] = self.episodic_retriever
            return self.library.retrieve_top_k(**kwargs)
        finally:
            self._last_retrieval_latency_ms = (time.perf_counter() - started) * 1000.0

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
            "retrieved_indices": [card.source_index for card in cards],
            "retrieved_source_tasks": [card.source_task for card in cards],
            "episodic_store_size": len(getattr(self.library, "records", ())),
            "episodic_retriever": self.episodic_retriever,
            "retrieval_latency_ms": getattr(
                self,
                "_last_retrieval_latency_ms",
                None,
            ),
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
            "episodic_retriever": self.episodic_retriever,
            "n_records": len(self.library.records),
        }

        if hasattr(self, "factorizer"):
            out["factorizer"] = self.factorizer.state_dict()
        if include_embedder_failed:
            out["embedder_failed"] = self.library.embedder_failed
        return out

    def _clear_episodic_memory(self) -> None:
        self.library.clear()

    def _episodic_checkpoint_state(self) -> Dict[str, Any]:
        with self.library._lock:
            records = copy.deepcopy(self.library.records)
        return {
            "top_k": self.top_k,
            "retriever": self.episodic_retriever,
            "records": records,
        }

    def _load_episodic_checkpoint_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("episodic checkpoint must be an object")
        expected = (self.top_k, self.episodic_retriever)
        actual = (
            int(state.get("top_k", -1)),
            str(state.get("retriever", "")),
        )
        if actual != expected:
            raise ValueError(
                f"episodic checkpoint config mismatch: {actual!r} != {expected!r}"
            )
        records = state.get("records")
        if not isinstance(records, list):
            raise ValueError("episodic checkpoint records must be a list")
        self.library.load_records(records)


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
        rendered, keys = self.registry.compile_with_keys(token_budget=self.token_budget)
        self._displayed_rule_keys = keys
        return rendered

    def _semantic_trace(self) -> Dict[str, Any]:
        return {
            "n_rules": len(self.registry),
            "active_rules": self.registry.renderable_count(),
            "rendered_rules": len(self._displayed_rule_keys),
        }

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
        self._pending.append(
            MismatchSample(
                task=str(task or ""),
                state_obs=str(state or ""),
                action=str(action or ""),
                prediction=str(prediction or ""),
                gold_next_obs=str(gold_next_state or ""),
            )
        )
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
            "factorizer": self.factorizer.state_dict(),
        }

    def _clear_semantic_memory(self) -> None:
        self.registry.clear()
        self._pending.clear()
        self._displayed_rule_keys.clear()
        self.factorizer.clear()

    def _semantic_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "batch_k": self.batch_k,
            "token_budget": self.token_budget,
            "filter_enabled": self.registry.filter_enabled,
            "registry": self.registry.to_dict(),
            "pending": [asdict(sample) for sample in self._pending],
            "displayed_rule_keys": sorted(self._displayed_rule_keys),
            "factorizer": self.factorizer.checkpoint_state(),
        }

    def _load_semantic_checkpoint_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("semantic checkpoint must be an object")
        expected = (
            self.batch_k,
            self.token_budget,
            self.registry.filter_enabled,
        )
        actual = (
            int(state.get("batch_k", -1)),
            int(state.get("token_budget", -1)),
            bool(state.get("filter_enabled")),
        )
        if actual != expected:
            raise ValueError(
                f"semantic checkpoint config mismatch: {actual!r} != {expected!r}"
            )
        registry = state.get("registry")
        if not isinstance(registry, dict):
            raise ValueError("semantic checkpoint registry must be an object")
        self.registry = TypedRegistry.from_dict(registry)
        self.registry.filter_enabled = expected[2]
        pending = state.get("pending") or []
        if not isinstance(pending, list):
            raise ValueError("semantic checkpoint pending must be a list")
        self._pending = [MismatchSample(**sample) for sample in pending]
        self._displayed_rule_keys = {
            str(key) for key in (state.get("displayed_rule_keys") or [])
        }
        self.factorizer.load_checkpoint_state(state.get("factorizer") or {})

    def save_registry(self, path: str) -> None:
        self.registry.save(path)
