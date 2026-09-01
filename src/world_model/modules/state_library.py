
from __future__ import annotations

import copy
import hashlib
import re
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.state_library_assets import lib_dir, lib_path, load_library

from .delta_card import DeltaCard, build_delta_card
from .metrics import _embedding_client
from .vector_engine import (
    VectorEngine,
    build_vector_engine,
)


_ACTION_TOKEN_RE = re.compile(r"[a-z0-9]+")
EPISODIC_RETRIEVERS = frozenset({"jaccard_topk", "uniform_random"})


def _stable_uint32(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _action_tokens(action: str) -> set:

    return set(_ACTION_TOKEN_RE.findall((action or "").lower()))


def _action_jaccard(query_tokens: set, record_tokens: set) -> float:
    if not query_tokens or not record_tokens:
        return 0.0
    return len(query_tokens & record_tokens) / len(query_tokens | record_tokens)


def _format_query_text(state: str, action: str) -> str:

    return f"current state: {(state or '').strip()}\naction: {(action or '').strip()}"


def _format_memory_text(state: str, action: str, next_state: str) -> str:

    return (
        f"current state: {(state or '').strip()}\n"
        f"action: {(action or '').strip()}\n"
        f"next state: {(next_state or '').strip()}"
    )


class StateLibrary:


    def __init__(self, env_name: str, *, library_dir: Optional[str] = None) -> None:
        self.env_name = env_name.lower()
        self.library_dir = library_dir
        self.records: List[Dict] = []
        self._embedder_failed: bool = False
        self._lock = threading.Lock()

    @property
    def embedder_failed(self) -> bool:
        return self._embedder_failed

    def load_from_disk(self) -> int:

        with self._lock:
            self.records = list(load_library(self.env_name, self.library_dir))
            self._invalidate_index()
            return len(self.records)

    def load_records(self, records: Sequence[Dict]) -> int:

        with self._lock:
            self.records = [copy.deepcopy(r) for r in records]
            self._invalidate_index()
            return len(self.records)

    def clear(self) -> None:
        """Clear records and any derived retrieval index atomically."""
        with self._lock:
            self.records.clear()
            self._invalidate_index()

    def _invalidate_index(self) -> None:
        pass

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:

        enc = _embedding_client()
        mat = enc.encode(list(texts))
        mat = np.asarray(mat, dtype="float32")
        if mat.ndim != 2:
            mat = mat.reshape(len(texts), -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class EpisodicLibrary(StateLibrary):


    def __init__(self, env_name: str, *, library_dir: Optional[str] = None) -> None:
        super().__init__(env_name, library_dir=library_dir)

    def _invalidate_index(self) -> None:
        pass


    def append(
        self,
        *,
        task: str,
        state: str,
        action: str,
        gold_next_state: str,
        state_triples: Optional[Sequence] = None,
        next_state_triples: Optional[Sequence] = None,
    ) -> None:
        record = {
            "task": task or "",
            "action": action or "",
            "state_obs_raw": state or "",
            "state_triples": list(state_triples) if state_triples else [],
            "next_observation_raw": gold_next_state or "",
            "next_state_triples": list(next_state_triples) if next_state_triples else [],
        }
        with self._lock:
            self.records.append(record)


    def _rank_by_action(self, action: str, k: int) -> List[Tuple[int, float]]:

        if k <= 0 or not self.records:
            return []
        q_tokens = _action_tokens(action)
        if not q_tokens:
            return []
        with self._lock:
            scored = [
                (i, _action_jaccard(q_tokens, _action_tokens(str(r.get("action", "")))))
                for i, r in enumerate(self.records)
            ]
        scored = [(i, s) for i, s in scored if s > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _build_card(self, idx: int, score: float) -> DeltaCard:
        r = self.records[int(idx)]
        return build_delta_card(
            action=str(r.get("action", "")),
            next_observation_raw=str(r.get("next_observation_raw", "")),
            source_score=float(score),
            state_obs_raw=str(r.get("state_obs_raw") or ""),
            source_index=int(idx),
            source_task=str(r.get("task") or ""),
        )

    def retrieve_top_k(
        self,
        state: str,
        action: str,
        k: int,
        state_triples: Optional[Sequence] = None,
        *,
        retriever: str = "jaccard_topk",
    ) -> List[DeltaCard]:
        retriever = str(retriever).strip().lower()
        if retriever not in EPISODIC_RETRIEVERS:
            raise ValueError(
                "episodic_retriever must be one of "
                f"{sorted(EPISODIC_RETRIEVERS)!r}; got {retriever!r}"
            )
        if retriever == "uniform_random":
            reference_count = len(self._rank_by_action(action, k))
            with self._lock:
                n_records = len(self.records)
                sample_size = min(reference_count, n_records)
                if sample_size == 0:
                    indices = []
                elif sample_size == n_records:
                    indices = list(range(n_records))
                else:
                    rng = np.random.default_rng(
                        _stable_uint32(self.env_name, state, action, n_records)
                    )
                    indices = sorted(
                        int(index)
                        for index in rng.choice(
                            n_records,
                            size=sample_size,
                            replace=False,
                        )
                    )
            return [self._build_card(index, 0.0) for index in indices]
        return [
            self._build_card(idx, score)
            for idx, score in self._rank_by_action(action, k)
        ]

    def retrieve_records_by_action(self, action: str, k: int) -> List[Dict]:

        return [self.records[idx] for idx, _score in self._rank_by_action(action, k)]


class RawmPhiLibrary(StateLibrary):


    def __init__(self, env_name: str, *, library_dir: Optional[str] = None) -> None:
        super().__init__(env_name, library_dir=library_dir)
        self._transition_mat: Optional[np.ndarray] = None
        self._transition_engine: Optional[VectorEngine] = None

    def _clear_transition_engine(self) -> None:
        if self._transition_engine is not None:
            self._transition_engine.close()
        self._transition_engine = None

    def _invalidate_index(self) -> None:
        self._transition_mat = None
        self._clear_transition_engine()


    def append(
        self,
        *,
        task: str,
        state: str,
        action: str,
        gold_next_state: str,
        state_triples: Optional[Sequence] = None,
        next_state_triples: Optional[Sequence] = None,
    ) -> None:
        record = {
            "task": task or "",
            "action": action or "",
            "state_obs_raw": state or "",
            "state_triples": list(state_triples) if state_triples else [],
            "next_observation_raw": gold_next_state or "",
            "next_state_triples": list(next_state_triples) if next_state_triples else [],
        }
        memory_text = _format_memory_text(
            state=record["state_obs_raw"],
            action=record["action"],
            next_state=record["next_observation_raw"],
        )
        new_row: Optional[np.ndarray] = None
        if self._transition_mat is not None and not self._embedder_failed:
            try:
                new_row = self._encode_texts([memory_text])
            except Exception:
                new_row = None
        with self._lock:
            self.records.append(record)
            if new_row is not None and self._transition_mat is not None and \
               new_row.shape[1] == self._transition_mat.shape[1]:
                self._transition_mat = np.vstack([self._transition_mat, new_row])
                if self._transition_engine is not None:
                    try:
                        self._transition_engine.add(new_row, len(self.records) - 1)
                    except Exception:
                        self._invalidate_index()
            else:
                self._invalidate_index()


    def _ensure_transition_embeddings(self) -> bool:
        if self._transition_mat is not None and not self._embedder_failed:
            return True
        if self._embedder_failed:
            raise RuntimeError(
                f"RawmPhiLibrary({self.env_name}): embedder previously failed"
            )
        with self._lock:
            if self._transition_mat is not None:
                return True
            if not self.records:
                return False
            texts = [
                _format_memory_text(
                    state=str(r.get("state_obs_raw", "")),
                    action=str(r.get("action", "")),
                    next_state=str(r.get("next_observation_raw", "")),
                )
                for r in self.records
            ]
            n_records = len(self.records)
            records_id = id(self.records)
        try:
            normed = self._encode_texts(texts)
        except Exception:
            self._embedder_failed = True
            raise
        with self._lock:
            if id(self.records) == records_id and len(self.records) == n_records:
                self._transition_mat = normed
                self._clear_transition_engine()
                self._transition_engine = build_vector_engine(self.env_name, normed)
                return True
            return False

    def _rank_by_state_action(
        self, state: str, action: str, k: int
    ) -> List[Tuple[int, float]]:
        if k <= 0:
            return []
        if not self._ensure_transition_embeddings():
            return []
        q = self._encode_texts([_format_query_text(state, action)])
        assert self._transition_mat is not None
        k_eff = min(k, len(self.records))
        if k_eff <= 0:
            return []
        with self._lock:
            engine = self._transition_engine
            if engine is not None:
                try:
                    ranked = engine.search(q, k_eff)
                except Exception:
                    ranked = []
                if ranked:
                    return [
                        (idx, score)
                        for idx, score in ranked
                        if 0 <= idx < len(self.records)
                    ]
        raise RuntimeError(
            f"RawmPhiLibrary({self.env_name}): transition retrieval requires "
            "FAISS or hnswlib; no vector index is available."
        )

    def retrieve_records_by_state_action(
        self, state: str, action: str, k: int
    ) -> List[Dict]:

        return [
            self.records[idx]
            for idx, _score in self._rank_by_state_action(state, action, k)
        ]


def get_library(
    env_name: str,
    *,
    library_dir: Optional[str] = None,
) -> EpisodicLibrary:

    lib = EpisodicLibrary(env_name, library_dir=library_dir)
    lib.load_records(list(load_library(env_name, library_dir)))
    return lib


__all__ = [
    "EPISODIC_RETRIEVERS",
    "StateLibrary",
    "EpisodicLibrary",
    "RawmPhiLibrary",
    "get_library",
    "load_library",
    "lib_dir",
    "lib_path",
]
