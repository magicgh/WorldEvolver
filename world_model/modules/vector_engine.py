
from __future__ import annotations

from dataclasses import dataclass
import os
import re
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np


try:
    import faiss

    _FAISS_OK = True
    _FAISS_GPU_OK = (
        hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "index_cpu_to_gpu")
        and faiss.get_num_gpus() > 0
    )
except Exception:
    faiss = None
    _FAISS_OK = False
    _FAISS_GPU_OK = False

try:
    import hnswlib

    _HNSW_OK = True
except Exception:
    hnswlib = None
    _HNSW_OK = False

_HNSW_EF_CONSTRUCTION = 200
_HNSW_M = 48
_HNSW_MIN_EF = 100
_HNSW_MAX_EF = 1000


def _parse_vector_index_gpu_id() -> Optional[int]:

    spec = os.environ.get("WORLDEVOLVER_VECTOR_INDEX_DEVICE", "").strip().lower()
    match = re.match(r"^cuda:(\d+)$", spec)
    if match is None:
        return None
    return int(match.group(1))


class VectorEngine:


    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError

    def add(self, row: np.ndarray, row_id: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass
class FAISSVectorEngine(VectorEngine):


    index: Any
    resources: Any = None

    @classmethod
    def build(cls, vectors: np.ndarray) -> "FAISSVectorEngine":
        dim = int(vectors.shape[1])
        idx_cpu = faiss.IndexFlatIP(dim)
        idx_cpu.add(np.ascontiguousarray(vectors))
        gpu_id = _parse_vector_index_gpu_id()
        if _FAISS_GPU_OK and gpu_id is not None:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(
                resources, gpu_id, idx_cpu,
            )
            return cls(index=index, resources=resources)
        return cls(index=idx_cpu)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        scores, idxs = self.index.search(np.ascontiguousarray(query), k)
        ranked: List[Tuple[int, float]] = []
        for rank, raw_idx in enumerate(idxs[0].tolist()):
            if raw_idx >= 0:
                ranked.append((int(raw_idx), float(scores[0][rank])))
        return ranked

    def add(self, row: np.ndarray, row_id: int) -> None:
        self.index.add(np.ascontiguousarray(row))

    def close(self) -> None:
        self.index = None
        self.resources = None


@dataclass
class HNSWVectorEngine(VectorEngine):


    index: Any

    @classmethod
    def build(cls, vectors: np.ndarray) -> "HNSWVectorEngine":
        dim = int(vectors.shape[1])
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=max(1, int(vectors.shape[0])),
            ef_construction=_HNSW_EF_CONSTRUCTION,
            M=_HNSW_M,
        )
        index.add_items(
            np.ascontiguousarray(vectors),
            np.arange(vectors.shape[0], dtype=np.int64),
        )
        index.set_ef(
            max(
                _HNSW_MIN_EF,
                min(_HNSW_MAX_EF, int(vectors.shape[0])),
            )
        )
        return cls(index=index)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        idxs, distances = self.index.knn_query(np.ascontiguousarray(query), k=k)
        ranked: List[Tuple[int, float]] = []
        for rank, raw_idx in enumerate(idxs[0].tolist()):
            if raw_idx >= 0:
                ranked.append((int(raw_idx), float(1.0 - distances[0][rank])))
        return ranked

    def add(self, row: np.ndarray, row_id: int) -> None:
        resize = getattr(self.index, "resize_index", None)
        if resize is not None:
            resize(row_id + 1)
        self.index.add_items(
            np.ascontiguousarray(row),
            np.array([row_id], dtype=np.int64),
        )

    def close(self) -> None:
        self.index = None


def build_vector_engine(env_name: str, vectors: np.ndarray) -> Optional[VectorEngine]:

    if _FAISS_OK:
        try:
            return FAISSVectorEngine.build(vectors)
        except Exception as exc:
            warnings.warn(
                f"StateLibrary({env_name}): FAISS index build failed "
                f"({type(exc).__name__}: {exc}); falling back to hnswlib retrieval.",
                RuntimeWarning,
                stacklevel=2,
            )
    if _HNSW_OK:
        try:
            return HNSWVectorEngine.build(vectors)
        except Exception as exc:
            warnings.warn(
                f"StateLibrary({env_name}): HNSW index build failed "
                f"({type(exc).__name__}: {exc}); retrieval is disabled for "
                "this index.",
                RuntimeWarning,
                stacklevel=2,
            )
    return None


__all__ = [
    "VectorEngine",
    "FAISSVectorEngine",
    "HNSWVectorEngine",
    "build_vector_engine",
]
