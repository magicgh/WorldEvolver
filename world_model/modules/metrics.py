
from __future__ import annotations

import os
import re
import threading
from typing import Iterable, List, Optional


_PREDICTION_PREFIX_RE = re.compile(r"^prediction\s*:\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:

    s = (text or "").strip()
    s = _PREDICTION_PREFIX_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def em_at_1(pred: str, gold: str) -> float:

    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def _tokenize(text: str) -> List[str]:

    return _TOKEN_RE.findall(normalize(text))


def token_f1(pred: str, gold: str) -> float:

    p_tokens = _tokenize(pred)
    g_tokens = _tokenize(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    p_count: dict = {}
    for t in p_tokens:
        p_count[t] = p_count.get(t, 0) + 1
    common = 0
    for t in g_tokens:
        if p_count.get(t, 0) > 0:
            common += 1
            p_count[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()

_QWEN3_EMBED_8B = "Qwen/Qwen3-Embedding-8B"
_EMBED_BATCH_SIZE = 64
_DEFAULT_EMBED_BASE_URL = "http://localhost:30001"
_DEFAULT_EMBED_TIMEOUT = 120.0


def _embed_model_name() -> str:
    return os.environ.get("WORLDEVOLVER_EMBED_MODEL") or _QWEN3_EMBED_8B


def _embedding_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


class _OpenAIEmbeddingClient:


    def __init__(
        self,
        base_url: str,
        model: str = _QWEN3_EMBED_8B,
        api_key: str = "EMPTY",
        timeout: float = _DEFAULT_EMBED_TIMEOUT,
    ) -> None:
        import requests

        self.endpoint = _embedding_endpoint(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._requests = requests

    def encode(self, texts, batch_size: int = _EMBED_BATCH_SIZE):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        rows = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self._requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float",
                },
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
            data = sorted(data, key=lambda item: item.get("index", 0))
            rows.extend(item["embedding"] for item in data)
        return np.asarray(rows, dtype=np.float32)


def _embedding_client():

    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        _EMBEDDER = _OpenAIEmbeddingClient(
            base_url=(
                os.environ.get("WORLDEVOLVER_EMBED_BASE_URL")
                or _DEFAULT_EMBED_BASE_URL
            ),
            model=_embed_model_name(),
            api_key=os.environ.get("WORLDEVOLVER_EMBED_API_KEY", "EMPTY"),
            timeout=float(os.environ.get(
                "WORLDEVOLVER_EMBED_TIMEOUT",
                _DEFAULT_EMBED_TIMEOUT,
            )),
        )
        return _EMBEDDER


def _cosine_np(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32").reshape(-1)
    b = np.asarray(b, dtype="float32").reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine(pred: str, gold: str) -> Optional[float]:

    try:
        enc = _embedding_client()
        vecs = enc.encode([pred or "", gold or ""])
        return _cosine_np(vecs[0], vecs[1])
    except Exception:
        return None


def cosine_batch(preds: Iterable[str], golds: Iterable[str]) -> Optional[List[float]]:

    p_list = list(preds)
    g_list = list(golds)
    if len(p_list) != len(g_list):
        raise ValueError(f"len mismatch: preds={len(p_list)} golds={len(g_list)}")
    if not p_list:
        return []
    try:
        enc = _embedding_client()
        all_texts = p_list + g_list
        vecs = enc.encode(all_texts)
        n = len(p_list)
        return [_cosine_np(vecs[i], vecs[i + n]) for i in range(n)]
    except Exception:
        return None
