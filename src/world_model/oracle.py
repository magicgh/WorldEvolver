
from __future__ import annotations

import random
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data.word2world import Word2WorldDataset


class OracleProvider(ABC):


    name: str = "base"

    @abstractmethod
    def predict(
        self,
        env: Any,
        action: str,
        true_obs: str,
        reward: float,
        done: bool,
        info: Optional[dict],
        history: Optional[list] = None,
    ) -> Optional[str]:
        pass

    def reset(self, env: Any) -> None:
        return None


class NoneOracle(OracleProvider):


    name = "none"

    def predict(
        self,
        env: Any,
        action: str,
        true_obs: str,
        reward: float,
        done: bool,
        info: Optional[dict],
        history: Optional[list] = None,
    ) -> Optional[str]:
        return None


class PerfectOracle(OracleProvider):


    name = "perfect"

    def predict(
        self,
        env: Any,
        action: str,
        true_obs: str,
        reward: float,
        done: bool,
        info: Optional[dict],
        history: Optional[list] = None,
    ) -> Optional[str]:
        if true_obs is None:
            return None
        return str(true_obs)


_POOL_CACHE: Dict[Tuple[str, Optional[str], bool], List[str]] = {}
_POOL_LOCK = threading.Lock()


def _assistant_observations(records: Iterable[Dict[str, Any]]) -> List[str]:
    pool: List[str] = []
    for traj in records:
        for msg in traj.get("messages") or []:
            if msg.get("role") != "assistant":
                continue
            content = (msg.get("content") or "").strip()
            if content:
                pool.append(content)
    return pool


def get_observation_pool(
    env_name: str,
    data_dir: Optional[str] = None,
    from_hf: bool = True,
) -> List[str]:

    dataset = Word2WorldDataset(
        env_name,
        data_dir=data_dir,
        from_hf=from_hf,
    )
    cache_source = (
        f"hf:{dataset.hf_name}"
        if from_hf
        else str(dataset.data_dir.resolve())
    )
    cache_key = (dataset.env_name, cache_source, from_hf)
    with _POOL_LOCK:
        cached = _POOL_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

    try:
        records = dataset.load()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"NoisyOracle requires the full Word2World eval data for "
            f"env={dataset.env_name!r}. Missing local file at "
            f"{str(dataset.path)!r}; install it under {str(dataset.data_dir)!r} "
            "or install the optional 'huggingface_hub' package and pass "
            "word2world_from_hf=True."
        ) from exc

    pool = _assistant_observations(records)
    if not pool:
        source = f"HuggingFace {dataset.hf_name!r}" if from_hf else str(dataset.path)
        raise RuntimeError(
            f"NoisyOracle loaded {source} for env={dataset.env_name!r} but "
            "found zero assistant observations."
        )

    with _POOL_LOCK:
        _POOL_CACHE[cache_key] = list(pool)
    return list(pool)


class NoisyOracle(OracleProvider):


    name = "noisy"

    def __init__(
        self,
        env_name: str,
        seed: Optional[int] = None,
        data_dir: Optional[str] = None,
        word2world_from_hf: bool = True,
    ) -> None:
        self.env_name = Word2WorldDataset._normalize_env(env_name)
        self.pool: List[str] = list(
            get_observation_pool(
                self.env_name,
                data_dir=data_dir,
                from_hf=word2world_from_hf,
            )
        )
        if not self.pool:
            raise ValueError(f"NoisyOracle pool for env={env_name!r} is empty.")
        self.rng = random.Random(seed)

    def reset(self, env: Any) -> None:
        return None

    def predict(
        self,
        env: Any,
        action: str,
        true_obs: str,
        reward: float,
        done: bool,
        info: Optional[dict],
        history: Optional[list] = None,
    ) -> Optional[str]:
        return self.rng.choice(self.pool)


def build_oracle(mode: str, env_name: str, **kwargs: Any) -> OracleProvider:

    key = (mode or "none").lower()
    if key == "none":
        return NoneOracle()
    if key == "noisy":
        return NoisyOracle(env_name=env_name, **kwargs)
    if key == "perfect":
        return PerfectOracle()
    raise ValueError(
        f"Unknown oracle mode {key!r}; expected one of 'none', 'noisy', 'perfect'."
    )


__all__ = [
    "OracleProvider",
    "NoneOracle",
    "NoisyOracle",
    "PerfectOracle",
    "build_oracle",
    "get_observation_pool",
]
