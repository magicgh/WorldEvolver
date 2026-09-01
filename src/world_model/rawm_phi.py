
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from common.registry import registry as ab_registry

from .base import WorldModelBackend, llm_generate_with_logprobs
from .modules.memory_mixins import prompt_with_blocks
from .modules.state_library import RawmPhiLibrary

_DEFAULT_TOP_K = 3
_RAWM_ENCODER = "Qwen/Qwen3-Embedding-8B"


_RAWM_SYSTEM = (
    "You are a world model. Given the agent's current state and "
    "proposed action, predict what the agent will observe next.\n\n"
    'Output format: a single paragraph starting with "Prediction:" '
    "describing the next observation in the style of the environment's "
    "own text output."
)


def _render_retrieved_block(retrieved: List[Dict]) -> str:

    if not retrieved:
        return ""
    lines: List[str] = ["## Retrieved similar past transitions"]
    for r in retrieved:
        s = (r.get("state_obs_raw") or "").strip()
        a = (r.get("action") or "").strip()
        ns = (r.get("next_observation_raw") or "").strip()
        lines.append("")
        lines.append(f"Current state: {s}\nAction: {a}\nNext state: {ns}")
    return "\n".join(lines)


def _render_user_msg(
    state: str,
    action: str,
    goal: Optional[str] = None,
) -> str:

    parts: List[str] = []
    g = (goal or "").strip()
    if g:
        parts.append(f"Task: {g}")
    parts.append(
        f"Current state: {(state or '').strip()}\n" f"Action: {(action or '').strip()}"
    )
    return "\n\n".join(parts)


@ab_registry.register_wm("wm-rawm-phi")
class WMRawmPhi(WorldModelBackend):


    name = "wm-rawm-phi"

    def __init__(
        self,
        llm: Any,
        env_name: str,
        top_k: int = _DEFAULT_TOP_K,
        state_library_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, env_name=env_name, **kwargs)
        self.top_k = max(0, int(top_k))
        self.library = RawmPhiLibrary(
            self.env_name,
            library_dir=state_library_dir,
        )


    def _retrieve(self, state: str, action: str) -> List[Dict]:
        if self.top_k <= 0 or not self.library.records:
            self._last_retrieval_latency_ms = 0.0
            return []
        started = time.perf_counter()
        try:
            return self.library.retrieve_records_by_state_action(
                state=state,
                action=action,
                k=self.top_k,
            )
        finally:
            self._last_retrieval_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0


    def _system_prompt(self, retrieved: List[Dict]) -> str:

        return prompt_with_blocks(_RAWM_SYSTEM, _render_retrieved_block(retrieved))

    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        retrieved = self._retrieve(state, action)


        system = self._system_prompt(retrieved)
        user = _render_user_msg(state, action, goal)
        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            system,
            user,
            max_tokens=self.wm_max_tokens,
        )
        trace = {
            "n_retrieved": len(retrieved),
            "retrieval_latency_ms": getattr(
                self,
                "_last_retrieval_latency_ms",
                None,
            ),
        }
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
        task = (info or {}).get("task", "")
        self.library.append(
            task=str(task or ""),
            state=str(state or ""),
            action=str(action or ""),
            gold_next_state=str(gold_next_state or ""),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_name": self.env_name,
            "top_k": self.top_k,
            "n_records": len(self.library.records),
            "encoder": _RAWM_ENCODER,
            "degraded": bool(self.library.embedder_failed),
        }
