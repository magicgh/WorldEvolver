
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional

from common.registry import registry

from .prompt_templates import WORLD_MODEL_PROMPTS


def _coerce_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc


DEFAULT_WM_MAX_TOKENS = _coerce_int(
    os.environ.get("WORLDEVOLVER_WM_MAX_TOKENS", "1000"),
    name="WORLDEVOLVER_WM_MAX_TOKENS",
)


@contextmanager
def disable_llm_stop_for_wm(llm: Any, *, max_tokens: Optional[int] = None):

    sentinel = object()
    old_stop = getattr(llm, "stop", sentinel)
    old_max_tokens = getattr(llm, "max_tokens", sentinel)
    target_max_tokens = DEFAULT_WM_MAX_TOKENS if max_tokens is None else max_tokens
    try:
        if old_stop is not sentinel:
            llm.stop = None
        if old_max_tokens is not sentinel:
            llm.max_tokens = max(
                1,
                _coerce_int(target_max_tokens, name="wm max_tokens"),
            )
        yield
    finally:
        if old_stop is not sentinel:
            llm.stop = old_stop
        if old_max_tokens is not sentinel:
            llm.max_tokens = old_max_tokens


def llm_generate_with_logprobs(
    llm: Any,
    system: str,
    user: str,
    *,
    max_tokens: Optional[int] = None,
):

    with disable_llm_stop_for_wm(llm, max_tokens=max_tokens):
        fn = getattr(llm, "generate_with_logprobs", None)
        if fn is not None:
            result = fn(system, user)
            if isinstance(result, tuple) and len(result) == 3:
                ok, text, mean_lp = result
                if ok and text is not None:
                    return ok, text, mean_lp
            if isinstance(result, tuple) and len(result) == 2:
                ok, text = result
                if ok and text is not None:
                    return ok, text, None
        fn_raw = getattr(llm, "generate_raw", None)
        if fn_raw is not None:
            ok, text = fn_raw(system, user)
            return ok, text, None

        ok, text = llm.generate(system, user)
        return ok, text, None


class WorldModelBackend(ABC):


    name: str = "base"

    def __init__(self, llm: Any, env_name: str, **kwargs: Any) -> None:


        self.llm = llm
        self.env_name = (env_name or "").lower()
        self.use_history = _coerce_int(kwargs.get("use_history", 0), name="use_history")
        raw_wm_max_tokens = kwargs.get("wm_max_tokens", kwargs.get("max_tokens", DEFAULT_WM_MAX_TOKENS))
        self.wm_max_tokens = max(
            1,
            _coerce_int(raw_wm_max_tokens, name="wm_max_tokens"),
        )
        unknown = sorted(set(kwargs) - {"use_history", "max_tokens", "wm_max_tokens"})
        if unknown:
            names = ", ".join(unknown)
            raise TypeError(f"{self.name} received unknown WM config key(s): {names}")
        self.config: Dict[str, Any] = {
            "use_history": self.use_history,
            "wm_max_tokens": self.wm_max_tokens,
        }
        if self.use_history != 0:
            raise NotImplementedError(
                f"{self.name} received use_history={self.use_history}; "
                "current WorldEvolver WM cells are locked to state-only "
                "use_history=0 to match the Stage 2/3 baseline protocol."
            )


    @abstractmethod
    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        pass


    def update(
        self,
        state: str,
        action: str,
        prediction: Optional[str],
        gold_next_state: str,
        info: Optional[dict] = None,
    ) -> None:

        return None

    def reset(self, env: Any = None) -> None:

        return None


    def state_dict(self) -> Dict[str, Any]:

        return {"name": self.name, "env_name": self.env_name, "config": dict(self.config)}


if not hasattr(registry, "register_wm"):
    if not hasattr(registry, "mapping"):


        raise RuntimeError(
            "worldevolver.common.registry has no .mapping; cannot install wm namespace"
        )
    registry.mapping.setdefault("wm_name_mapping", {})

    def _register_wm(name: str):
        def _decorator(cls):
            registry.mapping["wm_name_mapping"][name] = cls
            return cls
        return _decorator

    def _get_wm_class(name: str):
        return registry.mapping["wm_name_mapping"].get(name)

    registry.register_wm = _register_wm
    registry.get_wm_class = _get_wm_class


def _format_env_name(env_name: Optional[str] = None) -> str:
    key = (env_name or "").strip().lower()
    if key == "alfworld":
        return "ALFWorld"
    if key == "scienceworld":
        return "ScienceWorld"
    return key or "WorldEvolver"


def _load_base_prompt(
    env_name: Optional[str] = None,
    prompt_kind: str = "base",
) -> str:

    try:
        template = WORLD_MODEL_PROMPTS[prompt_kind]
    except KeyError as exc:
        known = sorted(WORLD_MODEL_PROMPTS)
        raise KeyError(f"Unknown WM prompt kind {prompt_kind!r}; known={known!r}") from exc
    return template.format(env_name=_format_env_name(env_name)).strip()


def _build_user_msg(state: str, action: str, goal: Optional[str]) -> str:

    g = (goal or "").strip()
    s = (state or "").strip()
    a = (action or "").strip()
    if g:
        return f"Task: {g}\n\nState:\n{s}\n\nAction: {a}"
    return f"State:\n{s}\n\nAction: {a}"


@registry.register_wm("wm-base")
class WMBase(WorldModelBackend):


    name = "wm-base"

    def __init__(self, llm: Any, env_name: str, **kwargs: Any) -> None:
        super().__init__(llm=llm, env_name=env_name, **kwargs)
        self._base_prompt = _load_base_prompt(self.env_name)

    def _system_prompt(self) -> str:
        return self._base_prompt

    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        system = self._system_prompt()
        user = _build_user_msg(state, action, goal)
        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            system,
            user,
            max_tokens=self.wm_max_tokens,
        )
        if not ok or text is None:
            return {
                "prediction": None,
                "mean_logprob": mean_lp,
                "system": system,
                "error": "llm_failed",
            }
        return {
            "prediction": text.strip(),
            "mean_logprob": mean_lp,
            "system": system,
        }


def build_wm(name: str, *, llm: Any, env_name: str, **kwargs: Any) -> WorldModelBackend:

    key = (name or "").strip().lower()
    cls = registry.get_wm_class(key)
    if cls is None:
        known = sorted(registry.mapping.get("wm_name_mapping", {}).keys())
        raise KeyError(
            f"Unknown WM backend {name!r}. Registered: {known!r}. "
            f"Use one of the registered backend names."
        )
    return cls(llm=llm, env_name=env_name, **kwargs)
