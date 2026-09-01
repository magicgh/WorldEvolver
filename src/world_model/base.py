
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional

from common.registry import registry

from .modules.metrics import configure_embedding_client
from .modules.vector_engine import configure_vector_index_device
from .prompt_templates import WORLD_MODEL_PROMPTS

RESET_SCOPES = frozenset({"none", "trajectory", "task"})


def _coerce_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc


DEFAULT_WM_MAX_TOKENS = 1000


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


def _text_token_count(llm: Any, text: Optional[str]) -> tuple[int, str]:
    value = str(text or "")
    encoding = getattr(llm, "_encoding", None)
    if encoding is not None:
        try:
            return len(encoding.encode(value)), "client_encoding"
        except Exception:
            pass
    return len(value.split()), "whitespace"


def prompt_telemetry(
    llm: Any,
    system: str,
    user: str,
    prediction: Optional[str],
    *,
    episodic_block: str = "",
    semantic_block: str = "",
) -> Dict[str, Any]:
    """Return additive prompt-footprint metadata for one completed WM call."""
    usage = getattr(llm, "last_usage", None) if prediction is not None else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    prompt_token_source = "api_usage" if isinstance(prompt_tokens, int) else None
    if prompt_tokens is None:
        counter = getattr(llm, "num_tokens_from_messages", None)
        if callable(counter):
            try:
                prompt_tokens = int(
                    counter(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
                    )
                )
                prompt_token_source = "client_counter"
            except Exception:
                prompt_tokens = None

    episodic_tokens, block_token_source = _text_token_count(llm, episodic_block)
    semantic_tokens, _ = _text_token_count(llm, semantic_block)
    prediction_tokens, prediction_token_source = _text_token_count(llm, prediction)
    return {
        "total_prompt_tokens": prompt_tokens,
        "prompt_token_source": prompt_token_source,
        "episodic_block_tokens": episodic_tokens,
        "semantic_block_tokens": semantic_tokens,
        "memory_block_token_source": block_token_source,
        "prediction_token_length": prediction_tokens,
        "prediction_token_source": prediction_token_source,
    }


class WorldModelBackend(ABC):


    name: str = "base"

    def __init__(self, llm: Any, env_name: str, **kwargs: Any) -> None:


        self.llm = llm
        self.env_name = (env_name or "").lower()
        self.use_history = _coerce_int(kwargs.get("use_history", 0), name="use_history")
        self.reset_scope = str(kwargs.get("reset_scope", "none")).strip().lower()
        if self.reset_scope not in RESET_SCOPES:
            raise ValueError(
                f"reset_scope must be one of {sorted(RESET_SCOPES)!r}; "
                f"got {self.reset_scope!r}"
            )
        raw_wm_max_tokens = kwargs.get(
            "wm_max_tokens", kwargs.get("max_tokens", DEFAULT_WM_MAX_TOKENS)
        )
        self.wm_max_tokens = max(
            1,
            _coerce_int(raw_wm_max_tokens, name="wm_max_tokens"),
        )
        unknown = sorted(
            set(kwargs)
            - {
                "use_history",
                "max_tokens",
                "wm_max_tokens",
                "reset_scope",
            }
        )
        if unknown:
            names = ", ".join(unknown)
            raise TypeError(f"{self.name} received unknown WM config key(s): {names}")
        self.config: Dict[str, Any] = {
            "use_history": self.use_history,
            "wm_max_tokens": self.wm_max_tokens,
            "reset_scope": self.reset_scope,
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

    def clear_memory(self) -> None:
        return None

    def memory_counts(self) -> Dict[str, int]:
        state = self.state_dict()
        counts = {
            key: int(state.get(key, 0) or 0)
            for key in ("n_records", "n_rules", "pending")
        }
        factorizer = state.get("factorizer")
        counts["factorizer_cached"] = (
            int(factorizer.get("cached_observations", 0) or 0)
            if isinstance(factorizer, dict)
            else 0
        )
        return counts

    def clear_memory_if(self, scope: str) -> Optional[Dict[str, Any]]:
        if self.reset_scope != scope:
            return None
        before = self.memory_counts()
        self.clear_memory()
        after = self.memory_counts()
        remaining = {key: value for key, value in after.items() if value != 0}
        if remaining:
            raise RuntimeError(
                f"{self.name} failed {scope!r} memory reset: {remaining!r}"
            )
        return {"scope": scope, "before": before, "after": after}

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "env_name": self.env_name,
            "config": dict(self.config),
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or int(state.get("version", 0)) != 1:
            raise ValueError("unsupported world-model checkpoint")
        if state.get("name") != self.name:
            raise ValueError(
                f"checkpoint WM mismatch: {state.get('name')!r} != {self.name!r}"
            )
        if str(state.get("env_name", "")).lower() != self.env_name:
            raise ValueError(
                "checkpoint environment mismatch: "
                f"{state.get('env_name')!r} != {self.env_name!r}"
            )
        if state.get("config") != self.config:
            raise ValueError("checkpoint world-model configuration mismatch")


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
        raise KeyError(
            f"Unknown WM prompt kind {prompt_kind!r}; known={known!r}"
        ) from exc
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

    embedding_keys = {
        "embed_base_url",
        "embed_model",
        "embed_api_key",
        "embed_timeout",
    }
    embedding_config = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key in embedding_keys
    }
    if embedding_config:
        configure_embedding_client(
            base_url=embedding_config.get("embed_base_url"),
            model=embedding_config.get(
                "embed_model",
                "Qwen/Qwen3-Embedding-8B",
            ),
            api_key=embedding_config.get("embed_api_key"),
            timeout=embedding_config.get("embed_timeout", 120.0),
        )
    vector_index_device = kwargs.pop("vector_index_device", None)
    if vector_index_device is not None:
        configure_vector_index_device(vector_index_device)

    key = (name or "").strip().lower()
    cls = registry.get_wm_class(key)
    if cls is None:
        known = sorted(registry.mapping.get("wm_name_mapping", {}).keys())
        raise KeyError(
            f"Unknown WM backend {name!r}. Registered: {known!r}. "
            f"Use one of the registered backend names."
        )
    return cls(llm=llm, env_name=env_name, **kwargs)
