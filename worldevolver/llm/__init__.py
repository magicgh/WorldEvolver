from .openai_gpt import OPENAI_GPT
from common.registry import registry

__all__ = [
    "OPENAI_GPT",
    "load_llm",
]


def load_llm(name, config):
    cls = registry.get_llm_class(name)
    if cls is None:
        raise ImportError(
            f"LLM backend {name!r} is not registered or failed to import. "
            f"Registered: {sorted((registry.mapping.get('llm_name_mapping') or {}).keys())!r}. "
        )
    resolved_config = dict(config or {})
    resolved_config.setdefault("name", name)
    return cls.from_config(resolved_config)
