
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..base import disable_llm_stop_for_wm
from .metrics import normalize


FACTORIZE_SYSTEM = (
    "You extract compact world-state triples from text observations. "
    "Return only valid JSON; do not explain."
)

FACTORIZE_USER_TEMPLATE = """Extract (subject, predicate, object) triples that describe the world state from the observation below. Output ONLY a JSON array of triples, each triple as a 3-element array of strings.

Environment: {env}

Observation: {obs}

Respond with JSON array only, no other text. Example output format:
[["fridge 1", "is", "open"], ["mug 1", "on", "countertop 2"], ["water 1", "state", "liquid"]]"""


_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
_OUTER_BRACKETS = re.compile(r"\[.*\]", re.S)


def parse_triples_checked(text: Optional[str]) -> Tuple[List[List[str]], bool]:

    if not text:
        return [], False
    cleaned = _CODE_FENCE.sub("", str(text)).strip()
    try:
        out = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _OUTER_BRACKETS.search(cleaned)
        if match is None:
            return [], False
        try:
            out = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], False
    if not isinstance(out, list):
        return [], False
    triples: List[List[str]] = []
    for item in out:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triples.append([str(item[0]), str(item[1]), str(item[2])])
    if out and not triples:
        return [], False
    return triples, True


def parse_triples(text: Optional[str]) -> List[List[str]]:

    triples, _ok = parse_triples_checked(text)
    return triples


def _generate_raw_text_checked(
    llm: Any,
    system: str,
    user: str,
    *,
    max_tokens: Optional[int] = None,
) -> Tuple[bool, str]:

    for name in ("generate_raw", "generate"):
        fn = getattr(llm, name, None)
        if fn is None:
            continue
        with disable_llm_stop_for_wm(llm, max_tokens=max_tokens):
            result = fn(system, user)
        if isinstance(result, tuple):
            if len(result) >= 2 and result[0]:
                return True, str(result[1] or "")
            return False, ""
        return True, str(result or "")
    return False, ""


class LLMFactorizer:


    def __init__(
        self,
        llm: Any,
        env_name: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.env_name = (env_name or "").lower()
        self.max_tokens = max_tokens
        self._cache: Dict[str, Tuple[List[List[str]], bool]] = {}
        self._failures = 0

    def factorize(self, observation: str) -> List[List[str]]:
        triples, _ok = self.factorize_checked(observation)
        return triples

    def factorize_checked(self, observation: str) -> Tuple[List[List[str]], bool]:

        obs = normalize(str(observation or ""))
        if len(obs) < 2:
            return [], True
        cached = self._cache.get(obs)
        if cached is not None:
            triples, ok = cached
            return [list(t) for t in triples], ok
        prompt = FACTORIZE_USER_TEMPLATE.format(
            env=self.env_name,
            obs=obs,
        )
        generated_ok, raw = _generate_raw_text_checked(
            self.llm,
            FACTORIZE_SYSTEM,
            prompt,
            max_tokens=self.max_tokens,
        )
        triples, parse_ok = parse_triples_checked(raw)
        ok = generated_ok and parse_ok
        if not ok:
            self._failures += 1
        self._cache[obs] = (triples, ok)
        return [list(t) for t in triples], ok

    def state_dict(self) -> Dict[str, Any]:
        return {
            "env_name": self.env_name,
            "cached_observations": len(self._cache),
            "factorization_failures": self._failures,
        }


__all__ = [
    "FACTORIZE_SYSTEM",
    "FACTORIZE_USER_TEMPLATE",
    "LLMFactorizer",
    "parse_triples_checked",
    "parse_triples",
]
