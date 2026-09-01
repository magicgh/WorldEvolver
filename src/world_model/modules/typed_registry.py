
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Sequence

from ..base import disable_llm_stop_for_wm


DEFAULT_TOKEN_BUDGET = 2048
_EXTRACTED_ENTRY_KEYS = {"key", "text", "source_examples"}
_PERSISTED_ENTRY_KEYS = {
    "key",
    "text",
    "evidence",
    "last_updated_step",
    "source_examples",
}


@dataclass
class RegistryEntry:
    key: str
    text: str


    evidence: float = 1.0
    last_updated_step: int = 0
    source_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RegistryEntry":
        if set(d) - _PERSISTED_ENTRY_KEYS:
            raise ValueError("Registry entry contains non-schema fields")
        return cls(
            key=str(d.get("key", "")),
            text=str(d.get("text", "")),
            evidence=float(d.get("evidence", 1)),
            last_updated_step=int(d.get("last_updated_step", 0)),
            source_examples=list(d.get("source_examples", [])),
        )


def _approx_tokens(text: str, chars_per_token: int = 4) -> int:

    return max(1, len(text) // max(1, int(chars_per_token)))


def _unique_key(entries: dict, raw_key: str) -> str:
    key = raw_key
    if key not in entries:
        return key
    base = key
    suffix = 2
    while key in entries:
        key = f"{base}.{suffix}"
        suffix += 1
    return key


class TypedRegistry:


    def __init__(self, *, filter_enabled: bool = True) -> None:
        self.entries: dict = {}
        self.global_step: int = 0


        self.filter_enabled: bool = filter_enabled


    def to_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TypedRegistry":
        out = cls()
        out.global_step = int(d.get("global_step", 0))
        for k, v in (d.get("entries") or {}).items():
            try:
                out.entries[k] = RegistryEntry.from_dict(v)
            except ValueError:
                continue
        return out

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TypedRegistry":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


    def register_or_merge(
        self,
        new_entry: RegistryEntry,
        *,
        contradicts_if_text_differs: bool = True,
    ) -> RegistryEntry:

        self.global_step += 1
        new_entry.last_updated_step = self.global_step


        existing = self.entries.get(new_entry.key)
        if existing is not None:


            if (not contradicts_if_text_differs) or self._equiv(existing.text, new_entry.text):
                existing.last_updated_step = self.global_step
                for src in new_entry.source_examples:
                    if src not in existing.source_examples:
                        existing.source_examples.append(src)
                return existing
            new_entry.key = _unique_key(self.entries, new_entry.key)
            self.entries[new_entry.key] = new_entry
            return new_entry


        self.entries[new_entry.key] = new_entry
        return new_entry

    def merge(self, new_entries: Iterable[RegistryEntry]) -> List[RegistryEntry]:
        return [self.register_or_merge(e) for e in new_entries]

    def clear(self) -> None:
        self.entries.clear()
        self.global_step = 0

    @staticmethod
    def _equiv(a: str, b: str) -> bool:
        norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
        return norm(a) == norm(b)


    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: str) -> Optional[RegistryEntry]:
        return self.entries.get(key)

    def renderable_count(self) -> int:
        return len(self._renderable_entries())


    def _score(self, e: RegistryEntry) -> tuple:

        return (
            -e.evidence,
            -e.last_updated_step,
        )

    def _renderable_entries(
        self,
        extra_entries: Sequence[RegistryEntry] = (),
    ) -> list[RegistryEntry]:
        entries = list(self.entries.values()) + list(extra_entries)
        if self.filter_enabled:
            return [e for e in entries if e.evidence > 0]
        return entries

    def uncapped_compile_tokens(
        self,
        extra_entries: Sequence[RegistryEntry] = (),
    ) -> int:

        pool = self._renderable_entries(extra_entries)
        if not pool:
            return 0
        used = _approx_tokens("## Frame Axioms and Persistence Rules\n")
        for e in sorted(pool, key=self._score):
            used += _approx_tokens(f"- {e.text.strip()}") + 1
        return used

    def compile_with_keys(
        self,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> tuple[str, set]:

        if not self.entries or token_budget <= 0:
            return "", set()


        pool = self._renderable_entries()
        if not pool:
            return "", set()
        ordered = sorted(pool, key=self._score)
        lines: List[str] = ["## Frame Axioms and Persistence Rules"]
        used = _approx_tokens(lines[0] + "\n")
        keys: set = set()
        for e in ordered:
            line = f"- {e.text.strip()}"
            cost = _approx_tokens(line) + 1
            if used + cost > token_budget:
                continue
            used += cost
            lines.append(line)
            keys.add(e.key)
        if len(lines) == 1:
            return "", set()
        return "\n".join(lines), keys

    def compile(self, *, token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
        return self.compile_with_keys(token_budget=token_budget)[0]

    def credit_mismatch(self, displayed_keys) -> None:

        if not displayed_keys or not self.entries:
            return
        keys = [k for k in displayed_keys if k in self.entries]
        if not keys:
            return
        share = 1.0 / float(len(keys))
        self.global_step += 1
        for key in keys:
            entry = self.entries.get(key)
            if entry is None:
                continue
            entry.evidence -= share
            entry.last_updated_step = self.global_step

    def credit_match(self, displayed_keys) -> None:

        if not displayed_keys or not self.entries:
            return
        keys = [k for k in displayed_keys if k in self.entries]
        if not keys:
            return
        share = 1.0 / float(len(keys))
        self.global_step += 1
        for key in keys:
            entry = self.entries.get(key)
            if entry is None:
                continue
            entry.evidence += share
            entry.last_updated_step = self.global_step


@dataclass
class MismatchSample:


    task: str
    state_obs: str
    action: str
    prediction: str
    gold_next_obs: str

    def to_extraction_block(self) -> str:
        return (
            f"task: {self.task}\n"
            f"state: {self.state_obs}\n"
            f"action: {self.action}\n"
            f"prediction: {self.prediction}\n"
            f"gold: {self.gold_next_obs}"
        )


def parse_extracted_json(raw: str) -> List[RegistryEntry]:

    raw = (raw or "").strip()
    if not raw:
        return []

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml
            obj = yaml.safe_load(raw)
        except Exception:
            obj = None
    if obj is None:
        return []
    if isinstance(obj, dict) and "entries" in obj:
        obj = obj["entries"]
    elif isinstance(obj, dict):

        obj = [obj]
    if not isinstance(obj, list):
        return []
    out: List[RegistryEntry] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        if set(item) - _EXTRACTED_ENTRY_KEYS:
            continue
        key = str(item.get("key", "")).strip()
        text = str(item.get("text", "")).strip()
        if not key or not text:
            continue
        out.append(
            RegistryEntry(
                key=key,
                text=text,
                source_examples=[str(s) for s in item.get("source_examples", [])],
            )
        )
    return out


def build_extraction_system_msg(env_name: str = "") -> str:
    env = str(env_name or "").strip() or "the current"
    return (
        f"You are extracting PRESERVATION RULES for the {env} text "
        "environment from world model prediction mismatches.\n\n"
        "A preservation rule says what should stay the same after an "
        "action. Look for cases where the prediction changed a fact, but "
        "the gold next observation shows that the fact did not change.\n\n"
        "Return one JSON object per general rule:\n"
        "- key: a short lowercase dot-separated name such as "
        "\"examine.object_location\". Use the same key for the same rule "
        "across different mismatches.\n"
        "- text: one clear sentence stating the rule generically for this "
        "environment, not for one specific object instance.\n\n"
        "Example:\n"
        "[{\"key\":\"examine.object_location\","
        "\"text\":\"Examining an object does not move it.\"}]\n\n"
        "Only output rules about facts that should NOT have changed. If "
        "the batch does not show a reusable preservation rule, output []. "
        "Do not invent rules to fill the list.\n\n"
        "Output strict JSON only: a single list of objects with key and "
        "text fields. No markdown fences and no commentary."
    )


def build_extraction_user_msg(
    samples: Sequence[MismatchSample],
    *,
    env_name: str = "",
) -> str:

    body_parts = []
    for i, s in enumerate(samples, 1):
        body_parts.append(f"Mismatch {i}:\n{s.to_extraction_block()}")
    env = str(env_name or "").strip() or "unspecified"
    return (
        f"Environment: {env}\n\n"
        "From the following mismatches, extract preservation rules "
        "(what does NOT change after the action). "
        "Return strict JSON only — a JSON list of objects.\n\n"
        + "\n\n".join(body_parts)
    )


def extract_entries(
    llm,
    samples: Sequence[MismatchSample],
    *,
    env_name: str = "",
    max_retries: int = 2,
    max_tokens: Optional[int] = None,
) -> List[RegistryEntry]:

    if not samples:
        return []
    system = build_extraction_system_msg(env_name)
    user = build_extraction_user_msg(samples, env_name=env_name)
    last_raw: str = ""


    _gen = getattr(llm, "generate_raw", None) or llm.generate
    for _ in range(max_retries):
        with disable_llm_stop_for_wm(llm, max_tokens=max_tokens):
            ok, raw = _gen(system, user)
        if ok and raw:
            last_raw = raw
            entries = parse_extracted_json(raw)
            if entries:
                return entries


            if "[" in raw:
                return []
    if last_raw:
        print(
            f"[typed_registry] extraction failed: could not parse JSON "
            f"after {max_retries} attempts. Raw={last_raw!r}",
            file=sys.stderr,
            flush=True,
        )
    return []
