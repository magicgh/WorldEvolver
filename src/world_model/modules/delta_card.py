
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Triple = Tuple[str, str, str]


@dataclass
class DeltaCard:


    action_raw: str
    next_obs_exemplar: str
    state_obs_raw: str = ""
    source_score: Optional[float] = None
    source_index: Optional[int] = None
    source_task: str = ""


def _norm_triple(triple: Sequence) -> Optional[Triple]:
    if not isinstance(triple, (list, tuple)) or len(triple) < 3:
        return None
    subject, predicate, obj = (
        str(triple[0]).strip().lower(),
        str(triple[1]).strip().lower(),
        str(triple[2]).strip().lower(),
    )
    if not subject:
        return None
    return (subject, predicate, obj)


def _to_triple_set(triples: Iterable[Sequence]) -> List[Triple]:

    out: set[Triple] = set()
    for triple in triples:
        normalized = _norm_triple(triple)
        if normalized is not None:
            out.add(normalized)
    return sorted(out)


def build_delta_card(
    *,
    action: str,
    next_observation_raw: str,
    source_score: Optional[float] = None,
    state_obs_raw: str = "",
    source_index: Optional[int] = None,
    source_task: str = "",
) -> DeltaCard:

    return DeltaCard(
        action_raw=action.strip(),
        next_obs_exemplar=(next_observation_raw or "").strip(),
        state_obs_raw=(state_obs_raw or "").strip(),
        source_score=source_score,
        source_index=source_index,
        source_task=str(source_task or ""),
    )


def render_delta_cards(cards: Sequence[DeltaCard]) -> str:

    if not cards:
        return ""
    lines: List[str] = ["## Retrieved similar past transitions"]
    for card in cards:
        lines.append("")
        if card.state_obs_raw:
            lines.append(f"Current state: {card.state_obs_raw}")
        lines.append(f"Action: {card.action_raw}")
        if card.next_obs_exemplar:
            lines.append(f"Next state: {card.next_obs_exemplar}")
    return "\n".join(lines)


__all__ = [
    "Triple",
    "DeltaCard",
    "build_delta_card",
    "render_delta_cards",
]
