
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional


def confidence_pct_from_mean_logprob(mean_logprob: float) -> float:

    try:
        confidence = 100.0 * math.exp(float(mean_logprob))
    except (OverflowError, ValueError):
        return 0.0
    return max(0.0, min(100.0, confidence))


class SelectiveForesightGate:


    def __init__(self, confidence_pct: float, *, cell_name: str) -> None:
        pct = float(confidence_pct)
        if not 0 <= pct <= 100:
            raise ValueError(
                f"sf_confidence_pct must be in [0, 100]; got {confidence_pct!r}"
            )
        self.confidence_pct = pct
        self.cell_name = cell_name
        self._confidence_pcts: list[float] = []
        self._warned_no_logprobs = False

    def reset(self) -> None:
        self._confidence_pcts = []

    @property
    def count(self) -> int:
        return len(self._confidence_pcts)

    def apply(self, out: Dict[str, Any], mean_lp: Optional[float]) -> Dict[str, Any]:
        if self.confidence_pct <= 0:
            return out
        if mean_lp is None:
            if not self._warned_no_logprobs:
                warnings.warn(
                    f"{self.cell_name}: SF gate enabled "
                    f"(sf_confidence_pct={self.confidence_pct}) "
                    "but underlying WM did not return mean_logprob; SF gate falls "
                    "back to 'always pass'. Wire OpenAI-compatible token "
                    "logprobs through response.choices[0].logprobs and "
                    "surface them through llm.generate_with_logprobs to "
                    "enable the gate.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_no_logprobs = True
            out["sf_gate"] = "open_no_logprobs"
            return out
        confidence = confidence_pct_from_mean_logprob(float(mean_lp))
        self._confidence_pcts.append(confidence)
        out["sf_confidence_pct"] = confidence
        if confidence >= self.confidence_pct:
            out["sf_gate"] = "open"
            return out
        out["foresight"] = None
        out["sf_gate"] = "abstain"
        return out
