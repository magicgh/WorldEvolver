
from __future__ import annotations

from typing import Optional


class ForesightMixin:


    def add_foresight(self, foresight: Optional[str]) -> None:

        if foresight is None:
            return
        text = str(foresight).strip().replace("\n", " ")
        if not text:
            return
        self.memory.append((self.PREDICTION_TAG, text))

    def _has_foresight_memory(self) -> bool:
        return any(
            tag == self.PREDICTION_TAG for tag, _ in (getattr(self, "memory", None) or [])
        )

    def _instruction_for_prompt(self) -> str:
        instruction = getattr(self, "instruction", "") or ""
        if (
            not self._has_foresight_memory()
            or self.FORESIGHT_INTRO in instruction
        ):
            return instruction
        stripped = instruction.rstrip()
        return (
            f"{stripped} {self.FORESIGHT_INTRO}"
            if stripped
            else self.FORESIGHT_INTRO
        )

    def make_prompt(self, *args, **kwargs):

        if not self._has_foresight_memory():
            return super().make_prompt(*args, **kwargs)
        saved_instruction = self.instruction
        self.instruction = self._instruction_for_prompt()
        try:


            return super().make_prompt(*args, **kwargs)
        finally:
            self.instruction = saved_instruction
