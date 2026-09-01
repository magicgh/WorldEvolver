
from __future__ import annotations

from typing import Any, Dict, Optional

from common.registry import registry as ab_registry

from .base import WorldModelBackend, llm_generate_with_logprobs


_ITP_SYSTEM = (
    "You are a world model performing in-imagination prediction. "
    "Given the agent's current state and the action it is about to take, "
    "imagine in detail what the agent will observe immediately after the "
    "action executes. Be concrete: describe the resulting state in the "
    "style of the environment's own text output, including any new objects "
    "that become visible, any state changes (open/closed, on/off, "
    "moved/picked up), and the agent's new vantage point.\n\n"
    "Output format: a single paragraph starting with \"Prediction:\" "
    "describing the imagined next observation."
)


_ITP_DECIDE_K_SYSTEM_TPL = (
    "You are a planning assistant. Your job is to decide how many steps of "
    "look-ahead are needed right now.\n"
    "Given a task instruction and the interaction history, output a single "
    "integer K in the range [0, {kmax}].\n"
    "Output ONLY the integer number, without any extra text."
)

_ITP_FORESIGHT_SYSTEM_TPL = (
    "You are a world model for the {env_name} environment. Given an "
    "action/observation history, imagine the next few steps, describing "
    "likely observations and key objects."
)


def _build_user_msg(state: str, action: str, goal: Optional[str]) -> str:


    s = (state or "").strip()
    a = (action or "").strip()
    g = (goal or "").strip()
    if g:
        return f"Goal: {g}\n\nCurrent state:\n{s}\n\nAction: {a}"
    return f"Current state:\n{s}\n\nAction: {a}"


@ab_registry.register_wm("wm-itp-i")
class WMItpI(WorldModelBackend):


    name = "wm-itp-i"

    def predict(
        self,
        state: str,
        action: str,
        goal: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        user = _build_user_msg(state, action, goal)


        ok, text, mean_lp = llm_generate_with_logprobs(
            self.llm,
            _ITP_SYSTEM,
            user,
            max_tokens=self.wm_max_tokens,
        )
        if not ok or text is None:
            return {
                "prediction": None,
                "mean_logprob": mean_lp,
                "system": _ITP_SYSTEM,
                "error": "llm_failed",
            }
        return {
            "prediction": text.strip(),
            "mean_logprob": mean_lp,
            "system": _ITP_SYSTEM,
        }


    def imagine_horizon(
        self,
        task: str,
        history: str,
        max_k: int = 5,
        env_name: Optional[str] = None,
    ) -> str:

        max_k = max(0, int(max_k))
        if max_k == 0:
            return ""


        k = self._decide_k(task, history, max_k)
        if k <= 0:
            return ""
        env_name = env_name or self.env_name or "WorldEvolver"
        sys_prompt = _ITP_FORESIGHT_SYSTEM_TPL.format(env_name=env_name)
        user = (
            f"History so far:\n{history or ''}\n\n"
            f"Predict the next {int(k)} step(s). Return a concise one-line plan "
            f"inside <foresight>...</foresight>, semicolon-separated. "
            f"Example: <foresight>1. Go to kitchen; 2. Open the fridge; 3. Take the milk</foresight>"
        )
        ok, text, _ = llm_generate_with_logprobs(
            self.llm,
            sys_prompt,
            user,
            max_tokens=self.wm_max_tokens,
        )
        if not ok or not text:
            return ""


        import re
        return re.sub(r"</?foresight\s*>", "", text, flags=re.IGNORECASE).strip()


    def _decide_k(self, task: str, history: str, max_k: int) -> int:

        sys_prompt = _ITP_DECIDE_K_SYSTEM_TPL.format(kmax=max_k)
        user = (
            f"Task instruction:\n{task or ''}\n\n"
            f"History trajectory (thoughts, actions, observations so far):\n"
            f"{history or ''}\n\n"
            f"Question: Output a single integer K in [0, {max_k}]."
        )
        ok, text, _ = llm_generate_with_logprobs(
            self.llm,
            sys_prompt,
            user,
            max_tokens=self.wm_max_tokens,
        )
        if not ok or not text:
            return 0
        try:
            return max(0, min(max_k, int(str(text).strip().split()[0])))
        except Exception:
            return 1 if max_k >= 1 else 0
