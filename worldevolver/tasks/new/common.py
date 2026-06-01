
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def first_config_value(key: str, default: Any, *configs: Dict[str, Any]) -> Any:

    for cfg in configs:
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
    return default


def config_bool(value: Any) -> bool:

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class BestTrial:


    done: bool = False
    score: float = 0.0
    grounding: float = 0.0
    record: List[Any] = field(default_factory=list)

    def consider(self, score: float, done: bool, grounding: float, record: List[Any]) -> None:

        if done and not self.done:
            self.done = True
        if score > self.score:
            self.score = score
            self.grounding = grounding
            self.record = record


class CLSWMTaskMixin:


    WM_BYPASS_ACTIONS = frozenset({"check valid actions"})

    @classmethod
    def should_bypass_wm_for_action(cls, action: Any) -> bool:

        normalized = " ".join(str(action).strip().lower().split())
        return normalized in cls.WM_BYPASS_ACTIONS

    @staticmethod
    def _strip_prediction_prefix(text: str) -> str:

        if text is None:
            return ""
        s = str(text).strip()
        if not s:
            return ""
        for prefix in ("Prediction:", "prediction:"):
            if s.startswith(prefix):
                return s[len(prefix):].lstrip()
        return s

    @staticmethod
    def format_action_conditioned_foresight(action: Any, foresight: Any) -> str:

        action_text = str(action).strip().replace("\n", " ")
        foresight_text = CLSWMTaskMixin._strip_prediction_prefix(foresight).replace("\n", " ")
        return (
            f'If you take action "{action_text}", world model predicts '
            f"the next observation: {foresight_text}"
        )

    @staticmethod
    def format_observation_only_foresight(foresight: Any) -> str:

        foresight_text = CLSWMTaskMixin._strip_prediction_prefix(foresight).replace("\n", " ")
        return f"World model predicts the next observation: {foresight_text}"

    @staticmethod
    def sandbox_parent_task_logger(kwargs: Dict[str, Any]) -> Optional[str]:


        run_config = kwargs.get("run_config")
        user_log_path = run_config.get("log_path") if isinstance(run_config, dict) else None
        if user_log_path is None:
            user_log_path = kwargs.get("log_path")
        if user_log_path is None:
            return None

        sandbox_log_path = os.path.join(user_log_path, "_parent_unused")
        os.makedirs(os.path.join(sandbox_log_path, "logs"), exist_ok=True)
        if isinstance(run_config, dict):
            kwargs["run_config"] = dict(run_config)
            kwargs["run_config"]["log_path"] = sandbox_log_path
        kwargs["log_path"] = sandbox_log_path
        return user_log_path

    @staticmethod
    def prepare_task_log_path(log_path: Optional[str]) -> str:

        if not log_path:
            raise ValueError(
                "CLS-WM task requires log_path or run_config.log_path for "
                "its Stage-specific TaskLogger."
            )
        os.makedirs(os.path.join(log_path, "logs"), exist_ok=True)
        return log_path

    def reset_or_reuse_agent(self, goal: str, init_obs: str, trial: int) -> None:

        self.agent.reset(goal=goal, init_obs=init_obs)

    def maybe_inject_foresight(
        self,
        goal: str,
        env_name: str,
        trajectory: List[Dict[str, Any]],
        step_id: int,
        state: Optional[str] = None,
    ) -> bool:


        wm = getattr(self, "wm", None)
        agent = getattr(self, "agent", None)
        max_k = int(getattr(self, "itp_max_k", 0) or 0)
        if (
            wm is None
            or max_k <= 0
            or not hasattr(wm, "imagine_horizon")
            or agent is None
            or not hasattr(agent, "add_foresight")
        ):
            return False

        state_history = self.itp_state_history(state)
        try:
            foresight = wm.imagine_horizon(goal, state_history, max_k, env_name=env_name)
        except TypeError:
            foresight = wm.imagine_horizon(goal, state_history, max_k)
        if not foresight:
            return False


        wrapped = (
            "World model predicts the next observation(s): " + str(foresight).strip().replace("\n", " ")
        )
        agent.add_foresight(wrapped)
        trajectory.append({"Foresight": wrapped, "id": step_id})
        return True

    @staticmethod
    def itp_state_history(state: Optional[str]) -> str:

        lines = [line.strip() for line in str(state or "").splitlines()]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def infer_alfworld_inventory_from_actions(actions: Iterable[Any]) -> List[str]:

        inventory: List[str] = []
        for action in actions or []:
            if not isinstance(action, str) or not action.startswith("put "):
                continue
            obj = action[len("put "):].split(" in/on ", 1)[0].strip()
            if obj and obj not in inventory:
                inventory.append(obj)
        return inventory

    @classmethod
    def itp_alfworld_state(
        cls,
        observation: Optional[str],
        admissible_actions: Iterable[Any],
    ) -> str:

        obs = cls.itp_state_history(observation)
        inventory = ", ".join(cls.infer_alfworld_inventory_from_actions(admissible_actions))
        return f"Observation: {obs}\nInventory: {inventory}".strip()

    def snapshot_agent_runtime(self) -> Dict[str, Any]:

        agent = getattr(self, "agent", None)
        if agent is None:
            return {}
        return {
            "memory": copy.deepcopy(getattr(agent, "memory", None)),
            "think_count": getattr(agent, "think_count", None),
            "force_action": getattr(agent, "force_action", None),
        }

    def restore_agent_runtime(self, snapshot: Dict[str, Any]) -> None:

        agent = getattr(self, "agent", None)
        if agent is None:
            return
        for field_name, value in snapshot.items():
            if value is not None:
                setattr(agent, field_name, copy.deepcopy(value))

    def remove_foresight_entry(self, foresight: Optional[str]) -> bool:

        if foresight is None:
            return False
        agent = getattr(self, "agent", None)
        memory = getattr(agent, "memory", None)
        if not isinstance(memory, list):
            return False
        text = str(foresight).strip().replace("\n", " ")
        tag = getattr(agent, "PREDICTION_TAG", "Foresight")
        for idx in range(len(memory) - 1, -1, -1):
            if memory[idx] == (tag, text):
                memory.pop(idx)
                return True
        return False

    def choose_action_with_current_foresight(
        self,
        run_agent: Callable[[], Tuple[bool, Any]],
        state: str,
        goal: str,
        trajectory: List[Dict[str, Any]],
        step_id: int,
        normalize_action: Optional[Callable[[Any], Any]] = None,
    ) -> Tuple[bool, Any, Optional[str]]:

        wm = getattr(self, "wm", None)
        agent = getattr(self, "agent", None)
        normalize = normalize_action or (lambda action: action)

        if wm is None:
            success, action = run_agent()
            return success, normalize(action), None

        draft_snapshot = self.snapshot_agent_runtime()
        success, draft_action = run_agent()
        if not success:
            return False, draft_action, None
        draft_action = normalize(draft_action)
        if self.should_bypass_wm_for_action(draft_action):
            return True, draft_action, None

        wm_out = wm.predict(state=state, action=draft_action, goal=goal)
        prediction = wm_out.get("prediction")
        foresight = wm_out.get("foresight", prediction)

        if (
            foresight is None
            or not str(foresight).strip()
            or agent is None
            or not hasattr(agent, "add_foresight")
        ):
            return True, draft_action, prediction

        visible_foresight = self.format_action_conditioned_foresight(
            draft_action, foresight,
        )
        self.restore_agent_runtime(draft_snapshot)
        agent.add_foresight(visible_foresight)
        trajectory.append({"Foresight": visible_foresight, "id": step_id})

        success, action = run_agent()
        if not success:
            self.restore_agent_runtime(draft_snapshot)
            return False, action, prediction
        action = normalize(action)

        if action != draft_action:
            self.remove_foresight_entry(visible_foresight)
            wm_out = wm.predict(state=state, action=action, goal=goal)
            prediction = wm_out.get("prediction")
        return True, action, prediction

    def uses_itp_foresight(self) -> bool:

        wm = getattr(self, "wm", None)
        return (
            int(getattr(self, "itp_max_k", 0) or 0) > 0
            and wm is not None
            and hasattr(wm, "imagine_horizon")
        )

    def should_call_step_prediction(self) -> bool:

        return not self.uses_itp_foresight()

    @staticmethod
    def initial_trajectory(goal: str, init_obs: str) -> List[Dict[str, Any]]:
        return [
            {"Goal": goal, "id": 0},
            {"Observation": init_obs, "id": 0},
        ]

    def safe_example_prompt(self) -> Any:
        try:
            return self.agent.get_example_prompt()
        except AttributeError:
            return None

    def dump_trace_json(
        self,
        filename: str,
        label_key: str,
        label_value: str,
        index: int,
        trial: int,
        success: bool,
        progress: float,
        trajectory: List[Dict[str, Any]],
    ) -> None:
        trace_dir = getattr(self, "trace_dir", None)
        if trace_dir is None:
            return
        path = os.path.join(trace_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    label_key: label_value,
                    "index": index,
                    "trial": trial,
                    "success": success,
                    "progress": progress,
                    "trajectory": trajectory,
                },
                f,
                indent=2,
            )

    def summarize_stage_scores(
        self,
        srs: Iterable[float],
        scores: Iterable[float],
        grounding_accs: Iterable[float],
        score_state_records: List[Any],
        difficulties: Iterable[str],
        srs_per_trial: Optional[List[List[bool]]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        wall_clock_seconds: Optional[float] = None,
    ) -> Tuple[List[float], List[float], List[float], List[Any], float, float, float, float]:
        srs = list(srs)
        scores = list(scores)
        grounding_accs = list(grounding_accs)
        difficulties = list(difficulties)

        sr = sum(srs) * 1.0 / len(srs)
        pr = sum(scores) * 1.0 / len(scores)
        gr = sum(grounding_accs) * 1.0 / len(grounding_accs)

        hard_sr = [s for s, d in zip(srs, difficulties) if d == "hard"]
        hard_sr_value = sum(hard_sr) / len(hard_sr) if hard_sr else 0
        hard_pr = [p for p, d in zip(scores, difficulties) if d == "hard"]
        hard_pr_value = sum(hard_pr) / len(hard_pr) if hard_pr else 0
        easy_sr = [s for s, d in zip(srs, difficulties) if d == "easy"]
        easy_sr_value = sum(easy_sr) / len(easy_sr) if easy_sr else 0
        easy_pr = [p for p, d in zip(scores, difficulties) if d == "easy"]
        easy_pr_value = sum(easy_pr) / len(easy_pr) if easy_pr else 0

        self.run_logger.log_summary(
            sr, pr, gr, score_state_records,
            hard_sr_value, hard_pr_value, easy_sr_value, easy_pr_value,
        )


        if (srs_per_trial is not None
                or token_usage is not None
                or wall_clock_seconds is not None):
            self._write_extras_summary(
                srs_per_trial=srs_per_trial,
                token_usage=token_usage,
                wall_clock_seconds=wall_clock_seconds,
                srs=srs,
            )

        return (
            srs, scores, grounding_accs, score_state_records,
            easy_sr_value, hard_sr_value, easy_pr_value, hard_pr_value,
        )

    def _write_extras_summary(
        self,
        *,
        srs_per_trial: Optional[List[List[bool]]],
        token_usage: Optional[Dict[str, int]],
        wall_clock_seconds: Optional[float],
        srs: List[float],
    ) -> None:


        log_path = getattr(self.run_logger, "log_path", None)
        task_name = getattr(self.run_logger, "task_name", "run")
        if not log_path:
            return
        results_root = os.path.dirname(os.path.dirname(log_path))
        out_path = os.path.join(results_root, f"run_extras_{task_name}.json")
        payload: Dict[str, Any] = {}
        if srs_per_trial is not None:


            max_l = max((len(row) for row in srs_per_trial), default=0)
            per_trial_avg: List[Optional[float]] = []
            for trial_idx in range(max_l):
                vals = [
                    row[trial_idx]
                    for row in srs_per_trial
                    if trial_idx < len(row)
                ]
                per_trial_avg.append(
                    sum(vals) / len(vals) if vals else None
                )
            payload["srs_per_trial"] = [
                [bool(b) for b in row] for row in srs_per_trial
            ]
            payload["sr_at_trial"] = per_trial_avg

            cumulative: List[float] = []
            for k in range(max_l):
                hit = [
                    any(row[: k + 1]) if row else False
                    for row in srs_per_trial
                ]
                cumulative.append(sum(hit) / len(hit) if hit else 0.0)
            payload["pass_at_k"] = cumulative
        if token_usage is not None:
            payload["token_usage"] = dict(token_usage)
        if wall_clock_seconds is not None:
            payload["wall_clock_seconds"] = float(wall_clock_seconds)
            if srs:
                payload["wall_clock_per_task_seconds"] = (
                    float(wall_clock_seconds) / len(srs)
                )
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            print(
                f"[stage3-extras] failed to write {out_path}: {exc!r}",
                flush=True,
            )
