
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit


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


def evaluation_order(items: Iterable[Any], *, shuffle: bool, seed: int) -> List[Any]:
    """Return a deterministic, opt-in permutation of evaluation items."""
    ordered = list(items)
    if shuffle:
        random.Random(int(seed)).shuffle(ordered)
    return ordered


_FINGERPRINT_IGNORED_KEYS = {
    "api_key",
    "authorization",
    "checkpoint_path",
    "log_path",
    "resume",
    "trace_dir",
}


def experiment_config_fingerprint(*configs: Any) -> str:
    """Hash behavior-affecting config while ignoring outputs and secrets."""

    def normalize(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                str(name): normalize(item, str(name))
                for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if str(name).lower() not in _FINGERPRINT_IGNORED_KEYS
                and not str(name).lower().endswith("_api_key")
                and not str(name).lower().endswith("_token")
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, set):
            return sorted(normalize(item) for item in value)
        if key.lower() == "base_url" and value:
            parsed = urlsplit(str(value))
            host = (parsed.hostname or "").lower()
            netloc = (
                host
                if host in {"localhost", "127.0.0.1", "::1"}
                else parsed.netloc.lower()
            )
            return urlunsplit(
                (parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", "")
            )
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    encoded = json.dumps(normalize(configs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_replace(path: str, chunks: Iterable[str]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=os.path.splitext(path)[1],
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(chunks)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str, payload: Dict[str, Any], *, indent: int = 2) -> None:
    """Replace one JSON file atomically after flushing it to disk."""
    _atomic_replace(
        path,
        json.JSONEncoder(
            indent=indent,
            default=_json_default,
        ).iterencode(payload),
    )


def atomic_write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Replace one JSONL file atomically after flushing it to disk."""
    _atomic_replace(
        path,
        (json.dumps(record, default=_json_default) + "\n" for record in records),
    )


def load_json_object(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a JSON object: {path}")
    return payload


def wm_protocol_name(wm: Any) -> str:
    """Derive the WE-* protocol label from a world model's memory controls."""
    reset = getattr(wm, "reset_scope", "none") != "none"
    random_retrieval = (
        getattr(wm, "episodic_retriever", "jaccard_topk") == "uniform_random"
    )
    if reset and random_retrieval:
        return "WE-Reset-Random"
    if reset:
        return "WE-Reset"
    if random_retrieval:
        return "WE-Random"
    return "WE-Full"


def default_resume_path(
    trace_dir: str, *, stage: int, environment: str, wm_name: str
) -> str:
    return os.path.join(
        trace_dir,
        f".stage{stage}_{environment}_{wm_name}_resume.json",
    )


def check_checkpoint_identity(
    payload: Dict[str, Any], expected: Dict[str, Any], *, stage_label: str
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{stage_label} checkpoint {key} mismatch: "
                f"{payload.get(key)!r} != {value!r}"
            )


def restore_token_usage(llm: Any, usage: Any) -> None:
    total_usage = getattr(llm, "total_usage", None)
    if isinstance(usage, dict) and isinstance(total_usage, dict):
        total_usage.clear()
        total_usage.update({str(key): int(value) for key, value in usage.items()})


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
    # v4: traces gained "Draft Action" entries; pre-telemetry checkpoints
    # must not resume into mixed-schema trace sets.
    RESUME_CHECKPOINT_VERSION = 4
    STAGE3_PROGRESS_KEYS = (
        "scores",
        "grounding_accs",
        "srs",
        "score_state_records",
        "difficulties",
        "srs_per_trial",
    )

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
                return s[len(prefix) :].lstrip()
        return s

    @staticmethod
    def format_action_conditioned_foresight(action: Any, foresight: Any) -> str:

        action_text = str(action).strip().replace("\n", " ")
        foresight_text = CLSWMTaskMixin._strip_prediction_prefix(foresight).replace(
            "\n", " "
        )
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
        user_log_path = (
            run_config.get("log_path") if isinstance(run_config, dict) else None
        )
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

    def configure_stage3_resume(
        self,
        *,
        enabled: bool,
        checkpoint_path: Optional[str],
        environment: str,
    ) -> None:
        self.resume = bool(enabled)
        self.resume_checkpoint_path = checkpoint_path
        if not self.resume:
            return
        if self.resume_checkpoint_path is None:
            trace_dir = getattr(self, "trace_dir", None)
            if not trace_dir:
                raise ValueError("Stage 3 resume requires trace_dir or checkpoint_path")
            self.resume_checkpoint_path = default_resume_path(
                trace_dir, stage=3, environment=environment, wm_name=self.wm_name
            )

    def _stage3_agent_llm(self) -> Any:
        agent = getattr(self, "agent", None)
        return getattr(agent, "llm", None) or getattr(agent, "llm_model", None)

    def _stage3_token_usage(self) -> Any:
        return copy.deepcopy(
            getattr(self._stage3_agent_llm(), "total_usage", None)
        )

    def _stage3_task_logger(self) -> Any:
        return getattr(self, "run_logger", None) or getattr(
            self, "agentboard", None
        )

    def _stage3_resume_identity(
        self,
        *,
        environment: str,
        num_tasks: int,
    ) -> Dict[str, Any]:
        return {
            "version": self.RESUME_CHECKPOINT_VERSION,
            "stage": 3,
            "environment": environment,
            "wm_name": self.wm_name,
            "num_tasks": int(num_tasks),
            "num_trials": int(self.num_trials),
            "agent_type": type(self.agent).__name__,
            "max_num_steps": int(self.max_num_steps),
            "persist_memory": bool(self.persist_memory),
            "itp_max_k": int(getattr(self, "itp_max_k", 0) or 0),
            "seed": int(getattr(self, "seed", 42)),
            "shuffle_evaluation_order": bool(
                getattr(self, "shuffle_evaluation_order", False)
            ),
            "evaluation_order": list(getattr(self, "evaluation_order", ())),
            "llm_engine": getattr(self._stage3_agent_llm(), "engine", None),
            "config_fingerprint": getattr(self, "resume_config_fingerprint", None),
        }

    def _stage3_has_uncheckpointed_output(self, environment: str) -> bool:
        trace_dir = getattr(self, "trace_dir", None)
        prefix = f"{environment}_{self.wm_name}_"
        if trace_dir and os.path.isdir(trace_dir):
            if any(name.startswith(prefix) for name in os.listdir(trace_dir)):
                return True
        logger = self._stage3_task_logger()
        return any(
            path and os.path.isfile(path) and os.path.getsize(path) > 0
            for path in (
                getattr(logger, "log_path", None),
                getattr(logger, "log_summary_path", None),
            )
        )

    @staticmethod
    def _task_index_from_example_id(value: Any) -> Optional[int]:
        match = re.match(r"^(\d+)-t\d+$", str(value or ""))
        return int(match.group(1)) if match else None

    def _prune_stage3_outputs(self, environment: str, first_unfinished: int) -> None:
        """Remove artifacts not covered by the last committed checkpoint."""
        trace_dir = getattr(self, "trace_dir", None)
        pattern = re.compile(
            rf"^{re.escape(environment)}_{re.escape(self.wm_name)}_(\d{{4}})_t\d+\.json$"
        )
        if trace_dir and os.path.isdir(trace_dir):
            for name in os.listdir(trace_dir):
                match = pattern.match(name)
                if match and int(match.group(1)) >= first_unfinished:
                    os.unlink(os.path.join(trace_dir, name))

        task_logger = self._stage3_task_logger()
        detail_path = getattr(task_logger, "log_path", None)
        if detail_path and os.path.isfile(detail_path):
            with open(detail_path, "r", encoding="utf-8") as handle:
                text = handle.read()
            decoder = json.JSONDecoder()
            kept: List[Dict[str, Any]] = []
            pos = 0
            while pos < len(text):
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos >= len(text):
                    break
                try:
                    item, pos = decoder.raw_decode(text, pos)
                except json.JSONDecodeError:
                    break
                index = self._task_index_from_example_id(item.get("id"))
                if index is None or index < first_unfinished:
                    kept.append(item)
            _atomic_replace(
                detail_path,
                (json.dumps(item, indent=2) + "\n" for item in kept),
            )

        summary_path = getattr(task_logger, "log_summary_path", None)
        if summary_path and os.path.isfile(summary_path):
            kept_lines = []
            with open(summary_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    match = re.match(r"^\[EXP\] (\d+)-t\d+:", line)
                    if match is None or int(match.group(1)) < first_unfinished:
                        kept_lines.append(line)
            _atomic_replace(summary_path, kept_lines)

    def load_stage3_resume_state(
        self,
        *,
        environment: str,
        num_tasks: int,
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self, "resume", False):
            return None
        path = self.resume_checkpoint_path
        if not os.path.isfile(path):
            if self._stage3_has_uncheckpointed_output(environment):
                raise RuntimeError(
                    "Stage 3 resume requested but outputs exist without a checkpoint: "
                    f"{path}"
                )
            return None
        payload = load_json_object(path)
        expected = self._stage3_resume_identity(
            environment=environment,
            num_tasks=num_tasks,
        )
        check_checkpoint_identity(payload, expected, stage_label="Stage 3")
        first_unfinished = int(payload.get("next_task_index", 0))
        if not 0 <= first_unfinished <= int(num_tasks):
            raise ValueError("Stage 3 checkpoint has invalid next_task_index")
        progress = payload.get("progress")
        if not isinstance(progress, dict):
            raise ValueError("Stage 3 checkpoint progress must be an object")
        for key in self.STAGE3_PROGRESS_KEYS:
            values = progress.get(key)
            if not isinstance(values, list) or len(values) != first_unfinished:
                raise ValueError(
                    f"Stage 3 checkpoint progress length mismatch for {key}"
                )
        if self.wm is not None:
            self.wm.load_checkpoint_state(payload.get("wm_state"))
        metrics = payload.get("planning_metrics")
        if metrics is not None:
            self._stage3_planning_metrics = metrics
        restore_token_usage(self._stage3_agent_llm(), payload.get("token_usage"))
        self._prune_stage3_outputs(environment, first_unfinished)
        return payload

    def save_stage3_resume_state(
        self,
        *,
        environment: str,
        num_tasks: int,
        next_task_index: int,
        progress: Dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        if not getattr(self, "resume", False):
            return
        payload = {
            **self._stage3_resume_identity(
                environment=environment,
                num_tasks=num_tasks,
            ),
            "next_task_index": int(next_task_index),
            "complete": int(next_task_index) >= int(num_tasks),
            "elapsed_seconds": float(elapsed_seconds),
            "progress": copy.deepcopy(progress),
            "wm_state": (self.wm.checkpoint_state() if self.wm is not None else None),
            "planning_metrics": copy.deepcopy(
                getattr(self, "_stage3_planning_metrics", None)
            ),
            "token_usage": self._stage3_token_usage(),
        }
        atomic_write_json(self.resume_checkpoint_path, payload)

    def initialize_stage3_progress(
        self,
        *,
        environment: str,
        num_tasks: int,
    ) -> Tuple[Dict[str, List[Any]], int, float]:
        self.reset_stage3_planning_metrics()
        resume_state = self.load_stage3_resume_state(
            environment=environment,
            num_tasks=num_tasks,
        )
        state = resume_state or {}
        saved = state.get("progress") or {}
        progress = {
            key: list(saved.get(key) or []) for key in self.STAGE3_PROGRESS_KEYS
        }
        start_index = int(state.get("next_task_index", 0))
        elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        if resume_state is None:
            self.save_stage3_resume_state(
                environment=environment,
                num_tasks=num_tasks,
                next_task_index=0,
                progress=progress,
                elapsed_seconds=0.0,
            )
        return progress, start_index, elapsed_seconds

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

        def _imagine() -> Any:
            try:
                return wm.imagine_horizon(
                    goal,
                    state_history,
                    max_k,
                    env_name=env_name,
                )
            except TypeError:
                return wm.imagine_horizon(goal, state_history, max_k)

        foresight = self._stage3_wm_call("predict", _imagine)
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
            obj = action[len("put ") :].split(" in/on ", 1)[0].strip()
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
        inventory = ", ".join(
            cls.infer_alfworld_inventory_from_actions(admissible_actions)
        )
        return f"Observation: {obs}\nInventory: {inventory}".strip()

    def reset_stage3_planning_metrics(self) -> None:
        self._stage3_planning_metrics = {
            "agent_steps": 0,
            "predict_invocations": 0,
            "update_invocations": 0,
            "predict_llm_calls": 0,
            "update_llm_calls": 0,
            "retrieval_latency_ms": [],
            "memory_growth": [],
            "reset_events": [],
            "wm_calls": [],
        }
        self._stage3_call_context: Dict[str, Any] = {}

    def set_stage3_planning_context(
        self,
        task_index: int,
        trial: int,
        step_id: int,
    ) -> None:
        self._stage3_call_context = {
            "task": task_index,
            "trial": trial,
            "step": step_id,
            "call_idx": 0,
        }

    def reset_stage3_world_model_memory(
        self,
        scope: str,
        *,
        task_index: int,
        task_id: Optional[str] = None,
        trial: Optional[int] = None,
    ) -> None:
        wm = getattr(self, "wm", None)
        metrics = getattr(self, "_stage3_planning_metrics", None)
        if wm is None or metrics is None:
            return
        clear_memory_if = getattr(wm, "clear_memory_if", None)
        if not callable(clear_memory_if):
            return
        reset = clear_memory_if(scope)
        if reset is not None:
            metrics["reset_events"].append(
                {
                    "event": "memory_reset",
                    "benchmark": "agentboard",
                    "environment": getattr(wm, "env_name", None),
                    "protocol": self._stage3_protocol_name(),
                    "task_index": task_index,
                    "task_id": task_id,
                    "trial": trial,
                    "before_trial": trial + 1 if trial is not None else None,
                    **reset,
                }
            )

    def _stage3_llm_calls(self) -> Optional[int]:
        usage = getattr(getattr(self.wm, "llm", None), "total_usage", None)
        try:
            return int(usage["n_calls"])
        except (KeyError, TypeError, ValueError):
            return None

    def _stage3_wm_call(self, kind: str, call: Callable[[], Any]) -> Any:
        metrics = getattr(self, "_stage3_planning_metrics", None)
        if metrics is None:
            return call()
        metrics[f"{kind}_invocations"] += 1
        before = self._stage3_llm_calls()
        try:
            result = call()
        finally:
            after = self._stage3_llm_calls()
            if before is not None and after is not None:
                metrics[f"{kind}_llm_calls"] += max(0, after - before)
        if kind == "predict" and isinstance(result, dict):
            latency = result.get("retrieval_latency_ms")
            if latency is not None:
                metrics["retrieval_latency_ms"].append(float(latency))
            context = dict(getattr(self, "_stage3_call_context", {}) or {})
            call_idx = int(context.get("call_idx", 0)) + 1
            self._stage3_call_context["call_idx"] = call_idx
            context["call_idx"] = call_idx
            detail = {
                **context,
                "call_type": (
                    "draft_prediction" if call_idx == 1 else "action_aligned_prediction"
                ),
            }
            detail.update(
                {
                    "event": "world_model_call",
                    "protocol": self._stage3_protocol_name(),
                    "benchmark": "agentboard",
                    "environment": getattr(self.wm, "env_name", None),
                }
            )
            for key in (
                "n_retrieved",
                "retrieved_indices",
                "retrieved_source_tasks",
                "episodic_store_size",
                "episodic_retriever",
                "n_rules",
                "active_rules",
                "rendered_rules",
                "episodic_block_tokens",
                "semantic_block_tokens",
                "total_prompt_tokens",
                "prompt_token_source",
                "memory_block_token_source",
                "prediction_token_length",
                "mean_logprob",
                "sf_confidence_pct",
                "sf_gate",
                "retrieval_latency_ms",
                "error",
            ):
                if key in result:
                    detail[key] = result.get(key)
            detail.update(
                {
                    "k_me": getattr(self.wm, "top_k", 0),
                    "k_ms": getattr(self.wm, "batch_k", 0),
                    "episodic_store_size_before": result.get("episodic_store_size"),
                    "episodic_retrieved_count": result.get("n_retrieved"),
                    "semantic_stored_rule_count": result.get("n_rules"),
                    "semantic_active_rule_count": result.get("active_rules"),
                    "semantic_rendered_rule_count": result.get("rendered_rules"),
                    "mean_token_logprob": result.get("mean_logprob"),
                }
            )
            metrics["wm_calls"].append(detail)
        return result

    def _stage3_protocol_name(self) -> str:
        return wm_protocol_name(getattr(self, "wm", None))

    def update_wm_after_stage3_step(
        self,
        *,
        state: str,
        action: Any,
        prediction: Optional[str],
        gold_next_state: str,
        info: Optional[dict],
    ) -> None:
        if self.wm is None or prediction is None:
            return
        self._stage3_wm_call(
            "update",
            lambda: self.wm.update(
                state=state,
                action=action,
                prediction=str(prediction),
                gold_next_state=gold_next_state,
                info=info,
            ),
        )

    def record_stage3_planning_step(
        self,
        task_index: int,
        trial: int,
        step_id: int,
    ) -> None:
        metrics = getattr(self, "_stage3_planning_metrics", None)
        if metrics is None or self.wm is None:
            return
        metrics["agent_steps"] += 1
        state = self.wm.state_dict()
        snapshot = {
            key: state[key]
            for key in ("n_records", "n_rules", "pending")
            if key in state
        }
        metrics["memory_growth"].append(
            {
                "task": task_index,
                "trial": trial,
                "step": step_id,
                **snapshot,
            }
        )

    def summarize_stage3_planning_metrics(self) -> Optional[Dict[str, Any]]:
        metrics = getattr(self, "_stage3_planning_metrics", None)
        if not metrics or not metrics["agent_steps"]:
            return None
        steps = metrics["agent_steps"]
        wm_llm_calls = metrics["predict_llm_calls"] + metrics["update_llm_calls"]
        latencies = metrics["retrieval_latency_ms"]
        return {
            "agent_steps": steps,
            "wm_invocations_per_agent_step": (
                metrics["predict_invocations"] + metrics["update_invocations"]
            )
            / steps,
            "wm_llm_calls": {
                "predict": metrics["predict_llm_calls"],
                "update": metrics["update_llm_calls"],
                "per_agent_step": wm_llm_calls / steps,
            },
            "retrieval_latency_ms": {
                "calls": len(latencies),
                "total": sum(latencies),
                "mean": sum(latencies) / len(latencies) if latencies else None,
            },
            "memory_growth": metrics["memory_growth"],
            "reset_events": metrics["reset_events"],
            "wm_calls": metrics["wm_calls"],
        }

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

        wm_out = self._stage3_wm_call(
            "predict",
            lambda: wm.predict(state=state, action=draft_action, goal=goal),
        )
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
            draft_action,
            foresight,
        )
        self.restore_agent_runtime(draft_snapshot)
        agent.add_foresight(visible_foresight)
        trajectory.append({"Foresight": visible_foresight, "id": step_id})
        # Preserve the pre-foresight draft the moment foresight is shown:
        # without it, how often foresight changes the selected action is
        # unrecoverable. The outcome is derived by joining the same-id
        # "Action" entry — absent (rerun failed), equal (unchanged), or
        # different (changed). ITP foresight never emits this entry, so
        # draft-bearing steps are exactly the action-conditioned ones.
        trajectory.append({"Draft Action": draft_action, "id": step_id})

        success, action = run_agent()
        if not success:
            self.restore_agent_runtime(draft_snapshot)
            return False, action, prediction
        action = normalize(action)

        if action != draft_action:
            self.remove_foresight_entry(visible_foresight)
            wm_out = self._stage3_wm_call(
                "predict",
                lambda: wm.predict(state=state, action=action, goal=goal),
            )
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
        agent = getattr(self, "agent", None)
        atomic_write_json(
            path,
            {
                label_key: label_value,
                # Traces from different code versions or agents must never
                # be pooled silently (v4 introduced "Draft Action" entries;
                # React/ReflAct cells share the same wm_name).
                "trace_schema_version": self.RESUME_CHECKPOINT_VERSION,
                "agent_type": type(agent).__name__ if agent is not None else None,
                "index": index,
                "trial": trial,
                "success": success,
                "progress": progress,
                "trajectory": trajectory,
            },
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
    ) -> Tuple[
        List[float], List[float], List[float], List[Any], float, float, float, float
    ]:
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
            srs,
            scores,
            grounding_accs,
            score_state_records,
            easy_sr_value,
            hard_sr_value,
            easy_pr_value,
            hard_pr_value,
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
                vals = [row[trial_idx] for row in srs_per_trial if trial_idx < len(row)]
                per_trial_avg.append(sum(vals) / len(vals) if vals else None)
            payload["srs_per_trial"] = [[bool(b) for b in row] for row in srs_per_trial]
            payload["success_at_1"] = sum(
                bool(row and row[0]) for row in srs_per_trial
            ) / max(1, len(srs_per_trial))
            payload["best_at_5"] = sum(any(row[:5]) for row in srs_per_trial) / max(
                1, len(srs_per_trial)
            )
            payload["sr_at_trial"] = per_trial_avg

            cumulative: List[float] = []
            for k in range(max_l):
                hit = [any(row[: k + 1]) if row else False for row in srs_per_trial]
                cumulative.append(sum(hit) / len(hit) if hit else 0.0)
            payload["pass_at_k"] = cumulative
        if token_usage is not None:
            payload["token_usage"] = dict(token_usage)
        if wall_clock_seconds is not None:
            payload["wall_clock_seconds"] = float(wall_clock_seconds)
            if srs:
                payload["wall_clock_per_task_seconds"] = float(
                    wall_clock_seconds
                ) / len(srs)
        task_order = getattr(self, "evaluation_order", None)
        if task_order is not None:
            payload["evaluation_order"] = {
                "shuffled": bool(getattr(self, "shuffle_evaluation_order", False)),
                "seed": int(getattr(self, "seed", 42)),
                "task_ids": list(task_order),
            }
        wm_planning_metrics = self.summarize_stage3_planning_metrics()
        if wm_planning_metrics is not None:
            payload["wm_planning_metrics"] = wm_planning_metrics
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            print(
                f"[stage3-extras] failed to write {out_path}: {exc!r}",
                flush=True,
            )
