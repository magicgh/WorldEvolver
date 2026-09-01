
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from common.registry import registry
from llm import load_llm
from utils.logging.agent_logger import AgentLogger

from ..base_task import BaseTask
from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    check_checkpoint_identity,
    config_bool,
    default_resume_path,
    evaluation_order,
    experiment_config_fingerprint,
    load_json_object,
    restore_token_usage,
    wm_protocol_name,
)
from data.word2world import Word2WorldDataset
from world_model import build_wm
from world_model.modules.metrics import cosine_batch, em_at_1, token_f1
from world_model.modules.selective_foresight import confidence_pct_from_mean_logprob

logger = AgentLogger(__name__)
_DEFAULT_SF_THRESHOLDS = (25.0, 50.0, 75.0, 100.0)
_LOG_EVERY_STEPS = 1


def _format_threshold_key(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _parse_threshold_sequence(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    out = sorted({max(1.0, min(100.0, float(p))) for p in parts})
    return out or list(default)


class Stage2Word2WorldTask(BaseTask):


    ENV_NAME: str = ""
    DEFAULT_TRAJ_LIMIT: Optional[int] = None

    def __init__(
        self,
        llm_name: str = "gpt",
        llm_config: Optional[Dict[str, Any]] = None,
        wm_name: str = "wm-base",
        wm_config: Optional[Dict[str, Any]] = None,
        max_trajs: Optional[int] = None,
        log_path: Optional[str] = None,
        trace_dir: Optional[str] = None,
        compute_cosine: bool = True,
        online_updates: bool = True,
        sf_thresholds: Optional[Sequence[float]] = None,
        word2world_data_dir: Optional[str] = None,
        word2world_from_hf: bool = True,
        shuffle_evaluation_order: bool = False,
        seed: int = 42,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
        llm: Any = None,
    ) -> None:
        super().__init__()
        if not self.ENV_NAME:
            raise ValueError("Stage2Word2WorldTask subclasses must set ENV_NAME")
        self.llm_name = llm_name
        self.llm_config = llm_config or {}
        self.wm_name = wm_name
        self.wm_config = dict(wm_config or {})
        self.max_trajs = max_trajs if max_trajs is not None else self.DEFAULT_TRAJ_LIMIT
        self.log_path = log_path
        self.trace_dir = trace_dir
        self.compute_cosine = bool(compute_cosine)
        self.online_updates = bool(online_updates)
        self.shuffle_evaluation_order = bool(shuffle_evaluation_order)
        self.seed = int(seed)
        self.evaluation_order: List[str] = []
        self.resume = bool(resume)
        self.sf_thresholds = _parse_threshold_sequence(
            sf_thresholds, _DEFAULT_SF_THRESHOLDS
        )
        self.resume_config_fingerprint = experiment_config_fingerprint(
            self.llm_config,
            self.wm_config,
            {
                "environment": self.ENV_NAME,
                "max_trajs": self.max_trajs,
                "compute_cosine": self.compute_cosine,
                "online_updates": self.online_updates,
                "sf_thresholds": self.sf_thresholds,
                "word2world_data_dir": word2world_data_dir,
                "word2world_from_hf": word2world_from_hf,
                "shuffle_evaluation_order": self.shuffle_evaluation_order,
                "seed": self.seed,
            },
        )
        self.dataset = Word2WorldDataset(
            self.ENV_NAME,
            data_dir=word2world_data_dir,
            from_hf=word2world_from_hf,
        )


        self.llm = llm if llm is not None else load_llm(llm_name, self.llm_config)
        self.wm = build_wm(
            wm_name,
            llm=self.llm,
            env_name=self.ENV_NAME,
            **self.wm_config,
        )

        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        if self.trace_dir is not None:
            os.makedirs(self.trace_dir, exist_ok=True)
        self.resume_checkpoint_path = checkpoint_path
        if self.resume:
            if self.trace_dir is None:
                raise ValueError("Stage 2 resume requires trace_dir")
            if self.resume_checkpoint_path is None:
                self.resume_checkpoint_path = default_resume_path(
                    self.trace_dir,
                    stage=2,
                    environment=self.ENV_NAME,
                    wm_name=self.wm_name,
                )

    def _trace_path(self) -> str:
        return os.path.join(
            self.trace_dir,
            f"stage2_{self.ENV_NAME}_{self.wm_name}_steps.jsonl",
        )

    def _reset_events_path(self) -> str:
        return os.path.join(
            self.trace_dir,
            f"stage2_{self.ENV_NAME}_{self.wm_name}_reset_events.jsonl",
        )

    def _resume_identity(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "stage": 2,
            "environment": self.ENV_NAME,
            "wm_name": self.wm_name,
            "max_trajs": self.max_trajs,
            "online_updates": self.online_updates,
            "seed": int(getattr(self, "seed", 42)),
            "shuffle_evaluation_order": bool(
                getattr(self, "shuffle_evaluation_order", False)
            ),
            "evaluation_order": list(getattr(self, "evaluation_order", ())),
            "llm_engine": getattr(self.llm, "engine", None),
            "config_fingerprint": getattr(self, "resume_config_fingerprint", None),
        }

    def _ordered_trajectories(self):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        trajectory_ids: List[str] = []
        for step in self.dataset.iter_steps(max_trajs=self.max_trajs):
            traj_id = str(step["traj_id"])
            if traj_id not in grouped:
                grouped[traj_id] = []
                trajectory_ids.append(traj_id)
            grouped[traj_id].append(step)
        trajectory_ids = evaluation_order(
            trajectory_ids,
            shuffle=bool(getattr(self, "shuffle_evaluation_order", False)),
            seed=int(getattr(self, "seed", 42)),
        )
        return [(traj_id, grouped[traj_id]) for traj_id in trajectory_ids]

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        if not os.path.isfile(path):
            return []
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL record {line_number} is not an object: {path}"
                    )
                records.append(value)
        return records

    def _load_resume_state(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        if not getattr(self, "resume", False):
            return [], [], []
        checkpoint_path = self.resume_checkpoint_path
        trace_path = self._trace_path()
        if not os.path.isfile(checkpoint_path):
            if os.path.isfile(trace_path) and os.path.getsize(trace_path) > 0:
                raise RuntimeError(
                    "Stage 2 resume requested but trace exists without a checkpoint: "
                    f"{checkpoint_path}"
                )
            return [], [], []
        payload = load_json_object(checkpoint_path)
        check_checkpoint_identity(
            payload, self._resume_identity(), stage_label="Stage 2"
        )
        completed = [str(value) for value in payload.get("completed_trajectories", [])]
        completed_set = set(completed)
        per_step = [
            record
            for record in self._read_jsonl(trace_path)
            if str(record.get("traj_id")) in completed_set
        ]
        if len(per_step) != int(payload.get("n_steps", -1)):
            raise ValueError("Stage 2 checkpoint and trace disagree on completed steps")
        reset_events = [
            event
            for event in self._read_jsonl(self._reset_events_path())
            if str(event.get("traj_id")) in completed_set
        ]
        self.wm.load_checkpoint_state(payload.get("wm_state"))
        restore_token_usage(self.llm, payload.get("token_usage"))
        return per_step, reset_events, completed

    def _save_resume_state(
        self,
        per_step: Sequence[Dict[str, Any]],
        reset_events: Sequence[Dict[str, Any]],
        completed_trajectories: Sequence[str],
        *,
        complete: bool,
    ) -> None:
        if not getattr(self, "resume", False):
            return
        atomic_write_jsonl(self._trace_path(), per_step)
        atomic_write_jsonl(self._reset_events_path(), reset_events)
        atomic_write_json(
            self.resume_checkpoint_path,
            {
                **self._resume_identity(),
                "completed_trajectories": list(completed_trajectories),
                "n_steps": len(per_step),
                "complete": bool(complete),
                "wm_state": self.wm.checkpoint_state(),
                "token_usage": getattr(self.llm, "total_usage", None),
            },
        )

    def evaluate(self) -> Dict[str, Any]:
        trajectories = self._ordered_trajectories()
        self.evaluation_order = [traj_id for traj_id, _steps in trajectories]
        per_step, reset_events, completed_trajectories = self._load_resume_state()
        completed_set = set(completed_trajectories)
        em_running = sum(float(record["em"]) for record in per_step)
        f1_running = sum(float(record["token_f1"]) for record in per_step)
        n_steps = len(per_step)
        last_traj: Optional[str] = None

        if getattr(self, "resume", False) and not os.path.isfile(
            self.resume_checkpoint_path
        ):
            self._save_resume_state(
                per_step,
                reset_events,
                completed_trajectories,
                complete=False,
            )

        ordered_steps = (
            step for _traj_id, steps in trajectories for step in steps
        )
        for step in ordered_steps:
            traj_id = str(step["traj_id"])
            if traj_id in completed_set:
                continue
            if traj_id != last_traj:
                if last_traj is not None:
                    completed_trajectories.append(last_traj)
                    completed_set.add(last_traj)
                    self._save_resume_state(
                        per_step,
                        reset_events,
                        completed_trajectories,
                        complete=False,
                    )
                self.wm.reset()
                clear_memory_if = getattr(self.wm, "clear_memory_if", None)
                reset = (
                    clear_memory_if("trajectory") if callable(clear_memory_if) else None
                )
                if reset is not None:
                    reset_events.append(
                        {
                            "event": "memory_reset",
                            "benchmark": "word2world",
                            "environment": self.ENV_NAME,
                            "protocol": wm_protocol_name(self.wm),
                            "traj_id": traj_id,
                            "trajectory_id": traj_id,
                            "task_id": step["task"],
                            **reset,
                        }
                    )
                last_traj = traj_id

            pred_dict = self.wm.predict(
                state=step["state"],
                action=step["action"],
                goal=step["task"],
            )
            prediction = pred_dict.get("prediction")
            foresight = pred_dict.get("foresight", prediction)
            scored_prediction = "" if prediction is None else str(prediction)
            em = em_at_1(scored_prediction, step["gold_next_state"])
            f1 = token_f1(scored_prediction, step["gold_next_state"])
            em_running += em
            f1_running += f1
            n_steps += 1

            record = {
                "event": "world_model_call",
                "protocol": wm_protocol_name(self.wm),
                "benchmark": "word2world",
                "environment": self.ENV_NAME,
                "traj_id": traj_id,
                "trajectory_id": traj_id,
                "task_id": step["task"],
                "step_idx": step["step_idx"],
                "call_idx": 1,
                "call_type": "draft_prediction",
                "task": step["task"],
                "action": step["action"],
                "prediction": prediction,
                "foresight": foresight,
                "gold_next_state": step["gold_next_state"],
                "em": em,
                "token_f1": f1,
                "mean_logprob": pred_dict.get("mean_logprob"),
                "mean_token_logprob": pred_dict.get("mean_logprob"),
                "confidence_q": (
                    math.exp(float(pred_dict["mean_logprob"]))
                    if pred_dict.get("mean_logprob") is not None
                    else None
                ),
                "sf_confidence_pct": pred_dict.get("sf_confidence_pct"),
                "sf_gate": pred_dict.get("sf_gate"),
                "n_retrieved": pred_dict.get("n_retrieved"),
                "episodic_retrieved_count": pred_dict.get("n_retrieved"),
                "retrieved_indices": pred_dict.get("retrieved_indices"),
                "retrieved_source_tasks": pred_dict.get("retrieved_source_tasks"),
                "n_rules": pred_dict.get("n_rules"),
                "semantic_stored_rule_count": pred_dict.get("n_rules"),
                "active_rules": pred_dict.get("active_rules"),
                "semantic_active_rule_count": pred_dict.get("active_rules"),
                "rendered_rules": pred_dict.get("rendered_rules"),
                "semantic_rendered_rule_count": pred_dict.get("rendered_rules"),
                "k_me": getattr(self.wm, "top_k", 0),
                "k_ms": getattr(self.wm, "batch_k", 0),
                "episodic_store_size_before": pred_dict.get("episodic_store_size"),
                "episodic_retriever": pred_dict.get("episodic_retriever"),
                "episodic_block_tokens": pred_dict.get("episodic_block_tokens"),
                "semantic_block_tokens": pred_dict.get("semantic_block_tokens"),
                "total_prompt_tokens": pred_dict.get("total_prompt_tokens"),
                "prompt_token_source": pred_dict.get("prompt_token_source"),
                "memory_block_token_source": pred_dict.get("memory_block_token_source"),
                "prediction_token_length": pred_dict.get("prediction_token_length"),
                "prediction_token_source": pred_dict.get("prediction_token_source"),
                "embedder_failed": pred_dict.get("embedder_failed"),
                "error": pred_dict.get("error"),
            }
            per_step.append(record)

            if self.online_updates and prediction is not None:
                self.wm.update(
                    state=step["state"],
                    action=step["action"],
                    prediction=str(prediction),
                    gold_next_state=step["gold_next_state"],
                    info={"task": step["task"]},
                )
            get_counts = getattr(self.wm, "memory_counts", None)
            counts = get_counts() if callable(get_counts) else {}
            record.update(
                {
                    "episodic_store_size_after": counts.get("n_records"),
                    "semantic_rule_count_after": counts.get("n_rules"),
                    "pending_mismatch_count_after": counts.get("pending"),
                }
            )

            if n_steps % _LOG_EVERY_STEPS == 0:
                logger.info(
                    "stage2/%s wm=%s step %d traj=%s EM=%.3f F1=%.3f",
                    self.ENV_NAME,
                    self.wm_name,
                    n_steps,
                    traj_id,
                    em_running / n_steps,
                    f1_running / n_steps,
                )

        if last_traj is not None:
            completed_trajectories.append(last_traj)
            completed_set.add(last_traj)
        self._save_resume_state(
            per_step,
            reset_events,
            completed_trajectories,
            complete=True,
        )

        if self.compute_cosine and per_step:
            cos_vec = cosine_batch(
                (r["prediction"] for r in per_step),
                (r["gold_next_state"] for r in per_step),
            )
            if cos_vec is not None:
                for r, c in zip(per_step, cos_vec):
                    r["cosine"] = c
            else:
                logger.info(
                    "stage2/%s cosine skipped (embedder unavailable)", self.ENV_NAME
                )

        summary = self._summarize(
            per_step,
            sf_thresholds=getattr(self, "sf_thresholds", _DEFAULT_SF_THRESHOLDS),
        )
        summary["evaluation_order"] = {
            "shuffled": bool(getattr(self, "shuffle_evaluation_order", False)),
            "seed": int(getattr(self, "seed", 42)),
            "trajectory_ids": list(self.evaluation_order),
        }
        summary["n_reset_events"] = len(reset_events)
        self._dump(per_step, summary, reset_events)
        return summary

    @staticmethod
    def _summarize(
        per_step: List[Dict[str, Any]],
        sf_thresholds: Sequence[float] = _DEFAULT_SF_THRESHOLDS,
    ) -> Dict[str, Any]:
        if not per_step:
            return {
                "n_steps": 0,
                "em": 0.0,
                "token_f1": 0.0,
                "cosine": None,
                "sf_sweep": {},
            }
        n = len(per_step)
        em = sum(r["em"] for r in per_step) / n
        f1 = sum(r["token_f1"] for r in per_step) / n
        cos_vals = [r["cosine"] for r in per_step if r.get("cosine") is not None]
        cosine = sum(cos_vals) / len(cos_vals) if cos_vals else None
        return {
            "n_steps": n,
            "em": em,
            "token_f1": f1,
            "cosine": cosine,
            "sf_sweep": Stage2Word2WorldTask._summarize_sf_threshold(
                per_step,
                sf_thresholds,
            ),
        }

    @staticmethod
    def _summarize_sf_threshold(
        per_step: List[Dict[str, Any]],
        thresholds: Sequence[float] = _DEFAULT_SF_THRESHOLDS,
    ) -> Dict[str, Dict[str, Any]]:
        scored = [
            {
                **r,
                "sf_confidence_pct": confidence_pct_from_mean_logprob(
                    float(r["mean_logprob"])
                ),
            }
            for r in per_step
            if r.get("mean_logprob") is not None and r.get("prediction") is not None
        ]
        if not scored:
            return {}
        n_total = len(per_step)
        ranked = sorted(
            scored,
            key=lambda row: float(row["sf_confidence_pct"]),
            reverse=True,
        )
        out: Dict[str, Dict[str, Any]] = {}
        for pct in _parse_threshold_sequence(thresholds, _DEFAULT_SF_THRESHOLDS):
            keep_n = min(len(ranked), max(1, math.ceil(len(ranked) * pct / 100.0)))
            kept = ranked[:keep_n]
            cos_vals = [r["cosine"] for r in kept if r.get("cosine") is not None]
            conf_vals = [float(r["sf_confidence_pct"]) for r in kept]
            out[f"top_{_format_threshold_key(pct)}pct"] = {
                "retention_pct": pct,
                "confidence_threshold_pct": min(conf_vals),
                "n_kept": keep_n,
                "n_scored": len(ranked),
                "scored_coverage": keep_n / len(ranked),
                "coverage": keep_n / max(1, n_total),
                "em": (sum(float(r["em"]) for r in kept) / keep_n if kept else None),
                "token_f1": (
                    sum(float(r["token_f1"]) for r in kept) / keep_n if kept else None
                ),
                "cosine": (
                    sum(float(v) for v in cos_vals) / len(cos_vals)
                    if cos_vals
                    else None
                ),
                "mean_confidence_pct": (
                    sum(conf_vals) / len(conf_vals) if conf_vals else None
                ),
            }
        return out

    def _dump(
        self,
        per_step: List[Dict[str, Any]],
        summary: Dict[str, Any],
        reset_events: Sequence[Dict[str, Any]] = (),
    ) -> None:
        if self.log_path is None:
            return
        summary_path = os.path.join(
            self.log_path,
            f"stage2_{self.ENV_NAME}_{self.wm_name}_summary.json",
        )
        atomic_write_json(
            summary_path,
            {"wm_name": self.wm_name, "env_name": self.ENV_NAME, **summary},
        )
        if self.trace_dir is not None:
            atomic_write_jsonl(self._trace_path(), per_step)
            if reset_events:
                atomic_write_jsonl(self._reset_events_path(), reset_events)
        logger.finish(
            "stage2/%s wm=%s n_steps=%d EM=%.3f F1=%.3f cos=%s -> %s",
            self.ENV_NAME,
            self.wm_name,
            summary["n_steps"],
            summary["em"],
            summary["token_f1"],
            f"{summary['cosine']:.3f}" if summary["cosine"] is not None else "n/a",
            summary_path,
        )

    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):


        wm_name = agent_config.get("name", "wm-base")
        wm_config = agent_config.get("wm_config", {}) or {}
        max_trajs = run_config.get("max_trajs")
        if max_trajs is not None:
            max_trajs = int(max_trajs)
        compute_cosine = config_bool(run_config.get("compute_cosine", True))
        online_updates = config_bool(run_config.get("online_updates", True))
        sf_thresholds = _parse_threshold_sequence(
            run_config.get("sf_thresholds"),
            _DEFAULT_SF_THRESHOLDS,
        )
        word2world_data_dir = run_config.get("word2world_data_dir")
        word2world_from_hf = config_bool(
            run_config.get("word2world_from_hf", run_config.get("from_hf", True))
        )
        resume = config_bool(run_config.get("resume", False))
        shuffle_evaluation_order = config_bool(
            agent_config.get(
                "shuffle_evaluation_order",
                run_config.get("shuffle_evaluation_order", False),
            )
        )
        seed = int(agent_config.get("seed", run_config.get("seed", 42)))
        return cls(
            llm_name=llm_config.get("name", "gpt"),
            llm_config=llm_config,
            wm_name=wm_name,
            wm_config=wm_config,
            max_trajs=max_trajs,
            log_path=run_config.get("log_path"),
            trace_dir=run_config.get("trace_dir"),
            compute_cosine=compute_cosine,
            online_updates=online_updates,
            sf_thresholds=sf_thresholds,
            word2world_data_dir=word2world_data_dir,
            word2world_from_hf=word2world_from_hf,
            shuffle_evaluation_order=shuffle_evaluation_order,
            seed=seed,
            resume=resume,
            checkpoint_path=run_config.get("checkpoint_path"),
            llm=llm,
        )


@registry.register_task("stage2_word2world_alfworld")
class AlfWorldS2(Stage2Word2WorldTask):


    ENV_NAME: str = "alfworld"


@registry.register_task("stage2_word2world_scienceworld")
class ScienceworldS2(Stage2Word2WorldTask):


    ENV_NAME: str = "scienceworld"
