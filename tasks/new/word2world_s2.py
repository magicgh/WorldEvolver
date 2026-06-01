
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from common.registry import registry
from llm import load_llm
from utils.logging.agent_logger import AgentLogger

from ..base_task import BaseTask
from .common import config_bool
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
        self.sf_thresholds = _parse_threshold_sequence(sf_thresholds, _DEFAULT_SF_THRESHOLDS)
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

    def evaluate(self) -> Dict[str, Any]:

        per_step: List[Dict[str, Any]] = []
        em_running = 0.0
        f1_running = 0.0
        n_steps = 0
        last_traj: Optional[str] = None

        for step in self.dataset.iter_steps(max_trajs=self.max_trajs):
            traj_id = step["traj_id"]
            if traj_id != last_traj:


                self.wm.reset()
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
                "traj_id": traj_id,
                "step_idx": step["step_idx"],
                "task": step["task"],
                "action": step["action"],
                "prediction": prediction,
                "foresight": foresight,
                "gold_next_state": step["gold_next_state"],
                "em": em,
                "token_f1": f1,
                "mean_logprob": pred_dict.get("mean_logprob"),
                "sf_confidence_pct": pred_dict.get("sf_confidence_pct"),
                "sf_gate": pred_dict.get("sf_gate"),
                "n_retrieved": pred_dict.get("n_retrieved"),
                "n_rules": pred_dict.get("n_rules"),
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

            if n_steps % _LOG_EVERY_STEPS == 0:
                logger.info(
                    "stage2/%s wm=%s step %d traj=%s EM=%.3f F1=%.3f",
                    self.ENV_NAME, self.wm_name, n_steps, traj_id,
                    em_running / n_steps, f1_running / n_steps,
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
                logger.info("stage2/%s cosine skipped (embedder unavailable)", self.ENV_NAME)

        summary = self._summarize(
            per_step,
            sf_thresholds=getattr(self, "sf_thresholds", _DEFAULT_SF_THRESHOLDS),
        )
        self._dump(per_step, summary)
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
        thresholds: Sequence[int] = _DEFAULT_SF_THRESHOLDS,
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
        out: Dict[str, Dict[str, Any]] = {}
        for pct in _parse_threshold_sequence(thresholds, _DEFAULT_SF_THRESHOLDS):
            kept = [r for r in scored if float(r["sf_confidence_pct"]) >= pct]
            keep_n = len(kept)
            cos_vals = [r["cosine"] for r in kept if r.get("cosine") is not None]
            conf_vals = [float(r["sf_confidence_pct"]) for r in kept]
            out[f"confidence_{_format_threshold_key(pct)}"] = {
                "confidence_threshold_pct": pct,
                "n_kept": keep_n,
                "coverage": keep_n / max(1, n_total),
                "em": (
                    sum(float(r["em"]) for r in kept) / keep_n
                    if kept else None
                ),
                "token_f1": (
                    sum(float(r["token_f1"]) for r in kept) / keep_n
                    if kept else None
                ),
                "cosine": (
                    sum(float(v) for v in cos_vals) / len(cos_vals)
                    if cos_vals else None
                ),
                "mean_confidence_pct": (
                    sum(conf_vals) / len(conf_vals) if conf_vals else None
                ),
            }
        return out

    def _dump(self, per_step: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        summary_path = os.path.join(
            self.log_path,
            f"stage2_{self.ENV_NAME}_{self.wm_name}_summary.json",
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {"wm_name": self.wm_name, "env_name": self.ENV_NAME, **summary},
                f,
                indent=2,
            )
        if self.trace_dir is not None:
            trace_path = os.path.join(
                self.trace_dir,
                f"stage2_{self.ENV_NAME}_{self.wm_name}_steps.jsonl",
            )
            with open(trace_path, "w", encoding="utf-8") as f:
                for r in per_step:
                    f.write(json.dumps(r) + "\n")
        logger.finish(
            "stage2/%s wm=%s n_steps=%d EM=%.3f F1=%.3f cos=%s -> %s",
            self.ENV_NAME, self.wm_name, summary["n_steps"], summary["em"],
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
            llm=llm,
        )


@registry.register_task("stage2_word2world_alfworld")
class AlfWorldS2(Stage2Word2WorldTask):


    ENV_NAME: str = "alfworld"


@registry.register_task("stage2_word2world_scienceworld")
class ScienceworldS2(Stage2Word2WorldTask):


    ENV_NAME: str = "scienceworld"
