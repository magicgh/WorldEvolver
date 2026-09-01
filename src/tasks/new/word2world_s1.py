
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agents import load_agent
from common.registry import registry
from data.word2world import Word2WorldDataset
from llm import load_llm
from utils.logging.agent_logger import AgentLogger
from utils.logging.logger import TaskLogger
from world_model.oracle import build_oracle

from ..base_task import BaseTask
from .common import CLSWMTaskMixin, config_bool


logger = AgentLogger(__name__)
_DEFAULT_TRAJECTORIES = 195


def _norm_action(action: Any, *, env_name: str) -> str:
    text = " ".join(str(action or "").strip().split())
    if text.lower().startswith("action:"):
        text = text.split(":", 1)[1].strip()
    if env_name == "alfworld" and "put" in text:
        text = text.replace(" in ", " in/on ").replace(" on ", " in/on ")
    if text.endswith("."):
        text = text[:-1].strip()
    return " ".join(text.lower().split())


_USER_OBS_HEADER = "# User Environment Information (Displayed to User)"
_ORACLE_HEADER = "# Environment Information (Only visible to Assistant)"


def _strip_oracle_section(text: str) -> str:

    if text is None or text == "":
        return text
    if not isinstance(text, str):
        return text
    has_oracle = _ORACLE_HEADER in text
    user_idx = text.rfind(_USER_OBS_HEADER)
    if user_idx < 0:
        if has_oracle:
            raise ValueError(
                "Word2World observation has the assistant-only oracle "
                "header but no user-visible header; refusing to strip "
                "blindly because that would either leak oracle facts "
                "into the agent prompt or silently drop the entire "
                "observation."
            )
        return text


    after = text[user_idx + len(_USER_OBS_HEADER):].lstrip("\n").lstrip()
    if _ORACLE_HEADER in after:
        raise ValueError(
            "Word2World observation has an assistant-only oracle header "
            "AFTER the last user-visible header; refusing to fail open. "
            "The downstream agent prompt would otherwise contain the "
            "leaked oracle facts."
        )
    return after


class Stage1Word2WorldTask(CLSWMTaskMixin, BaseTask):


    ENV_NAME: str = ""

    def __init__(
        self,
        llm_name: str = "gpt",
        llm_config: Optional[Dict[str, Any]] = None,
        agent_name: str = "WMReactAgent",
        agent_config: Optional[Dict[str, Any]] = None,
        oracle_mode: str = "none",
        max_num_steps: int = 30,
        num_exams: Optional[int] = None,
        log_path: Optional[str] = None,
        baseline_dir: Optional[str] = None,
        trace_dir: Optional[str] = None,
        seed: int = 42,
        word2world_data_dir: Optional[str] = None,
        word2world_from_hf: bool = True,
        llm: Any = None,
    ) -> None:


        if not self.ENV_NAME:
            raise ValueError("Stage1Word2WorldTask subclasses must set ENV_NAME")

        self.llm_name = llm_name
        self.llm_config = llm_config or {}
        self.agent_name = agent_name
        self.agent_config = dict(agent_config or {})
        self.oracle_mode = (oracle_mode or "none").lower()
        self.max_num_steps = int(max_num_steps)
        self.num_exams = (
            _DEFAULT_TRAJECTORIES if num_exams is None else int(num_exams)
        )
        self.baseline_dir = baseline_dir or "data/baseline_results"
        self.trace_dir = trace_dir
        self.dataset = Word2WorldDataset(
            self.ENV_NAME,
            data_dir=word2world_data_dir,
            from_hf=word2world_from_hf,
        )
        self.llm = llm if llm is not None else load_llm(llm_name, self.llm_config)
        self.agent = load_agent(agent_name, self.agent_config, self.llm)
        self.oracle = build_oracle(
            self.oracle_mode,
            env_name=self.ENV_NAME,
            seed=int(seed),
            data_dir=word2world_data_dir,
            word2world_from_hf=word2world_from_hf,
        )
        self.run_logger = TaskLogger(
            task_name=f"stage1_{self.ENV_NAME}_{self.oracle_mode}",
            log_path=self.prepare_task_log_path(log_path),
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
        )
        if self.trace_dir is not None:
            os.makedirs(self.trace_dir, exist_ok=True)

    def _oracle_signal(self, teacher_action: str, gold_next_state: str) -> Optional[str]:
        prediction = self.oracle.predict(
            env=None,
            action=teacher_action,
            true_obs=gold_next_state,
            reward=1.0,
            done=False,
            info=None,
            history=getattr(self.agent, "memory", None),
        )
        if prediction is None:
            return None


        return self.format_observation_only_foresight(prediction)

    def _run_trajectory(self, index: int, traj: Dict[str, Any]):
        task = traj["task"]
        messages = traj["messages"]
        init_obs_full = (messages[0].get("content") or "").strip()


        init_obs = _strip_oracle_section(init_obs_full)
        traj_id = str(traj.get("traj_id") or f"traj_{index}")
        self.agent.reset(goal=task, init_obs=init_obs)
        self.oracle.reset(None)

        trajectory = self.initial_trajectory(task, init_obs)
        score_change_record: List[Any] = []
        matches = 0
        n_steps = 0
        current_state: Optional[str] = None
        pending_action: Optional[str] = None

        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                current_state = content
                continue
            if role == "user":
                pending_action = content
                continue
            if role != "assistant" or current_state is None or pending_action is None:
                continue

            teacher_action = pending_action
            gold_next_state = content
            foresight = self._oracle_signal(teacher_action, gold_next_state)
            if foresight is not None and hasattr(self.agent, "add_foresight"):
                self.agent.add_foresight(foresight)
                trajectory.append({"Foresight": foresight, "id": n_steps})

            success, action = self.agent.run()


            action_text = str(action) if action is not None else ""
            pred_norm = _norm_action(action_text, env_name=self.ENV_NAME)
            gold_norm = _norm_action(teacher_action, env_name=self.ENV_NAME)
            matched = bool(pred_norm and pred_norm == gold_norm)
            matches += 1 if matched else 0
            n_steps += 1
            progress = matches / n_steps
            score_change_record.append((n_steps - 1, progress))

            trajectory.append({"Action": action_text, "id": n_steps - 1})
            trajectory.append({"Teacher Action": teacher_action, "id": n_steps - 1})
            trajectory.append({"Action Match": matched, "id": n_steps - 1})
            trajectory.append({"Observation": gold_next_state, "id": n_steps - 1})
            trajectory.append({"Progress Rate": progress, "id": n_steps - 1})


            self.agent.update(
                action=teacher_action,
                state=_strip_oracle_section(gold_next_state),
            )
            current_state = gold_next_state
            pending_action = None

        score = matches / max(1, n_steps)
        env_details = {
            "task_name": traj_id,
            "goal": task,
            "difficulty": "easy",
            "oracle_mode": self.oracle_mode,
            "teacher_forced": True,
            "n_steps": n_steps,
        }
        self.run_logger.log_example(
            traj_id,
            score >= 1.0 and n_steps > 0,
            score,
            score,
            score_change_record,
            env_details,
            trajectory,
            self.safe_example_prompt(),
        )
        self.dump_trace_json(
            f"{self.ENV_NAME}_{self.oracle_mode}_{index:04d}.json",
            "oracle_mode",
            self.oracle_mode,
            index,
            0,
            score >= 1.0 and n_steps > 0,
            score,
            trajectory,
        )
        return score, score_change_record

    def evaluate(self):
        scores: List[float] = []
        records: List[Any] = []
        difficulties: List[str] = []
        for index, traj in enumerate(
            self.dataset.iter_trajectories(max_trajs=self.num_exams)
        ):
            score, record = self._run_trajectory(index, traj)
            scores.append(score)
            records.append(record)
            difficulties.append("easy")
            logger.finish(
                "stage1/%s oracle=%s traj=%s action_acc=%.3f",
                self.ENV_NAME,
                self.oracle_mode,
                traj["traj_id"],
                score,
            )
        return self.summarize_stage_scores(
            scores,
            scores,
            scores,
            records,
            difficulties,
        )

    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):
        oracle_mode = (
            env_config.get("oracle_mode")
            or run_config.get("oracle_mode")
            or agent_config.get("oracle_mode")
            or "none"
        )
        requested_trials = int(
            run_config.get("num_trials", agent_config.get("num_trials", 1))
        )
        if requested_trials != 1:
            logger.info(
                "Stage 1 is Word2World teacher-forced and uses L=1; "
                "ignoring requested num_trials=%d.",
                requested_trials,
            )
        num_exams = run_config.get("max_trajs", run_config.get("num_exam"))
        if num_exams is not None:
            num_exams = int(num_exams)
        word2world_from_hf_value = env_config.get(
            "word2world_from_hf",
            run_config.get(
                "word2world_from_hf",
                agent_config.get("word2world_from_hf", True),
            ),
        )
        return cls(
            llm_name=llm_config.get("name", "gpt"),
            llm_config=llm_config,
            agent_name=agent_config.get("name", "WMReactAgent"),
            agent_config=agent_config,
            oracle_mode=oracle_mode,
            max_num_steps=int(run_config.get("max_num_steps", 30)),
            num_exams=num_exams,
            log_path=run_config.get("log_path"),
            baseline_dir=run_config.get("baseline_dir", "data/baseline_results"),
            trace_dir=run_config.get("trace_dir"),
            seed=int(run_config.get("seed", env_config.get("seed", 42))),
            word2world_data_dir=(
                env_config.get("word2world_data_dir")
                or run_config.get("word2world_data_dir")
                or agent_config.get("word2world_data_dir")
            ),
            word2world_from_hf=config_bool(word2world_from_hf_value),
            llm=llm,
        )


@registry.register_task("stage1_word2world_alfworld")
class AlfWorldWord2WorldS1(Stage1Word2WorldTask):


    ENV_NAME = "alfworld"


@registry.register_task("stage1_word2world_scienceworld")
class ScienceworldWord2WorldS1(Stage1Word2WorldTask):


    ENV_NAME = "scienceworld"
