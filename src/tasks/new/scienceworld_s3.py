
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from common.registry import registry
from environment import load_environment

from utils.logging.agent_logger import AgentLogger
from utils.logging.logger import TaskLogger

from ..scienceworld import EvalScienceworld
from .common import (
    BestTrial,
    CLSWMTaskMixin,
    config_bool,
    evaluation_order,
    experiment_config_fingerprint,
    first_config_value as _first_config_value,
)
from world_model import build_wm

logger = AgentLogger(__name__)


@registry.register_task("stage3_scienceworld")
class ScienceworldS3(CLSWMTaskMixin, EvalScienceworld):


    def __init__(
        self,
        wm_name: str = "wm-base",
        wm_config: Optional[Dict[str, Any]] = None,
        num_trials: int = 5,
        persist_memory: bool = True,
        trace_dir: Optional[str] = None,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
        shuffle_evaluation_order: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        self.resume_config_fingerprint = experiment_config_fingerprint(
            kwargs.get("llm_config", {}),
            kwargs.get("agent_config", {}),
            kwargs.get("env_config", {}),
            {
                "wm_name": wm_name,
                "wm_config": wm_config or {},
                "num_trials": num_trials,
                "persist_memory": persist_memory,
                "max_num_steps": (kwargs.get("run_config") or {}).get(
                    "max_num_steps"
                ),
                "num_exams": (kwargs.get("run_config") or {}).get("num_exam"),
                "seed": seed,
                "shuffle_evaluation_order": shuffle_evaluation_order,
            },
        )
        user_log_path = self.sandbox_parent_task_logger(kwargs)

        super().__init__(**kwargs)

        self.wm_name = wm_name
        self.wm_config = dict(wm_config or {})
        self.num_trials = max(1, int(num_trials))
        self.persist_memory = bool(persist_memory)
        self.trace_dir = trace_dir
        self.resume = bool(resume)
        self.shuffle_evaluation_order = bool(shuffle_evaluation_order)
        self.seed = int(seed)
        self.evaluation_order: List[str] = []

        agent_llm = getattr(self.agent, "llm", None) or getattr(
            self.agent, "llm_model", None
        )
        if agent_llm is None:
            raise RuntimeError(
                "ScienceworldS3: agent does not expose .llm or .llm_model "
                "— cannot build WM."
            )


        self.itp_max_k = int(self.wm_config.pop("max_k", 0))


        if not wm_name:
            self.wm = None
            self.wm_name = "no-wm"
        else:
            self.wm = build_wm(
                wm_name,
                llm=agent_llm,
                env_name="scienceworld",
                **self.wm_config,
            )

        if self.trace_dir is not None:
            os.makedirs(self.trace_dir, exist_ok=True)
        self.configure_stage3_resume(
            enabled=self.resume,
            checkpoint_path=checkpoint_path,
            environment="scienceworld",
        )
        self.run_logger = TaskLogger(
            task_name=f"stage3_scienceworld_{self.wm_name}",
            log_path=self.prepare_task_log_path(user_log_path),
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
            resume=self.resume,
        )

    def _wm_state(self, observation: str) -> str:

        return f"{str(observation or '').strip()}\n{self.env.inventory()}".strip()


    def _run_single_trial(self, index, trial, task_name, var, modified_goal):


        self.env.load(task_name, var, simplificationStr=self.simplification_str)
        initialObs, _ = self.env.reset()
        init_obs = self._wm_state(initialObs)

        self.reset_or_reuse_agent(modified_goal, init_obs, trial)

        if self.wm is not None:
            self.reset_stage3_world_model_memory(
                "trajectory",
                task_index=index,
                task_id=modified_goal,
                trial=trial,
            )
            self.wm.reset(self.env)

        reward = 0.0
        last_reward = 0.0
        grounding_acc_count = 0
        score_change_record: list = []
        is_done = False

        trajectory = self.initial_trajectory(modified_goal, init_obs)
        logger.info(
            "Stage3 SW | Example {} | Trial {} | wm={} | Init obs: {}".format(
                index,
                trial,
                self.wm_name,
                init_obs,
            )
        )

        last_wm_state = init_obs

        steps_taken = 0
        for i in range(self.max_num_steps):
            prediction = None
            if self.wm is not None:
                self.set_stage3_planning_context(index, trial, i)
            if self.uses_itp_foresight():


                self.maybe_inject_foresight(
                    modified_goal,
                    "scienceworld",
                    trajectory,
                    i,
                    state=last_wm_state,
                )
                success, action = self.agent.run()
                if not success:
                    break
            elif self.wm is not None and self.should_call_step_prediction():
                success, action, prediction = self.choose_action_with_current_foresight(
                    run_agent=self.agent.run,
                    state=last_wm_state,
                    goal=modified_goal,
                    trajectory=trajectory,
                    step_id=i,
                )
                if not success:
                    break
            else:
                success, action = self.agent.run()
                if not success:
                    break
            trajectory.append({"Action": action, "id": i})

            observation, reward, is_done, info = self.env.step(action)
            if action in self.env.get_action_space(abstract=False):
                grounding_acc_count += 1

            trajectory.append({"Observation": observation, "id": i})
            trajectory.append({"Progress Rate": reward, "id": i})


            if self.wm is not None:
                self.update_wm_after_stage3_step(
                    state=last_wm_state,
                    action=action,
                    prediction=prediction,
                    gold_next_state=observation,
                    info={"task": modified_goal},
                )
                self.record_stage3_planning_step(index, trial, i)

            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward
            steps_taken = i
            last_wm_state = self._wm_state(observation)
            self.agent.update(action=action, state=observation)

            if is_done:
                env_details = {
                    "task_name": task_name,
                    "goal": self.agent.goal,
                    "difficulty": self.env.difficulty,
                    "trial": trial,
                    "wm_name": self.wm_name,
                }
                self.run_logger.log_example(
                    f"{index}-t{trial}",
                    True,
                    1.0,
                    grounding_acc_count / (i + 1),
                    score_change_record,
                    env_details,
                    trajectory,
                )
                self._dump_trace(index, trial, trajectory, success=True, progress=1.0)
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        env_details = {
            "task_name": task_name,
            "goal": self.agent.goal,
            "difficulty": self.env.difficulty,
            "trial": trial,
            "wm_name": self.wm_name,
        }
        example_prompt = self.safe_example_prompt()
        progress_rate = reward
        self.run_logger.log_example(
            f"{index}-t{trial}",
            False,
            progress_rate,
            grounding_acc_count / max(1, steps_taken + 1),
            score_change_record,
            env_details,
            trajectory,
            example_prompt,
        )
        self._dump_trace(
            index, trial, trajectory, success=False, progress=progress_rate
        )
        return (
            progress_rate,
            False,
            grounding_acc_count / max(1, steps_taken + 1),
            score_change_record,
            steps_taken,
        )

    def _dump_trace(
        self, index: int, trial: int, trajectory, success: bool, progress: float
    ):
        self.dump_trace_json(
            f"scienceworld_{self.wm_name}_{index:04d}_t{trial}.json",
            "wm_name",
            self.wm_name,
            index,
            trial,
            success,
            progress,
            trajectory,
        )


    def evaluate(self):
        self.env = load_environment("scienceworld", self.env_cfg)
        labels = self.env.labels
        if self.num_exams > len(labels):
            raise IndexError(
                f"Stage 3 requested {self.num_exams} ScienceWorld tasks, "
                f"but only {len(labels)} labels are available"
            )
        task_items = evaluation_order(
            list(labels.items())[: self.num_exams],
            shuffle=self.shuffle_evaluation_order,
            seed=self.seed,
        )
        self.evaluation_order = [str(label) for label, _value in task_items]
        checkpoint_progress, start_index, elapsed_before = (
            self.initialize_stage3_progress(
                environment="scienceworld",
                num_tasks=self.num_exams,
            )
        )
        scores = checkpoint_progress["scores"]
        grounding_accs = checkpoint_progress["grounding_accs"]
        srs = checkpoint_progress["srs"]
        score_state_records = checkpoint_progress["score_state_records"]
        difficulties = checkpoint_progress["difficulties"]
        srs_per_trial = checkpoint_progress["srs_per_trial"]
        segment_start = time.perf_counter()
        ran_task = False

        for index, (_label_key, v) in enumerate(task_items):
            if index < start_index:
                continue
            ran_task = True
            self.reset_stage3_world_model_memory(
                "task",
                task_index=index,
                task_id=str(_label_key),
                trial=0,
            )
            task_name = v["task_name"]
            var = v["var"]
            modified_goal = v["modified_goal"]
            logger.goal(
                "Stage3 SW | Example {} | Goal: task_name={} var={} {}".format(
                    index,
                    task_name,
                    var,
                    modified_goal,
                )
            )

            best = BestTrial()
            task_trial_results: List[bool] = []
            for trial in range(self.num_trials):
                score, done, gr_acc, record, num_steps = self._run_single_trial(
                    index,
                    trial,
                    task_name,
                    var,
                    modified_goal,
                )
                best.consider(score, done, gr_acc, record)
                task_trial_results.append(bool(done))
                logger.finish(
                    "Stage3 SW | Example {} | Trial {} | wm={} | done={} pr={:.3f} steps={}".format(
                        index,
                        trial,
                        self.wm_name,
                        done,
                        score,
                        num_steps + 1,
                    )
                )
                if best.done and not self.persist_memory:
                    break

            difficulties.append(self.env.difficulty)
            srs.append(1.0 if best.done else 0.0)
            scores.append(best.score)
            grounding_accs.append(best.grounding)
            score_state_records.append(best.record)
            srs_per_trial.append(task_trial_results)
            self.save_stage3_resume_state(
                environment="scienceworld",
                num_tasks=self.num_exams,
                next_task_index=index + 1,
                progress=checkpoint_progress,
                elapsed_seconds=(elapsed_before + time.perf_counter() - segment_start),
            )

        wall_clock_seconds = elapsed_before + (
            time.perf_counter() - segment_start if ran_task else 0.0
        )
        return self.summarize_stage_scores(
            srs,
            scores,
            grounding_accs,
            score_state_records,
            difficulties,
            srs_per_trial=srs_per_trial,
            token_usage=self._stage3_token_usage(),
            wall_clock_seconds=wall_clock_seconds,
        )


    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):
        agent_config = dict(agent_config or {})
        agent_config.setdefault("need_goal", True)
        llm_name = llm_config.get("name", "gpt")
        agent_name = agent_config.get("agent_name", agent_config.get("name", "WMReactAgent"))


        if agent_name == "WMReflactAgent" and not agent_config.get("instruction"):
            from agents.wm_reflact_agent import load_reflact_prompt, reflact_prompt_path

            load_reflact_prompt("scienceworld")
            agent_config = dict(agent_config)
            agent_config["init_prompt_path"] = str(reflact_prompt_path("scienceworld"))
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        log_path = run_config.get("log_path", None)

        if "max_num_steps" in run_config:
            run_config["max_num_steps"] = int(run_config["max_num_steps"])
        if "scienceworld_num_exam" in run_config:
            run_config["num_exam"] = int(run_config["scienceworld_num_exam"])
        if "num_exam" in run_config:
            run_config["num_exam"] = int(run_config["num_exam"])

        wm_name = _first_config_value(
            "wm_name",
            "wm-base",
            env_config,
            run_config,
            agent_config,
        )
        wm_config = _first_config_value(
            "wm_config",
            {},
            env_config,
            run_config,
            agent_config,
        )
        wm_config = wm_config or {}
        num_trials = int(
            run_config.get("num_trials", agent_config.get("num_trials", 5))
        )
        persist_memory = config_bool(
            run_config.get("persist_memory", agent_config.get("persist_memory", True))
        )
        trace_dir = run_config.get("trace_dir") or agent_config.get("trace_dir")
        resume = config_bool(run_config.get("resume", False))
        checkpoint_path = run_config.get("checkpoint_path")
        shuffle_evaluation_order = config_bool(
            agent_config.get(
                "shuffle_evaluation_order",
                run_config.get("shuffle_evaluation_order", False),
            )
        )
        seed = int(
            agent_config.get("seed", run_config.get("seed", env_config.get("seed", 42)))
        )

        return cls(
            llm_name=llm_name,
            llm_config=llm_config,
            agent_name=agent_name,
            agent_config=agent_config,
            env_config=env_config,
            run_config=run_config,
            llm=llm,
            baseline_dir=baseline_dir,
            log_path=log_path,
            wm_name=wm_name,
            wm_config=wm_config,
            num_trials=num_trials,
            persist_memory=persist_memory,
            trace_dir=trace_dir,
            resume=resume,
            checkpoint_path=checkpoint_path,
            shuffle_evaluation_order=shuffle_evaluation_order,
            seed=seed,
        )
