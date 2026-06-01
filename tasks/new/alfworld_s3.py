
from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional

from common.registry import registry
from environment import load_environment

from utils.logging.agent_logger import AgentLogger
from utils.logging.logger import TaskLogger

from ..alfworld import Evalalfworld, prefixes
from .common import (
    BestTrial,
    CLSWMTaskMixin,
    config_bool,
    first_config_value as _first_config_value,
)
from world_model import build_wm

logger = AgentLogger(__name__)


@registry.register_task("stage3_alfworld")
class AlfWorldS3(CLSWMTaskMixin, Evalalfworld):


    def __init__(
        self,
        wm_name: str = "wm-base",
        wm_config: Optional[Dict[str, Any]] = None,
        num_trials: int = 5,
        persist_memory: bool = True,
        trace_dir: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):


        user_log_path = self.sandbox_parent_task_logger(kwargs)

        super().__init__(**kwargs)

        self.wm_name = wm_name
        self.wm_config = dict(wm_config or {})
        self.num_trials = max(1, int(num_trials))
        self.persist_memory = bool(persist_memory)
        self.trace_dir = trace_dir
        self.seed = int(seed)


        agent_llm = getattr(self.agent, "llm", None) or getattr(self.agent, "llm_model", None)
        if agent_llm is None:
            raise RuntimeError(
                "AlfWorldS3: agent does not expose .llm or .llm_model — "
                "cannot build WM. WM model must be the same model as the "
                "agent (per 04_baselines.md)."
            )


        self.itp_max_k = int(self.wm_config.pop("max_k", 0))


        if not wm_name:
            self.wm = None
            self.wm_name = "no-wm"
        else:
            self.wm = build_wm(
                wm_name, llm=agent_llm, env_name="alfworld", **self.wm_config,
            )

        self.run_logger = TaskLogger(
            task_name=f"stage3_alfworld_{self.wm_name}",
            log_path=self.prepare_task_log_path(user_log_path),
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
        )
        if self.trace_dir is not None:
            os.makedirs(self.trace_dir, exist_ok=True)


    def _run_single_trial(self, index, trial, ob, examples):


        init_ob = ob.split("\n")[0]
        goal = ob.split("\n")[1].split("Your task is to:")[1].strip()

        self.reset_or_reuse_agent(goal, init_ob, trial)

        if self.wm is not None:
            self.wm.reset(self.env)

        init_prompt_dict = copy.deepcopy(self.prompts)
        init_prompt_dict["examples"] = examples
        reward = 0.0
        last_reward = 0.0
        grounding_acc_count = 0
        score_change_record: list = []
        trajectory = self.initial_trajectory(goal, init_ob)
        logger.info(
            "Stage3 ALF | Example {} | Trial {} | wm={} | Init obs: {}".format(
                index, trial, self.wm_name, init_ob,
            )
        )


        last_obs = init_ob

        steps_taken = 0
        for i in range(self.max_num_steps):
            prediction = None


            valid_actions = self.env.get_action_space()
            wm_state = self.itp_alfworld_state(last_obs, valid_actions)
            if self.uses_itp_foresight():


                self.maybe_inject_foresight(
                    goal,
                    "alfworld",
                    trajectory,
                    i,
                    state=wm_state,
                )
                success, action = self.agent.run(init_prompt_dict=init_prompt_dict)
                if not success:
                    break
                action = self.parseAction(action)
            elif self.wm is not None and self.should_call_step_prediction():
                success, action, prediction = self.choose_action_with_current_foresight(
                    run_agent=lambda: self.agent.run(init_prompt_dict=init_prompt_dict),
                    state=wm_state,
                    goal=goal,
                    trajectory=trajectory,
                    step_id=i,
                    normalize_action=self.parseAction,
                )
                if not success:
                    break
            else:
                success, action = self.agent.run(init_prompt_dict=init_prompt_dict)
                if not success:
                    break
                action = self.parseAction(action)
            if action in valid_actions:
                grounding_acc_count += 1.0

            trajectory.append({"Action": action, "id": i})

            observation, reward, done, info = self.env.step(action)
            if "Task accomplished!" in observation and reward < 1.0:
                raise Exception("Task accomplished error")

            trajectory.append({"Observation": observation, "id": i})
            trajectory.append({"Progress Rate": reward, "id": i})


            if self.wm is not None and prediction is not None:
                self.wm.update(
                    state=wm_state,
                    action=action,
                    prediction=str(prediction),
                    gold_next_state=observation,
                    info={"task": goal},
                )

            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward
            self.agent.update(action=action, state=observation)
            last_obs = observation
            steps_taken = i

            if done:
                game_name = self.env.cur_task_name.split("/")[0]
                env_details = {
                    "task_name": game_name,
                    "goal": self.agent.goal,
                    "difficulty": self.env.difficulty,
                    "trial": trial,
                    "num_trials": self.num_trials,
                    "wm_name": self.wm_name,
                }
                self.run_logger.log_example(
                    f"{index}-t{trial}",
                    True,
                    reward,
                    grounding_acc_count / (i + 1),
                    score_change_record,
                    env_details,
                    trajectory,
                )
                self._dump_trace(index, trial, trajectory, success=True, progress=reward)
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        game_name = self.env.cur_task_name.split("/")[0]
        env_details = {
            "task_name": game_name,
            "goal": self.agent.goal,
            "difficulty": self.env.difficulty,
            "trial": trial,
            "num_trials": self.num_trials,
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
        self._dump_trace(index, trial, trajectory, success=False, progress=progress_rate)
        return progress_rate, False, grounding_acc_count / max(1, steps_taken + 1), score_change_record, steps_taken

    def _dump_trace(self, index: int, trial: int, trajectory, success: bool, progress: float):
        self.dump_trace_json(
            f"alfworld_{self.wm_name}_{index:04d}_t{trial}.json",
            "wm_name",
            self.wm_name,
            index,
            trial,
            success,
            progress,
            trajectory,
        )

    def _load_game_env(self, gamefile: str):
        env_cfg = copy.deepcopy(self.env_cfg)
        env_cfg["game_files"] = [gamefile]
        return load_environment("alfworld", env_cfg)

    def _close_env(self) -> None:
        close = getattr(getattr(self, "env", None), "close", None)
        if callable(close):
            close()


    def evaluate(self):


        catalog_env = load_environment("alfworld", self.env_cfg)
        try:
            game_files = list(getattr(catalog_env, "game_files", []))
        finally:
            close = getattr(catalog_env, "close", None)
            if callable(close):
                close()
        if self.num_exams > len(game_files):
            raise IndexError(
                f"Stage 3 requested {self.num_exams} ALFWorld games, "
                f"but only {len(game_files)} are available"
            )
        scores: list = []
        score_state_records: list = []
        grounding_accs: list = []
        srs: list = []
        difficulties: list = []


        srs_per_trial: List[List[bool]] = []
        wall_clock_start = time.perf_counter()


        for idx, gamefile in enumerate(game_files[: self.num_exams]):
            best = BestTrial()
            task_trial_results: List[bool] = []


            trial_name = str(gamefile)
            for trial in range(self.num_trials):
                self.env = self._load_game_env(gamefile)
                try:
                    trial_ob, trial_info = self.env.reset()
                    if trial == 0:
                        difficulties.append(self.env.difficulty)
                    ob = "\n".join(trial_ob[0].split("\n\n")[1:])
                    trial_name = "/".join(
                        trial_info["extra.gamefile"][0].split("/")[-3:-1]
                    )
                    for k, v in prefixes.items():
                        if trial_name.startswith(k):
                            examples = "".join(self.prompts["examples"][v])
                            score, is_done, gr_acc, record, steps = self._run_single_trial(
                                index=idx, trial=trial, ob=ob, examples=examples,
                            )
                            best.consider(score, is_done, gr_acc, record)
                            task_trial_results.append(bool(is_done))
                            logger.finish(
                                "Stage3 ALF | Example {} | Trial {} | wm={} | "
                                "done={} pr={:.3f} steps={}".format(
                                    idx, trial, self.wm_name, is_done, score, steps + 1,
                                )
                            )
                            break
                finally:
                    self._close_env()

                if best.done and not self.persist_memory:
                    break

            srs.append(1.0 if best.done else 0.0)
            scores.append(best.score)
            grounding_accs.append(best.grounding)
            score_state_records.append(best.record)


            if not task_trial_results:
                logger.error(
                    "Stage3 ALF | No prefix matched task {} (gamefile={}); "
                    "WorldEvolver prefixes dict likely drifted. Aborting eval.".format(
                        trial_name, gamefile,
                    )
                )
                raise RuntimeError(
                    f"Stage 3 ALF: no prefix matched {trial_name}"
                )
            srs_per_trial.append(task_trial_results)

        wall_clock_seconds = time.perf_counter() - wall_clock_start


        agent_llm = (
            getattr(self.agent, "llm", None)
            or getattr(self.agent, "llm_model", None)
        )
        token_usage = (
            copy.deepcopy(getattr(agent_llm, "total_usage", None))
            if agent_llm is not None else None
        )
        return self.summarize_stage_scores(
            srs, scores, grounding_accs, score_state_records, difficulties,
            srs_per_trial=srs_per_trial,
            token_usage=token_usage,
            wall_clock_seconds=wall_clock_seconds,
        )


    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):
        agent_config = dict(agent_config or {})
        agent_config.setdefault("need_goal", True)
        agent_name = agent_config.get("agent_name", agent_config.get("name", "WMReactAgent"))
        init_prompt_path = agent_config.get(
            "init_prompt_path", "prompts/alfworld_in_context_learning.json",
        )


        if agent_name == "WMReflactAgent" and not agent_config.get("instruction"):
            from agents.wm_reflact_agent import load_reflact_prompt, reflact_prompt_path

            load_reflact_prompt("alfworld")
            agent_config = dict(agent_config)


            init_prompt_path = str(reflact_prompt_path("alfworld"))
            agent_config["init_prompt_path"] = init_prompt_path
        max_num_steps = int(run_config.get("max_num_steps", 30))
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        num_exams = int(run_config.get("num_exam", 134))
        log_path = run_config.get("log_path", None)


        wm_name = _first_config_value(
            "wm_name", "wm-base", env_config, run_config, agent_config,
        )
        wm_config = _first_config_value(
            "wm_config", {}, env_config, run_config, agent_config,
        )
        wm_config = wm_config or {}
        num_trials = int(run_config.get("num_trials", agent_config.get("num_trials", 5)))
        persist_memory = config_bool(
            run_config.get("persist_memory", agent_config.get("persist_memory", True))
        )
        trace_dir = run_config.get("trace_dir") or agent_config.get("trace_dir")
        seed = int(run_config.get("seed", env_config.get("seed", 42)))

        return cls(
            llm_config=llm_config,
            agent_name=agent_name,
            max_num_steps=max_num_steps,
            num_exams=num_exams,
            init_prompt_path=init_prompt_path,
            agent_config=agent_config,
            env_config=env_config,
            llm=llm,
            baseline_dir=baseline_dir,
            log_path=log_path,
            wm_name=wm_name,
            wm_config=wm_config,
            num_trials=num_trials,
            persist_memory=persist_memory,
            trace_dir=trace_dir,
            seed=seed,
        )
