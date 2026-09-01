
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "word2world"
_FILENAME_BY_ENV = {
    "alfworld": "alfworld_test_with_env_195.json",
    "scienceworld": "sciworld_test_with_env_195.json",
}
_DEFAULT_HF_NAME = "X1AOX1A/LLMasWorldModels"
_GOAL_RE = re.compile(r"Your task is to:?\s*(.+?)(?:\n|$)", re.DOTALL)


class Word2WorldDataset:


    def __init__(
        self,
        env_name: str,
        data_dir: Optional[str | os.PathLike[str]] = None,
        from_hf: bool = True,
        hf_name: str = _DEFAULT_HF_NAME,
    ) -> None:
        self.env_name = self._normalize_env(env_name)
        self.data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self.from_hf = bool(from_hf)
        self.hf_name = hf_name

    @staticmethod
    def _normalize_env(env_name: str) -> str:
        key = (env_name or "").lower()
        if key in {"sciworld", "science_world"}:
            key = "scienceworld"
        if key not in _FILENAME_BY_ENV:
            raise KeyError(
                f"No Word2World data path for env_name={env_name!r}. "
                f"Known envs: {sorted(_FILENAME_BY_ENV)!r}"
            )
        return key

    @property
    def path(self) -> Path:
        return self.data_dir / _FILENAME_BY_ENV[self.env_name]

    def load(self) -> List[Dict]:
        if self.from_hf:
            return self._load_from_hf()
        path = self.path
        if not path.is_file():
            raise FileNotFoundError(
                f"Word2World eval file not found at {str(path)!r}. "
                "Stage 2 requires the actual 195+195 eval set. Put the "
                f"JSON files under {str(self.data_dir)!r}, or set "
                "word2world_from_hf: true in run_config to download them from "
                "HuggingFace."
            )
        return self._load_json_list(path, source=str(path))

    def _load_from_hf(self) -> List[Dict]:
        filename = f"llama_factory/{_FILENAME_BY_ENV[self.env_name]}"
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "word2world_from_hf=True requires the optional 'huggingface_hub' "
                "package"
            ) from exc

        kwargs = {
            "repo_id": self.hf_name,
            "filename": filename,
            "repo_type": "dataset",
        }
        path = hf_hub_download(**kwargs)
        return self._load_json_list(
            Path(path),
            source=f"HuggingFace dataset {self.hf_name!r}:{filename}",
        )

    def _load_json_list(self, path: Path, *, source: str) -> List[Dict]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Word2World file {source} must contain a JSON list")
        self._validate_records(data, source=source)
        return data

    def _validate_records(self, data: List[Dict], *, source: str) -> None:
        for i, traj in enumerate(data):
            if not isinstance(traj, dict):
                raise ValueError(f"Word2World file {source} row {i} must be an object")
            msgs = traj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise ValueError(
                    f"Word2World file {source} row {i} must contain non-empty messages"
                )

    @staticmethod
    def _extract_goal(system_text: str) -> str:
        match = _GOAL_RE.search(system_text or "")
        return match.group(1).strip() if match else ""

    def iter_trajectories(self, max_trajs: Optional[int] = None) -> Iterable[Dict]:
        for i, traj in enumerate(self.load()):
            if max_trajs is not None and i >= max_trajs:
                break
            msgs = traj["messages"]
            sys_msg = msgs[0]
            task = (
                self._extract_goal(sys_msg.get("content", ""))
                if sys_msg.get("role") == "system"
                else ""
            )
            yield {
                "traj_id": traj.get("id") or f"traj_{i}",
                "task": task,
                "messages": msgs,
            }

    def iter_steps(self, max_trajs: Optional[int] = None) -> Iterable[Dict]:
        for traj in self.iter_trajectories(max_trajs=max_trajs):
            msgs: List[Dict] = traj["messages"]
            prev_state: Optional[str] = None
            pending_action: Optional[str] = None
            step_idx = 0
            for msg in msgs:
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "system":
                    prev_state = content
                elif role == "user":
                    pending_action = content
                elif role == "assistant":
                    if prev_state is not None and pending_action is not None:
                        yield {
                            "traj_id": traj["traj_id"],
                            "step_idx": step_idx,
                            "task": traj["task"],
                            "state": prev_state,
                            "action": pending_action,
                            "gold_next_state": content,
                        }
                        step_idx += 1
                    prev_state = content
                    pending_action = None
