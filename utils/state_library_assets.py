
from __future__ import annotations

import json
import os
from typing import Dict, List


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_REPO_ROOT, ".."))


def lib_dir() -> str:

    candidates: List[str] = []
    env_dir = os.environ.get("WORLDEVOLVER_STATE_LIBRARY_DIR")
    if env_dir:
        candidates.append(os.path.abspath(os.path.expanduser(env_dir)))

    project_path = os.environ.get("PROJECT_PATH")
    if project_path:
        project_root = os.path.abspath(os.path.expanduser(project_path))
        candidates.append(
            os.path.join(project_root, "word2world_wm_ablation", "state_library")
        )
        candidates.append(
            os.path.join(
                os.path.dirname(project_root),
                "word2world_wm_ablation",
                "state_library",
            )
        )

    candidates.extend([
        os.path.join(_REPO_ROOT, "word2world_wm_ablation", "state_library"),
        os.path.join(_WORKSPACE_ROOT, "word2world_wm_ablation", "state_library"),
    ])
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0] if candidates else os.path.join(
        _WORKSPACE_ROOT, "word2world_wm_ablation", "state_library"
    )


def lib_path(env_name: str) -> str:

    return os.path.join(lib_dir(), f"{env_name.lower()}.json")


def load_library(env_name: str) -> List[Dict]:

    path = lib_path(env_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"State library for env={env_name!r} not found at {path!r}"
        )
    entries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(64).lstrip()
        f.seek(0)
        if first.startswith("["):
            entries = json.load(f)
            if not isinstance(entries, list):
                raise ValueError(f"State library {path!r} must contain a JSON list")
        else:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"State library {path!r} has malformed JSONL at line {line_no}"
                    ) from exc
                entries.append(row)
    for i, row in enumerate(entries):
        if not isinstance(row, dict):
            raise ValueError(f"State library {path!r} row {i} must be an object")
    return entries


__all__ = ["lib_dir", "lib_path", "load_library"]
