
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]


def lib_dir(root: Optional[str] = None) -> str:
    if root:
        return str(Path(root).expanduser().resolve())
    return str(_REPO_ROOT / "word2world_wm_ablation" / "state_library")


def lib_path(env_name: str, root: Optional[str] = None) -> str:

    return os.path.join(lib_dir(root), f"{env_name.lower()}.json")


def load_library(env_name: str, root: Optional[str] = None) -> List[Dict]:

    path = lib_path(env_name, root)
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
