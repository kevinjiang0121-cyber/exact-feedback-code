from __future__ import annotations

import json
import os
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(root: Path, path: Path | None) -> dict:
    selected = path or root / "config.local.json"
    if not selected.is_file():
        raise SystemExit(
            f"Missing {selected}. Copy config.example.json to config.local.json "
            "and fill the model paths."
        )
    cfg = json.loads(selected.read_text(encoding="utf-8"))
    cfg["_path"] = str(selected)
    return cfg


def model_path(root: Path, cfg: dict, key: str) -> Path:
    value = (cfg.get("models", {}).get(key) or {}).get("path", "")
    if not value:
        raise SystemExit(f"Model path is not configured: {key}")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def python_environment(root: Path) -> dict[str, str]:
    """Expose the private implementation package to child Python processes."""
    env = os.environ.copy()
    src = str(root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    return env
