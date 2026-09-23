"""Execution objects shared by new portable experiment implementations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import subprocess
from .config import model_path, python_environment


@dataclass(frozen=True)
class ModelBackend:
    root: Path
    config: dict
    key: str

    def local_path(self) -> Path:
        return model_path(self.root, self.config, self.key)


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]

    def execute(self, root: Path, dry_run: bool = False) -> None:
        print("+", subprocess.list2cmdline(self.argv))
        if not dry_run:
            subprocess.run(self.argv, cwd=root, env=python_environment(root), check=True)


class ExperimentImplementation(ABC):
    @abstractmethod
    def plan(self) -> list[Command]:
        """Validate protocol inputs and construct ordered execution steps."""

    def execute(self, root: Path, dry_run: bool = False) -> None:
        for command in self.plan():
            command.execute(root, dry_run=dry_run)
