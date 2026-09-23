"""Fixed-draft comparisons with all source/reviser cells and identity checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import sys
from exact_feedback.reproduction.artifacts import ArtifactStore
from exact_feedback.reproduction.experiment import Command, ExperimentImplementation, ModelBackend

LLAMA = "llama31_8b"
PEERS = ("qwen3_14b", "gemma2_9b", "glm4_9b")


@dataclass
class CrossoverExperiment(ExperimentImplementation):
    root: Path
    config: dict
    model: str
    output: Path
    smoke: int | None = None

    def plan(self) -> list[Command]:
        if self.model not in (LLAMA, *PEERS):
            raise ValueError(f"Unsupported crossover reviser: {self.model}")
        store = ArtifactStore(self.root / "data/controller_crossover_combined240_v1")
        data = store.rows("generation_combined240.jsonl")
        ids = [x["id"] for x in data]
        if len(ids) != 240 or len(set(ids)) != 240:
            raise ValueError("Crossover requires the frozen 240 unique cases")
        sources = (LLAMA, *PEERS) if self.model == LLAMA else (LLAMA, self.model)
        commands = []
        for source in sources:
            draft_rows = store.rows(f"planner_drafts/{source}.jsonl")
            if len(draft_rows) != 240 or {r["id"] for r in draft_rows} != set(ids):
                raise ValueError(f"Draft identity mismatch: {source}")
            source_path = store.path(f"planner_drafts/{source}.jsonl")
            data_path = store.path("generation_combined240.jsonl")
            if self.smoke is not None:
                if not 0 < self.smoke <= len(ids):
                    raise ValueError("Smoke size must be between 1 and 240")
                # Preserve source order and identical IDs in every cell.
                selected = set(ids[:self.smoke])
                inputs = self.output / "inputs"
                inputs.mkdir(parents=True, exist_ok=True)
                data_path = inputs / "cases.jsonl"
                source_path = inputs / f"{source}.jsonl"
                data_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in data[:self.smoke]), encoding="utf-8")
                source_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in draft_rows if r["id"] in selected), encoding="utf-8")
            out = self.output / f"{source}_planner__{self.model}_controller"
            commands.append(Command(tuple(map(str, [
                self.config.get("python", sys.executable), self.root / "src/exact_feedback/crossover/runner.py",
                "--model", ModelBackend(self.root, self.config, self.model).local_path(),
                "--controller-slug", self.model, "--planner-slug", source,
                "--data", data_path, "--initial-cases", source_path,
                "--output-dir", out, "--max-revisions", 8, "--dtype", "bfloat16",
            ]))))
        return commands
