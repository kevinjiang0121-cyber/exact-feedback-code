"""Portable, fail-closed access to a separate evidence package."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class ArtifactStore:
    root: Path

    def path(self, relative: str) -> Path:
        root = self.root.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Artifact escapes package root: {relative}")
        return target

    def rows(self, relative: str) -> list[dict]:
        rows = []
        with self.path(relative).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def verify(self, index: str = "inventory.json") -> dict:
        entries = json.loads(self.path(index).read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError("An evidence index must be a non-empty list")
        errors, seen = [], set()
        for entry in entries:
            rel = entry["path"]
            if rel in seen:
                raise ValueError(f"Duplicate inventory path: {rel}")
            seen.add(rel)
            p = self.path(rel)
            if not p.is_file():
                errors.append({"path": rel, "error": "missing"})
                continue
            digest = hashlib.sha256()
            with p.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if p.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                errors.append({"path": rel, "error": "integrity mismatch"})
        return {"files": len(entries), "integrity_pass": not errors, "errors": errors,
                "scope": "File integrity only; not claim coverage or anonymity approval."}
