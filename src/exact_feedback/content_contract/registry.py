#!/usr/bin/env python3
"""Build a content-addressed endpoint registry for the three API Primary-120 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-data", required=True, type=Path)
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "CASES"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frozen = read_jsonl(args.frozen_data)
    expected_ids = [item["id"] for item in frozen]
    model_paths = {name: Path(path) for name, path in args.model}
    texts: dict[str, str] = {}
    kinds: dict[str, set[str]] = defaultdict(set)
    observations: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    def register(text: str, kind: str) -> str:
        text_hash = digest(text)
        if text_hash in texts and texts[text_hash] != text:
            raise RuntimeError("SHA-256 collision")
        texts[text_hash] = text
        kinds[text_hash].add(kind)
        return text_hash

    for item in frozen:
        prompt_hash = register(item["instruction"], "prompt")
        reference_hash = register(item["reference"], "reference")
        for kind, text_hash in (("prompt", prompt_hash), ("reference", reference_hash)):
            observations.append(
                {
                    "model": None,
                    "id": item["id"],
                    "source": item["source"],
                    "length_band": item["length_band"],
                    "kind": kind,
                    "text_sha256": text_hash,
                }
            )

    for model, path in model_paths.items():
        rows = read_jsonl(path)
        if [row["id"] for row in rows] != expected_ids:
            raise RuntimeError(f"{model}: row order differs from frozen panel")
        by_id = {row["id"]: row for row in rows}
        for item in frozen:
            row = by_id[item["id"]]
            rounds = row.get("rounds") or []
            round0_hash = register(rounds[0]["text"], "round0") if rounds else None
            final_hash = register(rounds[-1]["text"], "final") if rounds else None
            for kind, text_hash in (("round0", round0_hash), ("final", final_hash)):
                if text_hash is not None:
                    observations.append(
                        {
                            "model": model,
                            "id": item["id"],
                            "source": item["source"],
                            "length_band": item["length_band"],
                            "kind": kind,
                            "text_sha256": text_hash,
                            "final_joint": bool(row.get("final_joint")),
                        }
                    )
            trajectories.append(
                {
                    "model": model,
                    "id": item["id"],
                    "source": item["source"],
                    "length_band": item["length_band"],
                    "prompt_sha256": digest(item["instruction"]),
                    "reference_sha256": digest(item["reference"]),
                    "round0_sha256": round0_hash,
                    "final_sha256": final_hash,
                    "protocol_complete": bool(row.get("protocol_complete", True)),
                    "final_joint": bool(row.get("final_joint")),
                }
            )

    registry = [
        {
            "text_sha256": text_hash,
            "text": texts[text_hash],
            "kinds": sorted(kinds[text_hash]),
            "byte_count": len(texts[text_hash].encode("utf-8")),
        }
        for text_hash in sorted(texts)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "text_registry.jsonl": registry,
        "observations.jsonl": observations,
        "trajectories.jsonl": trajectories,
    }
    for name, rows in outputs.items():
        (args.output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    audit = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_cases": len(frozen),
        "models": list(model_paths),
        "trajectories": len(trajectories),
        "observations": len(observations),
        "unique_texts": len(registry),
        "missing_endpoint_texts": sum(
            row["round0_sha256"] is None or row["final_sha256"] is None
            for row in trajectories
        ),
        "input_hashes": {
            "frozen_data": file_digest(args.frozen_data),
            **{model: file_digest(path) for model, path in model_paths.items()},
        },
        "output_hashes": {
            name: file_digest(args.output_dir / name) for name in outputs
        },
    }
    (args.output_dir / "registry_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
