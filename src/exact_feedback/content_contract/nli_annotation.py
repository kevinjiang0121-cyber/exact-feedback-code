#!/usr/bin/env python3
"""Targeted local-contradiction NLI over endpoint sentence pairs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = annotation["sentences"]
    output = []
    seen = set()
    for left in range(len(sentences)):
        for right in range(left + 1, len(sentences)):
            shared = sorted(
                set(sentences[left]["entity_units"]) & set(sentences[right]["entity_units"])
            )
            kind = "adjacent" if right == left + 1 else "entity_linked"
            if kind == "entity_linked" and not shared:
                continue
            key = (left, right)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "pair_kind": kind,
                    "shared_entities": shared,
                    "premise": sentences[left]["text"],
                    "hypothesis": sentences[right]["text"],
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--stanza", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    registry = {
        row["text_sha256"]: row
        for row in read_jsonl(args.registry)
        if set(row["kinds"]) & {"round0", "final"}
    }
    stanza_rows = {
        row["text_sha256"]: row
        for row in read_jsonl(args.stanza)
        if row["text_sha256"] in registry and row["status"] == "ok"
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    labels = {int(key): value.casefold() for key, value in model.config.id2label.items()}
    completed = set()
    if args.output.exists():
        completed = {row["text_sha256"] for row in read_jsonl(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, text_hash in enumerate(sorted(registry), 1):
            if text_hash in completed:
                continue
            annotation = stanza_rows.get(text_hash)
            if annotation is None:
                result = {
                    "schema_version": 1,
                    "text_sha256": text_hash,
                    "status": "abstain",
                    "reason": "missing_stanza_annotation",
                }
            else:
                pair_rows = pairs(annotation)
                for start in range(0, len(pair_rows), args.batch_size):
                    batch = pair_rows[start : start + args.batch_size]
                    encoded = tokenizer(
                        [row["premise"] for row in batch],
                        [row["hypothesis"] for row in batch],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to("cuda")
                    with torch.inference_mode():
                        probabilities = model(**encoded).logits.softmax(dim=-1).float().cpu()
                    for row, probs in zip(batch, probabilities.tolist()):
                        row["probabilities"] = {
                            labels[i]: float(value) for i, value in enumerate(probs)
                        }
                result = {
                    "schema_version": 1,
                    "text_sha256": text_hash,
                    "status": "ok",
                    "pairs": pair_rows,
                    "adjacent_pairs": sum(row["pair_kind"] == "adjacent" for row in pair_rows),
                    "entity_linked_pairs": sum(
                        row["pair_kind"] == "entity_linked" for row in pair_rows
                    ),
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"completed": index, "hash": text_hash, "status": result["status"]}), flush=True)
    annotations = read_jsonl(args.output)
    pair_count = sum(len(row.get("pairs") or []) for row in annotations)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "transformers_nli",
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "checkpoint": str(args.model_dir),
        "checkpoint_config_sha256": sha256(args.model_dir / "config.json"),
        "registry_sha256": sha256(args.registry),
        "stanza_annotations_sha256": sha256(args.stanza),
        "texts": len(annotations),
        "ok": sum(row["status"] == "ok" for row in annotations),
        "abstain": sum(row["status"] == "abstain" for row in annotations),
        "sentence_pairs": pair_count,
        "max_length": 512,
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
