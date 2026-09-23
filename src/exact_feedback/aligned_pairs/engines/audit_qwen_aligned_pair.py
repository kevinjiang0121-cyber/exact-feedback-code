#!/usr/bin/env python3
"""Hard-gate an aligned Base/Instruct pair before causal interventions.

The historical filename is retained so the frozen Qwen runner remains
reproducible. Architecture and tokenizer expectations are now explicit CLI
arguments, allowing the same audited gate to serve Gemma without weakening the
original Qwen defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import AutoTokenizer


ARCH_FIELDS = (
    "architectures",
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "tie_word_embeddings",
    "rope_theta",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tensor_shapes(model: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    weight_map = read_json(model / "model.safetensors.index.json")["weight_map"]
    shapes: dict[str, list[int]] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(model / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = list(handle.get_slice(key).get_shape())
    return shapes, weight_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--instruct", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-hidden-size", type=int)
    parser.add_argument("--expected-states", type=int, default=288)
    parser.add_argument(
        "--tokenizer-files",
        nargs="*",
        default=None,
        help="Lexical tokenizer assets that must be byte-identical. Auto-detect when omitted.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    base_config = read_json(args.base / "config.json")
    instruct_config = read_json(args.instruct / "config.json")
    architecture = {}
    for field in ARCH_FIELDS:
        same = base_config.get(field) == instruct_config.get(field)
        architecture[field] = {
            "base": base_config.get(field),
            "instruct": instruct_config.get(field),
            "same": same,
        }
        if not same:
            errors.append(f"architecture mismatch: {field}")
    if base_config.get("num_hidden_layers") != args.expected_layers:
        errors.append(f"expected {args.expected_layers} hidden layers")
    if args.expected_hidden_size is not None and base_config.get("hidden_size") != args.expected_hidden_size:
        errors.append(f"expected hidden size {args.expected_hidden_size}")

    base_shapes, base_map = tensor_shapes(args.base)
    instruct_shapes, instruct_map = tensor_shapes(args.instruct)
    key_sets_same = set(base_shapes) == set(instruct_shapes)
    shape_mismatches = sorted(
        key for key in set(base_shapes) & set(instruct_shapes)
        if base_shapes[key] != instruct_shapes[key]
    )
    if not key_sets_same:
        errors.append("tensor key sets differ")
    if shape_mismatches:
        errors.append("tensor shapes differ")

    candidates = ("tokenizer.model", "tokenizer.json", "vocab.json", "merges.txt")
    tokenizer_files = tuple(args.tokenizer_files) if args.tokenizer_files is not None else tuple(
        name for name in candidates if (args.base / name).exists() and (args.instruct / name).exists()
    )
    if not tokenizer_files:
        errors.append("no shared lexical tokenizer asset found")
    tokenizer_hashes = {}
    for name in tokenizer_files:
        base_path, instruct_path = args.base / name, args.instruct / name
        same = base_path.exists() and instruct_path.exists() and sha256(base_path) == sha256(instruct_path)
        tokenizer_hashes[name] = {
            "base": sha256(base_path) if base_path.exists() else None,
            "instruct": sha256(instruct_path) if instruct_path.exists() else None,
            "same": same,
        }
        if not same:
            errors.append(f"tokenizer lexical file mismatch: {name}")

    tokenizer = AutoTokenizer.from_pretrained(args.instruct, trust_remote_code=True, use_fast=True)
    template = tokenizer.chat_template
    if not template:
        errors.append("Instruct chat template missing")
    states = read_jsonl(args.states)
    if len(states) != args.expected_states or len({row["state_id"] for row in states}) != args.expected_states:
        errors.append(f"fixed-state coverage is not {args.expected_states} unique states")
    prompt_rows = []
    for state in states:
        rendered = tokenizer.apply_chat_template(
            state["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        prompt_rows.append({
            "state_id": state["state_id"],
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
            "tokens": len(token_ids),
        })

    result = {
        "schema_version": 1,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "base": str(args.base),
        "instruct": str(args.instruct),
        "architecture": architecture,
        "expected_architecture": {
            "num_hidden_layers": args.expected_layers,
            "hidden_size": args.expected_hidden_size,
        },
        "tensor_audit": {
            "base_keys": len(base_shapes),
            "instruct_keys": len(instruct_shapes),
            "key_sets_same": key_sets_same,
            "shape_mismatches": shape_mismatches,
            "shard_maps_same": base_map == instruct_map,
        },
        "tokenizer_lexical_hashes": tokenizer_hashes,
        "common_interface": {
            "source": str(args.instruct),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if template else None,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "fixed_states": len(prompt_rows),
            "min_tokens": min((row["tokens"] for row in prompt_rows), default=None),
            "max_tokens": max((row["tokens"] for row in prompt_rows), default=None),
            "rows": prompt_rows,
        },
        "states_sha256": sha256(args.states),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "errors": errors, "tensor_keys": len(base_shapes), "states": len(prompt_rows)}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
