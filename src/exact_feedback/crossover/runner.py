#!/usr/bin/env python3
"""Run a controller model from another model's frozen round-zero draft."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_feedback.common.exact_protocol import (
    contains_literal,
    feedback_prompt,
    generate,
    initial_prompt,
    read_jsonl,
    word_count,
)


def evaluate(text: str, item: dict[str, Any], revision: int) -> dict[str, Any]:
    count = word_count(text)
    missing = [anchor for anchor in item["anchors"] if not contains_literal(text, anchor)]
    exact = count == item["target"]
    return {
        "revision": revision,
        "text": text,
        "word_count": count,
        "error": count - item["target"],
        "exact_length": exact,
        "missing": missing,
        "joint_success": exact and not missing,
    }


def run_case(
    model: Any,
    tokenizer: Any,
    item: dict[str, Any],
    initial_text: str,
    max_revisions: int,
) -> dict[str, Any]:
    first = evaluate(initial_text, item, 0)
    rounds = [first]
    messages = [
        {"role": "user", "content": initial_prompt(item)},
        {"role": "assistant", "content": initial_text},
    ]
    for revision in range(1, max_revisions + 1):
        if rounds[-1]["joint_success"]:
            break
        messages.append(
            {
                "role": "user",
                "content": feedback_prompt(rounds[-1]["text"], item, rounds[-1]["missing"]),
            }
        )
        text = generate(model, tokenizer, messages, item["target"])
        current = evaluate(text, item, revision)
        rounds.append(current)
        messages.append({"role": "assistant", "content": text})

    return {
        "id": item["id"],
        "source": item["source"],
        "source_id": item["source_id"],
        "length_band": item["length_band"],
        "target": item["target"],
        "required": item["anchors"],
        "rounds": rounds,
        "one_shot_exact": rounds[0]["exact_length"],
        "one_shot_joint": rounds[0]["joint_success"],
        "final_exact": rounds[-1]["exact_length"],
        "final_joint": rounds[-1]["joint_success"],
        "ever_exact": any(row["exact_length"] for row in rounds),
        "revisions_used": len(rounds) - 1,
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "one_shot_joint": sum(row["one_shot_joint"] for row in rows) / len(rows),
        "final_joint": sum(row["final_joint"] for row in rows) / len(rows),
        "ever_exact": sum(row["ever_exact"] for row in rows) / len(rows),
        "median_revisions": statistics.median(row["revisions_used"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--controller-slug", required=True)
    parser.add_argument("--planner-slug", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--initial-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    items = read_jsonl(args.data)
    initial_rows = {row["id"]: row for row in read_jsonl(args.initial_cases)}
    if {item["id"] for item in items} != set(initial_rows):
        raise ValueError("Frozen data and initial-case IDs do not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    results = []
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            initial_text = initial_rows[item["id"]]["rounds"][0]["text"]
            row = run_case(model, tokenizer, item, initial_text, args.max_revisions)
            row["planner"] = args.planner_slug
            row["controller"] = args.controller_slug
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"id": row["id"], "final": row["final_joint"], "revisions": row["revisions_used"]}), flush=True)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_source[row["source"]].append(row)
    summary = {
        "planner": args.planner_slug,
        "controller": args.controller_slug,
        "overall": metrics(results),
        "by_source": {source: metrics(selected) for source, selected in sorted(by_source.items())},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
