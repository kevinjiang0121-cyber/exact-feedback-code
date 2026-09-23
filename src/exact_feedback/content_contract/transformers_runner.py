#!/usr/bin/env python3
"""Sequential Transformers backend for the full480 multidomain protocol."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.common.structured_protocol as protocol  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate(
    model: Any, tokenizer: Any, messages: list[dict[str, str]], item: dict[str, Any]
) -> str:
    rendered = render_chat(tokenizer, messages)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=protocol.max_tokens(item),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def build_row(item: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "domain": item["domain"],
        "family": item["family"],
        "structure": item["structure"],
        "split": item["split"],
        "benchmark_partition": item["benchmark_partition"],
        "source": item["source"],
        "rounds": rounds,
        "one_shot_joint": bool(rounds[0]["joint_success"]),
        "final_joint": bool(rounds[-1]["joint_success"]),
        "ever_joint": any(bool(row["joint_success"]) for row in rounds),
        "revisions_used": len(rounds) - 1,
    }


def run_case(
    model: Any, tokenizer: Any, item: dict[str, Any], max_revisions: int
) -> dict[str, Any]:
    messages = [{"role": "user", "content": protocol.initial_prompt(item)}]
    rounds = []
    for revision in range(max_revisions + 1):
        text = generate(model, tokenizer, messages, item)
        verifier = protocol.verify(item["structure"], text, item["targets"])
        rounds.append(
            {
                "revision": revision,
                "text": text,
                "joint_success": bool(verifier["joint_success"]),
                "violation_energy": sum(verifier["violation_vector"]),
                "verifier": verifier,
            }
        )
        if verifier["joint_success"] or revision == max_revisions:
            break
        messages.extend(
            [
                {"role": "assistant", "content": text},
                {"role": "user", "content": protocol.feedback_prompt(item, verifier)},
            ]
        )
    return build_row(item, rounds)


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(selected),
            "one_shot_joint": sum(row["one_shot_joint"] for row in selected) / len(selected),
            "final_joint": sum(row["final_joint"] for row in selected) / len(selected),
            "ever_joint": sum(row["ever_joint"] for row in selected) / len(selected),
            "median_revisions": statistics.median(row["revisions_used"] for row in selected),
        }

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    return {
        "model": model_name,
        "backend": "transformers_sequential",
        "overall": metrics(rows),
        "by_domain": {key: metrics(value) for key, value in sorted(by_domain.items())},
    }


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    items = [item for path in args.data for item in read_jsonl(path)]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate benchmark IDs")
    if args.limit is not None:
        items = items[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / "cases.partial.jsonl"
    completed = load_completed(partial) if args.resume else {}
    selected_ids = {item["id"] for item in items}
    if set(completed) - selected_ids:
        raise ValueError("partial file contains out-of-scope IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    mode = "a" if args.resume and partial.exists() else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for item in items:
            if item["id"] in completed:
                continue
            row = run_case(model, tokenizer, item, args.max_revisions)
            completed[item["id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "final": row["final_joint"],
                        "revisions": row["revisions_used"],
                    }
                ),
                flush=True,
            )

    ordered = [completed[item["id"]] for item in items]
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(ordered, args.model)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
