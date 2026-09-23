#!/usr/bin/env python3
"""Continue frozen revision-8 failures under the unchanged loop through revision 32."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
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
    render_chat,
    word_count,
)


def build_messages(item: dict[str, Any], rounds: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "user", "content": initial_prompt(item)}]
    for row in rounds:
        messages.extend(
            [
                {"role": "assistant", "content": row["text"]},
                {"role": "user", "content": feedback_prompt(row["text"], item, row["missing"])},
            ]
        )
    return messages


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
    selected: dict[str, Any],
    max_revision: int,
) -> dict[str, Any]:
    item = selected["item"]
    baseline = selected["baseline"]
    rounds = copy.deepcopy(baseline["rounds"])
    if baseline["final_joint"] or [row["revision"] for row in rounds] != list(range(9)):
        raise RuntimeError(f"{selected['id']}: invalid revision-8 failure")
    messages = build_messages(item, rounds)
    max_context = int(getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))
    termination_reason = "max_revision"
    context_limit_revision = None
    input_tokens_at_stop = None

    for revision in range(9, max_revision + 1):
        rendered = render_chat(tokenizer, messages)
        input_tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        max_new_tokens = max(128, int(item["target"]) * 3)
        if input_tokens + max_new_tokens > max_context:
            termination_reason = "context_limit"
            context_limit_revision = revision
            input_tokens_at_stop = input_tokens
            break
        text = generate(model, tokenizer, messages, item["target"])
        row = evaluate(text, item, revision)
        rounds.append(row)
        if row["joint_success"]:
            termination_reason = "joint_success"
            break
        messages.extend(
            [
                {"role": "assistant", "content": text},
                {"role": "user", "content": feedback_prompt(text, item, row["missing"])},
            ]
        )

    return {
        "id": selected["id"],
        "source": selected["source"],
        "source_id": item["source_id"],
        "length_band": item["length_band"],
        "target": item["target"],
        "required": item["anchors"],
        "selection_index": selected["selection_index"],
        "selection_hash": selected["selection_hash"],
        "selection_stratum": f"{selected['source']}|{selected['terminal_error_region']}",
        "baseline_line_sha256": selected["baseline_line_sha256"],
        "baseline_terminal_error": selected["terminal_error"],
        "rounds": rounds,
        "new_joint_success": any(row["joint_success"] for row in rounds if row["revision"] > 8),
        "closure_revision": next(
            (row["revision"] for row in rounds if row["revision"] > 8 and row["joint_success"]),
            None,
        ),
        "termination_reason": termination_reason,
        "context_limit_revision": context_limit_revision,
        "input_tokens_at_stop": input_tokens_at_stop,
        "model_max_position_embeddings": max_context,
        "final_exact": rounds[-1]["exact_length"],
        "final_joint": rounds[-1]["joint_success"],
        "final_revision": rounds[-1]["revision"],
    }


def summarize(rows: list[dict[str, Any]], model_slug: str) -> dict[str, Any]:
    closures = [row["closure_revision"] for row in rows if row["closure_revision"] is not None]
    return {
        "model": model_slug,
        "cases": len(rows),
        "new_joint_success_n": len(closures),
        "new_joint_success_rate": len(closures) / len(rows),
        "context_limit_n": sum(row["termination_reason"] == "context_limit" for row in rows),
        "persistent_to_revision32_n": sum(
            row["termination_reason"] == "max_revision" and row["final_revision"] == 32
            for row in rows
        ),
        "median_closure_revision": statistics.median(closures) if closures else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-revision", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    selected = read_jsonl(args.selection)
    if not selected:
        raise RuntimeError("empty persistence selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    rows = []
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for item in selected:
            row = run_case(model, tokenizer, item, args.max_revision)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "termination": row["termination_reason"],
                        "closure_revision": row["closure_revision"],
                        "final_revision": row["final_revision"],
                    }
                ),
                flush=True,
            )
    summary = summarize(rows, args.model_slug)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
