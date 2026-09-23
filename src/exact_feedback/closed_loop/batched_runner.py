#!/usr/bin/env python3
"""Batched exact-N closed-loop generation with serial-equivalent semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from exact_feedback.common.exact_protocol import (
    contains_literal,
    feedback_prompt,
    initial_prompt,
    read_jsonl,
    render_chat,
    summarize,
    word_count,
)


class PerRowTokenLimit(LogitsProcessor):
    """Force EOS only after each row has produced its serial token budget."""

    def __init__(self, prompt_width: int, limits: list[int], eos_token_id: int) -> None:
        self.prompt_width = prompt_width
        self.limits = torch.tensor(limits, dtype=torch.long)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids.shape[1] - self.prompt_width
        if self.limits.device != input_ids.device:
            self.limits = self.limits.to(input_ids.device)
        limited = generated >= self.limits
        if limited.any():
            scores[limited] = -torch.inf
            scores[limited, self.eos_token_id] = 0
        return scores


def batched_generate(
    model: Any,
    tokenizer: Any,
    message_lists: list[list[dict[str, str]]],
    targets: list[int],
) -> list[str]:
    rendered = [render_chat(tokenizer, messages) for messages in message_lists]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
    prompt_width = int(inputs["input_ids"].shape[1])
    limits = [max(128, target * 3) for target in targets]
    processors = LogitsProcessorList(
        [PerRowTokenLimit(prompt_width, limits, int(tokenizer.eos_token_id))]
    )
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max(limits) + 1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=processors,
            )
    except torch.OutOfMemoryError:
        del inputs
        torch.cuda.empty_cache()
        if len(message_lists) == 1:
            raise
        midpoint = len(message_lists) // 2
        return batched_generate(
            model, tokenizer, message_lists[:midpoint], targets[:midpoint]
        ) + batched_generate(
            model, tokenizer, message_lists[midpoint:], targets[midpoint:]
        )
    return [
        tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
        for row in output
    ]


def finalize(item: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
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


def run_cases_batched(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    max_revisions: int,
    batch_size: int,
    completed: dict[str, dict[str, Any]] | None = None,
    on_finalized: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    completed = completed or {}
    states = [
        {
            "item": item,
            "messages": [{"role": "user", "content": initial_prompt(item)}],
            "rounds": [],
            "result": completed.get(item["id"]),
        }
        for item in items
    ]
    for revision in range(max_revisions + 1):
        active = [index for index, state in enumerate(states) if state["result"] is None]
        active.sort(
            key=lambda index: (
                states[index]["item"]["target"],
                sum(len(message["content"]) for message in states[index]["messages"]),
            )
        )
        for start in range(0, len(active), batch_size):
            indices = active[start : start + batch_size]
            texts = batched_generate(
                model,
                tokenizer,
                [states[index]["messages"] for index in indices],
                [states[index]["item"]["target"] for index in indices],
            )
            for index, text in zip(indices, texts):
                state = states[index]
                item = state["item"]
                count = word_count(text)
                missing = [
                    anchor for anchor in item["anchors"] if not contains_literal(text, anchor)
                ]
                exact = count == item["target"]
                joint = exact and not missing
                state["rounds"].append(
                    {
                        "revision": revision,
                        "text": text,
                        "word_count": count,
                        "error": count - item["target"],
                        "exact_length": exact,
                        "missing": missing,
                        "joint_success": joint,
                    }
                )
                if joint or revision == max_revisions:
                    state["result"] = finalize(item, state["rounds"])
                    if on_finalized is not None:
                        on_finalized(state["result"])
                else:
                    state["messages"].extend(
                        [
                            {"role": "assistant", "content": text},
                            {
                                "role": "user",
                                "content": feedback_prompt(text, item, missing),
                            },
                        ]
                    )
    return [state["result"] for state in states]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    items = read_jsonl(args.data)
    if args.limit is not None:
        items = items[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    partial_path = args.output_dir / "cases.partial.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        for row in read_jsonl(partial_path):
            completed[row["id"]] = row
    partial_mode = "a" if args.resume else "w"
    with partial_path.open(partial_mode, encoding="utf-8") as partial_handle:
        def checkpoint(row: dict[str, Any]) -> None:
            partial_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            partial_handle.flush()

        results = run_cases_batched(
            model,
            tokenizer,
            items,
            args.max_revisions,
            args.batch_size,
            completed=completed,
            on_finalized=checkpoint,
        )
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(results, args.model)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
