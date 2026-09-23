#!/usr/bin/env python3
"""Run one fixed stochastic-decoding seed on the frozen 60-case subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_feedback.common.exact_protocol import (
    contains_literal,
    feedback_prompt,
    initial_prompt,
    read_jsonl,
    render_chat,
    summarize,
    word_count,
)


def generate_sampled(
    model: Any, tokenizer: Any, messages: list[dict[str, str]], target: int
) -> str:
    rendered = render_chat(tokenizer, messages)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max(128, target * 3),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def run_case(
    model: Any, tokenizer: Any, item: dict[str, Any], max_revisions: int, seed: int
) -> dict[str, Any]:
    messages = [{"role": "user", "content": initial_prompt(item)}]
    rounds = []
    for revision in range(max_revisions + 1):
        text = generate_sampled(model, tokenizer, messages, item["target"])
        count = word_count(text)
        missing = [anchor for anchor in item["anchors"] if not contains_literal(text, anchor)]
        exact = count == item["target"]
        joint = exact and not missing
        rounds.append(
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
            break
        messages.extend(
            [
                {"role": "assistant", "content": text},
                {"role": "user", "content": feedback_prompt(text, item, missing)},
            ]
        )
    return {
        "id": item["id"],
        "source": item["source"],
        "source_id": item["source_id"],
        "length_band": item["length_band"],
        "target": item["target"],
        "required": item["anchors"],
        "decoding": {"temperature": 0.7, "top_p": 0.9, "seed": seed},
        "rounds": rounds,
        "one_shot_exact": rounds[0]["exact_length"],
        "one_shot_joint": rounds[0]["joint_success"],
        "final_exact": rounds[-1]["exact_length"],
        "final_joint": rounds[-1]["joint_success"],
        "ever_exact": any(row["exact_length"] for row in rounds),
        "revisions_used": len(rounds) - 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    items = read_jsonl(args.data)
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
            row = run_case(model, tokenizer, item, args.max_revisions, args.seed)
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "seed": args.seed,
                        "target": row["target"],
                        "one_shot": row["one_shot_joint"],
                        "final": row["final_joint"],
                        "revisions": row["revisions_used"],
                    }
                ),
                flush=True,
            )
    summary = summarize(results, args.model)
    summary["decoding"] = {"temperature": 0.7, "top_p": 0.9, "seed": args.seed}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
