#!/usr/bin/env python3
"""Run one frozen prompt-template condition of the 60-case robustness study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exact_feedback.common.exact_protocol import (
    contains_literal,
    generate,
    read_jsonl,
    summarize,
    word_count,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def render_initial(item: dict[str, Any], template: dict[str, str]) -> str:
    return template["initial"].format(
        instruction=item["instruction"],
        target=item["target"],
        anchors="; ".join(item["anchors"]),
    )


def render_feedback(
    text: str, item: dict[str, Any], missing: list[str], template: dict[str, str]
) -> str:
    current = word_count(text)
    delta = item["target"] - current
    action = f"add exactly {delta} words" if delta > 0 else f"remove exactly {-delta} words"
    return template["feedback"].format(
        current=current,
        target=item["target"],
        action=action,
        action_cap=action.capitalize(),
        missing="none" if not missing else "; ".join(missing),
    )


def run_case(
    model: Any,
    tokenizer: Any,
    item: dict[str, Any],
    template_name: str,
    template: dict[str, str],
    max_revisions: int,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": render_initial(item, template)}]
    rounds = []
    for revision in range(max_revisions + 1):
        text = generate(model, tokenizer, messages, item["target"])
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
                {"role": "user", "content": render_feedback(text, item, missing, template)},
            ]
        )
    return {
        "id": item["id"],
        "source": item["source"],
        "source_id": item["source_id"],
        "length_band": item["length_band"],
        "target": item["target"],
        "required": item["anchors"],
        "template": template_name,
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
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--template", required=True, choices=("baseline", "structured", "concise"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    items = read_jsonl(args.data)
    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    template = templates[args.template]
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
            row = run_case(model, tokenizer, item, args.template, template, args.max_revisions)
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "template": args.template,
                        "target": row["target"],
                        "one_shot": row["one_shot_joint"],
                        "final": row["final_joint"],
                        "revisions": row["revisions_used"],
                    }
                ),
                flush=True,
            )
    summary = summarize(results, args.model)
    summary["template"] = args.template
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
