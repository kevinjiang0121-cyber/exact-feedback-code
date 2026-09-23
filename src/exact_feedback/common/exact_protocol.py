#!/usr/bin/env python3
"""Run exact-N closed-loop generation on frozen human-authored requests."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any



WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def contains_literal(text: str, phrase: str) -> bool:
    tokens = WORD_RE.findall(phrase)
    if not tokens:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in tokens) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], target: int) -> str:
    import torch
    rendered = render_chat(tokenizer, messages)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max(128, target * 3),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def initial_prompt(item: dict[str, Any]) -> str:
    anchors = "; ".join(item["anchors"])
    return (
        f"{item['instruction']}\n\n"
        f"Length constraint: write exactly {item['target']} words under the deterministic word counter. "
        f"The response must explicitly retain these required content items: {anchors}. "
        "Return only the requested response. Do not include a preface, word-count note, analysis, or commentary."
    )


def feedback_prompt(text: str, item: dict[str, Any], missing: list[str]) -> str:
    count = word_count(text)
    delta = item["target"] - count
    action = f"add exactly {delta} words" if delta > 0 else f"remove exactly {-delta} words"
    missing_text = "none" if not missing else "; ".join(missing)
    return (
        f"DETERMINISTIC VERIFIER: previous response = {count} words; target = {item['target']}; "
        f"required action = {action}. Missing required content items: {missing_text}. "
        "Revise the complete response to satisfy the exact target and retain every required content item. "
        "Return only the revised response with no analysis, count, or commentary."
    )


def run_case(model: Any, tokenizer: Any, item: dict[str, Any], max_revisions: int) -> dict[str, Any]:
    messages = [{"role": "user", "content": initial_prompt(item)}]
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
        "rounds": rounds,
        "one_shot_exact": rounds[0]["exact_length"],
        "one_shot_joint": rounds[0]["joint_success"],
        "final_exact": rounds[-1]["exact_length"],
        "final_joint": rounds[-1]["joint_success"],
        "ever_exact": any(row["exact_length"] for row in rounds),
        "revisions_used": len(rounds) - 1,
    }


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(selected),
            "one_shot_exact": sum(row["one_shot_exact"] for row in selected) / len(selected),
            "one_shot_joint": sum(row["one_shot_joint"] for row in selected) / len(selected),
            "final_exact": sum(row["final_exact"] for row in selected) / len(selected),
            "final_joint": sum(row["final_joint"] for row in selected) / len(selected),
            "ever_exact": sum(row["ever_exact"] for row in selected) / len(selected),
            "median_revisions": statistics.median(row["revisions_used"] for row in selected),
        }

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    return {
        "model": model_name,
        "overall": metrics(rows),
        "by_source": {source: metrics(selected) for source, selected in sorted(by_source.items())},
    }


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    items = read_jsonl(args.data)
    if args.limit is not None:
        items = items[: args.limit]
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
            row = run_case(model, tokenizer, item, args.max_revisions)
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "target": row["target"],
                        "one_shot": row["one_shot_joint"],
                        "final": row["final_joint"],
                        "revisions": row["revisions_used"],
                    }
                ),
                flush=True,
            )
    summary = summarize(results, args.model)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
