#!/usr/bin/env python3
"""Generate one deterministic revision for every frozen Phase3A state."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version

from exact_feedback.common.exact_protocol import contains_literal, read_jsonl, word_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_chat(
    tokenizer: Any, template: str, messages: list[dict[str, str]]
) -> str:
    tokenizer.chat_template = template
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in ("discovery", "confirmation", "all"):
        subset = rows if split == "all" else [
            row for row in rows if row["split"] == split
        ]
        if not subset:
            continue
        by_split[split] = {
            "states": len(subset),
            "direction_correct_rate": sum(
                row["direction_correct"] for row in subset
            ) / len(subset),
            "positive_action_gain_rate": sum(
                row["action_gain"] > 0 for row in subset
            ) / len(subset),
            "median_action_gain": statistics.median(
                row["action_gain"] for row in subset
            ),
            "mean_action_gain": statistics.fmean(
                row["action_gain"] for row in subset
            ),
            "exact_rate": sum(row["exact_length"] for row in subset)
            / len(subset),
            "anchor_retention_rate": sum(
                not row["missing_anchors"] for row in subset
            ) / len(subset),
            "joint_rate": sum(row["joint_success"] for row in subset)
            / len(subset),
        }
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("all", "discovery", "confirmation"),
        default="all",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    args = parser.parse_args()

    all_states = read_jsonl(args.states)
    states = (
        all_states
        if args.split == "all"
        else [row for row in all_states if row["split"] == args.split]
    )
    expected = {"all": 288, "discovery": 96, "confirmation": 192}[
        args.split
    ]
    if (
        len(states) != expected
        or len({row["state_id"] for row in states}) != expected
    ):
        raise RuntimeError(
            f"expected {expected} unique {args.split} fixed states"
        )
    template = args.chat_template.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    prompts = [
        render_chat(tokenizer, template, state["messages"])
        for state in states
    ]
    prompt_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in prompts
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "backend": "vllm",
        "vllm_version": vllm_version,
        "model": args.model,
        "model_slug": args.model_slug,
        "dtype": args.dtype,
        "states": len(states),
        "split": args.split,
        "states_sha256": sha256(args.states),
        "chat_template_sha256": sha256(args.chat_template),
        "runner_sha256": sha256(runner_path),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens_rule": "max(128, target * 3)",
        "prompt_hash_sequence_sha256": hashlib.sha256(
            "\n".join(prompt_hashes).encode("ascii")
        ).hexdigest(),
    }
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=0,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        seed=0,
    )
    print(
        json.dumps(
            {
                "event": "engine_ready",
                "model": args.model_slug,
                "seconds": round(time.perf_counter() - load_started, 3),
            }
        ),
        flush=True,
    )
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=max(128, int(state["target"]) * 3),
            seed=0,
        )
        for state in states
    ]
    run_started = time.perf_counter()
    outputs = llm.generate(
        prompts, sampling_params=sampling_params, use_tqdm=False
    )
    if len(outputs) != len(states):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(states)} states"
        )

    rows: list[dict[str, Any]] = []
    for state, rendered_hash, output in zip(
        states, prompt_hashes, outputs, strict=True
    ):
        if not output.outputs:
            raise RuntimeError(f"{state['state_id']}: no generation candidate")
        candidate = output.outputs[0]
        text = candidate.text.strip()
        output_count = word_count(text)
        realized_delta = output_count - int(state["current_word_count"])
        required_delta = int(state["required_delta"])
        missing = [
            anchor
            for anchor in state["anchors"]
            if not contains_literal(text, anchor)
        ]
        exact = output_count == int(state["target"])
        row = {
            key: state[key]
            for key in (
                "state_id",
                "case_id",
                "source",
                "split",
                "state_family",
                "current_word_count",
                "target",
                "signed_error",
                "required_delta",
            )
        }
        row.update(
            {
                "model": args.model_slug,
                "rendered_sha256": rendered_hash,
                "text": text,
                "output_word_count": output_count,
                "output_error": output_count - int(state["target"]),
                "realized_delta": realized_delta,
                "action_gain": realized_delta / required_delta,
                "direction_correct": realized_delta * required_delta > 0,
                "exact_length": exact,
                "missing_anchors": missing,
                "joint_success": exact and not missing,
                "generated_token_count": len(candidate.token_ids),
                "finish_reason": candidate.finish_reason,
                "stop_reason": candidate.stop_reason,
            }
        )
        rows.append(row)

    cases_path = args.output_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "generation_seconds": round(
                time.perf_counter() - run_started, 3
            ),
            "cases_sha256": sha256(cases_path),
            "summary_sha256": sha256(summary_path),
        }
    )
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
