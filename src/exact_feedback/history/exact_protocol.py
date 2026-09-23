#!/usr/bin/env python3
"""Run paired full-history and history-reset generations on frozen states."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    template: str | None,
) -> str:
    if template is not None:
        tokenizer.chat_template = template
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in ("first_recurrence", "observed_rescue_control", "all"):
        family_rows = rows if family == "all" else [
            row for row in rows if row["state_family"] == family
        ]
        output[family] = {}
        for condition in ("full_history", "history_reset"):
            subset = [row for row in family_rows if row["condition"] == condition]
            output[family][condition] = {
                "states": len(subset),
                "repeat_current_rate": statistics.fmean(
                    row["repeats_current"] for row in subset
                ),
                "repeat_any_prior_rate": statistics.fmean(
                    row["repeats_any_prior"] for row in subset
                ),
                "direction_correct_rate": statistics.fmean(
                    row["direction_correct"] for row in subset
                ),
                "contraction_rate": statistics.fmean(
                    row["contracts_absolute_error"] for row in subset
                ),
                "exact_rate": statistics.fmean(
                    row["exact_length"] for row in subset
                ),
                "joint_rate": statistics.fmean(
                    row["joint_success"] for row in subset
                ),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    args = parser.parse_args()

    all_states = read_jsonl(args.states)
    states = [row for row in all_states if row["model"] == args.model_slug]
    if len(states) != 96 or len({row["state_id"] for row in states}) != 96:
        raise RuntimeError(f"expected 96 unique states for {args.model_slug}")
    template = (
        args.chat_template.read_text(encoding="utf-8")
        if args.chat_template is not None
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cells: list[tuple[dict[str, Any], str, list[dict[str, str]]]] = []
    for state in states:
        cells.extend(
            [
                (state, "full_history", state["full_messages"]),
                (state, "history_reset", state["reset_messages"]),
            ]
        )
    prompts = [render_chat(tokenizer, messages, template) for _, _, messages in cells]
    prompt_hashes = [text_hash(prompt) for prompt in prompts]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "backend": "vllm",
        "vllm_version": vllm_version,
        "model": args.model,
        "model_slug": args.model_slug,
        "interface": "custom_template" if template is not None else "native_chat_template",
        "dtype": args.dtype,
        "states": len(states),
        "generation_cells": len(cells),
        "states_sha256": sha256(args.states),
        "chat_template_sha256": sha256(args.chat_template) if args.chat_template else None,
        "runner_sha256": sha256(runner_path),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "temperature": 0.0,
        "seed": 0,
        "prompt_hash_sequence_sha256": text_hash("\n".join(prompt_hashes)),
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
        for state, _, _ in cells
    ]
    run_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    if len(outputs) != len(cells):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(cells)} cells")

    rows: list[dict[str, Any]] = []
    for (state, condition, _), prompt_hash, output in zip(
        cells, prompt_hashes, outputs, strict=True
    ):
        if not output.outputs:
            raise RuntimeError(f"{state['state_id']}/{condition}: no generation")
        candidate = output.outputs[0]
        text = candidate.text.strip()
        count = word_count(text)
        realized_delta = count - int(state["current_word_count"])
        required_delta = int(state["required_delta"])
        missing = [
            anchor for anchor in state["anchors"] if not contains_literal(text, anchor)
        ]
        exact = count == int(state["target"])
        prior_hashes = {text_hash(value) for value in state["prior_texts"]}
        row = {
            key: state[key]
            for key in (
                "state_id",
                "model",
                "case_id",
                "source",
                "length_band",
                "state_family",
                "revision",
                "target",
                "current_word_count",
                "current_error",
                "error_bin",
                "required_delta",
            )
        }
        row.update(
            {
                "condition": condition,
                "rendered_sha256": prompt_hash,
                "text": text,
                "output_word_count": count,
                "output_error": count - int(state["target"]),
                "realized_delta": realized_delta,
                "action_gain": (
                    realized_delta / required_delta if required_delta else None
                ),
                "direction_correct": realized_delta * required_delta > 0,
                "contracts_absolute_error": abs(count - int(state["target"]))
                < abs(int(state["current_error"])),
                "repeats_current": text_hash(text) == text_hash(state["current_text"]),
                "repeats_any_prior": text_hash(text) in prior_hashes,
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
            "generation_seconds": round(time.perf_counter() - run_started, 3),
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
