#!/usr/bin/env python3
"""Generate one deterministic revision for a frozen feedback-policy sweep."""

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
    tokenizer: Any,
    messages: list[dict[str, str]],
    template: str | None,
) -> str:
    if template is not None:
        tokenizer.chat_template = template
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in ("discovery", "confirmation", "all"):
        subset = (
            rows
            if split == "all"
            else [row for row in rows if row["split"] == split]
        )
        if not subset:
            continue
        report[split] = {
            "states": len(subset),
            "direction_correct_rate": sum(
                row["direction_correct"] for row in subset
            )
            / len(subset),
            "contraction_rate": sum(row["contraction"] for row in subset)
            / len(subset),
            "zero_action_rate": sum(row["zero_action"] for row in subset)
            / len(subset),
            "overshoot_rate": sum(row["overshoot"] for row in subset)
            / len(subset),
            "median_action_gain": statistics.median(
                row["action_gain"] for row in subset
            ),
            "median_terminal_absolute_error": statistics.median(
                abs(row["output_error"]) for row in subset
            ),
            "anchor_retention_rate": sum(
                not row["missing_anchors"] for row in subset
            )
            / len(subset),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("discovery", "confirmation", "all"),
        default="all",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()

    all_states = read_jsonl(args.states)
    states = (
        all_states
        if args.split == "all"
        else [row for row in all_states if row["split"] == args.split]
    )
    if args.case_limit is not None:
        case_ids = sorted({row["case_id"] for row in states})[
            : args.case_limit
        ]
        selected_ids = set(case_ids)
        states = [
            row for row in states if row["case_id"] in selected_ids
        ]
    state_ids = [row["state_id"] for row in states]
    if not states or len(state_ids) != len(set(state_ids)):
        raise RuntimeError("response-surface states must be nonempty and unique")

    template = (
        args.chat_template.read_text(encoding="utf-8")
        if args.chat_template
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    prompts = [
        render_chat(tokenizer, state["messages"], template)
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
        "case_limit": args.case_limit,
        "states_sha256": sha256(args.states),
        "chat_template_sha256": (
            sha256(args.chat_template) if args.chat_template else None
        ),
        "runner_sha256": sha256(runner_path),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
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
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
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
        raise RuntimeError("vLLM output count does not match frozen states")

    rows: list[dict[str, Any]] = []
    for state, prompt_hash, output in zip(
        states, prompt_hashes, outputs, strict=True
    ):
        if not output.outputs:
            raise RuntimeError(f"{state['state_id']}: no candidate")
        candidate = output.outputs[0]
        text = candidate.text.strip()
        output_count = word_count(text)
        current = int(state["current_word_count"])
        target = int(state["target"])
        required_delta = int(state["required_delta"])
        realized_delta = output_count - current
        initial_abs_error = abs(target - current)
        final_abs_error = abs(target - output_count)
        missing = [
            anchor
            for anchor in state["anchors"]
            if not contains_literal(text, anchor)
        ]
        row = {
            key: state[key]
            for key in (
                "state_id",
                "case_id",
                "source",
                "split",
                "length_stratum",
                "state_family",
                "current_word_count",
                "target",
                "signed_error_current_minus_target",
                "required_delta",
            )
        }
        row.update(
            {
                "model": args.model_slug,
                "rendered_sha256": prompt_hash,
                "text": text,
                "output_word_count": output_count,
                "output_error": output_count - target,
                "realized_delta": realized_delta,
                "action_gain": realized_delta / required_delta,
                "direction_correct": realized_delta * required_delta > 0,
                "contraction": final_abs_error < initial_abs_error,
                "zero_action": realized_delta == 0,
                "overshoot": final_abs_error > initial_abs_error,
                "exact_length": output_count == target,
                "missing_anchors": missing,
                "joint_success": output_count == target and not missing,
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
