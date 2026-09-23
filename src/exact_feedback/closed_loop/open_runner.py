#!/usr/bin/env python3
"""Run the frozen exact-length loop with vLLM continuous batching."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version

from exact_feedback.common.exact_protocol import (
    contains_literal,
    feedback_prompt,
    initial_prompt,
    read_jsonl,
    render_chat,
    summarize,
    word_count,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_row(item: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
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


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        completed[row["id"]] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chat-template-file", type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--distributed-executor-backend", choices=("mp", "ray"), default="mp")
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    chat_template_path = (
        args.chat_template_file.resolve() if args.chat_template_file else None
    )
    if chat_template_path is not None and not chat_template_path.is_file():
        raise FileNotFoundError(f"Chat template not found: {chat_template_path}")

    items = read_jsonl(args.data)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("No input cases selected")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Input case IDs are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "cases.partial.jsonl"
    completed = load_completed(partial_path) if args.resume else {}
    selected_ids = {item["id"] for item in items}
    unknown = sorted(set(completed) - selected_ids)
    if unknown:
        raise ValueError(f"Partial file contains IDs absent from selected data: {unknown[:5]}")

    runner_path = Path(__file__).resolve()
    manifest = {
        "backend": "vllm",
        "vllm_version": vllm_version,
        "model": args.model,
        "chat_template_file": (
            str(chat_template_path) if chat_template_path is not None else None
        ),
        "chat_template_sha256": (
            sha256(chat_template_path) if chat_template_path is not None else None
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "runner": str(runner_path),
        "runner_sha256": sha256(runner_path),
        "max_revisions": args.max_revisions,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "distributed_executor_backend": args.distributed_executor_backend,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "selected_cases": len(items),
        "resumed_cases": len(completed),
    }
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pending_items = [item for item in items if item["id"] not in completed]
    if pending_items:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if chat_template_path is not None:
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        load_started = time.perf_counter()
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype=args.dtype,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            distributed_executor_backend=args.distributed_executor_backend,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=0,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            seed=0,
        )
        print(
            json.dumps(
                {
                    "event": "engine_ready",
                    "seconds": round(time.perf_counter() - load_started, 3),
                    "pending_cases": len(pending_items),
                }
            ),
            flush=True,
        )

        states = {
            item["id"]: {
                "item": item,
                "messages": [{"role": "user", "content": initial_prompt(item)}],
                "rounds": [],
            }
            for item in pending_items
        }
        run_started = time.perf_counter()
        mode = "a" if args.resume and partial_path.exists() else "w"
        with partial_path.open(mode, encoding="utf-8") as partial_handle:
            for revision in range(args.max_revisions + 1):
                active = list(states.values())
                if not active:
                    break
                prompts = [
                    render_chat(tokenizer, state["messages"]) for state in active
                ]
                sampling_params = [
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=max(128, state["item"]["target"] * 3),
                        seed=0,
                    )
                    for state in active
                ]
                batch_started = time.perf_counter()
                outputs = llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                if len(outputs) != len(active):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for {len(active)} prompts"
                    )

                finalized: list[str] = []
                successes = 0
                for state, output in zip(active, outputs, strict=True):
                    if not output.outputs:
                        raise RuntimeError(
                            f"No generation candidate for {state['item']['id']}"
                        )
                    item = state["item"]
                    text = output.outputs[0].text.strip()
                    count = word_count(text)
                    missing = [
                        anchor
                        for anchor in item["anchors"]
                        if not contains_literal(text, anchor)
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
                    if joint or revision == args.max_revisions:
                        row = build_row(item, state["rounds"])
                        completed[item["id"]] = row
                        partial_handle.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        partial_handle.flush()
                        finalized.append(item["id"])
                        successes += int(joint)
                        print(
                            json.dumps(
                                {
                                    "event": "case_finalized",
                                    "id": item["id"],
                                    "target": item["target"],
                                    "one_shot": row["one_shot_joint"],
                                    "final": row["final_joint"],
                                    "revisions": row["revisions_used"],
                                }
                            ),
                            flush=True,
                        )
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
                for case_id in finalized:
                    del states[case_id]
                print(
                    json.dumps(
                        {
                            "event": "revision_complete",
                            "revision": revision,
                            "active_at_start": len(active),
                            "finalized": len(finalized),
                            "successes": successes,
                            "remaining": len(states),
                            "seconds": round(time.perf_counter() - batch_started, 3),
                            "elapsed_seconds": round(
                                time.perf_counter() - run_started, 3
                            ),
                        }
                    ),
                    flush=True,
                )

    ordered_rows = [completed[item["id"]] for item in items]
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(ordered_rows, args.model)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
