#!/usr/bin/env python3
"""Continue paired full-history/reset arms from exact frozen trigger states."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version

from exact_feedback.common.exact_protocol import contains_literal, feedback_prompt, read_jsonl, word_count
from exact_feedback.history.exact_protocol import render_chat


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_final(state: dict[str, Any]) -> dict[str, Any]:
    spec = state["spec"]
    rounds = state["rounds"]
    final = rounds[-1]
    return {
        "state_id": spec["state_id"],
        "model": spec["model"],
        "case_id": spec["case_id"],
        "source": spec["source"],
        "length_band": spec["length_band"],
        "condition": state["condition"],
        "trigger_revision": spec["revision"],
        "trigger_recurrence_origin_revision": spec["recurrence_origin_revision"],
        "target": spec["target"],
        "anchors": spec["anchors"],
        "trigger_word_count": spec["current_word_count"],
        "trigger_error": spec["current_error"],
        "trigger_error_bin": spec["error_bin"],
        "trigger_missing": spec["current_missing"],
        "trigger_text_sha256": text_hash(spec["current_text"]),
        "rounds": rounds,
        "first_step_escape": rounds[0]["escapes_prior_recurrence"],
        "first_step_direction_correct": rounds[0]["direction_correct"],
        "first_step_contraction": rounds[0]["contracts_absolute_error"],
        "later_recurrence": any(row["repeats_any_prior"] for row in rounds[1:]),
        "final_exact": final["exact_length"],
        "final_anchor_retention": not bool(final["missing"]),
        "final_joint": final["joint_success"],
        "final_error": final["error"],
        "post_trigger_generations": len(rounds),
        "final_revision": final["revision"],
    }


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
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument(
        "--expected-states",
        type=int,
        help="Override the frozen full-run count only for deterministic smoke subsets.",
    )
    args = parser.parse_args()

    all_states = read_jsonl(args.states)
    specs = [row for row in all_states if row["model"] == args.model_slug]
    expected = args.expected_states or {"llama31_8b": 83, "glm4_9b": 120}[args.model_slug]
    if len(specs) != expected or len({row["state_id"] for row in specs}) != expected:
        raise RuntimeError(f"expected {expected} unique states, got {len(specs)}")
    template = args.chat_template.read_text(encoding="utf-8") if args.chat_template else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "backend": "vllm",
        "vllm_version": vllm_version,
        "model": args.model,
        "model_slug": args.model_slug,
        "states": len(specs),
        "generation_arms": len(specs) * 2,
        "states_sha256": sha256(args.states),
        "runner_sha256": sha256(runner),
        "chat_template_sha256": sha256(args.chat_template) if args.chat_template else None,
        "dtype": args.dtype,
        "temperature": 0.0,
        "seed": 0,
        "max_revisions": args.max_revisions,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
    }
    manifest_path = args.output_dir / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    loaded_at = time.perf_counter()
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
    print(json.dumps({
        "event": "engine_ready",
        "model": args.model_slug,
        "seconds": round(time.perf_counter() - loaded_at, 3),
    }), flush=True)

    states: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for condition, key in (("full_history", "full_messages"), ("history_reset", "reset_messages")):
            cell_id = f"{spec['state_id']}__{condition}"
            states[cell_id] = {
                "cell_id": cell_id,
                "condition": condition,
                "spec": spec,
                "messages": list(spec[key]),
                "rounds": [],
                "current_word_count": int(spec["current_word_count"]),
                "current_error": int(spec["current_error"]),
                "prior_hashes": {text_hash(text) for text in spec["prior_texts"]},
            }

    completed: list[dict[str, Any]] = []
    partial_path = args.output_dir / "cases.partial.jsonl"
    started = time.perf_counter()
    with partial_path.open("w", encoding="utf-8") as handle:
        for step in range(1, args.max_revisions + 1):
            active = [
                state for state in states.values()
                if int(state["spec"]["revision"]) + step <= args.max_revisions
            ]
            if not active:
                break
            prompts = [render_chat(tokenizer, state["messages"], template) for state in active]
            params = [SamplingParams(
                temperature=0.0,
                max_tokens=max(128, int(state["spec"]["target"]) * 3),
                seed=0,
            ) for state in active]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            finalized = []
            for state, output in zip(active, outputs, strict=True):
                if not output.outputs:
                    raise RuntimeError(f"no output for {state['cell_id']}")
                spec = state["spec"]
                revision = int(spec["revision"]) + step
                text = output.outputs[0].text.strip()
                count = word_count(text)
                target = int(spec["target"])
                missing = [anchor for anchor in spec["anchors"] if not contains_literal(text, anchor)]
                realized_delta = count - int(state["current_word_count"])
                required_delta = -int(state["current_error"])
                current_hash = text_hash(text)
                round_row = {
                    "revision": revision,
                    "text": text,
                    "word_count": count,
                    "error": count - target,
                    "missing": missing,
                    "exact_length": count == target,
                    "joint_success": count == target and not missing,
                    "realized_delta": realized_delta,
                    "required_delta": required_delta,
                    "direction_correct": realized_delta * required_delta > 0,
                    "contracts_absolute_error": abs(count - target) < abs(int(state["current_error"])),
                    "repeats_any_prior": current_hash in state["prior_hashes"],
                    "escapes_prior_recurrence": current_hash not in state["prior_hashes"],
                }
                state["rounds"].append(round_row)
                state["prior_hashes"].add(current_hash)
                state["current_word_count"] = count
                state["current_error"] = count - target
                if round_row["joint_success"] or revision == args.max_revisions:
                    row = build_final(state)
                    completed.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    finalized.append(state["cell_id"])
                else:
                    item = {
                        "instruction": "",
                        "target": target,
                        "anchors": spec["anchors"],
                    }
                    state["messages"].extend([
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": feedback_prompt(text, item, missing)},
                    ])
            for cell_id in finalized:
                del states[cell_id]
            print(json.dumps({
                "event": "continuation_step_complete",
                "step": step,
                "active_at_start": len(active),
                "finalized": len(finalized),
                "remaining": len(states),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)

    expected_rows = expected * 2
    if len(completed) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} completed arms, got {len(completed)}")
    completed.sort(key=lambda row: (row["state_id"], row["condition"]))
    cases_path = args.output_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in completed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "model": args.model_slug,
        "states": expected,
        "arms": expected_rows,
        "full_history_final_joint": sum(row["final_joint"] for row in completed if row["condition"] == "full_history") / expected,
        "history_reset_final_joint": sum(row["final_joint"] for row in completed if row["condition"] == "history_reset") / expected,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest.update({
        "cases_output_sha256": sha256(cases_path),
        "summary_sha256": sha256(summary_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
