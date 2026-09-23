#!/usr/bin/env python3
"""Run the frozen COLLIE c05/c10 exact-feedback loop with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, __version__ as vllm_version


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "python_packages"))
sys.path.insert(0, str(ROOT / "third_party" / "Collie-master"))

from exact_feedback.common.constraint_protocol import verify  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def initial_prompt(item: dict[str, Any]) -> str:
    return (
        f"{item['task'].strip()}\n\n"
        "The constraints are checked by a deterministic verifier. Return only "
        "the requested text, with no preface, analysis, count, or commentary."
    )


def display(value: Any) -> str:
    return "<missing>" if value is None else repr(value)


def feedback_prompt(item: dict[str, Any], state: dict[str, Any]) -> str:
    if item["family"] == "lexical_position_c05":
        position_lines = []
        for position, target, observed, match in zip(
            state["target_positions_zero_based"],
            state["target_words"],
            state["observed_words"],
            state["position_matches"],
        ):
            position_lines.append(
                f"- word {position + 1}: target={display(target)}, "
                f"observed={display(observed)}, correct={str(match).lower()}"
            )
        details = "\n".join(position_lines)
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- total words: observed={state['observed_word_count']}, "
            f"target={state['target_word_count']}, "
            f"residual(observed-target)={state['count_residual']}\n"
            f"{details}"
        )
    else:
        unit_lines = []
        for index, (count, deficit, excess) in enumerate(
            zip(
                state["sentence_word_counts"],
                state["lower_deficits"],
                state["upper_excesses"],
            ),
            start=1,
        ):
            unit_lines.append(
                f"- sentence {index}: words={count}, "
                f"below-minimum-by={deficit}, above-maximum-by={excess}"
            )
        details = "\n".join(unit_lines) if unit_lines else "- no sentences detected"
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- sentences: observed={state['observed_sentence_count']}, "
            f"target={state['target_sentence_count']}, "
            f"residual(observed-target)={state['sentence_count_residual']}\n"
            f"- allowed words per sentence: {state['lower_bound']} to "
            f"{state['upper_bound']} inclusive\n"
            f"{details}"
        )
    return (
        f"{report}\n"
        "Revise the complete previous response so every reported constraint is "
        "satisfied. Return only the revised text, with no analysis, count, or commentary."
    )


def max_tokens(item: dict[str, Any]) -> int:
    targets = item["targets"]
    if item["family"] == "lexical_position_c05":
        expected_words = int(targets[0])
    else:
        expected_words = int(targets[0]) * int(targets[2])
    return max(128, expected_words * 3)


def build_row(item: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "family": item["family"],
        "structure": item["structure"],
        "split": item["split"],
        "source": item["source"],
        "upstream_key": item["upstream_key"],
        "upstream_index": item["upstream_index"],
        "targets": item["targets"],
        "rounds": rounds,
        "one_shot_joint": rounds[0]["joint_success"],
        "final_joint": rounds[-1]["joint_success"],
        "ever_joint": any(row["joint_success"] for row in rounds),
        "revisions_used": len(rounds) - 1,
    }


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(selected),
            "one_shot_joint": sum(row["one_shot_joint"] for row in selected) / len(selected),
            "final_joint": sum(row["final_joint"] for row in selected) / len(selected),
            "ever_joint": sum(row["ever_joint"] for row in selected) / len(selected),
            "median_revisions": statistics.median(row["revisions_used"] for row in selected),
        }

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    return {
        "model": model_name,
        "overall": metrics(rows),
        "by_family": {
            family: metrics(selected) for family, selected in sorted(by_family.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chat-template-file", type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    sampling_temperature = float(os.environ.get("MULTIDOMAIN_TEMPERATURE", "0.0"))
    sampling_top_p = float(os.environ.get("MULTIDOMAIN_TOP_P", "1.0"))
    sampling_seed = int(os.environ.get("MULTIDOMAIN_SEED", "0"))

    items = [item for path in args.data for item in read_jsonl(path)]
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("Input data must be nonempty with unique case IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "cases.partial.jsonl"
    completed = load_completed(partial_path) if args.resume else {}
    selected_ids = {item["id"] for item in items}
    if set(completed) - selected_ids:
        raise ValueError("Partial file contains IDs absent from selected data")

    runner_path = Path(__file__).resolve()
    manifest = {
        "backend": "vllm",
        "vllm_version": vllm_version,
        "model": args.model,
        "data": [str(path.resolve()) for path in args.data],
        "data_sha256": {str(path.resolve()): sha256(path) for path in args.data},
        "runner": str(runner_path),
        "runner_sha256": sha256(runner_path),
        "max_revisions": args.max_revisions,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "selected_cases": len(items),
        "resumed_cases": len(completed),
        "temperature": sampling_temperature,
        "top_p": sampling_top_p,
        "seed": sampling_seed,
    }
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pending = [item for item in items if item["id"] not in completed]
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if args.chat_template_file:
            tokenizer.chat_template = args.chat_template_file.read_text(encoding="utf-8")
        load_started = time.perf_counter()
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype=args.dtype,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=True,
            seed=sampling_seed,
        )
        print(json.dumps({"event": "engine_ready", "seconds": time.perf_counter() - load_started}), flush=True)

        states = {
            item["id"]: {
                "item": item,
                "messages": [{"role": "user", "content": initial_prompt(item)}],
                "rounds": [],
            }
            for item in pending
        }
        mode = "a" if args.resume and partial_path.exists() else "w"
        with partial_path.open(mode, encoding="utf-8") as handle:
            for revision in range(args.max_revisions + 1):
                active = list(states.values())
                if not active:
                    break
                prompts = [render_chat(tokenizer, state["messages"]) for state in active]
                params = [
                    SamplingParams(
                        temperature=sampling_temperature,
                        top_p=sampling_top_p,
                        max_tokens=max_tokens(state["item"]),
                        seed=sampling_seed,
                    )
                    for state in active
                ]
                outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
                finalized = []
                for state, output in zip(active, outputs, strict=True):
                    item = state["item"]
                    text = output.outputs[0].text.strip()
                    verifier_state = verify(item["family"], text, item["targets"])
                    round_row = {
                        "revision": revision,
                        "text": text,
                        "joint_success": verifier_state["joint_success"],
                        "violation_energy": sum(verifier_state["violation_vector"]),
                        "verifier": verifier_state,
                    }
                    state["rounds"].append(round_row)
                    if verifier_state["joint_success"] or revision == args.max_revisions:
                        row = build_row(item, state["rounds"])
                        completed[item["id"]] = row
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        finalized.append(item["id"])
                        print(json.dumps({"event": "case_finalized", "id": item["id"], "final": row["final_joint"], "revisions": row["revisions_used"]}), flush=True)
                    else:
                        state["messages"].extend([
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": feedback_prompt(item, verifier_state)},
                        ])
                for case_id in finalized:
                    del states[case_id]
                print(json.dumps({"event": "revision_complete", "revision": revision, "active": len(active), "finalized": len(finalized), "remaining": len(states)}), flush=True)

    ordered = [completed[item["id"]] for item in items]
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(ordered, args.model)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
