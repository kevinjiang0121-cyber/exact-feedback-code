#!/usr/bin/env python3
"""Run the frozen GLM tail-1/tail-2 partial-history pilot via vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[3]
import exact_feedback.common.structured_protocol as protocol  # noqa: E402
from exact_feedback.content_contract.constraint_runner import render_chat  # noqa: E402

EXPECTED_STATES_SHA = "037d099f6c77f5757e088da5f3aca45664542098670757e73c7c935f3fce2e5f"
CONDITIONS = (
    ("full_history", "full_messages"),
    ("history_reset", "reset_messages"),
    ("tail_1", "tail_1_messages"),
    ("tail_2", "tail_2_messages"),
)


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow a deterministic smoke subset; full runs retain the frozen hash/count gates.",
    )
    args = parser.parse_args()

    states_sha = hashlib.sha256(args.states.read_bytes()).hexdigest()
    if not args.allow_subset and states_sha != EXPECTED_STATES_SHA:
        raise RuntimeError(f"states hash mismatch: {states_sha}")
    specs = read(args.states)
    if (not args.allow_subset and len(specs) != 60) or {row["model"] for row in specs} != {"glm4_9b"}:
        raise RuntimeError("frozen pilot requires exactly 60 GLM states")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "cases.jsonl").exists():
        raise RuntimeError("canonical output already exists")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        seed=0,
    )
    cells: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for condition, key in CONDITIONS:
            arm_id = f"{spec['state_id']}__{condition}"
            cells[arm_id] = {
                "id": arm_id,
                "condition": condition,
                "spec": spec,
                "messages": list(spec[key]),
                "rounds": [],
                "prior": {digest(text) for text in spec["prior_texts"]},
                "energy": float(spec["current_violation_energy"]),
            }

    completed = []
    for step in range(1, args.max_revisions + 1):
        active = [
            cell for cell in cells.values()
            if int(cell["spec"]["revision"]) + step <= args.max_revisions
        ]
        if not active:
            break
        prompts = [render_chat(tokenizer, cell["messages"]) for cell in active]
        params = [
            SamplingParams(
                temperature=0,
                seed=0,
                max_tokens=protocol.max_tokens(cell["spec"]["item"]),
            )
            for cell in active
        ]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        done = []
        for cell, output in zip(active, outputs, strict=True):
            text = output.outputs[0].text.strip()
            spec = cell["spec"]
            verifier = protocol.verify(spec["structure"], text, spec["item"]["targets"])
            energy = float(sum(verifier["violation_vector"]))
            round_row = {
                "revision": int(spec["revision"]) + step,
                "text": text,
                "joint_success": bool(verifier["joint_success"]),
                "violation_energy": energy,
                "verifier": verifier,
                "escapes_prior_recurrence": digest(text) not in cell["prior"],
                "contracts_violation_energy": energy < cell["energy"],
            }
            cell["rounds"].append(round_row)
            cell["prior"].add(digest(text))
            cell["energy"] = energy
            if round_row["joint_success"] or round_row["revision"] == args.max_revisions:
                completed.append({
                    "state_id": spec["state_id"],
                    "model": "glm4_9b",
                    "domain": spec["domain"],
                    "structure": spec["structure"],
                    "source": spec["source"],
                    "condition": cell["condition"],
                    "trigger_revision": spec["revision"],
                    "rounds": cell["rounds"],
                    "first_step_escape": cell["rounds"][0]["escapes_prior_recurrence"],
                    "first_step_contraction": cell["rounds"][0]["contracts_violation_energy"],
                    "final_joint": round_row["joint_success"],
                })
                done.append(cell["id"])
            else:
                cell["messages"].extend([
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": protocol.feedback_prompt(spec["item"], verifier)},
                ])
        for arm_id in done:
            del cells[arm_id]
        print(json.dumps({"event": "step", "step": step, "remaining": len(cells)}), flush=True)

    expected_arms = len(specs) * len(CONDITIONS)
    if len(completed) != expected_arms:
        raise RuntimeError(f"expected {expected_arms} arms, got {len(completed)}")
    cases = args.output_dir / "cases.jsonl"
    cases.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in sorted(completed, key=lambda row: (row["state_id"], row["condition"]))
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_slug": "glm4_9b",
        "backend": "vllm_native_chat_partial_history",
        "states": len(specs),
        "arms": expected_arms,
        "conditions": [condition for condition, _ in CONDITIONS],
        "temperature": 0,
        "seed": 0,
        "states_sha256": states_sha,
        "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
