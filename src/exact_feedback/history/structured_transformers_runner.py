#!/usr/bin/env python3
"""Run resumable paired reset continuations with a Transformers backend.

This runner exists for Falcon H1, whose fused Mamba environment is not the vLLM
environment used by the dense Transformer confirmation models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
import exact_feedback.common.structured_protocol as protocol  # noqa: E402
from exact_feedback.content_contract.transformers_runner import generate  # noqa: E402


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_arm(
    model: Any,
    tokenizer: Any,
    spec: dict[str, Any],
    condition: str,
    max_revisions: int,
) -> dict[str, Any]:
    key = "full_messages" if condition == "full_history" else "reset_messages"
    messages = list(spec[key])
    prior = {digest(text) for text in spec["prior_texts"]}
    energy = float(spec["current_violation_energy"])
    rounds = []
    for revision in range(int(spec["revision"]) + 1, max_revisions + 1):
        text = generate(model, tokenizer, messages, spec["item"])
        verifier = protocol.verify(spec["structure"], text, spec["item"]["targets"])
        next_energy = float(sum(verifier["violation_vector"]))
        row = {
            "revision": revision,
            "text": text,
            "joint_success": bool(verifier["joint_success"]),
            "violation_energy": next_energy,
            "verifier": verifier,
            "escapes_prior_recurrence": digest(text) not in prior,
            "contracts_violation_energy": next_energy < energy,
        }
        rounds.append(row)
        prior.add(digest(text))
        energy = next_energy
        if row["joint_success"] or revision == max_revisions:
            break
        messages.extend([
            {"role": "assistant", "content": text},
            {"role": "user", "content": protocol.feedback_prompt(spec["item"], verifier)},
        ])
    return {
        "state_id": spec["state_id"],
        "model": spec["model"],
        "domain": spec["domain"],
        "structure": spec["structure"],
        "source": spec["source"],
        "condition": condition,
        "trigger_revision": spec["revision"],
        "rounds": rounds,
        "first_step_escape": rounds[0]["escapes_prior_recurrence"],
        "first_step_contraction": rounds[0]["contracts_violation_energy"],
        "final_joint": rounds[-1]["joint_success"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-revisions", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    specs = [row for row in read(args.states) if row["model"] == args.model_slug]
    if not specs:
        raise RuntimeError(f"no states for {args.model_slug}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / "cases.partial.jsonl"
    completed = {}
    if partial.exists():
        for row in read(partial):
            completed[f"{row['state_id']}__{row['condition']}"] = row
    expected_ids = {
        f"{spec['state_id']}__{condition}"
        for spec in specs
        for condition in ("full_history", "history_reset")
    }
    if set(completed) - expected_ids:
        raise RuntimeError("partial output contains out-of-scope arms")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    mode = "a" if partial.exists() else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for spec in specs:
            for condition in ("full_history", "history_reset"):
                arm_id = f"{spec['state_id']}__{condition}"
                if arm_id in completed:
                    continue
                row = run_arm(model, tokenizer, spec, condition, args.max_revisions)
                completed[arm_id] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({
                    "event": "arm_complete",
                    "arm_id": arm_id,
                    "completed": len(completed),
                    "expected": len(expected_ids),
                    "final_joint": row["final_joint"],
                }), flush=True)

    if set(completed) != expected_ids:
        raise RuntimeError(f"expected {len(expected_ids)} arms, got {len(completed)}")
    cases_path = args.output_dir / "cases.jsonl"
    ordered = [completed[key] for key in sorted(completed)]
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_slug": args.model_slug,
        "backend": "transformers_fused_mamba_resumable",
        "states": len(specs),
        "arms": len(ordered),
        "temperature": 0,
        "states_sha256": hashlib.sha256(args.states.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
