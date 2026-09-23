#!/usr/bin/env python3
"""Run paired full-history/reset arms from frozen multidomain recurrent states."""

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


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-slug", required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=.88)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-revisions", type=int, default=8)
    a = p.parse_args()
    specs = [row for row in read(a.states) if row["model"] == a.model_slug]
    if not specs:
        raise RuntimeError(f"no states for {a.model_slug}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, tokenizer=a.model, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
              max_num_seqs=a.max_num_seqs, enable_prefix_caching=True, seed=0)
    cells: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for condition, key in (("full_history", "full_messages"), ("history_reset", "reset_messages")):
            cid = f"{spec['state_id']}__{condition}"
            cells[cid] = {"id": cid, "condition": condition, "spec": spec,
                          "messages": list(spec[key]), "rounds": [],
                          "prior": {h(t) for t in spec["prior_texts"]},
                          "energy": float(spec["current_violation_energy"])}
    completed = []
    for step in range(1, a.max_revisions + 1):
        active = [cell for cell in cells.values() if cell["spec"]["revision"] + step <= a.max_revisions]
        if not active:
            break
        prompts = [render_chat(tokenizer, cell["messages"]) for cell in active]
        params = [SamplingParams(temperature=0, seed=0, max_tokens=protocol.max_tokens(cell["spec"]["item"])) for cell in active]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        done = []
        for cell, output in zip(active, outputs, strict=True):
            text = output.outputs[0].text.strip()
            spec = cell["spec"]
            verifier = protocol.verify(spec["structure"], text, spec["item"]["targets"])
            energy = float(sum(verifier["violation_vector"]))
            row = {"revision": spec["revision"] + step, "text": text,
                   "joint_success": bool(verifier["joint_success"]), "violation_energy": energy,
                   "verifier": verifier, "escapes_prior_recurrence": h(text) not in cell["prior"],
                   "contracts_violation_energy": energy < cell["energy"]}
            cell["rounds"].append(row)
            cell["prior"].add(h(text))
            cell["energy"] = energy
            if row["joint_success"] or row["revision"] == a.max_revisions:
                completed.append({"state_id": spec["state_id"], "model": a.model_slug,
                                  "domain": spec["domain"], "structure": spec["structure"],
                                  "source": spec["source"], "condition": cell["condition"],
                                  "trigger_revision": spec["revision"], "rounds": cell["rounds"],
                                  "first_step_escape": cell["rounds"][0]["escapes_prior_recurrence"],
                                  "first_step_contraction": cell["rounds"][0]["contracts_violation_energy"],
                                  "final_joint": row["joint_success"]})
                done.append(cell["id"])
            else:
                cell["messages"].extend([{"role": "assistant", "content": text}, {"role": "user", "content": protocol.feedback_prompt(spec["item"], verifier)}])
        for cid in done:
            del cells[cid]
        print(json.dumps({"event": "step", "step": step, "remaining": len(cells)}), flush=True)
    if len(completed) != len(specs) * 2:
        raise RuntimeError(f"expected {len(specs)*2} arms, got {len(completed)}")
    out = a.output_dir / "cases.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(completed, key=lambda r: (r["state_id"], r["condition"]))), encoding="utf-8")
    manifest = {"schema_version": 1, "model": a.model, "model_slug": a.model_slug,
                "states": len(specs), "arms": len(completed), "temperature": 0, "seed": 0,
                "states_sha256": hashlib.sha256(a.states.read_bytes()).hexdigest(),
                "cases_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (a.output_dir / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
