#!/usr/bin/env python3
"""Recount and verify one feedback-policy response-surface result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from exact_feedback.common.exact_protocol import contains_literal, read_jsonl, word_count


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("discovery", "confirmation", "all"),
        required=True,
    )
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    all_states = read_jsonl(args.states)
    states = (
        all_states
        if args.split == "all"
        else [row for row in all_states if row["split"] == args.split]
    )
    if args.case_limit is not None:
        case_ids = sorted({row["case_id"] for row in states})[: args.case_limit]
        selected_ids = set(case_ids)
        states = [row for row in states if row["case_id"] in selected_ids]
    results = read_jsonl(args.cases)
    manifest = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    errors: list[str] = []

    expected_ids = [row["state_id"] for row in states]
    result_ids = [row["state_id"] for row in results]
    if result_ids != expected_ids:
        errors.append("state order or coverage mismatch")
    if len(result_ids) != len(set(result_ids)):
        errors.append("duplicate result state IDs")
    if manifest["states_sha256"] != sha256(args.states):
        errors.append("states hash mismatch")
    if manifest["split"] != args.split:
        errors.append("runtime split mismatch")
    if manifest.get("case_limit") != args.case_limit:
        errors.append("runtime case-limit mismatch")

    state_map = {row["state_id"]: row for row in states}
    for row in results:
        state = state_map.get(row["state_id"])
        if state is None:
            continue
        output_count = word_count(row["text"])
        current = int(state["current_word_count"])
        target = int(state["target"])
        required = int(state["required_delta"])
        realized = output_count - current
        initial_abs_error = abs(target - current)
        final_abs_error = abs(target - output_count)
        missing = [
            anchor
            for anchor in state["anchors"]
            if not contains_literal(row["text"], anchor)
        ]
        expected: dict[str, Any] = {
            "output_word_count": output_count,
            "output_error": output_count - target,
            "realized_delta": realized,
            "action_gain": realized / required,
            "direction_correct": realized * required > 0,
            "contraction": final_abs_error < initial_abs_error,
            "zero_action": realized == 0,
            "overshoot": final_abs_error > initial_abs_error,
            "exact_length": output_count == target,
            "missing_anchors": missing,
            "joint_success": output_count == target and not missing,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(f"{row['state_id']}: {key} mismatch")

    report = {
        "schema_version": 1,
        "verdict": "PASS" if not errors else "FAIL",
        "split": args.split,
        "frozen_rows": len(states),
        "result_rows": len(results),
        "states_sha256": sha256(args.states),
        "cases_sha256": sha256(args.cases),
        "errors": errors[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
