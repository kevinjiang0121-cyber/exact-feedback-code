#!/usr/bin/env python3
"""Audit and select frozen aligned-Qwen restoration groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = (
    "instruct_sham",
    "restore_noncontiguous",
    "restore_blocks_00_08",
    "restore_blocks_09_17",
    "restore_blocks_18_26",
    "restore_blocks_27_35",
    "restore_output",
)
RESTORATIONS = CONDITIONS[2:]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fixed_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    result = {}
    for case_id, cells in grouped.items():
        if len(cells) != 4:
            raise RuntimeError(f"{case_id}: expected four states")
        result[case_id] = {
            "zero_action": float(np.mean([int(row["realized_delta"]) == 0 for row in cells])),
            "overshoot": float(np.mean([abs(int(row["output_error"])) > abs(int(row["signed_error"])) for row in cells])),
            "direction": float(np.mean([bool(row["direction_correct"]) for row in cells])),
            "calibration_error": float(np.mean([abs(float(row["action_gain"]) - 1.0) for row in cells])),
        }
    return result


def paired(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]], metric: str, rng: np.random.Generator) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    values = np.asarray([left[x][metric] - right[x][metric] for x in ids])
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    distribution = np.mean(values[draws], axis=1)
    return {"cases": len(ids), "effect": float(np.mean(values)), "cluster_bootstrap_95": [float(x) for x in np.quantile(distribution, [0.025, 0.5, 0.975])]}


def closed_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    recurrence, self_loop = [], []
    for row in rows:
        texts = [x["text"] for x in row["rounds"]]
        recurrence.append(len(texts) != len(set(texts)))
        self_loop.append(not row["final_joint"] and len(texts) > 1 and texts[-1] == texts[-2])
    return {
        "final_joint_rate": float(np.mean([bool(x["final_joint"]) for x in rows])),
        "terminal_absolute_error": float(np.mean([abs(int(x["rounds"][-1]["error"])) for x in rows])),
        "recurrence_rate": float(np.mean(recurrence)),
        "terminal_self_loop_rate": float(np.mean(self_loop)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20260830)
    fixed, closed, integrity = {}, {}, {"passes": True, "conditions": {}}
    for condition in CONDITIONS:
        root = args.root / condition
        fixed_rows = read_jsonl(root / "fixed_state" / "cases.jsonl")
        closed_rows = read_jsonl(root / "closed_loop" / "cases.jsonl")
        audit = json.loads((root / "closed_loop" / "integrity_audit.json").read_text(encoding="utf-8"))
        passed = len(fixed_rows) == 96 and len({x["state_id"] for x in fixed_rows}) == 96 and len(closed_rows) == 120 and len({x["id"] for x in closed_rows}) == 120 and audit.get("verdict") == "PASS"
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition], closed[condition] = fixed_case(fixed_rows), closed_rows
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")
    sham, control = fixed["instruct_sham"], fixed["restore_noncontiguous"]
    results, eligibility = {}, []
    for condition in CONDITIONS:
        cell = {"fixed_rates": {metric: float(np.mean([x[metric] for x in fixed[condition].values()])) for metric in ("zero_action", "overshoot", "direction", "calibration_error")}, "closed_loop": closed_summary(closed[condition])}
        if condition != "instruct_sham":
            cell["paired_vs_sham"] = {metric: paired(fixed[condition], sham, metric, rng) for metric in ("zero_action", "overshoot", "direction", "calibration_error")}
        results[condition] = cell
    control_zero = results["restore_noncontiguous"]["paired_vs_sham"]["zero_action"]["effect"]
    control_over = results["restore_noncontiguous"]["paired_vs_sham"]["overshoot"]["effect"]
    sham_final = results["instruct_sham"]["closed_loop"]["final_joint_rate"]
    for order, condition in enumerate(RESTORATIONS):
        zero = results[condition]["paired_vs_sham"]["zero_action"]["effect"]
        over = results[condition]["paired_vs_sham"]["overshoot"]["effect"]
        final_degradation = sham_final - results[condition]["closed_loop"]["final_joint_rate"]
        selecting = []
        if zero > 0 and zero > control_zero:
            selecting.append("zero_action")
        if over > 0 and over > control_over:
            selecting.append("overshoot")
        eligible = bool(selecting) and final_degradation >= 0
        eligibility.append({"condition": condition, "order": order, "eligible": eligible, "selecting_signatures": selecting, "zero_action_effect": zero, "overshoot_effect": over, "final_joint_degradation": final_degradation, "rank_score": zero + over})
    ranked = sorted((x for x in eligibility if x["eligible"]), key=lambda x: (-x["rank_score"], -x["final_joint_degradation"], x["order"]))
    report = {"schema_version": 1, "protocol": "docs/QWEN_ALIGNED_MECHANISM_PROTOCOL_20260830.md", "integrity": integrity, "results": results, "eligibility": eligibility, "selected_at_most_two": [x["condition"].removeprefix("restore_") for x in ranked[:2]], "claim_boundary": "Discovery selection only; necessity requires frozen dose and case-disjoint confirmation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity": integrity, "eligibility": eligibility, "selected_at_most_two": report["selected_at_most_two"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
