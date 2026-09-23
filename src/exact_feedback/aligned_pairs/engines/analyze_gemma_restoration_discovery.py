#!/usr/bin/env python3
"""Select Gemma restoration groups against the frozen Base--IT utility vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


UTILITIES = ("direction", "nonzero_action", "overshoot_avoidance", "calibration_utility")
MIN_BASE_CONTRAST = {
    "direction": 0.05,
    "nonzero_action": 0.05,
    "overshoot_avoidance": 0.05,
    "calibration_utility": 0.25,
}


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
            "direction": float(np.mean([bool(x["direction_correct"]) for x in cells])),
            "nonzero_action": float(np.mean([int(x["realized_delta"]) != 0 for x in cells])),
            "overshoot_avoidance": float(np.mean([
                abs(int(x["output_error"])) <= abs(int(x["signed_error"])) for x in cells
            ])),
            "calibration_utility": -float(np.mean([
                min(abs(float(x["action_gain"]) - 1.0), 10.0) for x in cells
            ])),
        }
    return result


def paired(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    values = np.asarray([left[x][metric] - right[x][metric] for x in ids])
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    distribution = np.mean(values[draws], axis=1)
    return {
        "cases": len(ids),
        "effect": float(np.mean(values)),
        "cluster_bootstrap_95": [float(x) for x in np.quantile(distribution, [0.025, 0.5, 0.975])],
    }


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
    parser.add_argument("--group-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    group_spec = json.loads(args.group_spec.read_text(encoding="utf-8"))
    groups = [name for name in group_spec["groups"] if name not in ("noncontiguous", "output")]
    conditions = (
        "base_common_interface",
        "instruct_sham",
        "restore_noncontiguous",
        *(f"restore_{name}" for name in groups),
        "restore_output",
    )
    restorations = [name for name in conditions if name.startswith("restore_") and name != "restore_noncontiguous"]
    rng = np.random.default_rng(20260903)
    fixed, closed, integrity = {}, {}, {"passes": True, "conditions": {}}
    for condition in conditions:
        root = args.root / condition
        fixed_rows = read_jsonl(root / "fixed_state" / "cases.jsonl")
        closed_rows = read_jsonl(root / "closed_loop" / "cases.jsonl")
        audit = json.loads((root / "closed_loop" / "integrity_audit.json").read_text(encoding="utf-8"))
        passed = (
            len(fixed_rows) == 96
            and len({x["state_id"] for x in fixed_rows}) == 96
            and len(closed_rows) == 120
            and len({x["id"] for x in closed_rows}) == 120
            and audit.get("verdict") == "PASS"
        )
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition], closed[condition] = fixed_case(fixed_rows), closed_rows
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")

    base = fixed["base_common_interface"]
    instruct = fixed["instruct_sham"]
    control = fixed["restore_noncontiguous"]
    base_contrast = {metric: paired(base, instruct, metric, rng) for metric in UTILITIES}
    control_effect = {metric: paired(control, instruct, metric, rng) for metric in UTILITIES}
    results: dict[str, Any] = {}
    eligibility = []
    instruct_final = closed_summary(closed["instruct_sham"])["final_joint_rate"]

    for condition in conditions:
        cell: dict[str, Any] = {
            "fixed_rates": {
                metric: float(np.mean([x[metric] for x in fixed[condition].values()]))
                for metric in UTILITIES
            },
            "closed_loop": closed_summary(closed[condition]),
        }
        if condition != "instruct_sham":
            cell["paired_vs_instruct"] = {
                metric: paired(fixed[condition], instruct, metric, rng) for metric in UTILITIES
            }
        results[condition] = cell

    for order, condition in enumerate(restorations):
        selecting, progress = [], {}
        for metric in UTILITIES:
            base_delta = base_contrast[metric]["effect"]
            effect = results[condition]["paired_vs_instruct"][metric]["effect"]
            control_delta = control_effect[metric]["effect"]
            meaningful = abs(base_delta) >= MIN_BASE_CONTRAST[metric]
            aligned = meaningful and effect * base_delta > 0
            exceeds_control = aligned and abs(effect) > abs(control_delta)
            if exceeds_control:
                selecting.append(metric)
                progress[metric] = float(np.clip(effect / base_delta, 0.0, 2.0))
        final_degradation = instruct_final - results[condition]["closed_loop"]["final_joint_rate"]
        eligible = bool(selecting) and final_degradation >= 0
        eligibility.append({
            "condition": condition,
            "order": order,
            "eligible": eligible,
            "selecting_utilities": selecting,
            "progress_toward_base": progress,
            "rank_score": float(sum(progress.values())),
            "final_joint_degradation": final_degradation,
        })

    ranked = sorted(
        (x for x in eligibility if x["eligible"]),
        key=lambda x: (-x["rank_score"], -x["final_joint_degradation"], x["order"]),
    )
    report = {
        "schema_version": 1,
        "protocol": "docs/GEMMA_ALIGNED_MECHANISM_PROTOCOL_20260903.md",
        "integrity": integrity,
        "utilities": list(UTILITIES),
        "minimum_base_contrast": MIN_BASE_CONTRAST,
        "base_minus_instruct": base_contrast,
        "control_minus_instruct": control_effect,
        "results": results,
        "eligibility": eligibility,
        "selected_at_most_two": [x["condition"].removeprefix("restore_") for x in ranked[:2]],
        "claim_boundary": "Discovery selection only; confirmation is required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "integrity": integrity,
        "base_minus_instruct": base_contrast,
        "eligibility": eligibility,
        "selected_at_most_two": report["selected_at_most_two"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
