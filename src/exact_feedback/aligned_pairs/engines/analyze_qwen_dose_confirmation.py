#!/usr/bin/env python3
"""Adjudicate Qwen restoration dose and case-disjoint confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def case_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    result = {}
    for case_id, cells in grouped.items():
        result[case_id] = {
            "zero_action": float(np.mean([int(x["realized_delta"]) == 0 for x in cells])),
            "overshoot": float(np.mean([abs(int(x["output_error"])) > abs(int(x["signed_error"])) for x in cells])),
        }
    return result


def paired(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]], metric: str, rng: np.random.Generator) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    values = np.asarray([left[x][metric] - right[x][metric] for x in ids])
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    boot = np.mean(values[draws], axis=1)
    return {"cases": len(ids), "effect": float(np.mean(values)), "cluster_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.5, 0.975])]}


def trajectory_metrics(row: dict[str, Any]) -> dict[str, float]:
    texts = [cell["text"] for cell in row["rounds"]]
    return {
        "final_joint": float(bool(row["final_joint"])),
        "terminal_error_utility": -float(abs(int(row["rounds"][-1]["error"]))),
        "recurrence_avoidance": float(len(texts) == len(set(texts))),
    }


def paired_trajectory(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], metric: str, rng: np.random.Generator) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    effects = {x: trajectory_metrics(left[x])[metric] - trajectory_metrics(right[x])[metric] for x in ids}
    sources: dict[str, list[str]] = {}
    for case_id in ids:
        sources.setdefault(left[case_id]["source"], []).append(case_id)
    draws = []
    for _ in range(10000):
        sampled = []
        for source_ids in sources.values():
            sampled.extend(source_ids[int(i)] for i in rng.integers(0, len(source_ids), size=len(source_ids)))
        draws.append(float(np.mean([effects[x] for x in sampled])))
    return {"cases": len(ids), "effect": float(np.mean(list(effects.values()))), "source_stratified_paired_bootstrap_95": [float(x) for x in np.quantile(draws, [0.025, 0.5, 0.975])]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-analysis", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    discovery = json.loads(args.discovery_analysis.read_text(encoding="utf-8"))
    groups = discovery["selected_at_most_two"]
    rng = np.random.default_rng(20260830)
    if not groups:
        report = {"schema_version": 1, "integrity": {"passes": True}, "selected_groups": [], "confirmed_groups": [], "gate": "STOP_NO_DISCOVERY_SELECTION"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    conditions = ["instruct_sham", "restore_noncontiguous"] + [name for group in groups for name in (f"restore50_{group}", f"restore_{group}")]
    fixed, closed_rows, integrity = {}, {}, {"passes": True, "conditions": {}}
    for condition in conditions:
        root = args.root / condition
        rows = read_jsonl(root / "fixed_state_confirmation" / "cases.jsonl")
        closed = read_jsonl(root / "combined480" / "cases.jsonl")
        audit = json.loads((root / "combined480" / "integrity_audit.json").read_text(encoding="utf-8"))
        passed = len(rows) == 192 and len({x["state_id"] for x in rows}) == 192 and len(closed) == 480 and len({x["id"] for x in closed}) == 480 and audit.get("verdict") == "PASS"
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition] = case_metrics(rows)
        closed_rows[condition] = {x["id"]: x for x in closed}
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")
    sham, control = fixed["instruct_sham"], fixed["restore_noncontiguous"]
    control_effects = {metric: paired(control, sham, metric, rng) for metric in ("zero_action", "overshoot")}
    discovery_lookup = {x["condition"].removeprefix("restore_"): x for x in discovery["eligibility"]}
    results, confirmed = {}, []
    for group in groups:
        half, full = fixed[f"restore50_{group}"], fixed[f"restore_{group}"]
        half_effects = {metric: paired(half, sham, metric, rng) for metric in ("zero_action", "overshoot")}
        full_effects = {metric: paired(full, sham, metric, rng) for metric in ("zero_action", "overshoot")}
        trajectory = {
            "half_vs_sham": {metric: paired_trajectory(closed_rows[f"restore50_{group}"], closed_rows["instruct_sham"], metric, rng) for metric in ("final_joint", "terminal_error_utility", "recurrence_avoidance")},
            "full_vs_sham": {metric: paired_trajectory(closed_rows[f"restore_{group}"], closed_rows["instruct_sham"], metric, rng) for metric in ("final_joint", "terminal_error_utility", "recurrence_avoidance")},
            "full_vs_control": {metric: paired_trajectory(closed_rows[f"restore_{group}"], closed_rows["restore_noncontiguous"], metric, rng) for metric in ("final_joint", "terminal_error_utility", "recurrence_avoidance")},
        }
        selecting = discovery_lookup[group]["selecting_signatures"]
        cells = {}
        for metric in selecting:
            h, f, c = half_effects[metric]["effect"], full_effects[metric]["effect"], control_effects[metric]["effect"]
            ci = full_effects[metric]["cluster_bootstrap_95"]
            cells[metric] = {
                "discovery_direction_replicates": f > 0,
                "exceeds_control": f > c,
                "dose_order_noncontradiction": h >= -abs(f) and h <= f + abs(f),
                "interval_excludes_zero": ci[0] > 0,
            }
        passed = bool(cells) and any(x["interval_excludes_zero"] for x in cells.values()) and all(x["discovery_direction_replicates"] and x["exceeds_control"] and x["dose_order_noncontradiction"] for x in cells.values())
        results[group] = {"selecting_signatures": selecting, "half_vs_sham": half_effects, "full_vs_sham": full_effects, "control_vs_sham": control_effects, "trajectory_effects": trajectory, "signature_gates": cells, "necessity_pass": passed}
        if passed:
            confirmed.append(group)
    report = {"schema_version": 1, "protocol": "docs/QWEN_ALIGNED_MECHANISM_PROTOCOL_20260830.md", "integrity": integrity, "selected_groups": groups, "results": results, "confirmed_groups": confirmed, "gate": "PASS_TO_Q4_Q5" if confirmed else "STOP_NO_CONFIRMED_GROUP"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
