#!/usr/bin/env python3
"""Analyze Qwen Instruct-to-Base controlled parameter grafts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cases(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    return {case_id: {"zero_action": float(np.mean([int(x["realized_delta"]) == 0 for x in cells])), "overshoot_avoidance": float(np.mean([abs(int(x["output_error"])) <= abs(int(x["signed_error"])) for x in cells]))} for case_id, cells in grouped.items()}


def effect(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]], metric: str, rng: np.random.Generator) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    values = np.asarray([left[x][metric] - right[x][metric] for x in ids])
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    boot = np.mean(values[draws], axis=1)
    return {"cases": len(ids), "effect": float(np.mean(values)), "cluster_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.5, 0.975])]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dose-analysis", required=True, type=Path)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dose = json.loads(args.dose_analysis.read_text(encoding="utf-8"))
    groups = [x for x in args.groups.split(",") if x] or [x for x in dose["confirmed_groups"] if x != "output"]
    base_name, control_name = "base_common_interface", "graft_noncontiguous_plus_output"
    conditions = [base_name, "graft_output", control_name] + [name for group in groups for name in (f"graft_{group}", f"graft_{group}_plus_output")]
    expected_fixed, expected_closed = (96, 120) if args.split == "discovery" else (192, 480)
    fixed, integrity = {}, {"passes": True, "conditions": {}}
    for condition in conditions:
        root = args.root / condition
        fixed_rows = read_jsonl(root / "fixed_state" / "cases.jsonl")
        closed_rows = read_jsonl(root / "closed_loop" / "cases.jsonl")
        audit = json.loads((root / "closed_loop" / "integrity_audit.json").read_text(encoding="utf-8"))
        passed = len(fixed_rows) == expected_fixed and len({x["state_id"] for x in fixed_rows}) == expected_fixed and len(closed_rows) == expected_closed and len({x["id"] for x in closed_rows}) == expected_closed and audit.get("verdict") == "PASS"
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition] = cases(fixed_rows)
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")
    rng = np.random.default_rng(20260830)
    base, control = fixed[base_name], fixed[control_name]
    results, eligible = {}, []
    for group in groups:
        joint_name = f"graft_{group}_plus_output"
        selecting = dose["results"][group]["selecting_signatures"]
        selecting_metrics = ["zero_action" if x == "zero_action" else "overshoot_avoidance" for x in selecting]
        effects_base = {metric: effect(fixed[joint_name], base, metric, rng) for metric in ("zero_action", "overshoot_avoidance")}
        effects_control = {metric: effect(fixed[joint_name], control, metric, rng) for metric in ("zero_action", "overshoot_avoidance")}
        other = [x for x in ("zero_action", "overshoot_avoidance") if x not in selecting_metrics]
        primary = any(effects_base[x]["effect"] > 0 and effects_control[x]["effect"] > 0 for x in selecting_metrics)
        other_ok = all(effects_base[x]["effect"] >= 0 for x in other)
        interval = any(effects_base[x]["cluster_bootstrap_95"][0] > 0 or effects_control[x]["cluster_bootstrap_95"][0] > 0 for x in selecting_metrics) if args.split == "confirmation" else True
        passed = primary and other_ok and interval
        results[group] = {"joint_condition": joint_name, "selecting_signatures": selecting, "joint_vs_base": effects_base, "joint_vs_control": effects_control, "primary_transfer_pass": primary, "other_signature_noncontradiction": other_ok, "interval_pass": interval, "graft_pass": passed}
        if passed:
            eligible.append(group)
    report = {"schema_version": 1, "protocol": "docs/QWEN_ALIGNED_MECHANISM_PROTOCOL_20260830.md", "split": args.split, "integrity": integrity, "results": results, "eligible_groups": eligible}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

