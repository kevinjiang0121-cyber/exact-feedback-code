#!/usr/bin/env python3
"""Analyze Gemma Instruct-to-Base controlled parameter grafts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from exact_feedback.aligned_pairs.engines.analyze_gemma_restoration_discovery import (
    UTILITIES,
    closed_summary,
    fixed_case,
    paired,
    read_jsonl,
)


def oriented_positive_interval(interval: list[float], orientation: float) -> bool:
    return interval[0] > 0 if orientation > 0 else interval[1] < 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dose-analysis", required=True, type=Path)
    parser.add_argument("--discovery-analysis", required=True, type=Path)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dose = json.loads(args.dose_analysis.read_text(encoding="utf-8"))
    discovery = json.loads(args.discovery_analysis.read_text(encoding="utf-8"))
    groups = [x for x in args.groups.split(",") if x] or [x for x in dose["confirmed_groups"] if x != "output"]
    base_name, control_name = "base_common_interface", "graft_noncontiguous_plus_output"
    conditions = [base_name, "graft_output", control_name] + [
        name for group in groups for name in (f"graft_{group}", f"graft_{group}_plus_output")
    ]
    expected_fixed, expected_closed = (96, 120) if args.split == "discovery" else (192, 480)
    fixed, closed, integrity = {}, {}, {"passes": True, "conditions": {}}
    for condition in conditions:
        root = args.root / condition
        fixed_rows = read_jsonl(root / "fixed_state" / "cases.jsonl")
        closed_rows = read_jsonl(root / "closed_loop" / "cases.jsonl")
        audit = json.loads((root / "closed_loop" / "integrity_audit.json").read_text(encoding="utf-8"))
        passed = (
            len(fixed_rows) == expected_fixed
            and len({x["state_id"] for x in fixed_rows}) == expected_fixed
            and len(closed_rows) == expected_closed
            and len({x["id"] for x in closed_rows}) == expected_closed
            and audit.get("verdict") == "PASS"
        )
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition] = fixed_case(fixed_rows)
        closed[condition] = closed_summary(closed_rows)
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")

    rng = np.random.default_rng(20260903)
    base, control = fixed[base_name], fixed[control_name]
    base_contrast = discovery["base_minus_instruct"]
    condition_summaries = {
        condition: {
            "fixed_rates": {
                metric: float(np.mean([row[metric] for row in fixed[condition].values()]))
                for metric in UTILITIES
            },
            "closed_loop": closed[condition],
        }
        for condition in conditions
    }
    results, eligible = {}, []
    for group in groups:
        group_name = f"graft_{group}"
        joint_name = f"graft_{group}_plus_output"
        selecting = dose["results"][group]["selecting_utilities"]
        effects_group_base = {
            metric: paired(fixed[group_name], base, metric, rng)
            for metric in UTILITIES
        }
        effects_base = {metric: paired(fixed[joint_name], base, metric, rng) for metric in UTILITIES}
        effects_control = {metric: paired(fixed[joint_name], control, metric, rng) for metric in UTILITIES}
        orientations = {metric: -float(np.sign(base_contrast[metric]["effect"])) for metric in UTILITIES}
        primary = any(
            effects_base[x]["effect"] * orientations[x] > 0
            and effects_control[x]["effect"] * orientations[x] > 0
            for x in selecting
        )
        noncontradiction = all(effects_base[x]["effect"] * orientations[x] >= 0 for x in selecting)
        calibration_ok = (
            "calibration_utility" in selecting
            or effects_base["calibration_utility"]["effect"] * orientations["calibration_utility"] >= 0
        )
        if args.split == "confirmation":
            interval_pass = any(
                oriented_positive_interval(effects_base[x]["cluster_bootstrap_95"], orientations[x])
                or oriented_positive_interval(effects_control[x]["cluster_bootstrap_95"], orientations[x])
                for x in selecting
            )
        else:
            interval_pass = True
        passed = primary and noncontradiction and calibration_ok and interval_pass
        results[group] = {
            "joint_condition": joint_name,
            "selecting_utilities": selecting,
            "orientations_toward_instruct": orientations,
            "group_vs_base": effects_group_base,
            "joint_vs_base": effects_base,
            "joint_vs_control": effects_control,
            "closed_loop": {
                name: closed[name]
                for name in (base_name, "graft_output", control_name, group_name, joint_name)
            },
            "primary_transfer_pass": primary,
            "selecting_utility_noncontradiction": noncontradiction,
            "calibration_noncontradiction": calibration_ok,
            "interval_pass": interval_pass,
            "graft_pass": passed,
        }
        if passed:
            eligible.append(group)
    report = {
        "schema_version": 1,
        "protocol": "docs/GEMMA_ALIGNED_MECHANISM_PROTOCOL_20260903.md",
        "split": args.split,
        "integrity": integrity,
        "condition_summaries": condition_summaries,
        "results": results,
        "eligible_groups": eligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
