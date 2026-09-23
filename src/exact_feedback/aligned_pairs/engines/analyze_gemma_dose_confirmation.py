#!/usr/bin/env python3
"""Adjudicate Gemma restoration dose and case-disjoint confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from exact_feedback.aligned_pairs.engines.analyze_gemma_restoration_discovery import UTILITIES, fixed_case, paired, read_jsonl


def trajectory_metrics(row: dict[str, Any]) -> dict[str, float]:
    texts = [cell["text"] for cell in row["rounds"]]
    terminal_self_loop = (
        not bool(row["final_joint"])
        and len(texts) > 1
        and texts[-1] == texts[-2]
    )
    return {
        "final_joint": float(bool(row["final_joint"])),
        "terminal_error_utility": -float(abs(int(row["rounds"][-1]["error"]))),
        "recurrence_avoidance": float(len(texts) == len(set(texts))),
        "terminal_self_loop_avoidance": float(not terminal_self_loop),
    }


def paired_trajectory(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
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
    return {
        "cases": len(ids),
        "effect": float(np.mean(list(effects.values()))),
        "source_stratified_paired_bootstrap_95": [float(x) for x in np.quantile(draws, [0.025, 0.5, 0.975])],
    }


def oriented_interval_excludes_zero(interval: list[float], orientation: float) -> bool:
    return interval[0] > 0 if orientation > 0 else interval[1] < 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-analysis", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    discovery = json.loads(args.discovery_analysis.read_text(encoding="utf-8"))
    groups = discovery["selected_at_most_two"]
    rng = np.random.default_rng(20260903)
    if not groups:
        report = {
            "schema_version": 1,
            "protocol": "docs/GEMMA_ALIGNED_MECHANISM_PROTOCOL_20260903.md",
            "integrity": {"passes": True},
            "selected_groups": [],
            "confirmed_groups": [],
            "gate": "STOP_NO_DISCOVERY_SELECTION",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    conditions = ["instruct_sham", "restore_noncontiguous"] + [
        name for group in groups for name in (f"restore50_{group}", f"restore_{group}")
    ]
    fixed, closed_rows, integrity = {}, {}, {"passes": True, "conditions": {}}
    for condition in conditions:
        root = args.root / condition
        rows = read_jsonl(root / "fixed_state_confirmation" / "cases.jsonl")
        closed = read_jsonl(root / "combined480" / "cases.jsonl")
        audit = json.loads((root / "combined480" / "integrity_audit.json").read_text(encoding="utf-8"))
        passed = (
            len(rows) == 192
            and len({x["state_id"] for x in rows}) == 192
            and len(closed) == 480
            and len({x["id"] for x in closed}) == 480
            and audit.get("verdict") == "PASS"
        )
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition] = fixed_case(rows)
        closed_rows[condition] = {x["id"]: x for x in closed}
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")

    sham, control = fixed["instruct_sham"], fixed["restore_noncontiguous"]
    control_effects = {metric: paired(control, sham, metric, rng) for metric in UTILITIES}
    discovery_lookup = {
        x["condition"].removeprefix("restore_"): x for x in discovery["eligibility"]
    }
    base_contrast = discovery["base_minus_instruct"]
    results, confirmed = {}, []
    for group in groups:
        half = fixed[f"restore50_{group}"]
        full = fixed[f"restore_{group}"]
        half_effects = {metric: paired(half, sham, metric, rng) for metric in UTILITIES}
        full_effects = {metric: paired(full, sham, metric, rng) for metric in UTILITIES}
        trajectory = {
            "half_vs_sham": {
                metric: paired_trajectory(closed_rows[f"restore50_{group}"], closed_rows["instruct_sham"], metric, rng)
                for metric in (
                    "final_joint",
                    "terminal_error_utility",
                    "recurrence_avoidance",
                    "terminal_self_loop_avoidance",
                )
            },
            "full_vs_sham": {
                metric: paired_trajectory(closed_rows[f"restore_{group}"], closed_rows["instruct_sham"], metric, rng)
                for metric in (
                    "final_joint",
                    "terminal_error_utility",
                    "recurrence_avoidance",
                    "terminal_self_loop_avoidance",
                )
            },
            "full_vs_control": {
                metric: paired_trajectory(closed_rows[f"restore_{group}"], closed_rows["restore_noncontiguous"], metric, rng)
                for metric in (
                    "final_joint",
                    "terminal_error_utility",
                    "recurrence_avoidance",
                    "terminal_self_loop_avoidance",
                )
            },
        }
        selecting = discovery_lookup[group]["selecting_utilities"]
        gates = {}
        for metric in selecting:
            orientation = float(np.sign(base_contrast[metric]["effect"]))
            h = half_effects[metric]["effect"] * orientation
            f = full_effects[metric]["effect"] * orientation
            c = control_effects[metric]["effect"] * orientation
            gates[metric] = {
                "orientation_toward_base": orientation,
                "discovery_direction_replicates": f > 0,
                "exceeds_control": f > c,
                # The frozen rule only excludes a half-dose reversal larger
                # than the full-dose effect. A stronger half dose is nonlinear
                # but is not, by itself, a contradiction.
                "dose_order_noncontradiction": h >= -abs(f),
                "interval_excludes_zero": oriented_interval_excludes_zero(
                    full_effects[metric]["cluster_bootstrap_95"], orientation
                ),
            }
        final_not_improved = trajectory["full_vs_sham"]["final_joint"]["effect"] <= 0
        passed = (
            bool(gates)
            and all(x["discovery_direction_replicates"] and x["exceeds_control"] and x["dose_order_noncontradiction"] for x in gates.values())
            and any(x["interval_excludes_zero"] for x in gates.values())
            and final_not_improved
        )
        results[group] = {
            "selecting_utilities": selecting,
            "half_vs_sham": half_effects,
            "full_vs_sham": full_effects,
            "control_vs_sham": control_effects,
            "trajectory_effects": trajectory,
            "utility_gates": gates,
            "final_joint_not_improved": final_not_improved,
            "necessity_pass": passed,
        }
        if passed:
            confirmed.append(group)
    report = {
        "schema_version": 1,
        "protocol": "docs/GEMMA_ALIGNED_MECHANISM_PROTOCOL_20260903.md",
        "integrity": integrity,
        "selected_groups": groups,
        "results": results,
        "confirmed_groups": confirmed,
        "gate": "PASS_TO_G4_G5" if confirmed else "STOP_NO_CONFIRMED_GROUP",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
