#!/usr/bin/env python3
"""Analyze frozen Phase3D coordinated-controller graft discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


BASE = "base_meta_contract_sham"
MID = "graft_blocks_08_15"
OUTPUT = "graft_output"
CONTROL = "graft_noncontiguous_plus_output"
JOINT = "graft_blocks_08_15_plus_output"
CONDITIONS = (BASE, MID, OUTPUT, CONTROL, JOINT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def calibration_utility(row: dict[str, Any]) -> float:
    return -min(abs(float(row["action_gain"]) - 1.0), 10.0)


def interval(values: np.ndarray) -> list[float]:
    return [
        float(value)
        for value in np.quantile(values, [0.025, 0.5, 0.975])
    ]


def fixed_effect(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    metric: Callable[[dict[str, Any]], float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    by_case: dict[str, list[float]] = {}
    for state_id in ids:
        case_id = left[state_id]["case_id"]
        by_case.setdefault(case_id, []).append(
            metric(left[state_id]) - metric(right[state_id])
        )
    case_values = np.asarray(
        [np.mean(by_case[case_id]) for case_id in sorted(by_case)]
    )
    draws = rng.integers(
        0, len(case_values), size=(10000, len(case_values))
    )
    return {
        "cases": len(case_values),
        "states": len(ids),
        "effect": float(np.mean(case_values)),
        "cluster_bootstrap_95": interval(
            np.mean(case_values[draws], axis=1)
        ),
    }


def closed_effect(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    groups: dict[str, list[str]] = {}
    for case_id in ids:
        groups.setdefault(left[case_id]["source"], []).append(case_id)
    effects = {
        case_id: float(left[case_id]["final_joint"])
        - float(right[case_id]["final_joint"])
        for case_id in ids
    }
    draws = np.empty(10000)
    for index in range(10000):
        sampled = []
        for source_ids in groups.values():
            positions = rng.integers(
                0, len(source_ids), size=len(source_ids)
            )
            sampled.extend(source_ids[position] for position in positions)
        draws[index] = np.mean([effects[case_id] for case_id in sampled])
    return {
        "cases": len(ids),
        "effect": float(np.mean(list(effects.values()))),
        "source_stratified_paired_bootstrap_95": interval(draws),
        "left_only_success": sum(
            bool(left[x]["final_joint"]) and not bool(right[x]["final_joint"])
            for x in ids
        ),
        "right_only_success": sum(
            bool(right[x]["final_joint"]) and not bool(left[x]["final_joint"])
            for x in ids
        ),
    }


def fixed_factorial(
    rows: dict[str, dict[str, dict[str, Any]]],
    metric: Callable[[dict[str, Any]], float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(set.intersection(*(set(rows[x]) for x in (BASE, MID, OUTPUT, JOINT))))
    by_case: dict[str, list[float]] = {}
    for state_id in ids:
        value = (
            metric(rows[JOINT][state_id])
            - metric(rows[MID][state_id])
            - metric(rows[OUTPUT][state_id])
            + metric(rows[BASE][state_id])
        )
        by_case.setdefault(rows[BASE][state_id]["case_id"], []).append(value)
    case_values = np.asarray(
        [np.mean(by_case[case_id]) for case_id in sorted(by_case)]
    )
    draws = rng.integers(
        0, len(case_values), size=(10000, len(case_values))
    )
    return {
        "cases": len(case_values),
        "states": len(ids),
        "effect": float(np.mean(case_values)),
        "cluster_bootstrap_95": interval(
            np.mean(case_values[draws], axis=1)
        ),
    }


def closed_factorial(
    rows: dict[str, dict[str, dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(set.intersection(*(set(rows[x]) for x in (BASE, MID, OUTPUT, JOINT))))
    groups: dict[str, list[str]] = {}
    for case_id in ids:
        groups.setdefault(rows[BASE][case_id]["source"], []).append(case_id)
    effects = {
        case_id: (
            float(rows[JOINT][case_id]["final_joint"])
            - float(rows[MID][case_id]["final_joint"])
            - float(rows[OUTPUT][case_id]["final_joint"])
            + float(rows[BASE][case_id]["final_joint"])
        )
        for case_id in ids
    }
    draws = np.empty(10000)
    for index in range(10000):
        sampled = []
        for source_ids in groups.values():
            positions = rng.integers(
                0, len(source_ids), size=len(source_ids)
            )
            sampled.extend(source_ids[position] for position in positions)
        draws[index] = np.mean([effects[case_id] for case_id in sampled])
    return {
        "cases": len(ids),
        "effect": float(np.mean(list(effects.values()))),
        "source_stratified_paired_bootstrap_95": interval(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20260730)

    fixed = {}
    closed = {}
    integrity = {"passes": True, "conditions": {}}
    for condition in CONDITIONS:
        condition_root = args.root / condition
        fixed_rows = read_jsonl(condition_root / "fixed_state" / "cases.jsonl")
        closed_rows = read_jsonl(
            condition_root / "closed_loop" / "cases.jsonl"
        )
        audit = json.loads(
            (
                condition_root
                / "closed_loop"
                / "integrity_audit.json"
            ).read_text(encoding="utf-8")
        )
        passed = (
            len(fixed_rows) == 96
            and len({row["state_id"] for row in fixed_rows}) == 96
            and len(closed_rows) == 120
            and len({row["id"] for row in closed_rows}) == 120
            and audit.get("verdict") == "PASS"
        )
        integrity["conditions"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        fixed[condition] = {row["state_id"]: row for row in fixed_rows}
        closed[condition] = {row["id"]: row for row in closed_rows}
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")

    def direction(row: dict[str, Any]) -> float:
        return float(bool(row["direction_correct"]))
    results = {}
    for condition in CONDITIONS:
        fixed_values = list(fixed[condition].values())
        closed_values = list(closed[condition].values())
        results[condition] = {
            "fixed_direction_rate": float(
                np.mean([direction(row) for row in fixed_values])
            ),
            "fixed_calibration_utility": float(
                np.mean([calibration_utility(row) for row in fixed_values])
            ),
            "main120_final_joint_rate": float(
                np.mean([bool(row["final_joint"]) for row in closed_values])
            ),
        }

    effects = {
        "joint_vs_base": {
            "direction": fixed_effect(
                fixed[JOINT], fixed[BASE], direction, rng
            ),
            "calibration": fixed_effect(
                fixed[JOINT], fixed[BASE], calibration_utility, rng
            ),
            "final_joint": closed_effect(
                closed[JOINT], closed[BASE], rng
            ),
        },
        "joint_vs_control": {
            "direction": fixed_effect(
                fixed[JOINT], fixed[CONTROL], direction, rng
            ),
            "calibration": fixed_effect(
                fixed[JOINT], fixed[CONTROL], calibration_utility, rng
            ),
            "final_joint": closed_effect(
                closed[JOINT], closed[CONTROL], rng
            ),
        },
        "factorial_synergy": {
            "direction": fixed_factorial(fixed, direction, rng),
            "calibration": fixed_factorial(
                fixed, calibration_utility, rng
            ),
            "final_joint": closed_factorial(closed, rng),
        },
    }
    recovery_direction = effects["joint_vs_base"]["direction"]
    recovery_joint = effects["joint_vs_base"]["final_joint"]
    control_direction = effects["joint_vs_control"]["direction"]
    control_joint = effects["joint_vs_control"]["final_joint"]
    synergy_direction = effects["factorial_synergy"]["direction"]
    synergy_joint = effects["factorial_synergy"]["final_joint"]
    calibration_ok = (
        effects["joint_vs_base"]["calibration"]["effect"] >= 0
        and effects["joint_vs_control"]["calibration"]["effect"] >= 0
    )
    recovery_pass = (
        recovery_direction["cluster_bootstrap_95"][0] > 0
        and recovery_joint[
            "source_stratified_paired_bootstrap_95"
        ][0] > 0
    )
    control_pass = (
        control_direction["cluster_bootstrap_95"][0] > 0
        and control_joint[
            "source_stratified_paired_bootstrap_95"
        ][0] > 0
    )
    synergy_pass = (
        (
            synergy_direction["cluster_bootstrap_95"][0] > 0
            and synergy_joint["effect"] >= 0
        )
        or (
            synergy_joint[
                "source_stratified_paired_bootstrap_95"
            ][0] > 0
            and synergy_direction["effect"] >= 0
        )
    )
    gate = {
        "recovery_pass": recovery_pass,
        "control_superiority_pass": control_pass,
        "synergy_pass": synergy_pass,
        "calibration_noncontradiction_pass": calibration_ok,
        "controller_graft_discovery_pass": (
            recovery_pass and control_pass and synergy_pass and calibration_ok
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": "docs/PHASE3D_CONTROLLER_GRAFT_DISCOVERY_PROTOCOL_20260730.md",
        "integrity": integrity,
        "results": results,
        "effects": effects,
        "gate": gate,
        "claim_boundary": (
            "Discovery only. A method claim requires alpha=0.5 and "
            "case-disjoint Combined-480 confirmation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
