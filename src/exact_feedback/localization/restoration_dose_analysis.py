#!/usr/bin/env python3
"""Evaluate the frozen Phase3B dose and case-disjoint confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATES = ("restore_blocks_08_15", "restore_output")
DOSE_MAP = {
    "restore_blocks_08_15": "dose50_blocks_08_15",
    "restore_output": "dose50_output",
}
CONFIRMATION = (
    "sham_copy",
    "restore_noncontiguous",
    *CANDIDATES,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def fixed_direction(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["direction_correct"]) for row in rows]))


def final_joint(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(row["final_joint"]) for row in rows]))


def fixed_case_direction(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(
            bool(row["direction_correct"])
        )
    if any(len(values) != 4 for values in grouped.values()):
        raise RuntimeError("each fixed-state case must have four states")
    return {
        case_id: float(np.mean(values))
        for case_id, values in grouped.items()
    }


def quantiles(values: np.ndarray) -> list[float]:
    return [
        float(value)
        for value in np.quantile(values, [0.025, 0.5, 0.975])
    ]


def paired_fixed_interval(
    reference: list[dict[str, Any]],
    condition: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    ref = fixed_case_direction(reference)
    cond = fixed_case_direction(condition)
    ids = sorted(set(ref) & set(cond))
    effects = np.asarray([ref[x] - cond[x] for x in ids])
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    return {
        "cases": len(ids),
        "loss": float(np.mean(effects)),
        "cluster_bootstrap_95": quantiles(
            np.mean(effects[draws], axis=1)
        ),
    }


def paired_source_stratified_interval(
    reference: list[dict[str, Any]],
    condition: list[dict[str, Any]],
    rng: np.random.Generator,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    ref = {row["id"]: row for row in reference}
    cond = {row["id"]: row for row in condition}
    ids = sorted(set(ref) & set(cond))
    if allowed_ids is not None:
        ids = [case_id for case_id in ids if case_id in allowed_ids]
    groups: dict[str, list[str]] = {}
    for case_id in ids:
        groups.setdefault(ref[case_id]["source"], []).append(case_id)
    effects = {
        case_id: float(ref[case_id]["final_joint"])
        - float(cond[case_id]["final_joint"])
        for case_id in ids
    }
    draws = np.empty(10000, dtype=float)
    for index in range(10000):
        sampled = []
        for source_ids in groups.values():
            positions = rng.integers(0, len(source_ids), size=len(source_ids))
            sampled.extend(source_ids[position] for position in positions)
        draws[index] = np.mean([effects[case_id] for case_id in sampled])
    return {
        "cases": len(ids),
        "loss": float(np.mean(list(effects.values()))),
        "source_stratified_paired_bootstrap_95": quantiles(draws),
        "reference_only_success": sum(
            bool(ref[x]["final_joint"]) and not bool(cond[x]["final_joint"])
            for x in ids
        ),
        "condition_only_success": sum(
            bool(cond[x]["final_joint"]) and not bool(ref[x]["final_joint"])
            for x in ids
        ),
    }


def load_audit(path: Path) -> bool:
    return json.loads(path.read_text(encoding="utf-8")).get("verdict") == "PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20260730)

    discovery_fixed = {}
    discovery_closed = {}
    for condition in ("sham_copy", "restore_noncontiguous", *CANDIDATES):
        condition_root = args.discovery_root / condition
        discovery_fixed[condition] = read_jsonl(
            condition_root / "fixed_state" / "cases.jsonl"
        )
        discovery_closed[condition] = read_jsonl(
            condition_root / "closed_loop" / "cases.jsonl"
        )

    dose = {}
    integrity: dict[str, Any] = {"passes": True, "dose": {}, "confirmation": {}}
    for full_condition, dose_condition in DOSE_MAP.items():
        condition_root = args.root / "dose" / dose_condition
        fixed = read_jsonl(condition_root / "fixed_state" / "cases.jsonl")
        closed = read_jsonl(condition_root / "closed_loop" / "cases.jsonl")
        passed = (
            len(fixed) == 96
            and len({row["state_id"] for row in fixed}) == 96
            and len(closed) == 120
            and len({row["id"] for row in closed}) == 120
            and load_audit(
                condition_root / "closed_loop" / "integrity_audit.json"
            )
        )
        integrity["dose"][dose_condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        half_direction = paired_fixed_interval(
            discovery_fixed["sham_copy"], fixed, rng
        )
        full_direction = paired_fixed_interval(
            discovery_fixed["sham_copy"],
            discovery_fixed[full_condition],
            rng,
        )
        half_joint = paired_source_stratified_interval(
            discovery_closed["sham_copy"], closed, rng
        )
        full_joint = paired_source_stratified_interval(
            discovery_closed["sham_copy"],
            discovery_closed[full_condition],
            rng,
        )
        direction_monotonic = (
            0 < half_direction["loss"] <= full_direction["loss"]
        )
        joint_monotonic = 0 < half_joint["loss"] <= full_joint["loss"]
        dose[full_condition] = {
            "half_condition": dose_condition,
            "direction": {"half": half_direction, "full": full_direction},
            "final_joint": {"half": half_joint, "full": full_joint},
            "strict_direction_monotonic": direction_monotonic,
            "strict_final_joint_monotonic": joint_monotonic,
            "strict_dose_support": direction_monotonic and joint_monotonic,
        }

    combined = {}
    for condition in CONFIRMATION:
        condition_root = args.root / "confirmation" / condition / "combined480"
        rows = read_jsonl(condition_root / "cases.jsonl")
        passed = (
            len(rows) == 480
            and len({row["id"] for row in rows}) == 480
            and load_audit(condition_root / "integrity_audit.json")
        )
        integrity["confirmation"][condition] = passed
        integrity["passes"] = integrity["passes"] and passed
        combined[condition] = rows

    discovery_ids = {
        row["id"] for row in discovery_closed["sham_copy"]
    }
    all_ids = {row["id"] for row in combined["sham_copy"]}
    new_ids = all_ids - discovery_ids
    if len(new_ids) != 360:
        raise RuntimeError(f"expected 360 case-disjoint IDs, got {len(new_ids)}")

    confirmation = {}
    control_new = paired_source_stratified_interval(
        combined["sham_copy"],
        combined["restore_noncontiguous"],
        rng,
        new_ids,
    )
    control_all = paired_source_stratified_interval(
        combined["sham_copy"],
        combined["restore_noncontiguous"],
        rng,
    )
    for condition in CANDIDATES:
        candidate_new = paired_source_stratified_interval(
            combined["sham_copy"], combined[condition], rng, new_ids
        )
        candidate_all = paired_source_stratified_interval(
            combined["sham_copy"], combined[condition], rng
        )
        excess_new = paired_source_stratified_interval(
            combined["restore_noncontiguous"],
            combined[condition],
            rng,
            new_ids,
        )
        excess_all = paired_source_stratified_interval(
            combined["restore_noncontiguous"],
            combined[condition],
            rng,
        )
        discovery_direction_loss = (
            fixed_direction(discovery_fixed["sham_copy"])
            - fixed_direction(discovery_fixed[condition])
        )
        ci_positive = (
            candidate_new["source_stratified_paired_bootstrap_95"][0] > 0
        )
        excess_ci_positive = (
            excess_new["source_stratified_paired_bootstrap_95"][0] > 0
        )
        coherent = discovery_direction_loss > 0 and candidate_new["loss"] > 0
        items_1_to_4 = (
            discovery_direction_loss > 0
            and ci_positive
            and excess_ci_positive
            and coherent
        )
        confirmation[condition] = {
            "discovery_direction_loss": discovery_direction_loss,
            "new_360": candidate_new,
            "combined_480": candidate_all,
            "excess_over_noncontiguous_new_360": excess_new,
            "excess_over_noncontiguous_combined_480": excess_all,
            "gate_items_1_to_4_pass": items_1_to_4,
            "strict_dose_support": dose[condition]["strict_dose_support"],
            "gate_b_pass": (
                items_1_to_4 and dose[condition]["strict_dose_support"]
            ),
            "nonlinear_full_restoration_necessity_only": (
                items_1_to_4 and not dose[condition]["strict_dose_support"]
            ),
        }

    report = {
        "schema_version": 1,
        "protocol": "docs/PHASE3B_DOSE_CONFIRMATION_PROTOCOL_20260730.md",
        "integrity": integrity,
        "dose": dose,
        "confirmation_control": {
            "new_360": control_new,
            "combined_480": control_all,
        },
        "confirmation": confirmation,
        "claim_boundary": (
            "Gate B only. Phase 3C activation mediation remains unauthorized."
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
