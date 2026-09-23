#!/usr/bin/env python3
"""Analyze frozen Phase3C C1 bidirectional activation mediation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


UNITS = ("blocks_08_15", "output_boundary")
FAMILIES = ("correction", "near_target")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def clipped_calibration(row: dict[str, Any]) -> float:
    return -min(abs(float(row["action_gain"]) - 1.0), 10.0)


def cluster_interval(
    values: dict[str, list[float]], rng: np.random.Generator
) -> dict[str, Any]:
    case_ids = sorted(values)
    case_means = np.asarray(
        [float(np.mean(values[case_id])) for case_id in case_ids]
    )
    draws = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
    boot = np.mean(case_means[draws], axis=1)
    return {
        "cases": len(case_ids),
        "states": sum(len(values[case_id]) for case_id in case_ids),
        "mean": float(np.mean(case_means)),
        "cluster_bootstrap_95": [
            float(value)
            for value in np.quantile(boot, [0.025, 0.5, 0.975])
        ],
    }


def paired_effect(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    state_ids: list[str],
    metric: Callable[[dict[str, Any]], float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    by_case: dict[str, list[float]] = {}
    for state_id in state_ids:
        case_id = left[state_id]["case_id"]
        by_case.setdefault(case_id, []).append(
            metric(left[state_id]) - metric(right[state_id])
        )
    return cluster_interval(by_case, rng)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-clean", required=True, type=Path)
    parser.add_argument("--instruct-clean", required=True, type=Path)
    parser.add_argument("--patched", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20260730)

    base = {
        row["state_id"]: row
        for row in read_jsonl(args.base_clean)
        if row["split"] == "discovery"
    }
    instruct = {
        row["state_id"]: row
        for row in read_jsonl(args.instruct_clean)
        if row["split"] == "discovery"
    }
    patched_rows = read_jsonl(args.patched)
    patched = {
        (row["state_id"], row["unit"], row["direction"]): row
        for row in patched_rows
    }
    expected_ids = set(base) & set(instruct)
    integrity = {
        "passes": (
            len(base) == 96
            and len(instruct) == 96
            and len(expected_ids) == 96
            and len(patched_rows) == 384
            and len(patched) == 384
        ),
        "base_states": len(base),
        "instruct_states": len(instruct),
        "patched_rows": len(patched_rows),
        "unique_patched_cells": len(patched),
    }
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")

    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "direction_correct": lambda row: float(bool(row["direction_correct"])),
        "calibration_utility": clipped_calibration,
        "exact_length": lambda row: float(bool(row["exact_length"])),
        "anchor_retention": lambda row: float(not row["missing_anchors"]),
        "joint_success": lambda row: float(bool(row["joint_success"])),
    }
    results = {}
    eligible = []
    for unit in UNITS:
        results[unit] = {}
        for family in FAMILIES:
            state_ids = sorted(
                state_id
                for state_id in expected_ids
                if base[state_id]["state_family"] == family
            )
            denoised = {
                state_id: patched[(state_id, unit, "denoise")]
                for state_id in state_ids
            }
            noised = {
                state_id: patched[(state_id, unit, "noise")]
                for state_id in state_ids
            }
            base_subset = {state_id: base[state_id] for state_id in state_ids}
            instruct_subset = {
                state_id: instruct[state_id] for state_id in state_ids
            }
            cell = {"states": len(state_ids), "effects": {}}
            for metric_name, metric in metrics.items():
                total = paired_effect(
                    instruct_subset, base_subset, state_ids, metric, rng
                )
                denoising = paired_effect(
                    denoised, base_subset, state_ids, metric, rng
                )
                noising = paired_effect(
                    instruct_subset, noised, state_ids, metric, rng
                )
                interaction_by_state = {}
                for state_id in state_ids:
                    case_id = base[state_id]["case_id"]
                    value = (
                        metric(instruct_subset[state_id])
                        - metric(noised[state_id])
                        - metric(denoised[state_id])
                        + metric(base_subset[state_id])
                    )
                    interaction_by_state.setdefault(case_id, []).append(value)
                cell["effects"][metric_name] = {
                    "total_checkpoint_effect": total,
                    "denoising_indirect": denoising,
                    "noising_loss": noising,
                    "mediated_interaction": cluster_interval(
                        interaction_by_state, rng
                    ),
                }
            direction = cell["effects"]["direction_correct"]
            calibration = cell["effects"]["calibration_utility"]
            bidirectional = (
                direction["denoising_indirect"]["mean"]
                + direction["noising_loss"]["mean"]
            ) / 2
            calibration_bidirectional = (
                calibration["denoising_indirect"]["mean"]
                + calibration["noising_loss"]["mean"]
            ) / 2
            cell["mean_bidirectional_direction_effect"] = bidirectional
            cell["mean_bidirectional_calibration_effect"] = (
                calibration_bidirectional
            )
            cell["eligible"] = (
                direction["denoising_indirect"]["mean"] > 0
                and direction["noising_loss"]["mean"] > 0
                and bidirectional > 0
                and calibration_bidirectional >= 0
            )
            if cell["eligible"]:
                eligible.append(
                    {
                        "unit": unit,
                        "state_family": family,
                        "mean_bidirectional_direction_effect": bidirectional,
                        "mean_bidirectional_calibration_effect": (
                            calibration_bidirectional
                        ),
                    }
                )
            results[unit][family] = cell

    eligible.sort(
        key=lambda row: (
            row["mean_bidirectional_direction_effect"],
            row["mean_bidirectional_calibration_effect"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "protocol": "docs/PHASE3C_ACTIVATION_MEDIATION_DISCOVERY_PROTOCOL_20260730.md",
        "integrity": integrity,
        "results": results,
        "eligible_ranked": eligible,
        "selected_at_most_one": eligible[:1],
        "claim_boundary": (
            "Discovery only; activation mediation requires frozen controls, "
            "case-disjoint confirmation, downstream agreement, and stable interaction."
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
