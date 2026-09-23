#!/usr/bin/env python3
"""Audit and rank the frozen Phase3B restoration discovery screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


CONDITIONS = (
    "sham_copy",
    "restore_noncontiguous",
    "restore_blocks_00_07",
    "restore_blocks_08_15",
    "restore_blocks_16_23",
    "restore_blocks_24_31",
    "restore_output",
)
RESTORATIONS = CONDITIONS[2:]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def interval(values: np.ndarray) -> list[float]:
    return [
        float(value)
        for value in np.quantile(values, [0.025, 0.5, 0.975])
    ]


def fixed_case_metrics(
    rows: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    result = {}
    for case_id, states in grouped.items():
        if len(states) != 4:
            raise RuntimeError(f"{case_id}: expected four fixed states")
        result[case_id] = {
            "direction": float(
                np.mean([row["direction_correct"] for row in states])
            ),
            "calibration_error": float(
                np.mean(
                    [abs(float(row["action_gain"]) - 1.0) for row in states]
                )
            ),
            "exact": float(
                np.mean([row["exact_length"] for row in states])
            ),
        }
    return result


def fixed_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    gains = np.asarray([float(row["action_gain"]) for row in rows])
    return {
        "states": len(rows),
        "cases": len({row["case_id"] for row in rows}),
        "direction_correct_rate": float(
            np.mean([row["direction_correct"] for row in rows])
        ),
        "positive_action_gain_rate": float(np.mean(gains > 0)),
        "median_action_gain": float(np.median(gains)),
        "mean_absolute_calibration_error": float(
            np.mean(np.abs(gains - 1.0))
        ),
        "exact_rate": float(
            np.mean([row["exact_length"] for row in rows])
        ),
        "joint_rate": float(
            np.mean([row["joint_success"] for row in rows])
        ),
    }


def closed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recurrence = []
    terminal_loop = []
    contraction = []
    for row in rows:
        texts = [round_row["text"] for round_row in row["rounds"]]
        recurrence.append(len(texts) != len(set(texts)))
        terminal_loop.append(
            not row["final_joint"]
            and len(texts) > 1
            and texts[-1] == texts[-2]
        )
        if len(row["rounds"]) > 1:
            contraction.append(
                abs(int(row["rounds"][1]["error"]))
                < abs(int(row["rounds"][0]["error"]))
            )
    return {
        "cases": len(rows),
        "one_shot_joint_rate": float(
            np.mean([row["one_shot_joint"] for row in rows])
        ),
        "final_joint_rate": float(
            np.mean([row["final_joint"] for row in rows])
        ),
        "final_exact_rate": float(
            np.mean([row["final_exact"] for row in rows])
        ),
        "first_revision_contraction_rate": (
            float(np.mean(contraction)) if contraction else None
        ),
        "recurrence_rate": float(np.mean(recurrence)),
        "terminal_failure_self_loop_rate": float(
            np.mean(terminal_loop)
        ),
        "median_terminal_absolute_error": float(
            np.median(
                [
                    abs(int(row["rounds"][-1]["error"]))
                    for row in rows
                ]
            )
        ),
    }


def paired_fixed(
    sham: dict[str, dict[str, float]],
    condition: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    case_ids = sorted(set(sham) & set(condition))
    direction = np.asarray(
        [
            sham[case_id]["direction"]
            - condition[case_id]["direction"]
            for case_id in case_ids
        ]
    )
    calibration = np.asarray(
        [
            condition[case_id]["calibration_error"]
            - sham[case_id]["calibration_error"]
            for case_id in case_ids
        ]
    )
    draws = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
    return {
        "cases": len(case_ids),
        "direction_loss_mean": float(np.mean(direction)),
        "direction_loss_cluster_bootstrap": interval(
            np.mean(direction[draws], axis=1)
        ),
        "calibration_degradation_mean": float(
            np.mean(calibration)
        ),
        "calibration_degradation_cluster_bootstrap": interval(
            np.mean(calibration[draws], axis=1)
        ),
    }


def paired_closed(
    sham_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sham = {row["id"]: bool(row["final_joint"]) for row in sham_rows}
    condition = {
        row["id"]: bool(row["final_joint"]) for row in condition_rows
    }
    ids = sorted(set(sham) & set(condition))
    sham_only = sum(sham[x] and not condition[x] for x in ids)
    condition_only = sum(condition[x] and not sham[x] for x in ids)
    discordant = sham_only + condition_only
    return {
        "cases": len(ids),
        "final_joint_loss": float(
            np.mean([sham[x] - condition[x] for x in ids])
        ),
        "sham_only_success": sham_only,
        "condition_only_success": condition_only,
        "mcnemar_exact_two_sided_p": (
            1.0
            if discordant == 0
            else float(
                binomtest(
                    min(sham_only, condition_only),
                    discordant,
                    0.5,
                ).pvalue
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20260730)

    fixed_rows: dict[str, list[dict[str, Any]]] = {}
    closed_rows: dict[str, list[dict[str, Any]]] = {}
    integrity: dict[str, Any] = {"passes": True, "conditions": {}}
    for condition in CONDITIONS:
        condition_root = args.root / condition
        fixed = read_jsonl(
            condition_root / "fixed_state" / "cases.jsonl"
        )
        closed = read_jsonl(
            condition_root / "closed_loop" / "cases.jsonl"
        )
        audit = json.loads(
            (
                condition_root
                / "closed_loop"
                / "integrity_audit.json"
            ).read_text(encoding="utf-8")
        )
        errors = []
        if len(fixed) != 96 or len({row["state_id"] for row in fixed}) != 96:
            errors.append("fixed-state coverage")
        if len(closed) != 120 or len({row["id"] for row in closed}) != 120:
            errors.append("closed-loop coverage")
        if audit.get("verdict") != "PASS":
            errors.append("closed-loop recount audit")
        passed = not errors
        integrity["passes"] = integrity["passes"] and passed
        integrity["conditions"][condition] = {
            "passes": passed,
            "errors": errors,
        }
        fixed_rows[condition] = fixed
        closed_rows[condition] = closed

    sham_fixed = fixed_case_metrics(fixed_rows["sham_copy"])
    sham_closed = closed_rows["sham_copy"]
    results: dict[str, Any] = {}
    for condition in CONDITIONS:
        result = {
            "fixed_state": fixed_summary(fixed_rows[condition]),
            "closed_loop": closed_summary(closed_rows[condition]),
        }
        if condition != "sham_copy":
            result["paired_vs_sham"] = {
                "fixed_state": paired_fixed(
                    sham_fixed,
                    fixed_case_metrics(fixed_rows[condition]),
                    rng,
                ),
                "closed_loop": paired_closed(
                    sham_closed, closed_rows[condition]
                ),
            }
        results[condition] = result

    control = results["restore_noncontiguous"]["paired_vs_sham"]
    eligibility = []
    for condition in RESTORATIONS:
        paired = results[condition]["paired_vs_sham"]
        direction_loss = paired["fixed_state"]["direction_loss_mean"]
        joint_loss = paired["closed_loop"]["final_joint_loss"]
        control_dominates = (
            control["fixed_state"]["direction_loss_mean"]
            >= direction_loss
            and control["closed_loop"]["final_joint_loss"] >= joint_loss
        )
        eligible = (
            direction_loss > 0
            and joint_loss >= 0
            and not control_dominates
        )
        eligibility.append(
            {
                "condition": condition,
                "eligible": eligible,
                "direction_loss": direction_loss,
                "final_joint_loss": joint_loss,
                "noncontiguous_control_dominates": control_dominates,
            }
        )
    selected = [
        row["condition"]
        for row in sorted(
            [row for row in eligibility if row["eligible"]],
            key=lambda row: (
                row["direction_loss"],
                row["final_joint_loss"],
            ),
            reverse=True,
        )[:2]
    ]
    report = {
        "schema_version": 1,
        "protocol": "docs/PHASE3B_RESTORATION_DISCOVERY_PROTOCOL_20260730.md",
        "integrity": integrity,
        "results": results,
        "eligibility": eligibility,
        "selected_at_most_two": selected,
        "claim_boundary": (
            "Discovery selection only; necessity requires planned dose and "
            "confirmation effects exceeding controls."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "restoration_discovery_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "integrity": integrity,
                "eligibility": eligibility,
                "selected_at_most_two": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not integrity["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
