#!/usr/bin/env python3
"""Audit the held-out scale extension and the complete new-domain Combined-480."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.recurrence.structured_analysis as core  # noqa: E402


STAGES = {
    "discovery": {"split": "discovery", "rows": 120, "per_family": 60},
    "confirmation": {"split": "confirmation", "rows": 120, "per_family": 60},
    "scale_extension": {"split": "scale_extension", "rows": 240, "per_family": 120},
}
SEED = 20260806


def validate_stage(stage_dir: Path, stage: str) -> dict[str, list[dict[str, Any]]]:
    spec = STAGES[stage]
    paths = sorted(stage_dir.glob("*/cases.jsonl"))
    models = {path.parent.name for path in paths}
    if models != core.EXPECTED_MODELS:
        raise ValueError(f"{stage_dir}: models={sorted(models)}")
    by_model: dict[str, list[dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    for path in paths:
        rows = core.read_jsonl(path)
        if len(rows) != spec["rows"]:
            raise ValueError(f"{path}: expected {spec['rows']} rows, found {len(rows)}")
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate IDs")
        counts = Counter(str(row["family"]) for row in rows)
        expected_counts = {family: spec["per_family"] for family in core.EXPECTED_FAMILIES}
        if counts != expected_counts:
            raise ValueError(f"{path}: family counts={dict(counts)}")
        for row in rows:
            if str(row["split"]) != spec["split"]:
                raise ValueError(f"{path}: split={row['split']}")
            revisions = [int(item["revision"]) for item in row["rounds"]]
            if revisions != list(range(len(revisions))):
                raise ValueError(f"{path}: non-contiguous revisions for {row['id']}")
            if bool(row["final_joint"]):
                if not bool(row["rounds"][-1]["joint_success"]):
                    raise ValueError(f"{path}: inconsistent final success for {row['id']}")
            elif revisions[-1] != 8:
                raise ValueError(f"{path}: unresolved case stopped early: {row['id']}")
        if reference_ids is None:
            reference_ids = set(ids)
        elif set(ids) != reference_ids:
            raise ValueError(f"{path}: case set differs across checkpoints")
        by_model[path.parent.name] = rows
    return by_model


def final_success_summary(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model, rows in sorted(rows_by_model.items()):
        result[model] = {
            "cases": len(rows),
            "final_joint": sum(bool(row["final_joint"]) for row in rows) / len(rows),
            "by_family": {
                family: sum(
                    bool(row["final_joint"]) for row in rows if row["family"] == family
                )
                / sum(row["family"] == family for row in rows)
                for family in sorted(core.EXPECTED_FAMILIES)
            },
        }
    return result


def analyze_rows(rows: list[dict[str, Any]], bootstraps: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall": core.summarize_group(rows),
        "adjusted_energy_recovery": core.clustered_logistic(rows, "any_energy_recovery"),
        "bootstrap": core.stratified_case_bootstrap(rows, bootstraps, seed),
        "families": {},
        "sources": {},
        "controllers": {},
        "leave_one_controller_out_adjusted": {},
    }
    for index, family in enumerate(sorted(core.EXPECTED_FAMILIES)):
        subset = [row for row in rows if row["family"] == family]
        result["families"][family] = {
            "summary": core.summarize_group(subset),
            "adjusted_energy_recovery": core.clustered_logistic(
                subset, "any_energy_recovery"
            ),
            "bootstrap": core.stratified_case_bootstrap(
                subset, bootstraps, seed + index + 1
            ),
        }
    for model in sorted(core.EXPECTED_MODELS):
        subset = [row for row in rows if row["model"] == model]
        result["controllers"][model] = core.summarize_group(subset)
        retained = [row for row in rows if row["model"] != model]
        result["leave_one_controller_out_adjusted"][model] = core.clustered_logistic(
            retained, "any_energy_recovery"
        )
    for source in sorted({str(row["source"]) for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        result["sources"][source] = {
            "summary": core.summarize_group(subset),
            "adjusted_energy_recovery": core.clustered_logistic(
                subset, "any_energy_recovery"
            ),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=20000)
    args = parser.parse_args()

    dirs = {
        "discovery": args.discovery_dir,
        "confirmation": args.confirmation_dir,
        "scale_extension": args.extension_dir,
    }
    by_stage: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audit: dict[str, Any] = {}
    all_ids_by_model: dict[str, set[str]] = defaultdict(set)
    features: list[dict[str, Any]] = []
    for stage, stage_dir in dirs.items():
        by_model = validate_stage(stage_dir, stage)
        by_stage[stage] = by_model
        audit[stage] = {
            "models": len(by_model),
            "rows": sum(len(rows) for rows in by_model.values()),
        }
        for model, rows in by_model.items():
            ids = {str(row["id"]) for row in rows}
            if ids & all_ids_by_model[model]:
                raise ValueError(f"{model}: case overlap across stages")
            all_ids_by_model[model].update(ids)
            for row in rows:
                feature = core.landmark_row(stage, model, row)
                if feature is not None:
                    features.append(feature)
    if any(len(ids) != 480 for ids in all_ids_by_model.values()):
        raise ValueError("combined suite is not 480 unique cases per controller")
    reference = next(iter(all_ids_by_model.values()))
    if any(ids != reference for ids in all_ids_by_model.values()):
        raise ValueError("combined case sets differ across controllers")

    combined_rows_by_model = {
        model: [row for stage in by_stage.values() for row in stage[model]]
        for model in sorted(core.EXPECTED_MODELS)
    }
    extension_features = [row for row in features if row["stage"] == "scale_extension"]
    results = {
        "audit": audit,
        "combined_unique_cases_per_controller": 480,
        "combined_controller_case_trajectories": 2880,
        "landmark": core.LANDMARK,
        "extension": analyze_rows(extension_features, args.bootstraps, SEED),
        "combined480": analyze_rows(features, args.bootstraps, SEED + 100),
        "extension_final_success": final_success_summary(by_stage["scale_extension"]),
        "combined480_final_success": final_success_summary(combined_rows_by_model),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "landmark4_rows.csv", features)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
