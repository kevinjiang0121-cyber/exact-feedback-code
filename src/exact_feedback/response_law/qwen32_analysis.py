#!/usr/bin/env python3
"""Evaluate Qwen3-32B under the frozen feedback-policy estimator.

This is a scale extension, not a rewrite of the original eight-checkpoint
confirmation gate.  The checkpoint curve is fit on the original discovery
split and evaluated on the case-disjoint confirmation split; the pooled
baseline remains the one frozen before the original confirmation analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exact_feedback.response_law.analysis import (
    COMMANDS,
    correlation,
    curve,
    model_summary,
    prediction_errors,
    read_json,
    read_jsonl,
    sha256,
    stratified_bootstrap,
)


def require_pass(path: Path) -> dict[str, Any]:
    report = read_json(path)
    if report.get("verdict") != "PASS":
        raise ValueError(f"audit did not pass: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--prediction-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    discovery_dir = args.experiment_dir / "discovery" / "qwen3_32b"
    confirmation_dir = args.experiment_dir / "confirmation" / "qwen3_32b"
    discovery_audit = require_pass(discovery_dir / "local_integrity_audit.json")
    confirmation_audit = require_pass(
        confirmation_dir / "local_integrity_audit.json"
    )
    if discovery_audit["states_sha256"] != confirmation_audit["states_sha256"]:
        raise ValueError("discovery and confirmation state hashes differ")

    discovery_rows = read_jsonl(discovery_dir / "cases.jsonl")
    confirmation_rows = read_jsonl(confirmation_dir / "cases.jsonl")
    if {row["model"] for row in discovery_rows + confirmation_rows} != {
        "qwen3_32b"
    }:
        raise ValueError("unexpected model slug")

    prediction = read_json(args.prediction_spec)
    discovery_curve = curve(discovery_rows)
    confirmation_curve = curve(confirmation_rows)
    checkpoint_predictor = {
        command: point["median_action"]
        for command, point in discovery_curve["points"].items()
    }
    pooled_predictor = prediction["pooled_median_action_by_command"]

    commands = [
        str(command)
        for command in COMMANDS
        if str(command) in checkpoint_predictor
        and str(command) in confirmation_curve["points"]
    ]
    rho = correlation(
        [float(checkpoint_predictor[command]) for command in commands],
        [
            float(confirmation_curve["points"][command]["median_action"])
            for command in commands
        ],
    )
    means, clustered = prediction_errors(
        confirmation_rows, checkpoint_predictor, pooled_predictor
    )
    differences = {
        case_id: (
            source,
            checkpoint_loss - dict(clustered["pooled_curve"])[case_id][1],
        )
        for case_id, (source, checkpoint_loss) in clustered[
            "checkpoint_curve"
        ].items()
    }

    report = {
        "schema_version": 1,
        "status": "post_hoc_scale_extension_under_frozen_estimator",
        "model": "qwen3_32b",
        "states_sha256": discovery_audit["states_sha256"],
        "discovery_rows": len(discovery_rows),
        "confirmation_rows": len(confirmation_rows),
        "discovery_cases_sha256": discovery_audit["cases_sha256"],
        "confirmation_cases_sha256": confirmation_audit["cases_sha256"],
        "original_prediction_spec_sha256": sha256(args.prediction_spec),
        "commands": [int(command) for command in commands],
        "discovery_curve": discovery_curve,
        "confirmation_curve": confirmation_curve,
        "confirmation_summary": model_summary(confirmation_rows),
        "predictive_validation": {
            "discovery_confirmation_curve_spearman": rho,
            "mean_absolute_action_prediction_error": means,
            "checkpoint_minus_original_pooled_curve_mae": stratified_bootstrap(
                differences
            ),
        },
        "interpretation_boundary": (
            "The estimator and case split are inherited from the frozen system-ID "
            "protocol, but Qwen3-32B was authorized as a later scale extension and "
            "does not alter the original eight-checkpoint confirmation gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["predictive_validation"], indent=2))
    print(f"output={args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
