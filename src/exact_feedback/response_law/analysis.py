#!/usr/bin/env python3
"""Freeze discovery response curves and evaluate case-disjoint confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


MODELS = (
    "llama31_8b_base",
    "llama31_8b_instruct",
    "qwen3_8b_base",
    "qwen3_8b",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_14b",
    "gemma2_9b_it",
)
PAIRS = {
    "llama_posttraining": ("llama31_8b_instruct", "llama31_8b_base"),
    "qwen_posttraining": ("qwen3_8b", "qwen3_8b_base"),
}
METRICS = (
    "direction_correct",
    "contraction",
    "zero_action",
    "overshoot",
)
COMMANDS = (-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50)
SEED = 20260731
BOOTSTRAPS = 10000
GATE = {
    "minimum_models_with_curve_rho_at_least_0_70": 6,
    "llama_direction_gain_ci_lower_must_exceed": 0.0,
    "qwen_zero_action_change_ci_lower_must_exceed": 0.0,
    "qwen_overshoot_change_ci_upper_must_be_below": 0.0,
    "checkpoint_curve_minus_pooled_mae_ci_upper_must_be_below": 0.0,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def stratified_bootstrap(
    case_values: dict[str, tuple[str, float]],
    statistic: Callable[[list[float]], float] = statistics.mean,
) -> dict[str, Any]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for source, value in case_values.values():
        by_source[source].append(float(value))
    observed = statistic([value for values in by_source.values() for value in values])
    rng = random.Random(SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAPS):
        draw: list[float] = []
        for source in sorted(by_source):
            values = by_source[source]
            draw.extend(rng.choice(values) for _ in values)
        samples.append(statistic(draw))
    return {
        "estimate": float(observed),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "case_clusters": len(case_values),
        "bootstrap_replicates": BOOTSTRAPS,
        "stratified_by": "source",
    }


def grouped_case_values(
    rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]
) -> dict[str, tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, str] = {}
    for row in rows:
        grouped[row["case_id"]].append(float(value(row)))
        sources[row["case_id"]] = row["source"]
    return {
        case_id: (sources[case_id], statistics.mean(values))
        for case_id, values in grouped.items()
    }


def paired_case_values(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
) -> dict[str, tuple[str, float]]:
    left_by_state = {row["state_id"]: row for row in left}
    right_by_state = {row["state_id"]: row for row in right}
    if set(left_by_state) != set(right_by_state):
        raise ValueError(f"paired state mismatch for {metric}")
    grouped: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, str] = {}
    for state_id in sorted(left_by_state):
        lrow = left_by_state[state_id]
        rrow = right_by_state[state_id]
        grouped[lrow["case_id"]].append(
            float(lrow[metric]) - float(rrow[metric])
        )
        sources[lrow["case_id"]] = lrow["source"]
    return {
        case_id: (sources[case_id], statistics.mean(values))
        for case_id, values in grouped.items()
    }


def case_slopes(
    rows: list[dict[str, Any]], sign: int | None = None
) -> dict[str, tuple[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        command = int(row["required_delta"])
        if abs(command) > 10:
            continue
        if sign is not None and command * sign <= 0:
            continue
        grouped[row["case_id"]].append(row)
    answer: dict[str, tuple[str, float]] = {}
    for case_id, items in grouped.items():
        slopes: list[float] = []
        ordered = sorted(items, key=lambda row: int(row["required_delta"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                dx = int(right["required_delta"]) - int(left["required_delta"])
                if dx:
                    slopes.append(
                        (float(right["realized_delta"]) - float(left["realized_delta"]))
                        / dx
                    )
        if slopes:
            answer[case_id] = (
                ordered[0]["source"],
                float(statistics.median(slopes)),
            )
    return answer


def curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_command: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_command[int(row["required_delta"])].append(row)
    points: dict[str, Any] = {}
    medians: list[float] = []
    present_commands: list[int] = []
    for command in COMMANDS:
        items = by_command.get(command, [])
        if not items:
            continue
        actions = [float(row["realized_delta"]) for row in items]
        median_action = float(statistics.median(actions))
        medians.append(median_action)
        present_commands.append(command)
        points[str(command)] = {
            "n": len(items),
            "mean_action": float(statistics.mean(actions)),
            "median_action": median_action,
            "direction_correct_rate": statistics.mean(
                float(row["direction_correct"]) for row in items
            ),
            "contraction_rate": statistics.mean(
                float(row["contraction"]) for row in items
            ),
            "zero_action_rate": statistics.mean(
                float(row["zero_action"]) for row in items
            ),
            "overshoot_rate": statistics.mean(
                float(row["overshoot"]) for row in items
            ),
            "anchor_retention_rate": statistics.mean(
                float(not row["missing_anchors"]) for row in items
            ),
        }
    concordant = 0
    comparable = 0
    for index, left in enumerate(medians):
        for right in medians[index + 1 :]:
            comparable += 1
            concordant += int(right >= left)
    local = stratified_bootstrap(
        case_slopes(rows), statistic=statistics.median
    )
    add = stratified_bootstrap(
        case_slopes(rows, sign=1), statistic=statistics.median
    )
    remove = stratified_bootstrap(
        case_slopes(rows, sign=-1), statistic=statistics.median
    )
    return {
        "points": points,
        "commands": present_commands,
        "median_actions": medians,
        "monotone_pair_fraction": concordant / comparable,
        "local_theil_sen_slope": local,
        "add_local_theil_sen_slope": add,
        "remove_local_theil_sen_slope": remove,
        "add_remove_slope_difference": add["estimate"] - remove["estimate"],
    }


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2 + 1
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_rank = rank(left)
    right_rank = rank(right)
    left_mean = statistics.mean(left_rank)
    right_mean = statistics.mean(right_rank)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left_rank, right_rank)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left_rank))
    right_norm = math.sqrt(
        sum((value - right_mean) ** 2 for value in right_rank)
    )
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def validate_and_load(
    experiment_dir: Path, split: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[str, Any] = {}
    state_sets: list[set[str]] = []
    for model in MODELS:
        model_dir = experiment_dir / model
        audit = read_json(model_dir / "integrity_audit.json")
        if audit["verdict"] != "PASS" or audit["split"] != split:
            raise ValueError(f"invalid audit for {model}: {audit}")
        model_rows = read_jsonl(model_dir / "cases.jsonl")
        if len(model_rows) != int(audit["result_rows"]):
            raise ValueError(f"row count mismatch for {model}")
        if any(row["split"] != split for row in model_rows):
            raise ValueError(f"split contamination for {model}")
        rows[model] = model_rows
        state_sets.append({row["state_id"] for row in model_rows})
        evidence[model] = {
            "cases_sha256": sha256(model_dir / "cases.jsonl"),
            "runtime_manifest_sha256": sha256(
                model_dir / "runtime_manifest.json"
            ),
            "audit_sha256": sha256(model_dir / "integrity_audit.json"),
            "rows": len(model_rows),
        }
    if any(states != state_sets[0] for states in state_sets[1:]):
        raise ValueError("models do not share the same frozen state set")
    return rows, evidence


def model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"states": len(rows)}
    for metric in METRICS:
        report[metric] = stratified_bootstrap(
            grouped_case_values(rows, lambda row, key=metric: float(row[key]))
        )
    report["anchor_retention"] = stratified_bootstrap(
        grouped_case_values(rows, lambda row: float(not row["missing_anchors"]))
    )
    report["terminal_absolute_error_median"] = float(
        statistics.median(abs(float(row["output_error"])) for row in rows)
    )
    report["action_gain_median"] = float(
        statistics.median(float(row["action_gain"]) for row in rows)
    )
    report["curve"] = curve(rows)
    return report


def pair_summary(
    rows_by_model: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, (left, right) in PAIRS.items():
        report[name] = {
            "left": left,
            "right": right,
            "left_minus_right": {
                metric: stratified_bootstrap(
                    paired_case_values(
                        rows_by_model[left], rows_by_model[right], metric
                    )
                )
                for metric in METRICS
            },
        }
    return report


def prompt_identity(
    rows_by_model: dict[str, list[dict[str, Any]]]
) -> dict[str, bool]:
    answer = {}
    for name, (left, right) in PAIRS.items():
        left_hashes = {
            row["state_id"]: row["rendered_sha256"]
            for row in rows_by_model[left]
        }
        right_hashes = {
            row["state_id"]: row["rendered_sha256"]
            for row in rows_by_model[right]
        }
        answer[name] = left_hashes == right_hashes
    return answer


def discovery(
    experiment_dir: Path, output_dir: Path, script_path: Path
) -> None:
    rows_by_model, evidence = validate_and_load(experiment_dir, "discovery")
    summaries = {
        model: model_summary(rows_by_model[model]) for model in MODELS
    }
    pair_results = pair_summary(rows_by_model)
    identity = prompt_identity(rows_by_model)
    if not all(identity.values()):
        raise ValueError(f"aligned-pair prompt mismatch: {identity}")

    model_predictors = {
        model: {
            command: summaries[model]["curve"]["points"][command][
                "median_action"
            ]
            for command in summaries[model]["curve"]["points"]
        }
        for model in MODELS
    }
    pooled_predictors: dict[str, float] = {}
    for command in COMMANDS:
        values = [
            float(row["realized_delta"])
            for model in MODELS
            for row in rows_by_model[model]
            if int(row["required_delta"]) == command
        ]
        if values:
            pooled_predictors[str(command)] = float(statistics.median(values))

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "schema_version": 1,
        "phase": "discovery",
        "models": list(MODELS),
        "commands": list(COMMANDS),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "evidence": evidence,
        "aligned_pair_prompt_identity": identity,
        "model_summaries": summaries,
        "paired_posttraining": pair_results,
        "interpretation_status": "discovery_only_no_confirmation_claim",
    }
    analysis_path = output_dir / "discovery_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction = {
        "schema_version": 1,
        "frozen_for": "case_disjoint_confirmation",
        "models": list(MODELS),
        "commands": list(COMMANDS),
        "model_median_action_by_command": model_predictors,
        "pooled_median_action_by_command": pooled_predictors,
        "discovery_evidence": evidence,
        "discovery_analysis_sha256": sha256(analysis_path),
        "analysis_script_sha256": sha256(script_path),
        "confirmation_gate": GATE,
    }
    prediction_path = output_dir / "frozen_prediction_spec.json"
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis": str(analysis_path),
                "analysis_sha256": sha256(analysis_path),
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "prompt_identity": identity,
            },
            indent=2,
        )
    )


def prediction_errors(
    rows: list[dict[str, Any]],
    checkpoint_curve: dict[str, float],
    pooled_curve: dict[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, tuple[str, float]]]]:
    losses: dict[str, list[float]] = defaultdict(list)
    by_case: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sources: dict[str, str] = {}
    for row in rows:
        command = str(int(row["required_delta"]))
        action = float(row["realized_delta"])
        candidates = {
            "checkpoint_curve": float(checkpoint_curve[command]),
            "pooled_curve": float(pooled_curve[command]),
            "zero_action": 0.0,
            "ideal_identity": float(command),
        }
        for name, prediction in candidates.items():
            loss = abs(action - prediction)
            losses[name].append(loss)
            by_case[row["case_id"]][name].append(loss)
        sources[row["case_id"]] = row["source"]
    means = {
        name: float(statistics.mean(values)) for name, values in losses.items()
    }
    clustered = {
        name: {
            case_id: (sources[case_id], statistics.mean(case_losses[name]))
            for case_id, case_losses in by_case.items()
        }
        for name in losses
    }
    return means, clustered


def confirmation(
    experiment_dir: Path,
    output_dir: Path,
    prediction_path: Path,
    script_path: Path,
) -> None:
    prediction = read_json(prediction_path)
    if prediction["analysis_script_sha256"] != sha256(script_path):
        raise ValueError("analysis script changed after prediction freeze")
    if prediction["confirmation_gate"] != GATE:
        raise ValueError("confirmation gate changed after prediction freeze")
    rows_by_model, evidence = validate_and_load(
        experiment_dir, "confirmation"
    )
    summaries = {
        model: model_summary(rows_by_model[model]) for model in MODELS
    }
    pair_results = pair_summary(rows_by_model)
    identity = prompt_identity(rows_by_model)
    if not all(identity.values()):
        raise ValueError(f"aligned-pair prompt mismatch: {identity}")

    predictions: dict[str, Any] = {}
    aggregate_case_differences: dict[str, list[float]] = defaultdict(list)
    aggregate_sources: dict[str, str] = {}
    reproducible = 0
    for model in MODELS:
        discovery_actions = [
            float(
                prediction["model_median_action_by_command"][model][str(command)]
            )
            for command in COMMANDS
            if str(command)
            in prediction["model_median_action_by_command"][model]
            and str(command) in summaries[model]["curve"]["points"]
        ]
        confirmation_actions = [
            float(summaries[model]["curve"]["points"][str(command)]["median_action"])
            for command in COMMANDS
            if str(command)
            in prediction["model_median_action_by_command"][model]
            and str(command) in summaries[model]["curve"]["points"]
        ]
        rho = correlation(discovery_actions, confirmation_actions)
        reproducible += int(rho >= 0.70)
        means, clustered = prediction_errors(
            rows_by_model[model],
            prediction["model_median_action_by_command"][model],
            prediction["pooled_median_action_by_command"],
        )
        differences = {
            case_id: (
                source,
                checkpoint_loss
                - dict(clustered["pooled_curve"])[case_id][1],
            )
            for case_id, (source, checkpoint_loss) in clustered[
                "checkpoint_curve"
            ].items()
        }
        for case_id, (source, value) in differences.items():
            aggregate_case_differences[case_id].append(value)
            aggregate_sources[case_id] = source
        predictions[model] = {
            "discovery_confirmation_curve_spearman": rho,
            "mean_absolute_action_prediction_error": means,
            "checkpoint_minus_pooled_curve_mae": stratified_bootstrap(
                differences
            ),
        }

    aggregate = {
        case_id: (
            aggregate_sources[case_id],
            statistics.mean(values),
        )
        for case_id, values in aggregate_case_differences.items()
    }
    aggregate_interval = stratified_bootstrap(aggregate)
    llama_direction = pair_results["llama_posttraining"][
        "left_minus_right"
    ]["direction_correct"]
    qwen_metrics = pair_results["qwen_posttraining"]["left_minus_right"]
    checks = {
        "curve_reproducibility": {
            "pass": reproducible
            >= GATE["minimum_models_with_curve_rho_at_least_0_70"],
            "models_passing": reproducible,
            "required": GATE[
                "minimum_models_with_curve_rho_at_least_0_70"
            ],
        },
        "llama_posttraining_direction": {
            "pass": llama_direction["ci95"][0]
            > GATE["llama_direction_gain_ci_lower_must_exceed"],
            "interval": llama_direction,
        },
        "qwen_posttraining_signature": {
            "pass": qwen_metrics["zero_action"]["ci95"][0]
            > GATE["qwen_zero_action_change_ci_lower_must_exceed"]
            and qwen_metrics["overshoot"]["ci95"][1]
            < GATE["qwen_overshoot_change_ci_upper_must_be_below"],
            "expected_from_discovery": {
                "zero_action": "instruct_minus_base_positive",
                "overshoot": "instruct_minus_base_negative",
            },
            "zero_action_interval": qwen_metrics["zero_action"],
            "overshoot_interval": qwen_metrics["overshoot"],
        },
        "checkpoint_specific_prediction": {
            "pass": aggregate_interval["ci95"][1]
            < GATE[
                "checkpoint_curve_minus_pooled_mae_ci_upper_must_be_below"
            ],
            "checkpoint_minus_pooled_curve_mae": aggregate_interval,
        },
    }
    gate_pass = all(check["pass"] for check in checks.values())
    report = {
        "schema_version": 1,
        "phase": "case_disjoint_confirmation",
        "models": list(MODELS),
        "commands": list(COMMANDS),
        "prediction_spec_sha256": sha256(prediction_path),
        "analysis_script_sha256": sha256(script_path),
        "evidence": evidence,
        "aligned_pair_prompt_identity": identity,
        "model_summaries": summaries,
        "paired_posttraining": pair_results,
        "predictive_validation": predictions,
        "aggregate_checkpoint_minus_pooled_curve_mae": aggregate_interval,
        "gate": {
            "verdict": "PASS" if gate_pass else "FAIL",
            "checks": checks,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confirmation_analysis.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": sha256(output_path),
                "gate": report["gate"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("discovery", "confirmation"), required=True
    )
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-spec", type=Path)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    if args.phase == "discovery":
        if args.prediction_spec is not None:
            raise ValueError("--prediction-spec is not used for discovery")
        discovery(args.experiment_dir, args.output_dir, script_path)
    else:
        if args.prediction_spec is None:
            raise ValueError("--prediction-spec is required for confirmation")
        confirmation(
            args.experiment_dir,
            args.output_dir,
            args.prediction_spec,
            script_path,
        )


if __name__ == "__main__":
    main()
