#!/usr/bin/env python3
"""Prospective linked analysis for the ICLR strong-findings spotlight gate.

The predictor is frozen in ``frozen_prediction_spec.json``.  This script never
fits a response curve on confirmation or closed-loop outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SEED = 20260731
BOOTSTRAPS = 10000
MODELS = (
    "llama31_8b_base",
    "llama31_8b_instruct",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "gemma2_9b_it",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_family(source: str) -> str:
    for prefix in ("dolly", "no_robots", "oasst1", "writingprompts"):
        if source.startswith(prefix):
            return prefix
    return source


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def interval(samples: list[float]) -> list[float]:
    return [quantile(samples, 0.025), quantile(samples, 0.975)]


def stratified_case_bootstrap(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_family"])].append(row)
    point = metric(rows)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(bootstraps):
        draw: list[dict[str, Any]] = []
        for source in sorted(by_source):
            source_rows = by_source[source]
            draw.extend(rng.choice(source_rows) for _ in source_rows)
        samples.append(metric(draw))
    return {
        "estimate": point,
        "ci95": interval(samples),
        "case_clusters": len(rows),
        "bootstrap_replicates": bootstraps,
        "stratified_by": "source_family",
    }


def paired_case_id_bootstrap(
    rows: list[dict[str, Any]],
    models: tuple[str, ...],
    metric: Callable[[dict[str, list[dict[str, Any]]]], float],
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    """Resample shared case IDs, then give every checkpoint equal weight."""
    by_case_model = {
        (str(row["case_id"]), str(row["model"])): row for row in rows
    }
    source_by_case = {
        str(row["case_id"]): str(row["source_family"]) for row in rows
    }
    ids_by_source: dict[str, list[str]] = defaultdict(list)
    for case_id, source in source_by_case.items():
        ids_by_source[source].append(case_id)

    def materialize(case_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case_id in case_ids:
            for model in models:
                row = by_case_model.get((case_id, model))
                if row is not None:
                    grouped[model].append(row)
        return grouped

    all_ids = sorted(source_by_case)
    point = metric(materialize(all_ids))
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(bootstraps):
        draw: list[str] = []
        for source in sorted(ids_by_source):
            source_ids = ids_by_source[source]
            draw.extend(rng.choice(source_ids) for _ in source_ids)
        samples.append(metric(materialize(draw)))
    return {
        "estimate": point,
        "ci95": interval(samples),
        "case_clusters": len(all_ids),
        "checkpoint_weighting": "equal",
        "bootstrap_replicates": bootstraps,
        "stratified_by": "source_family",
    }


def make_curve(points: dict[str, float]) -> Callable[[float], float]:
    xy = sorted(
        {(float(command), float(action)) for command, action in points.items()}
        | {(0.0, 0.0)}
    )

    def predict(command: float) -> float:
        if command <= xy[0][0]:
            return xy[0][1]
        if command >= xy[-1][0]:
            return xy[-1][1]
        for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
            if x0 <= command <= x1:
                if x1 == x0:
                    return y0
                weight = (command - x0) / (x1 - x0)
                return y0 + weight * (y1 - y0)
        raise AssertionError(f"command outside interpolated curve: {command}")

    return predict


def action_flags(error: float, action: float) -> dict[str, float]:
    command = -error
    next_error = error + action
    correct_direction = action * command > 0
    return {
        "zero_action": float(abs(action) < 1e-12),
        "direction_correct": float(correct_direction),
        "contraction": float(abs(next_error) < abs(error)),
        "overshoot": float(
            correct_direction
            and abs(next_error) > 1e-12
            and math.copysign(1.0, next_error) != math.copysign(1.0, error)
        ),
    }


def macro_category_accuracy(
    actual: dict[str, float], predicted: dict[str, float]
) -> float:
    return statistics.mean(
        float(actual[key] == predicted[key])
        for key in ("zero_action", "direction_correct", "contraction", "overshoot")
    )


def trajectory_outcome(row: dict[str, Any]) -> dict[str, float]:
    rounds = row["rounds"]
    errors = [int(item["error"]) for item in rounds]
    signs = [1 if error > 0 else -1 for error in errors if error != 0]
    flips = sum(left != right for left, right in zip(signs, signs[1:]))
    stationary = bool(
        not row["final_joint"]
        and len(rounds) > 1
        and rounds[-1]["text"] == rounds[-2]["text"]
    )
    return {
        "terminal_joint": float(bool(row["final_joint"])),
        "eventual_exact": float(bool(row["ever_exact"])),
        "oscillation": float(flips >= 2),
        "stationary_failure": float(stationary),
    }


def simulate(initial_error: int, curve: Callable[[float], float], steps: int = 8) -> dict[str, float]:
    error = float(initial_error)
    errors = [error]
    actions: list[float] = []
    captured = abs(error) < 0.5
    for _ in range(steps):
        if captured:
            actions.append(0.0)
            errors.append(error)
            continue
        command = max(-50.0, min(50.0, -error))
        action = curve(command)
        actions.append(action)
        error += action
        if abs(error) < 0.5:
            error = 0.0
            captured = True
        errors.append(error)
    signs = [1 if value > 0 else -1 for value in errors if abs(value) >= 0.5]
    flips = sum(left != right for left, right in zip(signs, signs[1:]))
    stationary = not captured and abs(actions[-1]) < 0.5
    return {
        "terminal_joint": float(captured),
        "eventual_exact": float(captured),
        "oscillation": float(flips >= 2),
        "stationary_failure": float(stationary),
    }


def average(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def exact_spearman_p(left: list[float], right: list[float]) -> float:
    observed = abs(spearman(left, right))
    total = 0
    extreme = 0
    for permutation in itertools.permutations(right):
        total += 1
        if abs(spearman(left, list(permutation))) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_registry(root: Path) -> dict[str, list[Path]]:
    exp = root / "experiments"
    return {
        "llama31_8b_base": [
            exp / "llama_training_stage_origin_v1/llama31_8b_base/vllm_common_plain_combined480/cases.jsonl"
        ],
        "llama31_8b_instruct": [
            exp / "llama_training_stage_origin_v1/llama31_8b_instruct/vllm_common_plain_combined480/cases.jsonl"
        ],
        "qwen3_1_7b": [
            exp / "model_generality_combined480_v1/qwen3_1_7b/combined480/cases.jsonl"
        ],
        "qwen3_4b": [
            exp / "model_generality_combined480_v1/qwen3_4b/combined480/cases.jsonl"
        ],
        "qwen3_8b": [
            exp / "human_generation_main120_v1/qwen3_8b/cases.jsonl",
            exp / "human_generation_replication120_v1/qwen3_8b/cases.jsonl",
            exp / "human_generation_extension240_v1/qwen3_8b/cases.jsonl",
        ],
        "qwen3_14b": [
            exp / "human_generation_main120_v1/qwen3_14b/cases.jsonl",
            exp / "human_generation_replication120_v1/qwen3_14b/cases.jsonl",
            exp / "human_generation_extension240_v1/qwen3_14b/cases.jsonl",
        ],
        "gemma2_9b_it": [
            exp / "human_generation_main120_v1/gemma2_9b/cases.jsonl",
            exp / "human_generation_replication120_v1/gemma2_9b/cases.jsonl",
            exp / "human_generation_extension240_v1/gemma2_9b/cases.jsonl",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prediction-spec", type=Path)
    parser.add_argument("--discovery-analysis", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/policy_to_closed_loop_spotlight_gate_v1"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = args.prediction_spec or root / "experiments/feedback_policy_system_identification_v1/analysis/frozen_prediction_spec.json"
    states_path = root / "data/feedback_policy_response_surface_v1/states.jsonl"
    discovery_path = args.discovery_analysis or root / "experiments/feedback_policy_system_identification_v1/analysis/discovery_analysis.json"
    spec = read_json(spec_path)
    _discovery = read_json(discovery_path)
    excluded_ids = {str(row["case_id"]) for row in read_jsonl(states_path)}
    if len(excluded_ids) != 72:
        raise ValueError(f"expected 72 system-identification case IDs, got {len(excluded_ids)}")

    curves = {
        model: make_curve(spec["model_median_action_by_command"][model])
        for model in MODELS
    }
    pooled_curve = make_curve(spec["pooled_median_action_by_command"])
    registry = input_registry(root)
    evidence: dict[str, Any] = {
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "frozen_prediction_spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
        "discovery_analysis": {"path": str(discovery_path), "sha256": sha256(discovery_path)},
        "system_id_states": {"path": str(states_path), "sha256": sha256(states_path)},
        "closed_loop": {},
    }

    per_model_cases: dict[str, list[dict[str, Any]]] = {}
    transition_case_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for model in MODELS:
        all_rows: list[dict[str, Any]] = []
        model_evidence: list[dict[str, Any]] = []
        for path in registry[model]:
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            all_rows.extend(rows)
            model_evidence.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})
        ids = [str(row["id"]) for row in all_rows]
        if len(ids) != 480 or len(set(ids)) != 480:
            raise ValueError(f"{model}: expected 480 unique rows, got {len(ids)} rows/{len(set(ids))} IDs")
        retained = [row for row in all_rows if str(row["id"]) not in excluded_ids]
        if len(retained) != 408:
            raise ValueError(f"{model}: expected 408 retained cases, got {len(retained)}")
        per_model_cases[model] = retained
        evidence["closed_loop"][model] = model_evidence

        for row in retained:
            case_id = str(row["id"])
            source = source_family(str(row["source"]))
            transitions: list[dict[str, float]] = []
            first_transition: dict[str, float] | None = None
            for left, right in zip(row["rounds"], row["rounds"][1:]):
                error = float(left["error"])
                if error == 0 or abs(error) > 50:
                    continue
                actual_action = float(right["error"]) - error
                command = -error
                model_action = curves[model](command)
                pooled_action = pooled_curve(command)
                actual_flags = action_flags(error, actual_action)
                model_flags = action_flags(error, model_action)
                pooled_flags = action_flags(error, pooled_action)
                item = {
                    "model_abs_error": abs(actual_action - model_action),
                    "pooled_abs_error": abs(actual_action - pooled_action),
                    "zero_abs_error": abs(actual_action),
                    "ideal_abs_error": abs(actual_action - command),
                    "model_category_accuracy": macro_category_accuracy(actual_flags, model_flags),
                    "pooled_category_accuracy": macro_category_accuracy(actual_flags, pooled_flags),
                }
                transitions.append(item)
                if int(left["revision"]) == 0:
                    first_transition = item
            if transitions:
                case_summary = {
                    "model": model,
                    "case_id": case_id,
                    "source_family": source,
                    "transitions": len(transitions),
                }
                for key in transitions[0]:
                    case_summary[key] = statistics.mean(item[key] for item in transitions)
                case_summary["mae_delta_model_minus_pooled"] = (
                    case_summary["model_abs_error"] - case_summary["pooled_abs_error"]
                )
                case_summary["category_accuracy_delta"] = (
                    case_summary["model_category_accuracy"]
                    - case_summary["pooled_category_accuracy"]
                )
                transition_case_rows.append(case_summary)
            if first_transition is not None:
                transition_case_rows.append(
                    {
                        "model": model,
                        "case_id": case_id,
                        "source_family": source,
                        "transitions": 1,
                        **{f"first_{key}": value for key, value in first_transition.items()},
                        "row_type": "first_transition",
                    }
                )

            observed = trajectory_outcome(row)
            simulated_model = simulate(int(row["rounds"][0]["error"]), curves[model])
            simulated_pooled = simulate(int(row["rounds"][0]["error"]), pooled_curve)
            regime_rows.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "source_family": source,
                    **{f"observed_{key}": value for key, value in observed.items()},
                    **{f"model_simulated_{key}": value for key, value in simulated_model.items()},
                    **{f"pooled_simulated_{key}": value for key, value in simulated_pooled.items()},
                }
            )

    all_transition_rows = [row for row in transition_case_rows if row.get("row_type") != "first_transition"]
    first_transition_rows = [row for row in transition_case_rows if row.get("row_type") == "first_transition"]
    first_common_rows = [
        {
            "model": row["model"],
            "case_id": row["case_id"],
            "source_family": row["source_family"],
            "model_abs_error": row["first_model_abs_error"],
            "pooled_abs_error": row["first_pooled_abs_error"],
            "zero_abs_error": row["first_zero_abs_error"],
            "ideal_abs_error": row["first_ideal_abs_error"],
            "category_accuracy_delta": row["first_model_category_accuracy"]
            - row["first_pooled_category_accuracy"],
        }
        for row in first_transition_rows
    ]
    per_model_transition: list[dict[str, Any]] = []
    model_intervals: dict[str, Any] = {}
    for model in MODELS:
        rows = [row for row in all_transition_rows if row["model"] == model]
        delta = stratified_case_bootstrap(
            rows, lambda draw: average(draw, "mae_delta_model_minus_pooled"), args.bootstraps, args.seed
        )
        relative = stratified_case_bootstrap(
            rows,
            lambda draw: (
                average(draw, "pooled_abs_error") - average(draw, "model_abs_error")
            ) / average(draw, "pooled_abs_error"),
            args.bootstraps,
            args.seed + 1,
        )
        category = stratified_case_bootstrap(
            rows, lambda draw: average(draw, "category_accuracy_delta"), args.bootstraps, args.seed + 2
        )
        model_intervals[model] = {
            "mae_delta_model_minus_pooled": delta,
            "relative_mae_reduction": relative,
            "category_accuracy_delta": category,
        }
        per_model_transition.append(
            {
                "model": model,
                "cases": len(rows),
                "transitions": sum(int(row["transitions"]) for row in rows),
                "model_mae": average(rows, "model_abs_error"),
                "pooled_mae": average(rows, "pooled_abs_error"),
                "zero_mae": average(rows, "zero_abs_error"),
                "ideal_mae": average(rows, "ideal_abs_error"),
                "mae_delta": delta["estimate"],
                "mae_delta_ci_lo": delta["ci95"][0],
                "mae_delta_ci_hi": delta["ci95"][1],
                "relative_reduction": relative["estimate"],
                "relative_reduction_ci_lo": relative["ci95"][0],
                "relative_reduction_ci_hi": relative["ci95"][1],
                "category_accuracy_delta": category["estimate"],
            }
        )

    def checkpoint_means(grouped: dict[str, list[dict[str, Any]]], key: str) -> float:
        return statistics.mean(average(grouped[model], key) for model in MODELS)

    aggregate = {
        "mae_delta_model_minus_pooled": paired_case_id_bootstrap(
            all_transition_rows,
            MODELS,
            lambda grouped: checkpoint_means(grouped, "model_abs_error")
            - checkpoint_means(grouped, "pooled_abs_error"),
            args.bootstraps,
            args.seed + 10,
        ),
        "relative_mae_reduction": paired_case_id_bootstrap(
            all_transition_rows,
            MODELS,
            lambda grouped: (
                checkpoint_means(grouped, "pooled_abs_error")
                - checkpoint_means(grouped, "model_abs_error")
            )
            / checkpoint_means(grouped, "pooled_abs_error"),
            args.bootstraps,
            args.seed + 11,
        ),
        "category_accuracy_delta": paired_case_id_bootstrap(
            all_transition_rows,
            MODELS,
            lambda grouped: checkpoint_means(grouped, "category_accuracy_delta"),
            args.bootstraps,
            args.seed + 12,
        ),
    }

    first_step_aggregate = {
        "mae_delta_model_minus_pooled": paired_case_id_bootstrap(
            first_common_rows,
            MODELS,
            lambda grouped: checkpoint_means(grouped, "model_abs_error")
            - checkpoint_means(grouped, "pooled_abs_error"),
            args.bootstraps,
            args.seed + 13,
        ),
        "relative_mae_reduction": paired_case_id_bootstrap(
            first_common_rows,
            MODELS,
            lambda grouped: (
                checkpoint_means(grouped, "pooled_abs_error")
                - checkpoint_means(grouped, "model_abs_error")
            )
            / checkpoint_means(grouped, "pooled_abs_error"),
            args.bootstraps,
            args.seed + 14,
        ),
        "category_accuracy_delta": paired_case_id_bootstrap(
            first_common_rows,
            MODELS,
            lambda grouped: checkpoint_means(grouped, "category_accuracy_delta"),
            args.bootstraps,
            args.seed + 15,
        ),
    }

    per_model_regime: list[dict[str, Any]] = []
    for model in MODELS:
        rows = [row for row in regime_rows if row["model"] == model]
        result: dict[str, Any] = {"model": model, "cases": len(rows)}
        for outcome in ("terminal_joint", "eventual_exact", "oscillation", "stationary_failure"):
            result[f"observed_{outcome}"] = average(rows, f"observed_{outcome}")
            result[f"model_simulated_{outcome}"] = average(rows, f"model_simulated_{outcome}")
            result[f"pooled_simulated_{outcome}"] = average(rows, f"pooled_simulated_{outcome}")
        per_model_regime.append(result)

    rank_validation: dict[str, Any] = {}
    for outcome in ("terminal_joint", "oscillation", "stationary_failure"):
        observed = [float(row[f"observed_{outcome}"]) for row in per_model_regime]
        model_prediction = [float(row[f"model_simulated_{outcome}"]) for row in per_model_regime]
        pooled_prediction = [float(row[f"pooled_simulated_{outcome}"]) for row in per_model_regime]
        rank_validation[outcome] = {
            "model_specific_spearman": spearman(model_prediction, observed),
            "model_specific_exact_permutation_p": exact_spearman_p(model_prediction, observed),
            "pooled_spearman": spearman(pooled_prediction, observed),
            "pooled_exact_permutation_p": exact_spearman_p(pooled_prediction, observed),
            "model_specific_absolute_rate_error": statistics.mean(abs(x - y) for x, y in zip(model_prediction, observed)),
            "pooled_absolute_rate_error": statistics.mean(abs(x - y) for x, y in zip(pooled_prediction, observed)),
        }

    improved_models = [row["model"] for row in per_model_transition if float(row["mae_delta"]) < 0]
    non_llama_models = tuple(model for model in MODELS if not model.startswith("llama"))
    non_llama_rows = [row for row in all_transition_rows if row["model"] in non_llama_models]
    non_llama_delta = paired_case_id_bootstrap(
        non_llama_rows,
        non_llama_models,
        lambda grouped: statistics.mean(
            average(grouped[model], "model_abs_error")
            - average(grouped[model], "pooled_abs_error")
            for model in non_llama_models
        ),
        args.bootstraps,
        args.seed + 20,
    )
    high_rho_outcomes = [
        outcome for outcome, values in rank_validation.items()
        if float(values["model_specific_spearman"]) >= 0.70
    ]
    checks = {
        "aggregate_mae_interval_below_zero": aggregate["mae_delta_model_minus_pooled"]["ci95"][1] < 0,
        "relative_reduction_lower_bound_above_5pct": aggregate["relative_mae_reduction"]["ci95"][0] > 0.05,
        "at_least_five_of_seven_improve": len(improved_models) >= 5,
        "at_least_two_regime_rank_rho_ge_0_70": len(high_rho_outcomes) >= 2,
        "not_llama_only": non_llama_delta["ci95"][1] < 0,
    }
    if all(checks.values()):
        verdict = "STRONG_SPOTLIGHT_GATE_PASS"
    elif checks["aggregate_mae_interval_below_zero"] and len(improved_models) >= 4:
        verdict = "POSTER_PLUS_LINKED_PREDICTION"
    else:
        verdict = "NO_LINKED_UPLIFT_KEEP_IDENTIFY_CONTRAST_BOUNDARY"

    first_step_summary: dict[str, Any] = {}
    if first_transition_rows:
        for model in MODELS:
            rows = [row for row in first_transition_rows if row["model"] == model]
            if not rows:
                continue
            first_step_summary[model] = {
                "cases": len(rows),
                "model_mae": average(rows, "first_model_abs_error"),
                "pooled_mae": average(rows, "first_pooled_abs_error"),
                "delta": average(rows, "first_model_abs_error") - average(rows, "first_pooled_abs_error"),
            }

    recurrence_shift = {
        model: {
            "first_step_delta": first_step_summary[model]["delta"],
            "all_transition_delta": next(
                row["mae_delta"] for row in per_model_transition if row["model"] == model
            ),
            "all_minus_first_delta": next(
                row["mae_delta"] for row in per_model_transition if row["model"] == model
            )
            - first_step_summary[model]["delta"],
        }
        for model in MODELS
    }

    report = {
        "schema_version": 1,
        "verdict": verdict,
        "protocol": "docs/ICLR_SPOTLIGHT_GATE_PROTOCOL_20260731.md",
        "models": list(MODELS),
        "support": {"command_min": -50, "command_max": 50, "origin_added": [0, 0]},
        "excluded_system_id_cases": len(excluded_ids),
        "retained_cases_per_model": 408,
        "evidence": evidence,
        "transition_analysis": {
            "aggregate_equal_checkpoint_weight": aggregate,
            "per_model_intervals": model_intervals,
            "improved_models": improved_models,
            "non_llama_aggregate_mae_delta": non_llama_delta,
            "first_step_sensitivity": first_step_summary,
            "first_step_aggregate": first_step_aggregate,
            "recurrence_shift": recurrence_shift,
        },
        "regime_analysis": {
            "rank_validation": rank_validation,
            "high_rho_outcomes": high_rho_outcomes,
        },
        "checks": checks,
    }

    report_path = output_dir / "analysis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "transition_summary.csv", per_model_transition)
    write_csv(output_dir / "regime_summary.csv", per_model_regime)
    write_csv(output_dir / "transition_case_level.csv", all_transition_rows)
    write_csv(output_dir / "regime_case_level.csv", regime_rows)

    lines = [
        "# Policy-to-closed-loop spotlight gate",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Transition prediction",
        "",
        "| Checkpoint | Cases | Transitions | Specific MAE | Pooled MAE | Delta | Relative reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_model_transition:
        lines.append(
            f"| {row['model']} | {row['cases']} | {row['transitions']} | "
            f"{row['model_mae']:.3f} | {row['pooled_mae']:.3f} | {row['mae_delta']:+.3f} | "
            f"{100 * row['relative_reduction']:+.1f}% |"
        )
    agg_delta = aggregate["mae_delta_model_minus_pooled"]
    agg_relative = aggregate["relative_mae_reduction"]
    first_delta = first_step_aggregate["mae_delta_model_minus_pooled"]
    first_relative = first_step_aggregate["relative_mae_reduction"]
    lines += [
        "",
        f"Equal-checkpoint aggregate delta: {agg_delta['estimate']:+.3f} words "
        f"(95% CI {agg_delta['ci95'][0]:+.3f}, {agg_delta['ci95'][1]:+.3f}).",
        f"Aggregate relative reduction: {100 * agg_relative['estimate']:+.1f}% "
        f"(95% CI {100 * agg_relative['ci95'][0]:+.1f}%, {100 * agg_relative['ci95'][1]:+.1f}%).",
        f"First-step aggregate delta: {first_delta['estimate']:+.3f} words "
        f"(95% CI {first_delta['ci95'][0]:+.3f}, {first_delta['ci95'][1]:+.3f}); "
        f"relative reduction {100 * first_relative['estimate']:+.1f}% "
        f"(95% CI {100 * first_relative['ci95'][0]:+.1f}%, {100 * first_relative['ci95'][1]:+.1f}%).",
        "",
        "## Regime rank validation",
        "",
        "| Outcome | Model-specific rho | Exact p | Pooled rho | Exact p | Specific rate error | Pooled rate error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for outcome, values in rank_validation.items():
        lines.append(
            f"| {outcome} | {values['model_specific_spearman']:.3f} | "
            f"{values['model_specific_exact_permutation_p']:.4f} | "
            f"{values['pooled_spearman']:.3f} | "
            f"{values['pooled_exact_permutation_p']:.4f} | "
            f"{values['model_specific_absolute_rate_error']:.3f} | "
            f"{values['pooled_absolute_rate_error']:.3f} |"
        )
    lines += ["", "## Frozen checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: {'PASS' if value else 'FAIL'}")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "checks": checks, "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
