#!/usr/bin/env python3
"""Prospective cross-family capture--recurrence law audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exact_feedback.recurrence.unified_analysis import (
    MODELS,
    LINEAGES,
    main_registry,
    read_jsonl,
    sha256,
    source_family,
    state_tuple,
    validate_rows,
)


SEED = 20260801
BOOTSTRAPS = 10000
PERMUTATIONS = 100000
LANDMARKS = (3, 4, 5)
FAMILIES = ("Llama", "Gemma", "GLM", "Mistral", "Qwen", "Granite", "Falcon-H1")

B0_NUMERIC = (
    "current_error", "current_abs_error", "log_current_abs_error", "target", "current_missing_count",
)
B1_EXTRA = (
    "min_abs_error", "initial_to_current_abs_improvement", "contraction_count",
    "sign_flip_count", "length_noop_count", "largest_abs_action", "escape_by_landmark",
)
B2_EXTRA = (
    "recurrence_count", "repeat_fraction", "any_recurrence", "unique_output_ratio",
    "longest_identical_run", "period2_return",
)
CATEGORICAL = ("source_family", "length_band")


def family_for_model(model: str) -> str:
    lineage = LINEAGES[model]
    if lineage == "Ministral" or lineage == "Mistral":
        return "Mistral"
    if lineage == "Qwen":
        return "Qwen"
    return lineage


def longest_identical_run(states: list[tuple[str, int, tuple[str, ...]]]) -> int:
    best = 1
    current = 1
    for left, right in zip(states, states[1:]):
        if left == right:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def landmark_features(model: str, row: dict[str, Any], landmark: int) -> dict[str, Any] | None:
    rounds = row["rounds"]
    if len(rounds) < landmark + 2:
        return None
    observed = rounds[: landmark + 1]
    states = [state_tuple(item) for item in observed]
    errors = [int(item["error"]) for item in observed]
    actions = [right - left for left, right in zip(errors, errors[1:])]
    recurrence_flags: list[int] = []
    seen = {states[0]}
    for state in states[1:]:
        recurrence_flags.append(int(state in seen))
        seen.add(state)
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in errors]
    sign_flips = sum(left != 0 and right != 0 and left != right for left, right in zip(signs, signs[1:]))
    escape = any(0 < abs(left) <= 20 and abs(right) > 50 for left, right in zip(errors, errors[1:]))
    period2 = any(
        states[index] == states[index - 2] and states[index] != states[index - 1]
        for index in range(2, len(states))
    )
    current_error = errors[-1]
    return {
        "model": model,
        "family": family_for_model(model),
        "case_id": str(row["id"]),
        "source_family": source_family(str(row["source"])),
        "length_band": str(row.get("length_band", "")),
        "landmark": landmark,
        "late_rescue": int(bool(row["final_joint"])),
        "current_error": current_error,
        "current_abs_error": abs(current_error),
        "log_current_abs_error": math.log1p(abs(current_error)),
        "target": int(row["target"]),
        "current_missing_count": len(observed[-1].get("missing", [])),
        "min_abs_error": min(abs(value) for value in errors),
        "initial_to_current_abs_improvement": abs(errors[0]) - abs(current_error),
        "contraction_count": sum(abs(right) < abs(left) for left, right in zip(errors, errors[1:])),
        "sign_flip_count": sign_flips,
        "length_noop_count": sum(action == 0 for action in actions),
        "largest_abs_action": max((abs(action) for action in actions), default=0),
        "escape_by_landmark": int(escape),
        "recurrence_count": sum(recurrence_flags),
        "repeat_fraction": sum(recurrence_flags) / landmark,
        "any_recurrence": int(any(recurrence_flags)),
        "unique_output_ratio": len(set(states)) / len(states),
        "longest_identical_run": longest_identical_run(states),
        "period2_return": int(period2),
    }


def rows_to_columns(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> dict[str, list[Any]]:
    return {column: [row[column] for row in rows] for column in columns}


def design_matrix(rows: list[dict[str, Any]], numeric: tuple[str, ...], categorical: tuple[str, ...]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    numeric_values = np.asarray([[float(row[column]) for column in numeric] for row in rows], dtype=float)
    mean = numeric_values.mean(axis=0)
    std = numeric_values.std(axis=0)
    std[std < 1e-12] = 1.0
    blocks.append((numeric_values - mean) / std)
    for column in categorical:
        levels = sorted({str(row[column]) for row in rows})
        for level in levels[1:]:
            blocks.append(np.asarray([[float(str(row[column]) == level)] for row in rows]))
    return np.column_stack([np.ones(len(rows)), *blocks])


def make_predictor(numeric: tuple[str, ...]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(numeric)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ]
    )
    return Pipeline([
        ("features", transformer),
        ("model", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000)),
    ])


def records_for_sklearn(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{key: row[key] for key in (*columns, *CATEGORICAL)} for row in rows])


def leave_family_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    levels = {
        "b0": B0_NUMERIC,
        "b1": (*B0_NUMERIC, *B1_EXTRA),
        "b2": (*B0_NUMERIC, *B1_EXTRA, *B2_EXTRA),
    }
    predictions: list[dict[str, Any]] = []
    for family in FAMILIES:
        train = [row for row in rows if row["family"] != family]
        test = [row for row in rows if row["family"] == family]
        if not test:
            raise ValueError(f"empty held family: {family}")
        y_train = np.asarray([row["late_rescue"] for row in train], dtype=int)
        result_rows = [{
            "model": row["model"], "family": row["family"], "case_id": row["case_id"],
            "source_family": row["source_family"], "landmark": row["landmark"],
            "late_rescue": row["late_rescue"], "any_recurrence": row["any_recurrence"],
            "repeat_fraction": row["repeat_fraction"],
        } for row in test]
        for name, numeric in levels.items():
            predictor = make_predictor(numeric)
            predictor.fit(records_for_sklearn(train, numeric), y_train)
            probabilities = predictor.predict_proba(records_for_sklearn(test, numeric))[:, 1]
            for output, probability in zip(result_rows, probabilities):
                output[f"{name}_probability"] = float(probability)
                output[f"{name}_brier"] = (float(output["late_rescue"]) - probability) ** 2
        predictions.extend(result_rows)
    return predictions


def per_family_metrics(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for family in FAMILIES:
        rows = [row for row in predictions if row["family"] == family]
        y = np.asarray([row["late_rescue"] for row in rows], dtype=int)
        result: dict[str, Any] = {
            "family": family,
            "rows": len(rows),
            "models": len({row["model"] for row in rows}),
            "late_rescue_rate": float(y.mean()),
        }
        for level in ("b0", "b1", "b2"):
            probability = np.asarray([row[f"{level}_probability"] for row in rows], dtype=float)
            result[f"{level}_brier"] = float(brier_score_loss(y, probability))
            result[f"{level}_log_loss"] = float(log_loss(y, probability, labels=[0, 1]))
            result[f"{level}_auroc"] = float(roc_auc_score(y, probability)) if len(set(y)) == 2 else float("nan")
        result["b2_minus_b1_brier"] = result["b2_brier"] - result["b1_brier"]
        result["b2_relative_reduction"] = (result["b1_brier"] - result["b2_brier"]) / result["b1_brier"]
        results.append(result)
    return results


def bootstrap_weights(case_ids: list[str], source_by_case: dict[str, str], bootstraps: int, seed: int) -> np.ndarray:
    index = {case_id: position for position, case_id in enumerate(case_ids)}
    by_source: dict[str, list[int]] = defaultdict(list)
    for case_id in case_ids:
        by_source[source_by_case[case_id]].append(index[case_id])
    rng = np.random.default_rng(seed)
    weights = np.zeros((bootstraps, len(case_ids)), dtype=np.int16)
    for positions in by_source.values():
        draws = rng.multinomial(len(positions), np.full(len(positions), 1 / len(positions)), size=bootstraps)
        weights[:, np.asarray(positions)] = draws
    return weights


def equal_family_bootstrap(
    predictions: list[dict[str, Any]], case_ids: list[str], weights: np.ndarray,
) -> dict[str, Any]:
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    family_samples: dict[str, dict[str, np.ndarray]] = {}
    family_points: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        rows = [row for row in predictions if row["family"] == family]
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_case[row["case_id"]].append(row)
        family_samples[family] = {}
        family_points[family] = {}
        for level in ("b1", "b2"):
            values = np.full(len(case_ids), np.nan)
            for case_id, group in by_case.items():
                values[case_index[case_id]] = float(np.mean([row[f"{level}_brier"] for row in group]))
            valid = ~np.isnan(values)
            family_points[family][level] = float(np.mean(values[valid]))
            family_samples[family][level] = (weights[:, valid] @ values[valid]) / np.maximum(weights[:, valid].sum(axis=1), 1)
    b1_point = float(np.mean([family_points[family]["b1"] for family in FAMILIES]))
    b2_point = float(np.mean([family_points[family]["b2"] for family in FAMILIES]))
    b1_samples = np.vstack([family_samples[family]["b1"] for family in FAMILIES]).mean(axis=0)
    b2_samples = np.vstack([family_samples[family]["b2"] for family in FAMILIES]).mean(axis=0)
    difference = b2_samples - b1_samples
    relative = (b1_samples - b2_samples) / b1_samples
    return {
        "b1_brier": b1_point,
        "b2_brier": b2_point,
        "b2_minus_b1": {"estimate": b2_point - b1_point, "ci95": [float(np.quantile(difference, 0.025)), float(np.quantile(difference, 0.975))]},
        "relative_reduction": {"estimate": (b1_point - b2_point) / b1_point, "ci95": [float(np.quantile(relative, 0.025)), float(np.quantile(relative, 0.975))]},
    }


def common_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    control_numeric = (*B0_NUMERIC, *B1_EXTRA)
    values = np.asarray([[float(row[column]) for column in control_numeric] for row in rows], dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-12] = 1.0
    blocks: list[np.ndarray] = [
        np.ones((len(rows), 1)),
        np.asarray([[float(row["repeat_fraction"])] for row in rows]),
        (values - mean) / std,
    ]
    for column in ("model", "source_family", "length_band"):
        levels = sorted({str(row[column]) for row in rows})
        for level in levels[1:]:
            blocks.append(np.asarray([[float(str(row[column]) == level)] for row in rows]))
    x = np.column_stack(blocks)
    y = np.asarray([row["late_rescue"] for row in rows], dtype=float)

    def objective(beta_values: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta_values
        value = float(np.sum(np.logaddexp(0.0, linear) - y * linear))
        gradient = x.T @ (expit(linear) - y)
        return value, gradient

    fit = minimize(
        lambda coefficients: objective(coefficients)[0],
        np.zeros(x.shape[1]),
        jac=lambda coefficients: objective(coefficients)[1],
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    if not fit.success:
        raise RuntimeError(f"common GLM did not converge: {fit.message}")
    coefficients = fit.x
    probability = expit(x @ coefficients)
    hessian = x.T @ ((probability * (1 - probability))[:, None] * x)
    bread = np.linalg.pinv(hessian)
    individual_scores = x * (y - probability)[:, None]
    cluster_scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(x.shape[1]))
    for row, score in zip(rows, individual_scores):
        cluster_scores[row["case_id"]] += score
    meat = sum(np.outer(score, score) for score in cluster_scores.values())
    covariance = bread @ meat @ bread
    standard_error = float(math.sqrt(max(covariance[1, 1], 0.0)))
    coefficient = float(coefficients[1])
    z_value = coefficient / standard_error
    p_value = float(2 * norm.sf(abs(z_value)))
    return {
        "repeat_fraction_coefficient": coefficient,
        "clustered_standard_error": standard_error,
        "odds_ratio_full_repeat_fraction": math.exp(coefficient),
        "ci95_odds_ratio": [math.exp(coefficient - 1.96 * standard_error), math.exp(coefficient + 1.96 * standard_error)],
        "clustered_z": z_value,
        "clustered_p_two_sided": p_value,
        "case_clusters": len(cluster_scores),
        "rows": len(rows),
        "converged": bool(fit.success),
    }


def directional_contrasts(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    levels = sorted({str(row[key]) for row in rows})
    results = []
    for level in levels:
        group = [row for row in rows if str(row[key]) == level]
        exposed = [row["late_rescue"] for row in group if row["any_recurrence"]]
        unexposed = [row["late_rescue"] for row in group if not row["any_recurrence"]]
        results.append({
            key: level,
            "rows": len(group),
            "exposed": len(exposed),
            "unexposed": len(unexposed),
            "exposed_rescue_rate": float(np.mean(exposed)) if exposed else float("nan"),
            "unexposed_rescue_rate": float(np.mean(unexposed)) if unexposed else float("nan"),
            "risk_difference_exposed_minus_unexposed": float(np.mean(exposed) - np.mean(unexposed)) if exposed and unexposed else float("nan"),
            "estimable": bool(exposed and unexposed),
        })
    return results


def spearman(left: list[float], right: list[float]) -> float:
    x = rankdata(np.asarray(left, dtype=float))
    y = rankdata(np.asarray(right, dtype=float))
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def permutation_spearman(left: list[float], right: list[float], permutations: int, seed: int) -> dict[str, Any]:
    observed = spearman(left, right)
    rng = np.random.default_rng(seed)
    array = np.asarray(right, dtype=float)
    extreme = 0
    for _ in range(permutations):
        extreme += int(abs(spearman(left, rng.permutation(array).tolist())) >= abs(observed) - 1e-12)
    return {"rho": observed, "permutation_p_two_sided": (extreme + 1) / (permutations + 1), "permutations": permutations}


def exact_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    observed = spearman(left, right)
    total = 0
    extreme = 0
    for permuted in itertools.permutations(right):
        total += 1
        extreme += int(abs(spearman(left, list(permuted))) >= abs(observed) - 1e-12)
    return {"rho": observed, "exact_p_two_sided": extreme / total, "permutations": total}


def early_hazards(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    active_transitions = 0
    captures = 0
    recurrences = 0
    for row in rows:
        rounds = row["rounds"]
        states = [state_tuple(item) for item in rounds]
        for index in range(min(4, len(rounds) - 1)):
            active_transitions += 1
            captures += int(bool(rounds[index + 1].get("joint_success", False)))
            recurrences += int(states[index + 1] in set(states[: index + 1]))
    return {
        "model": model,
        "family": family_for_model(model),
        "active_transitions_r0_r4": active_transitions,
        "capture_events": captures,
        "recurrence_events": recurrences,
        "early_capture_hazard": captures / active_transitions,
        "early_recurrence_hazard": recurrences / active_transitions,
        "capture_recurrence_ratio": (captures + 0.5) / (recurrences + 0.5),
        "final_joint": float(np.mean([row["final_joint"] for row in rows])),
        "one_shot_joint": float(np.mean([row["one_shot_joint"] for row in rows])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/capture_recurrence_law_v1"))
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = main_registry(root)
    all_cases: dict[str, list[dict[str, Any]]] = {}
    integrity_errors: list[str] = []
    files: dict[str, list[dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    source_by_case: dict[str, str] = {}
    for model in MODELS:
        rows: list[dict[str, Any]] = []
        files[model] = []
        for path in registry[model]:
            part = read_jsonl(path)
            rows.extend(part)
            files[model].append({"path": str(path), "rows": len(part), "sha256": sha256(path)})
        integrity_errors.extend(validate_rows(model, rows))
        ids = {str(row["id"]) for row in rows}
        if reference_ids is None:
            reference_ids = ids
            source_by_case = {str(row["id"]): source_family(str(row["source"])) for row in rows}
        elif ids != reference_ids:
            integrity_errors.append(f"{model}: case set mismatch")
        all_cases[model] = rows
    if integrity_errors:
        raise ValueError("integrity failure:\n" + "\n".join(integrity_errors[:50]))
    assert reference_ids is not None
    case_ids = sorted(reference_ids)
    weights = bootstrap_weights(case_ids, source_by_case, args.bootstraps, args.seed)

    landmark_outputs: dict[int, dict[str, Any]] = {}
    primary_rows: list[dict[str, Any]] = []
    primary_predictions: list[dict[str, Any]] = []
    primary_family_metrics: list[dict[str, Any]] = []
    primary_model_contrasts: list[dict[str, Any]] = []
    primary_source_contrasts: list[dict[str, Any]] = []
    for landmark in LANDMARKS:
        rows = []
        for model in MODELS:
            for case in all_cases[model]:
                features = landmark_features(model, case, landmark)
                if features is not None:
                    rows.append(features)
        predictions = leave_family_out(rows)
        family_metrics = per_family_metrics(predictions)
        aggregate = equal_family_bootstrap(predictions, case_ids, weights)
        effect = common_effect(rows)
        model_contrasts = directional_contrasts(rows, "model")
        source_contrasts = directional_contrasts(rows, "source_family")
        landmark_outputs[landmark] = {
            "risk_rows": len(rows),
            "late_rescue_rate": float(np.mean([row["late_rescue"] for row in rows])),
            "equal_family_prediction": aggregate,
            "common_effect": effect,
            "negative_model_directions": sum(row["estimable"] and row["risk_difference_exposed_minus_unexposed"] < 0 for row in model_contrasts),
            "estimable_models": sum(row["estimable"] for row in model_contrasts),
            "negative_source_directions": sum(row["estimable"] and row["risk_difference_exposed_minus_unexposed"] < 0 for row in source_contrasts),
            "estimable_sources": sum(row["estimable"] for row in source_contrasts),
            "improved_families": [row["family"] for row in family_metrics if row["b2_minus_b1_brier"] < 0],
        }
        if landmark == 4:
            primary_rows = rows
            primary_predictions = predictions
            primary_family_metrics = family_metrics
            primary_model_contrasts = model_contrasts
            primary_source_contrasts = source_contrasts

    hazard_rows = [early_hazards(model, all_cases[model]) for model in MODELS]
    hazard_ratio = [row["capture_recurrence_ratio"] for row in hazard_rows]
    final_success = [row["final_joint"] for row in hazard_rows]
    hazard_association = permutation_spearman(hazard_ratio, final_success, args.permutations, args.seed + 50)
    qwen_models = ["qwen3_1_7b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b"]
    hazard_by_model = {row["model"]: row for row in hazard_rows}
    qwen_association = exact_spearman(
        [1.7, 4.0, 8.0, 14.0, 32.0],
        [hazard_by_model[model]["capture_recurrence_ratio"] for model in qwen_models],
    )

    primary = landmark_outputs[4]
    improved_families = primary["improved_families"]
    robustness_pass = all(
        landmark_outputs[landmark]["common_effect"]["repeat_fraction_coefficient"] < 0
        and landmark_outputs[landmark]["equal_family_prediction"]["b2_minus_b1"]["estimate"] < 0
        for landmark in (3, 5)
    )
    checks = {
        "adjusted_or_below_one_cluster_p_lt_0_01": primary["common_effect"]["odds_ratio_full_repeat_fraction"] < 1 and primary["common_effect"]["clustered_p_two_sided"] < 0.01,
        "negative_in_ten_models_and_four_sources": primary["negative_model_directions"] >= 10 and primary["negative_source_directions"] == 4,
        "lofo_brier_ci_below_zero_relative_ge_5pct_five_families": primary["equal_family_prediction"]["b2_minus_b1"]["ci95"][1] < 0 and primary["equal_family_prediction"]["relative_reduction"]["estimate"] >= 0.05 and len(improved_families) >= 5,
        "early_ratio_rho_ge_0_70_p_lt_0_05": hazard_association["rho"] >= 0.70 and hazard_association["permutation_p_two_sided"] < 0.05,
        "landmarks_3_and_5_preserve_negative_effect_and_b2_advantage": robustness_pass,
    }
    if all(checks.values()):
        verdict = "STRONG_CROSS_FAMILY_LAW"
    elif checks["adjusted_or_below_one_cluster_p_lt_0_01"] and checks["negative_in_ten_models_and_four_sources"] and checks["early_ratio_rho_ge_0_70_p_lt_0_05"]:
        verdict = "CONSISTENT_PROGNOSTIC_LAW"
    else:
        verdict = "NO_UNIVERSAL_LAW_KEEP_RECURRENCE_TAXONOMY"

    manifest = {
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "upstream_attractor_manifest": {
            "path": str(root / "experiments/unified_output_recurrence_audit_v2/input_manifest.json"),
            "sha256": sha256(root / "experiments/unified_output_recurrence_audit_v2/input_manifest.json"),
        },
        "files": files,
        "integrity": {"passed": True, "models": len(MODELS), "cases_per_model": len(case_ids)},
    }
    report = {
        "schema_version": 1,
        "verdict": verdict,
        "protocol": "docs/ICLR_CAPTURE_RECURRENCE_LAW_PROTOCOL_20260801.md",
        "population": {"models": len(MODELS), "families": len(FAMILIES), "cases_per_model": len(case_ids)},
        "landmarks": landmark_outputs,
        "early_hazard_association": hazard_association,
        "qwen_dense_size_vs_capture_recurrence_ratio": qwen_association,
        "checks": checks,
        "manifest": "input_manifest.json",
    }
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "landmark4_case_features.csv", primary_rows)
    write_csv(output_dir / "landmark4_held_family_predictions.csv", primary_predictions)
    write_csv(output_dir / "landmark4_family_metrics.csv", primary_family_metrics)
    write_csv(output_dir / "landmark4_model_contrasts.csv", primary_model_contrasts)
    write_csv(output_dir / "landmark4_source_contrasts.csv", primary_source_contrasts)
    write_csv(output_dir / "early_hazards.csv", hazard_rows)
    write_csv(output_dir / "landmark_sensitivity.csv", [
        {
            "landmark": landmark,
            "risk_rows": values["risk_rows"],
            "repeat_odds_ratio": values["common_effect"]["odds_ratio_full_repeat_fraction"],
            "repeat_cluster_p": values["common_effect"]["clustered_p_two_sided"],
            "b2_minus_b1_brier": values["equal_family_prediction"]["b2_minus_b1"]["estimate"],
            "b2_minus_b1_ci_lo": values["equal_family_prediction"]["b2_minus_b1"]["ci95"][0],
            "b2_minus_b1_ci_hi": values["equal_family_prediction"]["b2_minus_b1"]["ci95"][1],
            "relative_brier_reduction": values["equal_family_prediction"]["relative_reduction"]["estimate"],
            "negative_models": values["negative_model_directions"],
            "negative_sources": values["negative_source_directions"],
            "improved_families": ";".join(values["improved_families"]),
        }
        for landmark, values in sorted(landmark_outputs.items())
    ])

    effect = primary["common_effect"]
    prediction = primary["equal_family_prediction"]
    lines = [
        "# Capture--recurrence law audit", "", f"Verdict: **{verdict}**", "",
        f"Revision-4 risk set: {primary['risk_rows']:,} model-cases across seven held-family folds.", "",
        "## Common adjusted effect", "",
        f"Repeat-fraction OR: {effect['odds_ratio_full_repeat_fraction']:.3f} "
        f"(95% CI {effect['ci95_odds_ratio'][0]:.3f}, {effect['ci95_odds_ratio'][1]:.3f}); "
        f"case-clustered p={effect['clustered_p_two_sided']:.4g}.",
        f"Directional consistency: {primary['negative_model_directions']}/{primary['estimable_models']} models and "
        f"{primary['negative_source_directions']}/{primary['estimable_sources']} sources negative.", "",
        "## Leave-one-family-out prediction", "",
        f"Equal-family Brier B1={prediction['b1_brier']:.4f}, B2={prediction['b2_brier']:.4f}.",
        f"B2-B1={prediction['b2_minus_b1']['estimate']:+.4f} "
        f"(95% CI {prediction['b2_minus_b1']['ci95'][0]:+.4f}, {prediction['b2_minus_b1']['ci95'][1]:+.4f}); "
        f"relative reduction {100*prediction['relative_reduction']['estimate']:+.1f}%.",
        f"Improved held families: {', '.join(improved_families) if improved_families else 'none'}.", "",
        "## Early hazard ratio", "",
        f"Capture--recurrence ratio versus final success: rho={hazard_association['rho']:+.3f}, "
        f"permutation p={hazard_association['permutation_p_two_sided']:.5f}.",
        f"Qwen Dense size versus early ratio: rho={qwen_association['rho']:+.3f}, exact p={qwen_association['exact_p_two_sided']:.4f}.",
        "", "## Decision checks", "",
    ]
    lines.extend(f"- {key}: **{value}**" for key, value in checks.items())
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
