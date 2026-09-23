#!/usr/bin/env python3
"""Audit cross-task recurrence and verifier-state recovery at revision 4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


EXPECTED_MODELS = {
    "gemma2_9b",
    "glm4_9b",
    "granite_3_3_8b",
    "llama31_8b",
    "ministral_8b",
    "qwen3_14b",
}
EXPECTED_FAMILIES = {
    "lexical_position_c05",
    "compositional_structure_c10",
}
LANDMARK = 4
TOL = 1e-12
SEED = 20260805


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def validate_stage(stage_dir: Path, expected_split: str) -> dict[str, list[dict[str, Any]]]:
    paths = sorted(stage_dir.glob("*/cases.jsonl"))
    models = {path.parent.name for path in paths}
    if models != EXPECTED_MODELS:
        raise ValueError(f"{stage_dir}: models={sorted(models)}, expected={sorted(EXPECTED_MODELS)}")
    by_model: dict[str, list[dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    for path in paths:
        model = path.parent.name
        rows = read_jsonl(path)
        if len(rows) != 120:
            raise ValueError(f"{path}: expected 120 rows, found {len(rows)}")
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate case IDs")
        families = defaultdict(int)
        for row in rows:
            families[str(row["family"])] += 1
            if str(row["split"]) != expected_split:
                raise ValueError(f"{path}: unexpected split {row['split']}")
            rounds = row["rounds"]
            revisions = [int(item["revision"]) for item in rounds]
            if revisions != list(range(len(rounds))):
                raise ValueError(f"{path}: non-contiguous revisions for {row['id']}")
            if bool(row["final_joint"]):
                if not bool(rounds[-1]["joint_success"]):
                    raise ValueError(f"{path}: inconsistent final success for {row['id']}")
            elif revisions[-1] != 8:
                raise ValueError(f"{path}: unresolved case stopped before revision 8: {row['id']}")
        if set(families) != EXPECTED_FAMILIES or any(value != 60 for value in families.values()):
            raise ValueError(f"{path}: family counts={dict(families)}")
        if reference_ids is None:
            reference_ids = set(ids)
        elif set(ids) != reference_ids:
            raise ValueError(f"{path}: case set differs across checkpoints")
        by_model[model] = rows
    return by_model


def landmark_row(stage: str, model: str, row: dict[str, Any]) -> dict[str, Any] | None:
    rounds = row["rounds"]
    if len(rounds) <= LANDMARK or bool(rounds[LANDMARK]["joint_success"]):
        return None
    prefix = rounds[: LANDMARK + 1]
    later = rounds[LANDMARK + 1 :]
    if not later:
        raise ValueError(f"missing post-landmark rounds: {model}/{row['id']}")

    seen = {str(prefix[0]["text"])}
    recurrence_flags = []
    for item in prefix[1:]:
        text = str(item["text"])
        recurrence_flags.append(int(text in seen))
        seen.add(text)

    prefix_energy = [float(item["violation_energy"]) for item in prefix]
    later_energy = [float(item["violation_energy"]) for item in later]
    current_energy = prefix_energy[-1]
    best_post_energy = min(later_energy)
    current_vector = [float(value) for value in prefix[-1]["verifier"]["violation_vector"]]
    later_vectors = [
        [float(value) for value in item["verifier"]["violation_vector"]]
        for item in later
    ]
    positive_components = [index for index, value in enumerate(current_vector) if value > TOL]
    zero_components = [index for index, value in enumerate(current_vector) if value <= TOL]
    components_repaired = max(
        (
            sum(index < len(vector) and vector[index] <= TOL for index in positive_components)
            for vector in later_vectors
        ),
        default=0,
    )
    collateral = any(
        index >= len(vector) or vector[index] > TOL
        for vector in later_vectors
        for index in zero_components
    )
    prefix_texts = {str(item["text"]) for item in prefix}
    return {
        "stage": stage,
        "model": model,
        "family": str(row["family"]),
        "case_id": str(row["id"]),
        "source": str(row["source"]),
        "repeat_count": sum(recurrence_flags),
        "repeat_fraction": sum(recurrence_flags) / LANDMARK,
        "any_recurrence": int(any(recurrence_flags)),
        "current_energy": current_energy,
        "best_prior_energy": min(prefix_energy),
        "prior_contractions": sum(
            right < left - TOL for left, right in zip(prefix_energy, prefix_energy[1:])
        ),
        "prior_exact_noops": sum(
            str(left["text"]) == str(right["text"])
            for left, right in zip(prefix, prefix[1:])
        ),
        "joint_rescue": int(any(bool(item["joint_success"]) for item in later)),
        "any_energy_recovery": int(best_post_energy < current_energy - TOL),
        "best_energy_improvement": current_energy - best_post_energy,
        "components_repaired": components_repaired,
        "novel_escape": int(any(str(item["text"]) not in prefix_texts for item in later)),
        "collateral_violation": int(collateral),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recurrent = [row for row in rows if row["any_recurrence"]]
    nonrecurrent = [row for row in rows if not row["any_recurrence"]]

    def mean(group: list[dict[str, Any]], key: str) -> float | None:
        return float(np.mean([float(row[key]) for row in group])) if group else None

    return {
        "rows": len(rows),
        "case_clusters": len({row["case_id"] for row in rows}),
        "recurrent_rows": len(recurrent),
        "recurrence_rate": mean(rows, "any_recurrence"),
        "joint_rescue_recurrent": mean(recurrent, "joint_rescue"),
        "joint_rescue_nonrecurrent": mean(nonrecurrent, "joint_rescue"),
        "energy_recovery_recurrent": mean(recurrent, "any_energy_recovery"),
        "energy_recovery_nonrecurrent": mean(nonrecurrent, "any_energy_recovery"),
        "energy_recovery_risk_difference": (
            mean(recurrent, "any_energy_recovery") - mean(nonrecurrent, "any_energy_recovery")
            if recurrent and nonrecurrent
            else None
        ),
        "best_improvement_recurrent": mean(recurrent, "best_energy_improvement"),
        "best_improvement_nonrecurrent": mean(nonrecurrent, "best_energy_improvement"),
        "components_repaired_recurrent": mean(recurrent, "components_repaired"),
        "components_repaired_nonrecurrent": mean(nonrecurrent, "components_repaired"),
        "collateral_recurrent": mean(recurrent, "collateral_violation"),
        "collateral_nonrecurrent": mean(nonrecurrent, "collateral_violation"),
    }


def clustered_logistic(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    numeric = ("current_energy", "best_prior_energy", "prior_contractions", "prior_exact_noops")
    values = np.asarray([[float(row[key]) for key in numeric] for row in rows], dtype=float)
    means = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < TOL] = 1.0
    blocks = [
        np.ones((len(rows), 1)),
        np.asarray([[float(row["any_recurrence"])] for row in rows]),
        (values - means) / std,
    ]
    for column in ("model", "source", "family"):
        levels = sorted({str(row[column]) for row in rows})
        for level in levels[1:]:
            blocks.append(np.asarray([[float(str(row[column]) == level)] for row in rows]))
    x = np.column_stack(blocks)
    y = np.asarray([float(row[outcome]) for row in rows])
    if len(set(y)) < 2:
        return {"estimable": False, "reason": "outcome has one class", "rows": len(rows)}

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        return (
            float(np.sum(np.logaddexp(0.0, linear) - y * linear)),
            x.T @ (expit(linear) - y),
        )

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(x.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    if not fit.success:
        return {"estimable": False, "reason": str(fit.message), "rows": len(rows)}
    beta = fit.x
    probability = expit(x @ beta)
    hessian = x.T @ ((probability * (1.0 - probability))[:, None] * x)
    bread = np.linalg.pinv(hessian)
    scores = x * (y - probability)[:, None]
    clustered: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(x.shape[1]))
    for row, score in zip(rows, scores):
        clustered[row["case_id"]] += score
    meat = sum(np.outer(score, score) for score in clustered.values())
    covariance = bread @ meat @ bread
    standard_error = float(math.sqrt(max(covariance[1, 1], 0.0)))
    coefficient = float(beta[1])
    if standard_error <= TOL:
        return {"estimable": False, "reason": "zero clustered standard error", "rows": len(rows)}
    z_value = coefficient / standard_error
    return {
        "estimable": True,
        "rows": len(rows),
        "case_clusters": len(clustered),
        "coefficient_any_recurrence": coefficient,
        "odds_ratio": math.exp(coefficient),
        "ci95_odds_ratio": [
            math.exp(coefficient - 1.96 * standard_error),
            math.exp(coefficient + 1.96 * standard_error),
        ],
        "clustered_standard_error": standard_error,
        "clustered_z": z_value,
        "clustered_p_two_sided": float(2.0 * norm.sf(abs(z_value))),
    }


def stratified_case_bootstrap(
    rows: list[dict[str, Any]], bootstraps: int, seed: int
) -> dict[str, Any]:
    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"]].append(row)
    for case_id, case_rows in rows_by_case.items():
        first = case_rows[0]
        by_stratum[(first["family"], first["source"])].append(case_id)

    def contrasts(sample: list[dict[str, Any]]) -> tuple[float, float]:
        recurrent = [row for row in sample if row["any_recurrence"]]
        nonrecurrent = [row for row in sample if not row["any_recurrence"]]
        if not recurrent or not nonrecurrent:
            return float("nan"), float("nan")
        risk = np.mean([row["any_energy_recovery"] for row in recurrent]) - np.mean(
            [row["any_energy_recovery"] for row in nonrecurrent]
        )
        improvement = np.mean([row["best_energy_improvement"] for row in recurrent]) - np.mean(
            [row["best_energy_improvement"] for row in nonrecurrent]
        )
        return float(risk), float(improvement)

    point_risk, point_improvement = contrasts(rows)
    rng = np.random.default_rng(seed)
    risk_samples = []
    improvement_samples = []
    for _ in range(bootstraps):
        sample = []
        for case_ids in by_stratum.values():
            selected = rng.choice(case_ids, size=len(case_ids), replace=True)
            for case_id in selected:
                sample.extend(rows_by_case[str(case_id)])
        risk, improvement = contrasts(sample)
        if math.isfinite(risk):
            risk_samples.append(risk)
        if math.isfinite(improvement):
            improvement_samples.append(improvement)
    return {
        "bootstraps_requested": bootstraps,
        "energy_recovery_risk_difference": {
            "estimate": point_risk,
            "ci95": [float(np.quantile(risk_samples, 0.025)), float(np.quantile(risk_samples, 0.975))],
            "valid_bootstraps": len(risk_samples),
        },
        "best_energy_improvement_difference": {
            "estimate": point_improvement,
            "ci95": [
                float(np.quantile(improvement_samples, 0.025)),
                float(np.quantile(improvement_samples, 0.975)),
            ],
            "valid_bootstraps": len(improvement_samples),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=20000)
    args = parser.parse_args()

    all_features = []
    audit = {}
    for stage, path, split in (
        ("discovery", args.discovery_dir, "discovery"),
        ("confirmation", args.confirmation_dir, "confirmation"),
    ):
        by_model = validate_stage(path, split)
        audit[stage] = {"models": len(by_model), "rows": sum(map(len, by_model.values()))}
        for model, rows in by_model.items():
            for row in rows:
                feature = landmark_row(stage, model, row)
                if feature is not None:
                    all_features.append(feature)

    results: dict[str, Any] = {"audit": audit, "landmark": LANDMARK, "stages": {}}
    for stage in ("discovery", "confirmation"):
        stage_rows = [row for row in all_features if row["stage"] == stage]
        stage_result: dict[str, Any] = {"overall": summarize_group(stage_rows), "families": {}}
        for family in sorted(EXPECTED_FAMILIES):
            family_rows = [row for row in stage_rows if row["family"] == family]
            stage_result["families"][family] = {
                "summary": summarize_group(family_rows),
                "adjusted_energy_recovery": clustered_logistic(family_rows, "any_energy_recovery"),
                "bootstrap": stratified_case_bootstrap(
                    family_rows, args.bootstraps, SEED + len(stage_result["families"])
                ),
            }
        stage_result["pooled_adjusted_energy_recovery"] = clustered_logistic(
            stage_rows, "any_energy_recovery"
        )
        stage_result["pooled_bootstrap"] = stratified_case_bootstrap(
            stage_rows, args.bootstraps, SEED + 10
        )
        results["stages"][stage] = stage_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "landmark4_rows.csv", all_features)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

