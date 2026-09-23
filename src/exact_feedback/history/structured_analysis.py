#!/usr/bin/env python3
"""Audit and analyze the frozen Gemma/Qwen/Falcon reset confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from exact_feedback.history.structured_protocol import (
    BOOTSTRAPS,
    FORMAL_N,
    audit_arm,
    read_jsonl,
    sha256,
    summarize_cell,
)


SEED = 2026080702
MODELS = ("gemma2_9b", "qwen3_14b", "falcon_h1_7b")
STRUCTURES = ("c05", "c10", "c12")


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def cell_mean(pair_list: list[dict[str, dict[str, Any]]], metric: str) -> float:
    return sum(
        float(pair["history_reset"][metric]) - float(pair["full_history"][metric])
        for pair in pair_list
    ) / len(pair_list)


def macro_bootstrap(
    cells: list[list[dict[str, dict[str, Any]]]], metric: str, iterations: int, seed: int
) -> dict[str, Any]:
    point = sum(cell_mean(cell, metric) for cell in cells) / len(cells)
    rng = random.Random(seed)
    draws = []
    stratified = []
    for cell in cells:
        groups: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
        for pair in cell:
            groups[pair["full_history"]["source"]].append(pair)
        stratified.append(groups)
    for _ in range(iterations):
        cell_draws = []
        for groups in stratified:
            sample = [
                rng.choice(group)
                for source in sorted(groups)
                for group in [groups[source]]
                for _ in range(len(group))
            ]
            cell_draws.append(cell_mean(sample, metric))
        draws.append(sum(cell_draws) / len(cell_draws))
    return {"effect": point, "bootstrap_95ci": [quantile(draws, 0.025), quantile(draws, 0.975)]}


def robust_interaction(
    selected: list[tuple[str, str, dict[str, dict[str, Any]]]], metric: str
) -> dict[str, Any]:
    model_levels = list(MODELS)
    structure_levels = sorted({structure for _, structure, _ in selected})
    if len(structure_levels) < 2:
        return {"estimable": False, "reason": "fewer than two common structures"}
    rows = []
    outcomes = []
    for model, structure, pair in selected:
        model_terms = [float(model == level) for level in model_levels[1:]]
        structure_term = float(structure == structure_levels[1])
        interactions = [term * structure_term for term in model_terms]
        rows.append([1.0, *model_terms, structure_term, *interactions])
        outcomes.append(float(pair["history_reset"][metric]) - float(pair["full_history"][metric]))
    x = np.asarray(rows, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.clip(np.diag(x @ xtx_inv @ x.T), 0.0, 0.999999)
    scaled = residual / (1.0 - leverage)
    meat = x.T @ np.diag(scaled * scaled) @ x
    covariance = xtx_inv @ meat @ xtx_inv
    interaction_indices = list(range(x.shape[1] - len(model_levels) + 1, x.shape[1]))
    # For three models this selects the final two model-by-structure terms.
    interaction_indices = list(range(x.shape[1] - (len(model_levels) - 1), x.shape[1]))
    r = np.zeros((len(interaction_indices), x.shape[1]))
    for row_index, column_index in enumerate(interaction_indices):
        r[row_index, column_index] = 1.0
    contrast = r @ beta
    contrast_covariance = r @ covariance @ r.T
    statistic = float(contrast.T @ np.linalg.pinv(contrast_covariance) @ contrast)
    return {
        "estimable": True,
        "common_structures": structure_levels,
        "n": len(y),
        "wald_chi2": statistic,
        "df": len(interaction_indices),
        "p": float(chi2.sf(statistic, len(interaction_indices))),
        "interaction_coefficients": contrast.tolist(),
        "model_reference": model_levels[0],
        "structure_reference": structure_levels[0],
        "covariance": "HC3",
        "estimand": "linear probability model on within-state paired differences",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    states_path = args.experiment_dir / "states.jsonl"
    states = read_jsonl(states_path)
    state_map = {row["state_id"]: row for row in states}
    errors = []
    pairs_by_cell: dict[tuple[str, str], list[dict[str, dict[str, Any]]]] = defaultdict(list)
    input_hashes = {"states.jsonl": sha256(states_path)}

    for model in MODELS:
        cases_path = args.experiment_dir / model / "cases.jsonl"
        manifest_path = args.experiment_dir / model / "runtime_manifest.json"
        rows = read_jsonl(cases_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_states = {key: value for key, value in state_map.items() if value["model"] == model}
        if manifest.get("states") != len(expected_states) or manifest.get("arms") != 2 * len(expected_states):
            errors.append(f"{model}: runtime counts mismatch")
        if manifest.get("states_sha256") != sha256(states_path):
            errors.append(f"{model}: states hash mismatch")
        if manifest.get("cases_sha256") != sha256(cases_path):
            errors.append(f"{model}: cases hash mismatch")
        input_hashes[f"{model}/cases.jsonl"] = sha256(cases_path)
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for raw in rows:
            state_id = raw.get("state_id")
            if state_id not in expected_states:
                errors.append(f"{model}: unexpected state {state_id}")
                continue
            condition = raw.get("condition")
            if condition in grouped[state_id]:
                errors.append(f"{model}: duplicate {state_id}/{condition}")
                continue
            grouped[state_id][condition] = audit_arm(raw, expected_states[state_id], errors)
        if set(grouped) != set(expected_states):
            errors.append(f"{model}: state coverage mismatch")
        for state_id, pair in grouped.items():
            if set(pair) != {"full_history", "history_reset"}:
                errors.append(f"{model}: incomplete pair {state_id}")
                continue
            pairs_by_cell[(model, expected_states[state_id]["structure"])].append(pair)

    audit = {"verdict": "PASS" if not errors else "FAIL", "errors": errors, "input_sha256": input_hashes}
    (args.output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"audit failed with {len(errors)} errors")

    summaries = []
    for index, ((model, structure), pairs) in enumerate(sorted(pairs_by_cell.items())):
        summaries.append(summarize_cell(model, structure, pairs, args.iterations, SEED + index * 100_000))

    formal_keys = [
        (cell["model"], cell["structure"])
        for cell in summaries
        if cell["evidence_tier"] == "formal"
    ]
    formal_cells = [pairs_by_cell[key] for key in formal_keys]
    overall_escape = macro_bootstrap(formal_cells, "first_step_escape", args.iterations, SEED)
    by_model = {}
    for index, model in enumerate(MODELS):
        model_keys = [key for key in formal_keys if key[0] == model]
        by_model[model] = macro_bootstrap(
            [pairs_by_cell[key] for key in model_keys],
            "first_step_escape",
            args.iterations,
            SEED + (index + 1) * 1_000_000,
        )
    leave_one_out = {}
    for index, omitted in enumerate(MODELS):
        keys = [key for key in formal_keys if key[0] != omitted]
        leave_one_out[omitted] = macro_bootstrap(
            [pairs_by_cell[key] for key in keys],
            "first_step_escape",
            args.iterations,
            SEED + (index + 5) * 1_000_000,
        )

    common_structures = sorted(
        set.intersection(*[
            {structure for candidate_model, structure in formal_keys if candidate_model == model}
            for model in MODELS
        ])
    )
    interaction_rows = [
        (model, structure, pair)
        for model in MODELS
        for structure in common_structures
        for pair in pairs_by_cell[(model, structure)]
    ]
    interactions = {
        metric: robust_interaction(interaction_rows, metric)
        for metric in ("first_step_escape", "first_step_contraction", "final_joint")
    }
    positive_models = [model for model, result in by_model.items() if result["effect"] > 0]
    confirmation = {
        "macro_escape_ci_excludes_zero": overall_escape["bootstrap_95ci"][0] > 0,
        "at_least_two_positive_family_macros": len(positive_models) >= 2,
        "positive_family_macros": positive_models,
    }
    confirmation["pass"] = all([
        confirmation["macro_escape_ci_excludes_zero"],
        confirmation["at_least_two_positive_family_macros"],
    ])
    confirmation["interpretation"] = (
        "Cross-family causal confirmation: accumulated history stabilizes recurrence, while downstream contraction and capture remain heterogeneous."
        if confirmation["pass"]
        else "Targeted cross-family escape confirmation failed; retain the Llama/GLM result as scoped evidence and stop reset expansion."
    )

    analysis = {
        "schema_version": 1,
        "estimand": "within-state history-reset effect among recurrence-triggered states",
        "bootstrap_iterations": args.iterations,
        "formal_cell_minimum_n": FORMAL_N,
        "audit": audit,
        "cells": summaries,
        "equal_cell_macro_escape": overall_escape,
        "family_macro_escape": by_model,
        "leave_one_family_out_escape": leave_one_out,
        "common_formal_interaction_structures": common_structures,
        "model_by_structure_interactions": interactions,
        "confirmation_gate": confirmation,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.output_dir / "cell_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "structure", "pairs", "tier", "metric", "full", "reset", "delta", "ci_low", "ci_high", "p"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in summaries:
            for metric, item in cell["metrics"].items():
                low, high = item["source_stratified_paired_bootstrap_95ci"]
                writer.writerow({
                    "model": cell["model"], "structure": cell["structure"],
                    "pairs": cell["pairs"], "tier": cell["evidence_tier"], "metric": metric,
                    "full": item["full_history_mean"], "reset": item["history_reset_mean"],
                    "delta": item["paired_difference"], "ci_low": low, "ci_high": high,
                    "p": item.get("exact_mcnemar_p", item.get("wilcoxon_p")),
                })

    report = [
        "# Targeted Cross-Family History-Reset Confirmation",
        "",
        analysis["confirmation_gate"]["interpretation"],
        "",
        f"Equal-cell macro escape effect: {overall_escape['effect']:+.1%} "
        f"[{overall_escape['bootstrap_95ci'][0]:+.1%}, {overall_escape['bootstrap_95ci'][1]:+.1%}].",
        "",
        "| Model | Structure | N | Tier | Escape delta | Contraction delta | Avoid later recurrence | Final capture delta |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for cell in summaries:
        def display(metric: str) -> str:
            item = cell["metrics"][metric]
            low, high = item["source_stratified_paired_bootstrap_95ci"]
            return f"{item['paired_difference']:+.1%} [{low:+.1%}, {high:+.1%}]"
        report.append(
            f"| {cell['model']} | {cell['structure']} | {cell['pairs']} | {cell['evidence_tier']} | "
            f"{display('first_step_escape')} | {display('first_step_contraction')} | "
            f"{display('avoids_later_recurrence')} | {display('final_joint')} |"
        )
    report.extend(["", "## Interaction", ""])
    for metric, result in interactions.items():
        if result["estimable"]:
            report.append(f"- {metric}: HC3 Wald chi2({result['df']})={result['wald_chi2']:.3f}, p={result['p']:.4g} on {', '.join(common_structures)}.")
    report.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"audit": "PASS", "confirmation_gate": confirmation, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
