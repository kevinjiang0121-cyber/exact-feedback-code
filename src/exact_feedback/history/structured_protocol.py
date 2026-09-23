#!/usr/bin/env python3
"""Audit and analyze paired multidomain history-reset continuations.

The estimand is the within-state effect of replacing full dialogue history with
the frozen reset context among states selected because recurrence had already
occurred.  It is not an average treatment effect over all parent trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
import exact_feedback.common.structured_protocol as protocol  # noqa: E402


SEED = 20260807
BOOTSTRAPS = 20_000
FORMAL_N = 30
MAIN_STRUCTURES = {"c05", "c10", "c12"}
EXPLORATORY_STRUCTURES = {"c09"}
DESCRIPTIVE_STRUCTURES = {"c07"}

BINARY_METRICS = (
    "first_step_escape",
    "first_step_contraction",
    "ever_contraction",
    "avoids_any_recurrence",
    "avoids_later_recurrence",
    "final_joint",
)
CONTINUOUS_METRICS = (
    "first_step_energy_reduction",
    "final_energy_reduction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def close(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerance
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return all(close(a[key], b[key], tolerance) for key in a)
    return a == b


def arm_features(row: dict[str, Any], initial_energy: float) -> dict[str, Any]:
    rounds = row["rounds"]
    later_recurrence = any(not bool(step["escapes_prior_recurrence"]) for step in rounds[1:])
    any_recurrence = any(not bool(step["escapes_prior_recurrence"]) for step in rounds)
    return {
        **row,
        "first_step_escape": bool(row["first_step_escape"]),
        "first_step_contraction": bool(row["first_step_contraction"]),
        "ever_contraction": any(bool(step["contracts_violation_energy"]) for step in rounds),
        "avoids_any_recurrence": not any_recurrence,
        "avoids_later_recurrence": not later_recurrence,
        "final_joint": bool(row["final_joint"]),
        "first_step_energy_reduction": initial_energy - float(rounds[0]["violation_energy"]),
        "final_energy_reduction": initial_energy - float(rounds[-1]["violation_energy"]),
        "post_trigger_generations": len(rounds),
    }


def paired_differences(pairs: list[dict[str, dict[str, Any]]], metric: str) -> list[float]:
    return [
        float(pair["history_reset"][metric]) - float(pair["full_history"][metric])
        for pair in pairs
    ]


def bootstrap_ci(
    pairs: list[dict[str, dict[str, Any]]], metric: str, iterations: int, seed: int
) -> tuple[float, float]:
    strata: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        strata[pair["full_history"]["source"]].append(pair)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sample = [
            rng.choice(group)
            for source in sorted(strata)
            for group in [strata[source]]
            for _ in range(len(group))
        ]
        differences = paired_differences(sample, metric)
        draws.append(sum(differences) / len(differences))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def evidence_tier(structure: str, n: int) -> str:
    if structure in DESCRIPTIVE_STRUCTURES or n < FORMAL_N:
        return "descriptive"
    if structure in EXPLORATORY_STRUCTURES:
        return "exploratory"
    return "formal"


def summarize_cell(
    model: str,
    structure: str,
    pairs: list[dict[str, dict[str, Any]]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    tier = evidence_tier(structure, len(pairs))
    result: dict[str, Any] = {
        "model": model,
        "structure": structure,
        "pairs": len(pairs),
        "evidence_tier": tier,
        "metrics": {},
    }
    for index, metric in enumerate(BINARY_METRICS + CONTINUOUS_METRICS):
        full_values = [float(pair["full_history"][metric]) for pair in pairs]
        reset_values = [float(pair["history_reset"][metric]) for pair in pairs]
        differences = [reset - full for full, reset in zip(full_values, reset_values)]
        low, high = bootstrap_ci(pairs, metric, iterations, seed + index * 1009)
        summary = {
            "full_history_mean": sum(full_values) / len(full_values),
            "history_reset_mean": sum(reset_values) / len(reset_values),
            "paired_difference": sum(differences) / len(differences),
            "source_stratified_paired_bootstrap_95ci": [low, high],
        }
        if metric in BINARY_METRICS:
            reset_only = sum(full == 0 and reset == 1 for full, reset in zip(full_values, reset_values))
            full_only = sum(full == 1 and reset == 0 for full, reset in zip(full_values, reset_values))
            discordant = reset_only + full_only
            summary.update({
                "reset_only_discordant": reset_only,
                "full_only_discordant": full_only,
                "exact_mcnemar_p": binomtest(min(reset_only, full_only), discordant, 0.5).pvalue if discordant else 1.0,
            })
        else:
            nonzero = [value for value in differences if value != 0]
            summary["wilcoxon_p"] = float(wilcoxon(nonzero).pvalue) if nonzero else 1.0
        result["metrics"][metric] = summary
    result["clear_dynamic_effect"] = tier == "formal" and any(
        result["metrics"][metric]["source_stratified_paired_bootstrap_95ci"][0] > 0
        for metric in ("first_step_escape", "first_step_contraction")
    )
    return result


def audit_arm(
    row: dict[str, Any], state: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    label = f"{row.get('state_id')}::{row.get('condition')}"
    for key in ("model", "structure", "source", "trigger_revision"):
        expected_key = "revision" if key == "trigger_revision" else key
        if row.get(key) != state.get(expected_key):
            errors.append(f"{label}: {key} mismatch")
    rounds = row.get("rounds", [])
    if not rounds:
        errors.append(f"{label}: no rounds")
        return row
    expected_revisions = list(range(int(state["revision"]) + 1, int(state["revision"]) + 1 + len(rounds)))
    if [step.get("revision") for step in rounds] != expected_revisions:
        errors.append(f"{label}: nonsequential revisions")
    prior_hashes = {hashlib.sha256(text.encode()).hexdigest() for text in state["prior_texts"]}
    previous_energy = float(state["current_violation_energy"])
    for step in rounds:
        verified = protocol.verify(state["structure"], step["text"], state["item"]["targets"])
        energy = float(sum(verified["violation_vector"]))
        if not close(verified, step["verifier"]):
            errors.append(f"{label}: verifier mismatch at revision {step['revision']}")
        if not close(energy, step["violation_energy"]):
            errors.append(f"{label}: energy mismatch at revision {step['revision']}")
        if bool(verified["joint_success"]) != bool(step["joint_success"]):
            errors.append(f"{label}: success mismatch at revision {step['revision']}")
        text_hash = hashlib.sha256(step["text"].encode()).hexdigest()
        if bool(step["escapes_prior_recurrence"]) != (text_hash not in prior_hashes):
            errors.append(f"{label}: recurrence flag mismatch at revision {step['revision']}")
        # Verify the reconstructed energy above, then audit the strict comparison
        # on its serialized value. Python versions differ in float summation at
        # equality; those rounding differences must not relabel frozen outcomes.
        recorded_energy = float(step["violation_energy"])
        if bool(step["contracts_violation_energy"]) != (recorded_energy < previous_energy):
            errors.append(f"{label}: contraction flag mismatch at revision {step['revision']}")
        prior_hashes.add(text_hash)
        previous_energy = recorded_energy
    if bool(row["first_step_escape"]) != bool(rounds[0]["escapes_prior_recurrence"]):
        errors.append(f"{label}: first_step_escape mismatch")
    if bool(row["first_step_contraction"]) != bool(rounds[0]["contracts_violation_energy"]):
        errors.append(f"{label}: first_step_contraction mismatch")
    if bool(row["final_joint"]) != bool(rounds[-1]["joint_success"]):
        errors.append(f"{label}: final_joint mismatch")
    if rounds[-1]["joint_success"]:
        if any(step["joint_success"] for step in rounds[:-1]):
            errors.append(f"{label}: generation continued after success")
    elif rounds[-1]["revision"] != 8:
        errors.append(f"{label}: unsuccessful arm stopped before revision 8")
    return arm_features(row, float(state["current_violation_energy"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    structures = ("c05", "c07", "c09", "c10", "c12")
    models = ("llama31_8b", "glm4_9b")
    errors: list[str] = []
    pairs_by_cell: dict[tuple[str, str], list[dict[str, dict[str, Any]]]] = {}
    inputs: dict[str, str] = {}
    support = []

    for structure in structures:
        experiment_dir = args.experiments_root / f"{structure}_history_reset_v1"
        states_path = experiment_dir / "states.jsonl"
        states = read_jsonl(states_path)
        state_map = {state["state_id"]: state for state in states}
        if len(state_map) != len(states):
            errors.append(f"{structure}: duplicate state ids")
        inputs[str(states_path.relative_to(ROOT))] = sha256(states_path)
        for model in models:
            expected_states = {key: value for key, value in state_map.items() if value["model"] == model}
            cases_path = experiment_dir / model / "cases.jsonl"
            manifest_path = experiment_dir / model / "runtime_manifest.json"
            rows = read_jsonl(cases_path)
            inputs[str(cases_path.relative_to(ROOT))] = sha256(cases_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("states") != len(expected_states) or manifest.get("arms") != len(rows):
                errors.append(f"{model}/{structure}: runtime manifest counts mismatch")
            if manifest.get("states_sha256") != sha256(states_path):
                errors.append(f"{model}/{structure}: states hash mismatch")
            if manifest.get("cases_sha256") != sha256(cases_path):
                errors.append(f"{model}/{structure}: cases hash mismatch")
            grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for raw_row in rows:
                state_id = raw_row.get("state_id")
                if state_id not in expected_states:
                    errors.append(f"{model}/{structure}: unexpected state {state_id}")
                    continue
                condition = raw_row.get("condition")
                if condition in grouped[state_id]:
                    errors.append(f"{model}/{structure}: duplicate {state_id}/{condition}")
                    continue
                grouped[state_id][condition] = audit_arm(raw_row, expected_states[state_id], errors)
            missing = set(expected_states) - set(grouped)
            if missing:
                errors.append(f"{model}/{structure}: {len(missing)} missing states")
            incomplete = [state_id for state_id, pair in grouped.items() if set(pair) != {"full_history", "history_reset"}]
            if incomplete:
                errors.append(f"{model}/{structure}: {len(incomplete)} incomplete pairs")
            pairs = [pair for pair in grouped.values() if set(pair) == {"full_history", "history_reset"}]
            pairs_by_cell[(model, structure)] = pairs
            support.append({
                "model": model,
                "structure": structure,
                "pairs": len(pairs),
                "evidence_tier": evidence_tier(structure, len(pairs)),
            })

    if errors:
        audit = {"verdict": "FAIL", "errors": errors, "input_sha256": inputs}
        (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"audit failed with {len(errors)} errors")

    cells = []
    for index, ((model, structure), pairs) in enumerate(sorted(pairs_by_cell.items())):
        cells.append(summarize_cell(model, structure, pairs, args.iterations, SEED + index * 100_000))

    formal_cells = [cell for cell in cells if cell["evidence_tier"] == "formal"]
    clear_models = sorted({cell["model"] for cell in formal_cells if cell["clear_dynamic_effect"]})
    clear_structures = sorted({cell["structure"] for cell in formal_cells if cell["clear_dynamic_effect"]})
    supported_structures_by_model = {
        model: sorted(cell["structure"] for cell in formal_cells if cell["model"] == model)
        for model in models
    }
    common_formal_structures = sorted(set(supported_structures_by_model[models[0]]) & set(supported_structures_by_model[models[1]]))
    interaction_estimable = len(common_formal_structures) >= 2
    extension_gate = {
        "pair_integrity_pass": True,
        "clear_dynamic_effect_models": clear_models,
        "clear_dynamic_effect_structures": clear_structures,
        "requires_both_current_models": set(clear_models) == set(models),
        "requires_at_least_two_main_structures": len(set(clear_structures) & MAIN_STRUCTURES) >= 2,
    }
    extension_gate["pass"] = (
        extension_gate["requires_both_current_models"]
        and extension_gate["requires_at_least_two_main_structures"]
    )
    extension_gate["decision"] = (
        "Freeze a targeted Gemma2/Qwen3/Falcon confirmation design before launch."
        if extension_gate["pass"]
        else "Do not launch a second reset batch; retain the current scoped finding."
    )

    analysis = {
        "schema_version": 1,
        "estimand": "within-state history-reset effect among recurrence-triggered states",
        "not_estimand": "average treatment effect over all parent trajectories",
        "seed": SEED,
        "bootstrap_iterations": args.iterations,
        "bootstrap_unit": "paired trigger state, resampled within source",
        "formal_cell_minimum_n": FORMAL_N,
        "audit": {"verdict": "PASS", "errors": [], "input_sha256": inputs},
        "support_matrix": support,
        "cells": cells,
        "interaction": {
            "formally_estimable": interaction_estimable,
            "common_formal_structures": common_formal_structures,
            "reason": (
                "At least two structures have formal support in both models."
                if interaction_estimable
                else "Fewer than two structures have n>=30 recurrence-triggered pairs in both models; do not force a model-by-structure interaction test."
            ),
        },
        "targeted_extension_gate": extension_gate,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "audit.json").write_text(json.dumps(analysis["audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.output_dir / "cell_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model", "structure", "pairs", "evidence_tier", "metric",
            "full_history_mean", "history_reset_mean", "paired_difference",
            "ci_low", "ci_high", "p_value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for metric, summary in cell["metrics"].items():
                writer.writerow({
                    "model": cell["model"],
                    "structure": cell["structure"],
                    "pairs": cell["pairs"],
                    "evidence_tier": cell["evidence_tier"],
                    "metric": metric,
                    "full_history_mean": summary["full_history_mean"],
                    "history_reset_mean": summary["history_reset_mean"],
                    "paired_difference": summary["paired_difference"],
                    "ci_low": summary["source_stratified_paired_bootstrap_95ci"][0],
                    "ci_high": summary["source_stratified_paired_bootstrap_95ci"][1],
                    "p_value": summary.get("exact_mcnemar_p", summary.get("wilcoxon_p")),
                })

    report = [
        "# Cross-Constraint State-Matched History Reset",
        "",
        "Estimand: within-state history-reset effect among recurrence-triggered states.",
        "All intervals use source-stratified paired-state bootstrap resampling.",
        "",
        "## Support matrix",
        "",
        "| Model | Structure | Pairs | Tier |",
        "|---|---|---:|---|",
    ]
    for row in support:
        report.append(f"| {row['model']} | {row['structure']} | {row['pairs']} | {row['evidence_tier']} |")
    report.extend([
        "",
        "## Paired effects",
        "",
        "Positive deltas favor history reset. Energy reduction is measured in violation-energy units.",
        "",
        "| Model | Structure | N | Tier | Escape delta | Contraction delta | Avoid later recurrence delta | Final joint delta |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for cell in cells:
        metrics = cell["metrics"]
        def show(metric: str) -> str:
            item = metrics[metric]
            low, high = item["source_stratified_paired_bootstrap_95ci"]
            return f"{item['paired_difference']:+.1%} [{low:+.1%}, {high:+.1%}]"
        report.append(
            f"| {cell['model']} | {cell['structure']} | {cell['pairs']} | {cell['evidence_tier']} | "
            f"{show('first_step_escape')} | {show('first_step_contraction')} | "
            f"{show('avoids_later_recurrence')} | {show('final_joint')} |"
        )
    report.extend([
        "",
        "## Interaction boundary",
        "",
        analysis["interaction"]["reason"],
        "",
        "## Targeted-extension gate",
        "",
        analysis["targeted_extension_gate"]["decision"],
        "",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "audit": "PASS",
        "pairs": sum(row["pairs"] for row in support),
        "formal_cells": len(formal_cells),
        "interaction_estimable": interaction_estimable,
        "extension_gate": extension_gate,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
