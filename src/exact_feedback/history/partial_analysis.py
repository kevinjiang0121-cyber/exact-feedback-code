#!/usr/bin/env python3
"""Audit and analyze the frozen GLM partial-history matched pilot.

The unit is a deep recurrence-triggered state.  All effects are within-state
contrasts.  Bootstrap samples are drawn within constraint-by-source strata and
the macro estimand gives each of c05, c10, and c12 equal weight.
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

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
import exact_feedback.common.structured_protocol as protocol  # noqa: E402


SEED = 20260807
BOOTSTRAPS = 20_000
STRUCTURES = ("c05", "c10", "c12")
CONDITIONS = ("full_history", "history_reset", "tail_1", "tail_2")
PRIMARY_CONTRASTS = (("tail_1", "full_history"), ("tail_2", "full_history"))
SECONDARY_CONTRASTS = (
    ("tail_1", "history_reset"),
    ("tail_2", "history_reset"),
    ("tail_1", "tail_2"),
)
METRICS = (
    "first_step_escape",
    "first_step_contraction",
    "avoids_later_recurrence",
    "final_joint",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerance
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return all(close(a[key], b[key], tolerance) for key in a)
    return a == b


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def arm_features(row: dict[str, Any]) -> dict[str, Any]:
    rounds = row["rounds"]
    later_recurrence = any(not bool(step["escapes_prior_recurrence"]) for step in rounds[1:])
    return {
        **row,
        "first_step_escape": bool(row["first_step_escape"]),
        "first_step_contraction": bool(row["first_step_contraction"]),
        "avoids_later_recurrence": not later_recurrence,
        "final_joint": bool(row["final_joint"]),
    }


def audit_arm(row: dict[str, Any], state: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    label = f"{row.get('state_id')}::{row.get('condition')}"
    expected = {
        "model": state["model"],
        "structure": state["structure"],
        "source": state["source"],
        "trigger_revision": state["revision"],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            errors.append(f"{label}: {key} mismatch")

    rounds = row.get("rounds", [])
    if not rounds:
        errors.append(f"{label}: no rounds")
        return arm_features({**row, "rounds": [{}]})

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
        escaped = text_hash not in prior_hashes
        if bool(step["escapes_prior_recurrence"]) != escaped:
            errors.append(f"{label}: recurrence flag mismatch at revision {step['revision']}")
        # Recomputed energy is verified separately; use recorded float values
        # for the strict historical comparison at equality.
        contracted = float(step["violation_energy"]) < previous_energy
        if bool(step["contracts_violation_energy"]) != contracted:
            errors.append(f"{label}: contraction flag mismatch at revision {step['revision']}")
        prior_hashes.add(text_hash)
        previous_energy = float(step["violation_energy"])

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
    return arm_features(row)


def paired_values(
    states: list[dict[str, dict[str, Any]]], treatment: str, control: str, metric: str
) -> list[float]:
    return [float(state[treatment][metric]) - float(state[control][metric]) for state in states]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def exact_paired_binary(
    states: list[dict[str, dict[str, Any]]], treatment: str, control: str, metric: str
) -> dict[str, Any]:
    treatment_only = sum(
        not bool(state[control][metric]) and bool(state[treatment][metric]) for state in states
    )
    control_only = sum(
        bool(state[control][metric]) and not bool(state[treatment][metric]) for state in states
    )
    discordant = treatment_only + control_only
    p_value = binomtest(min(treatment_only, control_only), discordant, 0.5).pvalue if discordant else 1.0
    return {
        "treatment_only_discordant": treatment_only,
        "control_only_discordant": control_only,
        "discordant": discordant,
        "exact_mcnemar_p": float(p_value),
    }


def stratified_bootstrap(
    states: list[dict[str, dict[str, Any]]],
    treatment: str,
    control: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_cell_source: dict[tuple[str, str], list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for state in states:
        anchor = state[control]
        by_cell_source[(anchor["structure"], anchor["source"])].append(state)

    rng = random.Random(seed)
    macro_draws: list[float] = []
    cell_draws: dict[str, list[float]] = {structure: [] for structure in STRUCTURES}
    for _ in range(iterations):
        structure_effects = []
        for structure in STRUCTURES:
            sampled: list[dict[str, dict[str, Any]]] = []
            for source in sorted({source for cell, source in by_cell_source if cell == structure}):
                stratum = by_cell_source[(structure, source)]
                sampled.extend(rng.choice(stratum) for _ in range(len(stratum)))
            effect = mean(paired_values(sampled, treatment, control, metric))
            cell_draws[structure].append(effect)
            structure_effects.append(effect)
        macro_draws.append(mean(structure_effects))

    return {
        "macro_95ci": [quantile(macro_draws, 0.025), quantile(macro_draws, 0.975)],
        "cell_95ci": {
            structure: [quantile(draws, 0.025), quantile(draws, 0.975)]
            for structure, draws in cell_draws.items()
        },
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def summarize_contrast(
    states: list[dict[str, dict[str, Any]]],
    treatment: str,
    control: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "treatment": treatment,
        "control": control,
        "n": len(states),
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        cells: dict[str, Any] = {}
        for structure in STRUCTURES:
            subset = [state for state in states if state[control]["structure"] == structure]
            treatment_mean = mean([float(state[treatment][metric]) for state in subset])
            control_mean = mean([float(state[control][metric]) for state in subset])
            cells[structure] = {
                "n": len(subset),
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "paired_difference": treatment_mean - control_mean,
            }
        macro_treatment = mean([cells[structure]["treatment_mean"] for structure in STRUCTURES])
        macro_control = mean([cells[structure]["control_mean"] for structure in STRUCTURES])
        bootstrap = stratified_bootstrap(
            states, treatment, control, metric, iterations, seed + metric_index * 100_003
        )
        for structure in STRUCTURES:
            cells[structure]["source_stratified_paired_bootstrap_95ci"] = bootstrap["cell_95ci"][structure]
        result["metrics"][metric] = {
            "macro_treatment_mean": macro_treatment,
            "macro_control_mean": macro_control,
            "macro_paired_difference": macro_treatment - macro_control,
            "source_stratified_equal_cell_macro_bootstrap_95ci": bootstrap["macro_95ci"],
            "exact_paired_binary": exact_paired_binary(states, treatment, control, metric),
            "cells": cells,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT / "experiments" / "glm_partial_reset_matched_pilot_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    states_path = args.pilot_root / "states.jsonl"
    new_cases_path = args.pilot_root / "glm4_9b" / "cases.jsonl"
    runtime_manifest_path = args.pilot_root / "glm4_9b" / "runtime_manifest.json"
    states = read_jsonl(states_path)
    state_map = {state["state_id"]: state for state in states}
    errors: list[str] = []
    inputs: dict[str, str] = {
        str(states_path.relative_to(args.pilot_root.resolve())): sha256(states_path),
        str(new_cases_path.relative_to(args.pilot_root.resolve())): sha256(new_cases_path),
        str(runtime_manifest_path.relative_to(args.pilot_root.resolve())): sha256(runtime_manifest_path),
    }

    if len(states) != 60 or len(state_map) != 60:
        errors.append(f"expected 60 unique states, found {len(states)} rows/{len(state_map)} unique")
    if set(state["structure"] for state in states) != set(STRUCTURES):
        errors.append("unexpected structure support")
    for structure in STRUCTURES:
        if sum(state["structure"] == structure for state in states) != 20:
            errors.append(f"{structure}: expected 20 states")

    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    reused_baselines = runtime_manifest.get("arms") == 120
    if runtime_manifest.get("states") != 60 or runtime_manifest.get("arms") not in (120,240):
        errors.append("four-arm runtime manifest count mismatch")
    if runtime_manifest.get("states_sha256") != sha256(states_path):
        errors.append("new-arm runtime manifest states hash mismatch")
    if runtime_manifest.get("cases_sha256") != sha256(new_cases_path):
        errors.append("new-arm runtime manifest cases hash mismatch")

    all_rows = read_jsonl(new_cases_path)
    if not reused_baselines and runtime_manifest.get("conditions") != list(CONDITIONS):
        errors.append("runtime manifest does not contain the frozen four-arm design")
    if reused_baselines:
        for structure in STRUCTURES:
            old_cases_path = args.pilot_root.parent / f"{structure}_history_reset_v1" / "glm4_9b/cases.jsonl"
            old_manifest_path = old_cases_path.parent / "runtime_manifest.json"
            old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
            if old_manifest.get("cases_sha256") != sha256(old_cases_path):
                errors.append(f"{structure}: reused cases hash mismatch")
            inputs[str(old_cases_path.relative_to(args.pilot_root.parent))] = sha256(old_cases_path)
            inputs[str(old_manifest_path.relative_to(args.pilot_root.parent))] = sha256(old_manifest_path)
            selected_ids={state["state_id"] for state in states if state["structure"]==structure}
            all_rows.extend(row for row in read_jsonl(old_cases_path) if row.get("state_id") in selected_ids)
    if len(all_rows) != len(states) * len(CONDITIONS):
        errors.append(
            f"expected {len(states) * len(CONDITIONS)} four-arm rows, found {len(all_rows)}"
        )

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw_row in all_rows:
        state_id = raw_row.get("state_id")
        condition = raw_row.get("condition")
        if state_id not in state_map:
            errors.append(f"unexpected state {state_id}")
            continue
        if condition not in CONDITIONS:
            errors.append(f"{state_id}: unexpected condition {condition}")
            continue
        if condition in grouped[state_id]:
            errors.append(f"{state_id}: duplicate condition {condition}")
            continue
        grouped[state_id][condition] = audit_arm(raw_row, state_map[state_id], errors)

    missing_states = set(state_map) - set(grouped)
    if missing_states:
        errors.append(f"missing {len(missing_states)} states")
    incomplete = {
        state_id: sorted(set(CONDITIONS) - set(arms))
        for state_id, arms in grouped.items()
        if set(arms) != set(CONDITIONS)
    }
    if incomplete:
        errors.append(f"{len(incomplete)} states have incomplete four-arm support")

    audit = {
        "verdict": "FAIL" if errors else "PASS",
        "errors": errors,
        "states": len(states),
        "expected_arms": len(states) * len(CONDITIONS),
        "audited_arms": len(all_rows),
        "input_sha256": inputs,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"audit failed with {len(errors)} errors; see {args.output_dir / 'audit.json'}")

    complete_states = [grouped[state_id] for state_id in sorted(state_map)]
    contrasts: dict[str, Any] = {}
    for contrast_index, (treatment, control) in enumerate(PRIMARY_CONTRASTS + SECONDARY_CONTRASTS):
        key = f"{treatment}_minus_{control}"
        contrasts[key] = summarize_contrast(
            complete_states,
            treatment,
            control,
            args.iterations,
            SEED + contrast_index * 1_000_003,
        )

    raw_primary_p = {
        f"{treatment}_minus_{control}": contrasts[f"{treatment}_minus_{control}"]["metrics"]
        ["first_step_escape"]["exact_paired_binary"]["exact_mcnemar_p"]
        for treatment, control in PRIMARY_CONTRASTS
    }
    adjusted_primary_p = holm_adjust(raw_primary_p)
    condition_gates: dict[str, Any] = {}
    for treatment, control in PRIMARY_CONTRASTS:
        key = f"{treatment}_minus_{control}"
        escape = contrasts[key]["metrics"]["first_step_escape"]
        contraction = contrasts[key]["metrics"]["first_step_contraction"]
        positive_cells = sum(
            escape["cells"][structure]["paired_difference"] > 0 for structure in STRUCTURES
        )
        condition_gates[treatment] = {
            "contrast": key,
            "positive_macro_escape": escape["macro_paired_difference"] > 0,
            "escape_ci_excludes_zero_positive": escape[
                "source_stratified_equal_cell_macro_bootstrap_95ci"
            ][0]
            > 0,
            "holm_adjusted_escape_p": adjusted_primary_p[key],
            "holm_adjusted_escape_p_below_0_05": adjusted_primary_p[key] < 0.05,
            "positive_escape_cells": positive_cells,
            "positive_escape_in_at_least_two_cells": positive_cells >= 2,
            "macro_contraction_difference": contraction["macro_paired_difference"],
            "nonnegative_macro_contraction": contraction["macro_paired_difference"] >= 0,
        }
        condition_gates[treatment]["pass"] = all(
            (
                condition_gates[treatment]["positive_macro_escape"],
                condition_gates[treatment]["escape_ci_excludes_zero_positive"],
                condition_gates[treatment]["holm_adjusted_escape_p_below_0_05"],
                condition_gates[treatment]["positive_escape_in_at_least_two_cells"],
                condition_gates[treatment]["nonnegative_macro_contraction"],
            )
        )

    passing_conditions = [condition for condition, gate in condition_gates.items() if gate["pass"]]
    gate = {
        "pass": bool(passing_conditions),
        "passing_partial_conditions": passing_conditions,
        "raw_primary_escape_p": raw_primary_p,
        "holm_adjusted_primary_escape_p": adjusted_primary_p,
        "condition_gates": condition_gates,
        "decision": (
            "Partial-history mechanism gate passed. Report the bounded GLM causal result; do not expand without a new decision."
            if passing_conditions
            else "Partial-history mechanism gate failed. Stop this mechanism branch and retain temporal crossover as endogenous heterogeneity."
        ),
    }

    analysis = {
        "schema_version": 1,
        "estimand": "within-state partial-history effect among GLM deep-recurrence-triggered states",
        "not_estimand": "effect of trigger revision or average effect over all trajectories",
        "seed": SEED,
        "bootstrap_iterations": args.iterations,
        "bootstrap_design": "paired state resampling within constraint-by-source strata; equal-cell macro over c05/c10/c12",
        "audit": audit,
        "contrasts": contrasts,
        "frozen_gate": gate,
        "interpretation_boundary": (
            "A passing gate identifies an effect of removing older context while retaining recent state information. "
            "It does not distinguish semantic inertia from context length, position, or turn structure."
        ),
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "contrast_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "contrast", "metric", "level", "n", "treatment_mean", "control_mean",
            "paired_difference", "ci_low", "ci_high", "exact_mcnemar_p",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for contrast_name, contrast in contrasts.items():
            for metric, summary in contrast["metrics"].items():
                low, high = summary["source_stratified_equal_cell_macro_bootstrap_95ci"]
                writer.writerow({
                    "contrast": contrast_name,
                    "metric": metric,
                    "level": "equal_cell_macro",
                    "n": contrast["n"],
                    "treatment_mean": summary["macro_treatment_mean"],
                    "control_mean": summary["macro_control_mean"],
                    "paired_difference": summary["macro_paired_difference"],
                    "ci_low": low,
                    "ci_high": high,
                    "exact_mcnemar_p": summary["exact_paired_binary"]["exact_mcnemar_p"],
                })
                for structure in STRUCTURES:
                    cell = summary["cells"][structure]
                    cell_low, cell_high = cell["source_stratified_paired_bootstrap_95ci"]
                    writer.writerow({
                        "contrast": contrast_name,
                        "metric": metric,
                        "level": structure,
                        "n": cell["n"],
                        "treatment_mean": cell["treatment_mean"],
                        "control_mean": cell["control_mean"],
                        "paired_difference": cell["paired_difference"],
                        "ci_low": cell_low,
                        "ci_high": cell_high,
                        "exact_mcnemar_p": "",
                    })

    def effect(contrast_name: str, metric: str) -> str:
        item = contrasts[contrast_name]["metrics"][metric]
        low, high = item["source_stratified_equal_cell_macro_bootstrap_95ci"]
        return f"{item['macro_paired_difference']:+.1%} [{low:+.1%}, {high:+.1%}]"

    report = [
        "# GLM Partial-History Matched Pilot",
        "",
        "Audit: **PASS** (60 states, 240 four-condition arms, deterministic verifier recomputed).",
        "",
        "Estimand: within-state partial-history effect among GLM states first observed in deep recurrence (revision >= 4).",
        "Intervals: 20,000 constraint-by-source-stratified paired bootstrap draws; macro averages c05/c10/c12 equally.",
        "",
        "## Primary contrasts",
        "",
        "| Contrast | Escape | Contraction | Avoid later recurrence | Final capture | Holm escape p | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for treatment, control in PRIMARY_CONTRASTS:
        key = f"{treatment}_minus_{control}"
        report.append(
            f"| {treatment} - {control} | {effect(key, 'first_step_escape')} | "
            f"{effect(key, 'first_step_contraction')} | {effect(key, 'avoids_later_recurrence')} | "
            f"{effect(key, 'final_joint')} | {adjusted_primary_p[key]:.4g} | "
            f"{'PASS' if condition_gates[treatment]['pass'] else 'FAIL'} |"
        )
    report.extend([
        "",
        "## Constraint-level primary escape effects",
        "",
        "| Contrast | c05 | c10 | c12 |",
        "|---|---:|---:|---:|",
    ])
    for treatment, control in PRIMARY_CONTRASTS:
        key = f"{treatment}_minus_{control}"
        cells = contrasts[key]["metrics"]["first_step_escape"]["cells"]
        formatted = []
        for structure in STRUCTURES:
            low, high = cells[structure]["source_stratified_paired_bootstrap_95ci"]
            formatted.append(f"{cells[structure]['paired_difference']:+.1%} [{low:+.1%}, {high:+.1%}]")
        report.append(f"| {treatment} - {control} | " + " | ".join(formatted) + " |")
    report.extend([
        "",
        "## Frozen gate",
        "",
        f"**{'PASS' if gate['pass'] else 'FAIL'}**. {gate['decision']}",
        "",
        "## Interpretation boundary",
        "",
        analysis["interpretation_boundary"],
        "",
        "Secondary window-shape contrasts are stored in `analysis.json` and `contrast_summary.csv`; they do not rescue a failed primary result.",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "audit": audit["verdict"],
        "states": len(complete_states),
        "arms": len(all_rows),
        "gate": gate,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
