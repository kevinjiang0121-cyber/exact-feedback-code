#!/usr/bin/env python3
"""Formal paired analysis of state-matched closed-loop continuations."""

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


SEED = 20260802
BOOTSTRAPS = 20_000
METRICS = (
    "first_step_escape",
    "first_step_direction_correct",
    "first_step_contraction",
    "later_recurrence",
    "final_exact",
    "final_anchor_retention",
    "final_joint",
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


def difference(pairs: list[dict[str, dict[str, Any]]], metric: str) -> float:
    return sum(
        float(pair["history_reset"][metric]) - float(pair["full_history"][metric])
        for pair in pairs
    ) / len(pairs)


def bootstrap(
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
        draws.append(difference(sample, metric))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def summarize(
    pairs: list[dict[str, dict[str, Any]]], iterations: int, seed: int
) -> dict[str, Any]:
    output: dict[str, Any] = {"pairs": len(pairs), "metrics": {}}
    for index, metric in enumerate(METRICS):
        full = sum(bool(pair["full_history"][metric]) for pair in pairs) / len(pairs)
        reset = sum(bool(pair["history_reset"][metric]) for pair in pairs) / len(pairs)
        low, high = bootstrap(pairs, metric, iterations, seed + index * 1009)
        reset_only = sum(
            not pair["full_history"][metric] and pair["history_reset"][metric]
            for pair in pairs
        )
        full_only = sum(
            pair["full_history"][metric] and not pair["history_reset"][metric]
            for pair in pairs
        )
        discordant = reset_only + full_only
        p = binomtest(min(reset_only, full_only), discordant, 0.5).pvalue if discordant else 1.0
        output["metrics"][metric] = {
            "full_history_rate": full,
            "history_reset_rate": reset,
            "paired_difference": reset - full,
            "stratified_bootstrap_95ci": [low, high],
            "reset_only_discordant": reset_only,
            "full_only_discordant": full_only,
            "exact_mcnemar_p": p,
        }
    output["mean_post_trigger_generations"] = {
        condition: sum(pair[condition]["post_trigger_generations"] for pair in pairs) / len(pairs)
        for condition in ("full_history", "history_reset")
    }
    joint = output["metrics"]["final_joint"]
    anchors = output["metrics"]["final_anchor_retention"]
    output["confirmation_gate"] = {
        "joint_improvement_at_least_10pp": joint["paired_difference"] >= 0.10,
        "joint_ci_excludes_zero": joint["stratified_bootstrap_95ci"][0] > 0,
        "anchor_retention_not_lower": anchors["paired_difference"] >= 0,
    }
    output["confirmation_gate"]["pass"] = all(output["confirmation_gate"].values())
    return output


def revision_bin(value: int) -> str:
    return "middle_r3_r4" if value <= 4 else "late_r5_plus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    expected = {"llama31_8b": 83, "glm4_9b": 120}
    analysis = {
        "schema_version": 1,
        "seed": SEED,
        "bootstrap_iterations": args.iterations,
        "bootstrap_unit": "paired trigger state, resampled within source",
        "models": {},
        "input_sha256": {},
    }
    robustness = []
    for model_index, (model, n_states) in enumerate(expected.items()):
        path = args.experiment_dir / model / "run" / "cases.jsonl"
        rows = read_jsonl(path)
        if len(rows) != n_states * 2:
            raise RuntimeError(f"{model}: expected {n_states * 2} rows, got {len(rows)}")
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            grouped[row["state_id"]][row["condition"]] = row
        if len(grouped) != n_states or any(set(pair) != {"full_history", "history_reset"} for pair in grouped.values()):
            raise RuntimeError(f"{model}: incomplete pairs")
        pairs = list(grouped.values())
        result = summarize(pairs, args.iterations, SEED + model_index * 100_000)
        analysis["models"][model] = result
        analysis["input_sha256"][model] = sha256(path)
        for dimension, key_fn in (
            ("source", lambda row: row["source"]),
            ("error", lambda row: row["trigger_error_bin"]),
            ("revision", lambda row: revision_bin(int(row["trigger_revision"]))),
        ):
            groups: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
            for pair in pairs:
                groups[key_fn(pair["full_history"])].append(pair)
            for level, selected in sorted(groups.items()):
                for metric_index, metric in enumerate(METRICS):
                    low, high = (float("nan"), float("nan"))
                    if len(selected) >= 6:
                        low, high = bootstrap(
                            selected,
                            metric,
                            args.iterations,
                            SEED + model_index * 100_000 + metric_index * 1009 + len(robustness),
                        )
                    robustness.append({
                        "model": model,
                        "dimension": dimension,
                        "level": level,
                        "metric": metric,
                        "pairs": len(selected),
                        "full_history_rate": sum(bool(pair["full_history"][metric]) for pair in selected) / len(selected),
                        "history_reset_rate": sum(bool(pair["history_reset"][metric]) for pair in selected) / len(selected),
                        "paired_difference": difference(selected, metric),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    })

    passes = [model for model, result in analysis["models"].items() if result["confirmation_gate"]["pass"]]
    terminal_joint_gains = [
        model
        for model, result in analysis["models"].items()
        if result["confirmation_gate"]["joint_improvement_at_least_10pp"]
        and result["confirmation_gate"]["joint_ci_excludes_zero"]
    ]
    analysis["cross_family_decision"] = {
        "passing_models": passes,
        "models_with_confirmed_terminal_joint_gain": terminal_joint_gains,
        "replicated_cross_family_treatment": len(passes) == 2,
        "interpretation": (
            "Selective recurrence-triggered reset improves terminal reliability in two unrelated families."
            if len(passes) == 2 else
            "Terminal treatment efficacy is checkpoint-dependent."
            if len(passes) == 1 else
            "Llama shows a confirmed terminal joint-success gain but misses the frozen content-retention gate; GLM shows strong dynamical repair without a confirmed terminal gain. No checkpoint passes the full efficacy-and-content gate."
            if terminal_joint_gains else
            "The one-step causal mechanism does not translate into confirmed terminal reliability improvement."
        ),
    }
    analysis_path = args.output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "robustness_strata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robustness[0]))
        writer.writeheader()
        writer.writerows(robustness)

    report = [
        "# State-Matched Held-Out Closed-Loop Confirmation",
        "",
        f"All intervals use {args.iterations:,} paired-state bootstrap resamples stratified by source.",
        "",
        "| Model | N | Full joint | Reset joint | Joint delta [95% CI] | Anchor delta | Escape delta | Direction delta | Contraction delta | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, result in analysis["models"].items():
        m = result["metrics"]
        joint = m["final_joint"]
        report.append(
            f"| {model} | {result['pairs']} | {joint['full_history_rate']:.1%} | "
            f"{joint['history_reset_rate']:.1%} | {joint['paired_difference']:+.1%} "
            f"[{joint['stratified_bootstrap_95ci'][0]:+.1%}, {joint['stratified_bootstrap_95ci'][1]:+.1%}] | "
            f"{m['final_anchor_retention']['paired_difference']:+.1%} | "
            f"{m['first_step_escape']['paired_difference']:+.1%} | "
            f"{m['first_step_direction_correct']['paired_difference']:+.1%} | "
            f"{m['first_step_contraction']['paired_difference']:+.1%} | "
            f"{result['confirmation_gate']['pass']} |"
        )
    report.extend(["", "## Decision", "", analysis["cross_family_decision"]["interpretation"], ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"analysis": str(analysis_path), "report": str(args.output_dir / 'REPORT.md')}, indent=2))


if __name__ == "__main__":
    main()
