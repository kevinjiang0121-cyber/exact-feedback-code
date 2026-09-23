#!/usr/bin/env python3
"""Analyze the frozen 3-model x 3-template prompt-robustness experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


MODELS = ("llama31_8b", "gemma2_9b", "qwen3_14b")
TEMPLATES = ("baseline", "structured", "concise")
BOOTSTRAP_SEED = 20260724
BOOTSTRAP_SAMPLES = 20000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(left: dict[str, int], right: dict[str, int], seed: int) -> dict[str, Any]:
    ids = sorted(left)
    if ids != sorted(right):
        raise ValueError("paired conditions have different case IDs")
    observed = sum(left[case_id] - right[case_id] for case_id in ids) / len(ids)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        draws.append(sum(left[case_id] - right[case_id] for case_id in sampled) / len(ids))
    return {
        "difference": observed,
        "bootstrap_95_ci": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "samples": BOOTSTRAP_SAMPLES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    outcomes: dict[str, dict[str, dict[str, int]]] = {}
    cells: dict[str, Any] = {}
    expected_ids: list[str] | None = None
    for model in MODELS:
        outcomes[model] = {}
        for template in TEMPLATES:
            path = args.experiment_root / model / template / "cases.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            if len(rows) != 60:
                raise ValueError(f"{model}/{template} has {len(rows)} cases, expected 60")
            ids = sorted(row["id"] for row in rows)
            if len(set(ids)) != 60:
                raise ValueError(f"{model}/{template} contains duplicate IDs")
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"{model}/{template} case IDs differ from frozen set")
            values = {row["id"]: int(row["final_joint"]) for row in rows}
            outcomes[model][template] = values
            successes = sum(values.values())
            cells[f"{model}|{template}"] = {
                "successes": successes,
                "cases": 60,
                "final_joint_success": successes / 60,
                "wilson_95_ci": wilson(successes, 60),
                "failures": 60 - successes,
            }

    cross_model = {}
    for template_index, template in enumerate(TEMPLATES):
        for left_index, left in enumerate(MODELS):
            for right_index, right in enumerate(MODELS[left_index + 1 :], start=left_index + 1):
                key = f"{template}|{left}_minus_{right}"
                cross_model[key] = paired_bootstrap(
                    outcomes[left][template],
                    outcomes[right][template],
                    BOOTSTRAP_SEED + template_index * 100 + left_index * 10 + right_index,
                )

    cross_template = {}
    for model_index, model in enumerate(MODELS):
        for left_index, left in enumerate(TEMPLATES):
            for right_index, right in enumerate(TEMPLATES[left_index + 1 :], start=left_index + 1):
                key = f"{model}|{left}_minus_{right}"
                cross_template[key] = paired_bootstrap(
                    outcomes[model][left],
                    outcomes[model][right],
                    BOOTSTRAP_SEED + 1000 + model_index * 100 + left_index * 10 + right_index,
                )

    ordering = {
        template: sorted(
            MODELS,
            key=lambda model: (-cells[f"{model}|{template}"]["final_joint_success"], model),
        )
        for template in TEMPLATES
    }
    analysis = {
        "schema_version": 1,
        "design": {
            "models": list(MODELS),
            "templates": list(TEMPLATES),
            "cases_per_cell": 60,
            "primary_endpoint": "final_joint_success",
        },
        "cells": cells,
        "paired_cross_model": cross_model,
        "paired_cross_template": cross_template,
        "ordering": ordering,
        "diagnostics": {
            "every_cell_has_at_least_one_failure": all(cell["failures"] > 0 for cell in cells.values()),
            "ordering_matches_baseline": {
                template: ordering[template] == ordering["baseline"] for template in TEMPLATES
            },
        },
        "interpretation_rule": (
            "Do not select a template. If structured or concise erases or reverses the baseline model ordering, "
            "stop before stochastic decoding and narrow the claim. Otherwise proceed to the frozen decoding test."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt_robustness_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Prompt Robustness 60",
        "",
        "| Model | Baseline | Structured | Concise |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        rates = [cells[f"{model}|{template}"]["final_joint_success"] for template in TEMPLATES]
        lines.append(f"| {model} | {rates[0]:.3f} | {rates[1]:.3f} | {rates[2]:.3f} |")
    lines.extend(
        [
            "",
            f"Baseline ordering: `{ordering['baseline']}`.",
            f"Structured matches baseline: `{ordering['structured'] == ordering['baseline']}`.",
            f"Concise matches baseline: `{ordering['concise'] == ordering['baseline']}`.",
            "",
            "All confidence intervals and paired contrasts are in `prompt_robustness_analysis.json`.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(analysis["diagnostics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
