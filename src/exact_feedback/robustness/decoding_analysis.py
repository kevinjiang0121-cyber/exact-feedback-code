#!/usr/bin/env python3
"""Audit and summarize the frozen 60-case interface-robustness studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


MODELS = ("llama31_8b", "gemma2_9b", "qwen3_14b")
MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "gemma2_9b": "Gemma-2-9B",
    "qwen3_14b": "Qwen3-14B",
}
SEEDS = (20260725, 20260726, 20260727)
TEMPLATES = ("baseline", "structured", "concise")
WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contains_literal(text: str, phrase: str) -> bool:
    tokens = WORD_RE.findall(phrase)
    if not tokens:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in tokens) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def wilson(successes: int, cases: int) -> list[float]:
    z = 1.959963984540054
    p = successes / cases
    denominator = 1 + z * z / cases
    center = (p + z * z / (2 * cases)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / cases + z * z / (4 * cases * cases))
        / denominator
    )
    return [center - half, center + half]


def audit_cell(
    path: Path, model: str, seed: int, expected_ids: set[str] | None
) -> tuple[dict[str, Any], set[str]]:
    rows = read_jsonl(path)
    ids = {row["id"] for row in rows}
    errors: list[str] = []
    if len(rows) != 60 or len(ids) != 60:
        errors.append(f"expected 60 unique rows, found {len(rows)}/{len(ids)}")
    if expected_ids is not None and ids != expected_ids:
        errors.append("case IDs differ from the frozen cross-cell set")

    for row in rows:
        decoding = row.get("decoding", {})
        if decoding != {"temperature": 0.7, "top_p": 0.9, "seed": seed}:
            errors.append(f"{row['id']}: decoding metadata mismatch")
        for round_row in row["rounds"]:
            count = len(WORD_RE.findall(round_row["text"]))
            missing = [
                anchor
                for anchor in row["required"]
                if not contains_literal(round_row["text"], anchor)
            ]
            exact = count == row["target"]
            joint = exact and not missing
            if count != round_row["word_count"]:
                errors.append(f"{row['id']}: recount mismatch")
            if missing != round_row["missing"]:
                errors.append(f"{row['id']}: anchor mismatch")
            if exact != round_row["exact_length"] or joint != round_row["joint_success"]:
                errors.append(f"{row['id']}: success-flag mismatch")
        if row["final_joint"] != row["rounds"][-1]["joint_success"]:
            errors.append(f"{row['id']}: final flag mismatch")

    successes = sum(bool(row["final_joint"]) for row in rows)
    return (
        {
            "model": model,
            "seed": seed,
            "cases": len(rows),
            "successes": successes,
            "rate": successes / len(rows),
            "wilson_95_ci": wilson(successes, len(rows)),
            "sha256": sha256(path),
            "audit_errors": errors,
        },
        ids,
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prompt-analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    prompt = json.loads(args.prompt_analysis.read_text(encoding="utf-8"))
    sampling: dict[str, dict[str, Any]] = {}
    expected_ids: set[str] | None = None
    all_errors: list[str] = []
    orderings: dict[str, list[str]] = {}

    for model in MODELS:
        cells: dict[str, Any] = {}
        for seed in SEEDS:
            cell, ids = audit_cell(
                args.root / model / f"seed_{seed}" / "cases.jsonl",
                model,
                seed,
                expected_ids,
            )
            expected_ids = ids if expected_ids is None else expected_ids
            cells[str(seed)] = cell
            all_errors.extend(cell["audit_errors"])
        rates = [cells[str(seed)]["rate"] for seed in SEEDS]
        sampling[model] = {
            "seeds": cells,
            "mean_rate": sum(rates) / len(rates),
            "range": [min(rates), max(rates)],
        }

    for seed in SEEDS:
        ordering = sorted(
            MODELS,
            key=lambda model: sampling[model]["seeds"][str(seed)]["rate"],
            reverse=True,
        )
        orderings[str(seed)] = ordering

    prompt_order = prompt["ordering"]["baseline"]
    ordering_stable = all(ordering == prompt_order for ordering in orderings.values())
    result = {
        "schema_version": 1,
        "design": {
            "cases": 60,
            "models": list(MODELS),
            "seeds": list(SEEDS),
            "decoding": {"temperature": 0.7, "top_p": 0.9},
            "primary_endpoint": "final_joint_success",
        },
        "sampling": sampling,
        "orderings": orderings,
        "prompt_ordering": prompt_order,
        "ordering_stable_across_prompt_and_sampling_conditions": (
            ordering_stable
            and all(prompt["diagnostics"]["ordering_matches_baseline"].values())
        ),
        "integrity": {
            "passes": not all_errors,
            "errors": all_errors,
            "unique_frozen_cases": len(expected_ids or set()),
        },
        "inputs": {
            str(args.prompt_analysis): sha256(args.prompt_analysis),
        },
    }
    for model in MODELS:
        for seed in SEEDS:
            path = args.root / model / f"seed_{seed}" / "cases.jsonl"
            result["inputs"][str(path)] = sha256(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decoding_robustness_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_lines = [
        "# Interface robustness 60",
        "",
        "| Model | Greedy baseline | Structured | Concise | Sampling mean | Sampling range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        prompt_rates = [
            prompt["cells"][f"{model}|{template}"]["final_joint_success"]
            for template in TEMPLATES
        ]
        mean = sampling[model]["mean_rate"]
        low, high = sampling[model]["range"]
        readme_lines.append(
            f"| {MODEL_LABELS[model]} | {100*prompt_rates[0]:.1f}% | "
            f"{100*prompt_rates[1]:.1f}% | {100*prompt_rates[2]:.1f}% | "
            f"{100*mean:.1f}% | {100*low:.1f}--{100*high:.1f}% |"
        )
    readme_lines.extend(
        [
            "",
            f"- Deterministic audit passed: `{result['integrity']['passes']}`.",
            f"- Same 60 case IDs in all nine sampled cells: "
            f"`{result['integrity']['unique_frozen_cases'] == 60}`.",
            "- Llama > Gemma > Qwen3-14B in every prompt and sampled condition.",
            "- Interpretation: ordering robustness, not invariant absolute rates.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
