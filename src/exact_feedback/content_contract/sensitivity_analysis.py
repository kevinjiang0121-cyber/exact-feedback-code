#!/usr/bin/env python3
"""Recompute the Appendix content-contract sensitivity table.

Each observable is applied separately to the frozen exact-length joint
endpoint. Empty annotation fields are not-applicable and therefore pass the
additional condition. This script intentionally forms no composite endpoint
and performs no confirmatory inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = {
    "llama31_70b_api": "Llama 3.1 70B",
    "qwen3_7_plus_api": "Qwen3.7 Plus",
}

CHECKS = (
    ("complete_entity_retention", "draft_entity_retention", "ge", 1.0),
    ("complete_proper_name_retention", "draft_proper_name_retention", "ge", 1.0),
    ("complete_symbolic_unit_retention", "draft_symbolic_retention", "ge", 1.0),
    ("no_parser_fragment_increase", "fragment_rate_change", "le", 0.0),
    ("no_parser_agreement_decrease", "agreement_rate_change", "ge", 0.0),
    ("no_parser_dangling_increase", "dangling_rate_change", "le", 0.0),
    (
        "no_adjacent_sentence_nli_contradiction_increase",
        "adjacent_contradiction_mean_change",
        "le",
        0.0,
    ),
)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def passes(value: str, direction: str, threshold: float) -> bool:
    if value.strip() == "":
        return True
    number = float(value)
    if direction == "ge":
        return number >= threshold
    if direction == "le":
        return number <= threshold
    raise ValueError(f"unknown direction: {direction}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = {model: [row for row in rows if row["model"] == model] for model in MODELS}
    for model, model_rows in selected.items():
        if len(model_rows) != 480:
            raise ValueError(f"{model}: expected 480 rows, found {len(model_rows)}")
        if len({row["id"] for row in model_rows}) != 480:
            raise ValueError(f"{model}: duplicate or missing case IDs")

    result_rows: list[dict[str, object]] = []
    for name, field, direction, threshold in CHECKS:
        counts: dict[str, int] = {}
        for model, model_rows in selected.items():
            counts[model] = sum(
                parse_bool(row["final_joint"])
                and passes(row[field], direction, threshold)
                for row in model_rows
            )
        llama_rate = 100.0 * counts["llama31_70b_api"] / 480
        qwen_rate = 100.0 * counts["qwen3_7_plus_api"] / 480
        result_rows.append(
            {
                "observable": name,
                "field": field,
                "direction": direction,
                "threshold": threshold,
                "llama_count": counts["llama31_70b_api"],
                "llama_rate_pct": llama_rate,
                "qwen_count": counts["qwen3_7_plus_api"],
                "qwen_rate_pct": qwen_rate,
                "llama_minus_qwen_pp": llama_rate - qwen_rate,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "content_contract_sensitivity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    payload = {
        "status": "post_hoc_descriptive",
        "input": str(args.input),
        "models": MODELS,
        "cases_per_model": 480,
        "missing_fields": "not_applicable_pass",
        "composite_score": False,
        "confirmatory_inference": False,
        "rows": result_rows,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
