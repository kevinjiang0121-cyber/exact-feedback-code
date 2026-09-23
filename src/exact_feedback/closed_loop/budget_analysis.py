from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path


CAPS = tuple(range(9))
ASSAYS = ("exact_length", "lexical_constraints", "compositional_constraints")

OPEN_MODELS = (
    "llama31_8b",
    "gemma2_9b",
    "glm4_9b",
    "ministral_8b",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
    "qwen3_30b_a3b",
    "granite_3_3_8b",
    "falcon_h1_7b",
)

API_MODELS = (
    "gpt5_6_sol_api",
    "claude_opus5_api",
    "gemini_3_6_flash_api",
    "glm5_2_api",
    "deepseek_v4_flash_api",
    "qwen3_7_plus_api",
    "llama31_70b_api",
)

HISTORICAL_EXACT = {
    "llama31_8b",
    "gemma2_9b",
    "glm4_9b",
    "ministral_8b",
    "qwen3_8b",
    "qwen3_14b",
}

EXTRA_EXACT = {
    "qwen3_1_7b": "qwen3_1_7b/combined480/cases.jsonl",
    "qwen3_4b": "qwen3_4b/combined480/cases.jsonl",
    "qwen3_32b": "qwen3_32b/vllm_pp2_combined480/cases.jsonl",
    "qwen3_30b_a3b": "qwen3_30b_a3b/vllm_pp2_combined480/cases.jsonl",
    "granite_3_3_8b": "granite_3_3_8b/combined480/cases.jsonl",
    "falcon_h1_7b": "falcon_h1_7b/combined480/cases.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def capture_revision(row: dict) -> int | None:
    if row.get("api_error") is not None or row.get("protocol_complete") is False:
        return None
    rounds = row.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return None
    successful = [
        int(item["revision"])
        for item in rounds
        if item.get("joint_success") is True
    ]
    return min(successful) if successful else math.inf


def add_file(
    cells: dict[tuple[str, str, str], list[int | float]],
    audit_inputs: list[dict],
    *,
    panel: str,
    model: str,
    assay: str | None,
    path: Path,
    exclude_ids: set[str] | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    added = 0
    excluded = 0
    domains: dict[str, int] = defaultdict(int)
    for _, row in iter_jsonl(path):
        if exclude_ids and row.get("id") in exclude_ids:
            excluded += 1
            continue
        row_assay = assay or row.get("domain")
        if row_assay not in ASSAYS:
            raise ValueError(f"Unexpected assay {row_assay!r} in {path}")
        revision = capture_revision(row)
        if revision is None:
            excluded += 1
            continue
        cells[(panel, model, row_assay)].append(revision)
        domains[row_assay] += 1
        added += 1
    audit_inputs.append(
        {
            "path": str(path),
            "sha256": sha256(path),
            "evaluable_rows": added,
            "excluded_rows": excluded,
            "assay_rows": dict(sorted(domains.items())),
        }
    )


def load_cells(root: Path, selected_panel: str, frozen: bool = True):
    cells: dict[tuple[str, str, str], list[int | float]] = defaultdict(list)
    audit_inputs: list[dict] = []

    if selected_panel in ("all", "open"):
      for model in OPEN_MODELS:
        if model in HISTORICAL_EXACT:
            for experiment in (
                "human_generation_main120_v1",
                "human_generation_replication120_v1",
                "human_generation_extension240_v1",
            ):
                add_file(
                    cells,
                    audit_inputs,
                    panel="open",
                    model=model,
                    assay="exact_length",
                    path=root / "experiments" / experiment / model / "cases.jsonl",
                )
        else:
            add_file(
                cells,
                audit_inputs,
                panel="open",
                model=model,
                assay="exact_length",
                path=root
                / "experiments/model_generality_combined480_v1"
                / EXTRA_EXACT[model],
            )
        add_file(
            cells,
            audit_inputs,
            panel="open",
            model=model,
            assay=None,
            path=root
            / "experiments/multidomain_full480_open12_v1"
            / model
            / "cases.jsonl",
        )

    if selected_panel in ("all", "api"):
      gemini_exact_path = (
          root
          / "experiments/model_generality_combined480_api_v1"
          / "gemini_3_6_flash_api/gemini_3_6_flash_api/baseline/cases.jsonl"
      )
      common_exact_exclusions = {
          row.get("id")
          for _, row in iter_jsonl(gemini_exact_path)
          if capture_revision(row) is None
      }
      common_ids=None
      for model in API_MODELS:
          path=root/'experiments/model_generality_combined480_api_v1'/model/model/'baseline/cases.jsonl'
          records=[row for _,row in iter_jsonl(path)]
          ids={row['id'] for row in records}
          if len(ids)!=len(records):raise ValueError(f'Duplicate API case IDs: {model}')
          eligible={row['id'] for row in records if capture_revision(row) is not None}
          common_ids=eligible if common_ids is None else common_ids & eligible
      if not common_ids:raise ValueError('No common complete API exact-length cases')
      if frozen and (len(common_exact_exclusions) != 1 or None in common_exact_exclusions or len(common_ids)!=479):
          raise ValueError(
              f"Expected one frozen API exact-length exclusion, found {common_exact_exclusions}"
          )
      for model in API_MODELS:
        path=root/'experiments/model_generality_combined480_api_v1'/model/model/'baseline/cases.jsonl'
        excluded_ids={row['id'] for _,row in iter_jsonl(path)}-common_ids
        add_file(
            cells,
            audit_inputs,
            panel="api",
            model=model,
            assay="exact_length",
            path=root
            / "experiments/model_generality_combined480_api_v1"
            / model
            / model
            / "baseline/cases.jsonl",
            exclude_ids=excluded_ids,
        )
        for assay in ASSAYS[1:]:
            add_file(
                cells,
                audit_inputs,
                panel="api",
                model=model,
                assay=assay,
                path=root
                / "experiments/model_generality_multidomain_api_v1"
                / assay
                / model
                / "baseline/cases.jsonl",
            )
    return cells, audit_inputs


def reversal_count(rates, panel: str, models: tuple[str, ...], cap: int) -> int:
    reversals = 0
    for left, right in combinations(models, 2):
        signs = set()
        for assay in ASSAYS:
            delta = rates[(panel, left, assay, cap)] - rates[(panel, right, assay, cap)]
            if delta > 1e-12:
                signs.add(1)
            elif delta < -1e-12:
                signs.add(-1)
        reversals += signs == {1, -1}
    return reversals


def write_outputs(root: Path, output: Path, selected_panel: str, frozen: bool = True) -> None:
    cells, audit_inputs = load_cells(root, selected_panel, frozen)
    expected = {"open": 12 * 3, "api": 7 * 3, "all": 12 * 3 + 7 * 3}[selected_panel]
    if len(cells) != expected:
        raise ValueError(f"Expected {expected} controller-assay cells, found {len(cells)}")

    rows = []
    rates = {}
    for (panel, model, assay), revisions in sorted(cells.items()):
        if not revisions or (frozen and len(revisions) not in (479, 480)):
            raise ValueError(f"Unexpected evaluable count for {(panel, model, assay)}: {len(revisions)}")
        for cap in CAPS:
            successes = sum(value <= cap for value in revisions)
            rate = successes / len(revisions)
            rates[(panel, model, assay, cap)] = rate
            rows.append(
                {
                    "panel": panel,
                    "model": model,
                    "assay": assay,
                    "revision_cap": cap,
                    "evaluable_cases": len(revisions),
                    "cumulative_successes": successes,
                    "cumulative_success_rate": rate,
                }
            )

    summary = []
    for cap in CAPS:
        values = [row["cumulative_success_rate"] for row in rows if row["revision_cap"] == cap]
        summary.append(
            {
                "revision_cap": cap,
                "cell_min_rate": min(values),
                "cell_median_rate": statistics.median(values),
                "cell_max_rate": max(values),
                "open_reversing_pairs": reversal_count(rates, "open", OPEN_MODELS, cap)
                if selected_panel in ("all", "open")
                else "",
                "open_controller_pairs": len(tuple(combinations(OPEN_MODELS, 2)))
                if selected_panel in ("all", "open")
                else "",
                "api_reversing_pairs": reversal_count(rates, "api", API_MODELS, cap)
                if selected_panel in ("all", "api")
                else "",
                "api_controller_pairs": len(tuple(combinations(API_MODELS, 2)))
                if selected_panel in ("all", "api")
                else "",
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "cell_budget_success.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary_by_budget.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    audit = {
        "status": "PASS",
        "protocol": "docs/REVISION_BUDGET_SENSITIVITY_PROTOCOL_20260826.md",
        "caps": CAPS,
        "selected_panel": selected_panel,
        "controller_assay_cells": len(cells),
        "cell_budget_rows": len(rows),
        "input_files": audit_inputs,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("all", "open", "api"), default="all")
    parser.add_argument("--fresh", action="store_true", help="Use observed common complete cases; do not require the frozen refusal pattern")
    args = parser.parse_args()
    write_outputs(args.root.resolve(), args.output.resolve(), args.panel, frozen=not args.fresh)


if __name__ == "__main__":
    main()
