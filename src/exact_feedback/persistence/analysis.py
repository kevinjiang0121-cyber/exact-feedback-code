#!/usr/bin/env python3
"""Audit and summarize the frozen revision-32 persistence test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)
MODELS = (
    "llama31_8b",
    "gemma2_9b",
    "glm4_9b",
    "ministral_8b",
    "qwen3_8b",
    "qwen3_14b",
)
BUDGETS = (8, 12, 16, 20, 24, 28, 32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def contains_literal(text: str, phrase: str) -> bool:
    tokens = WORD_RE.findall(phrase)
    if not tokens:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in tokens) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def sign_flips(values: list[int]) -> int:
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {"passes": True, "models": {}}
    table = []
    curves = []
    case_rows = []

    for model in MODELS:
        selection = read_jsonl(args.selection_dir / f"{model}.jsonl")
        rows = read_jsonl(args.run_root / model / "cases.jsonl")
        exit_path = args.run_root / model / "exit_code.txt"
        exit_code = exit_path.read_text().strip() if exit_path.exists() else "0"
        selected_by_id = {row["id"]: row for row in selection}
        ids = [row["id"] for row in rows]
        mismatches = []
        for row in rows:
            selected = selected_by_id.get(row["id"])
            if selected is None:
                mismatches.append({"id": row["id"], "issue": "not_selected"})
                continue
            if row["rounds"][:9] != selected["baseline"]["rounds"]:
                mismatches.append({"id": row["id"], "issue": "baseline_rounds_changed"})
            if row["baseline_line_sha256"] != selected["baseline_line_sha256"]:
                mismatches.append({"id": row["id"], "issue": "baseline_hash_changed"})
            for round_row in row["rounds"]:
                recounted = word_count(round_row["text"])
                missing = [
                    anchor for anchor in row["required"]
                    if not contains_literal(round_row["text"], anchor)
                ]
                exact = recounted == row["target"]
                joint = exact and not missing
                if (
                    recounted != round_row["word_count"]
                    or missing != round_row["missing"]
                    or exact != round_row["exact_length"]
                    or joint != round_row["joint_success"]
                ):
                    mismatches.append(
                        {"id": row["id"], "revision": round_row["revision"], "issue": "recount"}
                    )

        model_pass = (
            exit_code == "0"
            and len(ids) == len(selection)
            and len(ids) == len(set(ids))
            and set(ids) == set(selected_by_id)
            and not mismatches
        )
        audit["passes"] = audit["passes"] and model_pass
        audit["models"][model] = {
            "passes": model_pass,
            "selected": len(selection),
            "rows": len(rows),
            "unique_ids": len(set(ids)),
            "exit_code": exit_code,
            "mismatches": mismatches,
        }

        closed = sum(row["closure_revision"] is not None for row in rows)
        context = sum(row["termination_reason"] == "context_limit" for row in rows)
        persistent = [
            row for row in rows
            if row["termination_reason"] == "max_revision" and row["final_revision"] == 32
        ]
        low, high = wilson(closed, len(rows))
        closure_revisions = [
            row["closure_revision"] for row in rows if row["closure_revision"] is not None
        ]
        table.append(
            {
                "model": model,
                "selected_revision8_failures": len(rows),
                "newly_closed_n": closed,
                "newly_closed_rate": closed / len(rows),
                "wilson95_low": low,
                "wilson95_high": high,
                "persistent_revision32_n": len(persistent),
                "context_exhaustion_n": context,
                "median_closure_revision": (
                    statistics.median(closure_revisions) if closure_revisions else ""
                ),
            }
        )
        for budget in BUDGETS:
            n = sum(
                row["closure_revision"] is not None and row["closure_revision"] <= budget
                for row in rows
            )
            curves.append(
                {
                    "model": model,
                    "revision_budget": budget,
                    "additional_closure_n": n,
                    "additional_closure_rate": n / len(rows),
                }
            )

        for row in rows:
            continuation_errors = [
                int(item["error"]) for item in row["rounds"] if item["revision"] > 8
            ]
            baseline_abs = abs(int(row["baseline_terminal_error"]))
            best_continuation_abs = (
                min(abs(value) for value in continuation_errors)
                if continuation_errors else None
            )
            is_persistent = (
                row["termination_reason"] == "max_revision" and row["final_revision"] == 32
            )
            case_rows.append(
                {
                    "model": model,
                    "id": row["id"],
                    "source": row["source"],
                    "selection_stratum": row["selection_stratum"],
                    "target": row["target"],
                    "baseline_terminal_error": row["baseline_terminal_error"],
                    "termination_reason": row["termination_reason"],
                    "closure_revision": row["closure_revision"] or "",
                    "final_revision": row["final_revision"],
                    "final_error": row["rounds"][-1]["error"],
                    "context_limit_revision": row["context_limit_revision"] or "",
                    "persistent_revision32": is_persistent,
                    "no_improvement_persistence": (
                        is_persistent
                        and best_continuation_abs is not None
                        and best_continuation_abs >= baseline_abs
                    ),
                    "terminal_stasis": (
                        is_persistent
                        and len(continuation_errors) >= 4
                        and len(set(continuation_errors[-4:])) == 1
                    ),
                    "persistent_oscillation": (
                        is_persistent and sign_flips(continuation_errors) >= 4
                    ),
                    "continuation_sign_flips": sign_flips(continuation_errors),
                    "best_continuation_abs_error": (
                        best_continuation_abs if best_continuation_abs is not None else ""
                    ),
                }
            )

    for name, rows in (
        ("table_persistence.csv", table),
        ("cumulative_closure.csv", curves),
        ("case_diagnostics.csv", case_rows),
    ):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "protocol": "docs/persistence_revision32_protocol_20260724.md",
        "table": table,
        "diagnostic_totals": {
            model: {
                "persistent_revision32_n": sum(
                    row["model"] == model and row["persistent_revision32"] for row in case_rows
                ),
                "no_improvement_persistence_n": sum(
                    row["model"] == model and row["no_improvement_persistence"] for row in case_rows
                ),
                "terminal_stasis_n": sum(
                    row["model"] == model and row["terminal_stasis"] for row in case_rows
                ),
                "persistent_oscillation_n": sum(
                    row["model"] == model and row["persistent_oscillation"] for row in case_rows
                ),
            }
            for model in MODELS
        },
        "integrity_passes": audit["passes"],
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "integrity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not audit["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
