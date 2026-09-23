#!/usr/bin/env python3
"""Aggregate success and exact-text recurrence for the open12 multidomain panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trajectory(row: dict[str, Any]) -> dict[str, Any]:
    rounds = row["rounds"]
    hashes = [text_hash(str(item["text"])) for item in rounds]
    first_seen: dict[str, int] = {}
    first_recurrence: int | None = None
    for revision, value in enumerate(hashes):
        if value in first_seen and first_recurrence is None:
            first_recurrence = revision
        first_seen.setdefault(value, revision)
    terminal_recurrence = len(hashes) > 1 and hashes[-1] in hashes[:-1]
    recurrence = first_recurrence is not None
    return {
        "one_shot": bool(row["one_shot_joint"]),
        "final": bool(row["final_joint"]),
        "recurrence": recurrence,
        "early_recurrence_r4": recurrence and first_recurrence <= 4,
        "terminal_recurrence": terminal_recurrence,
        "success_after_recurrence": recurrence and bool(row["final_joint"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    failures = [row for row in rows if not row["final"]]
    recurrent = [row for row in rows if row["recurrence"]]

    def rate(key: str, subset: list[dict[str, Any]] = rows) -> float | None:
        return sum(bool(row[key]) for row in subset) / len(subset) if subset else None

    return {
        "cases": n,
        "one_shot_successes": sum(row["one_shot"] for row in rows),
        "one_shot_rate": rate("one_shot"),
        "final_successes": sum(row["final"] for row in rows),
        "final_success_rate": rate("final"),
        "recurrence_cases": len(recurrent),
        "recurrence_rate": rate("recurrence"),
        "early_recurrence_r4_rate": rate("early_recurrence_r4"),
        "terminal_recurrence_rate": rate("terminal_recurrence"),
        "failures": len(failures),
        "recurrent_failures": sum(row["recurrence"] for row in failures),
        "recurrence_among_failures": rate("recurrence", failures),
        "successes_after_recurrence": sum(row["success_after_recurrence"] for row in recurrent),
        "success_rate_after_recurrence": rate("final", recurrent),
    }


def flatten(group: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    return {**group, **summary}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    excluded = {"backend_smokes"}
    model_paths = sorted(
        path for path in args.root.iterdir()
        if path.is_dir() and path.name not in excluded and (path / "cases.jsonl").exists()
    )
    if len(model_paths) != 12:
        raise ValueError(f"expected 12 completed models, found {len(model_paths)}")

    all_rows: list[dict[str, Any]] = []
    for model_path in model_paths:
        source_rows = read_jsonl(model_path / "cases.jsonl")
        if len(source_rows) != 960:
            raise ValueError(f"{model_path.name}: expected 960 rows, found {len(source_rows)}")
        for row in source_rows:
            all_rows.append({
                "model": model_path.name,
                "domain": row["domain"],
                "family": row["family"],
                "structure": row["structure"],
                **trajectory(row),
            })

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["model"],)].append(row)
        grouped[(row["model"], row["domain"])].append(row)
        grouped[(row["model"], row["family"])].append(row)

    overall_by_domain = []
    for domain in sorted({row["domain"] for row in all_rows}):
        rows = [row for row in all_rows if row["domain"] == domain]
        overall_by_domain.append(flatten({"domain": domain}, summarize(rows)))
    overall_by_family = []
    for family in sorted({row["family"] for row in all_rows}):
        rows = [row for row in all_rows if row["family"] == family]
        overall_by_family.append(flatten({"family": family}, summarize(rows)))

    by_model = [
        flatten({"model": key[0]}, summarize(rows))
        for key, rows in grouped.items() if len(key) == 1
    ]
    by_model_domain = [
        flatten({"model": key[0], "domain": key[1]}, summarize(rows))
        for key, rows in grouped.items()
        if len(key) == 2 and key[1] in {"lexical_constraints", "compositional_constraints"}
    ]
    by_model_family = [
        flatten({"model": key[0], "family": key[1]}, summarize(rows))
        for key, rows in grouped.items()
        if len(key) == 2 and key[1] not in {"lexical_constraints", "compositional_constraints"}
    ]
    by_model.sort(key=lambda row: row["model"])
    by_model_domain.sort(key=lambda row: (row["model"], row["domain"]))
    by_model_family.sort(key=lambda row: (row["model"], row["family"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "by_model.csv", by_model)
    write_csv(args.output_dir / "by_model_domain.csv", by_model_domain)
    write_csv(args.output_dir / "by_model_family.csv", by_model_family)
    write_csv(args.output_dir / "overall_by_domain.csv", overall_by_domain)
    write_csv(args.output_dir / "overall_by_family.csv", overall_by_family)
    payload = {
        "panel": {"models": 12, "cases_per_model": 960, "trajectories": len(all_rows)},
        "overall": summarize(all_rows),
        "overall_by_domain": overall_by_domain,
        "overall_by_family": overall_by_family,
        "by_model": by_model,
        "by_model_domain": by_model_domain,
        "by_model_family": by_model_family,
        "recurrence_definition": "any exact UTF-8 text hash repeated within a trajectory",
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["panel"], ensure_ascii=False))
    print(json.dumps(payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
