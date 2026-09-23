#!/usr/bin/env python3
"""Strict integrity audit for one 960-case multidomain controller result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.common.structured_protocol as protocol  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, nargs="+")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=960)
    parser.add_argument("--max-revisions", type=int, default=8)
    args = parser.parse_args()

    items = [item for path in args.data for item in read_jsonl(path)]
    rows = read_jsonl(args.cases)
    item_by_id = {item["id"]: item for item in items}
    errors: list[str] = []
    if len(items) != args.expected_rows or len(item_by_id) != args.expected_rows:
        errors.append(f"benchmark rows/unique={len(items)}/{len(item_by_id)}")
    if len(rows) != args.expected_rows or len({row["id"] for row in rows}) != len(rows):
        errors.append(f"result rows/unique={len(rows)}/{len({row['id'] for row in rows})}")
    if {row["id"] for row in rows} != set(item_by_id):
        errors.append("result ID set differs from benchmark")

    for row in rows:
        item = item_by_id.get(row["id"])
        if item is None:
            continue
        if row.get("structure") != item["structure"] or row.get("source") != item["source"]:
            errors.append(f"metadata mismatch:{row['id']}")
        revisions = [int(round_row["revision"]) for round_row in row["rounds"]]
        if revisions != list(range(len(revisions))):
            errors.append(f"revision sequence:{row['id']}")
            continue
        if revisions[-1] > args.max_revisions:
            errors.append(f"revision overflow:{row['id']}")
        for round_row in row["rounds"]:
            verifier = protocol.verify(item["structure"], round_row["text"], item["targets"])
            if bool(verifier["joint_success"]) != bool(round_row["joint_success"]):
                errors.append(f"joint mismatch:{row['id']}:{round_row['revision']}")
            energy = sum(float(value) for value in verifier["violation_vector"])
            if not math.isclose(energy, float(round_row["violation_energy"]), abs_tol=1e-12):
                errors.append(f"energy mismatch:{row['id']}:{round_row['revision']}")
        if bool(row["final_joint"]) != bool(row["rounds"][-1]["joint_success"]):
            errors.append(f"final mismatch:{row['id']}")
        if row["final_joint"] and revisions[-1] == args.max_revisions:
            pass
        elif not row["final_joint"] and revisions[-1] != args.max_revisions:
            errors.append(f"unresolved early stop:{row['id']}")

    report = {
        "verdict": "PASS" if not errors else "FAIL",
        "benchmark_rows": len(items),
        "result_rows": len(rows),
        "domains": dict(Counter(item["domain"] for item in items)),
        "structures": dict(Counter(item["structure"] for item in items)),
        "sources": dict(Counter(item["source"] for item in items)),
        "data_sha256": {str(path): sha256(path) for path in args.data},
        "cases_sha256": sha256(args.cases),
        "errors": errors[:100],
        "error_count": len(errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
