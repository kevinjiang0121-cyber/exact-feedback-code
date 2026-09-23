#!/usr/bin/env python3
"""Recompute exact-length loop fields and write a portable integrity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exact_feedback.common.exact_protocol import contains_literal, read_jsonl, word_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-rows",
        type=int,
        help="audit the ordered first N frozen rows (used by deterministic smoke runs)",
    )
    args = parser.parse_args()

    expected_rows = read_jsonl(args.data)
    if args.expected_rows is not None:
        if args.expected_rows < 1 or args.expected_rows > len(expected_rows):
            raise SystemExit("--expected-rows is outside the frozen input range")
        expected_rows = expected_rows[: args.expected_rows]
    expected = {row["id"]: row for row in expected_rows}
    rows = read_jsonl(args.result_dir / "cases.jsonl")
    ids = [row["id"] for row in rows]
    errors: list[str] = []
    if len(expected) != len(expected_rows):
        errors.append("duplicate input IDs")
    if ids != [row["id"] for row in expected_rows]:
        errors.append("result order or coverage mismatch")
    for row in rows:
        item = expected.get(row["id"])
        if item is None:
            continue
        for turn in row["rounds"]:
            count = word_count(turn["text"])
            missing = [
                anchor
                for anchor in item["anchors"]
                if not contains_literal(turn["text"], anchor)
            ]
            exact = count == item["target"]
            joint = exact and not missing
            if count != turn["word_count"]:
                errors.append(f"{row['id']} r{turn['revision']}: word_count")
            if missing != turn["missing"]:
                errors.append(f"{row['id']} r{turn['revision']}: missing anchors")
            if exact != turn["exact_length"] or joint != turn["joint_success"]:
                errors.append(f"{row['id']} r{turn['revision']}: success flags")
        if row["final_exact"] != row["rounds"][-1]["exact_length"]:
            errors.append(f"{row['id']}: final_exact")
        if row["final_joint"] != row["rounds"][-1]["joint_success"]:
            errors.append(f"{row['id']}: final_joint")

    report = {
        "passes": not errors,
        "expected_rows": len(expected_rows),
        "result_rows": len(rows),
        "errors": errors[:100],
    }
    (args.result_dir / "integrity_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
