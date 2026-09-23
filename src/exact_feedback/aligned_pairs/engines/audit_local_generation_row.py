#!/usr/bin/env python3
"""Strict deterministic integrity audit for one frozen local generation row."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def contains_literal(text: str, phrase: str) -> bool:
    tokens = WORD_RE.findall(phrase)
    if not tokens:
        return False
    pattern = (
        r"(?<!\w)"
        + r"\s+".join(re.escape(token) for token in tokens)
        + r"(?!\w)"
    )
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-data", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-revisions", type=int, default=8)
    args = parser.parse_args()

    frozen = read_jsonl(args.frozen_data)
    rows = read_jsonl(args.cases)
    expected_ids = [row["id"] for row in frozen]
    frozen_by_id = {row["id"]: row for row in frozen}
    errors: list[str] = []
    checked_rounds = 0

    if [row.get("id") for row in rows] != expected_ids:
        errors.append("row_order_or_ids")
    if len({row.get("id") for row in rows}) != len(rows):
        errors.append("duplicate_result_ids")

    for row in rows:
        case_id = row.get("id")
        item = frozen_by_id.get(case_id)
        if item is None:
            continue
        for field in ("source", "source_id", "length_band", "target"):
            if row.get(field) != item[field]:
                errors.append(f"{case_id}:frozen_{field}")
        if row.get("required") != item["anchors"]:
            errors.append(f"{case_id}:frozen_anchors")

        rounds = row.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            errors.append(f"{case_id}:empty_rounds")
            continue
        if len(rounds) > args.max_revisions + 1:
            errors.append(f"{case_id}:too_many_rounds")
        if [round_item.get("revision") for round_item in rounds] != list(
            range(len(rounds))
        ):
            errors.append(f"{case_id}:revision_sequence")

        recomputed_exact: list[bool] = []
        recomputed_joint: list[bool] = []
        for index, round_item in enumerate(rounds):
            checked_rounds += 1
            text = round_item.get("text")
            if not isinstance(text, str):
                errors.append(f"{case_id}:round{index}:text")
                continue
            count = word_count(text)
            missing = [
                anchor
                for anchor in item["anchors"]
                if not contains_literal(text, anchor)
            ]
            exact = count == item["target"]
            joint = exact and not missing
            recomputed_exact.append(exact)
            recomputed_joint.append(joint)
            expected = {
                "word_count": count,
                "error": count - item["target"],
                "exact_length": exact,
                "missing": missing,
                "joint_success": joint,
            }
            for field, value in expected.items():
                if round_item.get(field) != value:
                    errors.append(f"{case_id}:round{index}:{field}")

        if recomputed_joint[:-1] and any(recomputed_joint[:-1]):
            errors.append(f"{case_id}:continued_after_joint")
        if not recomputed_joint[-1] and len(rounds) != args.max_revisions + 1:
            errors.append(f"{case_id}:premature_stop")
        expected_row = {
            "one_shot_exact": recomputed_exact[0],
            "one_shot_joint": recomputed_joint[0],
            "final_exact": recomputed_exact[-1],
            "final_joint": recomputed_joint[-1],
            "ever_exact": any(recomputed_exact),
            "revisions_used": len(rounds) - 1,
        }
        for field, value in expected_row.items():
            if row.get(field) != value:
                errors.append(f"{case_id}:{field}")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "coverage": {
            "frozen_rows": len(frozen),
            "result_rows": len(rows),
            "unique_result_ids": len({row.get("id") for row in rows}),
            "checked_rounds": checked_rounds,
        },
        "outcomes": {
            "one_shot_joint": sum(bool(row.get("one_shot_joint")) for row in rows),
            "final_joint": sum(bool(row.get("final_joint")) for row in rows),
        },
        "hashes": {
            "frozen_data": sha256(args.frozen_data),
            "cases": sha256(args.cases),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
