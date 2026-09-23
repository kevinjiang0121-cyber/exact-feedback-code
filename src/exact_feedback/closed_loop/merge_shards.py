#!/usr/bin/env python3
"""Merge frozen batch-one exact-loop shards in canonical dataset order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exact_feedback.common.exact_protocol import read_jsonl, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shards", required=True, nargs="+", type=Path)
    args = parser.parse_args()

    expected = read_jsonl(args.data)
    rows = []
    for shard in args.shards:
        rows.extend(read_jsonl(shard / "cases.jsonl"))
    by_id = {row["id"]: row for row in rows}
    expected_ids = [row["id"] for row in expected]
    if len(by_id) != len(rows):
        raise RuntimeError("duplicate case IDs across shards")
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise RuntimeError(
            f"shard coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    ordered = [by_id[case_id] for case_id in expected_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(ordered, "qwen3_14b")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
