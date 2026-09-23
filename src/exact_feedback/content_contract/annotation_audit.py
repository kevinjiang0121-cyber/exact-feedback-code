#!/usr/bin/env python3
"""Strict join, span, probability, and manifest audit for API endpoint annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NLI_PROBABILITY_SUM_TOLERANCE = 5e-4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--stanza", required=True, type=Path)
    parser.add_argument("--stanza-manifest", required=True, type=Path)
    parser.add_argument("--nli", required=True, type=Path)
    parser.add_argument("--nli-manifest", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    registry_rows = read_jsonl(args.registry)
    registry = {row["text_sha256"]: row for row in registry_rows}
    trajectories = read_jsonl(args.trajectories)
    stanza_rows = read_jsonl(args.stanza)
    stanza = {row["text_sha256"]: row for row in stanza_rows}
    nli_rows = read_jsonl(args.nli)
    nli = {row["text_sha256"]: row for row in nli_rows}
    stanza_manifest = json.loads(args.stanza_manifest.read_text(encoding="utf-8"))
    nli_manifest = json.loads(args.nli_manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked_spans = 0
    checked_pairs = 0
    probability_sum_deviations: list[float] = []

    expected_registry = len(registry_rows)
    expected_trajectories = len(trajectories)
    if len(registry) != expected_registry:
        errors.append(f"registry_coverage={len(registry_rows)}/{len(registry)}")
    if len(stanza_rows) != expected_registry or len(stanza) != expected_registry:
        errors.append(f"stanza_coverage={len(stanza_rows)}/{len(stanza)}")
    if set(stanza) != set(registry):
        errors.append("stanza_registry_hash_set_mismatch")

    for text_hash, annotation in stanza.items():
        if annotation.get("status") != "ok":
            errors.append(f"{text_hash}:stanza_status")
            continue
        text = registry[text_hash]["text"]
        for sentence in annotation["sentences"]:
            start, end = sentence["start_char"], sentence["end_char"]
            checked_spans += 1
            if text[start:end] != sentence["text"]:
                errors.append(f"{text_hash}:sentence_span:{sentence['index']}")
        for family in ("entities", "proper_names", "symbolic_units"):
            for item in annotation[family]:
                checked_spans += 1
                if text[item["start_char"] : item["end_char"]] != item["text"]:
                    errors.append(f"{text_hash}:{family}_span")

    expected_nli = {
        text_hash
        for text_hash, row in registry.items()
        if set(row["kinds"]) & {"round0", "final"}
    }
    if len(nli_rows) != len(expected_nli) or len(nli) != len(expected_nli):
        errors.append(f"nli_coverage={len(expected_nli)}/{len(nli_rows)}/{len(nli)}")
    if set(nli) != expected_nli:
        errors.append("nli_hash_set_mismatch")
    for text_hash, annotation in nli.items():
        if annotation.get("status") != "ok":
            errors.append(f"{text_hash}:nli_status")
            continue
        sentences = stanza[text_hash]["sentences"]
        for pair in annotation["pairs"]:
            checked_pairs += 1
            left, right = pair["left_index"], pair["right_index"]
            if not (0 <= left < right < len(sentences)):
                errors.append(f"{text_hash}:pair_index")
                continue
            if pair["premise"] != sentences[left]["text"]:
                errors.append(f"{text_hash}:premise_mismatch")
            if pair["hypothesis"] != sentences[right]["text"]:
                errors.append(f"{text_hash}:hypothesis_mismatch")
            if pair["pair_kind"] == "adjacent" and right != left + 1:
                errors.append(f"{text_hash}:adjacency_rule")
            if pair["pair_kind"] == "entity_linked":
                shared = set(sentences[left]["entity_units"]) & set(
                    sentences[right]["entity_units"]
                )
                if right == left + 1 or sorted(shared) != pair["shared_entities"]:
                    errors.append(f"{text_hash}:entity_link_rule")
            probabilities = pair["probabilities"]
            if set(probabilities) != {"entailment", "neutral", "contradiction"}:
                errors.append(f"{text_hash}:nli_labels")
            values = list(probabilities.values())
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                errors.append(f"{text_hash}:nli_probability_range")
            probability_sum_deviations.append(abs(sum(values) - 1.0))

    probability_sum_deviations.sort()
    probability_sum_failures = sum(
        deviation > NLI_PROBABILITY_SUM_TOLERANCE
        for deviation in probability_sum_deviations
    )
    if probability_sum_failures:
        errors.append(
            "nli_probability_sum:"
            f"{probability_sum_failures}/{len(probability_sum_deviations)}"
        )

    def probability_quantile(fraction: float) -> float | None:
        if not probability_sum_deviations:
            return None
        index = min(
            len(probability_sum_deviations) - 1,
            int(fraction * (len(probability_sum_deviations) - 1)),
        )
        return probability_sum_deviations[index]

    expected_missing = sum(
        row["round0_sha256"] is None or row["final_sha256"] is None for row in trajectories
    )
    if analysis["coverage"] != {
        "trajectories": expected_trajectories,
        "ok": expected_trajectories - expected_missing,
        "provider_refusal_not_applicable": expected_missing,
        "annotation_abstain": 0,
    }:
        errors.append("analysis_coverage_mismatch")
    manifest_checks = {
        "stanza_registry": (
            stanza_manifest["registry_sha256"],
            sha256(args.registry),
        ),
        "stanza_output": (
            stanza_manifest["output_sha256"],
            sha256(args.stanza),
        ),
        "nli_registry": (
            nli_manifest["registry_sha256"],
            sha256(args.registry),
        ),
        "nli_stanza": (
            nli_manifest["stanza_annotations_sha256"],
            sha256(args.stanza),
        ),
        "nli_output": (
            nli_manifest["output_sha256"],
            sha256(args.nli),
        ),
    }
    for name, (recorded, actual) in manifest_checks.items():
        if recorded != actual:
            errors.append(f"manifest_hash:{name}")

    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "coverage": {
            "registry_texts": len(registry),
            "trajectories": len(trajectories),
            "stanza_annotations": len(stanza),
            "nli_annotations": len(nli),
            "checked_spans": checked_spans,
            "checked_sentence_pairs": checked_pairs,
        },
        "nli_probability_sum_audit": {
            "tolerance": NLI_PROBABILITY_SUM_TOLERANCE,
            "rationale": (
                "Explicit FP16 softmax serialization tolerance; probabilities "
                "must remain finite, individually within [0, 1], and sum within "
                "the stated absolute deviation."
            ),
            "failures": probability_sum_failures,
            "deviation_min": (
                probability_sum_deviations[0]
                if probability_sum_deviations
                else None
            ),
            "deviation_p50": probability_quantile(0.5),
            "deviation_p90": probability_quantile(0.9),
            "deviation_p99": probability_quantile(0.99),
            "deviation_p999": probability_quantile(0.999),
            "deviation_max": (
                probability_sum_deviations[-1]
                if probability_sum_deviations
                else None
            ),
        },
        "hashes": {
            "registry": sha256(args.registry),
            "trajectories": sha256(args.trajectories),
            "stanza": sha256(args.stanza),
            "nli": sha256(args.nli),
            "analysis": sha256(args.analysis),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
