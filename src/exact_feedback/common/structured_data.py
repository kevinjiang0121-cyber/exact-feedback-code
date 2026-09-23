#!/usr/bin/env python3
"""Build and audit two 480-case deterministic-verifier domains from COLLIE."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.common.constraint_protocol as base  # noqa: E402


DEFAULT_ORIGINAL = ROOT / "data" / "multiconstraint_v1"
DEFAULT_SCALE = ROOT / "data" / "multiconstraint_v1_scale_extension"
DEFAULT_OUTPUT = ROOT / "data" / "multidomain_full480_v1"
SOURCE_PREFIXES = base.SOURCE_PREFIXES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_tokens(text: str) -> list[str]:
    return [base.normalized_literal(token) for token in base.words(text)]


def verify_c07(text: str, targets: list[Any]) -> dict[str, Any]:
    tokens = normalized_tokens(text)
    normalized_targets = [base.normalized_literal(target) for target in targets]
    present = [target in tokens for target in normalized_targets]
    return {
        "joint_success": all(present),
        "target_words": targets,
        "target_present": present,
        "missing_count": sum(not value for value in present),
        "violation_vector": [0.0 if value else 1.0 for value in present],
    }


def verify_c09(text: str, targets: list[Any]) -> dict[str, Any]:
    target_sentences = int(targets[0])
    forbidden = list(targets[1:])
    units = base.sentences(text)
    tokens = normalized_tokens(text)
    normalized_forbidden = [base.normalized_literal(target) for target in forbidden]
    counts = [tokens.count(target) for target in normalized_forbidden]
    sentence_residual = len(units) - target_sentences
    return {
        "joint_success": sentence_residual == 0 and all(count == 0 for count in counts),
        "observed_sentence_count": len(units),
        "target_sentence_count": target_sentences,
        "sentence_count_residual": sentence_residual,
        "forbidden_words": forbidden,
        "forbidden_word_counts": counts,
        "violation_vector": [
            abs(sentence_residual) / max(target_sentences, 1),
            *[float(count) for count in counts],
        ],
    }


def verify_c12(text: str, targets: list[Any]) -> dict[str, Any]:
    target_sentences = int(targets[0])
    target_last_words = list(targets[1])
    units = base.sentences(text)
    observed_last_words = []
    for unit in units:
        tokens = base.words(unit)
        observed_last_words.append(tokens[-1] if tokens else None)
    matches = [
        index < len(observed_last_words)
        and base.normalized_literal(observed_last_words[index])
        == base.normalized_literal(target)
        for index, target in enumerate(target_last_words)
    ]
    sentence_residual = len(units) - target_sentences
    return {
        "joint_success": sentence_residual == 0 and all(matches),
        "observed_sentence_count": len(units),
        "target_sentence_count": target_sentences,
        "sentence_count_residual": sentence_residual,
        "target_last_words": target_last_words,
        "observed_last_words": observed_last_words,
        "last_word_matches": matches,
        "last_word_mismatch_count": sum(not value for value in matches),
        "violation_vector": [
            abs(sentence_residual) / max(target_sentences, 1),
            *[0.0 if value else 1.0 for value in matches],
        ],
    }


def verify(structure: str, text: str, targets: list[Any]) -> dict[str, Any]:
    if structure == "c05":
        return base.verify_c05(text, targets)
    if structure == "c07":
        return verify_c07(text, targets)
    if structure == "c09":
        return verify_c09(text, targets)
    if structure == "c10":
        return base.verify_c10(text, targets)
    if structure == "c12":
        return verify_c12(text, targets)
    raise KeyError(structure)


def corruption_checks(row: dict[str, Any]) -> dict[str, bool]:
    structure = row["structure"]
    witness = row["witness"]
    targets = row["targets"]
    if structure in {"c05", "c10"}:
        return base.corruption_checks(row["family"], row)
    if structure == "c07":
        target = base.normalized_literal(targets[0])
        corrupted = [
            "mismatchtoken" if base.normalized_literal(token) == target else token
            for token in base.words(witness)
        ]
        return {"required_word_corruption_detected": not verify(structure, " ".join(corrupted), targets)["joint_success"]}
    if structure == "c09":
        units = base.sentences(witness)
        return {
            "sentence_count_corruption_detected": verify(structure, units[0], targets)["sentence_count_residual"] != 0,
            "forbidden_word_corruption_detected": verify(structure, witness + " " + str(targets[1]), targets)["forbidden_word_counts"][0] > 0,
        }
    if structure == "c12":
        units = base.sentences(witness)
        first_tokens = base.words(units[0])
        first_tokens[-1] = "mismatchtoken"
        corrupted_units = [" ".join(first_tokens), *units[1:]]
        return {
            "sentence_count_corruption_detected": verify(structure, ". ".join(units[:-1]) + ".", targets)["sentence_count_residual"] != 0,
            "last_word_corruption_detected": not verify(structure, ". ".join(corrupted_units) + ".", targets)["last_word_matches"][0],
        }
    raise KeyError(structure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collie-root", type=Path, default=base.DEFAULT_COLLIE)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--scale-dir", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_path = args.collie_root / "data" / "all_data.dill"
    with data_path.open("rb") as handle:
        source_data = base.dill.load(handle)

    existing_paths = [
        *sorted(args.original_dir.glob("*.discovery.jsonl")),
        *sorted(args.original_dir.glob("*.confirmation.jsonl")),
        *sorted(args.scale_dir.glob("*.scale_extension.jsonl")),
    ]
    existing = [row for path in existing_paths for row in read_jsonl(path)]
    if len(existing) != 480:
        raise RuntimeError(f"expected 480 existing rows, found {len(existing)}")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_witnesses: set[str] = set()
    for source_row in existing:
        row = copy.deepcopy(source_row)
        row["domain"] = (
            "lexical_constraints" if row["structure"] == "c05" else "compositional_constraints"
        )
        row["benchmark_partition"] = "preserved_c05_c10_240"
        norm = base.normalized_witness(row["witness"])
        if row["id"] in seen_ids or norm in seen_witnesses:
            raise RuntimeError(f"duplicate in preserved rows: {row['id']}")
        upstream = source_data[row["upstream_key"]][int(row["upstream_index"])]
        local = verify(row["structure"], row["witness"], row["targets"])
        official = bool(upstream["constraint"].check(row["witness"], row["targets"]))
        if not local["joint_success"] or not official:
            raise RuntimeError(f"preserved verifier mismatch: {row['id']}")
        row["initial_verifier_state"] = local
        row["corruption_checks"] = corruption_checks(row)
        rows.append(row)
        seen_ids.add(row["id"])
        seen_witnesses.add(norm)

    additions = {
        "lexical_constraints": {"c07": 40, "c12": 40},
        "compositional_constraints": {"c09": 80},
    }
    added_counts: Counter[tuple[str, str, str]] = Counter()
    for domain, structures in additions.items():
        for structure, per_source in structures.items():
            for source in SOURCE_PREFIXES:
                source_key = f"{source}_{structure}"
                ranked = sorted(
                    enumerate(source_data[source_key]),
                    key=lambda pair: base.stable_key(
                        domain, structure, source, pair[1]["prompt"], pair[1]["example"]
                    ),
                )
                selected = 0
                for upstream_index, item in ranked:
                    case_id = f"{domain}__{structure}__{source}__{upstream_index:04d}"
                    norm = base.normalized_witness(item["example"])
                    if case_id in seen_ids or norm in seen_witnesses:
                        continue
                    local = verify(structure, item["example"], item["targets"])
                    official = bool(item["constraint"].check(item["example"], item["targets"]))
                    if not local["joint_success"] or not official:
                        continue
                    row = {
                        "id": case_id,
                        "domain": domain,
                        "family": f"{domain}_{structure}",
                        "structure": structure,
                        "split": "full480_extension",
                        "benchmark_partition": "full480_extension",
                        "source": source,
                        "upstream_key": source_key,
                        "upstream_index": upstream_index,
                        "task": item["prompt"],
                        "targets": item["targets"],
                        "witness": item["example"],
                        "witness_sha256": hashlib.sha256(item["example"].encode("utf-8")).hexdigest(),
                        "initial_verifier_state": local,
                    }
                    row["corruption_checks"] = corruption_checks(row)
                    rows.append(row)
                    seen_ids.add(case_id)
                    seen_witnesses.add(norm)
                    added_counts[(domain, structure, source)] += 1
                    selected += 1
                    if selected == per_source:
                        break
                if selected != per_source:
                    raise RuntimeError(
                        f"{domain}/{structure}/{source}: selected {selected} of {per_source}"
                    )

    if len(rows) != 960 or len(seen_ids) != 960 or len(seen_witnesses) != 960:
        raise RuntimeError("full benchmark is not 960-way unique")

    for row in rows:
        local = verify(row["structure"], row["witness"], row["targets"])
        upstream = source_data[row["upstream_key"]][int(row["upstream_index"])]
        official = bool(upstream["constraint"].check(row["witness"], row["targets"]))
        if not local["joint_success"] or not official:
            raise RuntimeError(f"final verifier mismatch: {row['id']}")
        if not all(row["corruption_checks"].values()):
            raise RuntimeError(f"corruption audit failed: {row['id']}")
        if not (
            local
            == verify(row["structure"], row["witness"], row["targets"])
            == verify(row["structure"], row["witness"], row["targets"])
        ):
            raise RuntimeError(f"determinism audit failed: {row['id']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for domain in ("lexical_constraints", "compositional_constraints"):
        path = args.output_dir / f"{domain}_combined480.jsonl"
        domain_rows = sorted(
            (row for row in rows if row["domain"] == domain),
            key=lambda row: (row["source"], row["structure"], row["id"]),
        )
        if len(domain_rows) != 480:
            raise RuntimeError(f"{domain}: expected 480 rows")
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in domain_rows),
            encoding="utf-8",
        )
        output_hashes[path.name] = base.sha256_file(path)

    domain_structure_counts = {
        domain: dict(Counter(row["structure"] for row in rows if row["domain"] == domain))
        for domain in ("lexical_constraints", "compositional_constraints")
    }
    domain_source_counts = {
        domain: dict(Counter(row["source"] for row in rows if row["domain"] == domain))
        for domain in ("lexical_constraints", "compositional_constraints")
    }
    audit = {
        "benchmark": "multidomain_full480_v1",
        "total_cases": len(rows),
        "cases_per_domain": 480,
        "unique_ids": len(seen_ids),
        "unique_normalized_witnesses": len(seen_witnesses),
        "preserved_cases": len(existing),
        "new_cases": len(rows) - len(existing),
        "domain_structure_counts": domain_structure_counts,
        "domain_source_counts": domain_source_counts,
        "added_counts": {"|".join(key): value for key, value in sorted(added_counts.items())},
        "all_witnesses_pass_local_and_official": True,
        "all_corruptions_detected": True,
        "three_pass_determinism": True,
        "collie_revision": "efce1c882e995cca65e1244545167a95dba72df9",
        "collie_data_sha256": base.sha256_file(data_path),
        "collie_constraints_sha256": base.sha256_file(args.collie_root / "collie" / "constraints.py"),
        "output_sha256": output_hashes,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
