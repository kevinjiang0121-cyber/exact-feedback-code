#!/usr/bin/env python3
"""Freeze and audit the COLLIE c05/c10 closed-loop extension suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COLLIE = ROOT / "third_party" / "Collie-master"
DEFAULT_VENDOR = ROOT / "third_party" / "python_packages"
DEFAULT_NLTK = ROOT / "third_party" / "nltk_data"
DEFAULT_OUTPUT = ROOT / "data" / "multiconstraint_v1"

for path in (DEFAULT_VENDOR, DEFAULT_COLLIE):
    sys.path.insert(0, str(path))
os.environ.setdefault("NLTK_DATA", str(DEFAULT_NLTK))

import dill  # noqa: E402
import nltk  # noqa: E402
from nltk import sent_tokenize, word_tokenize  # noqa: E402

nltk.data.path.insert(0, str(DEFAULT_NLTK))


SOURCE_PREFIXES = ("ccnews", "guten", "wiki")
FAMILY_SPECS = {
    "lexical_position_c05": "c05",
    "compositional_structure_c10": "c10",
}
POSITIONS = (3, 7, 10)  # zero based; COLLIE prompt renders 4th, 8th, 11th


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def normalized_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.lower().strip(string.punctuation + " ")
    return value if stripped == "" else stripped


def words(text: str) -> list[str]:
    units = [token for token in word_tokenize(text) if token not in string.punctuation]
    return [token.strip().strip(".") for token in units]


def sentences(text: str) -> list[str]:
    return [unit.strip().strip(".") for unit in sent_tokenize(text)]


def verify_c05(text: str, targets: list[Any]) -> dict[str, Any]:
    target_count, target_words = targets
    tokens = words(text)
    observed_words = [tokens[pos] if pos < len(tokens) else None for pos in POSITIONS]
    matches = [
        normalized_literal(observed) == normalized_literal(target)
        for observed, target in zip(observed_words, target_words)
    ]
    count_residual = len(tokens) - int(target_count)
    mismatch_count = sum(not match for match in matches)
    return {
        "joint_success": count_residual == 0 and mismatch_count == 0,
        "observed_word_count": len(tokens),
        "target_word_count": int(target_count),
        "count_residual": count_residual,
        "target_positions_zero_based": list(POSITIONS),
        "target_words": target_words,
        "observed_words": observed_words,
        "position_matches": matches,
        "position_mismatch_count": mismatch_count,
        "violation_vector": [
            abs(count_residual) / max(int(target_count), 1),
            *[0.0 if match else 1.0 for match in matches],
        ],
    }


def verify_c10(text: str, targets: list[Any]) -> dict[str, Any]:
    target_sentences, lower, upper = map(int, targets)
    units = sentences(text)
    counts = [len(words(unit)) for unit in units]
    sentence_residual = len(units) - target_sentences
    lower_deficits = [max(0, lower - count) for count in counts]
    upper_excesses = [max(0, count - upper) for count in counts]
    violations = [abs(sentence_residual) / max(target_sentences, 1)]
    violations.extend(value / max(lower, 1) for value in lower_deficits)
    violations.extend(value / max(upper, 1) for value in upper_excesses)
    return {
        "joint_success": (
            sentence_residual == 0
            and all(value == 0 for value in lower_deficits)
            and all(value == 0 for value in upper_excesses)
        ),
        "observed_sentence_count": len(units),
        "target_sentence_count": target_sentences,
        "sentence_count_residual": sentence_residual,
        "sentence_word_counts": counts,
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_deficits": lower_deficits,
        "upper_excesses": upper_excesses,
        "violation_vector": violations,
    }


def verify(family: str, text: str, targets: list[Any]) -> dict[str, Any]:
    if family == "lexical_position_c05":
        return verify_c05(text, targets)
    if family == "compositional_structure_c10":
        return verify_c10(text, targets)
    raise KeyError(family)


def normalized_witness(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def corruption_checks(family: str, row: dict[str, Any]) -> dict[str, bool]:
    witness = row["witness"]
    targets = row["targets"]
    if family == "lexical_position_c05":
        token_units = words(witness)
        position_corrupt = list(token_units)
        position_corrupt[POSITIONS[0]] = "mismatchtoken"
        position_result = verify(family, " ".join(position_corrupt), targets)
        count_result = verify(family, witness + " extra", targets)
        return {
            "position_corruption_detected": position_result["position_mismatch_count"] > 0,
            "count_corruption_detected": count_result["count_residual"] != 0,
        }

    sentence_units = sentences(witness)
    # A single retained sentence must violate every c10 target (3--5 sentences)
    # without relying on lossy reconstruction of quoted sentence boundaries.
    removed_result = verify(family, sentence_units[0], targets)
    upper = int(targets[2])
    inflated = list(sentence_units)
    inflated[0] = inflated[0] + " extra" * (upper + 1)
    bound_result = verify(family, ". ".join(inflated) + ".", targets)
    return {
        "sentence_count_corruption_detected": removed_result["sentence_count_residual"] != 0,
        "bound_corruption_detected": any(bound_result["upper_excesses"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collie-root", type=Path, default=DEFAULT_COLLIE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-source", type=int, default=40)
    args = parser.parse_args()

    data_path = args.collie_root / "data" / "all_data.dill"
    with data_path.open("rb") as handle:
        source_data = dill.load(handle)

    all_selected: list[dict[str, Any]] = []
    family_audits: dict[str, Any] = {}
    seen_witnesses: set[str] = set()

    for family, structure in FAMILY_SPECS.items():
        selected: list[dict[str, Any]] = []
        available: dict[str, int] = {}
        for source in SOURCE_PREFIXES:
            source_key = f"{source}_{structure}"
            candidates = source_data[source_key]
            available[source] = len(candidates)
            ranked = sorted(
                enumerate(candidates),
                key=lambda pair: stable_key(
                    family,
                    source,
                    pair[1]["prompt"],
                    pair[1]["example"],
                ),
            )
            source_rows: list[dict[str, Any]] = []
            for upstream_index, item in ranked:
                witness_norm = normalized_witness(item["example"])
                if witness_norm in seen_witnesses:
                    continue
                local_result = verify(family, item["example"], item["targets"])
                official_result = bool(item["constraint"].check(item["example"], item["targets"]))
                if not local_result["joint_success"] or not official_result:
                    continue
                split = "discovery" if len(source_rows) < args.per_source // 2 else "confirmation"
                row = {
                    "id": f"{family}__{source}__{upstream_index:04d}",
                    "family": family,
                    "structure": structure,
                    "split": split,
                    "source": source,
                    "upstream_key": source_key,
                    "upstream_index": upstream_index,
                    "task": item["prompt"],
                    "targets": item["targets"],
                    "witness": item["example"],
                    "witness_sha256": hashlib.sha256(item["example"].encode("utf-8")).hexdigest(),
                    "initial_verifier_state": local_result,
                }
                row["corruption_checks"] = corruption_checks(family, row)
                source_rows.append(row)
                seen_witnesses.add(witness_norm)
                if len(source_rows) == args.per_source:
                    break
            if len(source_rows) != args.per_source:
                raise RuntimeError(
                    f"{family}/{source}: selected {len(source_rows)} of {args.per_source}"
                )
            selected.extend(source_rows)

        repeated_determinism = all(
            verify(row["family"], row["witness"], row["targets"])
            == verify(row["family"], row["witness"], row["targets"])
            == verify(row["family"], row["witness"], row["targets"])
            for row in selected
        )
        all_corruptions_detected = all(
            all(row["corruption_checks"].values()) for row in selected
        )
        family_audits[family] = {
            "structure": structure,
            "available_by_source": available,
            "selected": len(selected),
            "discovery": sum(row["split"] == "discovery" for row in selected),
            "confirmation": sum(row["split"] == "confirmation" for row in selected),
            "all_witnesses_pass_local_and_official": all(
                row["initial_verifier_state"]["joint_success"] for row in selected
            ),
            "all_corruptions_detected": all_corruptions_detected,
            "three_pass_determinism": repeated_determinism,
        }
        all_selected.extend(selected)

    if len({row["id"] for row in all_selected}) != len(all_selected):
        raise RuntimeError("Duplicate case IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for family in FAMILY_SPECS:
        for split in ("discovery", "confirmation"):
            path = args.output_dir / f"{family}.{split}.jsonl"
            rows = [
                row for row in all_selected
                if row["family"] == family and row["split"] == split
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            outputs[path.name] = sha256_file(path)

    audit = {
        "collie_revision": "efce1c882e995cca65e1244545167a95dba72df9",
        "collie_data_sha256": sha256_file(data_path),
        "collie_constraints_sha256": sha256_file(args.collie_root / "collie" / "constraints.py"),
        "nltk_version": nltk.__version__,
        "nltk_data": str(DEFAULT_NLTK.resolve()),
        "families": family_audits,
        "unique_ids": len({row["id"] for row in all_selected}),
        "unique_normalized_witnesses": len(seen_witnesses),
        "output_sha256": outputs,
    }
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
