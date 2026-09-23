#!/usr/bin/env python3
"""Audit Extension-240 and analyze the frozen six-model Combined-480."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from exact_feedback.common.exact_protocol import contains_literal, word_count


MODELS = (
    "llama31_8b",
    "gemma2_9b",
    "qwen3_14b",
    "glm4_9b",
    "qwen3_8b",
    "ministral_8b",
)
LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "gemma2_9b": "Gemma-2-9B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_8b": "Qwen3-8B",
    "glm4_9b": "GLM-4-9B",
    "ministral_8b": "Ministral-8B",
}
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 20_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def family(source: str) -> str:
    if source.startswith("dolly_"):
        return "dolly"
    if source.startswith("no_robots_"):
        return "no_robots"
    if source.startswith("oasst1_"):
        return "oasst1"
    if source.startswith("writingprompts_"):
        return "writingprompts"
    raise ValueError(f"unknown source family: {source}")


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def stratified_bootstrap_rate(
    rows: list[dict[str, Any]], key: str, rng: np.random.Generator
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[family(row["source"])].append(float(bool(row[key])))
    totals = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    n_total = 0
    for values in grouped.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
        totals += array[indices].sum(axis=1)
        n_total += len(array)
    return np.quantile(totals / n_total, [0.025, 0.975]).tolist()


def stratified_bootstrap_independent_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    rng: np.random.Generator,
) -> list[float]:
    def samples(rows: list[dict[str, Any]]) -> np.ndarray:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[family(row["source"])].append(float(bool(row[key])))
        totals = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
        n_total = 0
        for values in grouped.values():
            array = np.asarray(values, dtype=np.float64)
            indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
            totals += array[indices].sum(axis=1)
            n_total += len(array)
        return totals / n_total

    differences = samples(left) - samples(right)
    return np.quantile(differences, [0.025, 0.975]).tolist()


def stratified_bootstrap_paired_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    rng: np.random.Generator,
) -> list[float]:
    right_by_id = {row["id"]: row for row in right}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in left:
        grouped[family(row["source"])].append(
            float(bool(row[key])) - float(bool(right_by_id[row["id"]][key]))
        )
    totals = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    n_total = 0
    for values in grouped.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
        totals += array[indices].sum(axis=1)
        n_total += len(array)
    return np.quantile(totals / n_total, [0.025, 0.975]).tolist()


def rate_summary(
    rows: list[dict[str, Any]], key: str, rng: np.random.Generator
) -> dict[str, Any]:
    successes = sum(bool(row[key]) for row in rows)
    return {
        "cases": len(rows),
        "successes": successes,
        "rate": successes / len(rows),
        "wilson_95": wilson(successes, len(rows)),
        "stratified_bootstrap_95": stratified_bootstrap_rate(rows, key, rng),
    }


def exact_mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    right_by_id = {row["id"]: row for row in right}
    left_only = sum(
        bool(row["final_joint"]) and not bool(right_by_id[row["id"]]["final_joint"])
        for row in left
    )
    right_only = sum(
        not bool(row["final_joint"]) and bool(right_by_id[row["id"]]["final_joint"])
        for row in left
    )
    discordant = left_only + right_only
    p = (
        1.0
        if discordant == 0
        else min(
            1.0,
            2
            * sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
            / (2**discordant),
        )
    )
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "exact_two_sided_p": p,
    }


def holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda value: value[1]["exact_two_sided_p"])
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, (len(rows) - rank) * row["exact_two_sided_p"])
        rows[index]["holm_p"] = min(1.0, running)


def audit_model_rows(
    rows: list[dict[str, Any]], data: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_result_ids")
    if set(ids) != set(data):
        errors.append(f"result_identity_missing={len(set(data)-set(ids))}_extra={len(set(ids)-set(data))}")
    for row in rows:
        case_id = row["id"]
        if case_id not in data:
            continue
        item = data[case_id]
        for key in ("source", "source_id", "length_band", "target"):
            if row[key] != item[key]:
                errors.append(f"{case_id}:metadata:{key}")
        if row["required"] != item["anchors"]:
            errors.append(f"{case_id}:required")
        rounds = row["rounds"]
        if [round_row["revision"] for round_row in rounds] != list(range(len(rounds))):
            errors.append(f"{case_id}:revision_sequence")
        if len(rounds) > 9:
            errors.append(f"{case_id}:revision_budget")
        for round_row in rounds:
            count = word_count(round_row["text"])
            missing = [
                anchor
                for anchor in item["anchors"]
                if not contains_literal(round_row["text"], anchor)
            ]
            exact = count == item["target"]
            joint = exact and not missing
            revision = round_row["revision"]
            if count != round_row["word_count"]:
                errors.append(f"{case_id}:r{revision}:count")
            if count - item["target"] != round_row["error"]:
                errors.append(f"{case_id}:r{revision}:error")
            if missing != round_row["missing"]:
                errors.append(f"{case_id}:r{revision}:anchors")
            if exact != round_row["exact_length"] or joint != round_row["joint_success"]:
                errors.append(f"{case_id}:r{revision}:flags")
        if (
            row["one_shot_exact"] != rounds[0]["exact_length"]
            or row["one_shot_joint"] != rounds[0]["joint_success"]
            or row["final_exact"] != rounds[-1]["exact_length"]
            or row["final_joint"] != rounds[-1]["joint_success"]
            or row["ever_exact"] != any(round_row["exact_length"] for round_row in rounds)
            or row["revisions_used"] != len(rounds) - 1
        ):
            errors.append(f"{case_id}:summary_flags")
    return errors


def cochran_q(model_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids = [row["id"] for row in model_rows[MODELS[0]]]
    maps = {
        model: {row["id"]: int(bool(row["final_joint"])) for row in rows}
        for model, rows in model_rows.items()
    }
    matrix = np.asarray([[maps[model][case_id] for model in MODELS] for case_id in ids])
    k = matrix.shape[1]
    column = matrix.sum(axis=0)
    row = matrix.sum(axis=1)
    total = float(column.sum())
    denominator = k * total - float((row**2).sum())
    statistic = (
        0.0
        if denominator == 0
        else (k - 1) * (k * float((column**2).sum()) - total**2) / denominator
    )
    return {"q": statistic, "df": k - 1, "p": float(chi2.sf(statistic, k - 1))}


def write_results_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Human Generation Combined-480 Results",
        "",
        "Primary outcome: final joint success (exact deterministic word count and all lexical anchors).",
        "",
        "| Model | Core-240 | Extension-240 | Extension - Core | Combined-480 (95% stratified bootstrap CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        value = report["models"][model]
        core = value["splits"]["core240"]
        extension = value["splits"]["extension240"]
        combined = value["splits"]["combined480"]
        delta = value["extension_minus_core"]
        lo, hi = combined["stratified_bootstrap_95"]
        dlo, dhi = delta["stratified_bootstrap_95"]
        lines.append(
            f"| {LABELS[model]} | {core['successes']}/{core['cases']} ({100*core['rate']:.1f}%) "
            f"| {extension['successes']}/{extension['cases']} ({100*extension['rate']:.1f}%) "
            f"| {100*delta['difference']:+.1f} pp [{100*dlo:+.1f}, {100*dhi:+.1f}] "
            f"| {combined['successes']}/{combined['cases']} ({100*combined['rate']:.1f}%) "
            f"[{100*lo:.1f}, {100*hi:.1f}] |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- Overall audit pass: `{report['integrity']['passes']}`",
        f"- Unique frozen cases: `{report['integrity']['combined_cases']}`",
        f"- Qwen shard merge equality: `{report['integrity']['qwen_shard_merge_equal']}`",
        f"- Recount/anchor/flag errors: `{report['integrity']['error_count']}`",
        "",
        "## Interpretation",
        "",
        f"- Core-to-Extension ordering equality: `{report['combined480_model_comparison']['ordering_equal']}`.",
        "- Llama remains a feasible-controller counterexample, while no model reaches the 95% reliability threshold.",
        "- The extension materially tightens case-level uncertainty without increasing the number of source families.",
        "- Combined-480 inference applies to all six evaluated models.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    data_paths = {
        "primary120": repo / "data/human_generation_main120_v1/generation_main120.jsonl",
        "replication120": repo
        / "data/human_generation_replication120_v1/generation_replication120.jsonl",
        "extension240": repo
        / "data/human_generation_extension240_v1/generation_extension240.jsonl",
    }
    roots = {
        "primary120": repo / "experiments/human_generation_main120_v1",
        "replication120": repo / "experiments/human_generation_replication120_v1",
        "extension240": repo / "experiments/human_generation_extension240_v1",
    }
    output_dir = repo / "experiments/human_generation_combined480_v1"
    data_output_dir = repo / "data/human_generation_combined480_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    data_lists = {name: read_jsonl(path) for name, path in data_paths.items()}
    data_maps = {
        name: {row["id"]: row for row in rows} for name, rows in data_lists.items()
    }
    all_data_rows = (
        data_lists["primary120"] + data_lists["replication120"] + data_lists["extension240"]
    )
    all_ids = [row["id"] for row in all_data_rows]
    source_keys = [(family(row["source"]), str(row["source_id"])) for row in all_data_rows]
    signatures = [
        (normalize(row["instruction"]), normalize(row["reference"])) for row in all_data_rows
    ]
    data_errors = []
    if len(all_ids) != 480 or len(set(all_ids)) != 480:
        data_errors.append("combined_case_identity")
    if len(set(source_keys)) != 480:
        data_errors.append("combined_source_identity")
    if len(set(signatures)) != 480:
        data_errors.append("combined_text_signature")

    result_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    audit_errors: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for model in MODELS:
        for split in ("primary120", "replication120", "extension240"):
            path = roots[split] / model / "cases.jsonl"
            rows = read_jsonl(path)
            result_rows[model][split] = rows
            audit_errors[model][split] = audit_model_rows(rows, data_maps[split])
        result_rows[model]["core240"] = (
            result_rows[model]["primary120"] + result_rows[model]["replication120"]
        )
        result_rows[model]["combined480"] = (
            result_rows[model]["core240"] + result_rows[model]["extension240"]
        )

    qwen_shards = (
        read_jsonl(roots["extension240"] / "qwen3_14b_shards/shard00/cases.jsonl")
        + read_jsonl(roots["extension240"] / "qwen3_14b_shards/shard01/cases.jsonl")
    )
    qwen_shard_map = {row["id"]: row for row in qwen_shards}
    qwen_canonical_map = {
        row["id"]: row for row in result_rows["qwen3_14b"]["extension240"]
    }
    qwen_shard_merge_equal = (
        len(qwen_shards) == 240
        and len(qwen_shard_map) == 240
        and qwen_shard_map == qwen_canonical_map
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    report: dict[str, Any] = {
        "schema_version": 1,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "unit": "independent case, stratified by four source families",
        },
        "inputs": {},
        "models": {},
    }
    input_files = list(data_paths.values())
    for model in MODELS:
        input_files.extend(
            roots[split] / model / "cases.jsonl"
            for split in ("primary120", "replication120", "extension240")
        )
    input_files.extend(
        [
            roots["extension240"] / "qwen3_14b_shards/shard00/cases.jsonl",
            roots["extension240"] / "qwen3_14b_shards/shard01/cases.jsonl",
        ]
    )
    report["inputs"] = {str(path.relative_to(repo)): digest(path) for path in input_files}

    for model in MODELS:
        split_report = {
            split: rate_summary(result_rows[model][split], "final_joint", rng)
            for split in (
                "primary120",
                "replication120",
                "core240",
                "extension240",
                "combined480",
            )
        }
        difference = (
            split_report["extension240"]["rate"] - split_report["core240"]["rate"]
        )
        difference_ci = stratified_bootstrap_independent_difference(
            result_rows[model]["extension240"],
            result_rows[model]["core240"],
            "final_joint",
            rng,
        )
        by_family = {}
        by_band = {}
        for value, target in (("source_family", by_family), ("length_band", by_band)):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in result_rows[model]["combined480"]:
                key = family(row["source"]) if value == "source_family" else row["length_band"]
                groups[key].append(row)
            for key, rows in sorted(groups.items()):
                successes = sum(bool(row["final_joint"]) for row in rows)
                target[key] = {
                    "cases": len(rows),
                    "successes": successes,
                    "rate": successes / len(rows),
                }
        report["models"][model] = {
            "label": LABELS[model],
            "splits": split_report,
            "extension_minus_core": {
                "difference": difference,
                "stratified_bootstrap_95": difference_ci,
                "ci_contains_zero": difference_ci[0] <= 0 <= difference_ci[1],
            },
            "combined480_by_source_family": by_family,
            "combined480_by_length_band": by_band,
            "supporting": {
                "combined480_one_shot_joint": rate_summary(
                    result_rows[model]["combined480"], "one_shot_joint", rng
                ),
                "combined480_final_exact": rate_summary(
                    result_rows[model]["combined480"], "final_exact", rng
                ),
            },
        }

    pairwise = []
    for left, right in (("llama31_8b", peer) for peer in MODELS[1:]):
        left_rows = result_rows[left]["combined480"]
        right_rows = result_rows[right]["combined480"]
        left_rate = sum(bool(row["final_joint"]) for row in left_rows) / 480
        right_rate = sum(bool(row["final_joint"]) for row in right_rows) / 480
        row = {
            "left": left,
            "right": right,
            "effect": left_rate - right_rate,
            "paired_stratified_bootstrap_95": stratified_bootstrap_paired_difference(
                left_rows, right_rows, "final_joint", rng
            ),
        }
        row.update(exact_mcnemar(left_rows, right_rows))
        pairwise.append(row)
    holm(pairwise)
    core_ordering = sorted(
        MODELS,
        key=lambda model: report["models"][model]["splits"]["core240"]["rate"],
        reverse=True,
    )
    extension_ordering = sorted(
        MODELS,
        key=lambda model: report["models"][model]["splits"]["extension240"]["rate"],
        reverse=True,
    )
    report["combined480_model_comparison"] = {
        "cochran_q": cochran_q(
            {model: result_rows[model]["combined480"] for model in MODELS}
        ),
        "core_ordering": core_ordering,
        "extension_ordering": extension_ordering,
        "ordering_equal": core_ordering == extension_ordering,
        "pairwise_mcnemar": pairwise,
    }

    flat_errors = data_errors + [
        f"{model}:{split}:{error}"
        for model, splits in audit_errors.items()
        for split, errors in splits.items()
        for error in errors
    ]
    if not qwen_shard_merge_equal:
        flat_errors.append("qwen_shard_merge")
    report["integrity"] = {
        "passes": not flat_errors,
        "combined_cases": len(all_ids),
        "unique_case_ids": len(set(all_ids)),
        "unique_source_keys": len(set(source_keys)),
        "unique_normalized_instruction_reference_pairs": len(set(signatures)),
        "qwen_shard_merge_equal": qwen_shard_merge_equal,
        "audit_errors": audit_errors,
        "errors": flat_errors,
        "error_count": len(flat_errors),
    }

    combined_data_path = data_output_dir / "generation_combined480.jsonl"
    combined_data_path.write_bytes(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_data_rows
        ).encode("utf-8")
    )
    data_manifest = {
        "schema_version": 1,
        "status": "frozen_after_all_component_model_runs",
        "cases": 480,
        "source_families": dict(Counter(family(row["source"]) for row in all_data_rows)),
        "components": {str(path.relative_to(repo)): digest(path) for path in data_paths.values()},
        "output": {
            str(combined_data_path.relative_to(repo)): digest(combined_data_path)
        },
    }
    (data_output_dir / "manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_results_markdown(output_dir / "RESULTS.md", report)

    csv_rows = []
    for model in MODELS:
        value = report["models"][model]
        for split in ("core240", "extension240", "combined480"):
            stats = value["splits"][split]
            csv_rows.append(
                {
                    "model": model,
                    "split": split,
                    "cases": stats["cases"],
                    "successes": stats["successes"],
                    "rate": stats["rate"],
                    "bootstrap_lo": stats["stratified_bootstrap_95"][0],
                    "bootstrap_hi": stats["stratified_bootstrap_95"][1],
                }
            )
    with (output_dir / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["integrity"]["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
