#!/usr/bin/env python3
"""Derived, construct-separated analysis of API endpoint Stanza/NLI annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "fragment_rate_change",
    "agreement_rate_change",
    "dangling_rate_change",
    "adjacent_entity_overlap_change",
    "adjacent_contradiction_mean_change",
    "draft_entity_retention",
    "draft_proper_name_retention",
    "draft_symbolic_retention",
    "reference_entity_coverage_final",
    "reference_proper_name_coverage_final",
    "reference_symbolic_coverage_final",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def set_retention(source: set[str], output: set[str]) -> float | None:
    return ratio(len(source & output), len(source))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def text_metrics(stanza_row: dict[str, Any], nli_row: dict[str, Any] | None) -> dict[str, Any]:
    sentences = int(stanza_row["sentence_count"])
    overlap = stanza_row["adjacent_entity_overlap"]
    entity_boundaries = [
        row
        for row in overlap
        if row["jaccard"] is not None
    ]
    adjacent_probs = [
        pair["probabilities"]["contradiction"]
        for pair in (nli_row or {}).get("pairs", [])
        if pair["pair_kind"] == "adjacent"
    ]
    linked_probs = [
        pair["probabilities"]["contradiction"]
        for pair in (nli_row or {}).get("pairs", [])
        if pair["pair_kind"] == "entity_linked"
    ]
    return {
        "sentence_count": sentences,
        "fragment_rate": ratio(len(stanza_row["fragment_rule_hits"]), sentences),
        "agreement_rate": ratio(len(stanza_row["agreement_mismatch_hits"]), sentences),
        "dangling_rate": ratio(len(stanza_row["dangling_conjunction_hits"]), sentences),
        "adjacent_entity_overlap_rate_all_boundaries": ratio(
            sum(row["has_overlap"] for row in overlap), len(overlap)
        ),
        "adjacent_entity_overlap_rate_entity_boundaries": ratio(
            sum(row["has_overlap"] for row in entity_boundaries), len(entity_boundaries)
        ),
        "adjacent_contradiction_mean": mean(adjacent_probs),
        "adjacent_contradiction_max": max(adjacent_probs) if adjacent_probs else None,
        "entity_linked_contradiction_mean": mean(linked_probs),
        "entity_linked_contradiction_max": max(linked_probs) if linked_probs else None,
        "entity_units": {
            f"{item['type']}:{item['normalized']}" for item in stanza_row["entities"]
        },
        "proper_name_units": {item["normalized"] for item in stanza_row["proper_names"]},
        "symbolic_units": {item["normalized"] for item in stanza_row["symbolic_units"]},
    }


def difference(final: float | None, initial: float | None) -> float | None:
    return final - initial if final is not None and initial is not None else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"trajectories": len(rows), "metrics": {}}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        output["metrics"][metric] = {
            "available": len(values),
            "missing": len(rows) - len(values),
            "mean": mean(values),
            "median": median(values),
            "q25": (
                values[0]
                if len(values) == 1
                else statistics.quantiles(values, n=4, method="inclusive")[0]
                if values
                else None
            ),
            "q75": (
                values[0]
                if len(values) == 1
                else statistics.quantiles(values, n=4, method="inclusive")[2]
                if values
                else None
            ),
        }
    return output


def bootstrap_pair_difference(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    sources: dict[str, str],
    metric: str,
    rng: random.Random,
    replicates: int,
) -> dict[str, Any]:
    eligible = [
        case_id
        for case_id in sorted(left.keys() & right.keys())
        if left[case_id].get(metric) is not None and right[case_id].get(metric) is not None
    ]
    strata: dict[str, list[str]] = defaultdict(list)
    for case_id in eligible:
        strata[sources[case_id]].append(case_id)
    observed = mean([left[i][metric] - right[i][metric] for i in eligible])
    draws = []
    for _ in range(replicates):
        sample = [
            case_id
            for ids in strata.values()
            for case_id in rng.choices(ids, k=len(ids))
        ]
        draws.append(mean([left[i][metric] - right[i][metric] for i in sample]))
    draws = sorted(value for value in draws if value is not None)
    def q(p: float) -> float | None:
        if not draws:
            return None
        index = (len(draws) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return draws[lo] if lo == hi else draws[lo] * (hi - index) + draws[hi] * (index - lo)
    return {
        "paired_cases": len(eligible),
        "mean_paired_difference": observed,
        "source_stratified_bootstrap_95": [q(0.025), q(0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--stanza", required=True, type=Path)
    parser.add_argument("--nli", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    trajectories = read_jsonl(args.trajectories)
    models = tuple(dict.fromkeys(row["model"] for row in trajectories))
    if not models:
        raise RuntimeError("no models found in trajectory registry")
    stanza = {
        row["text_sha256"]: row for row in read_jsonl(args.stanza) if row["status"] == "ok"
    }
    nli = {row["text_sha256"]: row for row in read_jsonl(args.nli) if row["status"] == "ok"}
    derived: dict[str, dict[str, Any]] = {
        text_hash: text_metrics(row, nli.get(text_hash)) for text_hash, row in stanza.items()
    }
    rows = []
    for trajectory in trajectories:
        hashes = {
            kind: trajectory[f"{kind}_sha256"]
            for kind in ("prompt", "reference", "round0", "final")
        }
        if hashes["round0"] is None or hashes["final"] is None:
            rows.append(
                {
                    "model": trajectory["model"],
                    "id": trajectory["id"],
                    "source": trajectory["source"],
                    "length_band": trajectory["length_band"],
                    "status": "not_applicable_provider_refusal",
                }
            )
            continue
        if any(text_hash not in derived for text_hash in hashes.values()):
            rows.append(
                {
                    "model": trajectory["model"],
                    "id": trajectory["id"],
                    "source": trajectory["source"],
                    "length_band": trajectory["length_band"],
                    "status": "abstain_missing_annotation",
                }
            )
            continue
        metrics = {kind: derived[text_hash] for kind, text_hash in hashes.items()}
        row = {
            "model": trajectory["model"],
            "id": trajectory["id"],
            "source": trajectory["source"],
            "length_band": trajectory["length_band"],
            "final_joint": trajectory["final_joint"],
            "status": "ok",
            "fragment_rate_change": difference(
                metrics["final"]["fragment_rate"], metrics["round0"]["fragment_rate"]
            ),
            "agreement_rate_change": difference(
                metrics["final"]["agreement_rate"], metrics["round0"]["agreement_rate"]
            ),
            "dangling_rate_change": difference(
                metrics["final"]["dangling_rate"], metrics["round0"]["dangling_rate"]
            ),
            "adjacent_entity_overlap_change": difference(
                metrics["final"]["adjacent_entity_overlap_rate_all_boundaries"],
                metrics["round0"]["adjacent_entity_overlap_rate_all_boundaries"],
            ),
            "adjacent_contradiction_mean_change": difference(
                metrics["final"]["adjacent_contradiction_mean"],
                metrics["round0"]["adjacent_contradiction_mean"],
            ),
            "draft_entity_retention": set_retention(
                metrics["round0"]["entity_units"], metrics["final"]["entity_units"]
            ),
            "draft_proper_name_retention": set_retention(
                metrics["round0"]["proper_name_units"], metrics["final"]["proper_name_units"]
            ),
            "draft_symbolic_retention": set_retention(
                metrics["round0"]["symbolic_units"], metrics["final"]["symbolic_units"]
            ),
            "reference_entity_coverage_final": set_retention(
                metrics["reference"]["entity_units"], metrics["final"]["entity_units"]
            ),
            "reference_proper_name_coverage_final": set_retention(
                metrics["reference"]["proper_name_units"], metrics["final"]["proper_name_units"]
            ),
            "reference_symbolic_coverage_final": set_retention(
                metrics["reference"]["symbolic_units"], metrics["final"]["symbolic_units"]
            ),
        }
        rows.append(row)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Separate Stanza-conditioned syntax/entity observables and "
            "DeBERTa-conditioned local contradiction probabilities; no overall quality score."
        ),
        "coverage": {
            "trajectories": len(rows),
            "ok": len(ok_rows),
            "provider_refusal_not_applicable": sum(
                row["status"] == "not_applicable_provider_refusal" for row in rows
            ),
            "annotation_abstain": sum(
                row["status"] == "abstain_missing_annotation" for row in rows
            ),
        },
        "models": {},
        "paired_model_differences": {},
    }
    for model in models:
        selected = [row for row in ok_rows if row["model"] == model]
        report["models"][model] = {
            **summarize(selected),
            "by_final_joint": {
                "success": summarize([row for row in selected if row["final_joint"]]),
                "failure": summarize([row for row in selected if not row["final_joint"]]),
            },
            "by_source": {
                source: summarize([row for row in selected if row["source"] == source])
                for source in sorted({row["source"] for row in selected})
            },
            "by_length_band": {
                band: summarize([row for row in selected if row["length_band"] == band])
                for band in sorted({row["length_band"] for row in selected})
            },
        }
    sources = {row["id"]: row["source"] for row in rows}
    indexed = {
        model: {row["id"]: row for row in ok_rows if row["model"] == model}
        for model in models
    }
    rng = random.Random(20260727)
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            key = f"{left}__minus__{right}"
            report["paired_model_differences"][key] = {
                metric: bootstrap_pair_difference(
                    indexed[left], indexed[right], sources, metric, rng, args.bootstrap
                )
                for metric in METRICS
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "endpoint_annotation_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "model", "id", "source", "length_band", "final_joint", "status", *METRICS
    ]
    with (args.output_dir / "endpoint_trajectory_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
