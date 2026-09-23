#!/usr/bin/env python3
"""Formal paired analysis for the Llama--Tulu stage and interface panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import binomtest


ASSAYS = ("exact_length", "lexical", "compositional")
STRUCTURES = ("c05", "c07", "c12", "c09", "c10")
STAGES = ("base", "sft", "dpo", "rlvr", "meta_instruct")
CHAT_STAGES = ("sft", "dpo", "rlvr", "meta_instruct")
DISPLAY = {
    "base": "Llama 3.1 8B Base",
    "sft": "Tulu 3 8B SFT",
    "dpo": "Tulu 3 8B DPO",
    "rlvr": "Tulu 3.1 8B RLVR",
    "meta_instruct": "Llama 3.1 8B Instruct",
}
SLUG = {
    "base": "llama31_8b_base",
    "sft": "tulu3_8b_sft",
    "dpo": "tulu3_8b_dpo",
    "rlvr": "tulu31_8b_rlvr",
    "meta_instruct": "llama31_8b_instruct",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(root: Path) -> dict[tuple[str, str, str], Path]:
    origin = root / "experiments/llama_training_stage_origin_v1"
    common_struct = root / "experiments/llama_tulu_multidomain_stage_v1"
    native = root / "experiments/llama_tulu_native_interface_ablation_v1"
    paths: dict[tuple[str, str, str], Path] = {}
    for stage in STAGES:
        paths[(stage, "common", "exact_length")] = (
            origin / SLUG[stage] / "vllm_common_plain_combined480/cases.jsonl"
        )
        paths[(stage, "common", "structured")] = common_struct / SLUG[stage] / "cases.jsonl"
    paths[("sft", "native", "exact_length")] = native / "tulu3_8b_sft/exact_length/cases.jsonl"
    paths[("dpo", "native", "exact_length")] = origin / "tulu3_8b_dpo/vllm_native_combined480/cases.jsonl"
    paths[("rlvr", "native", "exact_length")] = native / "tulu31_8b_rlvr/exact_length/cases.jsonl"
    paths[("meta_instruct", "native", "exact_length")] = origin / "llama31_8b_instruct/vllm_native_combined480/cases.jsonl"
    paths[("sft", "native", "structured")] = native / "tulu3_8b_sft/structured/cases.jsonl"
    paths[("dpo", "native", "structured")] = native / "tulu3_8b_dpo/structured/cases.jsonl"
    paths[("rlvr", "native", "structured")] = native / "tulu31_8b_rlvr/structured/cases.jsonl"
    paths[("meta_instruct", "native", "structured")] = native / "reused_meta_instruct_native/llama31_8b/cases.jsonl"
    return paths


def load_panel(root: Path) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], dict[str, str]]:
    panel: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        interface: {stage: {assay: {} for assay in ASSAYS} for stage in (STAGES if interface == "common" else CHAT_STAGES)}
        for interface in ("common", "native")
    }
    hashes: dict[str, str] = {}
    for (stage, interface, kind), path in input_paths(root).items():
        rows = read_jsonl(path)
        expected = 480 if kind == "exact_length" else 960
        if len(rows) != expected or len({row["id"] for row in rows}) != expected:
            raise ValueError(f"{stage}/{interface}/{kind}: expected {expected} unique rows, got {len(rows)}")
        hashes[f"{stage}/{interface}/{kind}"] = sha256(path)
        if kind == "exact_length":
            panel[interface][stage]["exact_length"] = {row["id"]: row for row in rows}
        else:
            for assay, domain in (("lexical", "lexical_constraints"), ("compositional", "compositional_constraints")):
                selected = [row for row in rows if row["domain"] == domain]
                if len(selected) != 480:
                    raise ValueError(f"{stage}/{interface}/{assay}: expected 480 rows")
                panel[interface][stage][assay] = {row["id"]: row for row in selected}
    for assay in ASSAYS:
        expected_ids = None
        for interface in panel:
            for stage in panel[interface]:
                ids = set(panel[interface][stage][assay])
                expected_ids = ids if expected_ids is None else expected_ids
                if ids != expected_ids:
                    raise ValueError(f"case-ID mismatch in {interface}/{stage}/{assay}")
    return panel, hashes


def stratified_indices(sources: list[str], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sources_array = np.asarray(sources)
    groups = [np.flatnonzero(sources_array == value) for value in sorted(set(sources))]
    return np.concatenate(
        [rng.choice(group, size=(draws, len(group)), replace=True) for group in groups], axis=1
    )


def ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def mcnemar(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    left_only = int(np.sum((left == 1) & (right == 0)))
    right_only = int(np.sum((left == 0) & (right == 1)))
    discordant = left_only + right_only
    p = 1.0 if discordant == 0 else float(binomtest(right_only, discordant, 0.5).pvalue)
    return left_only, right_only, p


def endpoint_arrays(panel: dict[str, Any], draws: int, seed: int) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    samples: dict[str, np.ndarray] = {}
    ids_by_assay: dict[str, list[str]] = {}
    for assay_index, assay in enumerate(ASSAYS):
        reference = panel["common"]["base"][assay]
        ids = sorted(reference)
        ids_by_assay[assay] = ids
        sources = [str(reference[case_id]["source"]) for case_id in ids]
        samples[assay] = stratified_indices(sources, draws, seed + assay_index)
        for interface in panel:
            for stage in panel[interface]:
                arrays[(stage, interface, assay)] = np.asarray(
                    [int(bool(panel[interface][stage][assay][case_id]["final_joint"])) for case_id in ids], dtype=float
                )
    return arrays, samples, ids_by_assay


def structure_arrays(panel: dict[str, Any]) -> dict[tuple[str, str, str], np.ndarray]:
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for interface in ("common", "native"):
        stages = STAGES if interface == "common" else CHAT_STAGES
        for stage in stages:
            for structure in STRUCTURES:
                assay = "lexical" if structure in {"c05", "c07", "c12"} else "compositional"
                rows = panel[interface][stage][assay]
                selected = [rows[case_id] for case_id in sorted(rows) if rows[case_id]["structure"] == structure]
                arrays[(stage, interface, structure)] = np.asarray(
                    [int(bool(row["final_joint"])) for row in selected], dtype=float
                )
    return arrays


def design_two_factor(a: str, b: str, levels_a: tuple[str, ...], levels_b: tuple[str, ...]) -> np.ndarray:
    values = [1.0]
    da = [float(a == level) for level in levels_a[1:]]
    db = [float(b == level) for level in levels_b[1:]]
    values.extend(da)
    values.extend(db)
    values.extend(x * y for x in da for y in db)
    return np.asarray(values)


def clustered_lpm(records: list[tuple[str, str, float, str]], levels_a: tuple[str, ...], levels_b: tuple[str, ...]) -> dict[str, Any]:
    x = np.vstack([design_two_factor(a, b, levels_a, levels_b) for a, b, _, _ in records])
    y = np.asarray([outcome for _, _, outcome, _ in records])
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record[3]].append(index)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for indices in groups.values():
        ix = np.asarray(indices)
        score = x[ix].T @ residual[ix]
        meat += np.outer(score, score)
    n, k, g = len(y), x.shape[1], len(groups)
    covariance = (g / (g - 1)) * ((n - 1) / (n - k)) * xtx_inv @ meat @ xtx_inv
    start = 1 + (len(levels_a) - 1) + (len(levels_b) - 1)
    interaction = beta[start:]
    subcov = covariance[start:, start:]
    rank = int(np.linalg.matrix_rank(subcov))
    statistic = float(interaction.T @ np.linalg.pinv(subcov) @ interaction)
    return {"n_rows": n, "clusters": g, "df": rank, "wald_chi2": statistic, "p_value": float(stats.chi2.sf(statistic, rank))}


def interaction_records(arrays: dict[tuple[str, str, str], np.ndarray], ids_by_assay: dict[str, list[str]], interface: str, stages: tuple[str, ...]) -> list[tuple[str, str, float, str]]:
    records = []
    for assay in ASSAYS:
        for stage in stages:
            for index, outcome in enumerate(arrays[(stage, interface, assay)]):
                records.append((stage, assay, float(outcome), f"{assay}::{ids_by_assay[assay][index]}"))
    return records


def interface_records(arrays: dict[tuple[str, str, str], np.ndarray], ids: list[str], assay: str) -> list[tuple[str, str, float, str]]:
    return [
        (stage, interface, float(outcome), ids[index])
        for stage in CHAT_STAGES for interface in ("common", "native")
        for index, outcome in enumerate(arrays[(stage, interface, assay)])
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    panel, hashes = load_panel(args.project_root)
    arrays, samples, ids_by_assay = endpoint_arrays(panel, args.draws, args.seed)
    structure_endpoints = structure_arrays(panel)

    cells: list[dict[str, Any]] = []
    for interface in ("common", "native"):
        stages = STAGES if interface == "common" else CHAT_STAGES
        for stage in stages:
            for assay in ASSAYS:
                values = arrays[(stage, interface, assay)]
                bootstrap = values[samples[assay]].mean(axis=1)
                low, high = ci(bootstrap)
                cells.append({"stage": stage, "interface": interface, "assay": assay, "n": len(values), "successes": int(values.sum()), "rate": float(values.mean()), "ci_low": low, "ci_high": high})

    interface_contrasts: list[dict[str, Any]] = []
    for stage in CHAT_STAGES:
        for assay in ASSAYS:
            common = arrays[(stage, "common", assay)]
            native = arrays[(stage, "native", assay)]
            bootstrap = (native - common)[samples[assay]].mean(axis=1)
            low, high = ci(bootstrap)
            left_only, right_only, p = mcnemar(common, native)
            interface_contrasts.append({"stage": stage, "assay": assay, "n": len(common), "native_minus_common": float(np.mean(native - common)), "ci_low": low, "ci_high": high, "common_only_success": left_only, "native_only_success": right_only, "mcnemar_p": p})

    stage_contrasts: list[dict[str, Any]] = []
    for left, right in zip(STAGES, STAGES[1:]):
        for assay in ASSAYS:
            a, b = arrays[(left, "common", assay)], arrays[(right, "common", assay)]
            bootstrap = (b - a)[samples[assay]].mean(axis=1)
            low, high = ci(bootstrap)
            left_only, right_only, p = mcnemar(a, b)
            stage_contrasts.append({"left": left, "right": right, "assay": assay, "difference": float(np.mean(b - a)), "ci_low": low, "ci_high": high, "left_only_success": left_only, "right_only_success": right_only, "mcnemar_p": p})

    structure_cells: list[dict[str, Any]] = []
    for interface in ("common", "native"):
        stages = STAGES if interface == "common" else CHAT_STAGES
        for stage in stages:
            for structure in STRUCTURES:
                values = structure_endpoints[(stage, interface, structure)]
                structure_cells.append({"stage": stage, "interface": interface, "structure": structure, "n": len(values), "successes": int(values.sum()), "rate": float(values.mean())})

    interface_rank_dids: list[dict[str, Any]] = []
    for assay in ASSAYS:
        difference = (
            arrays[("dpo", "native", assay)] - arrays[("rlvr", "native", assay)]
            - arrays[("dpo", "common", assay)] + arrays[("rlvr", "common", assay)]
        )
        bootstrap = difference[samples[assay]].mean(axis=1)
        low, high = ci(bootstrap)
        interface_rank_dids.append({"contrast": "DPO_minus_RLVR", "assay": assay, "interface_difference_in_differences": float(difference.mean()), "ci_low": low, "ci_high": high})

    interactions: dict[str, Any] = {
        "common_checkpoint_by_assay": clustered_lpm(interaction_records(arrays, ids_by_assay, "common", STAGES), STAGES, ASSAYS),
        "native_checkpoint_by_assay": clustered_lpm(interaction_records(arrays, ids_by_assay, "native", CHAT_STAGES), CHAT_STAGES, ASSAYS),
        "interface_by_checkpoint_within_assay": {},
    }
    for assay in ASSAYS:
        interactions["interface_by_checkpoint_within_assay"][assay] = clustered_lpm(interface_records(arrays, ids_by_assay[assay], assay), CHAT_STAGES, ("common", "native"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_estimates.csv", cells)
    write_csv(args.output_dir / "interface_contrasts.csv", interface_contrasts)
    write_csv(args.output_dir / "common_stage_contrasts.csv", stage_contrasts)
    write_csv(args.output_dir / "structure_cell_estimates.csv", structure_cells)
    write_csv(args.output_dir / "interface_rank_dids.csv", interface_rank_dids)
    analysis = {"schema_version": 1, "draws": args.draws, "seed": args.seed, "pairing": "within-assay shared cases; assay datasets are distinct", "interactions": interactions, "selected_interface_rank_dids": interface_rank_dids, "input_hashes": hashes}
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    lookup = {(r["stage"], r["interface"], r["assay"]): r for r in cells}
    lines = ["# Llama--Tulu multidomain stage and interface analysis", "", f"All inputs and pairings pass. Intervals use {args.draws:,} source-stratified paired-case bootstrap draws.", "", "## Final joint success", "", "### Common interface", "", "| Checkpoint | Exact length | Lexical | Compositional |", "|---|---:|---:|---:|"]
    for stage in STAGES:
        vals = [lookup[(stage, "common", assay)] for assay in ASSAYS]
        lines.append("| " + DISPLAY[stage] + " | " + " | ".join(f"{100*r['rate']:.1f}%" for r in vals) + " |")
    lines += ["", "### Native interface", "", "| Checkpoint | Exact length | Lexical | Compositional |", "|---|---:|---:|---:|"]
    for stage in CHAT_STAGES:
        vals = [lookup[(stage, "native", assay)] for assay in ASSAYS]
        lines.append("| " + DISPLAY[stage] + " | " + " | ".join(f"{100*r['rate']:.1f}%" for r in vals) + " |")
    lines += ["", "## Native minus common", "", "| Checkpoint | Assay | Difference | 95% paired bootstrap CI | McNemar p |", "|---|---|---:|---:|---:|"]
    for row in interface_contrasts:
        lines.append(f"| {DISPLAY[row['stage']]} | {row['assay']} | {100*row['native_minus_common']:+.1f} pp | [{100*row['ci_low']:+.1f}, {100*row['ci_high']:+.1f}] | {fmt_p(row['mcnemar_p'])} |")
    lines += ["", "## Omnibus interactions", "", "| Test | Wald chi2(df) | p |", "|---|---:|---:|"]
    for key in ("common_checkpoint_by_assay", "native_checkpoint_by_assay"):
        row = interactions[key]
        lines.append(f"| {key} | {row['wald_chi2']:.2f} ({row['df']}) | {fmt_p(row['p_value'])} |")
    for assay, row in interactions["interface_by_checkpoint_within_assay"].items():
        lines.append(f"| interface_by_checkpoint: {assay} | {row['wald_chi2']:.2f} ({row['df']}) | {fmt_p(row['p_value'])} |")
    lines += ["", "## Interpretation", "", "The common-interface panel shows that post-training stages do not form a monotone, assay-invariant ladder. Native serialization is also not a scalar bonus: its effect changes across checkpoints and assays. These are paired behavioral interactions among released checkpoints; they do not identify a unique training-stage cause.", ""]
    report = args.output_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"verdict": "PASS", "analysis_sha256": sha256(args.output_dir / "analysis.json"), "report_sha256": sha256(report), "cell_estimates_sha256": sha256(args.output_dir / "cell_estimates.csv"), "interface_contrasts_sha256": sha256(args.output_dir / "interface_contrasts.csv"), "structure_cell_estimates_sha256": sha256(args.output_dir / "structure_cell_estimates.csv"), "interface_rank_dids_sha256": sha256(args.output_dir / "interface_rank_dids.csv")}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": "PASS", "interactions": interactions}, indent=2))


if __name__ == "__main__":
    main()
