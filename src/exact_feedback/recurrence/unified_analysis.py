#!/usr/bin/env python3
"""Unified 12 x 480 audit of exact attractors and local-basin escapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


SEED = 20260731
BOOTSTRAPS = 10000
PERMUTATIONS = 100000
WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", flags=re.UNICODE)
MODELS = (
    "llama31_8b_instruct",
    "gemma2_9b_it",
    "glm4_9b",
    "ministral_8b",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
    "qwen3_30b_a3b",
    "granite_3_3_8b",
    "falcon_h1_7b",
)
LINEAGES = {
    "llama31_8b_instruct": "Llama",
    "gemma2_9b_it": "Gemma",
    "glm4_9b": "GLM",
    "ministral_8b": "Mistral",
    "qwen3_1_7b": "Qwen",
    "qwen3_4b": "Qwen",
    "qwen3_8b": "Qwen",
    "qwen3_14b": "Qwen",
    "qwen3_32b": "Qwen",
    "qwen3_30b_a3b": "Qwen",
    "granite_3_3_8b": "Granite",
    "falcon_h1_7b": "Falcon-H1",
}
THRESHOLDS = (2, 5, 10, 20, 50)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_family(source: str) -> str:
    for prefix in ("dolly", "no_robots", "oasst1", "writingprompts"):
        if source.startswith(prefix):
            return prefix
    return source


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def contains_literal(text: str, phrase: str) -> bool:
    tokens = WORD_RE.findall(phrase)
    if not tokens:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in tokens) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def main_registry(root: Path) -> dict[str, list[Path]]:
    exp = root / "experiments"
    old = {
        "llama31_8b_instruct": "llama31_8b",
        "gemma2_9b_it": "gemma2_9b",
        "glm4_9b": "glm4_9b",
        "ministral_8b": "ministral_8b",
        "qwen3_8b": "qwen3_8b",
        "qwen3_14b": "qwen3_14b",
    }
    registry: dict[str, list[Path]] = {}
    for model, directory in old.items():
        registry[model] = [
            exp / f"human_generation_main120_v1/{directory}/cases.jsonl",
            exp / f"human_generation_replication120_v1/{directory}/cases.jsonl",
            exp / f"human_generation_extension240_v1/{directory}/cases.jsonl",
        ]
    registry.update({
        "qwen3_1_7b": [exp / "model_generality_combined480_v1/qwen3_1_7b/combined480/cases.jsonl"],
        "qwen3_4b": [exp / "model_generality_combined480_v1/qwen3_4b/combined480/cases.jsonl"],
        "qwen3_32b": [exp / "model_generality_combined480_v1/qwen3_32b/vllm_pp2_combined480/cases.jsonl"],
        "qwen3_30b_a3b": [exp / "model_generality_combined480_v1/qwen3_30b_a3b/vllm_pp2_combined480/cases.jsonl"],
        "granite_3_3_8b": [exp / "model_generality_combined480_v1/granite_3_3_8b/combined480/cases.jsonl"],
        "falcon_h1_7b": [exp / "model_generality_combined480_v1/falcon_h1_7b/combined480/cases.jsonl"],
    })
    return registry


def linked_registry(root: Path) -> dict[str, list[Path]]:
    exp = root / "experiments"
    return {
        "llama31_8b_base": [exp / "llama_training_stage_origin_v1/llama31_8b_base/vllm_common_plain_combined480/cases.jsonl"],
        "llama31_8b_instruct": [exp / "llama_training_stage_origin_v1/llama31_8b_instruct/vllm_common_plain_combined480/cases.jsonl"],
        "qwen3_1_7b": [exp / "model_generality_combined480_v1/qwen3_1_7b/combined480/cases.jsonl"],
        "qwen3_4b": [exp / "model_generality_combined480_v1/qwen3_4b/combined480/cases.jsonl"],
        "qwen3_8b": [
            exp / "human_generation_main120_v1/qwen3_8b/cases.jsonl",
            exp / "human_generation_replication120_v1/qwen3_8b/cases.jsonl",
            exp / "human_generation_extension240_v1/qwen3_8b/cases.jsonl",
        ],
        "qwen3_14b": [
            exp / "human_generation_main120_v1/qwen3_14b/cases.jsonl",
            exp / "human_generation_replication120_v1/qwen3_14b/cases.jsonl",
            exp / "human_generation_extension240_v1/qwen3_14b/cases.jsonl",
        ],
        "gemma2_9b_it": [
            exp / "human_generation_main120_v1/gemma2_9b/cases.jsonl",
            exp / "human_generation_replication120_v1/gemma2_9b/cases.jsonl",
            exp / "human_generation_extension240_v1/gemma2_9b/cases.jsonl",
        ],
    }


def state_tuple(round_row: dict[str, Any]) -> tuple[str, int, tuple[str, ...]]:
    text_hash = hashlib.sha256(str(round_row["text"]).encode("utf-8")).hexdigest()
    return text_hash, int(round_row["error"]), tuple(sorted(str(item).lower() for item in round_row.get("missing", [])))


def analyze_case(model: str, row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rounds = row["rounds"]
    states = [state_tuple(item) for item in rounds]
    _errors = [int(item["error"]) for item in rounds]
    seen: dict[tuple[str, int, tuple[str, ...]], int] = {}
    recurrences: list[tuple[int, int, int]] = []
    for index, state in enumerate(states):
        if state in seen and not bool(rounds[index].get("joint_success", False)):
            previous = seen[state]
            recurrences.append((previous, index, index - previous))
        else:
            seen.setdefault(state, index)
    periods = sorted({period for _, _, period in recurrences})
    any_output_recurrence = int(bool(recurrences))
    persistent_fixed = int(
        not bool(row["final_joint"])
        and len(states) >= 3
        and states[-1] == states[-2] == states[-3]
    )
    persistent_fixed_last2 = int(
        not bool(row["final_joint"])
        and len(states) >= 2
        and states[-1] == states[-2]
    )
    persistent_fixed_last4 = int(
        not bool(row["final_joint"])
        and len(states) >= 4
        and len(set(states[-4:])) == 1
    )
    persistent_cycle_period = 0
    if not bool(row["final_joint"]):
        for period in (2, 3, 4):
            if len(states) < 2 * period:
                continue
            left_block = states[-2 * period : -period]
            right_block = states[-period:]
            if left_block == right_block and len(set(right_block)) >= 2:
                persistent_cycle_period = period
                break
    persistent_cycle = int(persistent_cycle_period > 0)
    period2_three_cycles = int(
        not bool(row["final_joint"])
        and len(states) >= 6
        and states[-6:-4] == states[-4:-2] == states[-2:]
        and len(set(states[-2:])) == 2
    )
    terminal_unconfirmed_period_5_7 = int(
        any(end == len(states) - 1 and 5 <= period <= 7 for _, end, period in recurrences)
    )
    persistent_recurrence = int(bool(persistent_fixed or persistent_cycle))
    landmark4_risk = int(len(states) >= 6)
    landmark_states = states[:5]
    recurrence_by_r4 = int(
        landmark4_risk and len(set(landmark_states)) < len(landmark_states)
    )

    episodes: list[dict[str, Any]] = []
    risk = {threshold: 0 for threshold in THRESHOLDS}
    escape = {threshold: 0 for threshold in THRESHOLDS}
    primary_episode_count = 0
    support_episode_count = 0
    expansion_episode_count = 0
    length_noop_episodes = 0
    exact_noop_episodes = 0
    for index, (left, right) in enumerate(zip(rounds, rounds[1:])):
        error = int(left["error"])
        next_error = int(right["error"])
        for threshold in THRESHOLDS:
            if 0 < abs(error) <= threshold:
                risk[threshold] = 1
                if abs(next_error) > 50:
                    escape[threshold] = 1
        primary = 0 < abs(error) <= 20 and abs(next_error) > 50
        support = 0 < abs(error) <= 50 and abs(next_error) > 50
        expansion = 0 < abs(error) and abs(next_error) >= 50 and abs(next_error) / abs(error) >= 5
        length_noop = error == next_error
        exact_noop = states[index] == states[index + 1]
        primary_episode_count += int(primary)
        support_episode_count += int(support)
        expansion_episode_count += int(expansion)
        length_noop_episodes += int(length_noop)
        exact_noop_episodes += int(exact_noop)
        if primary or support or expansion or exact_noop:
            episodes.append({
                "model": model,
                "case_id": str(row["id"]),
                "source_family": source_family(str(row["source"])),
                "revision": int(left["revision"]),
                "error": error,
                "next_error": next_error,
                "action": next_error - error,
                "primary_escape": int(primary),
                "support_escape": int(support),
                "catastrophic_expansion": int(expansion),
                "exact_noop": int(exact_noop),
                "left_text_sha256": states[index][0],
                "right_text_sha256": states[index + 1][0],
            })

    final_joint = int(bool(row["final_joint"]))
    result: dict[str, Any] = {
        "model": model,
        "lineage": LINEAGES.get(model, model),
        "case_id": str(row["id"]),
        "source_family": source_family(str(row["source"])),
        "target": int(row["target"]),
        "length_band": str(row.get("length_band", "")),
        "rounds": len(rounds),
        "transitions": max(0, len(rounds) - 1),
        "one_shot_joint": int(bool(row["one_shot_joint"])),
        "final_exact": int(bool(row["final_exact"])),
        "final_joint": final_joint,
        "failed": 1 - final_joint,
        "any_output_recurrence": any_output_recurrence,
        "persistent_fixed_output": persistent_fixed,
        "persistent_fixed_last2": persistent_fixed_last2,
        "persistent_fixed_last4": persistent_fixed_last4,
        "persistent_output_cycle": persistent_cycle,
        "persistent_cycle_period": persistent_cycle_period,
        "persistent_period_2": int(persistent_cycle_period == 2),
        "persistent_period_3": int(persistent_cycle_period == 3),
        "persistent_period_4": int(persistent_cycle_period == 4),
        "persistent_period2_three_cycles": period2_three_cycles,
        "terminal_unconfirmed_period_5_7": terminal_unconfirmed_period_5_7,
        "persistent_output_recurrence": persistent_recurrence,
        "landmark4_risk": landmark4_risk,
        "recurrence_by_r4": recurrence_by_r4,
        "landmark4_final_failure": int(bool(landmark4_risk and not final_joint)),
        "primary_basin_risk": risk[20],
        "primary_basin_escape": escape[20],
        "support_basin_risk": risk[50],
        "support_basin_escape": escape[50],
        "catastrophic_expansion": int(expansion_episode_count > 0),
        "instability_burden": int(bool(persistent_recurrence or escape[20])),
        "recurrence_and_primary_escape": int(bool(persistent_recurrence and escape[20])),
        "length_noop": int(length_noop_episodes > 0),
        "exact_noop": int(exact_noop_episodes > 0),
        "primary_escape_episodes": primary_episode_count,
        "support_escape_episodes": support_episode_count,
        "catastrophic_expansion_episodes": expansion_episode_count,
        "length_noop_episodes": length_noop_episodes,
        "exact_noop_episodes": exact_noop_episodes,
        "observed_recurrence_periods": ";".join(str(value) for value in periods),
    }
    for threshold in THRESHOLDS:
        result[f"risk_le_{threshold}"] = risk[threshold]
        result[f"escape_from_le_{threshold}"] = escape[threshold]
    return result, episodes


def validate_rows(model: str, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(row["id"]) for row in rows]
    if len(rows) != 480 or len(set(ids)) != 480:
        errors.append(f"{model}: rows/unique={len(rows)}/{len(set(ids))}")
    for row in rows:
        target = int(row["target"])
        required = [str(item) for item in row["required"]]
        rounds = row.get("rounds", [])
        if not rounds or len(rounds) > 9:
            errors.append(f"{model}/{row['id']}: invalid rounds={len(rounds)}")
            continue
        for expected_revision, item in enumerate(rounds):
            text = str(item["text"])
            counted = word_count(text)
            missing = [anchor for anchor in required if not contains_literal(text, anchor)]
            if int(item["revision"]) != expected_revision:
                errors.append(f"{model}/{row['id']}: revision sequence")
            if counted != int(item["word_count"]) or counted - target != int(item["error"]):
                errors.append(f"{model}/{row['id']}/r{expected_revision}: recount")
            if sorted(value.lower() for value in missing) != sorted(str(value).lower() for value in item.get("missing", [])):
                errors.append(f"{model}/{row['id']}/r{expected_revision}: anchors")
        final = rounds[-1]
        recomputed_exact = int(final["error"]) == 0
        recomputed_joint = recomputed_exact and not final.get("missing", [])
        if bool(row["final_exact"]) != recomputed_exact or bool(row["final_joint"]) != recomputed_joint:
            errors.append(f"{model}/{row['id']}: final flags")
    return errors


def build_weights(case_ids: list[str], source_by_case: dict[str, str], bootstraps: int, seed: int) -> np.ndarray:
    index = {case_id: position for position, case_id in enumerate(case_ids)}
    by_source: dict[str, list[int]] = defaultdict(list)
    for case_id in case_ids:
        by_source[source_by_case[case_id]].append(index[case_id])
    rng = np.random.default_rng(seed)
    weights = np.zeros((bootstraps, len(case_ids)), dtype=np.int16)
    for positions in by_source.values():
        draws = rng.multinomial(len(positions), np.full(len(positions), 1 / len(positions)), size=bootstraps)
        weights[:, np.asarray(positions)] = draws
    return weights


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def spearman(left: list[float], right: list[float]) -> float:
    x = rankdata(np.asarray(left, dtype=float))
    y = rankdata(np.asarray(right, dtype=float))
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def permutation_spearman(left: list[float], right: list[float], permutations: int, seed: int) -> dict[str, Any]:
    observed = spearman(left, right)
    rng = np.random.default_rng(seed)
    right_array = np.asarray(right, dtype=float)
    extreme = 0
    for _ in range(permutations):
        value = spearman(left, rng.permutation(right_array).tolist())
        extreme += int(abs(value) >= abs(observed) - 1e-12)
    return {"rho": observed, "permutation_p_two_sided": (extreme + 1) / (permutations + 1), "permutations": permutations}


def exact_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    observed = spearman(left, right)
    values = list(right)
    total = 0
    extreme = 0
    for permuted in itertools.permutations(values):
        total += 1
        extreme += int(abs(spearman(left, list(permuted))) >= abs(observed) - 1e-12)
    return {"rho": observed, "exact_p_two_sided": extreme / total, "permutations": total}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calibration_bridge(root: Path, excluded_ids: set[str]) -> dict[str, Any]:
    registry = linked_registry(root)
    burdens: dict[str, float] = {}
    for model, paths in registry.items():
        cases: list[dict[str, Any]] = []
        for path in paths:
            cases.extend(read_jsonl(path))
        retained = [row for row in cases if str(row["id"]) not in excluded_ids]
        analyzed = [analyze_case(model, row)[0] for row in retained]
        burdens[model] = float(np.mean([row["instability_burden"] for row in analyzed]))

    regime_path = root / "experiments/policy_to_closed_loop_spotlight_gate_v1/regime_summary.csv"
    with regime_path.open(encoding="utf-8", newline="") as handle:
        regime = {row["model"]: row for row in csv.DictReader(handle)}
    models = list(registry)
    x = np.asarray([burdens[model] for model in models], dtype=float)
    y = np.asarray([
        float(regime[model]["observed_terminal_joint"]) - float(regime[model]["model_simulated_terminal_joint"])
        for model in models
    ], dtype=float)
    baseline_predictions = np.zeros(len(models))
    burden_predictions = np.zeros(len(models))
    for held_out in range(len(models)):
        train = np.asarray([index for index in range(len(models)) if index != held_out])
        baseline_predictions[held_out] = float(np.mean(y[train]))
        design = np.column_stack([np.ones(len(train)), x[train]])
        coefficients = np.linalg.lstsq(design, y[train], rcond=None)[0]
        burden_predictions[held_out] = coefficients[0] + coefficients[1] * x[held_out]
    full_coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(models)), x]), y, rcond=None)[0]
    baseline_mae = float(np.mean(np.abs(y - baseline_predictions)))
    burden_mae = float(np.mean(np.abs(y - burden_predictions)))
    return {
        "models": models,
        "instability_burden": burdens,
        "signed_calibration_residual": {model: float(value) for model, value in zip(models, y)},
        "full_panel_intercept": float(full_coefficients[0]),
        "full_panel_slope": float(full_coefficients[1]),
        "loco_baseline_mae": baseline_mae,
        "loco_burden_mae": burden_mae,
        "relative_loco_mae_reduction": (baseline_mae - burden_mae) / baseline_mae,
        "condition_negative_slope_and_20pct_reduction": bool(full_coefficients[1] < 0 and burden_mae <= 0.8 * baseline_mae),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/unified_output_recurrence_audit_v2"))
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = main_registry(root)
    all_cases: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "models": {},
    }
    integrity_errors: list[str] = []
    reference_contract: dict[str, tuple[Any, ...]] | None = None
    reference_ids: set[str] | None = None
    for model in MODELS:
        rows: list[dict[str, Any]] = []
        files = []
        for path in registry[model]:
            if not path.exists():
                raise FileNotFoundError(path)
            file_rows = read_jsonl(path)
            rows.extend(file_rows)
            files.append({"path": str(path), "rows": len(file_rows), "bytes": path.stat().st_size, "sha256": sha256(path)})
        integrity_errors.extend(validate_rows(model, rows))
        ids = {str(row["id"]) for row in rows}
        contracts = {
            str(row["id"]): (
                str(row["source"]), int(row["target"]), tuple(str(value) for value in row["required"]), str(row.get("length_band", ""))
            )
            for row in rows
        }
        if reference_ids is None:
            reference_ids = ids
            reference_contract = contracts
        else:
            if ids != reference_ids:
                integrity_errors.append(f"{model}: case-ID set mismatch")
            assert reference_contract is not None
            for case_id in sorted(ids & reference_ids):
                if contracts[case_id] != reference_contract[case_id]:
                    integrity_errors.append(f"{model}/{case_id}: frozen contract mismatch")
        manifest["models"][model] = {"files": files, "rows": len(rows), "unique_ids": len(ids)}
        all_cases[model] = rows
    if integrity_errors:
        raise ValueError("integrity audit failed:\n" + "\n".join(integrity_errors[:50]))
    assert reference_ids is not None and reference_contract is not None
    case_ids = sorted(reference_ids)
    manifest["case_set_sha256"] = hashlib.sha256(("\n".join(case_ids) + "\n").encode()).hexdigest()
    manifest["integrity"] = {"passed": True, "errors": 0, "models": len(MODELS), "cases_per_model": len(case_ids)}

    case_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for row in all_cases[model]:
            analyzed, episodes = analyze_case(model, row)
            case_rows.append(analyzed)
            episode_rows.extend(episodes)

    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    model_index = {model: index for index, model in enumerate(MODELS)}
    source_by_case = {case_id: source_family(reference_contract[case_id][0]) for case_id in case_ids}
    weights = build_weights(case_ids, source_by_case, args.bootstraps, args.seed)

    def matrix(metric: str) -> np.ndarray:
        values = np.zeros((len(case_ids), len(MODELS)), dtype=float)
        for row in case_rows:
            values[case_index[row["case_id"]], model_index[row["model"]]] = float(row[metric])
        return values

    metric_names = (
        "final_exact", "final_joint", "primary_basin_risk", "primary_basin_escape",
        "support_basin_escape", "catastrophic_expansion", "any_output_recurrence",
        "persistent_fixed_output", "persistent_output_cycle",
        "persistent_output_recurrence", "landmark4_risk", "recurrence_by_r4",
        "instability_burden",
    )
    matrices = {metric: matrix(metric) for metric in metric_names}
    boot_means = {metric: (weights @ values) / weights.sum(axis=1, keepdims=True) for metric, values in matrices.items()}

    model_summary: list[dict[str, Any]] = []
    for model in MODELS:
        index = model_index[model]
        summary: dict[str, Any] = {"model": model, "lineage": LINEAGES[model], "cases": len(case_ids)}
        for metric in metric_names:
            values = matrices[metric][:, index]
            samples = boot_means[metric][:, index]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_ci_lo"] = float(np.quantile(samples, 0.025))
            summary[f"{metric}_ci_hi"] = float(np.quantile(samples, 0.975))
        risk_values = matrices["primary_basin_risk"][:, index]
        escape_values = matrices["primary_basin_escape"][:, index]
        risk_count = int(risk_values.sum())
        summary["primary_basin_entrants"] = risk_count
        summary["primary_escape_cases"] = int(escape_values.sum())
        summary["primary_escape_hazard"] = float(escape_values.sum() / risk_count) if risk_count else 0.0
        boot_num = weights @ escape_values
        boot_den = weights @ risk_values
        hazard_samples = np.divide(boot_num, boot_den, out=np.zeros_like(boot_num), where=boot_den > 0)
        summary["primary_escape_hazard_ci_lo"] = float(np.quantile(hazard_samples, 0.025))
        summary["primary_escape_hazard_ci_hi"] = float(np.quantile(hazard_samples, 0.975))
        model_rows = [row for row in case_rows if row["model"] == model]
        summary["persistent_period_2_cases"] = sum(row["persistent_period_2"] for row in model_rows)
        summary["persistent_period_3_cases"] = sum(row["persistent_period_3"] for row in model_rows)
        summary["persistent_period_4_cases"] = sum(row["persistent_period_4"] for row in model_rows)
        summary["recurrence_and_escape_cases"] = sum(row["recurrence_and_primary_escape"] for row in model_rows)
        model_summary.append(summary)

    landmark_risk = matrix("landmark4_risk")
    landmark_exposure = matrix("recurrence_by_r4")
    final_failure = 1.0 - matrices["final_joint"]
    landmark_boot_differences: list[np.ndarray] = []
    landmark_points: list[float] = []
    landmark_estimable_models: list[str] = []
    for model in MODELS:
        index = model_index[model]
        exposed = (landmark_risk[:, index] == 1) & (landmark_exposure[:, index] == 1)
        unexposed = (landmark_risk[:, index] == 1) & (landmark_exposure[:, index] == 0)
        if not exposed.any() or not unexposed.any():
            continue
        point = float(np.mean(final_failure[exposed, index]) - np.mean(final_failure[unexposed, index]))
        exp_num = weights[:, exposed] @ final_failure[exposed, index]
        exp_den = weights[:, exposed].sum(axis=1)
        unexp_num = weights[:, unexposed] @ final_failure[unexposed, index]
        unexp_den = weights[:, unexposed].sum(axis=1)
        samples = exp_num / np.maximum(exp_den, 1) - unexp_num / np.maximum(unexp_den, 1)
        landmark_points.append(point)
        landmark_boot_differences.append(samples)
        landmark_estimable_models.append(model)
        summary = next(row for row in model_summary if row["model"] == model)
        summary["landmark4_recurrence_cases"] = int(exposed.sum())
        summary["landmark4_no_recurrence_cases"] = int(unexposed.sum())
        summary["landmark4_failure_risk_difference"] = point
        summary["landmark4_failure_risk_difference_ci_lo"] = float(np.quantile(samples, 0.025))
        summary["landmark4_failure_risk_difference_ci_hi"] = float(np.quantile(samples, 0.975))
    landmark_aggregate_samples = np.vstack(landmark_boot_differences).mean(axis=0)
    landmark_aggregate = {
        "estimate": float(np.mean(landmark_points)),
        "ci95": percentile_interval(landmark_aggregate_samples),
        "estimable_models": landmark_estimable_models,
        "positive_checkpoint_estimates": sum(value > 0 for value in landmark_points),
    }

    threshold_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in case_rows if row["model"] == model]
        for threshold in THRESHOLDS:
            risk_key = f"risk_le_{threshold}"
            escape_key = f"escape_from_le_{threshold}"
            risk_count = sum(row[risk_key] for row in model_rows)
            escape_count = sum(row[escape_key] for row in model_rows)
            threshold_rows.append({
                "model": model,
                "threshold": threshold,
                "risk_cases": risk_count,
                "escape_cases": escape_count,
                "escape_hazard": escape_count / risk_count if risk_count else 0.0,
            })

    recurrence_sensitivity_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in case_rows if row["model"] == model]
        recurrence_sensitivity_rows.append({
            "model": model,
            "failed_cases": sum(row["failed"] for row in model_rows),
            "any_output_recurrence_cases": sum(row["any_output_recurrence"] for row in model_rows),
            "terminal_fixed_last2_cases": sum(row["persistent_fixed_last2"] for row in model_rows),
            "terminal_fixed_last3_primary_cases": sum(row["persistent_fixed_output"] for row in model_rows),
            "terminal_fixed_last4_cases": sum(row["persistent_fixed_last4"] for row in model_rows),
            "terminal_period2_two_cycles_cases": sum(row["persistent_period_2"] for row in model_rows),
            "terminal_period2_three_cycles_cases": sum(row["persistent_period2_three_cycles"] for row in model_rows),
            "terminal_period3_two_cycles_cases": sum(row["persistent_period_3"] for row in model_rows),
            "terminal_period4_two_cycles_cases": sum(row["persistent_period_4"] for row in model_rows),
            "terminal_unconfirmed_period5_7_return_cases": sum(row["terminal_unconfirmed_period_5_7"] for row in model_rows),
        })

    pairwise_rows: list[dict[str, Any]] = []
    for metric in ("primary_basin_escape", "persistent_output_recurrence", "instability_burden"):
        p_values: list[float] = []
        metric_rows: list[dict[str, Any]] = []
        samples = boot_means[metric]
        points = matrices[metric].mean(axis=0)
        for left, right in itertools.combinations(range(len(MODELS)), 2):
            differences = samples[:, left] - samples[:, right]
            nonpositive = int(np.sum(differences <= 0))
            nonnegative = int(np.sum(differences >= 0))
            raw_p = min(1.0, 2 * (min(nonpositive, nonnegative) + 1) / (args.bootstraps + 1))
            result = {
                "metric": metric,
                "left": MODELS[left],
                "right": MODELS[right],
                "risk_difference": float(points[left] - points[right]),
                "ci_lo": float(np.quantile(differences, 0.025)),
                "ci_hi": float(np.quantile(differences, 0.975)),
                "raw_bootstrap_p": raw_p,
            }
            metric_rows.append(result)
            p_values.append(raw_p)
        for result, adjusted in zip(metric_rows, holm_adjust(p_values)):
            result["holm_p"] = adjusted
        pairwise_rows.extend(metric_rows)

    final_success = [row["final_joint"] for row in model_summary]
    association: dict[str, Any] = {}
    for metric in ("persistent_output_recurrence", "primary_escape_hazard", "instability_burden"):
        burden = [row[metric] for row in model_summary]
        association[metric] = permutation_spearman(burden, final_success, args.permutations, args.seed + len(association))

    qwen_models = ["qwen3_1_7b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b"]
    sizes = [1.7, 4.0, 8.0, 14.0, 32.0]
    by_model = {row["model"]: row for row in model_summary}
    qwen_scaling = {
        metric: exact_spearman(sizes, [by_model[model][metric] for model in qwen_models])
        for metric in ("final_joint", "persistent_output_recurrence", "primary_escape_hazard", "instability_burden")
    }

    failed_rows = [row for row in case_rows if row["failed"]]
    persistent_rows = [row for row in case_rows if row["persistent_output_recurrence"]]
    n_persistent = len(persistent_rows)
    lineages_with_persistent_recurrence = sorted({row["lineage"] for row in persistent_rows})

    states_path = root / "data/feedback_policy_response_surface_v1/states.jsonl"
    excluded_ids = {str(row["case_id"]) for row in read_jsonl(states_path)}
    bridge = calibration_bridge(root, excluded_ids)
    checks = {
        "persistent_recurrence_in_at_least_three_lineages": len(lineages_with_persistent_recurrence) >= 3,
        "persistent_recurrence_covers_at_least_ten_percent_of_failures": n_persistent / len(failed_rows) >= 0.10,
        "landmark_failure_risk_difference_lower_ci_above_10pp": landmark_aggregate["ci95"][0] > 0.10,
        "at_least_eight_positive_checkpoint_landmark_differences": landmark_aggregate["positive_checkpoint_estimates"] >= 8,
        "persistent_recurrence_rho_le_minus_0_60_and_p_lt_0_05": association["persistent_output_recurrence"]["rho"] <= -0.60 and association["persistent_output_recurrence"]["permutation_p_two_sided"] < 0.05,
        "calibration_bridge_negative_slope_and_20pct_loco_gain": bridge["condition_negative_slope_and_20pct_reduction"],
    }
    if all(checks.values()):
        verdict = "STRONG_OUTPUT_DYNAMICS_ACCOUNT"
    elif all(checks[key] for key in (
        "persistent_recurrence_in_at_least_three_lineages",
        "persistent_recurrence_covers_at_least_ten_percent_of_failures",
        "landmark_failure_risk_difference_lower_ci_above_10pp",
        "at_least_eight_positive_checkpoint_landmark_differences",
        "persistent_recurrence_rho_le_minus_0_60_and_p_lt_0_05",
    )):
        verdict = "ROBUST_OUTPUT_RECURRENCE_TAXONOMY"
    else:
        verdict = "DESCRIPTIVE_OUTPUT_RECURRENCE_ONLY"

    report = {
        "schema_version": 1,
        "verdict": verdict,
        "protocol": "docs/ICLR_UNIFIED_OUTPUT_RECURRENCE_PROTOCOL_V2_20260731.md",
        "models": list(MODELS),
        "population": {"models": len(MODELS), "cases_per_model": len(case_ids), "trajectories": len(case_rows)},
        "integrity": manifest["integrity"],
        "aggregate": {
            "failed_model_cases": len(failed_rows),
            "persistent_output_recurrence_model_cases": n_persistent,
            "persistent_output_recurrence_share_of_failures": n_persistent / len(failed_rows),
            "lineages_with_persistent_output_recurrence": lineages_with_persistent_recurrence,
            "landmark4": landmark_aggregate,
        },
        "model_level_association": association,
        "qwen_dense_scaling": qwen_scaling,
        "calibration_residual_bridge": bridge,
        "checks": checks,
        "input_manifest": "input_manifest.json",
    }
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "case_level.csv", case_rows)
    write_csv(output_dir / "event_level.csv", episode_rows)
    write_csv(output_dir / "model_summary.csv", model_summary)
    write_csv(output_dir / "threshold_sensitivity.csv", threshold_rows)
    write_csv(output_dir / "recurrence_sensitivity.csv", recurrence_sensitivity_rows)
    write_csv(output_dir / "paired_comparisons.csv", pairwise_rows)

    lines = [
        "# Unified Combined-480 output-recurrence audit", "", f"Verdict: **{verdict}**", "",
        f"Integrity: PASS; {len(MODELS)} models x {len(case_ids)} matched cases = {len(case_rows):,} trajectories.", "",
        "| Model | Final joint | Basin entrants | Escape hazard | Persistent fixed | Persistent cycle | Persistent recurrence | Instability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            f"| {row['model']} | {100*row['final_joint']:.1f}% | {row['primary_basin_entrants']} | "
            f"{100*row['primary_escape_hazard']:.1f}% | {100*row['persistent_fixed_output']:.1f}% | "
            f"{100*row['persistent_output_cycle']:.1f}% | {100*row['persistent_output_recurrence']:.1f}% | "
            f"{100*row['instability_burden']:.1f}% |"
        )
    lines += [
        "", "## Aggregate", "",
        f"Persistent terminal output recurrence: {n_persistent}/{len(failed_rows)} failed model-cases ({100*n_persistent/len(failed_rows):.2f}%).",
        f"Independent lineages with persistent output recurrence: {', '.join(lineages_with_persistent_recurrence)}.",
        f"Revision-4 landmark failure-risk difference: {100*landmark_aggregate['estimate']:+.1f} pp "
        f"(95% CI {100*landmark_aggregate['ci95'][0]:+.1f}, {100*landmark_aggregate['ci95'][1]:+.1f}); "
        f"positive in {landmark_aggregate['positive_checkpoint_estimates']}/{len(landmark_estimable_models)} estimable checkpoints.",
        "", "## Endpoint association", "",
    ]
    for metric, values in association.items():
        lines.append(f"- {metric} versus final joint: rho={values['rho']:+.3f}, permutation p={values['permutation_p_two_sided']:.5f}.")
    lines += [
        "", "## Local-policy calibration bridge", "",
        f"Instability-burden slope: {bridge['full_panel_slope']:+.3f}.",
        f"LOCO MAE: intercept {bridge['loco_baseline_mae']:.3f}, burden {bridge['loco_burden_mae']:.3f}, "
        f"relative reduction {100*bridge['relative_loco_mae_reduction']:+.1f}%.",
        "", "## Decision checks", "",
    ]
    lines.extend(f"- {key}: **{value}**" for key, value in checks.items())
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
