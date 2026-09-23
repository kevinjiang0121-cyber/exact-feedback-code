#!/usr/bin/env python3
"""Seven-model paired bootstrap and model-by-task interaction analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


MODELS = [
    "gpt5_6_sol_api",
    "claude_opus5_api",
    "gemini_3_6_flash_api",
    "glm5_2_api",
    "qwen3_7_plus_api",
    "llama31_70b_api",
    "deepseek_v4_flash_api",
]
DISPLAY = {
    "gpt5_6_sol_api": "GPT-5.6 Sol",
    "claude_opus5_api": "Claude Opus 5",
    "gemini_3_6_flash_api": "Gemini 3.6 Flash",
    "glm5_2_api": "GLM-5.2",
    "qwen3_7_plus_api": "Qwen3.7 Plus",
    "llama31_70b_api": "Llama 3.1 70B",
    "deepseek_v4_flash_api": "DeepSeek V4 Flash",
}
ASSAYS = ["exact_length", "lexical_constraints", "compositional_constraints"]
STRUCTURES = ["c05", "c07", "c12", "c09", "c10"]
OUTCOMES = ["final_joint", "recurrence", "one_shot_joint", "terminal_recurrence"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_metrics(row: dict[str, Any]) -> dict[str, bool]:
    hashes = [hashlib.sha256(str(item["text"]).encode("utf-8")).digest() for item in row["rounds"]]
    recurrence = len(hashes) != len(set(hashes))
    terminal = len(hashes) > 1 and hashes[-1] in hashes[:-1]
    final_key = "final_joint_success" if "final_joint_success" in row else "final_joint"
    return {
        "final_joint": bool(row[final_key]),
        "one_shot_joint": bool(row["one_shot_joint"]),
        "recurrence": recurrence,
        "terminal_recurrence": terminal,
    }


def exact_path(root: Path, model: str) -> Path:
    return root / "experiments/model_generality_combined480_api_v1" / model / model / "baseline/cases.jsonl"


def new_path(root: Path, assay: str, model: str) -> Path:
    return root / "experiments/model_generality_multidomain_api_v1" / assay / model / "baseline/cases.jsonl"


def load_panel(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    panel: dict[str, dict[str, dict[str, Any]]] = {assay: {} for assay in ASSAYS}
    manifests = []
    for assay in ASSAYS:
        for model in MODELS:
            path = exact_path(root, model) if assay == "exact_length" else new_path(root, assay, model)
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            by_id = {}
            rejected = 0
            for row in rows:
                metrics = trajectory_metrics(row) if row.get("rounds") else None
                protocol_flag = row.get("protocol_complete")
                evaluable = (
                    protocol_flag is not False
                    and row.get("api_error") is None
                    and metrics is not None
                    and ("final_joint" in row or "final_joint_success" in row)
                )
                if not evaluable:
                    rejected += 1
                    continue
                case_id = str(row["id"])
                if case_id in by_id:
                    raise ValueError(f"duplicate {assay}/{model}/{case_id}")
                by_id[case_id] = {
                    "id": case_id,
                    "source": str(row["source"]),
                    "structure": "exact" if assay == "exact_length" else str(row["structure"]),
                    **metrics,
                }
            panel[assay][model] = by_id
            manifests.append({
                "assay": assay,
                "model": model,
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(rows),
                "evaluable": len(by_id),
                "excluded": rejected,
            })

    # Verify metadata equality on every shared case.
    for assay in ASSAYS:
        union = set().union(*(set(panel[assay][model]) for model in MODELS))
        for case_id in union:
            metadata = {
                (panel[assay][model][case_id]["source"], panel[assay][model][case_id]["structure"])
                for model in MODELS if case_id in panel[assay][model]
            }
            if len(metadata) != 1:
                raise ValueError(f"metadata mismatch {assay}/{case_id}: {metadata}")
    return panel, manifests


def common_ids(panel: dict[str, dict[str, dict[str, Any]]], assay: str, models: list[str]) -> list[str]:
    ids = set(panel[assay][models[0]])
    for model in models[1:]:
        ids &= set(panel[assay][model])
    return sorted(ids)


def stratified_draws(
    matrix: np.ndarray,
    sources: list[str],
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.zeros((draws, matrix.shape[1]), dtype=float)
    n_total = len(sources)
    for source in sorted(set(sources)):
        index = np.array([i for i, value in enumerate(sources) if value == source], dtype=int)
        n = len(index)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=draws)
        output += (counts @ matrix[index]) / n_total
    return output


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def holm(rows: list[dict[str, Any]], p_key: str = "p_raw") -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][p_key])
    running = 0.0
    m = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * rows[index][p_key])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def paired_bootstrap(
    panel: dict[str, dict[str, dict[str, Any]]],
    draws: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    draw_cache: dict[str, dict[str, np.ndarray]] = {}
    for assay_index, assay in enumerate(ASSAYS):
        ids = common_ids(panel, assay, MODELS)
        sources = [panel[assay][MODELS[0]][case_id]["source"] for case_id in ids]
        draw_cache[assay] = {}
        for outcome_index, outcome in enumerate(OUTCOMES):
            matrix = np.array(
                [[panel[assay][model][case_id][outcome] for model in MODELS] for case_id in ids],
                dtype=float,
            )
            rng = np.random.default_rng(seed + assay_index * 100 + outcome_index)
            estimates = stratified_draws(matrix, sources, draws, rng)
            draw_cache[assay][outcome] = estimates
            for model_index, model in enumerate(MODELS):
                low, high = quantile_interval(estimates[:, model_index])
                cell_rows.append({
                    "assay": assay,
                    "model": model,
                    "outcome": outcome,
                    "n_common": len(ids),
                    "estimate": float(matrix[:, model_index].mean()),
                    "ci_low": low,
                    "ci_high": high,
                })
            if outcome not in {"final_joint", "recurrence"}:
                continue
            family = []
            for a, b in itertools.combinations(range(len(MODELS)), 2):
                pair_ids = common_ids(panel, assay, [MODELS[a], MODELS[b]])
                if len(pair_ids) == len(ids):
                    pair_matrix = matrix[:, [a, b]]
                    delta_draws = estimates[:, a] - estimates[:, b]
                else:
                    pair_sources = [panel[assay][MODELS[a]][case_id]["source"] for case_id in pair_ids]
                    pair_matrix = np.array(
                        [[panel[assay][model][case_id][outcome] for model in [MODELS[a], MODELS[b]]] for case_id in pair_ids],
                        dtype=float,
                    )
                    pair_rng = np.random.default_rng(seed + assay_index * 10_000 + outcome_index * 100 + a * 10 + b)
                    pair_estimates = stratified_draws(pair_matrix, pair_sources, draws, pair_rng)
                    delta_draws = pair_estimates[:, 0] - pair_estimates[:, 1]
                low, high = quantile_interval(delta_draws)
                ya, yb = pair_matrix[:, 0].astype(bool), pair_matrix[:, 1].astype(bool)
                a_only = int(np.sum(ya & ~yb))
                b_only = int(np.sum(~ya & yb))
                discordant = a_only + b_only
                p_raw = stats.binomtest(a_only, discordant, 0.5).pvalue if discordant else 1.0
                family.append({
                    "assay": assay,
                    "outcome": outcome,
                    "model_a": MODELS[a],
                    "model_b": MODELS[b],
                    "n_common": len(pair_ids),
                    "rate_a": float(pair_matrix[:, 0].mean()),
                    "rate_b": float(pair_matrix[:, 1].mean()),
                    "delta_a_minus_b": float(pair_matrix[:, 0].mean() - pair_matrix[:, 1].mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "a_only": a_only,
                    "b_only": b_only,
                    "p_raw": float(p_raw),
                })
            holm(family)
            pair_rows.extend(family)
    cache_summary = {assay: {outcome: list(value.shape) for outcome, value in outcomes.items()} for assay, outcomes in draw_cache.items()}
    return cell_rows, pair_rows, cache_summary


def design_row(level: str, model: str, task: str, models: list[str], tasks: list[str]) -> np.ndarray:
    values = [1.0]
    values.extend(float(model == item) for item in models[1:])
    values.extend(float(task == item) for item in tasks[1:])
    values.extend(float(model == m and task == t) for m in models[1:] for t in tasks[1:])
    return np.asarray(values, dtype=float)


def clustered_lpm(
    records: list[dict[str, Any]],
    models: list[str],
    tasks: list[str],
    task_key: str,
    outcome: str,
) -> dict[str, Any]:
    x = np.vstack([design_row(task_key, row["model"], row[task_key], models, tasks) for row in records])
    y = np.asarray([row[outcome] for row in records], dtype=float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["cluster"]].append(index)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for index in groups.values():
        ix = np.asarray(index, dtype=int)
        score = x[ix].T @ residual[ix]
        meat += np.outer(score, score)
    n, k, g = len(y), x.shape[1], len(groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    interaction_start = 1 + (len(models) - 1) + (len(tasks) - 1)
    indices = np.arange(interaction_start, k)
    b = beta[indices]
    v = covariance[np.ix_(indices, indices)]
    rank = int(np.linalg.matrix_rank(v))
    statistic = float(b.T @ np.linalg.pinv(v) @ b)
    p_value = float(stats.chi2.sf(statistic, rank))
    return {
        "n_rows": n,
        "clusters": g,
        "parameters": k,
        "interaction_df": rank,
        "wald_chi2": statistic,
        "p_value": p_value,
        "beta": beta,
        "covariance": covariance,
    }


def records_for_assays(panel: dict[str, dict[str, dict[str, Any]]], models: list[str]) -> list[dict[str, Any]]:
    records = []
    for assay in ASSAYS:
        for case_id in common_ids(panel, assay, models):
            for model in models:
                row = panel[assay][model][case_id]
                records.append({
                    "model": model,
                    "assay": assay,
                    "cluster": f"{assay}::{case_id}",
                    **{outcome: row[outcome] for outcome in OUTCOMES},
                })
    return records


def records_for_structures(panel: dict[str, dict[str, dict[str, Any]]], models: list[str], structures: list[str]) -> list[dict[str, Any]]:
    records = []
    for assay in ["lexical_constraints", "compositional_constraints"]:
        ids = common_ids(panel, assay, models)
        for case_id in ids:
            structure = panel[assay][models[0]][case_id]["structure"]
            if structure not in structures:
                continue
            for model in models:
                row = panel[assay][model][case_id]
                records.append({
                    "model": model,
                    "structure": structure,
                    "cluster": f"{structure}::{case_id}",
                    **{outcome: row[outcome] for outcome in OUTCOMES},
                })
    return records


def contrast_vector(model_a: str, model_b: str, task_a: str, task_b: str, models: list[str], tasks: list[str]) -> np.ndarray:
    return (
        design_row("task", model_a, task_a, models, tasks)
        - design_row("task", model_b, task_a, models, tasks)
        - design_row("task", model_a, task_b, models, tasks)
        + design_row("task", model_b, task_b, models, tasks)
    )


def interaction_contrasts(fit: dict[str, Any], models: list[str], tasks: list[str], outcome: str) -> list[dict[str, Any]]:
    rows = []
    beta, covariance = fit["beta"], fit["covariance"]
    for model_a, model_b in itertools.combinations(models, 2):
        for task_a, task_b in itertools.combinations(tasks, 2):
            c = contrast_vector(model_a, model_b, task_a, task_b, models, tasks)
            estimate = float(c @ beta)
            variance = max(0.0, float(c @ covariance @ c))
            se = math.sqrt(variance)
            z = estimate / se if se else (math.inf if estimate else 0.0)
            p_raw = float(2 * stats.norm.sf(abs(z))) if se else (0.0 if estimate else 1.0)
            rows.append({
                "outcome": outcome,
                "model_a": model_a,
                "model_b": model_b,
                "task_a": task_a,
                "task_b": task_b,
                "difference_in_differences": estimate,
                "se_cluster": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "p_raw": p_raw,
            })
    holm(rows)
    return rows


def interaction_suite(panel: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload: dict[str, Any] = {"assay": {}, "structure": {}, "robustness": {}}
    assay_contrasts: list[dict[str, Any]] = []
    structure_contrasts: list[dict[str, Any]] = []
    assay_records = records_for_assays(panel, MODELS)
    structure_records = records_for_structures(panel, MODELS, STRUCTURES)
    for outcome in ["final_joint", "recurrence", "one_shot_joint"]:
        assay_fit = clustered_lpm(assay_records, MODELS, ASSAYS, "assay", outcome)
        structure_fit = clustered_lpm(structure_records, MODELS, STRUCTURES, "structure", outcome)
        payload["assay"][outcome] = {key: value for key, value in assay_fit.items() if key not in {"beta", "covariance"}}
        payload["structure"][outcome] = {key: value for key, value in structure_fit.items() if key not in {"beta", "covariance"}}
        assay_contrasts.extend(interaction_contrasts(assay_fit, MODELS, ASSAYS, outcome))
        structure_contrasts.extend(interaction_contrasts(structure_fit, MODELS, STRUCTURES, outcome))

    payload["robustness"]["leave_one_model_out"] = []
    for omitted in MODELS:
        models = [model for model in MODELS if model != omitted]
        records = records_for_assays(panel, models)
        fit = clustered_lpm(records, models, ASSAYS, "assay", "final_joint")
        payload["robustness"]["leave_one_model_out"].append({
            "omitted": omitted,
            **{key: value for key, value in fit.items() if key not in {"beta", "covariance"}},
        })
    payload["robustness"]["leave_one_structure_out"] = []
    for omitted in STRUCTURES:
        structures = [item for item in STRUCTURES if item != omitted]
        records = records_for_structures(panel, MODELS, structures)
        fit = clustered_lpm(records, MODELS, structures, "structure", "final_joint")
        payload["robustness"]["leave_one_structure_out"].append({
            "omitted": omitted,
            **{key: value for key, value in fit.items() if key not in {"beta", "covariance"}},
        })
    return payload, assay_contrasts, structure_contrasts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_p(value: float) -> str:
    return "<1e-300" if value < 1e-300 else f"{value:.3g}"


def render_report(
    cell_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    interactions: dict[str, Any],
    assay_contrasts: list[dict[str, Any]],
    structure_contrasts: list[dict[str, Any]],
    draws: int,
) -> str:
    lookup = {(row["assay"], row["model"], row["outcome"]): row for row in cell_rows}
    lines = [
        "# Seven-model paired bootstrap and interaction analysis",
        "",
        f"Input and pairing audit: **PASS**. Intervals use {draws:,} source-stratified paired-case bootstrap draws.",
        "Paired analyses use common evaluable cases; the per-comparison denominators are recorded in the CSV outputs.",
        "Wald degrees of freedom use the estimable covariance rank; ceiling cells reduce the nominal structure-interaction rank.",
        "",
        "## Final joint success on common paired cases",
        "",
        "| Model | Exact length | Lexical | Compositional |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        cells = [lookup[(assay, model, "final_joint")] for assay in ASSAYS]
        values = [f"{fmt_pct(row['estimate'])} [{fmt_pct(row['ci_low'])}, {fmt_pct(row['ci_high'])}]" for row in cells]
        lines.append(f"| {DISPLAY[model]} | " + " | ".join(values) + " |")

    lines += ["", "## Omnibus model-by-task interaction", "", "| Level | Outcome | Wald chi2(df) | p |", "|---|---|---:|---:|"]
    for level in ["assay", "structure"]:
        for outcome in ["final_joint", "recurrence", "one_shot_joint"]:
            row = interactions[level][outcome]
            lines.append(f"| {level} | {outcome} | {row['wald_chi2']:.2f} ({row['interaction_df']}) | {fmt_p(row['p_value'])} |")

    lines += ["", "## Paired-comparison closure", ""]
    for assay in ASSAYS:
        for outcome in ["final_joint", "recurrence"]:
            family = [row for row in pair_rows if row["assay"] == assay and row["outcome"] == outcome]
            significant = sum(row["p_holm"] < 0.05 for row in family)
            lines.append(f"- {assay}, {outcome}: {significant}/21 model pairs differ after Holm correction.")

    lines += ["", "## Largest adjusted assay interactions in final success", "", "| Models | Assays | Difference-in-differences | 95% CI | Holm p |", "|---|---|---:|---:|---:|"]
    selected = sorted(
        [row for row in assay_contrasts if row["outcome"] == "final_joint" and row["p_holm"] < 0.05],
        key=lambda row: abs(row["difference_in_differences"]), reverse=True,
    )[:10]
    for row in selected:
        lines.append(
            f"| {DISPLAY[row['model_a']]} vs {DISPLAY[row['model_b']]} | {row['task_a']} vs {row['task_b']} | "
            f"{100 * row['difference_in_differences']:+.1f} pp | "
            f"[{100 * row['ci_low']:+.1f}, {100 * row['ci_high']:+.1f}] | {fmt_p(row['p_holm'])} |"
        )

    lines += ["", "## Largest adjusted structure interactions in final success", "", "| Models | Structures | Difference-in-differences | 95% CI | Holm p |", "|---|---|---:|---:|---:|"]
    selected = sorted(
        [row for row in structure_contrasts if row["outcome"] == "final_joint" and row["p_holm"] < 0.05],
        key=lambda row: abs(row["difference_in_differences"]), reverse=True,
    )[:10]
    for row in selected:
        lines.append(
            f"| {DISPLAY[row['model_a']]} vs {DISPLAY[row['model_b']]} | {row['task_a']} vs {row['task_b']} | "
            f"{100 * row['difference_in_differences']:+.1f} pp | "
            f"[{100 * row['ci_low']:+.1f}, {100 * row['ci_high']:+.1f}] | {fmt_p(row['p_holm'])} |"
        )

    loo_model = interactions["robustness"]["leave_one_model_out"]
    loo_structure = interactions["robustness"]["leave_one_structure_out"]
    lines += [
        "",
        "## Robustness",
        "",
        f"- Final-success assay interaction has p<.05 in {sum(row['p_value'] < .05 for row in loo_model)}/{len(loo_model)} leave-one-model-out fits; maximum p={fmt_p(max(row['p_value'] for row in loo_model))}.",
        f"- Final-success structure interaction has p<.05 in {sum(row['p_value'] < .05 for row in loo_structure)}/{len(loo_structure)} leave-one-structure-out fits; maximum p={fmt_p(max(row['p_value'] for row in loo_structure))}.",
        "",
        "## Interpretation",
        "",
        "These tests measure behavioral interaction under matched verification. Assess evidence for task-dependent ordering from the estimates and uncertainty above; this comparison does not isolate architecture or training as the cause.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_808)
    parser.add_argument("--fresh", action="store_true", help="Allow observed refusal patterns instead of requiring the historical frozen pattern")
    args = parser.parse_args()

    panel, manifests = load_panel(args.project_root)
    expected_evaluable = {
        assay: {
            model: (479 if assay == "exact_length" and model == "gemini_3_6_flash_api" else 480)
            for model in MODELS
        }
        for assay in ASSAYS
    }
    for item in manifests:
        if item["rows"] != 480 or (not args.fresh and item["evaluable"] != expected_evaluable[item["assay"]][item["model"]]):
            raise ValueError(f"unexpected panel integrity: {item}")
    expected_common = {"exact_length": 479, "lexical_constraints": 480, "compositional_constraints": 480}
    observed_common = {assay: len(common_ids(panel, assay, MODELS)) for assay in ASSAYS}
    if (not args.fresh and observed_common != expected_common) or not all(observed_common.values()):
        raise ValueError(f"common-case mismatch: {observed_common}")
    cell_rows, pair_rows, draw_shapes = paired_bootstrap(panel, args.draws, args.seed)
    interactions, assay_contrasts, structure_contrasts = interaction_suite(panel)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_cell_estimates.csv", cell_rows)
    write_csv(args.output_dir / "paired_model_contrasts.csv", pair_rows)
    write_csv(args.output_dir / "assay_interaction_contrasts.csv", assay_contrasts)
    write_csv(args.output_dir / "structure_interaction_contrasts.csv", structure_contrasts)
    (args.output_dir / "input_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    serializable = {
        "protocol": "docs/API_SEVEN_MODEL_PAIRED_INTERACTION_PROTOCOL_20260808.md",
        "draws": args.draws,
        "seed": args.seed,
        "common_cases": observed_common,
        "draw_shapes": draw_shapes,
        "interactions": interactions,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    report = render_report(cell_rows, pair_rows, interactions, assay_contrasts, structure_contrasts, args.draws)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "common_cases": serializable["common_cases"],
        "assay_final_interaction": interactions["assay"]["final_joint"],
        "structure_final_interaction": interactions["structure"]["final_joint"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
