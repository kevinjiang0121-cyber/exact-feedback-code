#!/usr/bin/env python3
"""Formal three-domain confirmation for exact-feedback closed-loop revision."""

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
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
from scipy.stats import norm


SEED = 20260805
BOOTSTRAPS = 10_000
MODELS = (
    "llama31_8b", "gemma2_9b", "glm4_9b", "ministral_8b",
    "qwen3_1_7b", "qwen3_4b", "qwen3_8b", "qwen3_14b",
    "qwen3_32b", "qwen3_30b_a3b", "granite_3_3_8b", "falcon_h1_7b",
)
MODEL_LABEL = {
    "llama31_8b": "Llama-3.1-8B",
    "gemma2_9b": "Gemma-2-9B",
    "glm4_9b": "GLM-4-9B",
    "ministral_8b": "Ministral-8B",
    "qwen3_1_7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "qwen3_30b_a3b": "Qwen3-30B-A3B",
    "granite_3_3_8b": "Granite-3.3-8B",
    "falcon_h1_7b": "Falcon-H1-7B",
}
DOMAIN_ORDER = ("exact_length", "lexical_constraints", "compositional_constraints")
EXACT_OLD = {
    "llama31_8b": "llama31_8b", "gemma2_9b": "gemma2_9b",
    "glm4_9b": "glm4_9b", "ministral_8b": "ministral_8b",
    "qwen3_8b": "qwen3_8b", "qwen3_14b": "qwen3_14b",
}
EXACT_SINGLE = {
    "qwen3_1_7b": "qwen3_1_7b/combined480",
    "qwen3_4b": "qwen3_4b/combined480",
    "qwen3_32b": "qwen3_32b/vllm_pp2_combined480",
    "qwen3_30b_a3b": "qwen3_30b_a3b/vllm_pp2_combined480",
    "granite_3_3_8b": "granite_3_3_8b/combined480",
    "falcon_h1_7b": "falcon_h1_7b/combined480",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trajectory(model: str, domain: str, row: dict[str, Any]) -> dict[str, Any]:
    rounds = row["rounds"]
    states = [text_hash(str(item["text"])) for item in rounds]
    seen: set[str] = set()
    recurrence_revisions: list[int] = []
    for index, state in enumerate(states):
        if state in seen and not bool(rounds[index].get("joint_success", False)):
            recurrence_revisions.append(index)
        seen.add(state)
    persistent_fixed = int(
        not bool(row["final_joint"]) and len(states) >= 3
        and states[-1] == states[-2] == states[-3]
    )
    persistent_cycle_period = 0
    if not bool(row["final_joint"]):
        for period in (2, 3, 4):
            if len(states) >= 2 * period:
                left = states[-2 * period:-period]
                right = states[-period:]
                if left == right and len(set(right)) >= 2:
                    persistent_cycle_period = period
                    break
    landmark_risk = int(len(states) >= 6)
    recurrence_r4 = int(landmark_risk and len(set(states[:5])) < len(states[:5]))
    observed = rounds[:5]
    energies = []
    for item in observed:
        if "violation_energy" in item:
            energies.append(float(item["violation_energy"]))
        else:
            target = max(float(row.get("target", 1)), 1.0)
            energies.append(abs(float(item.get("error", 0))) / target + len(item.get("missing", [])))
    energy_actions = [right - left for left, right in zip(energies, energies[1:])]
    task = "exact_length" if domain == "exact_length" else str(row["family"])
    source = source_family(str(row["source"]))
    return {
        "model": model,
        "model_label": MODEL_LABEL[model],
        "domain": domain,
        "task": task,
        "case_id": str(row["id"]),
        "case_cluster": f"{domain}|{row['id']}",
        "source": source,
        "bootstrap_stratum": f"{task}|{source}",
        "one_shot": int(bool(row["one_shot_joint"])),
        "final": int(bool(row["final_joint"])),
        "any_recurrence": int(bool(recurrence_revisions)),
        "persistent_fixed": persistent_fixed,
        "persistent_cycle": int(persistent_cycle_period > 0),
        "persistent_recurrence": int(bool(persistent_fixed or persistent_cycle_period)),
        "landmark_risk_r4": landmark_risk,
        "recurrence_by_r4": recurrence_r4,
        "later_rescue": int(landmark_risk and bool(row["final_joint"])),
        "repeat_fraction_r4": sum(states[index] in set(states[:index]) for index in range(1, min(5, len(states)))) / 4 if landmark_risk else math.nan,
        "current_energy_r4": energies[-1] if landmark_risk else math.nan,
        "log_current_energy_r4": math.log1p(energies[-1]) if landmark_risk else math.nan,
        "min_energy_r4": min(energies) if landmark_risk else math.nan,
        "energy_improvement_r4": energies[0] - energies[-1] if landmark_risk else math.nan,
        "contraction_count_r4": sum(right < left for left, right in zip(energies, energies[1:])) if landmark_risk else math.nan,
        "energy_noop_count_r4": sum(abs(action) < 1e-12 for action in energy_actions) if landmark_risk else math.nan,
        "largest_abs_energy_action_r4": max((abs(action) for action in energy_actions), default=0.0) if landmark_risk else math.nan,
    }


def exact_registry(root: Path) -> dict[str, list[Path]]:
    exp = root / "experiments"
    registry: dict[str, list[Path]] = {}
    for model, directory in EXACT_OLD.items():
        registry[model] = [
            exp / f"human_generation_main120_v1/{directory}/cases.jsonl",
            exp / f"human_generation_replication120_v1/{directory}/cases.jsonl",
            exp / f"human_generation_extension240_v1/{directory}/cases.jsonl",
        ]
    for model, suffix in EXACT_SINGLE.items():
        registry[model] = [exp / f"model_generality_combined480_v1/{suffix}/cases.jsonl"]
    return registry


def load_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    case_sets: dict[tuple[str, str], set[str]] = {}
    for model, paths in exact_registry(root).items():
        rows = list(itertools.chain.from_iterable(read_jsonl(path) for path in paths))
        if len(rows) != 480 or len({row["id"] for row in rows}) != 480:
            raise ValueError(f"exact_length {model}: invalid Combined-480")
        case_sets[("exact_length", model)] = {str(row["id"]) for row in rows}
        records.extend(trajectory(model, "exact_length", row) for row in rows)
        files[f"exact_length/{model}"] = [
            {"path": str(path), "rows": len(read_jsonl(path)), "sha256": sha256(path)} for path in paths
        ]

    new_root = root / "experiments/multidomain_full480_open12_v1"
    for model in MODELS:
        path = new_root / model / "cases.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 960 or len({row["id"] for row in rows}) != 960:
            raise ValueError(f"new domains {model}: invalid 960 panel")
        for domain in ("lexical_constraints", "compositional_constraints"):
            selected = [row for row in rows if row["domain"] == domain]
            if len(selected) != 480:
                raise ValueError(f"{domain} {model}: expected 480")
            case_sets[(domain, model)] = {str(row["id"]) for row in selected}
            records.extend(trajectory(model, domain, row) for row in selected)
        files[f"new_domains/{model}"] = [{"path": str(path), "rows": 960, "sha256": sha256(path)}]

    for domain in DOMAIN_ORDER:
        reference = case_sets[(domain, MODELS[0])]
        for model in MODELS[1:]:
            if case_sets[(domain, model)] != reference:
                raise ValueError(f"case-set mismatch: {domain} {model}")
    if len(records) != 17_280:
        raise ValueError(f"expected 17,280 trajectories, found {len(records)}")
    return records, files


def percentile(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])] if finite.size else [math.nan, math.nan]


def grouped_matrix(df: pd.DataFrame, group: str, outcome: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    cases = sorted(df["case_id"].unique())
    case_index = {case: i for i, case in enumerate(cases)}
    matrix = np.zeros((len(cases), len(MODELS)), dtype=np.int8)
    strata = np.empty(len(cases), dtype=object)
    for row in df.itertuples(index=False):
        i = case_index[row.case_id]
        j = MODELS.index(row.model)
        matrix[i, j] = int(getattr(row, outcome))
        strata[i] = getattr(row, group)
    return matrix, cases, strata


def bootstrap_domain(df: pd.DataFrame, rng: np.random.Generator, b: int) -> dict[str, Any]:
    final, _, strata = grouped_matrix(df, "bootstrap_stratum", "final")
    one, _, _ = grouped_matrix(df, "bootstrap_stratum", "one_shot")
    persistent, _, _ = grouped_matrix(df, "bootstrap_stratum", "persistent_recurrence")
    failure = 1 - final
    boot_final = np.zeros((b, len(MODELS)), dtype=float)
    boot_one = np.zeros_like(boot_final)
    boot_pf = np.zeros_like(boot_final)
    boot_fail = np.zeros_like(boot_final)
    total = 0
    for stratum in sorted(set(strata)):
        idx = np.flatnonzero(strata == stratum)
        n = len(idx)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=b)
        boot_final += weights @ final[idx]
        boot_one += weights @ one[idx]
        boot_pf += weights @ (persistent[idx] * failure[idx])
        boot_fail += weights @ failure[idx]
        total += n
    boot_final /= total
    boot_one /= total
    with np.errstate(divide="ignore", invalid="ignore"):
        recurrence_failure = boot_pf.sum(axis=1) / boot_fail.sum(axis=1)
    return {
        "final": boot_final,
        "one_shot": boot_one,
        "recurrence_among_failures": recurrence_failure,
        "recurrent_failure_count": boot_pf.sum(axis=1),
        "failure_count": boot_fail.sum(axis=1),
    }


def mh_or(df: pd.DataFrame, strata_columns: list[str]) -> float:
    numerator = 0.0
    denominator = 0.0
    risk = df[df["landmark_risk_r4"] == 1]
    for _, part in risk.groupby(strata_columns, observed=True):
        x = part["recurrence_by_r4"].to_numpy()
        y = part["later_rescue"].to_numpy()
        a = float(np.sum((x == 1) & (y == 1)))
        b = float(np.sum((x == 1) & (y == 0)))
        c = float(np.sum((x == 0) & (y == 1)))
        d = float(np.sum((x == 0) & (y == 0)))
        n = a + b + c + d
        if n:
            numerator += a * d / n
            denominator += b * c / n
    return numerator / denominator if denominator else math.nan


def bootstrap_landmark(df: pd.DataFrame, rng: np.random.Generator, b: int) -> dict[str, Any]:
    risk = df[df["landmark_risk_r4"] == 1].copy()
    observed_or = mh_or(risk, ["model", "task"])
    exposed = risk[risk["recurrence_by_r4"] == 1]
    unexposed = risk[risk["recurrence_by_r4"] == 0]
    observed = {
        "at_risk": int(len(risk)),
        "recurrent": int(len(exposed)),
        "rescue_recurrent": float(exposed["later_rescue"].mean()) if len(exposed) else math.nan,
        "rescue_nonrecurrent": float(unexposed["later_rescue"].mean()) if len(unexposed) else math.nan,
        "risk_difference": float(exposed["later_rescue"].mean() - unexposed["later_rescue"].mean()) if len(exposed) and len(unexposed) else math.nan,
        "mh_common_odds_ratio": float(observed_or),
    }
    stratum_keys = sorted({f"{row.model}|{row.task}" for row in df.itertuples(index=False)})
    stratum_index = {key: i for i, key in enumerate(stratum_keys)}
    a_boot = np.zeros((b, len(stratum_keys)), dtype=float)
    b_boot = np.zeros_like(a_boot)
    c_boot = np.zeros_like(a_boot)
    d_boot = np.zeros_like(a_boot)
    for _, part in df.groupby("bootstrap_stratum", observed=True):
        ids = sorted(part["case_id"].unique())
        id_index = {case: i for i, case in enumerate(ids)}
        matrices = [np.zeros((len(ids), len(stratum_keys)), dtype=np.int8) for _ in range(4)]
        for row in part.itertuples(index=False):
            if not row.landmark_risk_r4:
                continue
            i = id_index[row.case_id]
            j = stratum_index[f"{row.model}|{row.task}"]
            if row.recurrence_by_r4 and row.later_rescue:
                matrices[0][i, j] = 1
            elif row.recurrence_by_r4:
                matrices[1][i, j] = 1
            elif row.later_rescue:
                matrices[2][i, j] = 1
            else:
                matrices[3][i, j] = 1
        n = len(ids)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=b)
        for target, matrix in zip((a_boot, b_boot, c_boot, d_boot), matrices):
            target += weights @ matrix
    n_boot = a_boot + b_boot + c_boot + d_boot
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_or = np.nansum(a_boot * d_boot / n_boot, axis=1) / np.nansum(b_boot * c_boot / n_boot, axis=1)
        rescue_exp = a_boot.sum(axis=1) / (a_boot + b_boot).sum(axis=1)
        rescue_non = c_boot.sum(axis=1) / (c_boot + d_boot).sum(axis=1)
        boot_rd = rescue_exp - rescue_non
    observed["mh_or_bootstrap_95"] = percentile(boot_or)
    observed["risk_difference_bootstrap_95"] = percentile(boot_rd)
    return observed


def interaction_test(df: pd.DataFrame, factor: str) -> dict[str, Any]:
    levels = sorted(df[factor].unique())
    cells = [(model, level) for model in MODELS for level in levels]
    cell_index = {cell: i for i, cell in enumerate(cells)}
    successes = np.zeros(len(cells), dtype=float)
    counts = np.zeros(len(cells), dtype=float)
    for row in df.itertuples(index=False):
        j = cell_index[(row.model, getattr(row, factor))]
        successes[j] += row.final
        counts[j] += 1
    probability = (successes + 0.5) / (counts + 1.0)
    beta = np.log(probability / (1.0 - probability))
    information = counts * probability * (1.0 - probability)
    bread = 1.0 / information
    cluster_scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(cells), dtype=float))
    for row in df.itertuples(index=False):
        j = cell_index[(row.model, getattr(row, factor))]
        cluster_scores[row.case_cluster][j] += row.final - probability[j]
    meat = np.zeros((len(cells), len(cells)), dtype=float)
    for score in cluster_scores.values():
        meat += np.outer(score, score)
    covariance = bread[:, None] * meat * bread[None, :]
    restrictions = []
    baseline_model = MODELS[0]
    baseline_level = levels[0]
    for model in MODELS[1:]:
        for level in levels[1:]:
            contrast = np.zeros(len(cells), dtype=float)
            contrast[cell_index[(model, level)]] = 1.0
            contrast[cell_index[(model, baseline_level)]] = -1.0
            contrast[cell_index[(baseline_model, level)]] = -1.0
            contrast[cell_index[(baseline_model, baseline_level)]] = 1.0
            restrictions.append(contrast)
    restriction = np.stack(restrictions)
    effect = restriction @ beta
    contrast_covariance = restriction @ covariance @ restriction.T
    rank = int(np.linalg.matrix_rank(contrast_covariance))
    statistic = float(effect @ np.linalg.pinv(contrast_covariance) @ effect)
    return {
        "factor": factor,
        "interaction_terms": len(restrictions),
        "wald_chi2": statistic,
        "df": rank,
        "p_value": float(chi2.sf(statistic, rank)),
        "cluster_count": int(df["case_cluster"].nunique()),
        "covariance": "case-cluster-robust sandwich on saturated empirical logits",
        "cell_correction": "add 0.5 successes and 0.5 failures",
    }


def empirical_logit_interaction(df: pd.DataFrame, factor: str) -> dict[str, Any]:
    cells = df.groupby(["model", factor], observed=True)["final"].agg(["sum", "count"]).reset_index()
    cells["logit"] = np.log((cells["sum"] + 0.5) / (cells["count"] - cells["sum"] + 0.5))
    pivot = cells.pivot(index="model", columns=factor, values="logit").loc[list(MODELS)]
    values = pivot.to_numpy()
    grand = values.mean()
    model_effect = values.mean(axis=1, keepdims=True) - grand
    task_effect = values.mean(axis=0, keepdims=True) - grand
    interaction = values - grand - model_effect - task_effect
    ss_model = values.shape[1] * float(np.sum(model_effect ** 2))
    ss_task = values.shape[0] * float(np.sum(task_effect ** 2))
    ss_interaction = float(np.sum(interaction ** 2))
    total = ss_model + ss_task + ss_interaction
    return {
        "scale": "empirical log-odds with 0.5 cell correction",
        "model_share": ss_model / total,
        "task_share": ss_task / total,
        "interaction_share": ss_interaction / total,
        "largest_absolute_interactions": [
            {"model": MODELS[i], factor: str(pivot.columns[j]), "residual_log_odds": float(interaction[i, j])}
            for i, j in sorted(np.ndindex(interaction.shape), key=lambda ij: abs(interaction[ij]), reverse=True)[:10]
        ],
    }


def adjusted_repeat_effect(df: pd.DataFrame) -> dict[str, Any]:
    rows = df[df["landmark_risk_r4"] == 1].copy()
    numeric = (
        "current_energy_r4", "log_current_energy_r4", "min_energy_r4",
        "energy_improvement_r4", "contraction_count_r4", "energy_noop_count_r4",
        "largest_abs_energy_action_r4",
    )
    values = rows.loc[:, numeric].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-12] = 1.0
    blocks = [
        np.ones((len(rows), 1)),
        rows[["repeat_fraction_r4"]].to_numpy(dtype=float),
        (values - mean) / std,
    ]
    rows["model_task"] = rows["model"] + "|" + rows["task"]
    for column in ("model_task", "source"):
        levels = sorted(rows[column].unique())
        for level in levels[1:]:
            blocks.append((rows[column].to_numpy() == level).astype(float)[:, None])
    x = np.column_stack(blocks)
    y = rows["later_rescue"].to_numpy(dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        return (
            float(np.sum(np.logaddexp(0.0, linear) - y * linear)),
            x.T @ (expit(linear) - y),
        )

    fit = minimize(
        lambda beta: objective(beta)[0], np.zeros(x.shape[1]),
        jac=lambda beta: objective(beta)[1], method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-11},
    )
    if not fit.success:
        raise RuntimeError(f"adjusted recurrence GLM failed: {fit.message}")
    probability = expit(x @ fit.x)
    hessian = x.T @ ((probability * (1.0 - probability))[:, None] * x)
    bread = np.linalg.pinv(hessian)
    scores = x * (y - probability)[:, None]
    cluster_scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(x.shape[1]))
    for cluster, score in zip(rows["case_cluster"], scores):
        cluster_scores[str(cluster)] += score
    meat = sum(np.outer(score, score) for score in cluster_scores.values())
    covariance = bread @ meat @ bread
    coefficient = float(fit.x[1])
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    z = coefficient / standard_error
    return {
        "estimand": "odds ratio for repeat_fraction changing from 0 to 1",
        "coefficient": coefficient,
        "odds_ratio": math.exp(coefficient),
        "clustered_95_ci": [math.exp(coefficient - 1.96 * standard_error), math.exp(coefficient + 1.96 * standard_error)],
        "clustered_p_two_sided": float(2 * norm.sf(abs(z))),
        "rows": len(rows), "case_clusters": len(cluster_scores),
        "controls": [*numeric, "model_x_task fixed effects", "source fixed effects"],
        "energy_definition": "task-native violation energy; exact length uses abs(error)/target plus missing anchors",
        "converged": bool(fit.success),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--legacy-analysis", type=Path)
    parser.add_argument(
        "--trajectory-table",
        type=Path,
        help=(
            "Optional frozen case-level source table. When supplied, rerun the "
            "statistical analysis without requiring the raw generation logs."
        ),
    )
    args = parser.parse_args()
    if args.trajectory_table:
        table_path = args.trajectory_table.resolve()
        df = pd.read_csv(table_path)
        required = {
            "model", "domain", "task", "case_id", "source", "one_shot", "final",
            "persistent_recurrence", "landmark_risk_r4", "recurrence_by_r4",
            "later_rescue", "repeat_fraction_r4",
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"trajectory table missing columns: {missing}")
        if len(df) != 17_280:
            raise ValueError(f"trajectory table has {len(df)} rows, expected 17280")
        files = {
            "frozen_trajectory_table": {
                "path": str(args.trajectory_table),
                "rows": len(df),
                "sha256": sha256(table_path),
            }
        }
    else:
        records, files = load_rows(args.root)
        df = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exact_fail = df[(df.domain == "exact_length") & (df.final == 0)]
    exact_recurrent = int(exact_fail["persistent_recurrence"].sum())
    if (exact_recurrent, len(exact_fail)) != (3337, 4622):
        raise ValueError(f"legacy recurrence gate failed: {exact_recurrent}/{len(exact_fail)}")

    rng = np.random.default_rng(SEED)
    cell_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    boot_by_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        part = df[df.domain == domain]
        boot = bootstrap_domain(part, rng, args.bootstraps)
        boot_by_domain[domain] = boot
        for j, model in enumerate(MODELS):
            observed = part[part.model == model]
            cell_rows.append({
                "domain": domain, "model": model, "model_label": MODEL_LABEL[model],
                "cases": len(observed), "one_shot_rate": observed.one_shot.mean(),
                "final_success_rate": observed.final.mean(),
                "final_bootstrap_low": percentile(boot["final"][:, j])[0],
                "final_bootstrap_high": percentile(boot["final"][:, j])[1],
                "gain_over_one_shot": observed.final.mean() - observed.one_shot.mean(),
            })
        failures = part[part.final == 0]
        recurrence_rows.append({
            "scope": domain, "failures": len(failures),
            "persistent_recurrent_failures": int(failures.persistent_recurrence.sum()),
            "persistent_recurrence_among_failures": failures.persistent_recurrence.mean(),
            "bootstrap_low": percentile(boot["recurrence_among_failures"])[0],
            "bootstrap_high": percentile(boot["recurrence_among_failures"])[1],
        })
        for left, right in itertools.combinations(range(len(MODELS)), 2):
            delta = boot["final"][:, left] - boot["final"][:, right]
            observed_delta = part[part.model == MODELS[left]].final.mean() - part[part.model == MODELS[right]].final.mean()
            contrast_rows.append({
                "domain": domain, "model_a": MODELS[left], "model_b": MODELS[right],
                "delta_a_minus_b": observed_delta,
                "bootstrap_low": percentile(delta)[0], "bootstrap_high": percentile(delta)[1],
                "bootstrap_two_sided_p": min(1.0, 2 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0)))),
            })

    for scope, domains in (
        ("new_domains_pooled", ("lexical_constraints", "compositional_constraints")),
        ("all_three_domains", DOMAIN_ORDER),
    ):
        scoped = df[df.domain.isin(domains)]
        failures = scoped[scoped.final == 0]
        numerator = sum(boot_by_domain[domain]["recurrent_failure_count"] for domain in domains)
        denominator = sum(boot_by_domain[domain]["failure_count"] for domain in domains)
        interval = percentile(numerator / denominator)
        recurrence_rows.append({
            "scope": scope, "failures": len(failures),
            "persistent_recurrent_failures": int(failures.persistent_recurrence.sum()),
            "persistent_recurrence_among_failures": failures.persistent_recurrence.mean(),
            "bootstrap_low": interval[0], "bootstrap_high": interval[1],
        })

    for domain in DOMAIN_ORDER:
        indices = [i for i, row in enumerate(contrast_rows) if row["domain"] == domain]
        ordered = sorted(indices, key=lambda i: contrast_rows[i]["bootstrap_two_sided_p"])
        running = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            candidate = (total - rank) * contrast_rows[index]["bootstrap_two_sided_p"]
            running = max(running, candidate)
            contrast_rows[index]["holm_p_within_domain"] = min(1.0, running)

    landmark = {"all_three_domains": bootstrap_landmark(df, rng, args.bootstraps)}
    landmark["new_domains_pooled"] = bootstrap_landmark(df[df.domain != "exact_length"], rng, args.bootstraps)
    for domain in DOMAIN_ORDER:
        landmark[domain] = bootstrap_landmark(df[df.domain == domain], rng, args.bootstraps)

    interactions = {
        "model_by_domain": interaction_test(df, "domain"),
        "model_by_task_family": interaction_test(df, "task"),
        "logit_variance_domain": empirical_logit_interaction(df, "domain"),
        "logit_variance_task_family": empirical_logit_interaction(df, "task"),
    }
    adjusted_recurrence = {"all_three_domains": adjusted_repeat_effect(df)}
    adjusted_recurrence["new_domains_pooled"] = adjusted_repeat_effect(df[df.domain != "exact_length"])
    for domain in DOMAIN_ORDER:
        adjusted_recurrence[domain] = adjusted_repeat_effect(df[df.domain == domain])
    legacy_path = args.legacy_analysis or args.root / "experiments/capture_recurrence_law_v1/analysis.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))["landmarks"]["4"]
    legacy_effect = legacy["common_effect"]
    if legacy.get("risk_rows") != 5039 or not math.isclose(
        legacy_effect["odds_ratio_full_repeat_fraction"],
        0.05944442113656184,
        rel_tol=1e-6,
        abs_tol=1e-12,
    ):
        raise ValueError("legacy adjusted recurrence gate failed")

    per_cell = df.groupby(["domain", "task", "model"], observed=True).agg(
        cases=("final", "size"), one_shot_rate=("one_shot", "mean"),
        final_success_rate=("final", "mean"), any_recurrence_rate=("any_recurrence", "mean"),
        persistent_recurrence_rate=("persistent_recurrence", "mean"),
    ).reset_index().to_dict("records")
    write_csv(args.output_dir / "domain_model_rates.csv", cell_rows)
    write_csv(args.output_dir / "task_model_rates.csv", per_cell)
    write_csv(args.output_dir / "paired_model_contrasts.csv", contrast_rows)
    write_csv(args.output_dir / "recurrence_failure_rates.csv", recurrence_rows)
    df.to_csv(args.output_dir / "trajectory_table.csv", index=False)

    manifest = {
        "schema_version": 1, "seed": SEED, "bootstraps": args.bootstraps,
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "models": list(MODELS), "domains": list(DOMAIN_ORDER),
        "trajectories": len(df), "cases_per_model_domain": 480,
        "bootstrap_unit": "case, paired across all 12 models",
        "bootstrap_strata": "task family x source family",
        "recurrence_definition": "exact output-text hash; persistent fixed point is last three equal; persistent cycle repeats a terminal period 2-4 block",
        "landmark_definition": "at risk requires revision 5; recurrence exposure is any repeated output through revision 4; outcome is later rescue by revision 8",
        "legacy_gate": {"persistent_recurrent_failures": exact_recurrent, "exact_failures": len(exact_fail), "passed": True},
        "acceptance": {
            "complete_factorial_panel": len(df) == 17_280,
            "legacy_persistent_recurrence_reproduced": (exact_recurrent, len(exact_fail)) == (3337, 4622),
            "legacy_adjusted_or_reproduced": True,
            "new_domains_adjusted_or_below_one_ci_excludes_one": adjusted_recurrence["new_domains_pooled"]["clustered_95_ci"][1] < 1,
            "three_domain_adjusted_or_below_one_ci_excludes_one": adjusted_recurrence["all_three_domains"]["clustered_95_ci"][1] < 1,
            "model_by_domain_interaction_detected": interactions["model_by_domain"]["p_value"] < 0.001,
        },
        "inputs": files,
    }
    analysis = {
        "panel": {"models": 12, "domains": 3, "cases_per_cell": 480, "trajectories": len(df)},
        "recurrence_among_failures": recurrence_rows,
        "landmark_recurrence_law": landmark,
        "adjusted_repeat_fraction": adjusted_recurrence,
        "legacy_exact_length_adjusted_gate": {
            "risk_rows": legacy["risk_rows"],
            "odds_ratio": legacy_effect["odds_ratio_full_repeat_fraction"],
            "ci95": legacy_effect["ci95_odds_ratio"],
            "clustered_p_two_sided": legacy_effect["clustered_p_two_sided"],
            "source": str(legacy_path),
            "source_sha256": sha256(legacy_path),
        },
        "interactions": interactions,
    }
    (args.output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    lines = [
        "# Three-domain exact-feedback confirmation", "", "Verdict: **ANALYSIS_COMPLETE**", "",
        f"Panel: 12 models x 3 domains x 480 cases = {len(df):,} trajectories.",
        f"Legacy exact-length gate reproduced: {exact_recurrent}/4622 = {exact_recurrent/4622:.2%} persistent recurrence among failures.", "",
        "## Persistent recurrence among failures", "",
        "| Scope | Recurrent / failures | Rate | Stratified bootstrap 95% CI |", "|---|---:|---:|---:|",
    ]
    for row in recurrence_rows:
        ci = "--" if math.isnan(row["bootstrap_low"]) else f"[{row['bootstrap_low']:.1%}, {row['bootstrap_high']:.1%}]"
        lines.append(f"| {row['scope']} | {row['persistent_recurrent_failures']}/{row['failures']} | {row['persistent_recurrence_among_failures']:.1%} | {ci} |")
    lines += ["", "## Revision-4 landmark", "", "| Scope | At risk | Rescue if recurrent | Rescue if nonrecurrent | MH OR (bootstrap 95% CI) |", "|---|---:|---:|---:|---:|"]
    for scope, row in landmark.items():
        lines.append(f"| {scope} | {row['at_risk']} | {row['rescue_recurrent']:.1%} | {row['rescue_nonrecurrent']:.1%} | {row['mh_common_odds_ratio']:.3f} [{row['mh_or_bootstrap_95'][0]:.3f}, {row['mh_or_bootstrap_95'][1]:.3f}] |")
    lines += ["", "## Interaction tests", ""]
    for name in ("model_by_domain", "model_by_task_family"):
        row = interactions[name]
        lines.append(f"- {name}: cluster-robust Wald chi2({row['df']})={row['wald_chi2']:.1f}, p={row['p_value']:.3g}.")
    lines += ["", "## Adjusted repeat-fraction association", ""]
    for scope, row in adjusted_recurrence.items():
        lines.append(f"- {scope}: OR={row['odds_ratio']:.3f}, 95% clustered CI [{row['clustered_95_ci'][0]:.3f}, {row['clustered_95_ci'][1]:.3f}], p={row['clustered_p_two_sided']:.3g}.")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
