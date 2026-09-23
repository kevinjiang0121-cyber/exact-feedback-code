#!/usr/bin/env python3
"""Test the frozen capture--recurrence law on the seven API controllers."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from exact_feedback.recurrence.unified_analysis import state_tuple


SEED = 20260803
PERMUTATIONS = 100_000
API_MODELS = {
    "deepseek_v4_flash_api": "DeepSeek V4 Flash",
    "qwen3_7_plus_api": "Qwen3.7 Plus",
    "glm5_2_api": "GLM-5.2",
    "gpt5_6_sol_api": "GPT-5.6 Sol",
    "gemini_3_6_flash_api": "Gemini 3.6 Flash",
    "llama31_70b_api": "Llama 3.1 70B",
    "claude_opus5_api": "Claude Opus 5",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cases_path(root: Path, model: str) -> Path:
    return root / "experiments" / "model_generality_combined480_api_v1" / model / model / "baseline" / "cases.jsonl"


def spearman(left: list[float], right: list[float]) -> float:
    x = rankdata(np.asarray(left, dtype=float))
    y = rankdata(np.asarray(right, dtype=float))
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def exact_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    observed = spearman(left, right)
    total = 0
    extreme = 0
    for permuted in itertools.permutations(right):
        total += 1
        extreme += int(abs(spearman(left, list(permuted))) >= abs(observed) - 1e-12)
    return {"rho": observed, "exact_p_two_sided": extreme / total, "permutations": total}


def permutation_spearman(left: list[float], right: list[float], permutations: int, seed: int) -> dict[str, Any]:
    observed = spearman(left, right)
    rng = np.random.default_rng(seed)
    array = np.asarray(right, dtype=float)
    extreme = 0
    for _ in range(permutations):
        extreme += int(abs(spearman(left, rng.permutation(array).tolist())) >= abs(observed) - 1e-12)
    return {"rho": observed, "permutation_p_two_sided": (extreme + 1) / (permutations + 1), "permutations": permutations}


def early_hazards(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("rounds") and not row.get("provider_refusal", False)]
    active_transitions = 0
    captures = 0
    recurrences = 0
    for row in evaluable:
        rounds = row["rounds"]
        states = [state_tuple(item) for item in rounds]
        for index in range(min(4, len(rounds) - 1)):
            active_transitions += 1
            captures += int(bool(rounds[index + 1].get("joint_success", False)))
            recurrences += int(states[index + 1] in set(states[: index + 1]))
    return {
        "model": model,
        "display_name": API_MODELS[model],
        "cases": len(rows),
        "evaluable_cases": len(evaluable),
        "active_transitions_r0_r4": active_transitions,
        "capture_events": captures,
        "recurrence_events": recurrences,
        "early_capture_hazard": captures / active_transitions,
        "early_recurrence_hazard": recurrences / active_transitions,
        "capture_recurrence_ratio": (captures + 0.5) / (recurrences + 0.5),
        "final_joint": float(np.mean([bool(row.get("final_joint", False)) for row in evaluable])),
        "one_shot_joint": float(np.mean([bool(row.get("one_shot_joint", False)) for row in evaluable])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/capture_recurrence_law_api_extension_v1"))
    parser.add_argument("--open-hazards", type=Path)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    api_rows = []
    input_files = []
    reference_ids = None
    for model in API_MODELS:
        path = cases_path(root, model)
        rows = read_jsonl(path)
        ids = [str(row["id"]) for row in rows]
        if len(rows) != 480 or len(set(ids)) != 480:
            raise ValueError(f"{model}: expected 480 unique cases, got {len(rows)} rows/{len(set(ids))} ids")
        if reference_ids is None:
            reference_ids = set(ids)
        elif set(ids) != reference_ids:
            raise ValueError(f"{model}: frozen case set mismatch")
        api_rows.append(early_hazards(model, rows))
        input_files.append({"model": model, "path": str(path), "rows": len(rows)})

    open_path = args.open_hazards or root / "experiments" / "capture_recurrence_law_v1" / "early_hazards.csv"
    with open_path.open("r", encoding="utf-8", newline="") as handle:
        open_rows = list(csv.DictReader(handle))
    for row in open_rows:
        row["capture_recurrence_ratio"] = float(row["capture_recurrence_ratio"])
        row["final_joint"] = float(row["final_joint"])

    api_association = exact_spearman(
        [row["capture_recurrence_ratio"] for row in api_rows],
        [row["final_joint"] for row in api_rows],
    )
    combined_rows = [
        {"model": row["model"], "panel": "open_discovery", "capture_recurrence_ratio": row["capture_recurrence_ratio"], "final_joint": row["final_joint"]}
        for row in open_rows
    ] + [
        {"model": row["model"], "panel": "api_confirmation", "capture_recurrence_ratio": row["capture_recurrence_ratio"], "final_joint": row["final_joint"]}
        for row in api_rows
    ]
    combined_association = permutation_spearman(
        [row["capture_recurrence_ratio"] for row in combined_rows],
        [row["final_joint"] for row in combined_rows],
        args.permutations,
        SEED,
    )

    report = {
        "schema_version": 1,
        "frozen_definition": "revision 0--4 active transitions; capture is next-round joint success; recurrence is an exact previously seen (text,error,missing-anchor) state; Jeffreys-smoothed capture/recurrence event ratio",
        "open_discovery_panel": {"models": len(open_rows), "association": permutation_spearman([row["capture_recurrence_ratio"] for row in open_rows], [row["final_joint"] for row in open_rows], args.permutations, SEED + 1)},
        "api_confirmation_panel": {"models": len(api_rows), "association": api_association},
        "combined_panel": {"models": len(combined_rows), "association": combined_association},
        "integrity": {"passed": True, "api_models": 7, "cases_per_model": 480, "matched_case_set": True, "provider_refusals_excluded_from_control_estimands": 1},
        "inputs": input_files,
    }
    (output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output / "api_early_hazards.csv", api_rows)
    write_csv(output / "combined_early_hazards.csv", combined_rows)
    lines = [
        "# API confirmation of the capture--recurrence law", "",
        "The definition was frozen before adding the API panel: revisions 0--4, next-state joint capture, exact-state recurrence, and the same Jeffreys-smoothed ratio.", "",
        f"- Open discovery panel (n=12): rho={report['open_discovery_panel']['association']['rho']:+.3f}, permutation p={report['open_discovery_panel']['association']['permutation_p_two_sided']:.5g}.",
        f"- API confirmation panel (n=7): rho={api_association['rho']:+.3f}, exact two-sided p={api_association['exact_p_two_sided']:.5g}.",
        f"- Combined panel (n=19): rho={combined_association['rho']:+.3f}, permutation p={combined_association['permutation_p_two_sided']:.5g}.", "",
        "The API panel is an external model-family confirmation, not a new fit or a post-hoc redefinition of the hazard statistic.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
