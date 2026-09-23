#!/usr/bin/env python3
"""Case-disjoint descriptive linear probes for Phase3A final-prompt states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
OUTCOMES = ("signed_error", "action_magnitude", "target")
FINAL_PROMPT_POSITION = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def outcome(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    if name == "action_magnitude":
        return np.asarray(
            [abs(int(row["signed_error"])) for row in rows],
            dtype=np.float32,
        )
    return np.asarray([row[name] for row in rows], dtype=np.float32)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train.astype(np.float32))
    test = scaler.transform(x_test.astype(np.float32))
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
    model.fit(train, y_train)
    return model.predict(test)


def select_alpha(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, float]]:
    splitter = GroupKFold(n_splits=6)
    scores: dict[str, float] = {}
    scale = max(float(np.var(y)), 1e-8)
    for alpha in ALPHAS:
        losses = []
        for train, test in splitter.split(x, y, groups):
            predicted = fit_predict(
                x[train], y[train], x[test], alpha
            )
            losses.append(float(np.mean((predicted - y[test]) ** 2) / scale))
        scores[str(alpha)] = float(np.mean(losses))
    selected = min(ALPHAS, key=lambda value: scores[str(value)])
    return float(selected), scores


def metrics(
    name: str, truth: np.ndarray, predicted: np.ndarray
) -> dict[str, float | None]:
    if float(np.std(truth)) == 0.0 or float(np.std(predicted)) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(truth, predicted)[0, 1])
    result = {
        "r2": float(r2_score(truth, predicted)),
        "mae": float(mean_absolute_error(truth, predicted)),
        "prediction_correlation": correlation,
    }
    if name == "signed_error":
        result["sign_accuracy"] = float(
            np.mean(np.sign(predicted) == np.sign(truth))
        )
    if name == "action_magnitude":
        result["magnitude_accuracy"] = float(
            np.mean((predicted >= 6) == (truth >= 6))
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    states = read_jsonl(args.states)
    discovery_indices = np.asarray(
        [index for index, row in enumerate(states) if row["split"] == "discovery"]
    )
    confirmation_indices = np.asarray(
        [index for index, row in enumerate(states) if row["split"] == "confirmation"]
    )
    discovery_rows = [states[index] for index in discovery_indices]
    confirmation_rows = [states[index] for index in confirmation_indices]
    groups = np.asarray([row["case_id"] for row in discovery_rows])
    if (
        len(discovery_indices) != 96
        or len(confirmation_indices) != 192
        or len(set(groups)) != 24
    ):
        raise RuntimeError("unexpected discovery/confirmation split")

    analysis: dict[str, Any] = {
        "schema_version": 1,
        "status": "descriptive_not_causal",
        "position": "final_prompt",
        "discovery_states": 96,
        "discovery_cases": 24,
        "confirmation_states": 192,
        "confirmation_cases": 48,
        "alpha_selection": (
            "six-fold GroupKFold by discovery case at layer 32; selected "
            "alpha frozen across all layers; confirmation never used for "
            "selection"
        ),
        "remaining_budget_note": (
            "required_delta equals negative signed_error exactly and is not "
            "treated as an independent probe outcome"
        ),
        "models": {},
    }
    for model in ("base", "instruct"):
        residual = np.load(
            args.results_root
            / model
            / "capture"
            / "residual_selected.npy",
            mmap_mode="r",
        )
        if residual.shape != (288, 33, 4, 4096):
            raise RuntimeError(f"{model}: unexpected residual shape")
        model_result: dict[str, Any] = {}
        final_discovery = np.asarray(
            residual[
                discovery_indices, 32, FINAL_PROMPT_POSITION, :
            ],
            dtype=np.float32,
        )
        for name in OUTCOMES:
            y_discovery = outcome(discovery_rows, name)
            y_confirmation = outcome(confirmation_rows, name)
            alpha, cv_scores = select_alpha(
                final_discovery, y_discovery, groups
            )
            layers = []
            for layer in range(33):
                x_discovery = np.asarray(
                    residual[
                        discovery_indices,
                        layer,
                        FINAL_PROMPT_POSITION,
                        :,
                    ],
                    dtype=np.float32,
                )
                x_confirmation = np.asarray(
                    residual[
                        confirmation_indices,
                        layer,
                        FINAL_PROMPT_POSITION,
                        :,
                    ],
                    dtype=np.float32,
                )
                predicted = fit_predict(
                    x_discovery,
                    y_discovery,
                    x_confirmation,
                    alpha,
                )
                layer_result = {"layer": layer}
                layer_result.update(
                    metrics(name, y_confirmation, predicted)
                )
                layers.append(layer_result)
            key = (
                "sign_accuracy"
                if name == "signed_error"
                else "magnitude_accuracy"
                if name == "action_magnitude"
                else "r2"
            )
            best = max(layers, key=lambda row: row[key])
            model_result[name] = {
                "selected_alpha": alpha,
                "discovery_group_cv_normalized_mse": cv_scores,
                "fixed_layer32_confirmation": layers[32],
                "descriptive_best_confirmation": best,
                "layers": layers,
            }
        analysis["models"][model] = model_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "residual_probe_analysis.json"
    output_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        model: {
            name: {
                "alpha": analysis["models"][model][name][
                    "selected_alpha"
                ],
                "layer32": analysis["models"][model][name][
                    "fixed_layer32_confirmation"
                ],
                "best": analysis["models"][model][name][
                    "descriptive_best_confirmation"
                ],
            }
            for name in OUTCOMES
        }
        for model in ("base", "instruct")
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
