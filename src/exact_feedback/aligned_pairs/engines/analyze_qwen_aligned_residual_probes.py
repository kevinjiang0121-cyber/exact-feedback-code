#!/usr/bin/env python3
"""Case-disjoint residual probes for an audited aligned pair."""

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
FINAL_PROMPT = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def outcome(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    if name == "action_magnitude":
        return np.asarray([abs(int(row["signed_error"])) for row in rows], dtype=np.float32)
    return np.asarray([row[name] for row in rows], dtype=np.float32)


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train.astype(np.float32))
    test = scaler.transform(x_test.astype(np.float32))
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
    model.fit(train, y_train)
    return model.predict(test)


def select_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    splitter = GroupKFold(n_splits=6)
    scale = max(float(np.var(y)), 1e-8)
    scores = {}
    for alpha in ALPHAS:
        losses = []
        for train, test in splitter.split(x, y, groups):
            prediction = fit_predict(x[train], y[train], x[test], alpha)
            losses.append(float(np.mean((prediction - y[test]) ** 2) / scale))
        scores[str(alpha)] = float(np.mean(losses))
    return float(min(ALPHAS, key=lambda x: scores[str(x)])), scores


def metrics(name: str, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    correlation = None if float(np.std(truth)) == 0 or float(np.std(prediction)) == 0 else float(np.corrcoef(truth, prediction)[0, 1])
    result: dict[str, Any] = {
        "r2": float(r2_score(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "prediction_correlation": correlation,
    }
    if name == "signed_error":
        result["sign_accuracy"] = float(np.mean(np.sign(prediction) == np.sign(truth)))
    elif name == "action_magnitude":
        result["magnitude_accuracy"] = float(np.mean((prediction >= 6) == (truth >= 6)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--protocol", default="docs/QWEN_ALIGNED_MECHANISM_PROTOCOL_20260830.md")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    states = read_jsonl(args.states)
    discovery = np.asarray([i for i, row in enumerate(states) if row["split"] == "discovery"])
    confirmation = np.asarray([i for i, row in enumerate(states) if row["split"] == "confirmation"])
    if len(discovery) != 96 or len(confirmation) != 192:
        raise RuntimeError("unexpected fixed-state split")
    discovery_rows, confirmation_rows = [states[i] for i in discovery], [states[i] for i in confirmation]
    groups = np.asarray([row["case_id"] for row in discovery_rows])
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": args.protocol,
        "seed": args.seed,
        "status": "descriptive_not_causal",
        "primary_layer_rule": "final residual layer, frozen before outcomes",
        "models": {},
    }
    for slug in ("base", "instruct"):
        residual = np.load(args.results_root / slug / "capture" / "residual_selected.npy", mmap_mode="r")
        expected_shape_tail = (4, args.expected_hidden_size)
        if residual.shape[0] != 288 or residual.shape[2:] != expected_shape_tail or residual.shape[1] != args.expected_layers + 1:
            raise RuntimeError(f"{slug}: unexpected residual shape {residual.shape}")
        final_layer = residual.shape[1] - 1
        final_discovery = np.asarray(residual[discovery, final_layer, FINAL_PROMPT, :], dtype=np.float32)
        model_result = {}
        for name in OUTCOMES:
            y_discovery, y_confirmation = outcome(discovery_rows, name), outcome(confirmation_rows, name)
            alpha, cv = select_alpha(final_discovery, y_discovery, groups)
            layers = []
            for layer in range(residual.shape[1]):
                x_train = np.asarray(residual[discovery, layer, FINAL_PROMPT, :], dtype=np.float32)
                x_test = np.asarray(residual[confirmation, layer, FINAL_PROMPT, :], dtype=np.float32)
                cell = {"layer": layer, **metrics(name, y_confirmation, fit_predict(x_train, y_discovery, x_test, alpha))}
                layers.append(cell)
            ranking_key = "sign_accuracy" if name == "signed_error" else "magnitude_accuracy" if name == "action_magnitude" else "r2"
            model_result[name] = {
                "selected_alpha": alpha,
                "discovery_group_cv_normalized_mse": cv,
                "primary_final_layer_confirmation": layers[final_layer],
                "descriptive_best_confirmation": max(layers, key=lambda row: row[ranking_key]),
                "layers": layers,
            }
        report["models"][slug] = model_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({slug: {name: report["models"][slug][name]["primary_final_layer_confirmation"] for name in OUTCOMES} for slug in ("base", "instruct")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
