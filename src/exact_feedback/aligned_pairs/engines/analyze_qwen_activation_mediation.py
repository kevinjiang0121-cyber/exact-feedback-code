#!/usr/bin/env python3
"""Analyze gated Qwen bidirectional activation transplantation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metric_map() -> dict[str, Callable[[dict[str, Any]], float]]:
    return {
        "zero_action_signature": lambda row: float(int(row["realized_delta"]) == 0),
        "overshoot_avoidance": lambda row: float(abs(int(row["output_error"])) <= abs(int(row["signed_error"]))),
        "calibration_utility": lambda row: -min(abs(float(row["action_gain"]) - 1.0), 10.0),
    }


def effect(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], ids: list[str], metric: Callable[[dict[str, Any]], float], rng: np.random.Generator) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for state_id in ids:
        grouped.setdefault(left[state_id]["case_id"], []).append(metric(left[state_id]) - metric(right[state_id]))
    values = np.asarray([np.mean(grouped[x]) for x in sorted(grouped)])
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    boot = np.mean(values[draws], axis=1)
    return {"cases": len(values), "states": len(ids), "mean": float(np.mean(values)), "cluster_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.5, 0.975])]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-clean", required=True, type=Path)
    parser.add_argument("--instruct-clean", required=True, type=Path)
    parser.add_argument("--patched", required=True, type=Path)
    parser.add_argument("--dose-analysis", required=True, type=Path)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dose = json.loads(args.dose_analysis.read_text(encoding="utf-8"))
    groups = [x for x in args.groups.split(",") if x] or dose["confirmed_groups"]
    units = ["output_boundary_qwen" if group == "output" else group for group in groups]
    base = {x["state_id"]: x for x in read_jsonl(args.base_clean) if x["split"] == args.split}
    instruct = {x["state_id"]: x for x in read_jsonl(args.instruct_clean) if x["split"] == args.split}
    patched_rows = read_jsonl(args.patched) if args.patched.exists() else []
    patched = {(x["state_id"], x["unit"], x["direction"]): x for x in patched_rows}
    expected_states = 96 if args.split == "discovery" else 192
    integrity = {"passes": len(base) == expected_states and len(instruct) == expected_states and len(patched) == expected_states * len(units) * 2, "base_states": len(base), "instruct_states": len(instruct), "patched_cells": len(patched), "expected_cells": expected_states * len(units) * 2}
    if not integrity["passes"]:
        raise RuntimeError(f"integrity failure: {integrity}")
    rng = np.random.default_rng(20260830)
    metrics = metric_map()
    results, eligible = {}, []
    ids = sorted(set(base) & set(instruct))
    for group, unit in zip(groups, units):
        denoised = {x: patched[(x, unit, "denoise")] for x in ids}
        noised = {x: patched[(x, unit, "noise")] for x in ids}
        cells = {}
        for metric_name, metric in metrics.items():
            denoise = effect(denoised, base, ids, metric, rng)
            noise = effect(instruct, noised, ids, metric, rng)
            cells[metric_name] = {"denoising_indirect": denoise, "noising_loss": noise, "bidirectional_mean": (denoise["mean"] + noise["mean"]) / 2}
        selecting = dose["results"][group]["selecting_signatures"]
        selecting_metrics = ["zero_action_signature" if x == "zero_action" else "overshoot_avoidance" for x in selecting]
        signature_pass = any(cells[x]["denoising_indirect"]["mean"] > 0 and cells[x]["noising_loss"]["mean"] > 0 for x in selecting_metrics)
        best_signature = max((cells[x]["bidirectional_mean"] for x in selecting_metrics), default=0.0)
        calibration_ok = cells["calibration_utility"]["bidirectional_mean"] >= -abs(best_signature)
        if args.split == "confirmation":
            interval_pass = any(cells[x]["denoising_indirect"]["cluster_bootstrap_95"][0] > 0 or cells[x]["noising_loss"]["cluster_bootstrap_95"][0] > 0 for x in selecting_metrics)
        else:
            interval_pass = True
        passed = signature_pass and calibration_ok and interval_pass
        results[group] = {"unit": unit, "selecting_signatures": selecting, "effects": cells, "signature_bidirectional_pass": signature_pass, "calibration_noncontradiction_pass": calibration_ok, "interval_pass": interval_pass, "mediation_pass": passed}
        if passed:
            eligible.append(group)
    report = {"schema_version": 1, "protocol": "docs/QWEN_ALIGNED_MECHANISM_PROTOCOL_20260830.md", "split": args.split, "integrity": integrity, "results": results, "eligible_groups": eligible}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
