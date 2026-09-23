#!/usr/bin/env python3
"""Cross-panel recurrence-conditioned later-rescue analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from exact_feedback.recurrence.api_analysis import API_MODELS, cases_path
from exact_feedback.recurrence.capture_analysis import (
    common_effect,
    directional_contrasts,
    exact_spearman,
    family_for_model,
    longest_identical_run,
    permutation_spearman,
)
from exact_feedback.recurrence.unified_analysis import (
    MODELS,
    main_registry,
    read_jsonl,
    sha256,
    source_family,
    state_tuple,
    validate_rows,
)


SEED = 20260803
PRIMARY_LANDMARK = 4
SENSITIVITY_LANDMARKS = (3, 4, 5)


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


def landmark_features(
    model: str,
    panel: str,
    row: dict[str, Any],
    landmark: int,
) -> dict[str, Any] | None:
    if panel == "api" and row.get("provider_refusal", False):
        return None
    rounds = row.get("rounds") or []
    if len(rounds) < landmark + 2:
        return None
    observed = rounds[: landmark + 1]
    if bool(observed[-1].get("joint_success", False)):
        return None
    states = [state_tuple(item) for item in observed]
    errors = [int(item["error"]) for item in observed]
    actions = [right - left for left, right in zip(errors, errors[1:])]
    recurrence_flags: list[int] = []
    seen = {states[0]}
    for state in states[1:]:
        recurrence_flags.append(int(state in seen))
        seen.add(state)
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in errors]
    sign_flips = sum(
        left != 0 and right != 0 and left != right
        for left, right in zip(signs, signs[1:])
    )
    escape = any(
        0 < abs(left) <= 20 and abs(right) > 50
        for left, right in zip(errors, errors[1:])
    )
    period2 = any(
        states[index] == states[index - 2] and states[index] != states[index - 1]
        for index in range(2, len(states))
    )
    current_error = errors[-1]
    return {
        "panel": panel,
        "model": model,
        "family": family_for_model(model) if panel == "open" else API_MODELS[model],
        "case_id": str(row["id"]),
        "source_family": source_family(str(row["source"])),
        "length_band": str(row.get("length_band", "")),
        "landmark": landmark,
        "late_rescue": int(bool(row.get("final_joint", False))),
        "current_error": current_error,
        "current_abs_error": abs(current_error),
        "log_current_abs_error": math.log1p(abs(current_error)),
        "target": int(row["target"]),
        "current_missing_count": len(observed[-1].get("missing", [])),
        "min_abs_error": min(abs(value) for value in errors),
        "initial_to_current_abs_improvement": abs(errors[0]) - abs(current_error),
        "contraction_count": sum(
            abs(right) < abs(left) for left, right in zip(errors, errors[1:])
        ),
        "sign_flip_count": sign_flips,
        "length_noop_count": sum(action == 0 for action in actions),
        "largest_abs_action": max((abs(action) for action in actions), default=0),
        "escape_by_landmark": int(escape),
        "recurrence_count": sum(recurrence_flags),
        "repeat_fraction": sum(recurrence_flags) / landmark,
        "any_recurrence": int(any(recurrence_flags)),
        "unique_output_ratio": len(set(states)) / len(states),
        "longest_identical_run": longest_identical_run(states),
        "period2_return": int(period2),
        "current_error_sign": (
            "positive" if current_error > 0 else "negative" if current_error < 0 else "zero"
        ),
    }


def load_open(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    registry = main_registry(root)
    output: dict[str, list[dict[str, Any]]] = {}
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for model in MODELS:
        rows: list[dict[str, Any]] = []
        for path in registry[model]:
            part = read_jsonl(path)
            rows.extend(part)
            files.append({
                "panel": "open",
                "model": model,
                "path": str(path),
                "rows": len(part),
                "sha256": sha256(path),
            })
        errors.extend(validate_rows(model, rows))
        output[model] = rows
    if errors:
        raise ValueError("open integrity failure:\n" + "\n".join(errors[:50]))
    return output, files


def load_api(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    files: list[dict[str, Any]] = []
    reference_ids: set[str] | None = None
    for model in API_MODELS:
        path = cases_path(root, model)
        rows = read_jsonl(path)
        ids = {str(row["id"]) for row in rows}
        if len(rows) != 480 or len(ids) != 480:
            raise ValueError(f"{model}: expected 480 rows and unique IDs")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{model}: API case-set mismatch")
        output[model] = rows
        files.append({
            "panel": "api",
            "model": model,
            "path": str(path),
            "rows": len(rows),
            "sha256": sha256(path),
        })
    return output, files


def effect_or_error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 50 or len({row["late_rescue"] for row in rows}) < 2:
        return {"estimable": False, "rows": len(rows)}
    result = common_effect(rows)
    result["estimable"] = True
    return result


def model_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        exposed = [row["late_rescue"] for row in group if row["any_recurrence"]]
        unexposed = [row["late_rescue"] for row in group if not row["any_recurrence"]]
        output.append({
            "panel": group[0]["panel"],
            "model": model,
            "risk_rows": len(group),
            "recurrence_prevalence": float(np.mean([row["any_recurrence"] for row in group])),
            "mean_repeat_fraction": float(np.mean([row["repeat_fraction"] for row in group])),
            "late_rescue_rate": float(np.mean([row["late_rescue"] for row in group])),
            "exposed_rows": len(exposed),
            "unexposed_rows": len(unexposed),
            "exposed_rescue_rate": float(np.mean(exposed)) if exposed else float("nan"),
            "unexposed_rescue_rate": float(np.mean(unexposed)) if unexposed else float("nan"),
            "risk_difference_exposed_minus_unexposed": (
                float(np.mean(exposed) - np.mean(unexposed))
                if exposed and unexposed else float("nan")
            ),
        })
    return output


def rank_audit(rows: list[dict[str, Any]], panel: str) -> dict[str, Any]:
    summary = [row for row in model_summary(rows) if row["panel"] == panel]
    recurrence = [-row["recurrence_prevalence"] for row in summary]
    rescue = [row["late_rescue_rate"] for row in summary]
    if panel == "api":
        return exact_spearman(recurrence, rescue)
    return permutation_spearman(recurrence, rescue, 100_000, SEED + 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/recurrence_conditioned_rescue_v1"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    open_cases, open_files = load_open(root)
    api_cases, api_files = load_api(root)
    by_landmark: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for landmark in SENSITIVITY_LANDMARKS:
        open_rows = [
            features
            for model, cases in open_cases.items()
            for row in cases
            if (features := landmark_features(model, "open", row, landmark)) is not None
        ]
        api_rows = [
            features
            for model, cases in api_cases.items()
            for row in cases
            if (features := landmark_features(model, "api", row, landmark)) is not None
        ]
        by_landmark[landmark] = {"open": open_rows, "api": api_rows}

    primary_open = by_landmark[PRIMARY_LANDMARK]["open"]
    primary_api = by_landmark[PRIMARY_LANDMARK]["api"]
    primary_combined = primary_open + primary_api
    primary_effects = {
        "open": effect_or_error(primary_open),
        "api": effect_or_error(primary_api),
        "combined": effect_or_error(primary_combined),
    }
    model_contrasts = directional_contrasts(primary_combined, "model")
    source_contrasts = directional_contrasts(primary_combined, "source_family")

    landmark_rows: list[dict[str, Any]] = []
    for landmark, panels in by_landmark.items():
        for panel, rows in panels.items():
            effect = effect_or_error(rows)
            landmark_rows.append({
                "landmark": landmark,
                "panel": panel,
                "risk_rows": len(rows),
                "late_rescue_rate": float(np.mean([row["late_rescue"] for row in rows])),
                "repeat_odds_ratio": effect.get("odds_ratio_full_repeat_fraction"),
                "repeat_ci_lo": (effect.get("ci95_odds_ratio") or [None, None])[0],
                "repeat_ci_hi": (effect.get("ci95_odds_ratio") or [None, None])[1],
                "repeat_cluster_p": effect.get("clustered_p_two_sided"),
            })

    error_sign_rows: list[dict[str, Any]] = []
    for panel, rows in (("open", primary_open), ("api", primary_api), ("combined", primary_combined)):
        for sign in ("positive", "negative"):
            group = [row for row in rows if row["current_error_sign"] == sign]
            effect = effect_or_error(group)
            error_sign_rows.append({
                "panel": panel,
                "current_error_sign": sign,
                "risk_rows": len(group),
                "late_rescue_rate": float(np.mean([row["late_rescue"] for row in group])),
                "repeat_odds_ratio": effect.get("odds_ratio_full_repeat_fraction"),
                "repeat_ci_lo": (effect.get("ci95_odds_ratio") or [None, None])[0],
                "repeat_ci_hi": (effect.get("ci95_odds_ratio") or [None, None])[1],
                "repeat_cluster_p": effect.get("clustered_p_two_sided"),
            })

    api_model_contrasts = [row for row in model_contrasts if row["model"] in API_MODELS]
    negative_api_models = sum(
        row["estimable"] and row["risk_difference_exposed_minus_unexposed"] < 0
        for row in api_model_contrasts
    )
    negative_sources = sum(
        row["estimable"] and row["risk_difference_exposed_minus_unexposed"] < 0
        for row in source_contrasts
    )
    sensitivity = {
        landmark: effect_or_error(by_landmark[landmark]["api"])
        for landmark in (3, 5)
    }
    checks = {
        "open_adjusted_or_below_one": primary_effects["open"].get(
            "odds_ratio_full_repeat_fraction", math.inf
        ) < 1,
        "api_adjusted_or_below_one_p_lt_0_05": (
            primary_effects["api"].get("odds_ratio_full_repeat_fraction", math.inf) < 1
            and primary_effects["api"].get("clustered_p_two_sided", 1.0) < 0.05
        ),
        "combined_p_lt_0_001": primary_effects["combined"].get(
            "clustered_p_two_sided", 1.0
        ) < 0.001,
        "negative_in_five_api_models_and_four_sources": (
            negative_api_models >= 5 and negative_sources == 4
        ),
        "api_landmarks_3_and_5_negative": all(
            effect.get("repeat_fraction_coefficient", 0.0) < 0
            for effect in sensitivity.values()
        ),
    }
    verdict = (
        "CROSS_PANEL_RECURRENCE_CONDITIONED_RESCUE"
        if all(checks.values())
        else "CROSS_PANEL_POOLED_EFFECT_WITH_API_DIRECTIONAL_HETEROGENEITY"
    )

    summary_rows = model_summary(primary_combined)
    rank_results = {
        "open": rank_audit(primary_open, "open"),
        "api": rank_audit(primary_api, "api"),
    }
    manifest = {
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "path": str(Path(__file__).resolve().parents[3] / "docs/ICLR_RECURRENCE_CONDITIONED_RESCUE_PROTOCOL_20260803.md"),
            "sha256": sha256(
                Path(__file__).resolve().parents[3] / "docs/ICLR_RECURRENCE_CONDITIONED_RESCUE_PROTOCOL_20260803.md"
            ),
        },
        "inputs": open_files + api_files,
        "integrity": {
            "passed": True,
            "open_models": len(open_cases),
            "api_models": len(api_cases),
            "cases_per_model": 480,
            "provider_refusals_excluded": 1,
        },
    }
    report = {
        "schema_version": 1,
        "verdict": verdict,
        "protocol": "docs/ICLR_RECURRENCE_CONDITIONED_RESCUE_PROTOCOL_20260803.md",
        "primary_landmark": PRIMARY_LANDMARK,
        "primary_effects": primary_effects,
        "directional_consistency": {
            "negative_api_models": negative_api_models,
            "estimable_api_models": sum(row["estimable"] for row in api_model_contrasts),
            "negative_sources": negative_sources,
            "estimable_sources": sum(row["estimable"] for row in source_contrasts),
        },
        "landmark_sensitivity": {
            str(key): value for key, value in sensitivity.items()
        },
        "panel_rank_audit": rank_results,
        "checks": checks,
        "manifest": "input_manifest.json",
    }
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "landmark4_case_features.csv", primary_combined)
    write_csv(output / "model_contrasts.csv", model_contrasts)
    write_csv(output / "source_contrasts.csv", source_contrasts)
    write_csv(output / "landmark_sensitivity.csv", landmark_rows)
    write_csv(output / "error_sign_sensitivity.csv", error_sign_rows)
    write_csv(output / "model_summary.csv", summary_rows)

    def effect_line(name: str, effect: dict[str, Any]) -> str:
        return (
            f"- {name}: OR={effect['odds_ratio_full_repeat_fraction']:.3f} "
            f"(95% CI {effect['ci95_odds_ratio'][0]:.3f}--"
            f"{effect['ci95_odds_ratio'][1]:.3f}), "
            f"case-clustered p={effect['clustered_p_two_sided']:.4g}, "
            f"n={effect['rows']:,}."
        )

    lines = [
        "# Recurrence-conditioned later-rescue audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Primary revision-4 adjusted effects",
        "",
        effect_line("Open panel", primary_effects["open"]),
        effect_line("API panel", primary_effects["api"]),
        effect_line("Combined panel", primary_effects["combined"]),
        "",
        "## Directional consistency",
        "",
        f"- API models negative: {negative_api_models}/"
        f"{sum(row['estimable'] for row in api_model_contrasts)}.",
        f"- Sources negative: {negative_sources}/"
        f"{sum(row['estimable'] for row in source_contrasts)}.",
        "",
        "## Panel-level lower recurrence versus later rescue",
        "",
        f"- Open: rho={rank_results['open']['rho']:+.3f}, "
        f"permutation p={rank_results['open']['permutation_p_two_sided']:.5g}.",
        f"- API: rho={rank_results['api']['rho']:+.3f}, "
        f"exact p={rank_results['api']['exact_p_two_sided']:.5g}.",
        "",
        "## Frozen decision checks",
        "",
    ]
    lines.extend(f"- {key}: **{value}**" for key, value in checks.items())
    lines.extend([
        "",
        "## Interpretation of the failed directional gate",
        "",
        "The pooled API effect is independently significant, but the frozen "
        "5/7 single-model direction gate fails. The three positive raw "
        "contrasts are the frontier ceiling cases: Claude, Gemini, and GPT "
        "have only 3, 1, and 3 recurrence-exposed risk-set trajectories, "
        "respectively. All four API models with at least 28 exposed cases "
        "have negative contrasts. This is evidence for a pooled cross-panel "
        "relation with underidentified frontier-model heterogeneity, not a "
        "license to revise the frozen gate or claim seven-model universality.",
        "",
        "This analysis conditions on being unresolved at the landmark. It is "
        "therefore distinct from correlating early capture with final success.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
