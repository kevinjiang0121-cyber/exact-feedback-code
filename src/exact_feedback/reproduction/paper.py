from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import python_environment
from .registry import API_PANEL, HISTORICAL_EXACT_SIX, MAIN_OPEN, TULU_PANEL


@dataclass(frozen=True)
class RequiredOutput:
    section: str
    path: str
    rows: int | None = None


ANALYSIS_SLUG = {
    "llama31_8b": "llama31_8b_instruct",
    "gemma2_9b": "gemma2_9b_it",
}

EXTRA_EXACT_DEST = {
    "qwen3_1_7b": "qwen3_1_7b/combined480",
    "qwen3_4b": "qwen3_4b/combined480",
    "qwen3_32b": "qwen3_32b/vllm_pp2_combined480",
    "qwen3_30b_a3b": "qwen3_30b_a3b/vllm_pp2_combined480",
    "granite_3_3_8b": "granite_3_3_8b/combined480",
    "falcon_h1_7b": "falcon_h1_7b/combined480",
}

TULU_LEGACY = {
    "llama31_8b_base": "llama31_8b_base",
    "tulu3_8b_sft": "tulu3_8b_sft",
    "tulu3_8b_dpo": "tulu3_8b_dpo",
    "tulu31_8b_rlvr": "tulu31_8b_rlvr",
    "llama31_8b": "llama31_8b_instruct",
}


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def requirements() -> list[RequiredOutput]:
    required: list[RequiredOutput] = []
    for model in HISTORICAL_EXACT_SIX:
        for split, rows in (("primary120", 120), ("replication120", 120), ("extension240", 240)):
            required.append(RequiredOutput("closed-loop exact/open", f"closed_loop_exact/{model}/{split}/cases.jsonl", rows))
            required.append(RequiredOutput("closed-loop exact/open", f"closed_loop_exact/{model}/{split}/integrity_audit.json"))
    for model in EXTRA_EXACT_DEST:
        required.append(RequiredOutput("closed-loop exact/open", f"closed_loop_exact/{model}/cases.jsonl", 480))
        required.append(RequiredOutput("closed-loop exact/open", f"closed_loop_exact/{model}/integrity_audit.json"))
    for model in MAIN_OPEN:
        required.append(RequiredOutput("closed-loop structured/open", f"closed_loop_structured/{model}/cases.jsonl", 960))
        required.append(RequiredOutput("closed-loop structured/open", f"closed_loop_structured/{model}/audit.json"))
    for model in API_PANEL:
        required.append(RequiredOutput("closed-loop exact/API", f"closed_loop_api_exact/{model}/{model}/baseline/cases.jsonl", 480))
        required.append(RequiredOutput("closed-loop exact/API", f"closed_loop_api_exact/{model}/{model}/baseline/integrity_audit.json"))
        for domain in ("lexical_constraints", "compositional_constraints"):
            required.append(RequiredOutput("closed-loop structured/API", f"closed_loop_api_structured/{model}/{domain}/{model}/baseline/cases.jsonl", 480))
            required.append(RequiredOutput("closed-loop structured/API", f"closed_loop_api_structured/{model}/{domain}/{model}/baseline/integrity_audit.json"))
    for model in (*TULU_PANEL,):
        required.append(RequiredOutput("Tulu common interface", f"tulu_common_interface/{model}/exact/cases.jsonl", 480))
        required.append(RequiredOutput("Tulu common interface", f"tulu_common_interface/{model}/exact/integrity_audit.json"))
        required.append(RequiredOutput("Tulu common interface", f"tulu_common_interface/{model}/structured/cases.jsonl", 960))
        required.append(RequiredOutput("Tulu common interface", f"tulu_common_interface/{model}/structured/audit.json"))
    for model in TULU_PANEL[1:]:
        required.append(RequiredOutput("Tulu native interface", f"tulu_native_interface/{model}/exact/cases.jsonl", 480))
        required.append(RequiredOutput("Tulu native interface", f"tulu_native_interface/{model}/exact/integrity_audit.json"))
        required.append(RequiredOutput("Tulu native interface", f"tulu_native_interface/{model}/structured/cases.jsonl", 960))
        required.append(RequiredOutput("Tulu native interface", f"tulu_native_interface/{model}/structured/audit.json"))
    for model in ("llama31_8b_base", "llama31_8b", "qwen3_8b_base", "qwen3_8b", "qwen3_1_7b", "qwen3_4b", "qwen3_14b", "gemma2_9b", "qwen3_32b"):
        for split in ("discovery", "confirmation"):
            required.append(RequiredOutput("fixed-draft response law", f"response_law/{model}/{split}/cases.jsonl"))
            required.append(RequiredOutput("fixed-draft response law", f"response_law/{model}/{split}/integrity_audit.json"))
    for line, models, rows in (
        ("prompt robustness", ("llama31_8b", "gemma2_9b", "qwen3_14b"), 60),
        ("decoding robustness", ("llama31_8b", "gemma2_9b", "qwen3_14b"), 60),
    ):
        arms = ("baseline", "structured", "concise") if line.startswith("prompt") else ("seed_20260725", "seed_20260726", "seed_20260727")
        key = "prompt_robustness" if line.startswith("prompt") else "decoding_robustness"
        for model in models:
            for arm in arms:
                required.append(RequiredOutput(line, f"{key}/{model}/{arm}/cases.jsonl", rows))
    for model in HISTORICAL_EXACT_SIX:
        required.append(RequiredOutput("revision-32 persistence", f"persistence_32/{model}/cases.jsonl"))
    for model, states in (("llama31_8b", 83), ("glm4_9b", 120)):
        required.append(RequiredOutput("exact history reset", f"history_reset_exact/{model}/cases.jsonl", states * 2))
    for model in ("gemma2_9b", "qwen3_14b", "falcon_h1_7b"):
        required.append(RequiredOutput("structured history reset", f"history_reset_structured/{model}/cases.jsonl"))
    required.append(RequiredOutput("partial-history intervention", "history_partial/glm4_9b/cases.jsonl", 240))
    for model in ("llama31_8b_base", "llama31_8b"):
        required.append(RequiredOutput("residual probes", f"residual_probe_capture/{model}/residual_selected.npy"))
    for line, rel in (
        ("parameter restoration", "parameter_restoration/paired/analysis/dose_confirmation.json"),
        ("activation transplant", "activation_transplant/paired/analysis.json"),
        ("controller graft", "controller_graft/paired/analysis.json"),
    ):
        required.append(RequiredOutput(line, rel))
    return required


def check_outputs(runs: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    by_section: dict[str, list[RequiredOutput]] = {}
    for item in requirements():
        by_section.setdefault(item.section, []).append(item)
    for section, items in by_section.items():
        section_errors = []
        for item in items:
            path = runs / item.path
            if not path.is_file():
                section_errors.append(f"missing {item.path}")
            elif item.rows is not None and path.suffix == ".jsonl":
                actual = count_jsonl(path)
                if actual != item.rows:
                    section_errors.append(f"{item.path}: rows={actual}, expected={item.rows}")
            if path.is_file() and path.name in {"integrity_audit.json", "audit.json"}:
                try:
                    verdict = json.loads(path.read_text(encoding="utf-8")).get("verdict")
                except (OSError, ValueError) as exc:
                    section_errors.append(f"{item.path}: unreadable audit ({exc})")
                else:
                    if verdict != "PASS":
                        section_errors.append(f"{item.path}: verdict={verdict!r}")
        if section_errors:
            failures.extend(f"{section}: {error}" for error in section_errors)
            print(f"PAPER_LINE_FAIL {section} ({len(section_errors)} issue(s))")
        else:
            print(f"PAPER_LINE_OK   {section} ({len(items)} artifact(s))")
    return (1 if failures else 0), failures


def copy_file(source: Path, target: Path, *, optional: bool = False) -> None:
    if not source.is_file():
        if optional:
            return
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare_compatibility(root: Path, runs: Path) -> Path:
    exp = root / "experiments"
    for model in HISTORICAL_EXACT_SIX:
        for split, legacy in (("primary120", "human_generation_main120_v1"), ("replication120", "human_generation_replication120_v1"), ("extension240", "human_generation_extension240_v1")):
            copy_file(runs / f"closed_loop_exact/{model}/{split}/cases.jsonl", exp / legacy / model / "cases.jsonl")
    for model, suffix in EXTRA_EXACT_DEST.items():
        copy_file(runs / f"closed_loop_exact/{model}/cases.jsonl", exp / "model_generality_combined480_v1" / suffix / "cases.jsonl")
    for model in MAIN_OPEN:
        copy_file(runs / f"closed_loop_structured/{model}/cases.jsonl", exp / "multidomain_full480_open12_v1" / model / "cases.jsonl")
    for model in API_PANEL:
        copy_file(runs / f"closed_loop_api_exact/{model}/{model}/baseline/cases.jsonl", exp / "model_generality_combined480_api_v1" / model / model / "baseline/cases.jsonl")
        for domain in ("lexical_constraints", "compositional_constraints"):
            copy_file(runs / f"closed_loop_api_structured/{model}/{domain}/{model}/baseline/cases.jsonl", exp / "model_generality_multidomain_api_v1" / domain / model / "baseline/cases.jsonl")
    for model in ("llama31_8b_base", "llama31_8b", "qwen3_8b_base", "qwen3_8b", "qwen3_1_7b", "qwen3_4b", "qwen3_14b", "gemma2_9b", "qwen3_32b"):
        slug = ANALYSIS_SLUG.get(model, model)
        for split in ("discovery", "confirmation"):
            source = runs / "response_law" / model / split
            target = exp / "feedback_policy_system_identification_v1" / split / slug
            for name in ("cases.jsonl", "runtime_manifest.json", "integrity_audit.json"):
                copy_file(source / name, target / name)
            if model == "qwen3_32b":
                copy_file(source / "integrity_audit.json", target / "local_integrity_audit.json")
    copy_file(root / "data/history_reset_closed_loop_state_matched_v1/states.jsonl", exp / "history_reset_closed_loop_state_matched_v1/states.jsonl")
    for model in ("llama31_8b", "glm4_9b"):
        for name in ("cases.jsonl", "runtime_manifest.json"):
            copy_file(runs / "history_reset_exact" / model / name, exp / "history_reset_closed_loop_state_matched_v1" / model / "run" / name, optional=name == "runtime_manifest.json")
    copy_file(root / "data/targeted_history_reset_confirmation_v1/states.jsonl", exp / "targeted_history_reset_confirmation_v1/states.jsonl")
    for model in ("gemma2_9b", "qwen3_14b", "falcon_h1_7b"):
        for name in ("cases.jsonl", "runtime_manifest.json"):
            copy_file(runs / "history_reset_structured" / model / name, exp / "targeted_history_reset_confirmation_v1" / model / name)
    copy_file(root / "data/glm_partial_reset_matched_pilot_v1/states.jsonl", exp / "glm_partial_reset_matched_pilot_v1/states.jsonl")
    for name in ("cases.jsonl", "runtime_manifest.json"):
        copy_file(runs / "history_partial/glm4_9b" / name, exp / "glm_partial_reset_matched_pilot_v1/glm4_9b" / name)
    residual = exp / "phase3a_fixed_state_capture_v1"
    for model, legacy in (("llama31_8b_base", "base"), ("llama31_8b", "instruct")):
        source = runs / "residual_probe_capture" / model
        for name in ("residual_selected.npy", "cases.jsonl", "runtime_manifest.json"):
            copy_file(source / name, residual / legacy / "capture" / name, optional=name != "residual_selected.npy")
    for model in TULU_PANEL:
        slug = TULU_LEGACY[model]
        copy_file(runs / f"tulu_common_interface/{model}/exact/cases.jsonl", exp / "llama_training_stage_origin_v1" / slug / "vllm_common_plain_combined480/cases.jsonl")
        copy_file(runs / f"tulu_common_interface/{model}/structured/cases.jsonl", exp / "llama_tulu_multidomain_stage_v1" / slug / "cases.jsonl")
    for model in TULU_PANEL[1:]:
        slug = TULU_LEGACY[model]
        exact = runs / f"tulu_native_interface/{model}/exact/cases.jsonl"
        structured = runs / f"tulu_native_interface/{model}/structured/cases.jsonl"
        if model in {"tulu3_8b_dpo", "llama31_8b"}:
            exact_target = exp / "llama_training_stage_origin_v1" / slug / "vllm_native_combined480/cases.jsonl"
        else:
            exact_target = exp / "llama_tulu_native_interface_ablation_v1" / slug / "exact_length/cases.jsonl"
        copy_file(exact, exact_target)
        if model == "llama31_8b":
            structured_target = exp / "llama_tulu_native_interface_ablation_v1/reused_meta_instruct_native/llama31_8b/cases.jsonl"
        else:
            structured_target = exp / "llama_tulu_native_interface_ablation_v1" / slug / "structured/cases.jsonl"
        copy_file(structured, structured_target)
    return exp


def analysis_commands(root: Path, runs: Path, quick: bool) -> list[list[str]]:
    py = sys.executable
    exp = root / "experiments"
    out = root / "analysis" / "recomputed"
    draws = "200" if quick else "20000"
    permutations = "200" if quick else "100000"
    recurrence_draws = "200" if quick else "10000"
    commands = [
        [py, str(root / "src/exact_feedback/response_law/analysis.py"), "--phase", "discovery", "--experiment-dir", str(exp / "feedback_policy_system_identification_v1/discovery"), "--output-dir", str(out / "response_law")],
        [py, str(root / "src/exact_feedback/response_law/analysis.py"), "--phase", "confirmation", "--experiment-dir", str(exp / "feedback_policy_system_identification_v1/confirmation"), "--output-dir", str(out / "response_law"), "--prediction-spec", str(out / "response_law/frozen_prediction_spec.json")],
        [py, str(root / "src/exact_feedback/response_law/qwen32_analysis.py"), "--experiment-dir", str(exp / "feedback_policy_system_identification_v1"), "--prediction-spec", str(out / "response_law/frozen_prediction_spec.json"), "--output", str(out / "response_law/qwen32_extension.json")],
        [py, str(root / "src/exact_feedback/robustness/prompt_analysis.py"), "--experiment-root", str(runs / "prompt_robustness"), "--output-dir", str(out / "prompt_robustness")],
        [py, str(root / "src/exact_feedback/robustness/decoding_analysis.py"), "--root", str(runs / "decoding_robustness"), "--prompt-analysis", str(out / "prompt_robustness/prompt_robustness_analysis.json"), "--output-dir", str(out / "decoding_robustness")],
        [py, str(root / "src/exact_feedback/response_law/transfer_analysis.py"), "--root", str(root), "--prediction-spec", str(out / "response_law/frozen_prediction_spec.json"), "--discovery-analysis", str(out / "response_law/discovery_analysis.json"), "--bootstraps", recurrence_draws, "--output-dir", str(exp / "policy_to_closed_loop_spotlight_gate_v1")],
        [py, str(root / "src/exact_feedback/recurrence/unified_analysis.py"), "--root", str(root), "--output-dir", str(exp / "unified_output_recurrence_audit_v2"), "--bootstraps", recurrence_draws, "--permutations", permutations],
        [py, str(root / "src/exact_feedback/recurrence/capture_analysis.py"), "--root", str(root), "--output-dir", str(out / "capture_recurrence_law"), "--bootstraps", recurrence_draws, "--permutations", permutations],
        [py, str(root / "src/exact_feedback/recurrence/api_analysis.py"), "--root", str(root), "--open-hazards", str(out / "capture_recurrence_law/early_hazards.csv"), "--output-dir", str(out / "api_capture_recurrence"), "--permutations", permutations],
        [py, str(root / "src/exact_feedback/recurrence/rescue_analysis.py"), "--root", str(root), "--output-dir", str(out / "conditioned_rescue")],
        [py, str(root / "src/exact_feedback/recurrence/three_domain_analysis.py"), "--root", str(root), "--output-dir", str(out / "three_domain"), "--bootstraps", recurrence_draws, "--legacy-analysis", str(out / "capture_recurrence_law/analysis.json")],
        [py, str(root / "src/exact_feedback/closed_loop/api_analysis.py"), "--project-root", str(root), "--output-dir", str(out / "api_interactions"), "--draws", draws, "--fresh"],
        [py, str(root / "src/exact_feedback/persistence/analysis.py"), "--selection-dir", str(root / "data/persistence_revision32_v1"), "--run-root", str(runs / "persistence_32"), "--output-dir", str(out / "persistence_32")],
        [py, str(root / "src/exact_feedback/history/exact_analysis.py"), "--experiment-dir", str(exp / "history_reset_closed_loop_state_matched_v1"), "--output-dir", str(out / "history_exact"), "--iterations", draws],
        [py, str(root / "src/exact_feedback/history/structured_analysis.py"), "--experiment-dir", str(exp / "targeted_history_reset_confirmation_v1"), "--output-dir", str(out / "history_structured"), "--iterations", draws],
        [py, str(root / "src/exact_feedback/history/partial_analysis.py"), "--pilot-root", str(exp / "glm_partial_reset_matched_pilot_v1"), "--output-dir", str(out / "history_partial"), "--iterations", draws],
        [py, str(root / "src/exact_feedback/localization/probe_analysis.py"), "--states", str(root / "data/phase3a_fixed_states_v1/states.jsonl"), "--results-root", str(exp / "phase3a_fixed_state_capture_v1"), "--output-dir", str(out / "residual_probes")],
        [py, str(root / "src/exact_feedback/aligned_models/analysis.py"), "--project-root", str(root), "--output-dir", str(out / "tulu_stage_interface"), "--draws", draws],
    ]
    for key in ("fixed_draft_crossover", "qwen_aligned", "gemma_aligned"):
        source = runs / key / "paired" if key.endswith("aligned") else runs / key
        commands.append([py, str(root / "reproduce.py"), "analyze-experiment", key,
                         "--runs-root", str(source), "--output", str(out / key)])
    commands.append([py,str(root / "reproduce.py"),"analyze-experiment","history_trigger",
                     "--runs-root",str(exp / "targeted_history_reset_confirmation_v1"),
                     "--output",str(out / "history_trigger")])
    commands.append([py,str(root / "src/exact_feedback/closed_loop/budget_analysis.py"),
                     "--root",str(root),"--output",str(out / "round_budget"),"--fresh"])
    commands.append([py,str(root / "src/exact_feedback/closed_loop/trajectory_diagnostics.py"),
                     "--root",str(root),"--output",str(out / "trajectory_diagnostics")])
    commands.append([py,str(root / "reproduce.py"),"analyze-experiment","failure_categories",
                     "--runs-root",str(root),"--output",str(out / "failure_categories"),"--fresh"])
    return commands


def content_contract_commands(root: Path, runs: Path, cfg: dict, quick: bool) -> list[list[str]]:
    annotations = cfg.get("annotations", {})
    stanza_resources = Path(annotations.get("stanza_resources_dir", ""))
    nli_model = Path(annotations.get("nli_model_path", ""))
    if not str(annotations.get("stanza_resources_dir", "")).strip():
        raise SystemExit("annotations.stanza_resources_dir is required")
    if not str(annotations.get("nli_model_path", "")).strip():
        raise SystemExit("annotations.nli_model_path is required")
    py = sys.executable
    out = root / "analysis/recomputed/content_contract"
    registry = out / "registry"
    stanza = out / "stanza.jsonl"
    nli = out / "nli.jsonl"
    endpoint = out / "endpoint"
    registry_command = [py, str(root / "src/exact_feedback/content_contract/registry.py"),
                        "--frozen-data", str(root / "data/human_generation_combined480_v1/generation_combined480.jsonl")]
    for model in API_PANEL:
        registry_command += ["--model", model, str(runs / f"closed_loop_api_exact/{model}/{model}/baseline/cases.jsonl")]
    registry_command += ["--output-dir", str(registry)]
    return [
        registry_command,
        [py, str(root / "src/exact_feedback/content_contract/stanza_annotation.py"), "--registry", str(registry / "text_registry.jsonl"), "--resources-dir", str(stanza_resources), "--output", str(stanza), "--manifest", str(out / "stanza_manifest.json")],
        [py, str(root / "src/exact_feedback/content_contract/nli_annotation.py"), "--registry", str(registry / "text_registry.jsonl"), "--stanza", str(stanza), "--model-dir", str(nli_model), "--output", str(nli), "--manifest", str(out / "nli_manifest.json")],
        [py, str(root / "src/exact_feedback/content_contract/endpoint_analysis.py"), "--registry", str(registry / "text_registry.jsonl"), "--trajectories", str(registry / "trajectories.jsonl"), "--stanza", str(stanza), "--nli", str(nli), "--output-dir", str(endpoint), "--bootstrap", "200" if quick else "20000"],
        [py, str(root / "src/exact_feedback/content_contract/annotation_audit.py"), "--registry", str(registry / "text_registry.jsonl"), "--trajectories", str(registry / "trajectories.jsonl"), "--stanza", str(stanza), "--stanza-manifest", str(out / "stanza_manifest.json"), "--nli", str(nli), "--nli-manifest", str(out / "nli_manifest.json"), "--analysis", str(endpoint / "endpoint_annotation_analysis.json"), "--output", str(out / "annotation_audit.json")],
        [py, str(root / "src/exact_feedback/content_contract/sensitivity_analysis.py"), "--input", str(endpoint / "endpoint_trajectory_metrics.csv"), "--output-dir", str(out / "sensitivity")],
    ]


def run_analysis(root: Path, runs: Path, cfg: dict, *, quick: bool, dry_run: bool, include_content_contract: bool) -> None:
    if not dry_run:
        status, failures = check_outputs(runs)
        from .extension_checks import check_extensions
        failures.extend(check_extensions(runs))
        status = 1 if failures else status
        if status:
            preview = "\n".join(f"- {item}" for item in failures[:20])
            raise SystemExit(f"paper-check failed; analysis was not started\n{preview}")
        prepare_compatibility(root, runs)
    commands = analysis_commands(root, runs, quick)
    if include_content_contract:
        commands += content_contract_commands(root, runs, cfg, quick)
    for argv in commands:
        print("+", subprocess.list2cmdline(argv))
        if not dry_run:
            subprocess.run(argv, cwd=root, check=True, env=python_environment(root))
