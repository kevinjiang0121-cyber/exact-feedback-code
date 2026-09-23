from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from .config import load_config, load_env, model_path
from .registry import EXPERIMENTS
from .planner import command, execute
from .paper import check_outputs, run_analysis


ROOT = Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified paper reproduction interface")
    p.add_argument("--config", type=Path)
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    offline = sub.add_parser("analyze-evidence")
    offline.add_argument("--materials", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--only", nargs="+")
    offline.add_argument("--dry-run", action="store_true")
    offline.add_argument("--jobs", type=int, choices=range(1,5), default=1)
    evidence = sub.add_parser("evidence-check")
    evidence.add_argument("--materials", type=Path, required=True)
    analysis = sub.add_parser("analyze-experiment")
    analysis.add_argument("experiment", choices=["fixed_draft_crossover", "qwen_aligned", "gemma_aligned", "round_budget", "failure_categories", "history_trigger"])
    analysis.add_argument("--runs-root", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--fresh", action="store_true", help="failure_categories: analyze newly generated runs without frozen outcome assertions")
    analysis.add_argument("--stages", nargs="+", choices=["probes","restoration","dose","activation","graft"], default=["probes","restoration","dose","activation","graft"])
    check = sub.add_parser("check")
    check.add_argument("experiment", nargs="?")
    check.add_argument(
        "--model", help="validate only one model in the selected experiment"
    )
    run = sub.add_parser("run")
    run.add_argument("experiment", choices=sorted(EXPERIMENTS))
    run.add_argument("--model")
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", type=int, metavar="N")
    mode.add_argument("--full", action="store_true")
    run.add_argument("--output", type=Path)
    run.add_argument("--dry-run", action="store_true")
    allp = sub.add_parser("run-all")
    all_mode = allp.add_mutually_exclusive_group(required=True)
    all_mode.add_argument("--smoke", type=int)
    all_mode.add_argument("--full", action="store_true")
    allp.add_argument("--dry-run", action="store_true")
    allp.add_argument(
        "--include-api", action="store_true", help="include paid API experiments"
    )
    paper_check = sub.add_parser("paper-check")
    paper_check.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    paper = sub.add_parser("analyze-paper")
    paper.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    paper.add_argument("--quick", action="store_true", help="validate the analysis pipeline with reduced resampling")
    paper.add_argument("--dry-run", action="store_true")
    paper.add_argument("--include-content-contract", action="store_true", help="run the optional Stanza and NLI annotation pipeline")
    return p


def validate(cfg: dict, keys: list[str], selected_model: str | None = None) -> int:
    errors = []
    for key in keys:
        if key not in EXPERIMENTS:
            errors.append(f"Unknown experiment: {key}")
            continue
        exp = EXPERIMENTS[key]
        for rel in exp.data:
            if not (ROOT / rel).is_file():
                errors.append(f"{key}: missing data {rel}")
        if exp.adapter.endswith("api"):
            models = (selected_model,) if selected_model else exp.models
            configured = cfg.get("api_models", {})
            for model in models:
                if model not in exp.models:
                    errors.append(f"{key}: unsupported model {model}")
                elif not (configured.get(model) or {}).get("id"):
                    errors.append(f"{key}: API model is not configured: {model}")
            key_env = cfg.get("api_key_env", "OPENROUTER_API_KEY")
            if not os.environ.get(key_env):
                errors.append(f"{key}: missing API key environment variable {key_env}")
        else:
            models = exp.models if exp.adapter in {"aligned_pair","materialize_restoration","materialize_graft","activation"} else ((selected_model,) if selected_model else exp.models)
            for model in models:
                if model not in exp.models:
                    errors.append(f"{key}: unsupported model {model}")
                    continue
                try:
                    path = model_path(ROOT, cfg, model)
                    if not path.exists():
                        errors.append(f"{key}: model path does not exist: {model}")
                    elif not path.is_dir() or not (path/'config.json').is_file():
                        errors.append(f"{key}: checkpoint config.json is missing: {model}")
                    elif not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
                        errors.append(f"{key}: checkpoint weights are missing: {model}")
                    else:
                        if not (path/'tokenizer_config.json').is_file():
                            errors.append(f"{key}: tokenizer_config.json is missing: {model}")
                        for filename in ['model.safetensors.index.json','pytorch_model.bin.index.json']:
                            index=path/filename
                            if index.is_file():
                                try:shards=set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
                                except (ValueError,KeyError,TypeError):
                                    errors.append(f"{key}: unreadable weight index: {model}")
                                    continue
                                if any(not (path/shard).is_file() for shard in shards):
                                    errors.append(f"{key}: checkpoint has missing weight shards: {model}")
                except SystemExit as exc:
                    errors.append(f"{key}: {exc}")
    if errors:
        print("CHECK_FAIL")
        print("\n".join(f"- {x}" for x in errors))
        return 1
    print(f"CHECK_OK experiments={len(keys)}")
    return 0


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "smoke", None) is not None and args.smoke < 1:
        raise SystemExit("--smoke must be a positive integer")
    if args.action == "list":
        for exp in EXPERIMENTS.values():
            print(f"{exp.key:28} {exp.title}")
        return
    if args.action == "analyze-evidence":
        from .evidence_analysis import EvidenceAnalysis
        results=EvidenceAnalysis(ROOT,args.materials,args.output).run(args.only,args.dry_run,args.jobs)
        print(json.dumps(results,indent=2))
        raise SystemExit(1 if any(v['status'] in {'failed','dependency_failed'} for v in results.values()) else 0)
    if args.action == "evidence-check":
        from .artifacts import ArtifactStore
        report = ArtifactStore(args.materials).verify()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["integrity_pass"] else 1)
    if args.action == "analyze-experiment":
        if args.experiment == "history_trigger":
            from exact_feedback.history.trigger_analysis import TriggerAnalyzer
            print(json.dumps(TriggerAnalyzer().analyze(args.runs_root,args.output),indent=2))
            return
        if args.experiment == "failure_categories":
            from exact_feedback.recurrence.failure_categories import FailureAnalyzer
            report=FailureAnalyzer().analyze(args.runs_root,args.output,frozen=not args.fresh)
            print(json.dumps(report['summary'],indent=2))
            return
        if args.experiment == "round_budget":
            import subprocess,sys
            subprocess.run([sys.executable,str(ROOT/'src/exact_feedback/closed_loop/budget_analysis.py'),
                            '--root',str(args.runs_root),'--output',str(args.output)],check=True)
            return
        if args.experiment in {"qwen_aligned", "gemma_aligned"}:
            from exact_feedback.aligned_pairs.experiment import AlignedPairExperiment
            AlignedPairExperiment(ROOT, {}, args.experiment.split("_")[0], args.output, args.runs_root).run(tuple(args.stages))
            return
        from exact_feedback.crossover.analysis import CrossoverAnalyzer
        if (args.runs_root/'experiments/controller_crossover_combined240_v1').is_dir():
            print(json.dumps(CrossoverAnalyzer().analyze_frozen(args.runs_root,args.output),indent=2))
            return
        report = CrossoverAnalyzer().analyze(
            ROOT / "data/controller_crossover_combined240_v1/generation_combined240.jsonl",
            args.runs_root, args.output)
        print(json.dumps(report, indent=2))
        return
    if args.action == "paper-check":
        runs = args.runs_root if args.runs_root.is_absolute() else ROOT / args.runs_root
        status, failures = check_outputs(runs)
        from .extension_checks import check_extensions
        failures.extend(check_extensions(runs))
        status = 1 if failures else status
        if failures:
            print("\n".join(f"- {item}" for item in failures))
        raise SystemExit(status)
    load_env(args.env_file)
    cfg = load_config(ROOT, args.config)
    if args.action == "analyze-paper":
        runs = args.runs_root if args.runs_root.is_absolute() else ROOT / args.runs_root
        run_analysis(ROOT, runs, cfg, quick=args.quick, dry_run=args.dry_run, include_content_contract=args.include_content_contract)
        return
    if args.action == "check":
        if args.model and not args.experiment:
            raise SystemExit("--model requires an experiment name")
        keys = [args.experiment] if args.experiment else list(EXPERIMENTS)
        raise SystemExit(validate(cfg, keys, args.model))
    if args.action == "run":
        exp = EXPERIMENTS[args.experiment]
        if args.smoke is not None and exp.key in {'qwen_aligned','gemma_aligned','residual_probe_capture','parameter_restoration','activation_transplant','controller_graft'}:
            raise SystemExit('This fixed-state pipeline requires --full; arbitrary state truncation would invalidate its analysis contract.')
        paired = exp.adapter in {
            "aligned_pair",
            "materialize_restoration",
            "materialize_graft",
            "activation",
        }
        model = args.model or (exp.models[0] if paired else None)
        if model is None:
            raise SystemExit(
                f"--model is required; choose one of: {', '.join(exp.models)}"
            )
        if model not in exp.models:
            raise SystemExit(
                f"{model} is not in {args.experiment}: {', '.join(exp.models)}"
            )
        if not args.dry_run and validate(cfg,[args.experiment],model):
            raise SystemExit(1)
        output_key = "paired" if paired else model
        out = args.output or ROOT / "runs" / args.experiment / output_key
        out = out if out.is_absolute() else ROOT / out
        execute(command(ROOT, cfg, exp, model, args.smoke, out), ROOT, args.dry_run)
        return
    if args.action == "run-all":
        if not args.full:
            raise SystemExit('run-all requires --full because aligned fixed-state pipelines have no validated arbitrary-subset mode. Use run <experiment> --smoke N for individual behavioral execution checks.')
        smoke = None if args.full else args.smoke
        keys=[e.key for e in EXPERIMENTS.values() if args.include_api or not e.adapter.endswith('api')]
        if not args.dry_run and validate(cfg,keys):raise SystemExit(1)
        for exp in EXPERIMENTS.values():
            if exp.adapter.endswith("api") and not args.include_api:
                print(f"SKIP_PAID_API {exp.key} (pass --include-api to enable)")
                continue
            models = (
                exp.models[:1]
                if exp.adapter
                in {"materialize_restoration", "materialize_graft", "activation", "aligned_pair"}
                else exp.models
            )
            for model in models:
                output_key = (
                    "paired"
                    if exp.adapter
                    in {"materialize_restoration", "materialize_graft", "activation", "aligned_pair"}
                    else model
                )
                out = ROOT / "runs" / exp.key / output_key
                execute(command(ROOT, cfg, exp, model, smoke, out), ROOT, args.dry_run)
