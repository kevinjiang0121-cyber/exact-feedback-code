from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .config import model_path, python_environment
from .registry import HISTORICAL_EXACT_SIX


RESTORATION_CONDITIONS = (
    "sham_copy",
    "restore_noncontiguous",
    "restore_blocks_00_07",
    "restore_blocks_08_15",
    "restore_blocks_16_23",
    "restore_blocks_24_31",
    "restore_output",
)
DOSE_CONDITIONS = ("dose50_blocks_08_15", "dose50_output")
CONFIRMATION_CONDITIONS = (
    "sham_copy",
    "restore_noncontiguous",
    "restore_blocks_08_15",
    "restore_output",
)
GRAFT_CONDITIONS = (
    "base_meta_contract_sham",
    "graft_blocks_08_15",
    "graft_output",
    "graft_noncontiguous_plus_output",
    "graft_blocks_08_15_plus_output",
)
ANALYSIS_MODEL_SLUG = {
    "llama31_8b": "llama31_8b_instruct",
    "gemma2_9b": "gemma2_9b_it",
}


def portable_path(path: Path, root: Path) -> str:
    """Keep package paths relative while allowing an explicit external output."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset_jsonl(
    source: Path,
    target: Path,
    limit: int,
    field: str | None = None,
    value: str | None = None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                if field is not None and json.loads(line).get(field) != value:
                    continue
                kept.append(line.rstrip("\n"))
            if len(kept) >= limit:
                break
    if len(kept) < limit:
        raise SystemExit(f"Requested {limit} rows but {source} has {len(kept)}")
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return target


def runtime_args(cfg: dict, model: str, *, pipeline: bool = False) -> list[str]:
    """Translate portable model settings into runner CLI flags."""
    spec = cfg.get("models", {}).get(model, {})
    args = []
    mapping = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_model_len": "--max-model-len",
        "max_num_seqs": "--max-num-seqs",
    }
    if pipeline:
        mapping["pipeline_parallel_size"] = "--pipeline-parallel-size"
    for key, flag in mapping.items():
        if key in spec:
            args += [flag, str(spec[key])]
    return args


def fixed_and_closed_commands(
    root: Path,
    py: str,
    checkpoint: Path,
    slug: str,
    states: Path,
    cases: Path,
    destination: Path,
    chat: Path,
    dtype: str,
    limit: int,
) -> list[list[str]]:
    closed = destination / "closed_loop"
    return [
        [
            py,
            str(root / "src/exact_feedback/localization/probe_runner.py"),
            "--model",
            str(checkpoint),
            "--model-slug",
            slug,
            "--states",
            str(states),
            "--chat-template",
            str(chat),
            "--output-dir",
            str(destination / "fixed_state"),
            "--split",
            "discovery",
            "--dtype",
            dtype,
        ],
        [
            py,
            str(root / "src/exact_feedback/closed_loop/open_runner.py"),
            "--model",
            str(checkpoint),
            "--data",
            str(cases),
            "--output-dir",
            str(closed),
            "--dtype",
            dtype,
            "--limit",
            str(limit),
        ],
        [
            py,
            str(root / "src/exact_feedback/closed_loop/audit.py"),
            "--data",
            str(cases),
            "--result-dir",
            str(closed),
        ],
    ]


from dataclasses import dataclass
from .experiment import ExperimentImplementation, Command

@dataclass
class ConfiguredExperiment(ExperimentImplementation):
    root: Path
    cfg: dict
    exp: object
    model: str
    smoke: int | None
    output: Path

    def plan(self):
        return [Command(tuple(argv)) for argv in self.commands()]

class CrossoverExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        from exact_feedback.crossover.experiment import CrossoverExperiment
        return [list(c.argv) for c in CrossoverExperiment(root, cfg, model, output, smoke).plan()]


class AlignedPairExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        if smoke is not None:
            raise SystemExit("Aligned pipelines currently expose full mode only; reduced-state integrity rules are not yet validated.")
        return [[py, str(root / "src/exact_feedback/aligned_pairs/pipeline.py"),
                 "--family", exp.key.split("_")[0], "--config", cfg["_path"],
                 "--output", str(output)]]


class ExactLocalExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        if model in HISTORICAL_EXACT_SIX:
            if smoke:
                destination = output / "smoke"
                return [
                    [
                        py,
                        str(root / "src/exact_feedback/common/exact_protocol.py"),
                        "--model",
                        str(mpath),
                        "--data",
                        str(inputs[1]),
                        "--output-dir",
                        str(destination),
                        "--dtype",
                        dtype,
                        "--limit",
                        str(smoke),
                    ],
                    [
                        py,
                        str(root / "src/exact_feedback/closed_loop/audit.py"),
                        "--data",
                        str(inputs[1]),
                        "--result-dir",
                        str(destination),
                        "--expected-rows",
                        str(smoke),
                    ],
                ]
            commands = []
            for split, data in (
                ("primary120", inputs[1]),
                ("replication120", inputs[2]),
            ):
                destination = output / split
                commands += [
                    [
                        py,
                        str(root / "src/exact_feedback/common/exact_protocol.py"),
                        "--model",
                        str(mpath),
                        "--data",
                        str(data),
                        "--output-dir",
                        str(destination),
                        "--dtype",
                        dtype,
                    ],
                    [
                        py,
                        str(root / "src/exact_feedback/closed_loop/audit.py"),
                        "--data",
                        str(data),
                        "--result-dir",
                        str(destination),
                    ],
                ]
            if model == "qwen3_14b":
                shard_dirs = []
                for index, data in enumerate(inputs[4:6]):
                    destination = output / "extension240_shards" / f"shard_{index:02d}"
                    shard_dirs.append(destination)
                    commands += [
                        [
                            py,
                            str(root / "src/exact_feedback/closed_loop/batched_runner.py"),
                            "--model",
                            str(mpath),
                            "--data",
                            str(data),
                            "--output-dir",
                            str(destination),
                            "--batch-size",
                            "1",
                            "--dtype",
                            dtype,
                            "--resume",
                        ],
                        [
                            py,
                            str(root / "src/exact_feedback/closed_loop/audit.py"),
                            "--data",
                            str(data),
                            "--result-dir",
                            str(destination),
                        ],
                    ]
                commands.append(
                    [
                        py,
                        str(root / "src/exact_feedback/closed_loop/merge_shards.py"),
                        "--data",
                        str(inputs[3]),
                        "--output-dir",
                        str(output / "extension240"),
                        "--shards",
                        *map(str, shard_dirs),
                    ]
                )
                commands.append(
                    [
                        py,
                        str(root / "src/exact_feedback/closed_loop/audit.py"),
                        "--data",
                        str(inputs[3]),
                        "--result-dir",
                        str(output / "extension240"),
                    ]
                )
            else:
                destination = output / "extension240"
                commands += [
                    [
                        py,
                        str(root / "src/exact_feedback/closed_loop/batched_runner.py"),
                        "--model",
                        str(mpath),
                        "--data",
                        str(inputs[3]),
                        "--output-dir",
                        str(destination),
                        "--batch-size",
                        "1",
                        "--dtype",
                        dtype,
                        "--resume",
                    ],
                    [
                        py,
                        str(root / "src/exact_feedback/closed_loop/audit.py"),
                        "--data",
                        str(inputs[3]),
                        "--result-dir",
                        str(destination),
                    ],
                ]
            return commands
        runner = (
            root / "src/exact_feedback/common/exact_protocol.py"
            if model == "falcon_h1_7b"
            else root / "src/exact_feedback/closed_loop/open_runner.py"
        )
        limit = smoke or 480
        return [
            [
                py,
                str(runner),
                "--model",
                str(mpath),
                "--data",
                str(inputs[0]),
                "--output-dir",
                str(output),
                "--dtype",
                dtype,
                "--limit",
                str(limit),
                *(
                    []
                    if model == "falcon_h1_7b"
                    else runtime_args(cfg, model, pipeline=True)
                ),
            ],
            [
                py,
                str(root / "src/exact_feedback/closed_loop/audit.py"),
                "--data",
                str(inputs[0]),
                "--result-dir",
                str(output),
                "--expected-rows",
                str(limit),
            ],
        ]


class StructuredLocalExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        data = [limited(p, p.name) for p in inputs]
        runner = (
            root / "src/exact_feedback/content_contract/transformers_runner.py"
            if model == "falcon_h1_7b"
            else root / "src/exact_feedback/content_contract/open_runner.py"
        )
        expected_rows = sum(
            sum(1 for line in path.open(encoding="utf-8") if line.strip())
            for path in data
        )
        return [
            [
                py,
                str(runner),
                "--model",
                str(mpath),
                "--data",
                *map(str, data),
                "--output-dir",
                str(output),
                "--dtype",
                dtype,
                *([] if model == "falcon_h1_7b" else runtime_args(cfg, model)),
            ],
            [
                py,
                str(root / "src/exact_feedback/content_contract/audit.py"),
                "--data",
                *map(str, data),
                "--cases",
                str(output / "cases.jsonl"),
                "--output",
                str(output / "audit.json"),
                "--expected-rows",
                str(expected_rows),
            ],
        ]


class ResponseSurfaceExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        model_slug = ANALYSIS_MODEL_SLUG.get(model, model)
        splits = ("all",) if smoke else ("discovery", "confirmation")
        commands = []
        for split in splits:
            destination = output if smoke else output / split
            run = [
                py,
                str(root / "src/exact_feedback/response_law/runner.py"),
                "--model",
                str(mpath),
                "--model-slug",
                model_slug,
                "--states",
                str(inputs[0]),
                "--output-dir",
                str(destination),
                "--split",
                split,
                "--dtype",
                dtype,
                *runtime_args(cfg, model, pipeline=True),
            ]
            if model in {"llama31_8b_base", "llama31_8b"}:
                run += ["--chat-template", str(root / "configs/llama_meta_chat_template.jinja")]
            elif model == "qwen3_8b_base":
                run += ["--chat-template", str(root / "configs/qwen3_base_response_chat_template.jinja")]
            audit = [
                py,
                str(root / "src/exact_feedback/response_law/audit.py"),
                "--states",
                str(inputs[0]),
                "--cases",
                str(destination / "cases.jsonl"),
                "--runtime-manifest",
                str(destination / "runtime_manifest.json"),
                "--split",
                split,
                "--output",
                str(destination / "integrity_audit.json"),
            ]
            if smoke:
                run += ["--case-limit", str(smoke)]
                audit += ["--case-limit", str(smoke)]
            commands += [run, audit]
        return commands


class PromptRobustnessExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        cases = limited(inputs[0], "cases.jsonl")
        return [
            [
                py,
                str(root / "src/exact_feedback/robustness/prompt_runner.py"),
                "--model",
                str(mpath),
                "--data",
                str(cases),
                "--templates",
                str(inputs[1]),
                "--template",
                template,
                "--output-dir",
                str(output / template),
                "--dtype",
                dtype,
            ]
            for template in ("baseline", "structured", "concise")
        ]


class DecodingRobustnessExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        cases = limited(inputs[0], "cases.jsonl")
        return [
            [
                py,
                str(root / "src/exact_feedback/robustness/decoding_runner.py"),
                "--model",
                str(mpath),
                "--data",
                str(cases),
                "--output-dir",
                str(output / f"seed_{seed}"),
                "--seed",
                str(seed),
                "--dtype",
                dtype,
            ]
            for seed in (20260725, 20260726, 20260727)
        ]


class PersistenceExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        selection = root / f"data/persistence_revision32_v1/{model}.jsonl"
        selection = limited(selection, selection.name)
        return [
            [
                py,
                str(root / "src/exact_feedback/persistence/runner.py"),
                "--model",
                str(mpath),
                "--model-slug",
                model,
                "--selection",
                str(selection),
                "--output-dir",
                str(output),
                "--max-revision",
                "32",
                "--dtype",
                dtype,
            ]
        ]


class HistoryExactExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        states = (
            subset_jsonl(inputs[0], scratch / "states.jsonl", smoke, "model", model)
            if smoke
            else inputs[0]
        )
        argv = [
            py,
            str(root / "src/exact_feedback/history/exact_runner.py"),
            "--model",
            str(mpath),
            "--model-slug",
            model,
            "--states",
            str(states),
            "--output-dir",
            str(output),
            "--chat-template",
            str(chat),
            "--dtype",
            dtype,
        ]
        if smoke:
            argv += ["--expected-states", str(smoke)]
        return [argv]


class HistoryStructuredExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        states = (
            subset_jsonl(inputs[0], scratch / "states.jsonl", smoke, "model", model)
            if smoke
            else inputs[0]
        )
        runner = (
            root / "src/exact_feedback/history/structured_transformers_runner.py"
            if model == "falcon_h1_7b"
            else root / "src/exact_feedback/history/structured_runner.py"
        )
        return [
            [
                py,
                str(runner),
                "--model",
                str(mpath),
                "--model-slug",
                model,
                "--states",
                str(states),
                "--output-dir",
                str(output),
                *(
                    ["--dtype", dtype]
                    if model == "falcon_h1_7b"
                    else runtime_args(cfg, model)
                ),
            ]
        ]


class HistoryPartialExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        states = limited(inputs[0], "states.jsonl")
        argv = [
            py,
            str(root / "src/exact_feedback/history/partial_runner.py"),
            "--model",
            str(mpath),
            "--states",
            str(states),
            "--output-dir",
            str(output),
        ]
        if smoke:
            argv.append("--allow-subset")
        return [argv]


class StateCaptureExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/llama_meta_chat_template.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        states = limited(inputs[0], "states.jsonl")
        return [
            [
                py,
                str(root / "src/exact_feedback/localization/capture.py"),
                "--model",
                str(mpath),
                "--model-slug",
                model,
                "--states",
                str(states),
                "--chat-template",
                str(chat),
                "--output-dir",
                str(output),
                "--dtype",
                dtype,
            ]
        ]


class MaterializeRestorationExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/llama_meta_chat_template.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        base = model_path(root, cfg, "llama31_8b_base")
        instruct = model_path(root, cfg, "llama31_8b")
        states = limited(inputs[0], "states.jsonl")
        main = limited(inputs[1], "main120.jsonl")
        combined = limited(inputs[2], "combined480.jsonl")
        conditions = (
            RESTORATION_CONDITIONS
            if not smoke
            else ("sham_copy", "restore_blocks_08_15")
        )
        checkpoints = output / "checkpoints" / "discovery"
        commands = [
            [
                py,
                str(root / "src/exact_feedback/localization/restoration.py"),
                "--base",
                str(base),
                "--instruct",
                str(instruct),
                "--output-root",
                str(checkpoints),
                "--conditions",
                *conditions,
            ]
        ]
        for condition in conditions:
            commands += fixed_and_closed_commands(
                root,
                py,
                checkpoints / condition,
                condition,
                states,
                main,
                output / "discovery" / condition,
                chat,
                dtype,
                smoke or 120,
            )
        if smoke:
            return commands
        dose_checkpoints = output / "checkpoints" / "dose"
        commands.append(
            [
                py,
                str(root / "src/exact_feedback/localization/restoration_dose.py"),
                "--base",
                str(base),
                "--instruct",
                str(instruct),
                "--output-root",
                str(dose_checkpoints),
            ]
        )
        for condition in DOSE_CONDITIONS:
            commands += fixed_and_closed_commands(
                root,
                py,
                dose_checkpoints / condition,
                condition,
                states,
                main,
                output / "dose" / condition,
                chat,
                dtype,
                120,
            )
        for condition in CONFIRMATION_CONDITIONS:
            destination = output / "confirmation" / condition / "combined480"
            commands.append(
                [
                    py,
                    str(root / "src/exact_feedback/closed_loop/open_runner.py"),
                    "--model",
                    str(checkpoints / condition),
                    "--data",
                    str(combined),
                    "--output-dir",
                    str(destination),
                    "--dtype",
                    dtype,
                    "--limit",
                    "480",
                ]
            )
            commands.append(
                [
                    py,
                    str(root / "src/exact_feedback/closed_loop/audit.py"),
                    "--data",
                    str(combined),
                    "--result-dir",
                    str(destination),
                ]
            )
        commands += [
            [
                py,
                str(root / "src/exact_feedback/localization/restoration_analysis.py"),
                "--root",
                str(output / "discovery"),
                "--output-dir",
                str(output / "analysis"),
            ],
            [
                py,
                str(root / "src/exact_feedback/localization/restoration_dose_analysis.py"),
                "--discovery-root",
                str(output / "discovery"),
                "--root",
                str(output),
                "--output",
                str(output / "analysis" / "dose_confirmation.json"),
            ],
        ]
        return commands


class MaterializeGraftExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/llama_meta_chat_template.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        base = model_path(root, cfg, "llama31_8b_base")
        instruct = model_path(root, cfg, "llama31_8b")
        states = limited(inputs[0], "states.jsonl")
        main = limited(inputs[1], "main120.jsonl")
        conditions = (
            GRAFT_CONDITIONS
            if not smoke
            else ("base_meta_contract_sham", "graft_blocks_08_15_plus_output")
        )
        checkpoints = output / "checkpoints"
        commands = [
            [
                py,
                str(root / "src/exact_feedback/localization/graft.py"),
                "--base",
                str(base),
                "--instruct",
                str(instruct),
                "--output-root",
                str(checkpoints),
                "--conditions",
                *conditions,
            ]
        ]
        for condition in conditions:
            commands += fixed_and_closed_commands(
                root,
                py,
                checkpoints / condition,
                condition,
                states,
                main,
                output / condition,
                chat,
                dtype,
                smoke or 120,
            )
        if not smoke:
            commands.append(
                [
                    py,
                    str(root / "src/exact_feedback/localization/graft_analysis.py"),
                    "--root",
                    str(output),
                    "--output",
                    str(output / "analysis.json"),
                ]
            )
        return commands


class ActivationExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/llama_meta_chat_template.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        states = limited(inputs[0], "states.jsonl")
        base = model_path(root, cfg, "llama31_8b_base")
        instruct = model_path(root, cfg, "llama31_8b")
        commands = [
            [
                py,
                str(root / "src/exact_feedback/localization/probe_runner.py"),
                "--model",
                str(base),
                "--model-slug",
                "llama31_8b_base",
                "--states",
                str(states),
                "--chat-template",
                str(chat),
                "--output-dir",
                str(output / "base_clean"),
                "--split",
                "discovery",
                "--dtype",
                dtype,
            ],
            [
                py,
                str(root / "src/exact_feedback/localization/probe_runner.py"),
                "--model",
                str(instruct),
                "--model-slug",
                "llama31_8b",
                "--states",
                str(states),
                "--chat-template",
                str(chat),
                "--output-dir",
                str(output / "instruct_clean"),
                "--split",
                "discovery",
                "--dtype",
                dtype,
            ],
            [
                py,
                str(root / "src/exact_feedback/localization/activation_runner.py"),
                "--base",
                str(model_path(root, cfg, "llama31_8b_base")),
                "--instruct",
                str(model_path(root, cfg, "llama31_8b")),
                "--states",
                str(states),
                "--chat-template",
                str(chat),
                "--output",
                str(output / "activation.jsonl"),
                "--split",
                "discovery",
                "--case-limit",
                str(smoke or 24),
                "--validate-sham",
            ],
        ]
        if not smoke:
            commands.append(
                [
                    py,
                    str(root / "src/exact_feedback/localization/activation_analysis.py"),
                    "--base-clean",
                    str(output / "base_clean" / "cases.jsonl"),
                    "--instruct-clean",
                    str(output / "instruct_clean" / "cases.jsonl"),
                    "--patched",
                    str(output / "activation.jsonl"),
                    "--output",
                    str(output / "analysis.json"),
                ]
            )
        return commands


class TuluCommonExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        commands = []
        template_args = (
            ["--chat-template-file", str(chat)] if exp.adapter == "tulu_common" else []
        )
        exact = (
            root / "data/human_generation_combined480_v1/generation_combined480.jsonl"
        )
        structured = [
            root / p
            for p in (
                "data/multidomain_full480_v1/lexical_constraints_combined480.jsonl",
                "data/multidomain_full480_v1/compositional_constraints_combined480.jsonl",
            )
        ]
        if smoke:
            structured = [limited(p, p.name) for p in structured]
        commands.append(
            [
                py,
                str(root / "src/exact_feedback/closed_loop/open_runner.py"),
                "--model",
                str(mpath),
                "--data",
                str(exact),
                "--output-dir",
                str(output / "exact"),
                "--dtype",
                dtype,
                "--limit",
                str(smoke or 480),
                *runtime_args(cfg, model, pipeline=True),
                *template_args,
            ]
        )
        commands.append(
            [
                py,
                str(root / "src/exact_feedback/closed_loop/audit.py"),
                "--data",
                str(exact),
                "--result-dir",
                str(output / "exact"),
                "--expected-rows",
                str(smoke or 480),
            ]
        )
        commands.append(
            [
                py,
                str(root / "src/exact_feedback/content_contract/open_runner.py"),
                "--model",
                str(mpath),
                "--data",
                *map(str, structured),
                "--output-dir",
                str(output / "structured"),
                "--dtype",
                dtype,
                *runtime_args(cfg, model),
                *template_args,
            ]
        )
        expected_rows = sum(
            sum(1 for line in path.open(encoding="utf-8") if line.strip())
            for path in structured
        )
        commands.append(
            [
                py,
                str(root / "src/exact_feedback/content_contract/audit.py"),
                "--data",
                *map(str, structured),
                "--cases",
                str(output / "structured" / "cases.jsonl"),
                "--output",
                str(output / "structured" / "audit.json"),
                "--expected-rows",
                str(expected_rows),
            ]
        )
        return commands


class ExactApiExperiment(ConfiguredExperiment):
    def commands(self):
        root, cfg, exp, model, smoke, output = self.root, self.cfg, self.exp, self.model, self.smoke, self.output
        py = cfg.get("python", sys.executable)
        mpath = model_path(root, cfg, model) if not exp.adapter.endswith("api") else None
        common = cfg.get("runtime", {})
        dtype = common.get("dtype", "bfloat16")
        chat = root / "configs/common_plain_chat_template_v1.jinja"
        inputs = [root / p for p in exp.data]
        scratch = root / "runs" / "_inputs" / exp.key / model

        def limited(path: Path, suffix: str) -> Path:
            return subset_jsonl(path, scratch / suffix, smoke) if smoke else path

        return api_commands(root, cfg, exp, model, smoke, output)


IMPLEMENTATIONS = {
    'crossover': CrossoverExperiment,
    'aligned_pair': AlignedPairExperiment,
    'exact_local': ExactLocalExperiment,
    'structured_local': StructuredLocalExperiment,
    'response_surface': ResponseSurfaceExperiment,
    'prompt_robustness': PromptRobustnessExperiment,
    'decoding_robustness': DecodingRobustnessExperiment,
    'persistence': PersistenceExperiment,
    'history_exact': HistoryExactExperiment,
    'history_structured': HistoryStructuredExperiment,
    'history_partial': HistoryPartialExperiment,
    'state_capture': StateCaptureExperiment,
    'materialize_restoration': MaterializeRestorationExperiment,
    'materialize_graft': MaterializeGraftExperiment,
    'activation': ActivationExperiment,
    'tulu_common': TuluCommonExperiment,
    'tulu_native': TuluCommonExperiment,
    'exact_api': ExactApiExperiment,
    'structured_api': ExactApiExperiment,
}

def experiment(root, cfg, exp, model, smoke, output):
    return IMPLEMENTATIONS[exp.adapter](root, cfg, exp, model, smoke, output)

def command(root, cfg, exp, model, smoke, output):
    return [list(step.argv) for step in experiment(root, cfg, exp, model, smoke, output).plan()]


def api_commands(
    root: Path, cfg: dict, exp, model: str, smoke: int | None, output: Path
) -> list[list[str]]:
    spec = cfg.get("api_models", {}).get(model)
    if not spec:
        raise SystemExit(f"API model is not configured: {model}")
    py = cfg.get("python", sys.executable)
    sources = [root / p for p in exp.data]
    if smoke:
        sources = [subset_jsonl(p, output / "_inputs" / p.name, smoke) for p in sources]
    commands = []
    for source in sources:
        domain = (
            "exact_length"
            if exp.adapter == "exact_api"
            else (
                "lexical_constraints"
                if "lexical" in source.name
                else "compositional_constraints"
            )
        )
        cfg_path = output / f"{domain}.config.json"
        domain_output = output if domain == "exact_length" else output / domain
        runner_cfg = {
            "schema_version": 1,
            "status": "reviewer_configured",
            "execution_enabled": True,
            "experiment_slug": f"{exp.key}_{model}_{domain}",
            "api": {
                "base_url": cfg.get("api_base_url", "https://openrouter.ai/api/v1"),
                "token_env": cfg.get("api_key_env", "OPENROUTER_API_KEY"),
                "transport": "requests",
                "timeout_seconds": 180,
                "max_attempts": 5,
            },
            "catalog_policy": {"require_text_input_and_output": True},
            "budget": {"max_total_usd": float(cfg.get("api_budget_usd", 1000))},
            "data": {
                "cases": portable_path(source, root),
                "expected_cases": sum(
                    1 for x in source.open(encoding="utf-8") if x.strip()
                ),
                "expected_cases_sha256": sha256(source),
            },
            "protocol": {
                "max_revisions": 8,
                "prompt_variant": "baseline",
                "provider": {
                    "order": spec.get("providers", []),
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            },
            "models": [
                {
                    "id": spec["id"],
                    "label": model,
                    "enabled": True,
                    "price_ceiling_usd_per_1m": spec.get(
                        "price_ceiling_usd_per_1m", {"prompt": 1000, "completion": 1000}
                    ),
                    "request_parameters": spec.get("request_parameters", {}),
                }
            ],
            "output_dir": portable_path(domain_output, root),
        }
        if domain == "exact_length":
            runner_cfg["data"]["templates"] = (
                "data/prompt_robustness60_v1/templates.json"
            )
            runner_cfg["data"]["template"] = "baseline"
        if domain != "exact_length":
            runner_cfg["data"]["domain"] = domain
            runner_cfg["route_gate"] = {
                "expected_provider": spec.get("provider", ""),
                "allowed_returned_models": [spec["id"]],
            }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(runner_cfg, indent=2) + "\n", encoding="utf-8")
        runner = (
            root / "src/exact_feedback/closed_loop/api_runner.py"
            if domain == "exact_length"
            else root / "src/exact_feedback/content_contract/api_runner.py"
        )
        commands.append(
            [py, str(runner), "--config", str(cfg_path), "--execute"]
        )
        result_dir = domain_output / model / "baseline"
        if domain == "exact_length":
            commands.append(
                [
                    py,
                    str(root / "src/exact_feedback/closed_loop/audit.py"),
                    "--data",
                    str(source),
                    "--result-dir",
                    str(result_dir),
                    "--expected-rows",
                    str(runner_cfg["data"]["expected_cases"]),
                ]
            )
        else:
            commands.append(
                [
                    py,
                    str(root / "src/exact_feedback/content_contract/audit.py"),
                    "--data",
                    str(source),
                    "--cases",
                    str(result_dir / "cases.jsonl"),
                    "--output",
                    str(result_dir / "integrity_audit.json"),
                    "--expected-rows",
                    str(runner_cfg["data"]["expected_cases"]),
                ]
            )
    return commands


def execute(commands: list[list[str]], root: Path, dry_run: bool) -> None:
    for argv in commands:
        print("+", subprocess.list2cmdline(argv))
        if dry_run:
            continue
        subprocess.run(argv, cwd=root, check=True, env=python_environment(root))
