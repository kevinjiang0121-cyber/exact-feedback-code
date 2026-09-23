#!/usr/bin/env python3
"""Run the full c05/c07/c09/c10/c12 deterministic-feedback benchmark with vLLM."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.common.structured_data as benchmark  # noqa: E402
import exact_feedback.common.structured_protocol as protocol  # noqa: E402
import exact_feedback.content_contract.constraint_runner as legacy  # noqa: E402


legacy_build_row = legacy.build_row
legacy_feedback_prompt = legacy.feedback_prompt
legacy_max_tokens = legacy.max_tokens
legacy_llm = legacy.LLM


class PipelineAwareLLM:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        pipeline_size = int(os.environ.get("MULTIDOMAIN_PIPELINE_PARALLEL_SIZE", "1"))
        if pipeline_size > 1:
            kwargs["pipeline_parallel_size"] = pipeline_size
        return legacy_llm(*args, **kwargs)


def feedback_prompt(item: dict[str, Any], state: dict[str, Any]) -> str:
    structure = str(item["structure"])
    if structure in {"c05", "c10"}:
        return legacy_feedback_prompt(item, state)
    if structure == "c07":
        lines = [
            f"- required word {legacy.display(target)}: present={str(present).lower()}"
            for target, present in zip(state["target_words"], state["target_present"])
        ]
        report = "DETERMINISTIC VERIFIER REPORT\n" + "\n".join(lines)
    elif structure == "c09":
        lines = [
            f"- forbidden word {legacy.display(word)}: observed-count={count}"
            for word, count in zip(state["forbidden_words"], state["forbidden_word_counts"])
        ]
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- sentences: observed={state['observed_sentence_count']}, "
            f"target={state['target_sentence_count']}, "
            f"residual(observed-target)={state['sentence_count_residual']}\n"
            + "\n".join(lines)
        )
    elif structure == "c12":
        lines = []
        for index, target in enumerate(state["target_last_words"]):
            observed = (
                state["observed_last_words"][index]
                if index < len(state["observed_last_words"])
                else None
            )
            lines.append(
                f"- sentence {index + 1} final word: target={legacy.display(target)}, "
                f"observed={legacy.display(observed)}, "
                f"correct={str(state['last_word_matches'][index]).lower()}"
            )
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- sentences: observed={state['observed_sentence_count']}, "
            f"target={state['target_sentence_count']}, "
            f"residual(observed-target)={state['sentence_count_residual']}\n"
            + "\n".join(lines)
        )
    else:
        raise KeyError(structure)
    return (
        f"{report}\n"
        "Revise the complete previous response so every reported constraint is "
        "satisfied. Return only the revised text, with no analysis, count, or commentary."
    )


def max_tokens(item: dict[str, Any]) -> int:
    structure = str(item["structure"])
    if structure in {"c05", "c10"}:
        return legacy_max_tokens(item)
    witness_words = len(benchmark.base.words(str(item["witness"])))
    return max(128, min(1024, witness_words * 3))


def build_row(item: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    row = legacy_build_row(item, rounds)
    row["domain"] = item["domain"]
    row["benchmark_partition"] = item["benchmark_partition"]
    return row


legacy.verify = protocol.verify
legacy.initial_prompt = protocol.initial_prompt
legacy.feedback_prompt = protocol.feedback_prompt
legacy.max_tokens = protocol.max_tokens
legacy.build_row = build_row
legacy.LLM = PipelineAwareLLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def augment_runtime_manifest() -> None:
    output_dir = None
    for index, argument in enumerate(sys.argv[:-1]):
        if argument == "--output-dir":
            output_dir = Path(sys.argv[index + 1])
            break
    if output_dir is None:
        return
    path = output_dir / "runtime_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256(Path(__file__)),
            "shared_protocol": str(Path(protocol.__file__).resolve()),
            "shared_protocol_sha256": sha256(Path(protocol.__file__)),
            "benchmark_verifier": str(Path(benchmark.__file__).resolve()),
            "benchmark_verifier_sha256": sha256(Path(benchmark.__file__)),
            "pipeline_parallel_size": int(
                os.environ.get("MULTIDOMAIN_PIPELINE_PARALLEL_SIZE", "1")
            ),
            "prompt_variant": os.environ.get("MULTIDOMAIN_PROMPT_VARIANT", "baseline"),
            "temperature": float(os.environ.get("MULTIDOMAIN_TEMPERATURE", "0.0")),
            "top_p": float(os.environ.get("MULTIDOMAIN_TOP_P", "1.0")),
            "seed": int(os.environ.get("MULTIDOMAIN_SEED", "0")),
        }
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    legacy.main()
    augment_runtime_manifest()
