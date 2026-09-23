#!/usr/bin/env python3
"""Shared deterministic verifier, prompts, and token budgets for full480 domains."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.common.structured_data as benchmark  # noqa: E402


def verify(kind: str, text: str, targets: list[Any]) -> dict[str, Any]:
    """Accept either a canonical structure code or the legacy family label."""
    structure = str(kind)
    if structure not in {"c05", "c07", "c09", "c10", "c12"}:
        matches = [code for code in ("c05", "c07", "c09", "c10", "c12") if code in structure]
        if len(matches) != 1:
            raise KeyError(kind)
        structure = matches[0]
    return benchmark.verify(structure, text, targets)


def display(value: Any) -> str:
    return "<missing>" if value is None else repr(value)


def initial_prompt(item: dict[str, Any]) -> str:
    variant = os.environ.get("MULTIDOMAIN_PROMPT_VARIANT", "baseline")
    if variant == "structured":
        return (
            f"TASK\n{item['task'].strip()}\n\n"
            "CONSTRAINTS\nCHECKING: deterministic verifier\n"
            "OUTPUT: only the requested text; no preface, analysis, count, or commentary."
        )
    if variant == "concise":
        return (
            f"{item['task'].strip()}\n\n"
            "A deterministic verifier checks every constraint. Output only the requested "
            "text, without a preface, analysis, count, or commentary."
        )
    if variant != "baseline":
        raise ValueError(f"unknown MULTIDOMAIN_PROMPT_VARIANT={variant!r}")
    return (
        f"{item['task'].strip()}\n\n"
        "The constraints are checked by a deterministic verifier. Return only "
        "the requested text, with no preface, analysis, count, or commentary."
    )


def feedback_prompt(item: dict[str, Any], state: dict[str, Any]) -> str:
    structure = str(item["structure"])
    if structure == "c05":
        lines = [
            f"- word {position + 1}: target={display(target)}, "
            f"observed={display(observed)}, correct={str(match).lower()}"
            for position, target, observed, match in zip(
                state["target_positions_zero_based"],
                state["target_words"],
                state["observed_words"],
                state["position_matches"],
            )
        ]
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- total words: observed={state['observed_word_count']}, "
            f"target={state['target_word_count']}, "
            f"residual(observed-target)={state['count_residual']}\n"
            + "\n".join(lines)
        )
    elif structure == "c07":
        report = "DETERMINISTIC VERIFIER REPORT\n" + "\n".join(
            f"- required word {display(target)}: present={str(present).lower()}"
            for target, present in zip(state["target_words"], state["target_present"])
        )
    elif structure == "c09":
        lines = [
            f"- forbidden word {display(word)}: observed-count={count}"
            for word, count in zip(state["forbidden_words"], state["forbidden_word_counts"])
        ]
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- sentences: observed={state['observed_sentence_count']}, "
            f"target={state['target_sentence_count']}, "
            f"residual(observed-target)={state['sentence_count_residual']}\n"
            + "\n".join(lines)
        )
    elif structure == "c10":
        lines = [
            f"- sentence {index}: words={count}, below-minimum-by={deficit}, "
            f"above-maximum-by={excess}"
            for index, (count, deficit, excess) in enumerate(
                zip(
                    state["sentence_word_counts"],
                    state["lower_deficits"],
                    state["upper_excesses"],
                ),
                start=1,
            )
        ]
        details = "\n".join(lines) if lines else "- no sentences detected"
        report = (
            "DETERMINISTIC VERIFIER REPORT\n"
            f"- sentences: observed={state['observed_sentence_count']}, "
            f"target={state['target_sentence_count']}, "
            f"residual(observed-target)={state['sentence_count_residual']}\n"
            f"- allowed words per sentence: {state['lower_bound']} to "
            f"{state['upper_bound']} inclusive\n{details}"
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
                f"- sentence {index + 1} final word: target={display(target)}, "
                f"observed={display(observed)}, "
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
    variant = os.environ.get("MULTIDOMAIN_PROMPT_VARIANT", "baseline")
    if variant == "structured":
        suffix = (
            "INSTRUCTION: Revise the complete previous response so every reported "
            "constraint is satisfied. Return only the revised text; no analysis, "
            "count, or commentary."
        )
    elif variant == "concise":
        suffix = (
            "Revise the whole response to satisfy every reported constraint. Output "
            "only the revision, without analysis, a count, or commentary."
        )
    elif variant == "baseline":
        suffix = (
            "Revise the complete previous response so every reported constraint is "
            "satisfied. Return only the revised text, with no analysis, count, or commentary."
        )
    else:
        raise ValueError(f"unknown MULTIDOMAIN_PROMPT_VARIANT={variant!r}")
    return f"{report}\n{suffix}"


def max_tokens(item: dict[str, Any]) -> int:
    structure = str(item["structure"])
    targets = item["targets"]
    if structure == "c05":
        expected_words = int(targets[0])
        return max(128, expected_words * 3)
    if structure == "c10":
        expected_words = int(targets[0]) * int(targets[2])
        return max(128, expected_words * 3)
    witness_words = len(benchmark.base.words(str(item["witness"])))
    return max(128, min(1024, witness_words * 3))
