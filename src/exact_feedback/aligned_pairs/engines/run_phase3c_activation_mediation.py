#!/usr/bin/env python3
"""Cross-checkpoint feedback-span activation patching for frozen Phase3C C1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_feedback.common.exact_protocol import contains_literal, word_count


UNITS = {
    "blocks_08_15": tuple(range(8, 16)),
    "output_boundary": (31,),
    "blocks_00_08": tuple(range(0, 9)),
    "blocks_09_17": tuple(range(9, 18)),
    "blocks_18_26": tuple(range(18, 27)),
    "blocks_27_35": tuple(range(27, 36)),
    "output_boundary_qwen": (35,),
    "blocks_00_09": tuple(range(0, 10)),
    "blocks_10_20": tuple(range(10, 21)),
    "blocks_21_30": tuple(range(21, 31)),
    "blocks_31_41": tuple(range(31, 42)),
    "output_boundary_gemma": (41,),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("empty feedback token sequence")
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    raise RuntimeError("feedback content token span not found in rendered prompt")


def replace_hidden(
    output: Any,
    donor: torch.Tensor,
    start: int,
    end: int,
    prompt_length: int,
) -> Any:
    hidden = output[0] if isinstance(output, tuple) else output
    if hidden.shape[1] != prompt_length:
        return output
    patched = hidden.clone()
    patched[:, start:end, :] = donor.to(
        device=patched.device, dtype=patched.dtype
    )
    if isinstance(output, tuple):
        return (patched,) + output[1:]
    return patched


def capture_spans(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: tuple[int, ...],
    start: int,
    end: int,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    with ExitStack() as stack:
        for layer in layers:
            def hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[:, start:end, :].detach().clone()

            stack.callback(model.model.layers[layer].register_forward_hook(hook).remove)
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
    if set(captured) != set(layers):
        raise RuntimeError(f"failed to capture layers {layers}")
    return captured


def generate_clean(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, int, str | None]:
    prompt_length = input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    new_ids = generated.sequences[0, prompt_length:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    finish = (
        "eos"
        if len(new_ids) > 0 and int(new_ids[-1]) == tokenizer.eos_token_id
        else "length"
    )
    return text, int(len(new_ids)), finish


def generate_patched(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: tuple[int, ...],
    donor: dict[int, torch.Tensor],
    start: int,
    end: int,
    max_new_tokens: int,
) -> tuple[str, int, str | None]:
    prompt_length = input_ids.shape[1]
    with ExitStack() as stack:
        for layer in layers:
            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                layer: int = layer,
            ) -> Any:
                return replace_hidden(
                    output,
                    donor[layer],
                    start,
                    end,
                    prompt_length,
                )

            stack.callback(model.model.layers[layer].register_forward_hook(hook).remove)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
    new_ids = generated.sequences[0, prompt_length:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    finish = (
        "eos"
        if len(new_ids) > 0 and int(new_ids[-1]) == tokenizer.eos_token_id
        else "length"
    )
    return text, int(len(new_ids)), finish


def render_state(
    tokenizer: Any, template: str, state: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
    tokenizer.chat_template = template
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            state["messages"], enable_thinking=False, **kwargs
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(state["messages"], **kwargs)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    feedback = state["messages"][-1]["content"].strip()
    feedback_ids = tokenizer.encode(feedback, add_special_tokens=False)
    sequence = encoded["input_ids"][0].tolist()
    span = find_last_subsequence(sequence, feedback_ids)
    return encoded["input_ids"], encoded["attention_mask"], span, prompt


def row_from_text(
    state: dict[str, Any],
    unit: str,
    direction: str,
    recipient: str,
    donor: str,
    span: tuple[int, int],
    prompt: str,
    text: str,
    generated_tokens: int,
    finish_reason: str | None,
) -> dict[str, Any]:
    output_count = word_count(text)
    realized_delta = output_count - int(state["current_word_count"])
    required_delta = int(state["required_delta"])
    missing = [
        anchor
        for anchor in state["anchors"]
        if not contains_literal(text, anchor)
    ]
    exact = output_count == int(state["target"])
    return {
        key: state[key]
        for key in (
            "state_id",
            "case_id",
            "source",
            "split",
            "state_family",
            "current_word_count",
            "target",
            "signed_error",
            "required_delta",
        )
    } | {
        "unit": unit,
        "direction": direction,
        "recipient": recipient,
        "donor": donor,
        "feedback_span_start": span[0],
        "feedback_span_end": span[1],
        "feedback_span_tokens": span[1] - span[0],
        "rendered_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "text": text,
        "output_word_count": output_count,
        "output_error": output_count - int(state["target"]),
        "realized_delta": realized_delta,
        "action_gain": realized_delta / required_delta,
        "direction_correct": realized_delta * required_delta > 0,
        "exact_length": exact,
        "missing_anchors": missing,
        "joint_success": exact and not missing,
        "generated_token_count": generated_tokens,
        "finish_reason": finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--instruct", required=True)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--units", default="blocks_08_15,output_boundary")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--validate-sham", action="store_true")
    args = parser.parse_args()

    units = [value.strip() for value in args.units.split(",") if value.strip()]
    if not units or any(unit not in UNITS for unit in units):
        raise ValueError(f"invalid units: {units}")
    states = [
        row for row in read_jsonl(args.states) if row["split"] == args.split
    ]
    case_ids = sorted({row["case_id"] for row in states})
    if args.case_limit is not None:
        case_ids = case_ids[: args.case_limit]
        states = [row for row in states if row["case_id"] in set(case_ids)]
    expected = 4 * len(case_ids)
    if len(states) != expected or len({row["state_id"] for row in states}) != expected:
        raise RuntimeError("fixed-state case coverage failure")

    tokenizer = AutoTokenizer.from_pretrained(args.instruct, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.get_vocab() != base_tokenizer.get_vocab():
        raise RuntimeError("Base/Instruct tokenizer vocabularies differ")
    template = args.chat_template.read_text(encoding="utf-8")
    rendered = {
        state["state_id"]: render_state(tokenizer, template, state)
        for state in states
    }
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    load_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    instruct = AutoModelForCausalLM.from_pretrained(
        args.instruct,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    models = {"base": base, "instruct": instruct}
    if base.config.num_hidden_layers != instruct.config.num_hidden_layers:
        raise RuntimeError("aligned checkpoints have different layer counts")
    layer_count = int(base.config.num_hidden_layers)
    if any(layer < 0 or layer >= layer_count for unit in units for layer in UNITS[unit]):
        raise RuntimeError(f"unit exceeds aligned {layer_count}-layer architecture")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            (row["state_id"], row["unit"], row["direction"])
            for row in read_jsonl(args.output)
        }
    sham_validation = {"requested": args.validate_sham, "passes": None, "cells": []}
    if args.validate_sham:
        state = sorted(states, key=lambda row: row["state_id"])[0]
        ids, mask, span, _prompt = rendered[state["state_id"]]
        ids = ids.to(device)
        mask = mask.to(device)
        max_tokens = max(128, int(state["target"]) * 3)
        for unit in units:
            layers = UNITS[unit]
            for model_name in ("base", "instruct"):
                model = models[model_name]
                clean_text, clean_tokens, _ = generate_clean(
                    model, tokenizer, ids, mask, max_tokens
                )
                self_donor = capture_spans(
                    model, ids, mask, layers, span[0], span[1]
                )
                patched_text, patched_tokens, _ = generate_patched(
                    model,
                    tokenizer,
                    ids,
                    mask,
                    layers,
                    self_donor,
                    span[0],
                    span[1],
                    max_tokens,
                )
                passed = (
                    clean_text == patched_text
                    and clean_tokens == patched_tokens
                )
                sham_validation["cells"].append(
                    {
                        "state_id": state["state_id"],
                        "unit": unit,
                        "model": model_name,
                        "clean_tokens": clean_tokens,
                        "patched_tokens": patched_tokens,
                        "byte_identical": clean_text == patched_text,
                        "passes": passed,
                    }
                )
                if not passed:
                    raise RuntimeError(
                        f"self-patch identity failed: {unit}/{model_name}"
                    )
                del self_donor
                torch.cuda.empty_cache()
        sham_validation["passes"] = True
    manifest = {
        "schema_version": 1,
        "protocol": "docs/PHASE3C_ACTIVATION_MEDIATION_DISCOVERY_PROTOCOL_20260730.md",
        "base": args.base,
        "instruct": args.instruct,
        "states_sha256": sha256(args.states),
        "chat_template_sha256": sha256(args.chat_template),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "split": args.split,
        "cases": len(case_ids),
        "states": len(states),
        "units": units,
        "directions": ["denoise", "noise"],
        "dtype": args.dtype,
        "sham_validation": sham_validation,
        "load_seconds": round(time.perf_counter() - load_started, 3),
        "max_tokens_rule": "max(128, target * 3)",
    }
    manifest_path = args.output.with_suffix(".runtime_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with args.output.open("a", encoding="utf-8") as handle:
        for state in sorted(states, key=lambda row: row["state_id"]):
            ids, mask, span, prompt = rendered[state["state_id"]]
            ids = ids.to(device)
            mask = mask.to(device)
            for unit in units:
                layers = UNITS[unit]
                for direction, recipient_name, donor_name in (
                    ("denoise", "base", "instruct"),
                    ("noise", "instruct", "base"),
                ):
                    key = (state["state_id"], unit, direction)
                    if key in completed:
                        continue
                    donor = capture_spans(
                        models[donor_name],
                        ids,
                        mask,
                        layers,
                        span[0],
                        span[1],
                    )
                    text, token_count, finish = generate_patched(
                        models[recipient_name],
                        tokenizer,
                        ids,
                        mask,
                        layers,
                        donor,
                        span[0],
                        span[1],
                        max(128, int(state["target"]) * 3),
                    )
                    row = row_from_text(
                        state,
                        unit,
                        direction,
                        recipient_name,
                        donor_name,
                        span,
                        prompt,
                        text,
                        token_count,
                        finish,
                    )
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.add(key)
                    print(
                        json.dumps(
                            {
                                "event": "patched_state",
                                "state_id": state["state_id"],
                                "unit": unit,
                                "direction": direction,
                                "direction_correct": row["direction_correct"],
                                "action_gain": row["action_gain"],
                            }
                        ),
                        flush=True,
                    )
                    del donor
                    torch.cuda.empty_cache()
    manifest["rows"] = len(completed)
    manifest["output_sha256"] = sha256(args.output)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
