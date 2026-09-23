#!/usr/bin/env python3
"""Capture matched Base/Instruct fixed-state residuals and prompt-end logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


POSITIONS = ("current", "target", "action_magnitude", "final_prompt")
META_EOT = "<|eot_id|>"
BASE_EOT = "<|end_of_text|>"


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


def render_chat(
    tokenizer: Any, template: str, messages: list[dict[str, str]]
) -> str:
    tokenizer.chat_template = template
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def feedback_spans(
    rendered: str, state: dict[str, Any]
) -> dict[str, tuple[int, int]]:
    feedback = state["messages"][-1]["content"]
    start = rendered.rfind(feedback)
    if start < 0:
        raise RuntimeError(f"{state['state_id']}: feedback not found")
    match = re.search(
        r"previous response = (?P<current>\d+) words; target = "
        r"(?P<target>\d+); required action = (?:add|remove) exactly "
        r"(?P<action_magnitude>\d+) words",
        feedback,
    )
    if match is None:
        raise RuntimeError(f"{state['state_id']}: feedback regex failed")
    expected = {
        "current": str(state["current_word_count"]),
        "target": str(state["target"]),
        "action_magnitude": str(abs(state["required_delta"])),
    }
    for name, value in expected.items():
        if match.group(name) != value:
            raise RuntimeError(
                f"{state['state_id']}: {name} {match.group(name)} != {value}"
            )
    return {
        name: (start + match.start(name), start + match.end(name))
        for name in ("current", "target", "action_magnitude")
    }


def token_indices(
    offsets: list[tuple[int, int]], span: tuple[int, int]
) -> list[int]:
    left, right = span
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if end > left and start < right
    ]


def locate(
    tokenizer: Any, template: str, state: dict[str, Any]
) -> tuple[str, dict[str, torch.Tensor], dict[str, list[int]]]:
    rendered = render_chat(tokenizer, template, state["messages"])
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [
        tuple(map(int, pair))
        for pair in encoded.pop("offset_mapping")[0].tolist()
    ]
    spans = feedback_spans(rendered, state)
    positions = {
        name: token_indices(offsets, span) for name, span in spans.items()
    }
    positions["final_prompt"] = [len(offsets) - 1]
    if any(not indices for indices in positions.values()):
        raise RuntimeError(f"{state['state_id']}: empty token span")
    return rendered, encoded, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    args = parser.parse_args()

    states = read_jsonl(args.states)
    if len(states) != 288 or len({row["state_id"] for row in states}) != 288:
        raise RuntimeError("expected 288 unique fixed states")
    template = args.chat_template.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    audits = []
    located = []
    for state in states:
        rendered, encoded, positions = locate(tokenizer, template, state)
        rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        audits.append(
            {
                "state_id": state["state_id"],
                "tokens": int(encoded["input_ids"].shape[1]),
                "rendered_sha256": rendered_sha,
                "span_token_counts": {
                    name: len(indices) for name, indices in positions.items()
                },
            }
        )
        located.append((rendered, encoded, positions))
    audit_path = args.output_dir / "token_span_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "states": len(states),
                "min_tokens": min(row["tokens"] for row in audits),
                "max_tokens": max(row["tokens"] for row in audits),
                "rows": audits,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    dtype = (
        torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    layers = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    residual_path = args.output_dir / "residual_selected.npy"
    residual = np.lib.format.open_memmap(
        residual_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(states), layers, len(POSITIONS), hidden_size),
    )
    metadata_path = args.output_dir / "metadata.jsonl"
    meta_eot_id = tokenizer.convert_tokens_to_ids(META_EOT)
    base_eot_id = tokenizer.convert_tokens_to_ids(BASE_EOT)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row_index, (state, cached) in enumerate(zip(states, located)):
            rendered, encoded, positions = cached
            inputs = {
                name: value.to(model.device) for name, value in encoded.items()
            }
            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            for layer_index, hidden in enumerate(outputs.hidden_states):
                for position_index, name in enumerate(POSITIONS):
                    indices = torch.tensor(
                        positions[name], device=hidden.device
                    )
                    vector = (
                        hidden[0]
                        .index_select(0, indices)
                        .mean(dim=0)
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    residual[
                        row_index, layer_index, position_index
                    ] = vector
            final_logits = outputs.logits[0, -1].float()
            top = int(torch.argmax(final_logits).item())
            metadata = {
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
            }
            metadata.update(
                {
                    "row_index": row_index,
                    "sequence_tokens": int(inputs["input_ids"].shape[1]),
                    "rendered_sha256": hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    "token_indices": positions,
                    "prompt_end_logits": {
                        "meta_eot_token_id": int(meta_eot_id),
                        "meta_eot": float(final_logits[meta_eot_id].item()),
                        "base_eot_token_id": int(base_eot_id),
                        "base_eot": float(final_logits[base_eot_id].item()),
                        "top_token_id": top,
                        "top_token": tokenizer.convert_ids_to_tokens(top),
                        "logsumexp": float(
                            torch.logsumexp(final_logits, dim=0).item()
                        ),
                    },
                }
            )
            handle.write(
                json.dumps(metadata, ensure_ascii=False) + "\n"
            )
            handle.flush()
            if (row_index + 1) % 12 == 0:
                residual.flush()
                print(
                    json.dumps(
                        {
                            "model": args.model_slug,
                            "completed": row_index + 1,
                            "total": len(states),
                        }
                    ),
                    flush=True,
                )
            del outputs
    residual.flush()
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_slug": args.model_slug,
        "dtype": args.dtype,
        "states": len(states),
        "layers_including_embedding": layers,
        "hidden_size": hidden_size,
        "positions": list(POSITIONS),
        "shape": list(residual.shape),
        "states_sha256": sha256(args.states),
        "chat_template_sha256": sha256(args.chat_template),
        "token_span_audit_sha256": sha256(audit_path),
        "metadata_sha256": sha256(metadata_path),
        "residual_sha256": sha256(residual_path),
    }
    (args.output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
