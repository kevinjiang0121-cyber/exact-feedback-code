#!/usr/bin/env python3
"""Stream Base tensors into Instruct shards for frozen Phase3B conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from safetensors import safe_open
from safetensors.torch import save_file


BANDS = {
    "restore_blocks_00_07": tuple(range(0, 8)),
    "restore_blocks_08_15": tuple(range(8, 16)),
    "restore_blocks_16_23": tuple(range(16, 24)),
    "restore_blocks_24_31": tuple(range(24, 32)),
    "restore_noncontiguous": (0, 4, 8, 12, 16, 20, 24, 28),
}
ORDER = (
    "sham_copy",
    "restore_noncontiguous",
    "restore_blocks_00_07",
    "restore_blocks_08_15",
    "restore_blocks_16_23",
    "restore_blocks_24_31",
    "restore_output",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index(path: Path) -> dict[str, Any]:
    return json.loads(
        (path / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )


def selector(condition: str) -> Callable[[str], bool]:
    if condition == "sham_copy":
        layers = tuple(range(0, 8))
        return lambda key: any(
            key.startswith(f"model.layers.{layer}.") for layer in layers
        )
    if condition in BANDS:
        layers = BANDS[condition]
        return lambda key: any(
            key.startswith(f"model.layers.{layer}.") for layer in layers
        )
    if condition == "restore_output":
        return lambda key: (
            key.startswith("model.norm.") or key.startswith("lm_head.")
        )
    raise ValueError(condition)


def copy_metadata_files(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        shutil.copy2(path, output / path.name)


def materialize(
    base: Path, instruct: Path, output_root: Path, condition: str
) -> dict[str, Any]:
    output = output_root / condition
    manifest_path = output / "RESTORATION_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["condition"] != condition:
            raise RuntimeError(f"{output}: condition mismatch")
        print(json.dumps({"condition": condition, "status": "reuse"}))
        return manifest
    if output.exists():
        raise RuntimeError(f"partial output exists without manifest: {output}")
    output.mkdir(parents=True)
    copy_metadata_files(instruct, output)

    base_index = index(base)
    instruct_index = index(instruct)
    base_map = base_index["weight_map"]
    instruct_map = instruct_index["weight_map"]
    if set(base_map) != set(instruct_map):
        raise RuntimeError("Base/Instruct tensor-key sets differ")
    if any(base_map[key] != instruct_map[key] for key in base_map):
        raise RuntimeError("Base/Instruct shard maps differ")

    choose = selector(condition)
    selected_keys = sorted(key for key in instruct_map if choose(key))
    if not selected_keys:
        raise RuntimeError(f"{condition}: selected no tensors")
    source_for_selected = instruct if condition == "sham_copy" else base
    selected_set = set(selected_keys)
    shard_names = sorted(set(instruct_map.values()))
    shard_records = []
    selected_numel = 0
    for shard_name in shard_names:
        shard_keys = sorted(
            key for key, shard in instruct_map.items() if shard == shard_name
        )
        touched = any(key in selected_set for key in shard_keys)
        destination = output / shard_name
        if not touched:
            os.link(instruct / shard_name, destination)
            mode = "hardlink_instruct"
        else:
            tensors = {}
            with safe_open(
                instruct / shard_name, framework="pt", device="cpu"
            ) as instruct_handle, safe_open(
                source_for_selected / shard_name,
                framework="pt",
                device="cpu",
            ) as selected_handle:
                metadata = instruct_handle.metadata()
                for key in shard_keys:
                    handle = (
                        selected_handle
                        if key in selected_set
                        else instruct_handle
                    )
                    tensor = handle.get_tensor(key)
                    tensors[key] = tensor
                    if key in selected_set:
                        selected_numel += tensor.numel()
            save_file(tensors, destination, metadata=metadata)
            mode = (
                "rewrite_instruct_sham"
                if condition == "sham_copy"
                else "rewrite_with_base_selection"
            )
        shard_records.append(
            {
                "file": shard_name,
                "mode": mode,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
        print(
            json.dumps(
                {
                    "condition": condition,
                    "shard": shard_name,
                    "mode": mode,
                }
            ),
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "condition": condition,
        "recipient_checkpoint": str(instruct),
        "donor_checkpoint": (
            str(instruct) if condition == "sham_copy" else str(base)
        ),
        "base_index_sha256": sha256(
            base / "model.safetensors.index.json"
        ),
        "instruct_index_sha256": sha256(
            instruct / "model.safetensors.index.json"
        ),
        "selected_tensor_count": len(selected_keys),
        "selected_parameter_count": selected_numel,
        "selected_keys": selected_keys,
        "shards": shard_records,
        "interface_and_generation_config_source": "Meta Instruct",
        "embedding_restored": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--instruct", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--conditions", nargs="*", choices=ORDER, default=list(ORDER)
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = [
        materialize(args.base, args.instruct, args.output_root, condition)
        for condition in args.conditions
    ]
    summary = {
        "schema_version": 1,
        "conditions": [
            {
                "condition": row["condition"],
                "selected_tensor_count": row["selected_tensor_count"],
                "selected_parameter_count": row[
                    "selected_parameter_count"
                ],
            }
            for row in manifests
        ],
    }
    (args.output_root / "MATRIX_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
