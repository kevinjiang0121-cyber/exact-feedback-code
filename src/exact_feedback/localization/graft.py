#!/usr/bin/env python3
"""Materialize frozen Instruct-to-Base coordinated controller grafts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


CONDITIONS = {
    "base_meta_contract_sham": {"layers": (), "output": False},
    "graft_blocks_08_15": {"layers": tuple(range(8, 16)), "output": False},
    "graft_output": {"layers": (), "output": True},
    "graft_noncontiguous_plus_output": {
        "layers": (0, 4, 8, 12, 16, 20, 24, 28),
        "output": True,
    },
    "graft_blocks_08_15_plus_output": {
        "layers": tuple(range(8, 16)),
        "output": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: Path) -> dict[str, Any]:
    return json.loads(
        (path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )


def copy_metadata(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        shutil.copy2(path, output / path.name)


def selected(condition: str, key: str) -> bool:
    spec = CONDITIONS[condition]
    in_layer = any(
        key.startswith(f"model.layers.{layer}.")
        for layer in spec["layers"]
    )
    in_output = spec["output"] and (
        key.startswith("model.norm.") or key.startswith("lm_head.")
    )
    return in_layer or in_output


def materialize(
    base: Path, instruct: Path, output_root: Path, condition: str
) -> dict[str, Any]:
    output = output_root / condition
    manifest_path = output / "GRAFT_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["condition"] != condition:
            raise RuntimeError(f"{output}: incompatible manifest")
        print(json.dumps({"condition": condition, "status": "reuse"}))
        return manifest
    if output.exists():
        raise RuntimeError(f"partial output without manifest: {output}")
    output.mkdir(parents=True)
    copy_metadata(instruct, output)

    base_index = load_index(base)
    instruct_index = load_index(instruct)
    base_map = base_index["weight_map"]
    instruct_map = instruct_index["weight_map"]
    if set(base_map) != set(instruct_map):
        raise RuntimeError("Base/Instruct tensor-key sets differ")
    if any(base_map[key] != instruct_map[key] for key in base_map):
        raise RuntimeError("Base/Instruct shard maps differ")

    selected_keys = sorted(
        key for key in base_map if selected(condition, key)
    )
    selected_set = set(selected_keys)
    selected_numel = 0
    shard_records = []
    for shard_name in sorted(set(base_map.values())):
        shard_keys = sorted(
            key for key, shard in base_map.items() if shard == shard_name
        )
        touched = any(key in selected_set for key in shard_keys)
        destination = output / shard_name
        if not touched:
            os.link(base / shard_name, destination)
            mode = "hardlink_base"
        else:
            tensors = {}
            with safe_open(
                base / shard_name, framework="pt", device="cpu"
            ) as base_handle, safe_open(
                instruct / shard_name, framework="pt", device="cpu"
            ) as instruct_handle:
                metadata = base_handle.metadata()
                for key in shard_keys:
                    handle = (
                        instruct_handle if key in selected_set else base_handle
                    )
                    tensor = handle.get_tensor(key)
                    tensors[key] = tensor
                    if key in selected_set:
                        selected_numel += tensor.numel()
            save_file(tensors, destination, metadata=metadata)
            mode = "rewrite_with_instruct_graft"
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
                {"condition": condition, "shard": shard_name, "mode": mode}
            ),
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "condition": condition,
        "recipient_tensor_checkpoint": str(base),
        "donor_tensor_checkpoint": str(instruct),
        "metadata_checkpoint": str(instruct),
        "base_index_sha256": sha256(base / "model.safetensors.index.json"),
        "instruct_index_sha256": sha256(
            instruct / "model.safetensors.index.json"
        ),
        "selected_layers": list(CONDITIONS[condition]["layers"]),
        "selected_output": CONDITIONS[condition]["output"],
        "selected_tensor_count": len(selected_keys),
        "selected_parameter_count": selected_numel,
        "selected_keys": selected_keys,
        "shards": shard_records,
        "embedding_grafted": False,
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
        "--conditions", nargs="*", choices=tuple(CONDITIONS),
        default=list(CONDITIONS),
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
                "selected_parameter_count": row["selected_parameter_count"],
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
