#!/usr/bin/env python3
"""Materialize frozen aligned-pair restoration, dose, and graft checkpoints.

The historical filename and default Qwen groups are retained. A frozen JSON
group specification can override only the layer grouping for another aligned
architecture.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


GROUPS = {
    "blocks_00_08": tuple(range(0, 9)),
    "blocks_09_17": tuple(range(9, 18)),
    "blocks_18_26": tuple(range(18, 27)),
    "blocks_27_35": tuple(range(27, 36)),
    "noncontiguous": (0, 4, 8, 12, 16, 20, 24, 28, 32),
    "output": (),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_map(path: Path) -> dict[str, str]:
    return json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]


def copy_metadata(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        shutil.copy2(path, output / path.name)


def is_selected(key: str, groups: list[str], group_map: dict[str, tuple[int, ...]]) -> bool:
    layers = tuple(layer for group in groups for layer in group_map[group])
    if any(key.startswith(f"model.layers.{layer}.") for layer in layers):
        return True
    return "output" in groups and (key.startswith("model.norm.") or key.startswith("lm_head."))


def condition_spec(name: str) -> dict[str, Any]:
    if name == "base_common_interface":
        return {"recipient": "base", "donor": "base", "groups": [], "alpha": 0.0}
    if name == "instruct_sham":
        return {"recipient": "instruct", "donor": "instruct", "groups": [], "alpha": 0.0}
    if name.startswith("restore50_"):
        return {"recipient": "instruct", "donor": "base", "groups": [name.removeprefix("restore50_")], "alpha": 0.5}
    if name.startswith("restore_"):
        return {"recipient": "instruct", "donor": "base", "groups": [name.removeprefix("restore_")], "alpha": 1.0}
    if name.startswith("graft_"):
        groups = name.removeprefix("graft_").split("_plus_")
        return {"recipient": "base", "donor": "instruct", "groups": groups, "alpha": 1.0}
    raise ValueError(f"unknown condition: {name}")


def tensor_from(handles: dict[str, Any], weight_map: dict[str, str], key: str) -> torch.Tensor:
    return handles[weight_map[key]].get_tensor(key)


def materialize(
    base: Path,
    instruct: Path,
    output_root: Path,
    name: str,
    group_map: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    spec = condition_spec(name)
    models = {"base": base, "instruct": instruct}
    recipient = models[spec["recipient"]]
    donor = models[spec["donor"]]
    output = output_root / name
    manifest_path = output / "INTERVENTION_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["condition"] != name:
            raise RuntimeError(f"{output}: incompatible manifest")
        print(json.dumps({"condition": name, "status": "reuse"}), flush=True)
        return manifest
    if output.exists():
        raise RuntimeError(f"partial output without manifest: {output}")
    output.mkdir(parents=True)
    copy_metadata(instruct, output)
    # Tokenizer, chat-template, and generation metadata always follow the
    # Instruct interface.  The safetensor index is not interface metadata: it
    # must describe the recipient's shard layout (Gemma Base has eight shards
    # while Gemma IT has four).
    shutil.copy2(
        recipient / "model.safetensors.index.json",
        output / "model.safetensors.index.json",
    )

    recipient_map, donor_map = load_map(recipient), load_map(donor)
    if set(recipient_map) != set(donor_map):
        raise RuntimeError("Base/Instruct tensor-key sets differ")
    unknown = sorted(set(spec["groups"]) - set(group_map))
    if unknown:
        raise RuntimeError(f"unknown frozen groups: {unknown}")
    selected = sorted(key for key in recipient_map if is_selected(key, spec["groups"], group_map))
    selected_set = set(selected)
    records, selected_numel = [], 0
    for shard in sorted(set(recipient_map.values())):
        keys = sorted(key for key, value in recipient_map.items() if value == shard)
        touched = any(key in selected_set for key in keys)
        destination = output / shard
        if not touched:
            os.link(recipient / shard, destination)
            mode = f"hardlink_{spec['recipient']}"
        else:
            tensors = {}
            needed_recipient = {recipient_map[key] for key in keys}
            needed_donor = {donor_map[key] for key in keys if key in selected_set}
            with ExitStack() as stack:
                rh = {s: stack.enter_context(safe_open(recipient / s, framework="pt", device="cpu")) for s in needed_recipient}
                dh = {s: stack.enter_context(safe_open(donor / s, framework="pt", device="cpu")) for s in needed_donor}
                metadata = rh[shard].metadata()
                for key in keys:
                    recipient_tensor = tensor_from(rh, recipient_map, key)
                    if key in selected_set:
                        donor_tensor = tensor_from(dh, donor_map, key)
                        if donor_tensor.shape != recipient_tensor.shape:
                            raise RuntimeError(f"shape mismatch: {key}")
                        alpha = float(spec["alpha"])
                        # Aligned checkpoints may use different storage dtypes
                        # (Gemma Base is published as FP32 while IT is BF16).
                        # Every experimental model is loaded in BF16, so save
                        # transplanted values in the recipient dtype as well.
                        tensor = (
                            donor_tensor.to(recipient_tensor.dtype)
                            if alpha == 1.0
                            else torch.lerp(
                                recipient_tensor.float(), donor_tensor.float(), alpha
                            ).to(recipient_tensor.dtype)
                        )
                        selected_numel += tensor.numel()
                    else:
                        tensor = recipient_tensor
                    tensors[key] = tensor
            save_file(tensors, destination, metadata=metadata)
            mode = f"rewrite_{spec['recipient']}_from_{spec['donor']}_alpha{spec['alpha']}"
        records.append({"file": shard, "mode": mode, "bytes": destination.stat().st_size, "sha256": sha256(destination)})
        print(json.dumps({"condition": name, "shard": shard, "mode": mode}), flush=True)

    manifest = {
        "schema_version": 1,
        "condition": name,
        "recipient_tensor_checkpoint": str(recipient),
        "donor_tensor_checkpoint": str(donor),
        "interface_metadata_checkpoint": str(instruct),
        "weight_index_checkpoint": str(recipient),
        "weight_index_sha256": sha256(output / "model.safetensors.index.json"),
        "groups": spec["groups"],
        "alpha": spec["alpha"],
        "selected_tensor_count": len(selected),
        "selected_parameter_count": selected_numel,
        "selected_keys": selected,
        "embedding_transplanted": False,
        "shards": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--instruct", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--group-spec", type=Path)
    args = parser.parse_args()
    group_map = GROUPS
    if args.group_spec is not None:
        raw = json.loads(args.group_spec.read_text(encoding="utf-8"))
        group_map = {name: tuple(int(x) for x in layers) for name, layers in raw["groups"].items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    # G0/G1 and G2 intentionally overlap across GPUs. Serialize checkpoint
    # construction so both stages may request common controls without racing
    # between directory creation and the final manifest write.
    lock_path = args.output_root / ".materialize.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifests = [
            materialize(args.base, args.instruct, args.output_root, name, group_map)
            for name in args.conditions
        ]
    summary = {"schema_version": 1, "conditions": [{"condition": x["condition"], "groups": x["groups"], "alpha": x["alpha"], "selected_parameter_count": x["selected_parameter_count"]} for x in manifests]}
    (args.output_root / "MATRIX_MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
