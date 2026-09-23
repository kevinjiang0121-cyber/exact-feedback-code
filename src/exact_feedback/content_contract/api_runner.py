#!/usr/bin/env python3
"""Auditable, resumable OpenRouter runner for the frozen multidomain panel.

Only protocol-complete cases enter the append-only canonical partial file.
Transport, provider-route, and API failures are logged separately and abort the
worker, so resumption never repeats a completed case or silently changes route.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

import exact_feedback.common.structured_protocol as protocol  # noqa: E402
import exact_feedback.closed_loop.api_runner as openrouter  # noqa: E402


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_existing(partial_path: Path, final_path: Path) -> tuple[list[dict[str, Any]], Path]:
    if final_path.exists() and partial_path.exists():
        raise RuntimeError("both final and partial case files exist")
    active = final_path if final_path.exists() else partial_path
    rows = openrouter.read_jsonl(active) if active.exists() else []
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate case IDs in {active}")
    return rows, active


def validate_route(config: dict[str, Any], api_meta: dict[str, Any]) -> None:
    gate = config["route_gate"]
    provider = api_meta.get("provider")
    returned_model = api_meta.get("returned_model")
    if provider != gate["expected_provider"]:
        raise RuntimeError(
            f"provider route mismatch: observed={provider!r}, "
            f"expected={gate['expected_provider']!r}"
        )
    if returned_model not in set(gate["allowed_returned_models"]):
        raise RuntimeError(
            f"returned model mismatch: observed={returned_model!r}, "
            f"allowed={gate['allowed_returned_models']!r}"
        )


def run_case(
    config: dict[str, Any],
    model_spec: dict[str, Any],
    item: dict[str, Any],
    token: str,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": protocol.initial_prompt(item)}]
    rounds: list[dict[str, Any]] = []
    max_revisions = int(config["protocol"]["max_revisions"])
    for revision in range(max_revisions + 1):
        text, api_meta = openrouter.chat_completion(
            config,
            model_spec,
            messages,
            protocol.max_tokens(item),
            token,
            f"{config['experiment_slug']}:{model_spec['label']}:{item['id']}:r{revision}",
        )
        validate_route(config, api_meta)
        verifier = protocol.verify(item["structure"], text, item["targets"])
        joint = bool(verifier["joint_success"])
        rounds.append(
            {
                "revision": revision,
                "text": text,
                "joint_success": joint,
                "violation_energy": float(sum(verifier["violation_vector"])),
                "verifier": verifier,
                "api": api_meta,
            }
        )
        if joint or revision == max_revisions:
            break
        messages.extend(
            [
                {"role": "assistant", "content": text},
                {"role": "user", "content": protocol.feedback_prompt(item, verifier)},
            ]
        )
    return {
        "id": item["id"],
        "family": item.get("family"),
        "structure": item["structure"],
        "domain": item.get("domain"),
        "source": item["source"],
        "targets": item["targets"],
        "configured_model": model_spec["id"],
        "model_label": model_spec["label"],
        "rounds": rounds,
        "api_error": None,
        "protocol_complete": True,
        "one_shot_joint": rounds[0]["joint_success"],
        "final_joint_success": rounds[-1]["joint_success"],
        "revisions_used": len(rounds) - 1,
    }


def audit_rows(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    require_complete: bool,
) -> None:
    if len(rows) > len(items):
        raise RuntimeError("output contains more rows than the frozen input")
    if require_complete and len(rows) != len(items):
        raise RuntimeError(f"expected {len(items)} rows, found {len(rows)}")
    max_revisions = int(config["protocol"]["max_revisions"])
    for index, row in enumerate(rows):
        item = items[index]
        if row.get("id") != item["id"]:
            raise RuntimeError(
                f"row order/ID mismatch at {index}: {row.get('id')} != {item['id']}"
            )
        if row.get("api_error") is not None or not row.get("protocol_complete"):
            raise RuntimeError(f"canonical row {row['id']} is not protocol-complete")
        rounds = row.get("rounds") or []
        if not rounds or len(rounds) > max_revisions + 1:
            raise RuntimeError(f"invalid round count for {row['id']}")
        for revision, round_row in enumerate(rounds):
            if round_row.get("revision") != revision:
                raise RuntimeError(f"nonsequential revisions for {row['id']}")
            expected = protocol.verify(
                item["structure"], round_row["text"], item["targets"]
            )
            if round_row.get("verifier") != expected:
                raise RuntimeError(f"verifier mismatch for {row['id']} r{revision}")
            if bool(round_row.get("joint_success")) != bool(expected["joint_success"]):
                raise RuntimeError(f"joint-success mismatch for {row['id']} r{revision}")
            validate_route(config, round_row.get("api") or {})
            if expected["joint_success"] and revision != len(rounds) - 1:
                raise RuntimeError(f"row continued after success: {row['id']}")
        if not rounds[-1]["joint_success"] and len(rounds) != max_revisions + 1:
            raise RuntimeError(f"row stopped before success/revision limit: {row['id']}")
        if bool(row.get("final_joint_success")) != bool(rounds[-1]["joint_success"]):
            raise RuntimeError(f"final-success mismatch for {row['id']}")


def summarize(
    rows: list[dict[str, Any]], live_prices: dict[str, float], budget: float
) -> dict[str, Any]:
    request_count = sum(len(row["rounds"]) for row in rows)
    prompt_tokens = sum(
        int((round_row.get("api", {}).get("usage") or {}).get("prompt_tokens") or 0)
        for row in rows
        for round_row in row["rounds"]
    )
    completion_tokens = sum(
        int((round_row.get("api", {}).get("usage") or {}).get("completion_tokens") or 0)
        for row in rows
        for round_row in row["rounds"]
    )
    spent = openrouter.rows_cost_usd(rows, live_prices)
    return {
        "cases": len(rows),
        "requests": request_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "observed_cost_usd": spent,
        "cost_ceiling_usd": budget,
        "mean_cost_per_case_usd": spent / len(rows) if rows else None,
        "projected_480_cost_usd": spent * 480 / len(rows) if rows else None,
        "one_shot_joint_rate": (
            sum(bool(row["one_shot_joint"]) for row in rows) / len(rows) if rows else None
        ),
        "final_joint_rate": (
            sum(bool(row["final_joint_success"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "median_revisions": (
            statistics.median(row["revisions_used"] for row in rows) if rows else None
        ),
    }


def execute(
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    selected: list[dict[str, Any]],
    token: str,
) -> None:
    if len(selected) != 1:
        raise RuntimeError("v2 safety runner requires exactly one enabled model")
    entry = selected[0]
    spec = entry["configured"]
    data_path = project_root / config["data"]["cases"]
    if openrouter.sha256_file(data_path) != config["data"]["expected_cases_sha256"]:
        raise RuntimeError("frozen input SHA-256 mismatch")
    items = openrouter.read_jsonl(data_path)
    if len(items) != int(config["data"]["expected_cases"]):
        raise RuntimeError("frozen input case-count mismatch")
    if any(item.get("domain") != config["data"]["domain"] for item in items):
        raise RuntimeError("input domain mismatch")
    os.environ["MULTIDOMAIN_PROMPT_VARIANT"] = config["protocol"]["prompt_variant"]

    model_dir = project_root / config["output_dir"] / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    partial_path = model_dir / "cases.partial.jsonl"
    final_path = model_dir / "cases.jsonl"
    failure_path = model_dir / "failed_requests.jsonl"
    rows, active_path = load_existing(partial_path, final_path)
    audit_rows(rows, items, config, require_complete=final_path.exists())
    if final_path.exists():
        print(f"already complete and audited: {final_path}")
        return

    prices = entry["live_price_usd_per_1m"]
    ceiling = float(config["budget"]["max_total_usd"])
    spent = openrouter.rows_cost_usd(rows, prices)
    for item in items[len(rows) :]:
        if spent >= ceiling:
            raise RuntimeError(
                f"case-boundary cost ceiling reached before {item['id']}: "
                f"${spent:.6f} >= ${ceiling:.2f}"
            )
        try:
            row = run_case(config, spec, item, token)
        except Exception as exc:
            append_jsonl(
                failure_path,
                {
                    "at_utc": openrouter.utc_now(),
                    "id": item["id"],
                    "completed_cases_preserved": len(rows),
                    "type": type(exc).__name__,
                    "message": str(exc)[:2000],
                },
            )
            raise RuntimeError(
                f"aborting on failed case {item['id']}; {len(rows)} completed cases preserved"
            ) from exc
        append_jsonl(partial_path, row)
        rows.append(row)
        spent += openrouter.rows_cost_usd([row], prices)
        progress = summarize(rows, prices, ceiling)
        print(
            json.dumps(
                {
                    "case": len(rows),
                    "of": len(items),
                    "id": row["id"],
                    "final_joint": row["final_joint_success"],
                    "revisions": row["revisions_used"],
                    "cost_usd": round(spent, 6),
                    "projected_480_cost_usd": round(
                        progress["projected_480_cost_usd"], 4
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    audit_rows(rows, items, config, require_complete=True)
    partial_path.replace(final_path)
    summary = summarize(rows, prices, ceiling)
    summary.update(
        {
            "completed_at_utc": openrouter.utc_now(),
            "catalog_price_usd_per_1m": prices,
            "input_sha256": openrouter.sha256_file(data_path),
            "cases_sha256": openrouter.sha256_file(final_path),
            "config_sha256": openrouter.sha256_file(config_path),
            "runner_sha256": openrouter.sha256_file(Path(__file__)),
            "verdict": "PASS",
        }
    )
    (model_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, **summary}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = openrouter.read_json(config_path)
    if args.env_file:
        openrouter.load_env_file(args.env_file.resolve())
    catalog = openrouter.fetch_catalog(config)
    selected = openrouter.validate_catalog(config, catalog)
    output_dir = project_root / config["output_dir"]
    snapshot = openrouter.write_catalog_snapshot(output_dir, config_path, selected)
    for entry in selected:
        prices = entry["live_price_usd_per_1m"]
        print(
            f"{entry['configured']['id']}: input=${prices['prompt']:g}/1M, "
            f"output=${prices['completion']:g}/1M"
        )
    print(f"catalog snapshot: {snapshot}")
    if args.preflight_only:
        print("preflight passed; no paid generation was requested")
        return
    if not config.get("execution_enabled"):
        raise RuntimeError("execution is disabled in the frozen config")
    token = os.environ.get(config["api"]["token_env"], "").strip()
    if not token:
        raise RuntimeError("OpenRouter token is empty")
    execute(project_root, config_path, config, selected, token)


if __name__ == "__main__":
    main()
