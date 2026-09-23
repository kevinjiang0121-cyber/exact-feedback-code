#!/usr/bin/env python3
"""Run the frozen exact-length 60-case screen through OpenRouter.

Catalog inspection is free and does not require a token. Paid generation is
impossible unless both a non-empty token and --execute are provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exact_feedback.common.exact_protocol import contains_literal, word_count
from exact_feedback.robustness.prompt_runner import render_feedback, render_initial


class NonRetryableAPIError(RuntimeError):
    """Provider rejected a request in a way that identical retries cannot fix."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, value)


def http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 180,
    transport: str = "urllib",
) -> tuple[dict[str, Any], dict[str, str]]:
    if transport == "requests":
        import requests

        response = requests.request(
            method,
            url,
            headers=headers or {},
            json=body,
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}: {response.text[:2000]}"
            )
        return response.json(), {
            key.lower(): value for key, value in response.headers.items()
        }
    if transport != "urllib":
        raise ValueError(f"unsupported HTTP transport: {transport}")

    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return json.loads(response.read().decode("utf-8")), response_headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:2000]}") from exc


def fetch_catalog(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_url = config["api"]["base_url"].rstrip("/")
    payload, _ = http_json(
        f"{base_url}/models",
        timeout=int(config["api"]["timeout_seconds"]),
        transport=config["api"].get("transport", "urllib"),
    )
    if not isinstance(payload.get("data"), list):
        raise RuntimeError("OpenRouter model catalog did not contain a data list")
    return payload["data"]


def per_million(raw_per_token: Any) -> float:
    return float(raw_per_token) * 1_000_000


def validate_catalog(
    config: dict[str, Any], catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in catalog}
    selected = []
    errors = []
    for spec in config["models"]:
        if not spec.get("enabled", True):
            continue
        model_id = spec["id"]
        row = by_id.get(model_id)
        if row is None:
            errors.append(f"{model_id}: missing from live catalog")
            continue
        modalities = row.get("architecture") or {}
        if config["catalog_policy"].get("require_text_input_and_output", True):
            if "text" not in modalities.get("input_modalities", []):
                errors.append(f"{model_id}: live catalog lacks text input")
            if "text" not in modalities.get("output_modalities", []):
                errors.append(f"{model_id}: live catalog lacks text output")
        supported = set(row.get("supported_parameters") or [])
        requested = set(spec.get("request_parameters") or {})
        unsupported = requested - supported
        if unsupported:
            errors.append(
                f"{model_id}: requested parameters absent from live catalog: "
                + ", ".join(sorted(unsupported))
            )
        live_prices = {
            "prompt": per_million(row["pricing"]["prompt"]),
            "completion": per_million(row["pricing"]["completion"]),
        }
        ceilings = spec["price_ceiling_usd_per_1m"]
        for kind, live_price in live_prices.items():
            if live_price > float(ceilings[kind]):
                errors.append(
                    f"{model_id}: live {kind} price ${live_price:g}/1M exceeds "
                    f"ceiling ${float(ceilings[kind]):g}/1M"
                )
        selected.append(
            {
                "configured": spec,
                "live": row,
                "live_price_usd_per_1m": live_prices,
            }
        )
    if errors:
        raise RuntimeError("catalog validation failed:\n- " + "\n- ".join(errors))
    return selected


def write_catalog_snapshot(
    output_dir: Path,
    config_path: Path,
    selected: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "retrieved_at_utc": utc_now(),
        "source": "https://openrouter.ai/api/v1/models",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "models": selected,
    }
    path = output_dir / "openrouter_catalog_snapshot.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        safe_response = json.dumps(response, ensure_ascii=False)
        error = response.get("error") or {}
        error_code = error.get("code")
        error_type = NonRetryableAPIError if error_code and int(error_code) < 500 else RuntimeError
        raise error_type(
            "OpenRouter response omitted a non-empty choices list: "
            + safe_response[:2000]
        )
    message = choices[0].get("message")
    if not isinstance(message, dict):
        safe_response = json.dumps(response, ensure_ascii=False)
        raise RuntimeError(
            "OpenRouter response omitted choices[0].message: "
            + safe_response[:2000]
        )
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()


def chat_completion(
    config: dict[str, Any],
    model_spec: dict[str, Any],
    messages: list[dict[str, str]],
    max_tokens: int,
    token: str,
    trace_name: str,
) -> tuple[str, dict[str, Any]]:
    api = config["api"]
    body = {
        "model": model_spec["id"],
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "provider": config["protocol"]["provider"],
        "trace": {"trace_name": trace_name},
        **model_spec.get("request_parameters", {}),
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": api["app_title"],
        "X-OpenRouter-Metadata": "enabled",
    }
    if api.get("http_referer"):
        headers["HTTP-Referer"] = api["http_referer"]
    attempts = int(api["max_attempts"])
    for attempt in range(1, attempts + 1):
        try:
            response, response_headers = http_json(
                f"{api['base_url'].rstrip('/')}/chat/completions",
                method="POST",
                headers=headers,
                body=body,
                timeout=int(api["timeout_seconds"]),
                transport=api.get("transport", "urllib"),
            )
            metadata = {
                "response_id": response.get("id"),
                "generation_id": response_headers.get("x-generation-id")
                or response.get("id"),
                "returned_model": response.get("model"),
                "provider": response.get("provider")
                or (response.get("openrouter_metadata") or {}).get("provider_name"),
                "system_fingerprint": response.get("system_fingerprint"),
                "usage": response.get("usage"),
                "openrouter_metadata": response.get("openrouter_metadata"),
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
            }
            return extract_text(response), metadata
        except NonRetryableAPIError:
            raise
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def run_case(
    config: dict[str, Any],
    model_spec: dict[str, Any],
    item: dict[str, Any],
    template_name: str,
    template: dict[str, str],
    token: str,
) -> dict[str, Any]:
    protocol = config["protocol"]
    messages = [{"role": "user", "content": render_initial(item, template)}]
    rounds = []
    for revision in range(int(protocol["max_revisions"]) + 1):
        max_tokens = max(
            int(protocol["max_tokens_floor"]),
            int(item["target"]) * int(protocol["max_tokens_per_target_word"]),
        )
        try:
            text, api_meta = chat_completion(
                config,
                model_spec,
                messages,
                max_tokens,
                token,
                f"{config['experiment_slug']}:{model_spec['label']}:{item['id']}:r{revision}",
            )
        except Exception as exc:
            error_text = str(exc)
            return {
                "id": item["id"],
                "source": item["source"],
                "source_id": item["source_id"],
                "length_band": item["length_band"],
                "target": item["target"],
                "required": item["anchors"],
                "template": template_name,
                "configured_model": model_spec["id"],
                "model_label": model_spec["label"],
                "rounds": rounds,
                "api_error": {
                    "revision": revision,
                    "type": type(exc).__name__,
                    "message": error_text[:2000],
                },
                "protocol_complete": False,
                "one_shot_exact": rounds[0]["exact_length"] if rounds else None,
                "one_shot_joint": rounds[0]["joint_success"] if rounds else None,
                "final_exact": None,
                "final_joint": None,
                "ever_exact": any(row["exact_length"] for row in rounds),
                "revisions_used": max(0, len(rounds) - 1),
            }
        count = word_count(text)
        missing = [
            anchor for anchor in item["anchors"] if not contains_literal(text, anchor)
        ]
        exact = count == item["target"]
        joint = exact and not missing
        rounds.append(
            {
                "revision": revision,
                "text": text,
                "word_count": count,
                "error": count - item["target"],
                "exact_length": exact,
                "missing": missing,
                "joint_success": joint,
                "api": api_meta,
            }
        )
        if joint or revision == int(protocol["max_revisions"]):
            break
        messages.extend(
            [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": render_feedback(text, item, missing, template),
                },
            ]
        )
    return {
        "id": item["id"],
        "source": item["source"],
        "source_id": item["source_id"],
        "length_band": item["length_band"],
        "target": item["target"],
        "required": item["anchors"],
        "template": template_name,
        "configured_model": model_spec["id"],
        "model_label": model_spec["label"],
        "rounds": rounds,
        "api_error": None,
        "protocol_complete": True,
        "one_shot_exact": rounds[0]["exact_length"],
        "one_shot_joint": rounds[0]["joint_success"],
        "final_exact": rounds[-1]["exact_length"],
        "final_joint": rounds[-1]["joint_success"],
        "ever_exact": any(row["exact_length"] for row in rounds),
        "revisions_used": len(rounds) - 1,
    }


def existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate case IDs in resumable output: {path}")
    return rows


def response_cost_usd(
    api_meta: dict[str, Any], live_price_usd_per_1m: dict[str, float]
) -> float:
    usage = api_meta.get("usage") or {}
    if usage.get("cost") is not None:
        return float(usage["cost"])
    prompt_tokens = float(usage.get("prompt_tokens") or 0)
    completion_tokens = float(usage.get("completion_tokens") or 0)
    return (
        prompt_tokens * float(live_price_usd_per_1m["prompt"])
        + completion_tokens * float(live_price_usd_per_1m["completion"])
    ) / 1_000_000


def rows_cost_usd(
    rows: list[dict[str, Any]], live_price_usd_per_1m: dict[str, float]
) -> float:
    return sum(
        response_cost_usd(round_row.get("api") or {}, live_price_usd_per_1m)
        for row in rows
        for round_row in row.get("rounds", [])
    )


def summarize_api_rows(
    rows: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("protocol_complete", True)]
    blocked = [row for row in rows if not row.get("protocol_complete", True)]

    def rate(selected: list[dict[str, Any]], key: str) -> float | None:
        return (
            sum(bool(row.get(key)) for row in selected) / len(selected)
            if selected
            else None
        )

    return {
        "model": model_name,
        "overall": {
            "cases_attempted": len(rows),
            "protocol_complete_cases": len(completed),
            "api_error_cases": len(blocked),
            "one_shot_joint_all_cases": rate(rows, "one_shot_joint"),
            "final_joint_all_cases_refusal_as_failure": rate(rows, "final_joint"),
            "final_joint_completed_cases": rate(completed, "final_joint"),
            "final_exact_all_cases_refusal_as_failure": rate(rows, "final_exact"),
            "final_exact_completed_cases": rate(completed, "final_exact"),
            "ever_exact_all_cases": rate(rows, "ever_exact"),
            "median_revisions_completed_cases": (
                statistics.median(row["revisions_used"] for row in completed)
                if completed
                else None
            ),
        },
        "api_errors": [
            {
                "id": row["id"],
                "source": row["source"],
                "revision": row["api_error"]["revision"],
                "type": row["api_error"]["type"],
                "message": row["api_error"]["message"],
            }
            for row in blocked
        ],
    }


def execute(
    project_root: Path,
    config: dict[str, Any],
    selected: list[dict[str, Any]],
    token: str,
) -> None:
    data_cfg = config["data"]
    cases_path = project_root / data_cfg["cases"]
    if sha256_file(cases_path) != data_cfg["expected_cases_sha256"]:
        raise RuntimeError("frozen cases SHA-256 mismatch")
    items = read_jsonl(cases_path)
    if len(items) != int(data_cfg["expected_cases"]):
        raise RuntimeError("frozen case count mismatch")
    templates = read_json(project_root / data_cfg["templates"])
    template_name = data_cfg["template"]
    template = templates[template_name]
    output_dir = project_root / config["output_dir"]

    for entry in selected:
        spec = entry["configured"]
        model_dir = output_dir / spec["label"] / template_name
        model_dir.mkdir(parents=True, exist_ok=True)
        cases_output = model_dir / "cases.jsonl"
        rows = existing_rows(cases_output)
        completed = {row["id"] for row in rows}
        spent_usd = rows_cost_usd(rows, entry["live_price_usd_per_1m"])
        total_ceiling = float(config["budget"]["max_total_usd"])
        consecutive_transport_errors = 0
        transport_error_types = {
            "URLError",
            "ConnectionError",
            "ConnectionResetError",
            "TimeoutError",
        }
        with cases_output.open("a", encoding="utf-8") as handle:
            for item in items:
                if item["id"] in completed:
                    continue
                if spent_usd >= total_ceiling:
                    raise RuntimeError(
                        f"cost ceiling reached before case {item['id']}: "
                        f"${spent_usd:.6f} >= ${total_ceiling:.2f}"
                    )
                row = run_case(config, spec, item, template_name, template, token)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                spent_usd += rows_cost_usd([row], entry["live_price_usd_per_1m"])
                error_type = (row.get("api_error") or {}).get("type")
                if error_type in transport_error_types:
                    consecutive_transport_errors += 1
                else:
                    consecutive_transport_errors = 0
                print(
                    json.dumps(
                        {
                            "model": spec["label"],
                            "id": row["id"],
                            "final_joint": row["final_joint"],
                            "revisions": row["revisions_used"],
                            "api_error": row.get("api_error"),
                            "cumulative_cost_usd": round(spent_usd, 6),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if consecutive_transport_errors >= 3:
                    raise RuntimeError(
                        "aborting after three consecutive transport-failed "
                        "cases; preserve and recover the failed rows before resume"
                    )
        summary = summarize_api_rows(rows, spec["id"])
        summary.update(
            {
                "template": template_name,
                "catalog_price_usd_per_1m": entry["live_price_usd_per_1m"],
                "observed_cost_usd": spent_usd,
                "cost_ceiling_usd": total_ceiling,
                "completed_at_utc": utc_now(),
            }
        )
        (model_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--catalog-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = read_json(config_path)
    if args.execute and not config.get("execution_enabled", False):
        raise RuntimeError(
            f"execution is disabled for config status={config.get('status', 'unspecified')}"
        )
    if args.env_file:
        load_env_file(args.env_file.resolve())

    catalog = fetch_catalog(config)
    selected = validate_catalog(config, catalog)
    output_dir = project_root / config["output_dir"]
    snapshot_path = write_catalog_snapshot(output_dir, config_path, selected)
    for entry in selected:
        prices = entry["live_price_usd_per_1m"]
        print(
            f"{entry['configured']['id']}: "
            f"input=${prices['prompt']:g}/1M, output=${prices['completion']:g}/1M"
        )
    print(f"catalog snapshot: {snapshot_path}")

    if args.catalog_only:
        print("catalog validation passed; no paid generation was requested")
        return

    token_name = config["api"]["token_env"]
    token = os.environ.get(token_name, "").strip()
    if not token:
        raise RuntimeError(
            f"{token_name} is empty; fill the local env file before using --execute"
        )
    execute(project_root, config, selected, token)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
