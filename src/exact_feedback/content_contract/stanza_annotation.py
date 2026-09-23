#!/usr/bin/env python3
"""Frozen Stanza syntax, entity, and entity-chain annotations for endpoint texts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import stanza


TRAILING_CONJ = {"and", "or", "but", "so", "yet", "nor"}
SYMBOL_RE = re.compile(
    r"https?://\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?<!\w)(?:[$€£¥]\s*)?\d+(?:[.,]\d+)*(?:\s*%|\s*(?:km|kg|cm|mm|m|g|lb|mph|°[CF]))?(?!\w)",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_unit(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def parse_feats(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return dict(part.split("=", 1) for part in raw.split("|") if "=" in part)


def annotate(nlp: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"]
    doc = nlp(text)
    entities = [
        {
            "text": ent.text,
            "type": ent.type,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "normalized": normalize_unit(ent.text),
        }
        for ent in doc.ents
    ]
    sentence_rows = []
    fragments = []
    agreement = []
    dangling = []
    proper_names = []
    for sentence_index, sentence in enumerate(doc.sentences):
        words = sentence.words
        sentence_start = min(
            (word.start_char for word in words if word.start_char is not None), default=0
        )
        sentence_end = max(
            (word.end_char for word in words if word.end_char is not None), default=sentence_start
        )
        sentence_text = text[sentence_start:sentence_end]
        finite = [
            word
            for word in words
            if word.upos in {"VERB", "AUX"} and parse_feats(word.feats).get("VerbForm") == "Fin"
        ]
        alphabetic = [word for word in words if any(char.isalpha() for char in word.text)]
        is_fragment = len(alphabetic) >= 3 and not finite
        if is_fragment:
            fragments.append({"sentence": sentence_index, "text": sentence_text})
        last_alpha = alphabetic[-1].lemma.casefold() if alphabetic else ""
        if last_alpha in TRAILING_CONJ:
            dangling.append(
                {"sentence": sentence_index, "token": alphabetic[-1].text, "kind": "trailing_conjunction"}
            )
        by_id = {int(word.id): word for word in words if isinstance(word.id, int)}
        for word in words:
            if word.upos == "PROPN":
                proper_names.append(
                    {
                        "text": word.text,
                        "start_char": word.start_char,
                        "end_char": word.end_char,
                        "normalized": normalize_unit(word.text),
                    }
                )
            if word.deprel.startswith("nsubj") and int(word.head or 0) in by_id:
                head = by_id[int(word.head)]
                subject_number = parse_feats(word.feats).get("Number")
                head_number = parse_feats(head.feats).get("Number")
                if subject_number and head_number and subject_number != head_number:
                    agreement.append(
                        {
                            "sentence": sentence_index,
                            "subject": word.text,
                            "predicate": head.text,
                            "subject_number": subject_number,
                            "predicate_number": head_number,
                        }
                    )
        sentence_entities = sorted(
            {
                f"{ent['type']}:{ent['normalized']}"
                for ent in entities
                if ent["start_char"] < sentence_end and ent["end_char"] > sentence_start
            }
            | {
                f"PROPN:{item['normalized']}"
                for item in proper_names
                if item["start_char"] is not None
                and item["start_char"] < sentence_end
                and item["end_char"] > sentence_start
            }
        )
        sentence_rows.append(
            {
                "index": sentence_index,
                "start_char": sentence_start,
                "end_char": sentence_end,
                "text": sentence_text,
                "entity_units": sentence_entities,
                "tokens": [
                    {
                        "id": word.id,
                        "text": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "feats": word.feats,
                        "head": word.head,
                        "deprel": word.deprel,
                        "start_char": word.start_char,
                        "end_char": word.end_char,
                    }
                    for word in words
                ],
            }
        )
    adjacent_overlap = []
    for left, right in zip(sentence_rows, sentence_rows[1:]):
        a, b = set(left["entity_units"]), set(right["entity_units"])
        adjacent_overlap.append(
            {
                "left": left["index"],
                "right": right["index"],
                "intersection": sorted(a & b),
                "has_overlap": bool(a & b),
                "jaccard": len(a & b) / len(a | b) if a | b else None,
            }
        )
    return {
        "schema_version": 1,
        "text_sha256": row["text_sha256"],
        "status": "ok",
        "sentence_count": len(sentence_rows),
        "token_count": sum(len(sentence.words) for sentence in doc.sentences),
        "sentences": sentence_rows,
        "entities": entities,
        "proper_names": proper_names,
        "symbolic_units": [
            {
                "text": match.group(0),
                "normalized": normalize_unit(match.group(0)),
                "start_char": match.start(),
                "end_char": match.end(),
            }
            for match in SYMBOL_RE.finditer(text)
        ],
        "fragment_rule_hits": fragments,
        "agreement_mismatch_hits": agreement,
        "dangling_conjunction_hits": dangling,
        "adjacent_entity_overlap": adjacent_overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--resources-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.registry)
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        completed = {row["text_sha256"]: row for row in read_jsonl(args.output)}
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse,ner",
        package="default",
        dir=str(args.resources_dir),
        download_method=None,
        use_gpu=False,
        verbose=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            if row["text_sha256"] in completed:
                continue
            try:
                result = annotate(nlp, row)
            except Exception as error:
                result = {
                    "schema_version": 1,
                    "text_sha256": row["text_sha256"],
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"completed": index, "hash": row["text_sha256"], "status": result["status"]}), flush=True)
    annotations = read_jsonl(args.output)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "stanza",
        "tool_version": importlib.metadata.version("stanza"),
        "processors": ["tokenize", "pos", "lemma", "depparse", "ner"],
        "package": "default",
        "use_gpu": False,
        "registry_sha256": sha256(args.registry),
        "resources_manifest_sha256": sha256(args.resources_dir / "resources.json"),
        "annotations": len(annotations),
        "ok": sum(row["status"] == "ok" for row in annotations),
        "errors": sum(row["status"] == "error" for row in annotations),
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
