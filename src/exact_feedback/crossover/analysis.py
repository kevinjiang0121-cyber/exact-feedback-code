#!/usr/bin/env python3
"""Integrity and paired attribution analysis for three Combined-240 crossovers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from exact_feedback.common.exact_protocol import contains_literal, word_count


PEERS = ("qwen3_14b", "gemma2_9b", "glm4_9b")


def read(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def combined(root_a: Path, root_b: Path, model: str) -> dict[str, dict[str, Any]]:
    return {**read(root_a / model / "cases.jsonl"), **read(root_b / model / "cases.jsonl")}


def mcnemar(left: dict[str, bool], right: dict[str, bool], ids: list[str]) -> dict[str, Any]:
    lo = sum(left[i] and not right[i] for i in ids)
    ro = sum(right[i] and not left[i] for i in ids)
    n = lo + ro
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(lo, ro) + 1)) / (2 ** n))
    return {
        "cases": len(ids), "left_only": lo, "right_only": ro,
        "effect_pp": 100 * (lo - ro) / len(ids), "exact_two_sided_p": p,
    }


def holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda value: value[1]["exact_two_sided_p"])
    running = 0.0
    m = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, (m - rank) * row["exact_two_sided_p"])
        rows[index]["holm_p"] = min(1.0, running)


def audit_rows(rows: dict[str, dict[str, Any]], data: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    for case, row in rows.items():
        item = data[case]
        for rnd in row["rounds"]:
            count = word_count(rnd["text"])
            missing = [a for a in item["anchors"] if not contains_literal(rnd["text"], a)]
            if count != rnd["word_count"] or missing != rnd["missing"] or (count == item["target"] and not missing) != rnd["joint_success"]:
                errors.append(f"{case}:r{rnd['revision']}")
    return errors


class CrossoverAnalyzer:
    def analyze_frozen(self, materials: Path, output_dir: Path) -> dict:
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as temporary:
            results=Path(temporary)
            for model in ('llama31_8b',*PEERS):
                target=results/model/f'{model}_planner__{model}_controller/cases.jsonl'
                target.parent.mkdir(parents=True)
                data=[]
                for split in ['human_generation_main120_v1','human_generation_replication120_v1']:
                    data.extend((materials/'experiments'/split/model/'cases.jsonl').read_text(encoding='utf-8').splitlines())
                target.write_text('\n'.join(data)+'\n',encoding='utf-8')
            for peer in PEERS:
                for source,reviser in [('llama31_8b',peer),(peer,'llama31_8b')]:
                    cell=f'{source}_planner__{reviser}_controller'
                    target=results/reviser/cell/'cases.jsonl';target.parent.mkdir(parents=True)
                    shutil.copy2(materials/'experiments/controller_crossover_combined240_v1'/cell/'cases.jsonl',target)
            return self.analyze(materials/'data/controller_crossover_combined240_v1/generation_combined240.jsonl',results,output_dir)

    def analyze(self, data_path: Path, results_root: Path, output_dir: Path) -> dict:
        data = read(data_path)
        llama = read(results_root / "llama31_8b" / "llama31_8b_planner__llama31_8b_controller/cases.jsonl")
        ids = sorted(data)
        if len(ids) != 240 or set(llama) != set(ids):
            raise RuntimeError("Diagonal/data identity failure")
    
        controller_tests = []
        planner_tests = []
        matrices = []
        integrity: dict[str, Any] = {"passes": True, "pairs": {}}
        report: dict[str, Any] = {"pairs": {}}
    
        for peer in PEERS:
            peer_diag = read(results_root / peer / f"{peer}_planner__{peer}_controller/cases.jsonl")
            l_to_p = read(results_root / peer / f"llama31_8b_planner__{peer}_controller" / "cases.jsonl")
            p_to_l = read(results_root / "llama31_8b" / f"{peer}_planner__llama31_8b_controller" / "cases.jsonl")
            cells = {"llama_llama": llama, "llama_peer": l_to_p, "peer_llama": p_to_l, "peer_peer": peer_diag}
            identity = all(set(rows) == set(ids) for rows in cells.values())
            l_zero = all(llama[i]["rounds"][0]["text"] == l_to_p[i]["rounds"][0]["text"] for i in ids)
            p_zero = all(peer_diag[i]["rounds"][0]["text"] == p_to_l[i]["rounds"][0]["text"] for i in ids)
            recount = {name: audit_rows(rows, data) for name, rows in cells.items()}
            pair_pass = identity and l_zero and p_zero and not any(recount.values())
            integrity["passes"] = integrity["passes"] and pair_pass
            integrity["pairs"][peer] = {"passes": pair_pass, "case_identity": identity, "llama_draft_byte_identity": l_zero, "peer_draft_byte_identity": p_zero, "recount_errors": recount}
    
            outcomes = {name: {i: bool(rows[i]["final_joint"]) for i in ids} for name, rows in cells.items()}
            rates = {name: sum(values.values()) / 240 for name, values in outcomes.items()}
            for planner, left, right in (("llama31_8b", "llama_llama", "llama_peer"), (peer, "peer_llama", "peer_peer")):
                result = mcnemar(outcomes[left], outcomes[right], ids)
                result.update({"pair": peer, "held_planner": planner, "left_controller": "llama31_8b", "right_controller": peer})
                controller_tests.append(result)
            for controller, left, right in (("llama31_8b", "llama_llama", "peer_llama"), (peer, "llama_peer", "peer_peer")):
                result = mcnemar(outcomes[left], outcomes[right], ids)
                result.update({"pair": peer, "held_controller": controller, "left_planner": "llama31_8b", "right_planner": peer})
                planner_tests.append(result)
    
            conditional = {}
            for planner, left, right in (("llama31_8b", "llama_llama", "llama_peer"), (peer, "peer_llama", "peer_peer")):
                invoked = [i for i in ids if not cells[left][i]["rounds"][0]["joint_success"]]
                conditional[planner] = mcnemar(outcomes[left], outcomes[right], invoked)
            report["pairs"][peer] = {"rates": rates, "controller_effect_given_round0_failure": conditional}
            for planner in ("llama31_8b", peer):
                for controller in ("llama31_8b", peer):
                    key = ("llama" if planner == "llama31_8b" else "peer") + "_" + ("llama" if controller == "llama31_8b" else "peer")
                    matrices.append({"pair": peer, "planner": planner, "controller": controller, "final_joint_n": int(rates[key] * 240), "final_joint_rate": rates[key]})
    
        holm(controller_tests)
        report["controller_tests"] = controller_tests
        report["planner_negative_controls"] = planner_tests
        report["integrity"] = integrity
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output_dir / "integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
        for name, rows in (("crossover_matrices.csv", matrices), ("controller_tests.csv", controller_tests), ("planner_tests.csv", planner_tests)):
            with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        if not integrity["passes"]:
            raise SystemExit(2)
    
        return report
