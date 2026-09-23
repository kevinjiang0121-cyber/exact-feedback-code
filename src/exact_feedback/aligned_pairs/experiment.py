"""Portable Qwen/Gemma pipelines; selection is computed from discovery outputs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import sys
from exact_feedback.reproduction.config import model_path, python_environment


@dataclass
class AlignedPairExperiment:
    root: Path
    config: dict
    family: str
    output: Path
    source: Path | None = None

    def __post_init__(self):
        self.root = self.root.resolve()
        self.output = self.output.resolve()
        if self.source is not None:
            self.source = self.source.resolve()
            if self.output == self.source or self.output.is_relative_to(self.source):
                raise ValueError("Write recomputed analysis outside the frozen evidence directory")
        if self.family not in {"qwen", "gemma"}:
            raise ValueError("Expected qwen or gemma")
        self.prefix = "q" if self.family == "qwen" else "g"
        self.inputs = self.source or self.output
        self.states = self.root / "data/phase3a_fixed_states_v1/states.jsonl"
        self.template = self.inputs / f"{self.prefix}0_parity" / ("qwen3_instruct_chat_template.jinja" if self.family == "qwen" else "gemma2_it_chat_template.jinja")
        self.models = self.output / "derived_checkpoints"
        self.group_spec = self.root / "configs/gemma_aligned_groups_v1.json"

    def stage(self, number: int, suffix: str) -> Path:
        return self.inputs / f"{self.prefix}{number}_{suffix}"

    def analysis_path(self, number: int, suffix: str, name: str) -> Path:
        return self.output / f"{self.prefix}{number}_{suffix}" / name

    def invoke(self, engine: str, *args) -> None:
        analysis = engine.startswith("analyze_")
        if self.source and not analysis:
            return
        argv = [self.config.get("python", sys.executable), str(self.root / "src/exact_feedback/aligned_pairs/engines" / (engine + ".py")), *map(str, args)]
        env = python_environment(self.root)
        env["TOKENIZERS_PARALLELISM"] = "false"
        if self.family == "gemma":
            env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        subprocess.run(argv, cwd=self.root, env=env, check=True)

    def materialize(self, conditions: list[str]) -> None:
        if self.source:
            return
        base = "qwen3_8b_base" if self.family == "qwen" else "gemma2_9b_base"
        instruct = "qwen3_8b" if self.family == "qwen" else "gemma2_9b"
        extra = ["--group-spec", self.group_spec] if self.family == "gemma" else []
        self.invoke("materialize_qwen_aligned_checkpoints", "--base", model_path(self.root,self.config,base), "--instruct", model_path(self.root,self.config,instruct), "--output-root", self.models, *extra, "--conditions", *conditions)

    def fixed(self, model: str, split: str, destination: Path) -> None:
        self.invoke("run_phase3a_fixed_state_generation_vllm", "--model", self.models/model,
                    "--model-slug",model,"--states",self.states,"--split",split,
                    "--chat-template",self.template,"--output-dir",destination,
                    "--dtype","bfloat16","--gpu-memory-utilization",.90,
                    "--max-model-len",8192,"--max-num-seqs",32)

    def closed(self, model: str, combined: bool, destination: Path) -> None:
        data = self.root / ("data/human_generation_combined480_v1/generation_combined480.jsonl" if combined else "data/human_generation_main120_v1/generation_main120.jsonl")
        extra = ["--enforce-eager","--no-enable-prefix-caching"] if self.family == "gemma" else []
        self.invoke("run_human_generation_loop_vllm","--model",self.models/model,
                    "--data",data,"--output-dir",destination,"--chat-template-file",self.template,
                    "--max-revisions",8,"--dtype","bfloat16","--gpu-memory-utilization",.90,
                    "--max-model-len",32768,"--max-num-seqs",32,*extra)
        self.invoke("audit_local_generation_row","--frozen-data",data,"--cases",destination/"cases.jsonl","--output",destination/"integrity_audit.json")
        if not self.source:
            verdict=json.loads((destination/"integrity_audit.json").read_text(encoding="utf-8"))
            if verdict.get("verdict") != "PASS":raise ValueError("Closed-loop integrity audit failed")

    def prepare(self) -> None:
        if self.source:return
        base="qwen3_8b_base" if self.family=="qwen" else "gemma2_9b_base"
        instruct="qwen3_8b" if self.family=="qwen" else "gemma2_9b"
        bp=model_path(self.root,self.config,base);ip=model_path(self.root,self.config,instruct)
        template=json.loads((ip/"tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
        if not isinstance(template,str):raise ValueError("Expected the frozen single Instruct template")
        self.template.parent.mkdir(parents=True,exist_ok=True);self.template.write_text(template,encoding="utf-8")
        extra=["--expected-layers",42,"--expected-hidden-size",3584,"--expected-states",288,"--tokenizer-files","tokenizer.model"] if self.family=="gemma" else []
        audit=self.template.parent/"parity_audit.json"
        self.invoke("audit_qwen_aligned_pair","--base",bp,"--instruct",ip,"--states",self.states,"--output",audit,*extra)
        if json.loads(audit.read_text(encoding="utf-8")).get("verdict")!="PASS":raise ValueError("Pair parity failed")
        self.materialize(["base_common_interface","instruct_sham"])

    def probes(self) -> None:
        root=self.stage(1,"residual")
        for name,condition in [("base","base_common_interface"),("instruct","instruct_sham")]:
            self.invoke("capture_phase3a_fixed_states","--model",self.models/condition,"--model-slug",name,
                        "--states",self.states,"--chat-template",self.template,"--output-dir",root/name/"capture","--dtype","bfloat16")
        extra=["--expected-layers",42,"--expected-hidden-size",3584,"--protocol","docs/GEMMA_ALIGNED_MECHANISM_PROTOCOL_20260903.md","--seed",20260903] if self.family=="gemma" else []
        self.invoke("analyze_qwen_aligned_residual_probes","--states",self.states,"--results-root",root,
                    "--output",self.analysis_path(1,"residual","analysis/residual_probe_analysis.json"),*extra)

    def restoration(self) -> None:
        groups=list(json.loads(self.group_spec.read_text())["groups"]) if self.family=="gemma" else ["blocks_00_08","blocks_09_17","blocks_18_26","blocks_27_35","output"]
        groups=[g for g in groups if g!="noncontiguous"]
        conditions=["instruct_sham","restore_noncontiguous",*["restore_"+g for g in groups]]
        if self.family == "gemma":
            conditions.insert(0,"base_common_interface")
        self.materialize(conditions);root=self.stage(2,"restoration_discovery")
        for condition in conditions:
            self.fixed(condition,"discovery",root/condition/"fixed_state")
            self.closed(condition,False,root/condition/"closed_loop")
        extra=["--group-spec",self.group_spec] if self.family=="gemma" else []
        self.invoke(f"analyze_{self.family}_restoration_discovery","--root",root,"--output",self.analysis_path(2,"restoration_discovery","analysis/restoration_discovery.json"),*extra)

    def dose(self) -> None:
        discovery=self.analysis_path(2,"restoration_discovery","analysis/restoration_discovery.json")
        groups=json.loads(discovery.read_text(encoding="utf-8"))["selected_at_most_two"]
        root=self.stage(3,"dose_confirmation")
        if groups:
            conditions=["instruct_sham","restore_noncontiguous"]+[prefix+g for g in groups for prefix in ["restore50_","restore_"]]
            self.materialize(conditions)
            for condition in conditions:
                self.fixed(condition,"confirmation",root/condition/"fixed_state_confirmation")
                self.closed(condition,True,root/condition/"combined480")
        self.invoke(f"analyze_{self.family}_dose_confirmation","--discovery-analysis",discovery,"--root",root,"--output",self.analysis_path(3,"dose_confirmation","analysis/dose_confirmation.json"))

    def transfer(self, kind: str) -> None:
        dose=self.analysis_path(3,"dose_confirmation","analysis/dose_confirmation.json")
        discovery=self.analysis_path(2,"restoration_discovery","analysis/restoration_discovery.json")
        groups=json.loads(dose.read_text(encoding="utf-8"))["confirmed_groups"]
        if kind=="graft" or self.family=="gemma":groups=[g for g in groups if g!="output"]
        if not groups:
            print(f"{kind}: no eligible groups under frozen selection rule")
            number=4 if kind=='activation' else 5
            target=self.analysis_path(number,kind,'discovery/analysis.json')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps({'status':'not_eligible','reason':'No groups passed the prespecified upstream selection rule.','confirmed_groups':groups},indent=2))
            return
        number=4 if kind=="activation" else 5
        root=self.stage(number,kind);split="discovery"
        extra=["--discovery-analysis",discovery] if self.family=="gemma" else []
        if kind=="activation":
            for name,condition in [("base","base_common_interface"),("instruct","instruct_sham")]:self.fixed(condition,split,root/"clean"/split/name)
            units=["output_boundary_qwen" if g=="output" else g for g in groups]
            self.invoke("run_phase3c_activation_mediation","--base",self.models/"base_common_interface","--instruct",self.models/"instruct_sham","--states",self.states,"--chat-template",self.template,"--split",split,"--output",root/split/"patched_states.jsonl","--units",",".join(units),"--dtype","bfloat16","--validate-sham")
            self.invoke(f"analyze_{self.family}_activation_mediation","--base-clean",root/"clean"/split/"base/cases.jsonl","--instruct-clean",root/"clean"/split/"instruct/cases.jsonl","--patched",root/split/"patched_states.jsonl","--dose-analysis",dose,"--split",split,"--groups",",".join(groups),"--output",self.analysis_path(number,kind,"discovery/analysis.json"),*extra)
        else:
            conditions=["base_common_interface","graft_output","graft_noncontiguous_plus_output"]+["graft_"+g+suffix for g in groups for suffix in ["","_plus_output"]]
            self.materialize(conditions)
            for condition in conditions:
                self.fixed(condition,split,root/split/condition/"fixed_state")
                self.closed(condition,False,root/split/condition/"closed_loop")
            self.invoke(f"analyze_{self.family}_controller_graft","--root",root/split,"--dose-analysis",dose,"--split",split,"--groups",",".join(groups),"--output",self.analysis_path(number,kind,"discovery/analysis.json"),*extra)
        # The manuscript reports completed discovery transfer comparisons only.
        # Incomplete Gemma graft confirmation is not an input or a reported result.

    def run(self, stages: tuple[str,...] = ("probes","restoration","dose","activation","graft")) -> None:
        self.output.mkdir(parents=True,exist_ok=True)
        self.prepare()
        for stage in stages:
            print(f"{self.family}: {stage}",flush=True)
            if stage in {"activation","graft"}:self.transfer(stage)
            else:getattr(self,stage)()
