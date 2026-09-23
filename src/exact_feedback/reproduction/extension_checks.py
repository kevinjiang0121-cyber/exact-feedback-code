"""Coverage checks for experiments added after the baseline reproduction release."""
from pathlib import Path
import json


def check_extensions(runs: Path) -> list[str]:
    failures=[]
    llama="llama31_8b";peers=("qwen3_14b","gemma2_9b","glm4_9b")
    for model in (llama,*peers):
        sources=(llama,*peers) if model==llama else (llama,model)
        for source in sources:
            path=runs/"fixed_draft_crossover"/model/f"{source}_planner__{model}_controller/cases.jsonl"
            if not path.is_file():failures.append(f"Missing crossover cell: {source} -> {model}");continue
            rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(rows)!=240 or len({r['id'] for r in rows})!=240:failures.append(f"Incomplete crossover cell: {source} -> {model}")
    for family,prefix in [("qwen","q"),("gemma","g")]:
        root=runs/f"{family}_aligned"/"paired"
        for rel in ["1_residual/analysis/residual_probe_analysis.json","2_restoration_discovery/analysis/restoration_discovery.json","3_dose_confirmation/analysis/dose_confirmation.json","4_activation/discovery/analysis.json","5_graft/discovery/analysis.json"]:
            path=root/(prefix+rel)
            if not path.is_file():failures.append(f"Missing {family} artifact: {rel}");continue
            value=json.loads(path.read_text(encoding="utf-8"))
            integrity=value.get("integrity",{})
            if integrity.get("passes") is False:failures.append(f"Failed {family} integrity: {rel}")
        # File presence is necessary, not a certificate that all scientific
        # conditions match. Native statistical analyses enforce their row gates.
    return failures
