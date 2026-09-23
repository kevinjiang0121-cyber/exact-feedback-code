"""Frozen exact-length diagnostics, separated from figure generation.

The trajectory classifier is preserved from the original API audit implementation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from collections import Counter
import argparse,csv,hashlib,json,math
from exact_feedback.closed_loop.budget_analysis import OPEN_MODELS,API_MODELS,HISTORICAL_EXACT,EXTRA_EXACT,iter_jsonl

def trajectory(row: dict[str, Any], model: str) -> dict[str, Any]:
    rounds = row.get("rounds") or []
    if not rounds:
        return {
            "model": model,
            "id": row["id"],
            "source": row["source"],
            "length_band": row["length_band"],
            "protocol_complete": False,
            "provider_refusal": True,
            "final_joint": False,
            "failure_family": "provider_refusal",
        }
    errors = [int(item["error"]) for item in rounds]
    abs_errors = [abs(value) for value in errors]
    nonzero_signs = [1 if value > 0 else -1 for value in errors if value != 0]
    sign_flips = sum(a != b for a, b in zip(nonzero_signs, nonzero_signs[1:]))
    text_hashes = [
        hashlib.sha256(item["text"].encode("utf-8")).hexdigest() for item in rounds
    ]
    recurrent = len(text_hashes) != len(set(text_hashes))
    transitions = list(zip(errors, errors[1:]))
    stationary = any(left == right for left, right in transitions)
    exact_rounds = [i for i, item in enumerate(rounds) if item["exact_length"]]
    final = rounds[-1]
    if row["final_joint"]:
        family = "success"
    elif final["exact_length"] and final["missing"]:
        family = "exact_anchor_failure"
    elif recurrent:
        family = "recurrent_text"
    elif sign_flips >= 2:
        family = "oscillatory"
    elif stationary:
        family = "stationary_error"
    elif min(abs_errors) <= 2:
        family = "near_target_no_capture"
    elif abs_errors[-1] < abs_errors[0]:
        family = "contracted_no_capture"
    else:
        family = "noncontracting_other"
    return {
        "model": model,
        "id": row["id"],
        "source": row["source"],
        "length_band": row["length_band"],
        "protocol_complete": bool(row.get("protocol_complete", True)),
        "provider_refusal": False,
        "one_shot_joint": bool(row["one_shot_joint"]),
        "final_joint": bool(row["final_joint"]),
        "revisions": len(rounds) - 1,
        "initial_error": errors[0],
        "final_error": errors[-1],
        "initial_abs_error": abs_errors[0],
        "final_abs_error": abs_errors[-1],
        "best_abs_error": min(abs_errors),
        "contracted": abs_errors[-1] < abs_errors[0],
        "ever_exact": bool(exact_rounds),
        "lost_exact": bool(exact_rounds and not final["exact_length"]),
        "sign_flips": sign_flips,
        "oscillation": sign_flips >= 2,
        "recurrent_text": recurrent,
        "stationary_error": stationary,
        "ever_anchor_loss": any(bool(item["missing"]) for item in rounds),
        "final_anchor_loss": bool(final["missing"]),
        "failure_family": family,
        "error_sequence": errors,
    }

class TrajectoryDiagnostics:
    def analyze(self, root: Path, output: Path):
        rows=[];inputs=[]
        for model in (*OPEN_MODELS,*API_MODELS):
            if model in API_MODELS:
                paths=[root/'experiments/model_generality_combined480_api_v1'/model/model/'baseline/cases.jsonl']
            elif model in HISTORICAL_EXACT:
                paths=[root/'experiments'/e/model/'cases.jsonl' for e in ['human_generation_main120_v1','human_generation_replication120_v1','human_generation_extension240_v1']]
            else:paths=[root/'experiments/model_generality_combined480_v1'/EXTRA_EXACT[model]]
            seen=set()
            for path in paths:
                inputs.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                for _,raw in iter_jsonl(path):
                    if raw['id'] in seen:raise ValueError(f'Duplicate case: {model}')
                    seen.add(raw['id'])
                    row=trajectory(raw,model);row['panel']='api' if model in API_MODELS else 'local';rows.append(row)
        complete=[r for r in rows if r['protocol_complete']]
        def summarize(group):
            failed=[r for r in group if not r['final_joint']]
            at_risk=len(group);rounds=[]
            for revision in range(9):
                captures=sum(r['final_joint'] and r['revisions']==revision for r in group)
                rounds.append({'revision':revision,'at_risk':at_risk,'captures':captures,'conditional_capture':captures/at_risk if at_risk else None,'cumulative_capture':sum(r['final_joint'] and r['revisions']<=revision for r in group)/len(group) if group else None})
                at_risk-=captures
            gaps=Counter()
            for r in failed:
                d=r['final_abs_error']-r['best_abs_error']
                label=str(d) if d<=2 else '3-5' if d<=5 else '6-10' if d<=10 else '11-20' if d<=20 else '>20'
                gaps[label]+=1
            return {'complete':len(group),'failures':len(failed),'failure_categories':dict(Counter(r['failure_family'] for r in failed)),
                    'lost_exact':sum(r['lost_exact'] for r in group),'best_at_terminal':sum(r['best_abs_error']==r['final_abs_error'] for r in failed),
                    'final_minus_best_bins':dict(gaps),'rounds':rounds}
        report={'panels':{p:summarize([r for r in complete if p=='pooled' or r['panel']==p]) for p in ['api','local','pooled']},
                'models':{m:summarize([r for r in complete if r['model']==m]) for m in (*OPEN_MODELS,*API_MODELS)},'inputs':inputs,
                'boundary':'Protocol-complete records; not the all-model common subset. Lost exact means leaving exact word count, not losing joint success.'}
        output.mkdir(parents=True,exist_ok=True)
        (output/'analysis.json').write_text(json.dumps(report,indent=2))
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (output/'trajectory_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=TrajectoryDiagnostics().analyze(a.root.resolve(),a.output.resolve())
    print(json.dumps({k:{f:v[f] for f in ['complete','failures','lost_exact','best_at_terminal']} for k,v in report['panels'].items()},indent=2))

if __name__=='__main__':main()
