"""Common failure labels, ported from the current manuscript's figure audit."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
from exact_feedback.reproduction.artifacts import ArtifactStore
from exact_feedback.reproduction.registry import MAIN_OPEN, API_PANEL


class FailureAnalyzer:
    @staticmethod
    def category(texts: list[str]) -> str:
        if len(texts)>=3 and texts[-1]==texts[-2]==texts[-3]:return 'fixed'
        if any(len(texts)>=2*k and texts[-2*k:-k]==texts[-k:] and len(set(texts[-k:]))>=2 for k in [2,3,4]):return 'cycle'
        return 'other_repeat' if len(set(texts))<len(texts) else 'no_repeat'

    def analyze(self, materials: Path, output: Path, frozen: bool = True) -> dict:
        store=ArtifactStore(materials);records=[];excluded=[]
        local='experiments/three_domain_confirmation_v1/trajectory_table.csv' if frozen else 'analysis/recomputed/three_domain/trajectory_table.csv'
        with store.path(local).open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                failed=r['final']=='0'
                category='fixed' if r['persistent_fixed']=='1' else 'cycle' if r['persistent_cycle']=='1' else 'other_repeat' if r['any_recurrence']=='1' else 'no_repeat'
                if category in {'fixed','cycle'} and r['any_recurrence']!='1':raise ValueError('Inconsistent local recurrence record')
                records.append(dict(domain=r['domain'],model=r['model'],id=r['case_id'],failed=failed,category=category if failed else 'success'))
        manifest_path='experiments/api_seven_model_paired_interaction_v1_rerun/input_manifest.json' if frozen else 'analysis/recomputed/api_interactions/input_manifest.json'
        manifest=json.loads(store.path(manifest_path).read_text(encoding='utf-8'))
        for entry in manifest:
            # Relocate a recorded provenance path only with its original hash.
            original=entry['path'].replace('\\','/')
            original='/' + original.lstrip('/')
            marker='/experiments/'
            if marker not in original:raise ValueError('Unrecognized provenance path')
            rel='experiments/'+original.split(marker,1)[1];path=store.path(rel)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Provenance hash mismatch: '+rel)
            rows=store.rows(rel)
            if len(rows)!=entry['rows']:raise ValueError('API row count mismatch')
            valid=0
            for row in rows:
                if not row.get('protocol_complete',True) or not row.get('rounds'):
                    excluded.append(dict(domain=entry['assay'],model=entry['model'],id=row['id']));continue
                valid+=1;success=row.get('final_joint',row.get('final_joint_success'))
                if success is None:raise ValueError('Missing success field')
                texts=[r['text'] for r in row['rounds']]
                records.append(dict(domain=entry['assay'],model=entry['model'],id=row['id'],failed=not success,category='success' if success else self.category(texts)))
            if valid!=entry['evaluable']:raise ValueError('Evaluable count mismatch')
        if len({(r['domain'],r['model'],r['id']) for r in records})!=len(records):raise ValueError('Duplicate trajectory identity')
        if frozen and (len(records)!=27359 or len(excluded)!=1):raise ValueError('Frozen complete trajectory matrix mismatch')
        summary={}
        for domain,nlocal,napi in [('exact_length',4622,650),('lexical_constraints',3858,947),('compositional_constraints',2002,91)]:
            selected=[r for r in records if r['domain']==domain];failures=[r for r in selected if r['failed']]
            observed_local=sum(r['model'] in MAIN_OPEN for r in failures);observed_api=sum(r['model'] in API_PANEL for r in failures)
            if frozen and (observed_local!=nlocal or observed_api!=napi):raise ValueError('Failure denominator mismatch')
            for model in (*MAIN_OPEN,*API_PANEL):
                expected=479 if domain=='exact_length' and model=='gemini_3_6_flash_api' else 480
                if frozen and sum(r['model']==model for r in selected)!=expected:raise ValueError('Cell coverage mismatch')
            summary[domain]=dict(complete=len(selected),failures=len(failures),local_failures=observed_local,api_failures=observed_api,counts=dict(Counter(r['category'] for r in failures)),by_model={m:dict(Counter(r['category'] for r in failures if r['model']==m)) for m in (*MAIN_OPEN,*API_PANEL)})
        report=dict(rule='fixed point, then terminal cycle p=2..4, then other exact output repetition, then no repeat; failures only',summary=summary,excluded=excluded)
        output.mkdir(parents=True,exist_ok=True)
        (output/'classification_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        with (output/'classified_trajectories.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        return report
