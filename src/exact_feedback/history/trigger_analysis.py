"""Retrospective trigger sensitivity; no new inference or model calls."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np

class TriggerAnalyzer:
    def analyze(self, source: Path, output: Path):
        def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        states=rows(source/'states.jsonl');lookup={s['state_id']:s for s in states}
        if len(lookup)!=len(states):raise ValueError('Duplicate state IDs')
        pairs=defaultdict(dict)
        for model in sorted({s['model'] for s in states}):
            for r in rows(source/model/'cases.jsonl'):
                if r['state_id'] not in lookup or r['condition'] in pairs[r['state_id']]:raise ValueError('Invalid or duplicate arm')
                pairs[r['state_id']][r['condition']]=r
        if set(pairs)!=set(lookup) or any(set(p)!={'full_history','history_reset'} for p in pairs.values()):raise ValueError('Incomplete pairing')
        cells=[];first=[]
        for model,structure in sorted({(s['model'],s['structure']) for s in states}):
            selected=[s for s in states if (s['model'],s['structure'])==(model,structure)]
            cell=dict(model=model,structure=structure)
            for tag,group in [('all',selected),('later',[s for s in selected if s['revision']>1])]:
                cell[tag+'_n']=len(group)
                for metric in ['first_step_escape','final_joint']:
                    rates={arm:sum(bool(pairs[s['state_id']][arm][metric]) for s in group)/len(group) if group else None for arm in ['full_history','history_reset']}
                    cell[tag+'_'+metric]={'full':rates['full_history'],'reset':rates['history_reset'],'delta':rates['history_reset']-rates['full_history'] if group else None}
                    if group:
                        rng=np.random.default_rng(20260908)
                        estimates=np.zeros(5000)
                        for origin in sorted({s['source'] for s in group}):
                            differences=np.array([float(pairs[s['state_id']]['history_reset'][metric])-float(pairs[s['state_id']]['full_history'][metric]) for s in group if s['source']==origin])
                            estimates+=rng.choice(differences,size=(5000,len(differences))).sum(axis=1)/len(group)
                        cell[tag+'_'+metric]['descriptive_ci95']=np.quantile(estimates,[.025,.975]).tolist()
            cells.append(cell)
        for s in states:
            if s['revision']==1:
                # Reset replays the prompt that produced the current failing draft.
                prior=s['full_messages'][:-2]
                replay=s['reset_messages']==prior
                r=pairs[s['state_id']]['history_reset']
                first.append(dict(state_id=s['state_id'],prompt_replayed=replay,text_repeated=r['rounds'][0]['text']==s['current_text']))
        report=dict(status='retrospective_descriptive',states=len(states),first_revision_states=len(first),replayed_prompts=sum(x['prompt_replayed'] for x in first),repeated_trigger_texts=sum(x['text_repeated'] for x in first),cells=cells)
        report['bootstrap']={'draws':5000,'seed':20260908,'unit':'paired state, stratified by source','boundary':'Retrospective subset; not a randomized timing effect or mediation estimate.'}
        output.mkdir(parents=True,exist_ok=True)
        (output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report
