"""Dependency-ordered offline statistics over the independent evidence package."""
from dataclasses import dataclass
from pathlib import Path
import json,sys,subprocess
from .config import python_environment

@dataclass(frozen=True)
class AnalysisJob:
    key: str
    argv: tuple[str,...]
    dependencies: tuple[str,...]=()

@dataclass
class EvidenceAnalysis:
    package: Path
    materials: Path
    output: Path

    def plan(self):
        p=self.package.resolve();m=self.materials.resolve();o=self.output.resolve();e=m/'experiments';jobs=[]
        def add(key,module,*args,deps=()):
            jobs.append(AnalysisJob(key,tuple(map(str,[sys.executable,p/'src/exact_feedback'/module,*args])),deps))
        add('response_discovery','response_law/analysis.py','--phase','discovery','--experiment-dir',e/'feedback_policy_system_identification_v1/discovery','--output-dir',o/'response')
        add('response_confirmation','response_law/analysis.py','--phase','confirmation','--experiment-dir',e/'feedback_policy_system_identification_v1/confirmation','--output-dir',o/'response','--prediction-spec',o/'response/frozen_prediction_spec.json',deps=('response_discovery',))
        add('response_qwen32','response_law/qwen32_analysis.py','--experiment-dir',e/'feedback_policy_system_identification_v1','--prediction-spec',o/'response/frozen_prediction_spec.json','--output',o/'response/qwen32.json',deps=('response_discovery',))
        add('natural_prediction','response_law/transfer_analysis.py','--root',m,'--prediction-spec',o/'response/frozen_prediction_spec.json','--discovery-analysis',o/'response/discovery_analysis.json','--bootstraps','10000','--output-dir',o/'natural_prediction',deps=('response_discovery',))
        add('round_budget','closed_loop/budget_analysis.py','--root',m,'--output',o/'round_budget')
        add('trajectory_diagnostics','closed_loop/trajectory_diagnostics.py','--root',m,'--output',o/'trajectory_diagnostics')
        add('prompt_robustness','robustness/prompt_analysis.py','--experiment-root',e/'prompt_robustness60_v1','--output-dir',o/'prompt')
        add('decoding_robustness','robustness/decoding_analysis.py','--root',e/'decoding_robustness60_v1','--prompt-analysis',o/'prompt/prompt_robustness_analysis.json','--output-dir',o/'decoding',deps=('prompt_robustness',))
        add('recurrence','recurrence/unified_analysis.py','--root',m,'--output-dir',o/'recurrence','--bootstraps','10000','--permutations','100000')
        add('capture','recurrence/capture_analysis.py','--root',m,'--output-dir',o/'capture','--bootstraps','10000','--permutations','100000')
        add('capture_api','recurrence/api_analysis.py','--root',m,'--open-hazards',o/'capture/early_hazards.csv','--output-dir',o/'capture_api','--permutations','100000',deps=('capture',))
        add('conditional_rescue','recurrence/rescue_analysis.py','--root',m,'--output-dir',o/'rescue')
        add('three_domain','recurrence/three_domain_analysis.py','--root',m,'--output-dir',o/'three_domain','--bootstraps','10000','--legacy-analysis',o/'capture/analysis.json',deps=('capture',))
        add('api_interaction','closed_loop/api_analysis.py','--project-root',m,'--output-dir',o/'api_interaction','--draws','20000')
        add('persistence','persistence/analysis.py','--selection-dir',m/'data/persistence_revision32_v1','--run-root',e/'persistence_revision32_v1','--output-dir',o/'persistence')
        add('history_exact','history/exact_analysis.py','--experiment-dir',e/'history_reset_closed_loop_state_matched_v1','--output-dir',o/'history_exact','--iterations','20000')
        add('history_structured','history/structured_analysis.py','--experiment-dir',e/'targeted_history_reset_confirmation_v1','--output-dir',o/'history_structured','--iterations','20000')
        add('history_partial','history/partial_analysis.py','--pilot-root',e/'glm_partial_reset_matched_pilot_v1','--output-dir',o/'history_partial','--iterations','20000')
        a=e/'model_generality_combined480_api_all7_audit_v1'
        add('endpoint_analysis','content_contract/endpoint_analysis.py','--registry',a/'endpoint_registry/text_registry.jsonl','--trajectories',a/'endpoint_registry/trajectories.jsonl','--stanza',a/'annotations/stanza_1_14_0/annotations.jsonl','--nli',a/'annotations/deberta_v3_large_nli_b3546ea/annotations.jsonl','--output-dir',o/'endpoint_analysis','--bootstrap','20000')
        add('content_contract','content_contract/sensitivity_analysis.py','--input',o/'endpoint_analysis/endpoint_trajectory_metrics.csv','--output-dir',o/'content_contract',deps=('endpoint_analysis',))
        add('annotation_integrity','content_contract/annotation_audit.py',
            '--registry',a/'endpoint_registry/text_registry.jsonl','--trajectories',a/'endpoint_registry/trajectories.jsonl',
            '--stanza',a/'annotations/stanza_1_14_0/annotations.jsonl','--stanza-manifest',a/'annotations/stanza_1_14_0/manifest.json',
            '--nli',a/'annotations/deberta_v3_large_nli_b3546ea/annotations.jsonl','--nli-manifest',a/'annotations/deberta_v3_large_nli_b3546ea/manifest.json',
            '--analysis',a/'endpoint_analysis/endpoint_annotation_analysis.json','--output',o/'annotation_integrity.json')
        add('stage_interface','aligned_models/analysis.py','--project-root',m,'--output-dir',o/'stage_interface','--draws','20000')
        l=e/'phase3_causal_localization_v1'
        add('llama_probes','localization/probe_analysis.py','--states',m/'data/phase3a_fixed_states_v1/states.jsonl','--results-root',l/'phase3a_fixed_states','--output-dir',o/'llama_probes')
        add('llama_restoration','localization/restoration_analysis.py','--root',l/'phase3b_restoration_discovery120','--output-dir',o/'llama_restoration')
        add('llama_dose','localization/restoration_dose_analysis.py','--discovery-root',l/'phase3b_restoration_discovery120','--root',l/'phase3b_dose_confirmation_v1','--output',o/'llama_dose.json')
        add('llama_graft','localization/graft_analysis.py','--root',l/'phase3d_controller_graft_discovery_v1','--output',o/'llama_graft.json')
        add('llama_activation','localization/activation_analysis.py','--base-clean',l/'phase3a_fixed_states/base/generation/cases.jsonl','--instruct-clean',l/'phase3a_fixed_states/instruct/generation/cases.jsonl','--patched',l/'phase3c_activation_mediation_discovery_v1/patched_states.jsonl','--output',o/'llama_activation.json')
        for family in ['qwen','gemma']:
            jobs.append(AnalysisJob(family+'_aligned',tuple(map(str,[sys.executable,p/'reproduce.py','analyze-experiment',family+'_aligned','--runs-root',e/(family+'_aligned_mechanism_v1'),'--output',o/(family+'_aligned')]))))
        for key,source in [('fixed_draft_crossover',m),('failure_categories',m),('history_trigger',e/'targeted_history_reset_confirmation_v1')]:
            jobs.append(AnalysisJob(key,tuple(map(str,[sys.executable,p/'reproduce.py','analyze-experiment',key,'--runs-root',source,'--output',o/key]))))
        profile_path=m/'analysis_profile.json'
        if profile_path.is_file():
            profile=json.loads(profile_path.read_text(encoding='utf-8'))
            if profile.get('profile') != 'submission_20260916':
                raise ValueError('Unknown evidence analysis profile')
            jobs=[j for j in jobs if j.key not in profile['requires_full_evidence']]
            for family in ['qwen','gemma']:
                jobs.append(AnalysisJob(family+'_interventions',tuple(map(str,[sys.executable,p/'reproduce.py','analyze-experiment',family+'_aligned','--runs-root',e/(family+'_aligned_mechanism_v1'),'--output',o/(family+'_interventions'),'--stages','restoration','dose','activation','graft']))))
        return jobs

    def run(self,selected=None,dry_run=False,workers=1):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock
        source=self.materials.resolve();out=self.output.resolve()
        if source==out or source in out.parents:raise ValueError('Analysis output must be outside evidence package')
        jobs={j.key:j for j in self.plan()};wanted=set(selected or jobs)
        if wanted-set(jobs):raise ValueError('Unknown analyses: '+repr(wanted-set(jobs)))
        def include(key):
            for dep in jobs[key].dependencies:wanted.add(dep);include(dep)
        for key in list(wanted):include(key)
        out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
        results={};env=python_environment(self.package);env.update(PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='')
        lock=Lock();futures={}
        def perform(key,job):
            if any(futures[dep].result()['status'] not in {'passed','planned'} for dep in job.dependencies):
                value={'status':'dependency_failed'}
                with lock:
                    results[key]=value
                    (out/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
                return value
            print(('PLAN ' if dry_run else 'ANALYZE ')+key,flush=True)
            if dry_run:value={'status':'planned','argv':list(job.argv)}
            else:
                log=out/'logs'/f'{key}.log'
                with log.open('w',encoding='utf-8') as f:r=subprocess.run(job.argv,cwd=self.package,env=env,stdout=f,stderr=subprocess.STDOUT)
                value={'status':'passed' if r.returncode==0 else 'failed','returncode':r.returncode,'log':'logs/'+log.name}
            with lock:
                results[key]=value
                (out/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            return value
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key,job in jobs.items():
                if key in wanted:futures[key]=pool.submit(perform,key,job)
            for future in futures.values():future.result()
        return results
