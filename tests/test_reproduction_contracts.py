"""CPU-only regression checks; synthetic fixtures are not experimental evidence."""
from pathlib import Path
import contextlib,hashlib,io,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from exact_feedback.closed_loop.budget_analysis import API_MODELS,ASSAYS,load_cells
from exact_feedback.reproduction.registry import EXPERIMENTS
from exact_feedback.reproduction.planner import command
from exact_feedback.reproduction.cli import validate
from exact_feedback.reproduction.artifacts import ArtifactStore

class ReproductionContracts(unittest.TestCase):
    def test_api_budget_uses_all_model_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in API_MODELS:
                for assay in ASSAYS:
                    p=root/'experiments'/('model_generality_combined480_api_v1' if assay=='exact_length' else 'model_generality_multidomain_api_v1')
                    p=p/model/model/'baseline/cases.jsonl' if assay=='exact_length' else p/assay/model/'baseline/cases.jsonl'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    rows=[{'id':str(i),'rounds':[{'revision':0,'joint_success':True}]} for i in range(3)]
                    if assay=='exact_length' and model==API_MODELS[0]:rows[0].update(protocol_complete=False)
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            cells,_=load_cells(root,'api',frozen=False)
            self.assertTrue(all(len(cells['api',m,'exact_length'])==2 for m in API_MODELS))
            with self.assertRaises(ValueError):load_cells(root,'api',frozen=True)

    def test_llama_templates_and_discovery_split(self):
        cfg=json.loads((ROOT/'config.example.json').read_text());cfg['_path']=str(ROOT/'config.example.json')
        for spec in cfg['models'].values():spec['path']='models/testing-only'
        with tempfile.TemporaryDirectory() as tmp:
            for key in ['parameter_restoration','controller_graft','activation_transplant','residual_probe_capture','response_law']:
                model='llama31_8b_base' if key in ['response_law','residual_probe_capture'] else EXPERIMENTS[key].models[0]
                steps=command(ROOT,cfg,EXPERIMENTS[key],model,None,Path(tmp)/key)
                for step in steps:
                    if '--chat-template' in step:
                        p=Path(step[step.index('--chat-template')+1])
                        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65')
                    if key in ['parameter_restoration','controller_graft'] and Path(step[1]).name=='probe_runner.py':
                        self.assertEqual(step[step.index('--split')+1],'discovery')

    def test_preflight_rejects_empty_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'models':{'llama31_8b':{'path':tmp}}}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validate(cfg,['closed_loop_exact'],'llama31_8b'),1)

    def test_inventory_rejects_traversal_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=ArtifactStore(root)
            with self.assertRaises(ValueError):store.path('../outside')
            (root/'a').write_bytes(b'altered')
            (root/'inventory.json').write_text(json.dumps([{'path':'a','bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}]))
            self.assertFalse(store.verify()['integrity_pass'])

if __name__=='__main__':unittest.main()
