"""Download or verify the frozen annotation resources; never annotate or generate."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
NLI_REPO='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
NLI_REVISION='b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'
NLI_CONFIG_SHA256='7f9d420b616691c5b575bab8839719aefbddc656d8e1316519414b097ce43e3d'

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(stanza_root,nli_root):
    manifest=json.loads((ROOT/'configs/annotation_resources.json').read_text())
    errors=[]
    for item in manifest['files']:
        if item['path'].endswith('.zip'):continue # Download cache, not a runtime input.
        p=stanza_root/item['path']
        if not p.is_file() or sha256(p)!=item['sha256']:errors.append(item['path'])
    p=nli_root/'config.json'
    if not p.is_file() or sha256(p)!=NLI_CONFIG_SHA256:errors.append('NLI config.json')
    if not (nli_root/'model.safetensors').is_file():errors.append('NLI model.safetensors')
    if errors:raise SystemExit('Resource verification failed: '+', '.join(errors))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--download',action='store_true',help='Download upstream model resources; otherwise only verify existing files')
    args=parser.parse_args();destination=args.output.resolve()
    stanza_root=destination/'stanza';nli_root=destination/'nli'
    if args.download:
        import stanza
        from huggingface_hub import snapshot_download
        if stanza.__version__!='1.14.0':raise SystemExit('Install requirements-annotation.txt first (Stanza 1.14.0 required)')
        stanza_root.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'configs/stanza_resources_frozen.json',stanza_root/'resources.json')
        stanza.download('en',model_dir=str(stanza_root),package='default',resources_version='1.14.0',download_json=False)
        snapshot_download(NLI_REPO,revision=NLI_REVISION,local_dir=str(nli_root))
    verify(stanza_root,nli_root)
    print(json.dumps({'annotations':{'stanza_resources_dir':str(stanza_root),'nli_model_path':str(nli_root)}},indent=2))

if __name__=='__main__':main()
