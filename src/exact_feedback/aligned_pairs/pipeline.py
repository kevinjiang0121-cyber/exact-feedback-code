from pathlib import Path
import argparse
from exact_feedback.reproduction.config import load_config
from exact_feedback.aligned_pairs.experiment import AlignedPairExperiment

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['qwen','gemma'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,help='Offline analysis of existing frozen results; never generates')
    p.add_argument('--stages',nargs='+',choices=['probes','restoration','dose','activation','graft'],default=['probes','restoration','dose','activation','graft'])
    a=p.parse_args();root=Path(__file__).resolve().parents[3]
    cfg={} if a.source else load_config(root,a.config)
    AlignedPairExperiment(root,cfg,a.family,a.output,a.source).run(tuple(a.stages))

if __name__=='__main__':main()
