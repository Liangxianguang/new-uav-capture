"""All-population diagnostic only; no optimizer, training or deployment permit."""
import argparse
import json
import sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'cwm_v36'),str(HERE.parent/'cwm_v35'),str(HERE.parent/'cwm_v28')]
import coordinate_audit
import cost_runtime
from audited_stream import AuditedCorpus,sha
from train_ranking_response import frozen_configs,save_json

def run(data,audit,output):
    if output.exists(): raise ValueError('Exclusive new diagnostic output required')
    output.mkdir(exist_ok=False)
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in HERE.glob('*.py')}
    save_json(output/'diagnostic_contract.json',dict(scope='ALL complete BASE calls at GRU and CV; diagnostic only',
        sources=sources,data_summary_sha256=sha(data/'summary.json'),audit_summary_sha256=sha(audit/'summary.json'),
        prior_failure_preserved=True,optimizer_allowed=False,enhanced_control_enabled=False,holdout_used=False))
    corpus=AuditedCorpus(data,audit)
    configs=frozen_configs(ROOT/'experiments/cwm_v1/baseline/capsule.zip',output/'r')
    # Local process alias only, no V23/V36 file or existing report mutation.
    cost_runtime.audit_deployed_score=coordinate_audit.audit_deployed_score
    certificates=cost_runtime.audit_public_origins(corpus,configs,output)
    cost_runtime.require_motion_permit(corpus,certificates,output)
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest: raise ValueError('Diagnostic source changed during run')
    result=dict(status='complete_native_coordinate_diagnostic_not_training_qualification',certificates=certificates,
        scope='New verification proposal only; not a published training/reload/archive contract',
        optimizer_run=False,enhanced_control_enabled=False,holdout_used=False)
    save_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','audit','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:run(args.data,args.audit,args.output)
    except Exception as error:
        if args.output.is_dir() and not (args.output/'summary.json').exists():
            save_json(args.output/'failure.json',dict(error=str(error),optimizer_run=False,enhanced_control_enabled=False))
        raise
