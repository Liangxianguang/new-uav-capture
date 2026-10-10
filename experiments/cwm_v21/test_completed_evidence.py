"""Semantic checks on completed V21 evidence; fail if training is unfinished."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from loss_training_release import audit,frozen_configs,arrays


def expect_rejection(read,configs,restored,changes,message):
    def altered(name):
        return changes[name] if name in changes else read(name)
    try:
        audit(altered,configs,restored)
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f'Unexpected semantic rejection: {error}') from error
        return
    raise AssertionError('Forged evidence was accepted')


def run(roots,restored):
    torch.set_num_threads(1)
    configs = frozen_configs(ROOT.parent/'cwm_v1/baseline/capsule.zip',restored)
    def read(name):
        stage,relative = name.split('/',1)
        return (roots[stage]/relative).read_bytes()
    positive = audit(read,configs,restored)
    identifier = 'raw_point_l2_seed990101'
    prediction_path = f'primary/{identifier}_development_validation_predictions.npz'
    forecast = arrays(read(prediction_path))
    forecast['call0_response'][0,0,0] += .01
    stream = io.BytesIO()
    np.savez_compressed(stream,**forecast)
    expect_rejection(read,configs,restored,{prediction_path:stream.getvalue()},'forecast or metrics')
    print(json.dumps({'test':'forged_public_forecast_rejected'}),flush=True)
    report = json.loads(read('primary/summary.json'))
    report['primary_research_eligible'] = not report['primary_research_eligible']
    expect_rejection(read,configs,restored,{'primary/summary.json':json.dumps(report).encode()},'qualification/factorial')
    print(json.dumps({'test':'forged_primary_qualification_rejected'}),flush=True)
    report = json.loads(read('primary/summary.json'))
    checkpoint_path = f'primary/{identifier}.pt'
    checkpoint = torch.load(io.BytesIO(read(checkpoint_path)),map_location='cpu',weights_only=True)
    checkpoint['torch_rng_state'][0] ^= 1
    stream = io.BytesIO()
    torch.save(checkpoint,stream)
    raw = stream.getvalue()
    next(r for r in report['models'] if r['configuration'] == 'raw_point_l2' and r['seed'] == 990101)['checkpoint_sha256'] = hashlib.sha256(raw).hexdigest()
    expect_rejection(read,configs,restored,{checkpoint_path:raw,'primary/summary.json':json.dumps(report).encode()},'weights/optimizer/RNG/history')
    print(json.dumps({'test':'forged_retraining_rng_rejected_even_with_updated_hash'}),flush=True)
    print(json.dumps({'status':'four_completed_evidence_semantic_checks_passed','positive_status':positive['status']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','public','primary','retrained','restored'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run({name:getattr(args,name) for name in ('data','public','primary','retrained')},args.restored)


if __name__ == '__main__':
    main()
