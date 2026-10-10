"""Real complete-shadow positive audit and hash-refreshed forgery rejection."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from release_real_shadow import (HERE,ROOT,BASELINE_SHA,Baseline,sha,digest,audit,arrays,
    independent_public_replay)


def check(artifact,output,report=None):
    output.mkdir(parents=True,exist_ok=False)
    capsule=HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule)!=BASELINE_SHA:
        raise ValueError('Original capsule differs')
    base=Baseline(capsule,output/'restored')
    protocol=json.loads((HERE/'protocol.json').read_text())
    expected=independent_public_replay(base,protocol,output/'independent')
    with zipfile.ZipFile(artifact) as z:
        members={name:z.read(name) for name in z.namelist()}
    passed=audit(members.__getitem__,base,expected,output/'independent')
    checks=['full_actual_completed_shadow_positive_audit_passed']
    cases=(('enabled','Not complete failed-model'),('omitted_call','Sequential coverage'),
        ('control','Failed shadow is controlling'),('checkpoint','checkpoint bytes differ'),
        ('filled_peer','Missing delayed peer plan silently filled'),
        ('history','Real public252 history/delayed-peer/GRU/CV input differs'),
        ('forecast','Real checkpoint forecast reinference differs'),
        ('command','DN-MPC plan or CBF command differs'))
    for kind,reason in cases:
        forged=dict(members)
        name='primary/summary.json'
        summary=json.loads(forged[name])
        if kind=='enabled':
            summary['enhanced_control_enabled']=True
        elif kind=='omitted_call':
            summary['modes']['real_shadow']['events'].pop()
        elif kind=='control':
            summary['modes']['real_shadow']['events'][0]['control_eligible']=True
        elif kind=='checkpoint':
            forged['model/primary.pt']=forged['model/primary.pt']+b'forged'
        elif kind=='filled_peer':
            next(e for e in summary['modes']['real_shadow']['events'] if e['status']=='missing_received_peer_context')['forecast_available']=True
        elif kind in ('history','forecast'):
            # Refresh array hashes and summaries, not a checksum-only rejection.
            mode='load_failure' if kind=='history' else 'real_shadow'
            for stage in ('primary','repeated'):
                stage_name=stage+'/summary.json'
                stage_summary=json.loads(forged[stage_name])
                event=next(e for e in stage_summary['modes'][mode]['events'] if 'arrays_path' in e)
                array_name=f"{stage}/{mode}/"+event['arrays_path']
                values=arrays(forged[array_name])
                values['history' if kind=='history' else 'prediction'].flat[0]+=.01
                raw=io.BytesIO();np.savez_compressed(raw,**values)
                forged[array_name]=raw.getvalue()
                event['arrays_sha256']=digest(forged[array_name])
                forged[stage_name]=json.dumps(stage_summary).encode()
            summary=json.loads(forged[name])
        elif kind=='command':
            for stage in ('primary','repeated'):
                command_name=f'{stage}/real_shadow/level5_920000640.commands.npz'
                values=arrays(forged[command_name]);values['commanded'].flat[0]+=.01
                raw=io.BytesIO();np.savez_compressed(raw,**values)
                forged[command_name]=raw.getvalue()
        forged[name]=json.dumps(summary).encode()
        try:
            audit(forged.__getitem__,base,expected,output/'independent')
        except ValueError as error:
            if reason not in str(error):
                raise AssertionError(f'{kind} rejected at unrelated reason: {error}') from error
            checks.append(kind+'_semantic_forgery_rejected')
        else:
            raise AssertionError(f'{kind} forgery was accepted')
    result={'status':'passed_full_actual_shadow_semantic_checks','checks':checks,
            'artifact_sha256':sha(artifact),'complete_positive_status':passed['status'],
            'enhanced_control_enabled':False}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if report is not None:
        report.parent.mkdir(parents=True,exist_ok=True)
        with report.open('x',encoding='utf8') as file:
            json.dump(result,file,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,default=HERE/'artifacts/real_sequential_shadow_20261010.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    check(args.artifact,args.output,args.report)
