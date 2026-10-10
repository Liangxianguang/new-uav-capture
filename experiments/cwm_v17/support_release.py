"""Reload fixed public models and recompute actual-local information audit."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from support_audit import compute,validate_protocol,frozen_configs,sha,check_archive


def validate_report(report,protocol):
    if (report['protocol'] != protocol or report['status'] != 'actual_local_information_diagnostic_finished_not_promoted'
            or any(report[k] for k in ('enhanced_control_enabled','new_model_trained','holdout_used','prior_gate_overridden'))):
        raise ValueError('Diagnostic-only original contract mismatch')
    if report['source_sha256'] != protocol['source_archive_sha256']:
        raise ValueError('Actual V15 diagnostic source changed')


def audit(read,source,candidate,configs,restored,mediator,protocol):
    report = json.loads(read('run/summary.json'))
    validate_report(report,protocol)
    if json.loads(read('run/source/cwm_v17/protocol.json')) != protocol:
        raise ValueError('Saved predeclared actual-local protocol changed')
    for name,digest in report['source_hashes'].items():
        if hashlib.sha256(read('run/source/'+name)).hexdigest() != digest or sha(ROOT.parent/name) != digest:
            raise ValueError('Run-used original/research diagnostic source mismatch')
    # A verifier re-runs the original source data audit, exact public candidate
    # mapping, frozen mediator/core inference and full original-cost scoring.
    expected,support,decisions = compute(source,candidate,configs,restored,mediator,protocol)
    for name,value in (('support',support),('decisions',decisions)):
        raw = read('run/'+name+'.json')
        if hashlib.sha256(raw).hexdigest() != report[name+'_sha256'] or json.loads(raw) != value:
            raise ValueError('Independent actual-local '+name+' recomputation differs')
    actual = {k:v for k,v in report.items() if k not in ('source_hashes','support_sha256','decisions_sha256')}
    if actual != expected:
        raise ValueError('Independent actual-local populations/information summaries differ')
    return {'status':'passed_fixed_public_models_and_actual_local_information_recomputation',
            'split_calls':report['split_calls'],'fixed_models':report['fixed_models'],
            'candidate_mapping_recomputed':True,'actual_original_solver_costs_reproduced':True,
            'oracle_not_model_input':True,'all_full_cost_rankings_recomputed':True,
            'populations':report['populations'],'paired_intersection':report['paired_intersection'],
            'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False,'prior_gate_overridden':False,
            'scope':protocol['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--source',type=Path,default=ROOT.parent/'cwm_v15/artifacts/mediated_response_training_20261010.zip')
    parser.add_argument('--candidate-source',type=Path,default=ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip')
    parser.add_argument('--mediator',type=Path,default=ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip')
    parser.add_argument('--capsule',type=Path,default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    validate_protocol(protocol)
    for path,key in ((args.source,'source_archive_sha256'),(args.candidate_source,'candidate_source_archive_sha256'),
                     (args.mediator,'mediator_archive_sha256'),(args.capsule,'baseline_capsule_sha256')):
        if sha(path) != protocol[key]:
            raise ValueError('Protected diagnostic dependency changed')
    for path in (args.source,args.candidate_source,args.mediator):
        check_archive(path,'ARTIFACT_MANIFEST.json',False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    restored = ROOT.parent.parent/'results/cwm_v17/audit_restored'
    configs = frozen_configs(args.capsule,restored)
    with zipfile.ZipFile(args.source) as source,zipfile.ZipFile(args.candidate_source) as candidate:
        if args.verify:
            integrity = check_archive(args.verify,'ARTIFACT_MANIFEST.json',False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read,source,candidate,configs,restored,args.mediator,protocol)
            print(json.dumps({'status':result['status'],'split_calls':result['split_calls'],'artifact_sha256':integrity['sha256']}),flush=True)
            return
        if args.run is None:
            parser.error('Completed support run required')
        members = {}
        for file in args.run.rglob('*'):
            relative = file.relative_to(args.run)
            if file.is_file() and 'restored' not in relative.parts:
                members['run/'+relative.as_posix()] = file.read_bytes()
        for file in ROOT.glob('*.py'):
            members['verification_source/'+file.name] = file.read_bytes()
        result = audit(members.__getitem__,source,candidate,configs,restored,args.mediator,protocol)
        path = ROOT/'artifacts/actual_local_information_20261010.zip'
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,raw in members.items():
                archive.writestr(name,raw)
            archive.writestr('ARTIFACT_MANIFEST.json',json.dumps({n:hashlib.sha256(raw).hexdigest() for n,raw in members.items()},indent=2))
        integrity = check_archive(path,'ARTIFACT_MANIFEST.json',False)
        with zipfile.ZipFile(path) as archive:
            audit(archive.read,source,candidate,configs,restored,args.mediator,protocol)
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    with (reports/'support_summary.json').open('xb') as file:
        file.write((args.run/'summary.json').read_bytes())
    with (reports/'release_manifest.json').open('x') as file:
        json.dump({**result,'artifact_sha256':integrity['sha256'],'artifact_bytes':path.stat().st_size},file,indent=2)
    print(json.dumps({'status':result['status'],'split_calls':result['split_calls'],
                      'artifact_sha256':integrity['sha256'],'artifact_bytes':path.stat().st_size}),flush=True)


if __name__ == '__main__':
    main()
