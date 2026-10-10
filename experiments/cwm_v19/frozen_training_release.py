"""Independently reload all V19 models, costs and exact repeated training state."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v7')]
from train_frozen_response import (audit_data, normalization, control_metrics, prepare_contexts,
    evaluate, decision_records, qualification, FrozenCommonResponse, LocalTwoHead,
    FrozenMediator, attach_public_estimates, state_digest, frozen_configs, sha, BASELINE_SHA)
from geometry_release import arrays,compare_arrays


def tensor_tree_equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left,right)
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and left.dtype == right.dtype and np.array_equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys() == right.keys() and all(tensor_tree_equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):
        return type(left) == type(right) and len(left) == len(right) and all(tensor_tree_equal(a,b) for a,b in zip(left,right))
    return left == right


def audit_sources(read,stage,report):
    for name,digest in report['source_hashes'].items():
        if hashlib.sha256(read(stage+'/source/'+name)).hexdigest() != digest or sha(ROOT.parent/name) != digest:
            raise ValueError('Snapshot/current run-used training source changed')


def verify_manifest(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate artifact paths')
        manifest = json.loads(archive.read('ARTIFACT_MANIFEST.json'))
        if set(manifest) != set(names)-{'ARTIFACT_MANIFEST.json'}:
            raise ValueError('Incomplete artifact manifest coverage')
        for name,digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Artifact member digest mismatch')
    return {'sha256':sha(path),'bytes':path.stat().st_size,'members':len(manifest)}


def audit(read,configs,restored):
    calls,data_report = audit_data(read,configs,restored)
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    indices = {split:[i for i,c in enumerate(calls) if c['record']['split'] == split] for split in ('train','development_validation')}
    mean,scale = normalization(calls,indices['train'])
    controls = control_metrics(calls,indices['development_validation'])
    contexts = prepare_contexts(calls,indices['development_validation'],read,configs)
    mediator = FrozenMediator(ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip',protocol)
    diagnostics = attach_public_estimates(calls,mediator)
    expected_estimates = {f'call{i}':c['estimated_commands'] for i,c in enumerate(calls)}
    sets,summaries = [],[]
    population = {(name,seed) for name in protocol['models'] for seed in protocol['training']['seeds']}
    for stage in ('primary','retrained'):
        report = json.loads(read(stage+'/summary.json'))
        if report['protocol'] != protocol or report['status'] != 'offline_frozen_common_motion_response_finished_not_promoted':
            raise ValueError('Frozen training protocol/status changed')
        if any(report[k] for k in ('enhanced_control_enabled','baseline_weights_included','holdout_used','prior_gate_overridden','common_motion_in_response_optimizer')):
            raise ValueError('Original/frozen offline contract violated')
        audit_sources(read,stage,report)
        if json.loads(read(stage+'/source/cwm_v19/training_protocol.json')) != protocol:
            raise ValueError('Saved training protocol changed')
        if report['data_summary_sha256'] != hashlib.sha256(read('data/summary.json')).hexdigest() or report['public_summary_sha256'] != hashlib.sha256(read('public/summary.json')).hexdigest():
            raise ValueError('Actual training source provenance changed')
        if (report['split_calls'] != {k:len(v) for k,v in indices.items()} or
                report['split_groups'] != {k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()}):
            raise ValueError('Actual training split differs')
        if report['controls'] != controls or report['full_development_calls'] != len(contexts):
            raise ValueError('Actual controls/full decision support differs')
        if report['mediator_checkpoint_sha256'] != mediator.checkpoint_sha256 or not report['mediator_unchanged']:
            raise ValueError('Frozen mediator contract changed')
        if not compare_arrays(expected_estimates,arrays(read(stage+'/public_estimates.npz'))) or json.loads(read(stage+'/mediator_diagnostics.json')) != diagnostics:
            raise ValueError('Public-only mediator reload differs')
        if not compare_arrays({'mean':mean,'scale':scale},arrays(read(stage+'/normalization.npz'))):
            raise ValueError('Train-only normalization differs')
        rows = report['models']
        if len(rows) != 9 or {(r['configuration'],r['seed']) for r in rows} != population:
            raise ValueError('Actual fixed9-model population changed')
        checkpoints,models,decisions = {},{},{}
        for row in rows:
            name,seed = row['configuration'],row['seed']
            identifier = f'{name}_seed{seed}'
            raw = read(f'{stage}/{identifier}.pt')
            if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
                raise ValueError('Checkpoint digest changed')
            checkpoint = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if (checkpoint['configuration'] != name or checkpoint['seed'] != seed or checkpoint['protocol'] != protocol
                    or any(checkpoint[k] for k in ('online_promoted','baseline_weights_included','mediator_optimized','common_motion_in_response_optimizer'))):
                raise ValueError('Actual checkpoint frozen contract changed')
            if any(checkpoint[k] != report[k] for k in ('source_hashes','data_summary_sha256','public_summary_sha256','mediator_checkpoint_sha256')):
                raise ValueError('Checkpoint provenance differs')
            if not np.array_equal(checkpoint['normalizer_mean'].numpy(),mean) or not np.array_equal(checkpoint['normalizer_scale'].numpy(),scale):
                raise ValueError('Checkpoint train normalizer differs')
            with torch.random.fork_rng(devices=[]):
                if name == 'motion_only':
                    model = LocalTwoHead('motion_only',protocol['training']['motion_scale_m'],protocol['training']['response_scale_m'])
                else:
                    model = FrozenCommonResponse(
                        LocalTwoHead('motion_only',protocol['training']['motion_scale_m'],protocol['training']['response_scale_m']),
                        LocalTwoHead('plain',protocol['training']['motion_scale_m'],protocol['training']['response_scale_m']),
                        name == 'mediated_response')
            model.load_state_dict(checkpoint['model_state'],strict=True)
            model.eval().requires_grad_(False)
            common = model if name == 'motion_only' else model.common
            if state_digest(common) != row['common_motion_state_digest'] or state_digest(common) != checkpoint['common_motion_state_digest']:
                raise ValueError('Shared frozen common motion digest differs')
            measured,outputs = evaluate(model,name,calls,indices['development_validation'],mean,scale)
            expected = {f'call{j}_{k}':v for j,o in enumerate(outputs) for k,v in o.items()}
            if measured != row['development'] or not compare_arrays(expected,arrays(read(f'{stage}/{identifier}_predictions.npz'))):
                raise ValueError('Independent model predictions/metrics differ')
            measured_decisions = decision_records(calls,indices['development_validation'],outputs,contexts)
            if json.loads(read(f'{stage}/{identifier}_decisions.json')) != measured_decisions:
                raise ValueError('Independent original full-cost model decisions differ')
            history = json.loads(read(f'{stage}/{identifier}_history.json'))
            epochs = protocol['training']['motion_epochs' if name == 'motion_only' else 'response_epochs']
            if [r['epoch'] for r in history] != list(range(1,epochs+1)) or not np.isfinite([r['mean_batch_loss_terms'] for r in history]).all():
                raise ValueError('Actual final fixed-budget history differs')
            if row['parameters'] != sum(p.numel() for p in model.parameters()) or row['initial_state_digest'] != checkpoint['initial_state_digest']:
                raise ValueError('Actual model budget/initial state differs')
            checkpoints[identifier],models[identifier],decisions[identifier] = checkpoint,model,measured_decisions
        for seed in protocol['training']['seeds']:
            common = models[f'motion_only_seed{seed}']
            raw,mediated = [models[f'{name}_seed{seed}'] for name in ('raw_response','mediated_response')]
            if not tensor_tree_equal(common.state_dict(),raw.common.state_dict()) or not tensor_tree_equal(common.state_dict(),mediated.common.state_dict()):
                raise ValueError('Response stages changed the matched common-motion weights')
            raw_cp,med_cp = [checkpoints[f'{name}_seed{seed}'] for name in ('raw_response','mediated_response')]
            if raw_cp['initial_state_digest'] != med_cp['initial_state_digest'] or raw_cp['sampler_rng_state'] != med_cp['sampler_rng_state']:
                raise ValueError('Raw/mediated initialization or sampler not matched')
            # Response optimizer must contain only response-core trainable params.
            for checkpoint,model in ((raw_cp,raw),(med_cp,mediated)):
                expected_count = sum(1 for n,p in model.response_core.named_parameters() if not n.startswith('motion_head.'))
                param_ids = [p for g in checkpoint['optimizer_state']['param_groups'] for p in g['params']]
                if len(param_ids) != expected_count or len(set(param_ids)) != expected_count:
                    raise ValueError('Response optimizer parameter budget differs')
        selected,gates = qualification(rows,controls,decisions,protocol)
        if selected != report['selected_median_seeds'] or gates != report['qualification']:
            raise ValueError('Actual gates/fixed ADE-median selections differ')
        sets.append(checkpoints)
        summaries.append(report)
    for name,checkpoint in sets[0].items():
        if not tensor_tree_equal(checkpoint,sets[1][name]):
            raise ValueError('Independent weights/optimizer/RNG/checkpoint state differs')
        if read('primary/'+name+'_history.json') != read('retrained/'+name+'_history.json'):
            raise ValueError('Independent fixed-budget loss histories differ')
    for key in ('protocol','source_hashes','controls','qualification','selected_median_seeds','split_calls','split_groups','full_development_calls'):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError('Independent training summaries differ')
    if not mediator.unchanged():
        raise ValueError('Frozen mediator changed in verification')
    return {'status':'passed_independent_frozen_common_motion_response_training_audit',
            'data_episodes':64,'data_groups':32,'split_calls':summaries[0]['split_calls'],
            'full_development_calls':len(contexts),'models_per_run':9,'independent_training_runs':2,
            'weights_optimizer_rng_history_equal':True,'public252_frame_reencoding_verified':True,
            'frozen_shared_common_motion_weights_equal':True,'public_model_prediction_bytes_equal':True,
            'all_original_full_cost_decisions_recomputed':True,'train_only_normalization_verified':True,
            'selected_median_seeds':summaries[0]['selected_median_seeds'],'qualification':summaries[0]['qualification'],
            'enhanced_control_enabled':False,'holdout_used':False,'scope':protocol['later_required']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','public','primary','retrained','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--verify',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original capsule changed')
    configs = frozen_configs(capsule,ROOT.parent.parent/'results/cwm_v19/release_restored')
    restored = ROOT.parent.parent/'results/cwm_v19/release_restored'
    if args.verify:
        integrity = verify_manifest(args.verify)
        with zipfile.ZipFile(args.verify) as archive:
            result = audit(archive.read,configs,restored)
            if json.loads(archive.read('release_summary.json')) != result:
                raise ValueError('Archived release summary differs from independent recomputation')
        print(json.dumps({'status':result['status'],**integrity}),flush=True)
        return
    if any(getattr(args,n) is None for n in ('data','public','primary','retrained','output')):
        parser.error('Completed data/public and two training runs required')
    roots = {n:getattr(args,n) for n in ('data','public','primary','retrained')}
    def read(name):
        stage,relative = name.split('/',1)
        return (roots[stage]/relative).read_bytes()
    result = audit(read,configs,restored)
    members = {}
    for stage,root in roots.items():
        for p in root.rglob('*'):
            relative = p.relative_to(root)
            if p.is_file() and not {'restored','__pycache__'}.intersection(relative.parts):
                members[stage+'/'+relative.as_posix()] = p.read_bytes()
    members['release_summary.json'] = json.dumps(result,indent=2).encode()
    for p in ROOT.glob('*.py'):
        members['verification_source/'+p.name] = p.read_bytes()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in members.items():
            archive.writestr(name,raw)
        archive.writestr('ARTIFACT_MANIFEST.json',json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in members.items()},indent=2))
    integrity = verify_manifest(args.output)
    with zipfile.ZipFile(args.output) as archive:
        recomputed = audit(archive.read,configs,restored)
        if json.loads(archive.read('release_summary.json')) != recomputed or recomputed != result:
            raise ValueError('Archived release summary differs from independent recomputation')
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    for name in ('data','public','primary'):
        dest = reports/('training_summary.json' if name == 'primary' else name+'_summary.json')
        with dest.open('xb') as file:
            file.write(read(name+'/summary.json'))
    with (reports/'release_manifest.json').open('x') as file:
        json.dump({**result,'artifact':integrity},file,indent=2)
    print(json.dumps({'status':result['status'],**integrity}),flush=True)


if __name__ == '__main__':
    main()
