"""Re-infer all fresh factorial models and audit two full independent runs."""
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
sys.path.insert(0,str(ROOT))
from train_loss_factorial import (audit_data, read_dataset, normalization, population_weights,
    control_metrics, prepare_contexts, evaluate, decision_records, qualification, validate_training,
    FrozenCommonResponse, LocalTwoHead, state_digest, frozen_configs, sha, BASELINE_SHA, decompose)
from frozen_training_release import tensor_tree_equal, verify_manifest
from geometry_release import arrays,compare_arrays
from artifact_transport import split_archive,resolve_artifact


def audit_sources(read,stage,report):
    for name,digest in report['source_hashes'].items():
        if hashlib.sha256(read(stage+'/source/'+name)).hexdigest() != digest or sha(ROOT.parent/name) != digest:
            raise ValueError('Saved/current factorial source differs')


def verify_protocol_bytes(saved,expected):
    # Existing training used this Windows runtime's Path.read_text default.
    # The UTF8 source contains a superscript in a descriptive field. Compare
    # source BYTES, not json.loads(bytes) against locale-decoded record text.
    # Recorded protocol is still strictly checked above against the same
    # frozen run source; no checkpoint, source, numeric protocol or gate changes.
    if saved != expected:
        raise ValueError('Saved preregistered protocol bytes changed')


def audit(read,configs,restored):
    calls,_ = audit_data(read,configs,restored)
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    validate_training(protocol)
    indices = {split:[i for i,c in enumerate(calls) if c['record']['split'] == split] for split in ('train','development_validation')}
    mean,scale = normalization(calls,indices['train'])
    weights,population = population_weights(calls,indices['train'])
    serialized_weights = json.loads(json.dumps({'population':population,'weights':weights}))
    controls = control_metrics(calls,indices['development_validation'])
    contexts = prepare_contexts(calls,indices['development_validation'],read,configs)
    diagnostics = json.loads((ROOT.parent/'cwm_v20/diagnostic_protocol.json').read_text())
    expected_population = {(name,seed) for name in protocol['models'] for seed in protocol['training']['seeds']}
    sets,summaries = [],[]
    for stage in ('primary','retrained'):
        report = json.loads(read(stage+'/summary.json'))
        if report['protocol'] != protocol or report['status'] != 'offline_fresh_response_loss_factorial_finished_not_promoted':
            raise ValueError('Published factorial protocol/status differs')
        if any(report[k] for k in ('enhanced_control_enabled','baseline_weights_included','holdout_used','prior_gate_overridden','common_motion_in_response_optimizer')):
            raise ValueError('Frozen factorial contract violated')
        audit_sources(read,stage,report)
        verify_protocol_bytes(read(stage+'/source/cwm_v21/training_protocol.json'),
                              (ROOT/'training_protocol.json').read_bytes())
        for source in ('data','public'):
            if report[source+'_summary_sha256'] != hashlib.sha256(read(source+'/summary.json')).hexdigest():
                raise ValueError('Data/public provenance differs')
        if (report['split_calls'] != {k:len(v) for k,v in indices.items()} or
            report['split_groups'] != {k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()}):
            raise ValueError('Training split population differs')
        if report['controls'] != controls or report['full_development_calls'] != len(contexts):
            raise ValueError('Controls/original cost support differs')
        if json.loads(read(stage+'/population_weights.json')) != serialized_weights or report['population_weights_sha256'] != hashlib.sha256(read(stage+'/population_weights.json')).hexdigest():
            raise ValueError('Full train-only weighting differs')
        if not compare_arrays({'mean':mean,'scale':scale},arrays(read(stage+'/normalization.npz'))):
            raise ValueError('Train-only normalizer differs')
        rows = report['models']
        if len(rows) != 15 or {(r['configuration'],r['seed']) for r in rows} != expected_population:
            raise ValueError('Fixed final fifteen-model population differs')
        checkpoints,models,decisions = {},{},{}
        for row in rows:
            name,seed = row['configuration'],row['seed']
            identifier = f'{name}_seed{seed}'
            raw = read(stage+'/'+identifier+'.pt')
            if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
                raise ValueError('Checkpoint hash differs')
            checkpoint = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if (checkpoint['configuration'] != name or checkpoint['seed'] != seed or checkpoint['protocol'] != protocol
                or any(checkpoint[k] for k in ('online_promoted','baseline_weights_included','mediator_optimized','common_motion_in_response_optimizer'))):
                raise ValueError('Checkpoint optional frozen contract differs')
            if any(checkpoint[k] != report[k] for k in ('source_hashes','data_summary_sha256','public_summary_sha256','population_weights_sha256')):
                raise ValueError('Checkpoint provenance differs')
            if not np.array_equal(checkpoint['normalizer_mean'].numpy(),mean) or not np.array_equal(checkpoint['normalizer_scale'].numpy(),scale):
                raise ValueError('Checkpoint normalizer differs')
            scales = protocol['training']
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                initial_model = LocalTwoHead('motion_only' if name == 'motion_only' else 'plain',scales['motion_scale_m'],scales['response_scale_m'])
                if name != 'motion_only':
                    initial_model.motion_head.requires_grad_(False)
                if state_digest(initial_model) != row['initial_state_digest']:
                    raise ValueError('Fixed per-seed initialization differs')
                common = LocalTwoHead('motion_only',scales['motion_scale_m'],scales['response_scale_m'])
                model = common if name == 'motion_only' else FrozenCommonResponse(common,
                    LocalTwoHead('plain',scales['motion_scale_m'],scales['response_scale_m']),False)
            model.load_state_dict(checkpoint['model_state'],strict=True)
            model.eval().requires_grad_(False)
            common = model if name == 'motion_only' else model.common
            if state_digest(common) != row['common_motion_state_digest'] or state_digest(common) != checkpoint['common_motion_state_digest']:
                raise ValueError('Common motion digest differs')
            if checkpoint['initial_state_digest'] != row['initial_state_digest'] or row['parameters'] != sum(p.numel() for p in model.parameters()):
                raise ValueError('Budget/initial state differs')
            measured_diagnostics = {}
            for split,chosen in indices.items():
                measured,outputs = evaluate(model,name,calls,chosen,mean,scale)
                expected = {f'call{j}_{k}':v for j,o in enumerate(outputs) for k,v in o.items()}
                key = 'train' if split == 'train' else 'development'
                if measured != row[key] or not compare_arrays(expected,arrays(read(f'{stage}/{identifier}_{split}_predictions.npz'))):
                    raise ValueError('Independent train/dev forecast or metrics differs')
                measured_diagnostics[split] = decompose(calls,chosen,outputs,diagnostics)
                if split == 'development_validation':
                    measured_decisions = decision_records(calls,chosen,outputs,contexts)
                    if measured_decisions != json.loads(read(f'{stage}/{identifier}_decisions.json')):
                        raise ValueError('Independent complete original costs/decisions differs')
                    decisions[identifier] = measured_decisions
            if measured_diagnostics != json.loads(read(f'{stage}/{identifier}_diagnostics.json')):
                raise ValueError('Independent bins/offsets/weighting diagnosis differs')
            history = json.loads(read(f'{stage}/{identifier}_history.json'))
            epochs = scales['motion_epochs' if name == 'motion_only' else 'response_epochs']
            if [h['epoch'] for h in history] != list(range(1,epochs+1)) or not np.isfinite([h['mean_batch_loss_terms'] for h in history]).all():
                raise ValueError('Fixed final training budget differs')
            expected_steps = epochs*((len(indices['train'])+scales['batch_calls']-1)//scales['batch_calls'])
            if not checkpoint['optimizer_state']['state'] or any(int(s['step'].item()) != expected_steps for s in checkpoint['optimizer_state']['state'].values()):
                raise ValueError('Optimizer update count differs from protocol')
            checkpoints[identifier],models[identifier] = checkpoint,model
        for seed in scales['seeds']:
            common = models[f'motion_only_seed{seed}']
            responses = [models[f'{name}_seed{seed}'] for name in protocol['response_configurations']]
            cps = [checkpoints[f'{name}_seed{seed}'] for name in protocol['response_configurations']]
            if any(not tensor_tree_equal(common.state_dict(),model.common.state_dict()) for model in responses):
                raise ValueError('Response altered matched frozen common motion')
            if any(cp['initial_state_digest'] != cps[0]['initial_state_digest'] or not tensor_tree_equal(cp['sampler_rng_state'],cps[0]['sampler_rng_state']) for cp in cps):
                raise ValueError('Factorial initialization/order mismatch')
            for cp,model in zip(cps,responses):
                expected_count = sum(1 for n,p in model.response_core.named_parameters() if not n.startswith('motion_head.'))
                param_ids = [p for g in cp['optimizer_state']['param_groups'] for p in g['params']]
                if len(param_ids) != expected_count or len(set(param_ids)) != expected_count:
                    raise ValueError('Response-only optimizer parameter budget differs')
        selected,gates,factorial = qualification(rows,controls,decisions,protocol)
        if (selected != report['selected_median_seeds'] or gates != report['qualification'] or factorial != report['factorial_comparisons']
            or report['primary_research_eligible'] != gates[protocol['primary_configuration']]['research_eligible']):
            raise ValueError('Fixed primary selection/qualification/factorial comparisons differs')
        sets.append(checkpoints)
        summaries.append(report)
    for identifier,checkpoint in sets[0].items():
        if not tensor_tree_equal(checkpoint,sets[1][identifier]) or read(f'primary/{identifier}_history.json') != read(f'retrained/{identifier}_history.json'):
            raise ValueError('Independent full weights/optimizer/RNG/history differs')
    if summaries[0] != summaries[1]:
        raise ValueError('Independent full training summaries differ')
    return {'status':'passed_independent_fresh_response_loss_factorial_audit','models_per_run':15,'independent_runs':2,
        'split_calls':summaries[0]['split_calls'],'full_development_calls':len(contexts),
        'all_train_development_forecasts_reinferred':True,'weights_optimizer_rng_history_equal':True,
        'public252_frames_and_original_costs_verified':True,'matched_frozen_common_motion_equal':True,
        'train_only_population_weights_verified':True,'all_bins_offsets_diagnostics_recomputed':True,
        'selected_median_seeds':selected,'qualification':gates,'factorial_comparisons':factorial,
        'primary_research_eligible':summaries[0]['primary_research_eligible'],
        'enhanced_control_enabled':False,'holdout_used':False,'scope':protocol['later_required']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','public','primary','retrained','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--parts-output',type=Path,help='Lossless <=45MiB parts for GitHub, full ZIP stays in results')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original capsule changed')
    restored = ROOT.parent.parent/'results/cwm_v21/release_restored'
    configs = frozen_configs(capsule,restored)
    if args.verify:
        artifact = resolve_artifact(args.verify,ROOT.parent.parent/'results/cwm_v21/assembled')
        integrity = verify_manifest(artifact)
        with zipfile.ZipFile(artifact) as archive:
            result = audit(archive.read,configs,restored)
            if json.loads(archive.read('release_summary.json')) != result:
                raise ValueError('Archived release summary differs')
        print(json.dumps({'status':result['status'],**integrity}),flush=True)
        return
    if any(getattr(args,name) is None for name in ('data','public','primary','retrained','output')):
        parser.error('Full fresh data/public/two complete training runs required')
    roots = {name:getattr(args,name) for name in ('data','public','primary','retrained')}
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
        if json.loads(archive.read('release_summary.json')) != recomputed or result != recomputed:
            raise ValueError('Archived evidence differs from pre-archive audit')
    transport = None
    if args.parts_output:
        transport = split_archive(args.output,args.parts_output)
        joined = resolve_artifact(args.parts_output,ROOT.parent.parent/'results/cwm_v21/assembled')
        if sha(joined) != integrity['sha256']:
            raise ValueError('Lossless transport differs from audited full archive')
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    for name in ('data','public','primary'):
        with (reports/('training_summary.json' if name == 'primary' else name+'_summary.json')).open('xb') as file:
            file.write(read(name+'/summary.json'))
    with (reports/'release_manifest.json').open('x') as file:
        json.dump({**result,'artifact':integrity,'transport':transport},file,indent=2)
    print(json.dumps({'status':result['status'],**integrity}),flush=True)


if __name__ == '__main__':
    main()
