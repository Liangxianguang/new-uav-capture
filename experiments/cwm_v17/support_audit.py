"""Actual versus expanded local action support under original full cost."""
import argparse
import collections
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v15')]
from train_mediated_response import (audit_data, frozen_configs, FrozenMediator, attach_public_estimates,
    LocalTwoHead, core_kind, evaluate, normalization, original_scores, rank_metrics, restore_public_call,
    sha, check_archive, descriptive_bootstrap, state_digest)
from geometry_release import arrays, compare_arrays


def validate_protocol(protocol):
    if any(protocol[k] for k in ('enhanced_control_enabled','new_model_trained','holdout_used','prior_gate_overridden')):
        raise ValueError('Original-only diagnostic contract violated')
    if (protocol['candidate_populations'] != ['actual_unique','expanded_union']
            or protocol['models'] != ['motion_only','raw_plain','mediated_plain']
            or protocol['splits'] != ['train','development_validation']):
        raise ValueError('Fixed diagnostic populations changed')


def map_actual(values, agent):
    actual, pool = np.asarray(values['actual_candidates']), np.asarray(values['proposed'])[:,:,agent]
    if actual.ndim != 3 or actual.shape[1:] != (8,3) or pool.shape[1:] != (8,3):
        raise ValueError('Original actual local candidate shape mismatch')
    indices, duplicated = [],0
    for candidate in actual:
        matches = np.flatnonzero((pool == candidate[None]).all((1,2)))
        if len(matches) != 1:
            raise ValueError('Exact public actual candidate mapping not unique')
        index = int(matches[0])
        if index in indices:
            duplicated += 1
        else:
            indices.append(index)
    if not indices:
        raise ValueError('No original local candidates')
    return np.asarray(indices,dtype=np.int64),duplicated


def support_record(call, indices, name, protocol):
    row,values = call['record'],call['values']
    nonanchor = ~(values['proposed'][indices] == values['anchor'][None]).all((1,2,3))
    common = values['valid'][indices] & values['anchor_valid'][None] & nonanchor[:,None]
    effect = np.linalg.norm(values['target'][indices]-values['anchor_target'][None],axis=-1)
    return {**{k:row[k] for k in ('episode_index','step','agent','group','split')},'population':name,
            'indices':indices.tolist(),'candidate_count':len(indices),'valid_nonanchor_points':int(common.sum()),
            'response_sum_m':float(effect[common].sum()),
            'over_threshold_points':int((effect[common] > protocol['effect_threshold_m']).sum()),
            'all_candidates_full_horizon_valid':bool(values['valid'][indices].all() and values['anchor_valid'].all())}


def response_summary(records):
    grouped = collections.defaultdict(lambda:{'calls':0,'candidates':0,'valid_nonanchor_points':0,'response_sum_m':0.,'over_threshold_points':0,'full_support_calls':0})
    for row in records:
        group = grouped[row['group']]
        group['calls'] += 1
        group['candidates'] += row['candidate_count']
        for key in ('valid_nonanchor_points','response_sum_m','over_threshold_points'):
            group[key] += row[key]
        group['full_support_calls'] += row['all_candidates_full_horizon_valid']
    for group in grouped.values():
        points = group['valid_nonanchor_points']
        group['mean_response_m'] = group['response_sum_m']/points if points else None
        group['over_threshold_fraction'] = group['over_threshold_points']/points if points else None
    supported = [g for g in grouped.values() if g['valid_nonanchor_points']]
    return {'calls':len(records),'groups':len(grouped),'groups_with_nonanchor_support':len(supported),
            'valid_nonanchor_points':sum(g['valid_nonanchor_points'] for g in grouped.values()),
            'full_support_calls':sum(g['full_support_calls'] for g in grouped.values()),
            'group_equal_mean_response_m':float(np.mean([g['mean_response_m'] for g in supported])) if supported else None,
            'group_equal_over_threshold_fraction':float(np.mean([g['over_threshold_fraction'] for g in supported])) if supported else None,
            'by_group':dict(grouped)}


def information_paths(values, indices):
    n = len(indices)
    backbone = np.repeat(values['backbone'][None],n,0)
    cv = values['reference'][None]+.1*np.arange(1,9)[:,None]*values['velocity'][None]
    cv = np.repeat(cv[None],n,0)
    truth,reference = values['target'][indices],np.repeat(values['anchor_target'][None],n,0)
    response = truth-reference
    return {'original_gru':backbone,'constant_velocity':cv,'fixed_reference_truth':reference,
            'gru_plus_exact_response':backbone+response,'cv_plus_exact_response':cv+response,
            'action_specific_truth':truth}


def scored_record(call, indices, name, original_call, outputs, protocol):
    row,values = call['record'],call['values']
    if not (values['valid'][indices].all() and values['anchor_valid'].all()):
        raise ValueError('Cannot score incomplete diagnostic decision population')
    paths = information_paths(values,indices)
    for model_name,output in outputs.items():
        paths[model_name] = output['prediction'][indices]
        paths[model_name+'_motion_component'] = values['backbone'][None]+output['motion'][indices]
        paths[model_name+'_motion_plus_exact_response'] = (values['backbone'][None]+output['motion'][indices]
                                                          + values['target'][indices]-values['anchor_target'][None])
    actions = values['proposed'][indices,:,row['agent']]
    costs = {k:original_scores(original_call,actions,v).tolist() for k,v in paths.items()}
    metrics = {k:rank_metrics(v,costs['action_specific_truth'],protocol['tie_absolute_cost_tolerance']) for k,v in costs.items()}
    if 'original_gru' not in costs or not np.isfinite(list(costs.values())).all():
        raise ValueError('Nonfinite/incomplete original full-engine score')
    # Preserve the real baseline choice; do not misattribute library expansion.
    source_choice = row['actual_choice_pool_index']
    local_choice = next((j for j,i in enumerate(indices) if i == source_choice),None)
    if local_choice is None:
        raise ValueError('Actual solver selection absent from fixed population')
    actual_regret = costs['action_specific_truth'][local_choice]-min(costs['action_specific_truth'])
    base_indices,duplicates = map_actual(values,row['agent'])
    if name == 'actual_unique':
        raw_indices = [next(j for j,i in enumerate(base_indices) if np.array_equal(values['proposed'][i,:,row['agent']],a))
                       for a in values['actual_candidates']]
        if not np.allclose(np.asarray(costs['original_gru'])[raw_indices],values['original_local_costs'],rtol=1e-10,atol=1e-8):
            raise ValueError('Actual original score does not reproduce copied solver costs')
        if metrics['original_gru']['choice'] != local_choice:
            raise ValueError('Actual original solver choice changed')
    output_metrics = {}
    for model_name,output in outputs.items():
        mask = values['valid'][indices] & values['anchor_valid'][None]
        nonanchor = ~(values['proposed'][indices] == values['anchor'][None]).all((1,2,3))
        response_mask = mask & nonanchor[:,None]
        error = np.linalg.norm(output['response'][indices]-(values['target'][indices]-values['anchor_target'][None]),axis=-1)
        output_metrics[model_name] = {'response_error_sum_m':float(error[response_mask].sum()),
                                      'valid_nonanchor_points':int(response_mask.sum())}
    return {**{k:row[k] for k in ('episode_index','step','agent','group','split')},'population':name,
            'indices':indices.tolist(),'costs':costs,'metrics':metrics,'actual_solver_local_choice':local_choice,
            'actual_solver_regret':float(actual_regret),'duplicate_actual_proposals':duplicates,'response_errors':output_metrics}


def decision_summary(records,protocol):
    if not records:
        return {'calls':0,'groups':0,'methods':{},'comparisons':{}}
    groups = [r['group'] for r in records]
    methods = set(records[0]['metrics'])
    if any(set(r['metrics']) != methods for r in records):
        raise ValueError('Inconsistent diagnostic information conditions')
    def bootstrap(values):
        measured = descriptive_bootstrap(values,groups,protocol['bootstrap']['draws'],protocol['bootstrap']['seed'])
        measured['interpretation'] = protocol['bootstrap']['interpretation']
        return measured
    result = {'calls':len(records),'groups':len(set(groups)),'methods':{},'comparisons':{},'response_errors':{}}
    for method in sorted(methods):
        result['methods'][method] = {'regret':bootstrap([r['metrics'][method]['regret_in_diagnostic_cost'] for r in records]),
                                     'gain_vs_actual_solver':bootstrap([r['actual_solver_regret']-r['metrics'][method]['regret_in_diagnostic_cost'] for r in records]),
                                     'choice_changes_vs_actual_solver':sum(r['metrics'][method]['choice'] != r['actual_solver_local_choice'] for r in records)}
    comparisons = list(protocol['comparisons'])
    for name in protocol['models']:
        if name in methods:
            comparisons += [[name+'_motion_component',name],[name+'_motion_component',name+'_motion_plus_exact_response']]
    for first,second in comparisons:
        result['comparisons'][first+'__to__'+second] = {
            'gain':bootstrap([r['metrics'][first]['regret_in_diagnostic_cost']-r['metrics'][second]['regret_in_diagnostic_cost'] for r in records]),
            'choice_changes':sum(r['metrics'][first]['choice'] != r['metrics'][second]['choice'] for r in records)}
    for name in records[0]['response_errors']:
        per_group = collections.defaultdict(lambda:np.zeros(2))
        for row in records:
            m = row['response_errors'][name]
            per_group[row['group']] += [m['response_error_sum_m'],m['valid_nonanchor_points']]
        supported = {g:float(x[0]/x[1]) for g,x in per_group.items() if x[1]}
        result['response_errors'][name] = {'by_group':supported,'groups':len(supported),
                                          'group_equal_mean_m':float(np.mean(list(supported.values()))) if supported else None}
    return result


def load_selected_models(archive,calls,indices,mean,scale,protocol):
    report = json.loads(archive.read('primary/summary.json'))
    outputs = {i:{} for i in indices}
    selections = {}
    for name in protocol['models']:
        seed = report['selected_median_seeds'][name]
        rows = [r for r in report['models'] if r['configuration'] == name]
        if seed != sorted(rows,key=lambda r:(r['development']['group_equal_ade_m'],r['seed']))[1]['seed']:
            raise ValueError('Frozen V15 median selection changed')
        row = next(r for r in rows if r['seed'] == seed)
        raw = archive.read(f'primary/{name}_seed{seed}.pt')
        if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
            raise ValueError('Fixed failed checkpoint digest mismatch')
        checkpoint = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if checkpoint['online_promoted'] or checkpoint['baseline_weights_included'] or checkpoint['mediator_optimized']:
            raise ValueError('Original frozen unpromoted checkpoint contract differs')
        config = checkpoint['protocol']['training']
        with torch.random.fork_rng(devices=[]):
            model = LocalTwoHead(core_kind(name),config['motion_scale_m'],config['response_scale_m'])
        model.load_state_dict(checkpoint['model_state'],strict=True)
        model.eval().requires_grad_(False)
        before = state_digest(model)
        measured,predictions = evaluate(model,name,calls,indices,mean,scale)
        if measured != row['development']:
            raise ValueError('Fixed V15 model public reload metrics differ')
        expected = {f'call{j}_{key}':v for j,o in enumerate(predictions) for key,v in o.items()}
        if not compare_arrays(expected,arrays(archive.read(f'primary/{name}_seed{seed}_predictions.npz'))):
            raise ValueError('Fixed V15 model prediction bytes differ')
        if state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Frozen model mutated during support audit')
        for i,output in zip(indices,predictions):
            outputs[i][name] = output
        selections[name] = {'seed':seed,'checkpoint_sha256':row['checkpoint_sha256']}
    return outputs,selections


def compute(archive,candidate,configs,restored,mediator_path,protocol):
    validate_protocol(protocol)
    source = json.loads(archive.read('primary/summary.json'))
    calls,_ = audit_data(archive.read,candidate.read,configs,restored)
    train = [i for i,c in enumerate(calls) if c['record']['split'] == 'train']
    dev = [i for i,c in enumerate(calls) if c['record']['split'] == 'development_validation']
    mean,scale = normalization(calls,train)
    mediator = FrozenMediator(mediator_path,source['protocol'])
    attach_public_estimates(calls,mediator)
    if not compare_arrays({f'call{i}':c['estimated_commands'] for i,c in enumerate(calls)},arrays(archive.read('primary/public_estimates.npz'))):
        raise ValueError('Frozen public mediator estimate reload differs')
    outputs,selections = load_selected_models(archive,calls,dev,mean,scale,protocol)
    support,decisions = [],[]
    duplicates = 0
    for i,call in enumerate(calls):
        row,values = call['record'],call['values']
        actual,duplicated = map_actual(values,row['agent'])
        duplicates += duplicated
        populations = {'actual_unique':actual,'expanded_union':np.arange(len(values['proposed']),dtype=np.int64)}
        raw = archive.read('data/'+row['context_path'])
        if hashlib.sha256(raw).hexdigest() != row['context_sha256']:
            raise ValueError('Actual delayed original context digest differs')
        original = restore_public_call(json.loads(raw),*configs,values['backbone'])
        for name,indices in populations.items():
            record = support_record(call,indices,name,protocol)
            support.append(record)
            if record['all_candidates_full_horizon_valid']:
                decision = scored_record(call,indices,name,original,outputs.get(i,{}),protocol)
                decisions.append(decision)
        if (i+1) % 256 == 0:
            print(json.dumps({'calls_audited':i+1,'decision_populations':len(decisions)}),flush=True)
    summary = {'status':'actual_local_information_diagnostic_finished_not_promoted','protocol':protocol,
               'source_sha256':protocol['source_archive_sha256'],'fixed_models':selections,
               'duplicate_actual_proposals':duplicates,'split_calls':{'train':len(train),'development_validation':len(dev)},
               'populations':{},'paired_intersection':{},'enhanced_control_enabled':False,
               'new_model_trained':False,'holdout_used':False,'prior_gate_overridden':False,'scope':protocol['scope']}
    identity = lambda r:(r['episode_index'],r['step'],r['agent'])
    for split in protocol['splits']:
        summary['populations'][split] = {}
        sets = {name:[r for r in decisions if r['split'] == split and r['population'] == name] for name in protocol['candidate_populations']}
        intersection = set(identity(r) for r in sets['actual_unique']) & set(identity(r) for r in sets['expanded_union'])
        summary['paired_intersection'][split] = {name:decision_summary([r for r in sets[name] if identity(r) in intersection],protocol)
                                               for name in protocol['candidate_populations']}
        for name in protocol['candidate_populations']:
            rows = [r for r in support if r['split'] == split and r['population'] == name]
            summary['populations'][split][name] = {'response_support':response_summary(rows),'decisions':decision_summary(sets[name],protocol)}
    if not mediator.unchanged():
        raise ValueError('Frozen mediator changed')
    return summary,support,decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
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
            raise ValueError('Pinned original/research source changed')
    for path in (args.source,args.candidate_source,args.mediator):
        check_archive(path,'ARTIFACT_MANIFEST.json',False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    configs = frozen_configs(args.capsule,args.output/'restored')
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {Path(__file__).resolve(),ROOT/'protocol.json'}
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        destination = args.output/'source'/source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
    with zipfile.ZipFile(args.source) as archive,zipfile.ZipFile(args.candidate_source) as candidate:
        summary,support,decisions = compute(archive,candidate,configs,args.output/'restored',args.mediator,protocol)
    if any(sha(ROOT.parent/name) != digest for name,digest in hashes.items()):
        raise ValueError('Run-used diagnostic sources changed')
    summary['source_hashes'] = hashes
    for name,value in (('support',support),('decisions',decisions)):
        path = args.output/(name+'.json')
        path.write_text(json.dumps(value,indent=2))
        summary[name+'_sha256'] = sha(path)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':summary['status'],'populations':{k:{n:v['response_support'] for n,v in s.items()} for k,s in summary['populations'].items()}}),flush=True)


if __name__ == '__main__':
    main()
