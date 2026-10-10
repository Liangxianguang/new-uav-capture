"""Fresh frozen-public-mediator response comparison, original control unchanged."""
import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v11'),str(ROOT.parent/'cwm_v10'),str(ROOT.parent/'cwm_v9'),str(ROOT.parent/'cwm_v8')]
from mediated_model import FrozenMediator,LocalTwoHead,attach_public_estimates,pack,physical_loss,state_digest
from data_audit_v15 import audit_data
from train_two_head import normalization,metrics,control_metrics
from local_shadow import restore_public_call,rank_metrics
from local_shadow_release import frozen_configs
from task_score import original_scores
from two_head_release import audit_sources
from geometry_release import arrays,compare_arrays,sha,BASELINE_SHA
from verify_release import check_archive
from s4_value import descriptive_bootstrap


def core_kind(name):
    if name == 'motion_only':
        return name
    if name in ('raw_plain','mediated_plain'):
        return 'plain'
    raise ValueError('Unexpected V15 configuration')


def evaluate(model,name,calls,indices,mean,scale):
    model.eval()
    inputs,slices = pack(calls,indices,mean,scale,name == 'mediated_plain')
    with torch.no_grad():
        values = [v.numpy() for v in model(*inputs)]
    outputs = [{key:v[start:end].copy() for key,v in zip(('prediction','motion','response'),values)} for start,end in slices]
    return metrics(calls,indices,outputs),outputs


def prepare_decisions(calls,indices,read,configs):
    context = {}
    for i in indices:
        row,values = calls[i]['record'],calls[i]['values']
        full = bool(values['valid'].all() and values['anchor_valid'].all())
        if full != row['all_candidates_full_horizon_valid']:
            raise ValueError('Actual full-support flag mismatch')
        if not full:
            continue
        raw = read('data/'+row['context_path'])
        if hashlib.sha256(raw).hexdigest() != row['context_sha256']:
            raise ValueError('Delayed original context changed')
        call = restore_public_call(json.loads(raw),*configs,values['backbone'])
        actions = values['proposed'][:,:,row['agent']]
        truth = original_scores(call,actions,values['target'])
        if not np.allclose(truth,row['costs']['action_specific_truth'],rtol=1e-10,atol=1e-8):
            raise ValueError('Original full-engine truth cost mismatch')
        context[i] = call
    if {calls[i]['record']['group'] for i in context} != {calls[i]['record']['group'] for i in indices}:
        raise ValueError('Missing complete decision support in development group')
    return context


def decision_records(calls,indices,outputs,contexts):
    records = []
    for i,output in zip(indices,outputs):
        if i not in contexts:
            continue
        row,values = calls[i]['record'],calls[i]['values']
        actions = values['proposed'][:,:,row['agent']]
        backbone = np.repeat(values['backbone'][None],len(actions),0)
        costs = {'model':original_scores(contexts[i],actions,output['prediction']).tolist(),
                 'own_motion_component':original_scores(contexts[i],actions,backbone+output['motion']).tolist(),
                 'constant_velocity':row['costs']['constant_velocity'],'original_gru':row['costs']['original_gru'],
                 'action_specific_truth':row['costs']['action_specific_truth']}
        truth = costs['action_specific_truth']
        records.append({**{k:row[k] for k in ('episode_index','step','agent','group')},'costs':costs,
                        'metrics':{k:rank_metrics(v,truth) for k,v in costs.items()}})
    return records


def qualification(models,controls,decisions,protocol):
    selected = {}
    for name in protocol['models']:
        rows = [r for r in models if r['configuration'] == name]
        if len(rows) != 3 or {r['seed'] for r in rows} != set(protocol['training']['seeds']):
            raise ValueError('Fixed three-seed population required')
        selected[name] = sorted(rows,key=lambda r:(r['development']['group_equal_ade_m'],r['seed']))[1]['seed']
    rows = [r for r in models if r['configuration'] == 'mediated_plain']
    raw_rows = [r for r in models if r['configuration'] == 'raw_plain']
    gate = protocol['development_gate']
    middle = lambda items,key: float(np.median([r['development'][key] for r in items]))
    ade = 'group_equal_ade_m'
    response = 'group_equal_paired_response_error_m'
    checks = {
        'each_seed_ade_vs_gru':all(r['development'][ade] <= controls['frozen_gru'][ade]*gate['maximum_each_seed_ade_ratio_vs_gru'] for r in rows),
        'median_ade_vs_cv':middle(rows,ade) <= controls['constant_velocity'][ade]*gate['maximum_median_ade_ratio_vs_cv'],
        'median_response_vs_zero':middle(rows,response) <= controls['frozen_gru'][response]*gate['maximum_median_response_error_ratio_vs_zero'],
        'median_response_vs_raw':middle(rows,response) <= middle(raw_rows,response)*gate['maximum_median_response_error_ratio_vs_raw'],
        'anchor_response_exact_zero':all(r['development']['anchor_response_exact_zero'] for r in models)}
    chosen = decisions[f"mediated_plain_seed{selected['mediated_plain']}"]
    comparisons = {k:decisions[f"{k}_seed{selected[k]}"] for k in ('motion_only','raw_plain')}
    identity = lambda data:[tuple(r[k] for k in ('episode_index','step','agent','group')) for r in data]
    if not chosen or any(identity(data) != identity(chosen) for data in comparisons.values()):
        raise ValueError('Decision comparison must use same complete support')
    gains = {}
    for reference in ('own_motion_component','cv','motion_only','raw'):
        if reference == 'motion_only' or reference == 'raw':
            name = 'raw_plain' if reference == 'raw' else reference
            base = [r['metrics']['model']['regret_in_diagnostic_cost'] for r in comparisons[name]]
        else:
            name = 'constant_velocity' if reference == 'cv' else reference
            base = [r['metrics'][name]['regret_in_diagnostic_cost'] for r in chosen]
        differences = [a-r['metrics']['model']['regret_in_diagnostic_cost'] for a,r in zip(base,chosen)]
        measured = descriptive_bootstrap(differences,[r['group'] for r in chosen],protocol['decision_bootstrap']['draws'],protocol['decision_bootstrap']['seed'])
        measured['interpretation'] = protocol['decision_bootstrap']['interpretation']
        gains[reference] = measured
        key = 'own_motion' if reference == 'own_motion_component' else reference
        checks['gain_vs_'+reference] = measured['percentile_95_interval'][0] > gate['minimum_selected_group_gain_lower_bound_vs_'+key]
    return selected,{'mediated_plain':{'research_eligible':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
                                     'selected_seed':selected['mediated_plain'],'selected_gains':gains,'online_promoted':False}}


def read_dataset(data,public):
    def read(name):
        stage,relative = name.split('/',1)
        return ((data if stage == 'data' else public)/relative).read_bytes()
    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','public','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--capsule',type=Path,default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    parser.add_argument('--candidate-source',type=Path,default=ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip')
    parser.add_argument('--mediator',type=Path,default=ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    data_protocol = json.loads((ROOT/'data_protocol.json').read_text())
    if any(protocol[k] for k in ('enhanced_control_enabled','original_gru_in_optimizer','mediator_in_optimizer','private_labels_model_inputs','holdout_used','prior_gate_override_allowed')):
        raise ValueError('Frozen original/public-only offline contract violated')
    if sha(args.capsule) != BASELINE_SHA or sha(args.candidate_source) != data_protocol['source_archive_sha256']:
        raise ValueError('Original/candidate source mismatch')
    for path in (args.candidate_source,args.mediator):
        check_archive(path,'ARTIFACT_MANIFEST.json',False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    configs = frozen_configs(args.capsule,args.output/'restored')
    read = read_dataset(args.data,args.public)
    with zipfile.ZipFile(args.candidate_source) as source:
        calls,report = audit_data(read,source.read,configs,args.output/'restored')
    indices = {split:[i for i,c in enumerate(calls) if c['record']['split'] == split] for split in ('train','development_validation')}
    counts = {k:sum(calls[i]['record']['all_candidates_full_horizon_valid'] for i in v) for k,v in indices.items()}
    if any(counts[k] < protocol['additional_data_gate']['minimum_full_support_'+('train' if k == 'train' else 'development')+'_calls'] for k in counts):
        raise ValueError('Insufficient complete paired decision support')
    mediator = FrozenMediator(args.mediator,protocol)
    diagnostics = attach_public_estimates(calls,mediator)
    np.savez_compressed(args.output/'public_estimates.npz',**{f'call{i}':c['estimated_commands'] for i,c in enumerate(calls)})
    (args.output/'mediator_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    mean,scale = normalization(calls,indices['train'])
    np.savez_compressed(args.output/'normalization.npz',mean=mean,scale=scale)
    contexts = prepare_decisions(calls,indices['development_validation'],read,configs)
    controls = control_metrics(calls,indices['development_validation'])
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                      and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {ROOT/'data_protocol.json',ROOT/'training_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for p in sources:
        dest = args.output/'source'/p.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    print(json.dumps({'status':'fresh_original_public_mediator_and_cost_audit_passed','calls':{k:len(v) for k,v in indices.items()},'full_support':counts}),flush=True)
    group_counts = collections.Counter(calls[i]['record']['group'] for i in indices['train'])
    config = protocol['training']
    models,decisions = [],{}
    for seed in config['seeds']:
        for name in protocol['models']:
            torch.manual_seed(seed)
            model = LocalTwoHead(core_kind(name),config['motion_scale_m'],config['response_scale_m'])
            initial = state_digest(model)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
            sampler = np.random.default_rng(seed)
            history = []
            for epoch in range(1,config['epochs']+1):
                model.train()
                order = sampler.permutation(indices['train'])
                loss_rows = []
                for start in range(0,len(order),config['batch_calls']):
                    batch = order[start:start+config['batch_calls']]
                    loss,terms = physical_loss(model,name,calls,batch,mean,scale,group_counts,config)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite mediated response loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
                    optimizer.step()
                    loss_rows.append([float(loss.detach()),*[float(v.detach()) for v in terms]])
                history.append({'epoch':epoch,'mean_batch_loss_motion_response_energy':np.mean(loss_rows,0).tolist()})
                if epoch % 20 == 0:
                    print(json.dumps({'configuration':name,'seed':seed,**history[-1]}),flush=True)
            measured,outputs = evaluate(model,name,calls,indices['development_validation'],mean,scale)
            identifier = f'{name}_seed{seed}'
            np.savez_compressed(args.output/f'{identifier}_predictions.npz',**{f'call{j}_{k}':v for j,o in enumerate(outputs) for k,v in o.items()})
            records = decision_records(calls,indices['development_validation'],outputs,contexts)
            decisions[identifier] = records
            (args.output/f'{identifier}_decisions.json').write_text(json.dumps(records,indent=2))
            (args.output/f'{identifier}_history.json').write_text(json.dumps(history,indent=2))
            checkpoint = {'configuration':name,'seed':seed,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                          'initial_state_digest':initial,'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':sampler.bit_generator.state,
                          'normalizer_mean':torch.from_numpy(mean),'normalizer_scale':torch.from_numpy(scale),'protocol':protocol,'source_hashes':hashes,
                          'data_summary_sha256':sha(args.data/'summary.json'),'public_summary_sha256':sha(args.public/'summary.json'),
                          'mediator_checkpoint_sha256':mediator.checkpoint_sha256,'online_promoted':False,'baseline_weights_included':False,'mediator_optimized':False}
            torch.save(checkpoint,args.output/f'{identifier}.pt')
            models.append({'configuration':name,'seed':seed,'initial_state_digest':initial,'parameters':sum(p.numel() for p in model.parameters()),
                           'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'checkpoint_sha256':sha(args.output/f'{identifier}.pt'),'development':measured})
    selected,gates = qualification(models,controls,decisions,protocol)
    if not mediator.unchanged() or any(sha(ROOT.parent/n) != d for n,d in hashes.items()):
        raise ValueError('Frozen mediator or run-used sources changed')
    summary = {'status':'offline_mediated_response_finished_not_promoted','protocol':protocol,'source_hashes':hashes,
               'models':models,'controls':controls,'qualification':gates,'selected_median_seeds':selected,
               'split_calls':{k:len(v) for k,v in indices.items()},'split_groups':{k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()},
               'full_support_calls':counts,'data_summary_sha256':sha(args.data/'summary.json'),'public_summary_sha256':sha(args.public/'summary.json'),
               'mediator_checkpoint_sha256':mediator.checkpoint_sha256,'mediator_state_digest':mediator.digest,'mediator_unchanged':True,
               'enhanced_control_enabled':False,'baseline_weights_included':False,'mediator_optimized':False,'private_labels_model_inputs':False,
               'holdout_used':False,'prior_gate_overridden':False,'scope':protocol['later_required']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':summary['status'],'qualification':gates,'selected':selected}),flush=True)


if __name__ == '__main__':
    main()
