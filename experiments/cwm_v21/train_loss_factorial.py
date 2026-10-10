"""Matched raw-action response loss/weighting experiment on fresh audited groups."""
import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'cwm_v19'))
from train_frozen_response import (FrozenCommonResponse, LocalTwoHead, model_inputs, evaluate,
    control_metrics, prepare_contexts, decision_records, normalization, batch_loss,
    state_digest, frozen_configs, sha, BASELINE_SHA, descriptive_bootstrap)
from loss_data_audit import audit_data, read_dataset
from factorial_loss import population_weights, factorial_response_loss
sys.path.insert(0,str(ROOT.parent/'cwm_v20'))
from response_diagnostics import decompose


def validate_training(protocol):
    if protocol != json.loads((ROOT/'training_protocol.json').read_text()):
        raise ValueError('Published V21 training protocol changed')
    if any(protocol[k] for k in ('enhanced_control_enabled','original_gru_in_optimizer','mediator_in_optimizer',
        'common_motion_in_response_optimizer','private_labels_model_inputs','holdout_used','prior_gate_override_allowed')):
        raise ValueError('Frozen original-only optional research contract violated')
    if protocol['primary_configuration'] != 'raw_point_l2' or len(protocol['models']) != 5:
        raise ValueError('Fixed primary/factorial population changed')


def selected_seeds(models, protocol):
    if len(models) != len(protocol['models'])*len(protocol['training']['seeds']):
        raise ValueError('Incomplete final model population')
    selected = {}
    for name in protocol['models']:
        rows = [r for r in models if r['configuration'] == name]
        if len(rows) != 3 or {r['seed'] for r in rows} != set(protocol['training']['seeds']):
            raise ValueError('Incomplete fixed three-seed population')
        selected[name] = sorted(rows,key=lambda r:(r['development']['group_equal_ade_m'],r['seed']))[1]['seed']
    return selected


def compare_decisions(chosen, baseline, protocol, baseline_key='model'):
    identity = lambda rows:[tuple(r[k] for k in ('episode_index','step','agent','group')) for r in rows]
    if not chosen or identity(chosen) != identity(baseline):
        raise ValueError('Original-cost decision support differs or empty')
    gains = [b['metrics'][baseline_key]['regret_in_diagnostic_cost']-r['metrics']['model']['regret_in_diagnostic_cost']
             for b,r in zip(baseline,chosen)]
    measured = descriptive_bootstrap(gains,[r['group'] for r in chosen],
        protocol['decision_bootstrap']['draws'],protocol['decision_bootstrap']['seed'])
    measured['interpretation'] = protocol['decision_bootstrap']['interpretation']
    return measured


def qualification(models, controls, decisions, protocol):
    validate_training(protocol)
    selected = selected_seeds(models,protocol)
    middle = lambda name,key:float(np.median([r['development'][key] for r in models if r['configuration'] == name]))
    thresholds = protocol['development_gate']
    response_key = 'group_equal_paired_response_error_m'
    gates = {}
    for name in protocol['response_configurations']:
        rows = [r for r in models if r['configuration'] == name]
        checks = {'each_seed_ade_vs_gru':all(r['development']['group_equal_ade_m'] <=
                    controls['frozen_gru']['group_equal_ade_m']*thresholds['maximum_each_seed_ade_ratio_vs_gru'] for r in rows),
                  'median_ade_vs_cv':middle(name,'group_equal_ade_m') <=
                    controls['constant_velocity']['group_equal_ade_m']*thresholds['maximum_median_ade_ratio_vs_cv'],
                  'median_response_vs_zero':middle(name,response_key) <=
                    controls['frozen_gru'][response_key]*thresholds['maximum_median_response_error_ratio_vs_zero'],
                  'anchor_response_exact_zero':all(r['development']['anchor_response_exact_zero'] for r in rows)}
        chosen = decisions[f'{name}_seed{selected[name]}']
        gains = {}
        for reference in ('own_motion','cv','motion_only'):
            if reference == 'motion_only':
                baseline = decisions[f'motion_only_seed{selected["motion_only"]}']
                key = 'model'
            else:
                baseline = chosen
                key = 'own_motion_component' if reference == 'own_motion' else 'constant_velocity'
            gains[reference] = compare_decisions(chosen,baseline,protocol,key)
            checks['gain_vs_'+reference] = gains[reference]['percentile_95_interval'][0] > thresholds['minimum_selected_gain_lower_bound_vs_'+reference]
        if name == protocol['primary_configuration']:
            control = 'raw_call_mse'
            gains[control] = compare_decisions(chosen,decisions[f'{control}_seed{selected[control]}'],protocol)
            checks['median_response_vs_call_mse'] = middle(name,response_key) <= middle(control,response_key)
            checks['gain_vs_call_mse'] = gains[control]['percentile_95_interval'][0] > 0.
        gates[name] = {'research_eligible':bool(all(checks.values())), 'checks':{k:bool(v) for k,v in checks.items()},
                       'selected_seed':selected[name],'selected_gains':gains,'online_promoted':False}
    factorial = {}
    for changed,control in protocol['factorial_comparisons']:
        factorial[f'{changed}_vs_{control}'] = {
            'all_seed_response_gain_m':{str(seed):next(r for r in models if r['configuration'] == control and r['seed'] == seed)['development'][response_key]
                - next(r for r in models if r['configuration'] == changed and r['seed'] == seed)['development'][response_key] for seed in protocol['training']['seeds']},
            'selected_decision_gain':compare_decisions(decisions[f'{changed}_seed{selected[changed]}'],
                decisions[f'{control}_seed{selected[control]}'],protocol), 'descriptive_only':True}
    return selected,gates,factorial


def run_epochs(model, optimizer, name, calls, indices, mean, scale, weights, counts, config, protocol):
    epochs = config['motion_epochs' if name == 'motion_only' else 'response_epochs']
    sampler = np.random.default_rng(config['seed'])
    history = []
    for epoch in range(1,epochs+1):
        model.train()
        losses = []
        order = sampler.permutation(indices)
        for start in range(0,len(order),config['batch_calls']):
            chosen = order[start:start+config['batch_calls']]
            if name == 'motion_only':
                loss,terms = batch_loss(model,calls,chosen,mean,scale,counts,
                    {**config,'motion_weight':1.,'response_weight':0.})
            else:
                inputs,slices = model_inputs(calls,chosen,mean,scale,False)
                _,_,response = model(*inputs)
                loss,terms = factorial_response_loss(response,calls,chosen,slices,weights,
                    protocol['response_configurations'][name],config['residual_penalty'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fixed-budget loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['gradient_clip_norm'])
            optimizer.step()
            losses.append([float(loss.detach()),*[float(t.detach()) for t in terms]])
        history.append({'epoch':epoch,'mean_batch_loss_terms':np.mean(losses,0).tolist()})
        if epoch % 20 == 0:
            print(json.dumps({'model':name,'seed':config['seed'],**history[-1]}),flush=True)
    return history,sampler.bit_generator.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','public','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    validate_training(protocol)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    capsule = ROOT.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original capsule changed')
    configs = frozen_configs(capsule,args.output/'restored')
    read = read_dataset(args.data,args.public)
    calls,data_report = audit_data(read,configs,args.output/'restored')
    indices = {split:[i for i,c in enumerate(calls) if c['record']['split'] == split] for split in ('train','development_validation')}
    weights,population = population_weights(calls,indices['train'])
    (args.output/'population_weights.json').write_text(json.dumps({'population':population,'weights':weights},indent=2))
    mean,scale = normalization(calls,indices['train'])
    np.savez_compressed(args.output/'normalization.npz',mean=mean,scale=scale)
    contexts = prepare_contexts(calls,indices['development_validation'],read,configs)
    controls = control_metrics(calls,indices['development_validation'])
    diagnostic = json.loads((ROOT.parent/'cwm_v20/diagnostic_protocol.json').read_text())
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} |
        {Path(__file__).resolve(),ROOT/'data_protocol.json',ROOT/'training_protocol.json',ROOT.parent/'cwm_v20/diagnostic_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        dest = args.output/'source'/source.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    print(json.dumps({'status':'fresh_v21_full_public_original_cost_audit_passed',
                      'calls':{k:len(v) for k,v in indices.items()},'full_dev_support':len(contexts)}),flush=True)
    config = protocol['training']
    counts = collections.Counter(calls[i]['record']['group'] for i in indices['train'])
    models,decisions = [],{}
    for seed in config['seeds']:
        torch.manual_seed(seed)
        common = LocalTwoHead('motion_only',config['motion_scale_m'],config['response_scale_m'])
        initial_motion = state_digest(common)
        optimizer = torch.optim.Adam([p for p in common.parameters() if p.requires_grad],lr=config['learning_rate'])
        history,rng = run_epochs(common,optimizer,'motion_only',calls,indices['train'],mean,scale,weights,counts,{**config,'seed':seed},protocol)
        motion_state = {'model_state':copy.deepcopy(common.state_dict()),'optimizer_state':copy.deepcopy(optimizer.state_dict()),
            'initial_state_digest':initial_motion,'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':rng}
        common.eval().requires_grad_(False)
        for p in common.parameters():
            p.grad = None
        common_digest = state_digest(common)
        for name in protocol['models']:
            if name == 'motion_only':
                model,trained,used_history,initial = common,motion_state,history,initial_motion
            else:
                torch.manual_seed(seed)
                core = LocalTwoHead('plain',config['motion_scale_m'],config['response_scale_m'])
                model = FrozenCommonResponse(copy.deepcopy(common),core,False)
                initial = state_digest(core)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
                if {id(p) for p in model.common.parameters()} & {id(p) for g in optimizer.param_groups for p in g['params']}:
                    raise ValueError('Frozen common motion entered response optimizer')
                used_history,rng = run_epochs(model,optimizer,name,calls,indices['train'],mean,scale,weights,counts,{**config,'seed':seed},protocol)
                if state_digest(model.common) != common_digest or any(p.grad is not None or p.requires_grad for p in model.common.parameters()):
                    raise ValueError('Frozen common motion mutated')
                trained = {'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'initial_state_digest':initial,
                           'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':rng}
            identifier = f'{name}_seed{seed}'
            split_metrics,split_diagnostics = {},{}
            for split,chosen in indices.items():
                measured,outputs = evaluate(model,name,calls,chosen,mean,scale)
                np.savez_compressed(args.output/f'{identifier}_{split}_predictions.npz',
                    **{f'call{j}_{k}':v for j,o in enumerate(outputs) for k,v in o.items()})
                split_metrics[split] = measured
                split_diagnostics[split] = decompose(calls,chosen,outputs,diagnostic)
                if not np.isclose(split_diagnostics[split]['overall']['group_equal_response_error_m'],
                                  measured['group_equal_paired_response_error_m'],rtol=0,atol=1e-12):
                    raise ValueError('Diagnostic support differs from gate metric')
                if split == 'development_validation':
                    decisions[identifier] = decision_records(calls,chosen,outputs,contexts)
            (args.output/f'{identifier}_decisions.json').write_text(json.dumps(decisions[identifier],indent=2))
            (args.output/f'{identifier}_history.json').write_text(json.dumps(used_history,indent=2))
            (args.output/f'{identifier}_diagnostics.json').write_text(json.dumps(split_diagnostics,indent=2,allow_nan=False))
            checkpoint = {**trained,'configuration':name,'seed':seed,'common_motion_state_digest':common_digest,
                'normalizer_mean':torch.from_numpy(mean),'normalizer_scale':torch.from_numpy(scale),'protocol':protocol,
                'source_hashes':hashes,'data_summary_sha256':sha(args.data/'summary.json'),
                'public_summary_sha256':sha(args.public/'summary.json'),'population_weights_sha256':sha(args.output/'population_weights.json'),
                'online_promoted':False,'baseline_weights_included':False,'mediator_optimized':False,
                'common_motion_in_response_optimizer':False}
            torch.save(checkpoint,args.output/f'{identifier}.pt')
            row = {'configuration':name,'seed':seed,'initial_state_digest':initial,'common_motion_state_digest':common_digest,
                'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(args.output/f'{identifier}.pt'),
                'train':split_metrics['train'],'development':split_metrics['development_validation']}
            models.append(row)
            print(json.dumps({'status':'final_model_measured','model':identifier,'development':row['development']['group_equal_paired_response_error_m']}),flush=True)
    selected,gates,factorial = qualification(models,controls,decisions,protocol)
    if any(sha(ROOT.parent/n) != digest for n,digest in hashes.items()):
        raise ValueError('Run-used sources changed')
    summary = {'status':'offline_fresh_response_loss_factorial_finished_not_promoted','protocol':protocol,'source_hashes':hashes,
        'models':models,'controls':controls,'qualification':gates,'factorial_comparisons':factorial,
        'primary_research_eligible':gates[protocol['primary_configuration']]['research_eligible'],'selected_median_seeds':selected,
        'split_calls':{k:len(v) for k,v in indices.items()},
        'split_groups':{k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()},
        'full_development_calls':len(contexts),'data_summary_sha256':sha(args.data/'summary.json'),
        'public_summary_sha256':sha(args.public/'summary.json'),'population_weights_sha256':sha(args.output/'population_weights.json'),
        'enhanced_control_enabled':False,'baseline_weights_included':False,'holdout_used':False,'prior_gate_overridden':False,
        'common_motion_in_response_optimizer':False,'scope':protocol['later_required']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({'status':summary['status'],'qualification':gates,'selected':selected}),flush=True)


if __name__ == '__main__':
    main()
