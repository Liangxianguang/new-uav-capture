"""Fixed two-stage geometry response training, original and common motion frozen."""
import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v15'),str(ROOT.parent/'cwm_v10')]
from frozen_motion_model import FrozenCommonResponse,LocalTwoHead,response_loss
from data_audit import audit_data,read_dataset
from mediated_model import FrozenMediator,attach_public_estimates,pack,state_digest
from train_two_head import normalization,metrics,batch_loss
from local_shadow_release import frozen_configs
from task_score import original_scores
from local_shadow import restore_public_call,rank_metrics
from geometry_release import sha,BASELINE_SHA
from s4_value import descriptive_bootstrap


def model_inputs(calls,indices,mean,scale,mediated=False):
    raw,slices = pack(calls,indices,mean,scale,False)
    if not mediated:
        return raw,slices
    estimates,_ = pack(calls,indices,mean,scale,True)
    return (*raw,estimates[2],estimates[3]),slices


def evaluate(model,name,calls,indices,mean,scale):
    model.eval()
    inputs,slices = model_inputs(calls,indices,mean,scale,name == 'mediated_response')
    with torch.no_grad():
        values = [v.numpy() for v in model(*inputs)]
    outputs = [{key:v[start:end].copy() for key,v in zip(('prediction','motion','response'),values)} for start,end in slices]
    return metrics(calls,indices,outputs),outputs


def control_metrics(calls,indices):
    controls = {}
    for name in ('frozen_gru','constant_velocity'):
        outputs = []
        for i in indices:
            values = calls[i]['values']
            backbone = np.repeat(values['backbone'][None],len(values['proposed']),0)
            motion,response = np.zeros_like(backbone),np.zeros_like(backbone)
            prediction = backbone.copy()
            if name == 'constant_velocity':
                path = values['reference'][None]+.1*np.arange(1,9)[:,None]*values['velocity'][None]
                prediction = np.repeat(path[None],len(backbone),0)
                motion = prediction-backbone
            outputs.append({'prediction':prediction,'motion':motion,'response':response})
        controls[name] = metrics(calls,indices,outputs)
    return controls


def prepare_contexts(calls,indices,read,configs):
    contexts = {}
    for i in indices:
        row,values = calls[i]['record'],calls[i]['values']
        if not (values['valid'].all() and values['anchor_valid'].all()):
            continue
        contexts[i] = restore_public_call(json.loads(read('data/'+row['context_path'])),*configs,values['backbone'])
    if {calls[i]['record']['group'] for i in contexts} != {calls[i]['record']['group'] for i in indices}:
        raise ValueError('No complete decision support in some development groups')
    return contexts


def decision_records(calls,indices,outputs,contexts):
    records = []
    for i,output in zip(indices,outputs):
        if i not in contexts:
            continue
        row,values = calls[i]['record'],calls[i]['values']
        actions = values['proposed'][:,:,row['agent']]
        backbone = np.repeat(values['backbone'][None],len(actions),0)
        cv = values['reference'][None]+.1*np.arange(1,9)[:,None]*values['velocity'][None]
        paths = {'model':output['prediction'],'own_motion_component':backbone+output['motion'],
                 'original_gru':backbone,'constant_velocity':np.repeat(cv[None],len(actions),0),
                 'action_specific_truth':values['target']}
        costs = {k:original_scores(contexts[i],actions,v).tolist() for k,v in paths.items()}
        records.append({**{k:row[k] for k in ('episode_index','step','agent','group')},'costs':costs,
                        'metrics':{k:rank_metrics(v,costs['action_specific_truth']) for k,v in costs.items()}})
    return records


def qualification(models,controls,decisions,protocol):
    selected = {}
    for name in protocol['models']:
        rows = [r for r in models if r['configuration'] == name]
        if len(rows) != 3 or {r['seed'] for r in rows} != set(protocol['training']['seeds']):
            raise ValueError('Fixed3-seed population required')
        selected[name] = sorted(rows,key=lambda r:(r['development']['group_equal_ade_m'],r['seed']))[1]['seed']
    gates = {}
    gate = protocol['development_gate']
    middle = lambda rows,key:float(np.median([r['development'][key] for r in rows]))
    raw_rows = [r for r in models if r['configuration'] == 'raw_response']
    for name in ('raw_response','mediated_response'):
        rows = [r for r in models if r['configuration'] == name]
        response_key = 'group_equal_paired_response_error_m'
        checks = {
            'each_seed_ade_vs_gru':all(r['development']['group_equal_ade_m'] <= controls['frozen_gru']['group_equal_ade_m']*gate['maximum_each_seed_ade_ratio_vs_gru'] for r in rows),
            'median_ade_vs_cv':middle(rows,'group_equal_ade_m') <= controls['constant_velocity']['group_equal_ade_m']*gate['maximum_median_ade_ratio_vs_cv'],
            'median_response_vs_zero':middle(rows,response_key) <= controls['frozen_gru'][response_key]*gate['maximum_median_response_error_ratio_vs_zero'],
            'anchor_response_exact_zero':all(r['development']['anchor_response_exact_zero'] for r in rows)}
        if name == 'mediated_response':
            checks['median_response_vs_raw'] = middle(rows,response_key) <= middle(raw_rows,response_key)*gate['maximum_median_mediated_response_error_ratio_vs_raw']
        chosen = decisions[f'{name}_seed{selected[name]}']
        references = ['own_motion','cv','motion_only']+(['raw'] if name == 'mediated_response' else [])
        gains = {}
        identity = lambda items:[tuple(r[k] for k in ('episode_index','step','agent','group')) for r in items]
        for reference in references:
            if reference in ('own_motion','cv'):
                key = 'own_motion_component' if reference == 'own_motion' else 'constant_velocity'
                base = [r['metrics'][key]['regret_in_diagnostic_cost'] for r in chosen]
            else:
                ref_name = 'raw_response' if reference == 'raw' else 'motion_only'
                records = decisions[f'{ref_name}_seed{selected[ref_name]}']
                if identity(records) != identity(chosen):
                    raise ValueError('Decision comparison support differs')
                base = [r['metrics']['model']['regret_in_diagnostic_cost'] for r in records]
            differences = [a-r['metrics']['model']['regret_in_diagnostic_cost'] for a,r in zip(base,chosen)]
            measured = descriptive_bootstrap(differences,[r['group'] for r in chosen],protocol['decision_bootstrap']['draws'],protocol['decision_bootstrap']['seed'])
            measured['interpretation'] = protocol['decision_bootstrap']['interpretation']
            gains[reference] = measured
            threshold = gate['minimum_selected_mediated_gain_lower_bound_vs_raw'] if reference == 'raw' else gate['minimum_selected_gain_lower_bound_vs_'+reference]
            checks['gain_vs_'+reference] = measured['percentile_95_interval'][0] > threshold
        gates[name] = {'research_eligible':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
                       'selected_seed':selected[name],'selected_gains':gains,'online_promoted':False}
    return selected,gates


def run_epochs(model,optimizer,name,calls,indices,mean,scale,counts,config):
    epochs = config['motion_epochs'] if name == 'motion_only' else config['response_epochs']
    sampler = np.random.default_rng(config['seed'])
    history = []
    for epoch in range(1,epochs+1):
        model.train()
        order = sampler.permutation(indices)
        losses = []
        for start in range(0,len(order),config['batch_calls']):
            batch = order[start:start+config['batch_calls']]
            if name == 'motion_only':
                loss,terms = batch_loss(model,calls,batch,mean,scale,counts,
                    {**config,'motion_weight':1.,'response_weight':0.})
            else:
                inputs,slices = model_inputs(calls,batch,mean,scale,name == 'mediated_response')
                _,_,response = model(*inputs)
                loss,terms = response_loss(response,calls,batch,slices,counts,config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fixed-budget stage loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['gradient_clip_norm'])
            optimizer.step()
            losses.append([float(loss.detach()),*[float(t.detach()) for t in terms]])
        history.append({'epoch':epoch,'mean_batch_loss_terms':np.mean(losses,0).tolist()})
        if epoch % 20 == 0:
            print(json.dumps({'configuration':name,'seed':config['seed'],**history[-1]}),flush=True)
    return history,sampler.bit_generator.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','public','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    if any(protocol[k] for k in ('enhanced_control_enabled','original_gru_in_optimizer','mediator_in_optimizer',
                                 'common_motion_in_response_optimizer','private_labels_model_inputs','holdout_used','prior_gate_override_allowed')):
        raise ValueError('Frozen original/common-motion offline contract violated')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    capsule = ROOT.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original baseline capsule changed')
    configs = frozen_configs(capsule,args.output/'restored')
    read = read_dataset(args.data,args.public)
    calls,data_report = audit_data(read,configs,args.output/'restored')
    indices = {split:[i for i,c in enumerate(calls) if c['record']['split'] == split] for split in ('train','development_validation')}
    mediator = FrozenMediator(ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip',protocol)
    diagnostics = attach_public_estimates(calls,mediator)
    np.savez_compressed(args.output/'public_estimates.npz',**{f'call{i}':c['estimated_commands'] for i,c in enumerate(calls)})
    (args.output/'mediator_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    mean,scale = normalization(calls,indices['train'])
    np.savez_compressed(args.output/'normalization.npz',mean=mean,scale=scale)
    contexts = prepare_contexts(calls,indices['development_validation'],read,configs)
    controls = control_metrics(calls,indices['development_validation'])
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                      and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} |
                     {Path(__file__).resolve(),ROOT/'data_protocol.json',ROOT/'training_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        dest = args.output/'source'/source.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    print(json.dumps({'status':'independent_public_geometry_original_cost_data_audit_passed',
                      'calls':{k:len(v) for k,v in indices.items()},'full_dev_support':len(contexts)}),flush=True)
    counts = collections.Counter(calls[i]['record']['group'] for i in indices['train'])
    config = protocol['training']
    models,decisions = [],{}
    for seed in config['seeds']:
        torch.manual_seed(seed)
        common = LocalTwoHead('motion_only',config['motion_scale_m'],config['response_scale_m'])
        motion_initial = state_digest(common)
        optimizer = torch.optim.Adam([p for p in common.parameters() if p.requires_grad],lr=config['learning_rate'])
        history,rng = run_epochs(common,optimizer,'motion_only',calls,indices['train'],mean,scale,counts,{**config,'seed':seed})
        motion_checkpoint = {'model_state':copy.deepcopy(common.state_dict()),'optimizer_state':copy.deepcopy(optimizer.state_dict()),
                             'initial_state_digest':motion_initial,'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':rng}
        common.eval().requires_grad_(False)
        # Old training gradients must be cleared as well as requires_grad flags.
        for parameter in common.parameters():
            parameter.grad = None
        common_digest = state_digest(common)
        for name in protocol['models']:
            if name == 'motion_only':
                model = common
                trained = motion_checkpoint
                this_history = history
                initial = motion_initial
            else:
                torch.manual_seed(seed)
                response_core = LocalTwoHead('plain',config['motion_scale_m'],config['response_scale_m'])
                model = FrozenCommonResponse(copy.deepcopy(common),response_core,name == 'mediated_response')
                initial = state_digest(response_core)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
                common_ids = {id(p) for p in model.common.parameters()}
                if any(id(p) in common_ids for group in optimizer.param_groups for p in group['params']):
                    raise ValueError('Common motion entered response optimizer')
                this_history,rng = run_epochs(model,optimizer,name,calls,indices['train'],mean,scale,counts,{**config,'seed':seed})
                if state_digest(model.common) != common_digest or any(p.grad is not None or p.requires_grad for p in model.common.parameters()):
                    raise ValueError('Frozen common motion changed during response training')
                trained = {'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                           'initial_state_digest':initial,'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':rng}
            measured,outputs = evaluate(model,name,calls,indices['development_validation'],mean,scale)
            identifier = f'{name}_seed{seed}'
            np.savez_compressed(args.output/f'{identifier}_predictions.npz',**{f'call{j}_{k}':v for j,o in enumerate(outputs) for k,v in o.items()})
            decisions[identifier] = decision_records(calls,indices['development_validation'],outputs,contexts)
            (args.output/f'{identifier}_decisions.json').write_text(json.dumps(decisions[identifier],indent=2))
            (args.output/f'{identifier}_history.json').write_text(json.dumps(this_history,indent=2))
            checkpoint = {**trained,'configuration':name,'seed':seed,'common_motion_state_digest':common_digest,
                          'normalizer_mean':torch.from_numpy(mean),'normalizer_scale':torch.from_numpy(scale),
                          'protocol':protocol,'source_hashes':hashes,'data_summary_sha256':sha(args.data/'summary.json'),
                          'public_summary_sha256':sha(args.public/'summary.json'),'mediator_checkpoint_sha256':mediator.checkpoint_sha256,
                          'online_promoted':False,'baseline_weights_included':False,'mediator_optimized':False,
                          'common_motion_in_response_optimizer':False}
            torch.save(checkpoint,args.output/f'{identifier}.pt')
            models.append({'configuration':name,'seed':seed,'initial_state_digest':initial,
                           'common_motion_state_digest':common_digest,'parameters':sum(p.numel() for p in model.parameters()),
                           'checkpoint_sha256':sha(args.output/f'{identifier}.pt'),'development':measured})
    selected,gates = qualification(models,controls,decisions,protocol)
    if not mediator.unchanged() or any(sha(ROOT.parent/n) != d for n,d in hashes.items()):
        raise ValueError('Frozen mediator or run-used training source changed')
    summary = {'status':'offline_frozen_common_motion_response_finished_not_promoted','protocol':protocol,'source_hashes':hashes,
               'models':models,'controls':controls,'qualification':gates,'selected_median_seeds':selected,
               'split_calls':{k:len(v) for k,v in indices.items()},'split_groups':{k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()},
               'full_development_calls':len(contexts),'data_summary_sha256':sha(args.data/'summary.json'),
               'public_summary_sha256':sha(args.public/'summary.json'),'mediator_checkpoint_sha256':mediator.checkpoint_sha256,
               'mediator_unchanged':True,'enhanced_control_enabled':False,'baseline_weights_included':False,
               'holdout_used':False,'prior_gate_overridden':False,'common_motion_in_response_optimizer':False,'scope':protocol['later_required']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':summary['status'],'qualification':gates,'selected':selected}),flush=True)


if __name__ == '__main__':
    main()
