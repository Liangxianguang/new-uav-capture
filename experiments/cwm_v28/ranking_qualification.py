"""Fixed final-epoch, ADE-median, group-equal V28 offline decision gates."""
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cwm_v8'))
from s4_value import descriptive_bootstrap
from decision_value import rank_metrics

IDENTITY=('episode_index','step','ordinal','agent','group')


def validated_choice(row,method):
    if len(row['indices'])!=len(set(row['indices'])):
        raise ValueError('Unique candidate population required')
    expected=rank_metrics(row['costs'][method],row['costs']['action_specific_truth'],1e-9)
    if row['metrics'][method]!=expected:
        raise ValueError('Saved choice/cost/regret does not match original cost vectors')
    return expected['realized_cost_at_choice']


def validate_training_protocol(protocol):
    if protocol!=json.loads((HERE/'training_protocol.json').read_text()):
        raise ValueError('Published fixed V28 training protocol differs')
    if (protocol['primary_configuration']!='cv_rank_l2' or protocol['models']!=['cv_motion_only','cv_l2','cv_rank_l2'] or
        protocol['training']['seeds']!=[994101,994102,994103] or
        any(protocol[k] for k in ('enhanced_control_enabled','original_gru_in_optimizer',
            'common_motion_in_response_optimizer','private_labels_model_inputs','holdout_used','prior_gate_override_allowed'))):
        raise ValueError('Original/default-off/fixed-population contract violated')


def compare_decisions(changed,baseline,protocol,key='model'):
    if (not changed or [tuple(r[k] for k in IDENTITY) for r in changed]!=[tuple(r[k] for k in IDENTITY) for r in baseline] or
        any(r['population']!='bounded_shared' for r in changed+baseline)):
        raise ValueError('Identical complete shared-library sequential support required')
    differences=[]
    for a,b in zip(changed,baseline):
        # Compare realized TRUE costs, never predicted costs or different minima.
        if a['indices']!=b['indices'] or a['costs']['action_specific_truth']!=b['costs']['action_specific_truth']:
            raise ValueError('Identical truth/library required for ranking gain')
        differences.append(validated_choice(b,key)-validated_choice(a,'model'))
    if not np.isfinite(differences).all(): raise ValueError('Finite complete decision gains required')
    settings=protocol['decision_bootstrap']
    result=descriptive_bootstrap(differences,[r['group'] for r in changed],settings['draws'],settings['seed'])
    return {**result,'interpretation':settings['interpretation']}


def qualify(models,controls,decisions,protocol):
    validate_training_protocol(protocol)
    if len(models)!=9 or {(r['configuration'],r['seed']) for r in models}!={(m,s) for m in protocol['models'] for s in protocol['training']['seeds']}:
        raise ValueError('All nine fixed final models required')
    metrics=('group_equal_ade_m','group_equal_paired_response_error_m','group_equal_ranking_kl')
    if any(not np.isfinite(r['development'][k]) for r in models for k in metrics):
        raise ValueError('Finite all-seed final metrics required')
    selected={name:sorted([r for r in models if r['configuration']==name],
        key=lambda r:(r['development']['group_equal_ade_m'],r['seed']))[1]['seed'] for name in protocol['models']}
    middle=lambda name,key:float(np.median([r['development'][key] for r in models if r['configuration']==name]))
    gate=protocol['development_gate'];gates={}
    for name in ('cv_l2','cv_rank_l2'):
        rows=[r for r in models if r['configuration']==name]
        checks={'each_seed_ade_vs_gru':all(r['development']['group_equal_ade_m']<=controls['frozen_gru']['group_equal_ade_m']*gate['maximum_each_seed_ade_ratio_vs_gru'] for r in rows),
            'median_ade_vs_cv':middle(name,'group_equal_ade_m')<=controls['constant_velocity']['group_equal_ade_m']*gate['maximum_median_ade_ratio_vs_cv'],
            'median_response_vs_zero':middle(name,'group_equal_paired_response_error_m')<=controls['frozen_gru']['group_equal_paired_response_error_m']*gate['maximum_median_response_error_ratio_vs_zero'],
            'anchor_response_exact_zero':all(r['development']['anchor_response_exact_zero'] is True for r in rows)}
        chosen=decisions[f'{name}_seed{selected[name]}'];gains={}
        for reference in ('own_motion','cv','motion_only','original_gru'):
            if reference=='motion_only':
                base=decisions[f"cv_motion_only_seed{selected['cv_motion_only']}"];key='model'
            else:
                base=chosen;key={'own_motion':'own_motion_component','cv':'constant_velocity','original_gru':'original_gru'}[reference]
            gains[reference]=compare_decisions(chosen,base,protocol,key)
            checks['gain_vs_'+reference]=gains[reference]['percentile_95_interval'][0]>gate['minimum_selected_gain_lower_bound_vs_'+reference]
        if name=='cv_rank_l2':
            checks['median_response_vs_cv_l2']=middle(name,'group_equal_paired_response_error_m')<=middle('cv_l2','group_equal_paired_response_error_m')*gate['maximum_primary_median_response_error_ratio_vs_cv_l2']
            checks['median_ranking_kl_vs_cv_l2']=middle(name,'group_equal_ranking_kl')<=middle('cv_l2','group_equal_ranking_kl')*gate['maximum_primary_median_ranking_kl_ratio_vs_cv_l2']
            gains['cv_l2']=compare_decisions(chosen,decisions[f"cv_l2_seed{selected['cv_l2']}"],protocol)
            checks['gain_vs_cv_l2']=gains['cv_l2']['percentile_95_interval'][0]>gate['minimum_selected_gain_lower_bound_vs_cv_l2']
        gates[name]={'research_eligible':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
            'selected_seed':selected[name],'selected_gains':gains,'online_promoted':False}
    return selected,gates


def library_contributions(populations,protocol):
    """Same score across libraries, common calls, REALIZED truth costs."""
    result={}
    shared=populations['bounded_shared']
    for smaller in ('actual_unique','public_geometry'):
        base=populations[smaller]
        if not shared or [tuple(r[k] for k in IDENTITY) for r in shared]!=[tuple(r[k] for k in IDENTITY) for r in base]:
            raise ValueError('Candidate contribution requires complete common call support')
        for a,b in zip(shared,base):
            mapping={i:v for i,v in zip(a['indices'],a['costs']['action_specific_truth'])}
            if any(i not in mapping or mapping[i]!=value for i,value in zip(b['indices'],b['costs']['action_specific_truth'])):
                raise ValueError('Smaller library truth is not unchanged shared subset')
        for method in ('model','own_motion_component','original_gru','constant_velocity'):
            gain=[validated_choice(b,method)-validated_choice(a,method) for a,b in zip(shared,base)]
            if not np.isfinite(gain).all(): raise ValueError('Finite realized candidate-library cost gains required')
            settings=protocol['decision_bootstrap']
            stats=descriptive_bootstrap(gain,[r['group'] for r in shared],settings['draws'],settings['seed'])
            result[smaller+'__to__bounded_shared__'+method]={**stats,'interpretation':settings['interpretation'],
                'quantity':'realized true cost at same scoring method choice, not difference of library regrets'}
    return result
