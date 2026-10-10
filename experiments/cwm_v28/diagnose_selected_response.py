"""Descriptive choice-switch diagnosis; NEVER chooses a model or deploys a gate.

Only the existing preregistered primary/ADE-median seed is read. All saved
train/development complete shared-cost calls and groups are retained. This is
not model re-inference, original-engine re-execution, new scenes, or holdout.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ranking_qualification import validate_training_protocol, validated_choice
from repeatability_identity import digest


def diagnose(directory):
    report = json.loads((directory/'summary.json').read_bytes())
    validate_training_protocol(report['protocol'])
    if (report['status'] != 'offline_fixed_fresh_ranking_training_finished_pending_two_run_release' or
        report['enhanced_control_enabled'] is not False or report['holdout_used'] is not False):
        raise ValueError('Complete offline fixed training summary required')
    configuration = report['protocol']['primary_configuration']
    seed = report['selected_median_seeds'][configuration]
    if report['qualification'][configuration]['selected_seed'] != seed:
        raise ValueError('Fixed selection provenance differs')
    row = next(v for v in report['models'] if (v['configuration'], v['seed']) == (configuration, seed))
    identifier = f'{configuration}_seed{seed}'
    checkpoint = directory/(identifier+'.pt')
    if digest(checkpoint) != row['checkpoint_sha256']:
        raise ValueError('Fixed selected model file identity differs')
    decisions = directory/(identifier+'_decisions.json')
    inputs = {p.name: digest(p) for p in (directory/'summary.json', checkpoint, decisions)}
    saved = json.loads(decisions.read_bytes())
    splits = {}
    for split in ('train', 'development_validation'):
        groups = collections.defaultdict(list)
        records = saved[split]['bounded_shared']
        if len(records) != report['full_cost_calls'][split]:
            raise ValueError('Complete shared-cost call support required')
        identities = set()
        for call in records:
            key = tuple(call[k] for k in ('episode_index', 'step', 'ordinal', 'agent', 'group'))
            if key in identities or call['population'] != 'bounded_shared':
                raise ValueError('Unique shared-cost call identity required')
            identities.add(key)
            model_cost = validated_choice(call, 'model')
            motion_cost = validated_choice(call, 'own_motion_component')
            truth = np.asarray(call['costs']['action_specific_truth'])
            if not np.isfinite(truth).all():
                raise ValueError('Finite complete truth cost vector required')
            minimum = float(truth.min())
            gain = motion_cost-model_cost
            switched = call['metrics']['model']['choice'] != call['metrics']['own_motion_component']['choice']
            groups[call['group']].append({'true_original_cost_gain': gain,
                'motion_choice_oracle_regret': motion_cost-minimum,
                'response_choice_oracle_regret': model_cost-minimum,
                'choice_switch_fraction': float(switched),
                'beneficial_choice_fraction': float(gain > 1e-9),
                'harmful_choice_fraction': float(gain < -1e-9),
                'true_cost_tie_fraction': float(abs(gain) <= 1e-9)})
        if sorted(groups) != report['split_groups'][split]:
            raise ValueError('Complete mirror-group support required')
        by_group = [{'group': group, 'calls': len(values),
            **{name: float(np.mean([v[name] for v in values])) for name in values[0]}}
            for group, values in sorted(groups.items())]
        equal = {name: float(np.mean([v[name] for v in by_group])) for name in groups[next(iter(groups))][0]}
        splits[split] = {'calls': len(records), 'groups': len(groups), 'group_equal': equal, 'by_group': by_group}
    expected = report['qualification'][configuration]['selected_gains']['own_motion']['group_equal_mean']
    if splits['development_validation']['group_equal']['true_original_cost_gain'] != expected:
        raise ValueError('Saved fixed qualification point estimate differs')
    if inputs != {name: digest(directory/name) for name in inputs}:
        raise ValueError('Saved selected-model evidence changed during diagnosis')
    return {'status': 'complete_saved_fixed_primary_choice_switch_diagnosis_not_reinference',
        'fixed_configuration': configuration, 'fixed_ade_median_seed': seed,
        'input_sha256': inputs, 'source_sha256': {name: digest(HERE/name) for name in
            ('diagnose_selected_response.py', 'ranking_qualification.py', 'repeatability_identity.py')},
        'splits': splits, 'primary_research_eligible': report['primary_research_eligible'],
        'enhanced_control_enabled': False, 'holdout_used': False,
        'new_scene_generated': False, 'gate_or_model_selected_by_diagnosis': False,
        'scope': 'ALL saved train/development full bounded_shared calls for the unchanged fixed primary ADE-median seed. Descriptive true original-cost choice switches and same-library oracle gaps, not causal identification, model reinference, native original-engine replay, capture/safety/latency, holdout, or permission to deploy truth-conditioned gates.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.primary), indent=2, allow_nan=False))
