"""Fixed BASE-only original realized cost, family/axis-block development gates."""
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cwm_v28'))
from ranking_qualification import validated_choice
from support_audit import map_actual
from task_score import original_scores
from local_shadow import rank_metrics
from audited_stream import sha
from contract import validate_contract

IDENTITY = ('episode_index', 'step', 'ordinal', 'agent', 'group', 'variant')


def build_decisions(call, output, context):
    row, v = call['record'], call['values']; n = int(v['base_candidate_count'])
    if not (v['valid'][:n].all() and v['anchor_valid'].all()):
        raise ValueError('Decisions require the COMPLETE unchanged BASE library')
    actions = v['proposed'][:n, :, row['agent']]
    cv = v['reference'][None]+.1*np.arange(1, 9)[:, None]*v['velocity'][None]
    paths = {'model': output['prediction'][:n], 'own_motion_component': output['reference'][:n],
        'original_gru': np.repeat(v['backbone'][None], n, 0), 'constant_velocity': np.repeat(cv[None], n, 0),
        'action_specific_truth': v['target'][:n]}
    costs = {name: original_scores(context, actions, path) for name, path in paths.items()}
    if not all(value.shape == (n,) and np.isfinite(value).all() for value in costs.values()):
        raise ValueError('Complete finite original full-cost vectors required')
    base = {**v, 'proposed': v['proposed'][:n]}
    actual, _ = map_actual(base, row['agent'])
    raw = [int(np.flatnonzero((actions == a[None]).all((1, 2)))[0]) for a in v['actual_candidates']]
    if not np.allclose(costs['original_gru'][raw], v['original_local_costs'], rtol=1e-10, atol=1e-8):
        raise ValueError('Actual original local full costs changed')
    populations = {}
    for population, indices in (('actual_unique', actual), ('public_geometry', v['public_geometry_indices']),
                                ('bounded_shared', np.arange(n, dtype=np.int64))):
        subset = {name: value[indices].tolist() for name, value in costs.items()}
        measures = {name: rank_metrics(value, subset['action_specific_truth'], 1e-9) for name, value in subset.items()}
        anchor = np.flatnonzero(indices == row['actual_choice_pool_index'])
        if len(anchor) != 1 or (population == 'actual_unique' and measures['original_gru']['choice'] != int(anchor[0])):
            raise ValueError('Original actual anchor/solver selection changed')
        populations[population] = {**{key: row[key] for key in IDENTITY}, 'population': population,
            'indices': indices.tolist(), 'costs': subset, 'metrics': measures,
            'original_anchor_realized_cost': float(costs['action_specific_truth'][row['actual_choice_pool_index']])}
    return populations


class FamilyAxisAccumulator:
    def __init__(self, axes):
        if len(set(axes)) != 4: raise ValueError('ALL four declared axes required')
        self.axes = axes; self.calls = 0
        self.grouped = collections.defaultdict(lambda: collections.defaultdict(lambda: [0., 0]))

    def add(self, value, row):
        if row['variant'] not in self.axes or not np.isfinite(value):
            raise ValueError('Finite declared geometry-axis gain required')
        entry = self.grouped[row['group']][row['variant']]
        entry[0] += float(value); entry[1] += 1; self.calls += 1

    def result(self, expected_groups, draws=2000, seed=1034701):
        if set(self.grouped) != set(expected_groups) or not self.calls or any(set(v) != set(self.axes) for v in self.grouped.values()):
            raise ValueError('ALL four axes in EVERY declared family required')
        names = sorted(self.grouped)
        per_family = {g: {a: self.grouped[g][a][0]/self.grouped[g][a][1] for a in self.axes} for g in names}
        block = np.array([np.mean(list(per_family[g].values())) for g in names], dtype=np.float64)
        if draws != 2000 or seed != 1034701: raise ValueError('Fixed family bootstrap budget/seed required')
        sampler = np.random.default_rng(seed)
        measured = block[sampler.integers(0, len(block), size=(draws, len(block)))].mean(1)
        return {'groups': len(block), 'calls': self.calls, 'group_equal_mean': float(block.mean()),
            'percentile_95_interval': np.percentile(measured, [2.5, 97.5]).tolist(),
            'by_axis_group_equal_mean': {a: float(np.mean([per_family[g][a] for g in names])) for a in self.axes},
            'by_family_axis_mean': per_family,
            'interpretation': 'Fresh V34 development family/axis-equal descriptive bootstrap; all mirrors/axes clustered. NOT untouched holdout, causal identification or capture confirmation.'}


def family_axis_stats(values, records, axes, expected_groups, draws=2000, seed=1034701):
    result = FamilyAxisAccumulator(axes)
    for value, row in itertools.zip_longest(values, records):
        if value is None or row is None: raise ValueError('Identical complete gain/record support required')
        result.add(value, row)
    return result.result(expected_groups, draws, seed)


class DecisionRows:
    """Repeatable JSONL iterable; no complete nine-model cost population in RAM."""
    def __init__(self, path, digest, count): self.path, self.digest, self.count = path, digest, count

    def __iter__(self):
        if sha(self.path) != self.digest: raise ValueError('Saved native original-cost decisions changed')
        count = 0
        with self.path.open(encoding='utf8') as file:
            for line in file:
                count += 1; yield json.loads(line)
        if count != self.count: raise ValueError('Complete saved original-cost decision population required')


def compare_decisions(changed, baseline, contract, expected_groups, method='model'):
    gains = FamilyAxisAccumulator(contract['geometry_axes'])
    for a, b in itertools.zip_longest(changed, baseline):
        if a is None or b is None or tuple(a[k] for k in IDENTITY) != tuple(b[k] for k in IDENTITY) or a['population'] != 'bounded_shared' or b['population'] != 'bounded_shared':
            raise ValueError('Identical complete BASE native decision support required')
        if a['indices'] != b['indices'] or a['costs']['action_specific_truth'] != b['costs']['action_specific_truth']:
            raise ValueError('Identical true costs/base candidates required')
        gains.add(validated_choice(b, method)-validated_choice(a, 'model'), a)
    return gains.result(expected_groups)


def qualify(models, controls, decisions, contract, plan, expected_groups):
    validate_contract(contract, plan)
    expected = {(name, seed) for name in plan['models'] for seed in plan['training']['seeds']}
    if len(models) != 9 or {(r['configuration'], r['seed']) for r in models} != expected or len(set(expected_groups)) != 16:
        raise ValueError('Complete nine final80epoch models and16 development families required')
    keys = ('group_equal_ade_m', 'group_equal_paired_response_error_m', 'group_equal_ranking_kl')
    if any(not np.isfinite(r['development'][key]) for r in models for key in keys):
        raise ValueError('Finite ALL-seed BASE development metrics required')
    selected = {name: sorted([r for r in models if r['configuration'] == name],
        key=lambda r: (r['development']['group_equal_ade_m'], r['seed']))[1]['seed'] for name in plan['models']}
    middle = lambda name, key: float(np.median([r['development'][key] for r in models if r['configuration'] == name]))
    threshold = plan['development_gate']; gates = {}
    for name in ('cv_full_query_rank_l2', 'cv_prefix_rank_l2'):
        rows = [r for r in models if r['configuration'] == name]
        checks = {'each_seed_ade_vs_gru': all(r['development']['group_equal_ade_m'] <= controls['frozen_gru']['group_equal_ade_m']*threshold['maximum_each_seed_ade_ratio_vs_gru'] for r in rows),
            'median_ade_vs_cv': middle(name, 'group_equal_ade_m') <= controls['constant_velocity']['group_equal_ade_m']*threshold['maximum_median_ade_ratio_vs_cv'],
            'median_response_vs_zero': middle(name, 'group_equal_paired_response_error_m') <= controls['frozen_gru']['group_equal_paired_response_error_m']*threshold['maximum_median_response_error_ratio_vs_zero'],
            'anchor_response_exact_zero': all(r['development']['anchor_response_exact_zero'] is True for r in rows)}
        chosen = decisions[f'{name}_seed{selected[name]}']; gains = {}
        references = ['own_motion', 'cv', 'motion_only', 'original_gru']
        if name == contract['primary_configuration']: references.append('full_query')
        for reference in references:
            if reference in ('motion_only', 'full_query'):
                reference_name = 'cv_motion_only' if reference == 'motion_only' else 'cv_full_query_rank_l2'
                base = decisions[f'{reference_name}_seed{selected[reference_name]}']; method = 'model'
            else:
                base = chosen; method = {'own_motion': 'own_motion_component', 'cv': 'constant_velocity', 'original_gru': 'original_gru'}[reference]
            gain = compare_decisions(chosen, base, contract, expected_groups, method)
            gains[reference] = gain; checks['gain_vs_'+reference] = gain['percentile_95_interval'][0] > threshold['minimum_selected_gain_lower_bound_vs_'+reference]
        gates[name] = {'research_eligible': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()},
            'selected_seed': selected[name], 'selected_gains': gains, 'online_promoted': False}
    return selected, gates


def library_contributions(populations, contract, expected_groups):
    result = {}; shared = populations['bounded_shared']
    for smaller in ('actual_unique', 'public_geometry'):
        base = populations[smaller]
        for method in ('model', 'own_motion_component', 'original_gru', 'constant_velocity'):
            gain = FamilyAxisAccumulator(contract['geometry_axes'])
            for a, b in itertools.zip_longest(shared, base):
                if a is None or b is None or tuple(a[k] for k in IDENTITY) != tuple(b[k] for k in IDENTITY):
                    raise ValueError('Identical native support required for library diagnostics')
                mapped = dict(zip(a['indices'], a['costs']['action_specific_truth']))
                if any(mapped.get(i) != value for i, value in zip(b['indices'], b['costs']['action_specific_truth'])):
                    raise ValueError('Smaller-library truth must be unchanged BASE subset')
                gain.add(validated_choice(b, method)-validated_choice(a, method), a)
            result[smaller+'__to__bounded_shared__'+method] = gain.result(expected_groups)
    return result
