"""Describe BOTH completed V23 runs; saved-cost checks are NOT engine reinference."""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ('model', 'own_motion_component', 'original_gru', 'constant_velocity', 'action_specific_truth')
SPLITS = ('train', 'development_validation')
POPULATIONS = ('actual_unique', 'geometry_union')
IDENTITY = ('episode_index', 'step', 'agent', 'group')
METRICS = ('group_equal_ade_m', 'group_equal_paired_response_error_m',
           'group_equal_normalized_cost_contrast_l1')


def saved_cost_metrics(costs, truth):
    """Independent stdlib rederivation of the saved argmin/regret, not of costs."""
    if not costs or len(costs) != len(truth) or not all(math.isfinite(x) for x in (*costs, *truth)):
        raise ValueError('Finite nonempty equal-length saved costs required')
    choice = min(range(len(costs)), key=costs.__getitem__)  # first tie, same as original argmin
    regret = truth[choice] - min(truth)
    return {'choice': choice, 'regret_in_diagnostic_cost': regret,
            'realized_cost_at_choice': truth[choice], 'realized_tie_optimal': regret <= 1e-9}


def describe_decisions(rows, population):
    if not rows:
        raise ValueError('Complete nonempty saved decision population required')
    identities = [tuple(r[k] for k in IDENTITY) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('Duplicate saved decision identity')
    by_group = collections.defaultdict(list)
    counts = collections.Counter()
    for row in rows:
        indices = row['indices']
        if row['population'] != population or not indices or len(set(indices)) != len(indices):
            raise ValueError('Saved candidate indices/population differ')
        if set(row['costs']) != set(METHODS) or set(row['metrics']) != set(METHODS):
            raise ValueError('Incomplete saved decision methods')
        truth = row['costs']['action_specific_truth']
        if len(truth) != len(indices):
            raise ValueError('Saved candidate cost dimensions differ')
        for method in METHODS:
            measured = saved_cost_metrics(row['costs'][method], truth)
            if measured != row['metrics'][method]:
                raise ValueError('Saved choice/regret inconsistent with saved costs')
        model, motion = (row['metrics'][k] for k in ('model', 'own_motion_component'))
        gain = motion['regret_in_diagnostic_cost'] - model['regret_in_diagnostic_cost']
        by_group[row['group']].append({
            'choice_change': int(model['choice'] != motion['choice']),
            'help': int(gain > 0), 'harm': int(gain < 0), 'gain': gain})
        counts[len(indices)] += 1
    equal = {key: statistics.mean(statistics.mean(r[key] for r in group)
                                 for group in by_group.values())
             for key in ('choice_change', 'help', 'harm', 'gain')}
    return {'calls': len(rows), 'groups': len(by_group),
            'candidate_count_histogram': {str(k): v for k, v in sorted(counts.items())},
            'choice_change_count': sum(r['choice_change'] for g in by_group.values() for r in g),
            'help_count': sum(r['help'] for g in by_group.values() for r in g),
            'harm_count': sum(r['harm'] for g in by_group.values() for r in g),
            'group_equal_choice_change_rate': equal['choice_change'],
            'group_equal_help_rate': equal['help'], 'group_equal_harm_rate': equal['harm'],
            'group_equal_gain_vs_own_motion': equal['gain']}


def summarize(primary, retrained):
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    raw = (primary / 'summary.json').read_bytes()
    if raw != (retrained / 'summary.json').read_bytes():
        raise ValueError('Two complete summaries differ')
    summary = json.loads(raw)
    if summary['protocol'] != protocol or summary['status'] != 'offline_fresh_deployed_cost_contrast_finished_not_promoted':
        raise ValueError('Completed fixed V23 protocol/status required')
    if any(summary[k] for k in ('enhanced_control_enabled', 'holdout_used', 'prior_gate_overridden',
                                'common_motion_in_response_optimizer', 'actual_unique_is_promotion_evidence')):
        raise ValueError('Offline optional-model safeguards differ')
    expected = {(name, seed) for name in protocol['models'] for seed in protocol['training']['seeds']}
    models = summary['models']
    if len(models) != len(expected) or {(r['configuration'], r['seed']) for r in models} != expected:
        raise ValueError('Both complete fixed18-model populations required')
    medians, decisions, hashes, support = {}, {}, {}, {}
    for name in protocol['models']:
        selected = [r for r in models if r['configuration'] == name]
        if any(not math.isfinite(r['development'][k]) for r in selected for k in METRICS):
            raise ValueError('Nonfinite development metrics')
        medians[name] = {k: statistics.median(r['development'][k] for r in selected) for k in METRICS}
        for row in sorted(selected, key=lambda r: r['seed']):
            identifier = f"{name}_seed{row['seed']}"
            path = identifier + '_decisions.json'
            decision_raw = (primary / path).read_bytes()
            if decision_raw != (retrained / path).read_bytes():
                raise ValueError('Independent saved decision evidence differs')
            hashes[path] = hashlib.sha256(decision_raw).hexdigest()
            data, described = json.loads(decision_raw), {}
            for split in SPLITS:
                described[split] = {}
                for population in POPULATIONS:
                    rows = data[split][population]
                    identities = [tuple(r[k] for k in IDENTITY) for r in rows]
                    if (len(rows) != summary['full_cost_calls'][split] or
                        {r['group'] for r in rows} != set(summary['split_groups'][split])):
                        raise ValueError('Complete saved train/dev support differs')
                    if (split, population) in support and support[(split, population)] != identities:
                        raise ValueError('Saved decisions do not use common support')
                    support[(split, population)] = identities
                    described[split][population] = describe_decisions(rows, population)
                actual, union = data[split]['actual_unique'], data[split]['geometry_union']
                if support[(split, 'actual_unique')] != support[(split, 'geometry_union')]:
                    raise ValueError('Actual/union saved decision supports differ')
                if any(not set(a['indices']).issubset(u['indices']) for a, u in zip(actual, union)):
                    raise ValueError('Actual candidates are not a subset of union')
                for a, u in zip(actual, union):
                    positions = [u['indices'].index(i) for i in a['indices']]
                    if any(a['costs'][method] != [u['costs'][method][i] for i in positions]
                           for method in METHODS):
                        raise ValueError('Actual/union saved costs do not match on shared candidates')
            decisions[identifier] = described
    return {'status': 'completed_training_saved_cost_description_not_full_reinference',
            'summary_sha256': hashlib.sha256(raw).hexdigest(),
            'protocol_version': protocol['version'], 'models_per_run': len(models), 'independent_runs': 2,
            'development_across_seed_medians': medians,
            'trainer_declared_qualification': summary['qualification'],
            'trainer_declared_primary_research_eligible': summary['primary_research_eligible'],
            'saved_decisions': decisions, 'saved_decision_sha256': hashes,
            'saved_argmin_regret_metrics_recomputed': True,
            'actual_union_shared_candidate_saved_costs_equal': True,
            'independent_saved_summaries_and_decisions_byte_equal': True,
            'original_engine_costs_reexecuted_by_this_report': False,
            'checkpoints_reinferred_by_this_report': False,
            'enhanced_control_enabled': False, 'holdout_used': False,
            'interpretation': 'Posthoc description of inspected train/dev evidence, not causal identification, '
                              'closed-loop capture gain, completed scientific release audit or promotion. '
                              'Both original-engine reinference audits continue separately. No new fitting, '
                              'seed selection, gate relaxation or policy enabled by this report.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'retrained', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.primary, args.retrained)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': report['status'], 'summary_sha256': report['summary_sha256']}), flush=True)


if __name__ == '__main__':
    main()
