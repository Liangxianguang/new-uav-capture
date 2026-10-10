"""Streaming original-only fresh geometry/prefix data. No optimizer/activation."""
import argparse
import collections
import copy
import hashlib
import json
import pickle
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'cwm_v28'))
import collect_ranking_data as prior
from prefix_library import augment, check_native_prefix
sys.path.insert(0, str(ROOT/'scripts'))
from publish_exact_github_api import GitHubAPI, credential_token

sha, save_json = prior.sha, prior.save_json
REMOTE, BRANCH = prior.REMOTE, prior.BRANCH


def validate_protocol(protocol):
    if protocol != json.loads((HERE/'data_protocol.json').read_bytes()):
        raise ValueError('Exact published fresh protocol required')
    if ((protocol['groups'], protocol['train_groups'], protocol['development_groups']) != (64, 48, 16) or
        protocol['prefix_lengths'] != [2, 5] or protocol['horizon_steps'] != 8 or protocol['history_steps'] != 8 or
        protocol['maximum_command_speed_mps'] != 5 or protocol['target_branch_rule_overrides'] or
        any(protocol[k] is not False for k in ('enhanced_control_enabled', 'new_model_trained', 'private_labels_model_inputs', 'holdout_used'))):
        raise ValueError('Original/default-off/fixed-population contract differs')
    episodes = [v['seed_start']+i for v in protocol['variants'] for i in range(2*protocol['groups'])]
    if len(episodes) != 512 or len(set(episodes)) != 512:
        raise ValueError('All512 unique preassigned four-axis identities required')
    actual = set(episodes); layouts = set(range(protocol['layout_seed_start'], protocol['layout_seed_start']+protocol['groups']))
    def check(node, inherited=0):
        if isinstance(node, dict):
            count = node.get('groups', inherited)
            for key, population, multiplier in (('seed_start', actual, 2), ('episode_seed_start', actual, 2), ('layout_seed_start', layouts, 1)):
                if key in node and count and population & set(range(node[key], node[key]+multiplier*count)):
                    raise ValueError('Prior or reserved population identity overlap')
            for child in node.values(): check(child, count)
        elif isinstance(node, list):
            for child in node: check(child, inherited)
    for path in sorted((ROOT/'experiments').glob('cwm_v*/*protocol*.json')):
        if path.parent != HERE: check(json.loads(path.read_bytes()))
    check(protocol['reserved_holdout_not_collected'])


def published():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    remote = GitHubAPI(credential_token()).head()
    if remote != commit: raise ValueError('Exact source/protocol HEAD must be published before collection')
    files = [*HERE.glob('*.py'), HERE/'data_protocol.json', HERE/'training_protocol.json',
        HERE.parent/'cwm_v33/prefix_response_core.py', ROOT/'scripts/publish_exact_github_api.py']
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT) != path.read_bytes():
            raise ValueError('Actual collection/protocol source differs from published bytes: '+name)
    return {'commit': commit, 'remote': REMOTE, 'branch': BRANCH, 'exact_pushed_sources_verified': True,
        'published_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        'verification': 'authenticated_github_git_data_api_exact_branch'}


class PairedTap(prior.SequentialTap):
    def __init__(self, capsule, restored, steps):
        super().__init__(capsule, restored, steps)
        self.collection_parent = self.evaluator.DistributedMinimaxDNMPC.__bases__[0]

    def run(self, record, trajectory=None, observer=None):
        # Original evaluator unchanged; only preassigned RNG identity is paired.
        execution_record = copy.deepcopy(record)
        execution_record['episode_index'] = record['controller_sampling_identity']
        row, steps = super().run(execution_record, trajectory, observer)
        row['episode_index'] = record['episode_index']
        return row, steps

    def plain(self, record, trajectory, observer):
        current = self.evaluator.DistributedMinimaxDNMPC
        self.evaluator.DistributedMinimaxDNMPC = self.collection_parent
        try: return self.run(record, trajectory, observer)
        finally: self.evaluator.DistributedMinimaxDNMPC = current


def scenes(base, protocol):
    from encirclement3d.showcase import s4_adaptive_branching_scenario, scenario_metadata, s4_branch_route_metrics, validate_s4_branching_scenario
    conditions = [(s, c) for s in protocol['target_speed_scales'] for c in protocol['observation_conditions']]
    records = []
    for group in range(protocol['groups']):
        speed, condition = conditions[group % len(conditions)]
        for member, bias in enumerate(('upper', 'lower')):
            seed = protocol['seed_start']+2*group+member
            parent = {'episode_seed': seed, 'controller_sampling_identity': seed,
                'layout_seed': protocol['layout_seed_start']+group, 'mirror_group_id': f"cwm-prefix-{protocol['layout_seed_start']+group}",
                'mirror_pair_member': bias, 'target_motion_mode': 'adaptive_branching', 'target_speed_scale': speed,
                'obstacle_count': 1, 'level': None, 'model_split': 'train' if group < protocol['train_groups'] else 'development_validation',
                'observation_condition': condition['name'], 'pursuit_overrides': copy.deepcopy(condition['overrides']),
                'execution': {'enabled': False, 'action_delay_steps': 0, 'command_noise_std': 0.}}
            env = base.env_class(base.configuration(parent), obstacle_count=1, target_speed_scale=speed)
            try:
                original = s4_adaptive_branching_scenario(env, parent['layout_seed'], bias, protocol['scene_variation'])
                error = None
            except ValueError as failure: original, error = None, str(failure)
            for variant in protocol['variants']:
                row = {**copy.deepcopy(parent), 'episode_index': variant['seed_start']+2*group+member,
                    'variant': variant['name'], 'geometry_axis': copy.deepcopy(variant), 'route_valid': False}
                if original is None: row['route_error'] = error
                else:
                    wall = copy.deepcopy(original.obstacles[0]); extents = np.asarray(wall.half_extents_xy).copy()
                    extents[1] *= variant['wall_y_multiplier']; shift = variant['wall_center_x_m']
                    wall = replace(wall, center_xy=wall.center_xy+np.array([shift, 0.]), half_extents_xy=extents)
                    positions = original.defender_positions.copy(); positions[:, 1] *= variant['defender_y_multiplier']
                    scenario = replace(original, obstacles=(wall,), defender_positions=positions,
                        obstacle_zone_x=tuple(np.asarray(original.obstacle_zone_x)+shift), name=original.name+'_'+variant['name'])
                    row['scenario'] = scenario_metadata(scenario)
                    try:
                        validate_s4_branching_scenario(env, scenario)
                        row['route_validation'] = s4_branch_route_metrics(env, scenario); row['route_valid'] = True
                    except ValueError as failure: row['route_error'] = str(failure)
                records.append(row)
    return records


def inspect(base, env, captured, history, adapter, noise_key):
    from encirclement3d.minimax_mpc import aggregate_scenario_costs
    snapshot = prior.public_call_snapshot(captured)
    call = prior.restore_public_call(snapshot, base.planner_config, base.distributed_config, captured['scenarios'].trajectories[0])
    row = {'agent': call['agent'], 'ordinal': call['ordinal'], 'noise_key': noise_key, 'actual_candidate_count': len(call['local_candidates'])}
    result = prior.expand_candidates(call, history, base.evaluator.belief_reference, adapter, mode='shadow', source='shared')
    if result['inputs'] is None:
        return {**row, 'status': result['info']['status'], 'proposal_info': result['info']}, None, snapshot
    h, relative, joint, anchor, backbone, cv = result['inputs']
    augmented, pairs = augment(joint, anchor, call['agent'])
    # Entire public candidate/probe construction above precedes ANY private labels.
    actual = call['local_candidates']; chosen = next(i for i, p in enumerate(actual) if np.array_equal(p, call['selected']))
    costs = prior.local_cost(call, actual, np.repeat(backbone[None], len(actual), 0))
    selected_cost = aggregate_scenario_costs(call['selected_costs'], call['scenarios'].normalized_weights,
        call['planner'].config.risk_mode, call['planner'].config.cvar_alpha)
    if not np.isclose(costs[chosen], selected_cost, rtol=1e-10, atol=1e-8) or int(np.argmin(costs)) != chosen:
        raise ValueError('Original selected local cost differs')
    branches = [prior.s4_rollout(base, env, commands, noise_key) for commands in augmented]
    anchored = prior.s4_rollout(base, env, anchor, noise_key)
    if not prior.compare_arrays(anchored, prior.s4_rollout(base, env, anchor, noise_key)):
        raise ValueError('Actual repeated anchor differs')
    values = {k: np.stack([b[k] for b in branches]) for k in anchored}
    values.update({'anchor_'+k: v for k, v in anchored.items()})
    reference, velocity = base.evaluator.belief_reference(call['observation'], call['planner'].config)
    values.update(history=h, relative=relative, proposed=augmented, anchor=anchor, backbone=backbone,
        reference=reference, velocity=velocity, actual_candidates=actual.copy(), original_local_costs=costs,
        public_geometry_indices=np.arange(result['info']['seed_count'], dtype=np.int64),
        bounded_shared_indices=np.arange(len(joint), dtype=np.int64), base_candidate_count=np.asarray(len(joint), dtype=np.int64),
        prefix_pairs=pairs, proposal_prediction=result['forecast']['prediction'],
        proposal_motion_reference=result['forecast']['reference'], proposal_response=result['forecast']['response'])
    prefix_count = check_native_prefix(values)
    anchor_index = next(i for i, p in enumerate(result['actions']) if np.array_equal(p, call['selected']))
    return {**row, 'status': 'collected_fresh_geometry_prefix_pairs', 'actual_choice_pool_index': anchor_index,
        'base_candidate_count': len(joint), 'augmented_candidate_count': len(augmented), 'prefix_pair_count': prefix_count,
        'proposal_info': result['info'], 'original_local_costs_equal': True, 'anchor_repeat_equal': True,
        'native_prefix_bytes_equal': True}, values, snapshot


class Support:
    def __init__(self):
        self.rows = []  # Small scalar support records only; never retain arrays.
        self.invalid = self.pairs = self.calls = 0

    def add(self, row, values, protocol):
        self.calls += 1; self.pairs += check_native_prefix(values)
        self.invalid += int(((~values['valid']) & (values['termination'] != 'after_terminal')).any(-1).sum())
        actual, _ = prior.map_actual(values, row['agent'])
        for name, indices in (('actual_unique', actual), ('public_geometry', values['public_geometry_indices']),
            ('bounded_shared', values['bounded_shared_indices']), ('prefix_augmented', np.arange(len(values['proposed'])))):
            scalar = prior.support_record({'record': row, 'values': values}, indices, name, protocol)
            scalar.pop('indices'); scalar['variant'] = row['variant']; self.rows.append(scalar)

    def result(self, records, episodes, protocol):
        response = {name: prior.response_summary([r for r in self.rows if r['population'] == name]) for name in protocol['candidate_populations']}
        base = response['bounded_shared']; gate = protocol['data_gate']
        failure = sum(r['status'] not in ('collected_fresh_geometry_prefix_pairs', 'missing_received_peer_context') for r in records)
        checks = {'original_target_valid': sum(e['target_invalid_episode'] for e in episodes) <= gate['maximum_original_target_invalid_episodes'],
            'branch_target_valid': self.invalid <= gate['maximum_probe_target_invalid_branches'],
            'all_groups_nonanchor_support': base['groups_with_nonanchor_support'] == gate['minimum_groups_with_nonanchor_support'],
            'signal_groups': sum(g['over_threshold_points'] > 0 for g in base['by_group'].values()) >= gate['minimum_signal_groups'],
            'response_fraction': base['group_equal_over_threshold_fraction'] is not None and base['group_equal_over_threshold_fraction'] >= gate['minimum_group_equal_over_threshold_fraction'],
            'full_support': base['full_support_calls'] >= gate['minimum_full_support_calls'],
            'proposal_failures': failure <= gate['maximum_proposal_failure_calls'], 'native_prefix_support': self.pairs > 0,
            'complete_episodes': len(episodes) == 512}
        splits, axes = {}, {}
        for split, short in (('train', 'train'), ('development_validation', 'development')):
            rows = [r for r in self.rows if r['population'] == 'bounded_shared' and r['split'] == split]
            stats = prior.response_summary(rows); splits[split] = stats
            expected = protocol['train_groups'] if split == 'train' else protocol['development_groups']
            checks[short+'_all_groups'] = stats['groups_with_nonanchor_support'] == expected
            checks[short+'_full_support'] = stats['full_support_calls'] >= gate['minimum_full_support_'+short+'_calls']
            checks[short+'_points'] = stats['valid_nonanchor_points'] >= gate['minimum_common_valid_nonanchor_points_each_split']
            checks[short+'_signals'] = sum(g['over_threshold_points'] > 0 for g in stats['by_group'].values()) >= gate['minimum_signal_groups_each_split']
            checks[short+'_response_fraction'] = stats['group_equal_over_threshold_fraction'] is not None and stats['group_equal_over_threshold_fraction'] >= gate['minimum_group_equal_over_threshold_fraction']
        for variant in protocol['variants']:
            stats = prior.response_summary([r for r in self.rows if r['population'] == 'bounded_shared' and r['variant'] == variant['name']])
            axes[variant['name']] = stats
            checks[variant['name']+'_groups'] = stats['groups_with_nonanchor_support'] == protocol['groups']
        return {'response_support': response, 'split_support': splits, 'axis_support': axes,
            'target_invalid_branches': self.invalid, 'proposal_failure_calls': failure, 'native_prefix_pairs_checked': self.pairs,
            'calls': self.calls, 'data_checks': {k: bool(v) for k, v in checks.items()}, 'data_gate_passed': all(checks.values()),
            'training_authorized_by_this_report': False}


def source_hashes():
    paths = {Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).suffix == '.py' and (Path(m.__file__).resolve().is_relative_to(ROOT/'experiments') or
            Path(m.__file__).resolve() == ROOT/'scripts/publish_exact_github_api.py')}
    paths |= {HERE/'data_protocol.json', HERE/'training_protocol.json', HERE.parent/'cwm_v33/prefix_response_core.py'}
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def main(output):
    if output.exists(): raise ValueError('Exclusive NEW full collection output required')
    protocol = json.loads((HERE/'data_protocol.json').read_bytes()); validate_protocol(protocol)
    preregistration = published()
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'; proposer = HERE.parent/'cwm_v26/artifacts/real_sequential_shadow_20261010.zip'
    if sha(capsule) != protocol['baseline_capsule_sha256'] or sha(proposer) != protocol['proposal_archive_sha256']:
        raise ValueError('Pinned original/proposer archives differ')
    output.mkdir(parents=True, exist_ok=False); save_json(output/'preregistration.json', preregistration)
    base = PairedTap(capsule, output/'r', set(protocol['snapshot_steps']))
    with zipfile.ZipFile(proposer) as archive: predictor = prior.checkpoint_predictor(archive.read('model/primary.pt'), protocol['proposal_checkpoint_sha256'])
    adapter = prior.OptionalResponseAdapter(prior.QualificationGate(mode='shadow', qualification={'research_eligible': False}), lambda: predictor)
    population = scenes(base, protocol)
    (output/'scenes.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in population), encoding='utf8')
    preflight = {'status': 'complete_fixed_geometry_population_preflight', 'episodes': len(population),
        'route_errors': [s for s in population if not s['route_valid']], 'holdout_used': False, 'training_authorized': False}
    save_json(output/'geometry_preflight.json', preflight)
    if len(population) != 512 or preflight['route_errors']:
        raise ValueError('Fixed complete population has route errors; preserve all, never replace/filter')
    hashes = source_hashes()
    for name in hashes:
        destination = output/'source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    records, episodes, support = [], [], Support()
    for scene in population:
        history, commands, plans = [], [], []
        def observer(env, observation, actions, sequence):
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            commands.append(actions.copy()); plans.append(sequence.copy())
            if env.step_count not in protocol['snapshot_steps']: return
            padded = np.stack([history[0]]*max(0, 8-len(history))+history[-8:]); before = prior.fingerprint(env)
            for captured in base.calls:
                public_before = pickle.dumps(prior.public_call_snapshot(captured), protocol=5)
                identity = f"{protocol['noise_seed']}/{scene['episode_index']}/{env.step_count}/{captured['agent']}/{captured['ordinal']}"
                noise_key = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')
                row, values, snapshot = inspect(base, env, captured, padded, adapter, noise_key)
                row.update(episode_index=scene['episode_index'], group=scene['mirror_group_id'], split=scene['model_split'],
                    step=int(env.step_count), variant=scene['variant'])
                name = f"{scene['episode_index']}_{env.step_count}_{captured['ordinal']}_{captured['agent']}"
                path = output/'calls'/(name+'.npz'); path.parent.mkdir(exist_ok=True)
                context = path.with_suffix('.json'); save_json(context, snapshot)
                row.update(context_path=context.relative_to(output).as_posix(), context_sha256=sha(context))
                if values is not None:
                    np.savez_compressed(path, **values)
                    row.update(arrays_path=path.relative_to(output).as_posix(), arrays_sha256=sha(path))
                    support.add(row, values, protocol)
                if pickle.dumps(prior.public_call_snapshot(captured), protocol=5) != public_before:
                    raise ValueError('Prefix labeling mutated public original context')
                records.append(row)
            if prior.fingerprint(env) != before: raise ValueError('Prefix labeling mutated original environment')
            save_json(output/'records.json', records)
            print(json.dumps({'episode': scene['episode_index'], 'step': int(env.step_count), 'supported_calls': support.calls}), flush=True)
        path = output/'observed'/f"{scene['episode_index']}.npz"
        row, _ = base.run(scene, path, observer)
        trace = {'commanded': np.stack(commands), 'planned': np.stack(plans)}
        np.savez_compressed(path.with_suffix('.commands.npz'), **trace); commands.clear(); plans.clear()
        def plain_observer(env, observation, actions, sequence): commands.append(actions.copy()); plans.append(sequence.copy())
        plain = output/'plain'/path.name; original, _ = base.plain(scene, plain, plain_observer)
        plain_trace = {'commanded': np.stack(commands), 'planned': np.stack(plans)}
        np.savez_compressed(plain.with_suffix('.commands.npz'), **plain_trace)
        if not prior.compare_arrays(prior.arrays(path.read_bytes()), prior.arrays(plain.read_bytes())) or not prior.compare_arrays(trace, plain_trace):
            raise ValueError('Collection changed original unhooked trajectory/plan/CBF')
        keys = ('safe_capture_success', 'collision', 'boundary_violation', 'timeout', 'target_invalid_episode', 'termination_reason')
        if any(row[k] != original[k] for k in keys): raise ValueError('Original unhooked outcomes differ')
        if bool(prior.target_bad(prior.arrays(path.read_bytes())['target_positions'], scene['scenario']).any()) != row['target_invalid_episode']:
            raise ValueError('Original target geometry validity differs')
        episodes.append({'episode_index': scene['episode_index'], 'split': scene['model_split'], 'variant': scene['variant'],
            **{k: row[k] for k in keys}, 'unhooked_original_trace_equal': True,
            'observed_sha256': sha(path), 'plain_sha256': sha(plain)})
        save_json(output/'episodes.json', episodes)
        print(json.dumps({'completed_episodes': len(episodes), 'total_episodes': 512, 'supported_calls': support.calls}), flush=True)
    prior.verify(base.root, base.capsule_manifest)
    if any(sha(ROOT/name) != value for name, value in hashes.items()): raise ValueError('Collection sources changed')
    summary = {'status': 'complete_fresh_geometry_prefix_collection_pending_independent_audit', 'protocol': protocol,
        'preregistration': preregistration, 'source_hashes': hashes, 'episodes': episodes, 'records': len(records),
        'scene_sha256': sha(output/'scenes.jsonl'), 'records_sha256': sha(output/'records.json'),
        **support.result(records, episodes, protocol), 'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False}
    save_json(output/'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'data_gate_passed': summary['data_gate_passed']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    main(args.output)
