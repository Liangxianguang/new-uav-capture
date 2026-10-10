"""ALL original calls/physical branches replayed independently before training."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import collect_prefix_data as collection
from prefix_library import check_native_prefix
prior = collection.prior
from task_score import original_scores, FrozenLocalScore
sha, save_json = prior.sha, prior.save_json


def equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def independent_library(joint, anchor, agent):
    """Independent suffix reconstruction; does not call collector's augment()."""
    library = list(joint.copy()); mapping = []
    for length in (2, 5):
        for original in range(len(joint)):
            if (joint[original] == anchor).all(): continue
            probe = joint[original].copy()
            own = np.concatenate((probe[:length, agent], anchor[length:, agent]), axis=0)
            probe[:, agent] = own
            found = [i for i, value in enumerate(library) if np.array_equal(value, probe)]
            if not found: found = [len(library)]; library.append(probe)
            mapping.append([original, length, found[0]])
    return np.stack(library), np.asarray(mapping, dtype=np.int64).reshape(-1, 3)


class IndependentTap(prior.Baseline):
    def __init__(self, capsule, restored, steps):
        super().__init__(capsule, restored)
        self.invocations, self.current_step = [], None
        parent = self.evaluator.DistributedMinimaxDNMPC; owner = self
        class IndependentPlanner(parent):
            def plan(self, observation, scenarios, **kwargs):
                owner.invocations.clear(); owner.current_step = int(kwargs.get('step_index', 0))
                return super().plan(observation, scenarios, **kwargs)
            def _select_local_sequence(self, observation, scenarios, agent_id, local_candidates, known, peer_sequences):
                chosen, costs = super()._select_local_sequence(observation, scenarios, agent_id, local_candidates, known, peer_sequences)
                if owner.current_step in steps:
                    owner.invocations.append({'agent': int(agent_id), 'ordinal': len(owner.invocations),
                        'observation': copy.deepcopy(observation), 'scenarios': copy.deepcopy(scenarios),
                        'planner': SimpleNamespace(config=self.config), 'local_candidates': np.stack(local_candidates).copy(),
                        'selected': chosen.copy(), 'selected_costs': costs.copy(),
                        'known': copy.deepcopy(known), 'peer_sequences': copy.deepcopy(peer_sequences)})
                return chosen, costs
        self.evaluator.DistributedMinimaxDNMPC = IndependentPlanner

    def run(self, record, trajectory=None, observer=None):
        copied = copy.deepcopy(record); copied['episode_index'] = record['controller_sampling_identity']
        row, steps = super().run(copied, trajectory, observer)
        row['episode_index'] = record['episode_index']
        return row, steps


def preflight(data):
    protocol = json.loads((HERE/'data_protocol.json').read_bytes()); collection.validate_protocol(protocol)
    report = json.loads((data/'summary.json').read_bytes())
    if (report['status'] != 'complete_fresh_geometry_prefix_collection_pending_independent_audit' or
        report['protocol'] != protocol or report['data_gate_passed'] is not True or
        report['training_authorized_by_this_report'] is not False or
        any(report[k] is not False for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used'))):
        raise ValueError('Complete original-only collection with unchanged successful data gates required')
    prereg = json.loads((data/'preregistration.json').read_bytes())
    if prereg != report['preregistration'] or prereg['remote'] != collection.REMOTE or prereg['branch'] != collection.BRANCH or prereg['exact_pushed_sources_verified'] is not True:
        raise ValueError('Exact source prepublication evidence differs')
    for name, digest in prereg['published_files'].items():
        raw = subprocess.check_output(['git', 'show', prereg['commit']+':'+name], cwd=ROOT)
        if raw != (ROOT/name).read_bytes() or sha(ROOT/name) != digest:
            raise ValueError('Preregistered source/protocol bytes differ')
    for name, digest in report['source_hashes'].items():
        if sha(ROOT/name) != digest or sha(data/'source'/name) != digest:
            raise ValueError('Current/stored collection scientific sources differ')
    if sha(data/'scenes.jsonl') != report['scene_sha256'] or sha(data/'records.json') != report['records_sha256']:
        raise ValueError('Full scene/call population identity differs')
    population = [json.loads(line) for line in (data/'scenes.jsonl').read_text(encoding='utf8').splitlines()]
    records = json.loads((data/'records.json').read_bytes()); episodes = json.loads((data/'episodes.json').read_bytes())
    if len(population) != 512 or len(episodes) != 512 or episodes != report['episodes'] or len(records) != report['records']:
        raise ValueError('Complete512-episode/record population required')
    return protocol, report, population, records, episodes


def evidence_manifest(data, report, population, records):
    used = {'summary.json', 'preregistration.json', 'scenes.jsonl', 'geometry_preflight.json', 'records.json', 'episodes.json'}
    used.update('source/'+name for name in report['source_hashes'])
    used.update(name for r in records for name in (r['context_path'], r.get('arrays_path')) if name)
    for scene in population:
        for kind in ('observed', 'plain'):
            used.update((f"{kind}/{scene['episode_index']}.npz", f"{kind}/{scene['episode_index']}.commands.npz"))
    for name in used:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('Only safe relative evidence paths are allowed')
    return {name: sha(data/name) for name in sorted(used)}


def audit(data, output):
    if output.exists(): raise ValueError('Exclusive NEW independent audit output required')
    summary_identity = sha(data/'summary.json')
    protocol, report, population, records, episodes = preflight(data)
    initial_manifest = evidence_manifest(data, report, population, records)
    if initial_manifest['summary.json'] != summary_identity:
        raise ValueError('Collection summary changed during audit preflight')
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'; proposer = HERE.parent/'cwm_v26/artifacts/real_sequential_shadow_20261010.zip'
    if sha(capsule) != protocol['baseline_capsule_sha256'] or sha(proposer) != protocol['proposal_archive_sha256']:
        raise ValueError('Original/proposer archive identity differs')
    output.mkdir(parents=True, exist_ok=False)
    base = IndependentTap(capsule, output/'r', set(protocol['snapshot_steps']))
    expected_scenes = collection.scenes(base, protocol)
    if population != expected_scenes or any(not s['route_valid'] for s in population):
        raise ValueError('Regenerated full fixed geometry population differs')
    with zipfile.ZipFile(proposer) as archive: predictor = prior.checkpoint_predictor(archive.read('model/primary.pt'), protocol['proposal_checkpoint_sha256'])
    adapter = prior.OptionalResponseAdapter(prior.QualificationGate(mode='shadow', qualification={'research_eligible': False}), lambda: predictor)
    sources = collection.source_hashes()
    for name in sources:
        destination = output/'source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    checked, cost_checks, support = [], [], collection.Support()
    by_episode = {s['episode_index']: [] for s in population}
    if len(by_episode) != 512: raise ValueError('Duplicate scene identity')
    for row in records:
        for key in ('episode_index', 'step', 'ordinal', 'agent', 'noise_key', 'actual_candidate_count'):
            if type(row[key]) is not int: raise ValueError('Native identity/count types differ')
        if row['episode_index'] not in by_episode: raise ValueError('Undeclared saved call')
        by_episode[row['episode_index']].append(row)
    for episode_position, (scene, episode) in enumerate(zip(population, episodes), 1):
        if (episode['episode_index'], episode['split'], episode['variant']) != (scene['episode_index'], scene['model_split'], scene['variant']):
            raise ValueError('Original episode identity/order differs')
        selected = by_episode[scene['episode_index']]; cursor = [0]; history, commands, plans = [], [], []
        def observer(env, observation, actions, sequence):
            feature = base.evaluator.policy_observations(env, observation).reshape(-1).copy()
            if feature.shape != (252,): raise ValueError('Original252 encoder differs')
            history.append(feature); commands.append(actions.copy()); plans.append(sequence.copy())
            if env.step_count not in protocol['snapshot_steps']: return
            padded = np.stack([history[0]]*max(0, 8-len(history))+history[-8:]); parent_before = prior.fingerprint(env)
            for captured in base.invocations:
                if cursor[0] >= len(selected): raise ValueError('Unrecorded native sequential call')
                row = selected[cursor[0]]; cursor[0] += 1
                identity = (scene['episode_index'], int(env.step_count), captured['ordinal'], captured['agent'])
                if tuple(row[k] for k in ('episode_index', 'step', 'ordinal', 'agent')) != identity or row['group'] != scene['mirror_group_id'] or row['split'] != scene['model_split'] or row['variant'] != scene['variant']:
                    raise ValueError('Call/split/geometry identity differs')
                stem = f"calls/{identity[0]}_{identity[1]}_{identity[2]}_{identity[3]}"
                if row['context_path'] != stem+'.json' or ('arrays_path' in row and row['arrays_path'] != stem+'.npz'):
                    raise ValueError('Saved native call paths differ')
                expected_snapshot = prior.public_call_snapshot(captured)
                if sha(data/row['context_path']) != row['context_sha256'] or json.loads((data/row['context_path']).read_bytes()) != expected_snapshot:
                    raise ValueError('Actual public received context differs')
                call = prior.restore_public_call(expected_snapshot, base.planner_config, base.distributed_config, captured['scenarios'].trajectories[0])
                for peer, message in call['known'].items():
                    if message.sender != peer or message.receiver != row['agent'] or message.delivery_step > env.step_count or env.step_count-message.sent_step > base.distributed_config.max_message_age_steps:
                        raise ValueError('Delayed message contract differs')
                generated = np.stack(call['planner']._local_candidate_sequences(call['observation'], call['scenarios'], call['agent'], call['observation']['defender_positions'][call['agent']], call['known']))
                if not equal(generated, call['local_candidates']): raise ValueError('Original public local generator differs')
                proposed = prior.expand_candidates(call, padded, base.evaluator.belief_reference, adapter, mode='shadow', source='shared')
                if proposed['inputs'] is None:
                    if 'arrays_path' in row or row['status'] != proposed['info']['status'] or row['proposal_info'] != proposed['info']:
                        raise ValueError('Missing/failure call was imputed')
                    checked.append({'identity': list(identity), 'status': row['status']}); continue
                if row['status'] != 'collected_fresh_geometry_prefix_pairs' or row['proposal_info'] != proposed['info']:
                    raise ValueError('Fixed proposer provenance differs')
                h, relative, joint, anchor, backbone, cv = proposed['inputs']
                augmented, pairs = independent_library(joint, anchor, row['agent'])
                path = data/row['arrays_path']
                if sha(path) != row['arrays_sha256']: raise ValueError('Saved physical arrays changed')
                values = prior.arrays(path.read_bytes())
                reference, velocity = base.evaluator.belief_reference(call['observation'], call['planner'].config)
                expected = {'history': h, 'relative': relative, 'proposed': augmented, 'anchor': anchor, 'backbone': backbone,
                    'reference': reference, 'velocity': velocity, 'actual_candidates': generated,
                    'public_geometry_indices': np.arange(proposed['info']['seed_count'], dtype=np.int64),
                    'bounded_shared_indices': np.arange(len(joint), dtype=np.int64),
                    'base_candidate_count': np.asarray(len(joint), dtype=np.int64), 'prefix_pairs': pairs}
                for key, value in expected.items():
                    if not equal(values[key], value): raise ValueError('Independent public prefix/input reconstruction differs: '+key)
                for key, field in (('prediction', 'proposal_prediction'), ('reference', 'proposal_motion_reference'), ('response', 'proposal_response')):
                    if not equal(values[field], proposed['forecast'][key]): raise ValueError('Fixed public proposer reinference differs')
                anchor_index = next(i for i, p in enumerate(proposed['actions']) if np.array_equal(p, call['selected']))
                if row['actual_choice_pool_index'] != anchor_index or row['base_candidate_count'] != len(joint) or row['augmented_candidate_count'] != len(augmented):
                    raise ValueError('Original/base/augmented candidate identity differs')
                noise_key = int.from_bytes(hashlib.sha256(f"{protocol['noise_seed']}/{identity[0]}/{identity[1]}/{identity[3]}/{identity[2]}".encode()).digest()[:4], 'little')
                if row['noise_key'] != noise_key: raise ValueError('Native paired noise key differs')
                branches = [prior.s4_rollout(base, env, commands, noise_key) for commands in augmented]
                anchored = prior.s4_rollout(base, env, anchor, noise_key)
                if not prior.compare_arrays(anchored, prior.s4_rollout(base, env, anchor, noise_key)): raise ValueError('Actual independent anchor repeat differs')
                for field in anchored:
                    if not equal(values[field], np.stack([b[field] for b in branches])) or not equal(values['anchor_'+field], anchored[field]):
                        raise ValueError('ALL independent native branches differ: '+field)
                if check_native_prefix(values) != row['prefix_pair_count']: raise ValueError('ALL native prefix counts differ')
                raw_costs = original_scores(call, generated, np.repeat(backbone[None], len(generated), 0))
                chosen = next(i for i, p in enumerate(generated) if np.array_equal(p, call['selected']))
                if not np.allclose(raw_costs, values['original_local_costs'], rtol=1e-10, atol=1e-8) or int(np.argmin(raw_costs)) != chosen:
                    raise ValueError('Original local costs/selected command differ')
                complete = np.flatnonzero(values['valid'].all(1) & values['anchor_valid'].all())
                if len(complete):
                    own = augmented[complete, :, row['agent']]; target = values['target'][complete]
                    native_costs = original_scores(call, own, target)
                    surrogate = FrozenLocalScore(call, own)(torch.from_numpy(target)).detach().numpy()
                    if not np.allclose(surrogate, native_costs, rtol=1e-10, atol=1e-8):
                        raise ValueError('Complete actual original label-cost surrogate differs')
                    cost_checks.append({'identity': list(identity), 'complete_indices': complete.tolist(), 'true_costs': native_costs.tolist(),
                        'base_library_complete': bool(values['valid'][:len(joint)].all() and values['anchor_valid'].all())})
                support.add(row, values, protocol)
                checked.append({'identity': list(identity), 'status': 'all_public_prefix_native_branches_original_costs_checked'})
            if prior.fingerprint(env) != parent_before: raise ValueError('Independent audit mutated native parent')
        path = output/'trajectories'/f"{scene['episode_index']}.npz"
        row, _ = base.run(scene, path, observer)
        if cursor[0] != len(selected): raise ValueError('Saved sequential calls omitted from original replay')
        trace = {'commanded': np.stack(commands), 'planned': np.stack(plans)}
        np.savez_compressed(path.with_suffix('.commands.npz'), **trace)
        for kind in ('observed', 'plain'):
            original = data/kind/path.name
            if sha(original) != episode[kind+'_sha256'] or not prior.compare_arrays(prior.arrays(path.read_bytes()), prior.arrays(original.read_bytes())) or not prior.compare_arrays(trace, prior.arrays(original.with_suffix('.commands.npz').read_bytes())):
                raise ValueError('Independent original trajectory/plan/CBF differs')
        for key in ('safe_capture_success', 'collision', 'boundary_violation', 'timeout', 'target_invalid_episode', 'termination_reason'):
            if row[key] != episode[key]: raise ValueError('Independent original episode outcome differs')
        if bool(prior.target_bad(prior.arrays(path.read_bytes())['target_positions'], scene['scenario']).any()) != row['target_invalid_episode']:
            raise ValueError('Independent original target geometry validity differs')
        save_json(output/'checked_calls.json', checked)
        print(json.dumps({'independently_audited_episodes': episode_position,
            'checked_calls': len(checked)}), flush=True)
    if len(checked) != len(records) or support.calls != report['calls']: raise ValueError('Incomplete independent ALL-call support')
    stats = support.result(records, episodes, protocol)
    if not stats['data_gate_passed'] or any(report[key] != value for key, value in stats.items()):
        raise ValueError('Independently recomputed whole-population data gate differs or failed')
    prior.verify(base.root, base.capsule_manifest)
    for folder, hashes in ((output, sources), (data, report['source_hashes'])):
        for name, digest in hashes.items():
            if sha(ROOT/name) != digest or sha(folder/'source'/name) != digest: raise ValueError('Actual used scientific sources changed')
    save_json(output/'full_label_cost_checks.json', cost_checks)
    if evidence_manifest(data, report, population, records) != initial_manifest:
        raise ValueError('Complete collection evidence changed during native audit')
    if sha(capsule) != protocol['baseline_capsule_sha256'] or sha(proposer) != protocol['proposal_archive_sha256']:
        raise ValueError('Pinned original/proposer archive changed during audit')
    save_json(output/'audited_data_manifest.json', initial_manifest)
    result = {'status': 'complete_independent_fresh_geometry_prefix_public_branch_cost_audit_passed',
        'protocol': protocol, 'data_summary_sha256': sha(data/'summary.json'), 'source_hashes': sources,
        'checked_calls': len(checked), 'supported_calls': support.calls, 'native_prefix_pairs_checked': support.pairs,
        'complete_cost_calls': len(cost_checks), 'checked_calls_sha256': sha(output/'checked_calls.json'),
        'full_label_cost_checks_sha256': sha(output/'full_label_cost_checks.json'),
        'audited_data_manifest_sha256': sha(output/'audited_data_manifest.json'),
        'data_gate_passed': True, 'training_authorized_by_this_report': True,
        'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
        'scope': 'Complete new-data audit only. No optimizer/model efficacy/active control/holdout/safety or latency qualification.'}
    save_json(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    print(json.dumps(audit(args.data, args.output)), flush=True)
