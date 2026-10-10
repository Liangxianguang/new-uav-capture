"""Reconstruct public features, geometry candidates and full original scores."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v18'),str(ROOT.parent/'cwm_v10')]
from fresh_cost_data import validate_protocol, assigned_split, split_statistics
from two_head_release import check_geometry, encoder_environment, audit_sources
from replay_cost_public import decode_public, compare_public, restore_public_call
from local_shadow import delayed_joint_context, local_cost
from geometry_probes import public_geometry_pool
from release_verify import _check_branch
from support_audit import map_actual, scored_record
from geometry_release import arrays, compare_arrays, target_bad, sha, BASELINE_SHA


def read_dataset(data, public):
    def read(name):
        stage, relative = name.split('/',1)
        if stage not in ('data','public'):
            raise ValueError('Unexpected data audit stage')
        return ((data if stage == 'data' else public)/relative).read_bytes()
    return read


def audit_data(read, configs, restored):
    from encirclement3d.minimax_mpc import belief_reference
    from encirclement3d.observation_encoding import policy_observations
    protocol = json.loads((ROOT/'data_protocol.json').read_text())
    validate_protocol(protocol)
    report = json.loads(read('data/summary.json'))
    public = json.loads(read('public/summary.json'))
    if report['protocol'] != protocol or public['protocol'] != protocol:
        raise ValueError('Actual fresh data/public protocol changed')
    if (report['status'] != 'fresh_cost_contrast_data_finished_not_promoted'
            or public['status'] != 'independent_original_public_cost_data_replay_frames_equal'
            or any(stage[key] for stage in (report, public) for key in
                   ('enhanced_control_enabled','new_model_trained','holdout_used'))
            or report['capsule_sha256'] != BASELINE_SHA or public['capsule_sha256'] != BASELINE_SHA):
        raise ValueError('Frozen original-only data/public contract changed')
    for stage, value in (('data',report),('public',public)):
        audit_sources(read,stage,value)
        if json.loads(read(stage+'/source/cwm_v23/data_protocol.json')) != protocol:
            raise ValueError('Saved preregistered data protocol changed')
    for name, key in (('scenes.jsonl','scene_sha256'),('records.json','records_sha256'),('decisions.json','decisions_sha256')):
        if hashlib.sha256(read('data/'+name)).hexdigest() != report[key]:
            raise ValueError('Actual dataset digest changed')
    if public['data_summary_sha256'] != hashlib.sha256(read('data/summary.json')).hexdigest():
        raise ValueError('Independent original replay provenance changed')
    scenes = [json.loads(line) for line in read('data/scenes.jsonl').decode().splitlines()]
    by_id = {s['episode_index']:s for s in scenes}
    if (len(scenes) != 64 or len(by_id) != 64 or {s['episode_seed'] for s in scenes} != set(range(993010,993074))
            or {s['layout_seed'] for s in scenes} != set(range(1993010,1993042))):
        raise ValueError('Actual fresh32 mirror-group population changed')
    # The legacy pure deterministic geometry check is reused, not its datasets,
    # candidates or weights. Its four-variant-cycle assignment is equivalent.
    geometry_protocol = {**protocol,'variant_cycles_split':['train']*6+['development_validation']*2}
    for scene in scenes:
        check_geometry(scene,geometry_protocol)
        if scene['model_split'] != assigned_split(scene['layout_seed']-protocol['layout_seed_start'],protocol):
            raise ValueError('Actual mirror-group split changed')
    episodes = json.loads(read('data/episodes.json'))
    public_rows = {r['episode_index']:r for r in public['episodes']}
    if episodes != report['episodes'] or len(episodes) != 64 or set(public_rows) != set(by_id) or len(public['episodes']) != 64:
        raise ValueError('Actual original/public episode coverage changed')
    encoder = encoder_environment(restored)
    frames_by_id, expected_ids = {}, set()
    for row in episodes:
        index = row['episode_index']
        if row['split'] != by_id[index]['model_split']:
            raise ValueError('Actual episode split changed')
        observed = arrays(read(f'data/observed/{index}.npz'))
        plain = arrays(read(f'data/plain/{index}.npz'))
        replay = arrays(read(f'public/trajectories/{index}.npz'))
        if not row['trajectory_byte_equal'] or not compare_arrays(observed,plain) or not compare_arrays(observed,replay):
            raise ValueError('Actual original observer/public replay arrays differ')
        for kind in ('observed','plain'):
            if hashlib.sha256(read(f'data/{kind}/{index}.npz')).hexdigest() != row[kind+'_sha256']:
                raise ValueError('Actual original trajectory digest changed')
        if not public_rows[index]['trajectory_byte_equal'] or hashlib.sha256(read(f'public/trajectories/{index}.npz')).hexdigest() != public_rows[index]['trajectory_sha256']:
            raise ValueError('Independent original public trajectory changed')
        if bool(target_bad(observed['target_positions'],by_id[index]['scenario']).any()) != row['target_invalid_episode']:
            raise ValueError('Actual original target geometry validity differs')
        raw = read(f'public/frames/{index}.json')
        if hashlib.sha256(raw).hexdigest() != public_rows[index]['frames_sha256']:
            raise ValueError('Saved independent public frames digest changed')
        frames = decode_public(json.loads(raw))
        if (len(frames) != len(observed['target_positions'])-1 or len(frames) != public_rows[index]['frames']
                or [f['step'] for f in frames] != list(range(len(frames)))):
            raise ValueError('Incomplete original public frame trace')
        for frame in frames:
            encoded = policy_observations(encoder,frame['observation']).reshape(-1)
            if encoded.shape != (252,) or encoded.tobytes() != frame['feature'].tobytes():
                raise ValueError('Independent original252 feature re-encoding differs')
            if not np.array_equal(frame['observation']['defender_positions'],observed['defender_positions'][frame['step']]):
                raise ValueError('Public state differs from original trajectory')
        frames_by_id[index] = frames
        expected_ids.update((index,t,agent) for t in protocol['snapshot_steps'] if t < len(frames) for agent in range(4))
    records = json.loads(read('data/records.json'))
    if len(records) != len(expected_ids) or {(r['episode_index'],r['step'],r['agent']) for r in records} != expected_ids:
        raise ValueError('Incomplete or duplicated actual local-call coverage')
    expected_public_rows = [{'episode_index':r['episode_index'],'step':r['step'],'agent':r['agent'],
                            'public_context_equal':True,'eligible_arrays':'arrays_path' in r} for r in records]
    if public['checked_calls'] != expected_public_rows:
        raise ValueError('Independent public local-call coverage differs')
    calls, decisions = [], []
    for row in records:
        scene = by_id[row['episode_index']]
        if row['split'] != scene['model_split'] or row['group'] != scene['mirror_group_id']:
            raise ValueError('Actual local-call split/group leakage')
        raw = read('data/'+row['context_path'])
        if hashlib.sha256(raw).hexdigest() != row['context_sha256']:
            raise ValueError('Actual local public context digest changed')
        snapshot = json.loads(raw)
        context = decode_public(snapshot)
        frame = frames_by_id[row['episode_index']][row['step']]
        extra = set(context['observation'])-set(frame['observation'])
        if extra != {'world_lower_bounds','world_upper_bounds'} or not np.array_equal(context['observation']['world_lower_bounds'],[-10.,-10.,.5]) or not np.array_equal(context['observation']['world_upper_bounds'],[10.,10.,10.]):
            raise ValueError('Original planner world boundary context changed')
        if any(k not in context['observation'] or not compare_public(v,context['observation'][k]) for k,v in frame['observation'].items()) or not np.array_equal(frame['backbone'],context['backbone']):
            raise ValueError('Actual local context/GRU differs from original public replay')
        call = restore_public_call(snapshot,*configs,context['backbone'])
        if call['agent'] != row['agent'] or call['ordinal'] != row['ordinal']:
            raise ValueError('Actual local call identity changed')
        for peer,message in call['known'].items():
            if (peer == row['agent'] or message.sender != peer or message.receiver != row['agent']
                    or message.delivery_step > row['step'] or row['step']-message.sent_step > configs[1].max_message_age_steps):
                raise ValueError('Actual delayed peer message provenance differs')
        reference,velocity = belief_reference(context['observation'],configs[0])
        raw_context = delayed_joint_context(call['observation'],call['known'],call['peer_sequences'],call['agent'],
                                            call['local_candidates'],call['selected'],reference,velocity)
        if raw_context is None:
            if 'arrays_path' in row or row['status'] != 'skipped_missing_delayed_peer_plan':
                raise ValueError('Unavailable delayed peer plan silently filled')
            continue
        if 'arrays_path' not in row:
            raise ValueError('Actual eligible local call omitted')
        raw = read('data/'+row['arrays_path'])
        if hashlib.sha256(raw).hexdigest() != row['arrays_sha256']:
            raise ValueError('Actual local branch array digest changed')
        values = arrays(raw)
        prefix = frames_by_id[row['episode_index']][:row['step']+1]
        history = np.stack([prefix[0]['feature']]*max(0,8-len(prefix))+[f['feature'] for f in prefix[-8:]])
        if not np.array_equal(values['history'],history) or not np.array_equal(values['backbone'],frame['backbone']):
            raise ValueError('Actual supplied252 history/GRU differs from original public replay')
        generated = call['planner']._local_candidate_sequences(call['observation'],call['scenarios'],call['agent'],
                                                               call['observation']['defender_positions'][call['agent']],call['known'])
        if not np.array_equal(np.stack(generated),call['local_candidates']) or not np.array_equal(values['actual_candidates'],call['local_candidates']):
            raise ValueError('Original actual candidate generation differs')
        pool = public_geometry_pool(call,reference,velocity,protocol['probe_magnitude_mps'])
        joint = delayed_joint_context(call['observation'],call['known'],call['peer_sequences'],call['agent'],pool,call['selected'],reference,velocity)
        if any(not np.array_equal(v,values[k]) for v,k in zip(joint,('proposed','anchor','relative'))):
            raise ValueError('Public geometry/delayed-peer/anchor reconstruction differs')
        if not np.array_equal(reference,values['reference']) or not np.array_equal(velocity,values['velocity']):
            raise ValueError('Original public belief reference differs')
        original_costs = local_cost(call,call['local_candidates'],np.repeat(values['backbone'][None],len(generated),0))
        choice = int(np.argmin(original_costs))
        if not np.allclose(original_costs,values['original_local_costs'],rtol=1e-10,atol=1e-8) or not np.array_equal(generated[choice],call['selected']):
            raise ValueError('Actual original local cost/choice differs')
        if not np.isclose(original_costs[choice],call['selected_costs'][0],rtol=1e-10,atol=1e-8):
            raise ValueError('Original copied selected local cost differs')
        equal = (values['proposed'] == values['anchor'][None]).all((1,2,3))
        if np.flatnonzero(equal).tolist() != [row['actual_choice_pool_index']]:
            raise ValueError('Original exact reference not unique in geometry pool')
        for j in range(len(pool)):
            branch = {k:values[k][j] for k in ('target','defenders','commanded','executed','valid','termination')}
            _check_branch(branch,'',scene['scenario'])
        _check_branch(values,'anchor_',scene['scenario'])
        for k in ('target','defenders','commanded','executed','valid','termination','branch_sign_label_only'):
            if not np.array_equal(values[k][row['actual_choice_pool_index']],values['anchor_'+k]):
                raise ValueError('Exact same-command branch differs from anchor')
        expected_key = int.from_bytes(hashlib.sha256(
            f"{protocol['noise_seed']}/{row['episode_index']}/{row['step']}/{row['agent']}".encode()).digest()[:4],'little')
        if row['noise_key'] != expected_key:
            raise ValueError('Frozen paired branch noise identity changed')
        current = {'record':row,'values':values}
        calls.append(current)
        actual,_ = map_actual(values,row['agent'])
        for name,indices in (('actual_unique',actual),('geometry_union',np.arange(len(pool)))):
            if values['valid'][indices].all() and values['anchor_valid'].all():
                decisions.append(scored_record(current,indices,name,call,{},protocol))
    if json.loads(read('data/decisions.json')) != decisions:
        raise ValueError('Independent full original costs/rankings differ')
    measured = split_statistics(calls,decisions,episodes,protocol)
    if any(report[k] != v for k,v in measured.items()) or not measured['training_eligible']:
        raise ValueError('Actual split data/support gate differs or failed')
    if report['calls'] != len(calls) or report['records'] != len(records) or report['skipped_calls'] != len(records)-len(calls):
        raise ValueError('Actual data call population changed')
    return calls,report
