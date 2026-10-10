"""Finite new-scene legality and paired-response diagnostic; control disabled."""
import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'cwm_v7'))
from geometry import BackboneTap, s4_commands, s4_rollout, fingerprint, verify
from geometry_release import arrays, compare_arrays, target_bad, sha, BASELINE_SHA


def validate_protocol(protocol):
    if any(protocol[k] for k in ('enhanced_control_enabled', 'new_model_trained', 'private_labels_model_inputs', 'holdout_used')):
        raise ValueError('Disabled original-only diagnostic contract violated')
    if protocol['target_branch_rule_overrides'] or protocol['baseline_capsule_sha256'] != BASELINE_SHA:
        raise ValueError('Original target/baseline changed')
    variants = protocol['variants']
    if len(variants) != 4 or len({v['name'] for v in variants}) != 4 or protocol['groups'] != 8:
        raise ValueError('Fixed four-variant eight-group population required')
    reference = (1.5, 1., 1.)
    keys = ('wall_center_x_m', 'defender_y_multiplier', 'wall_y_multiplier')
    if tuple(variants[0][k] for k in keys) != reference:
        raise ValueError('Reference geometry changed')
    if any(sum(v[k] != x for k, x in zip(keys, reference)) != 1 for v in variants[1:]):
        raise ValueError('New scenes must change exactly one geometry axis')
    episodes = [v['seed_start'] + i for v in variants for i in range(protocol['groups']*2)]
    if len(set(episodes)) != len(episodes) or any(966010 <= i <= 966025 for i in episodes):
        raise ValueError('Duplicate or holdout episode identity')
    if not set(range(protocol['layout_seed_start'], protocol['layout_seed_start']+8)).isdisjoint(range(1966010,1966018)):
        raise ValueError('Reserved holdout layouts reused')
    if protocol['response_horizons'] != [8,16] or protocol['horizon_steps'] != 80 or protocol['maximum_command_speed_mps'] != 5.:
        raise ValueError('Fixed diagnostic horizon/commands changed')


def transform(scenario, variant):
    shift = float(variant['wall_center_x_m'])
    wall = copy.deepcopy(scenario.obstacles[0])
    extent = np.asarray(wall.half_extents_xy).copy()
    extent[1] *= variant['wall_y_multiplier']
    wall = replace(wall, center_xy=wall.center_xy + np.array([shift,0.]), half_extents_xy=extent)
    positions = scenario.defender_positions.copy()
    positions[:,1] *= variant['defender_y_multiplier']
    return replace(scenario, obstacles=(wall,), defender_positions=positions,
                   obstacle_zone_x=tuple(np.asarray(scenario.obstacle_zone_x)+shift),
                   name=scenario.name+'_'+variant['name'])


def scene_records(base, protocol):
    from encirclement3d.showcase import (s4_adaptive_branching_scenario, scenario_metadata,
        s4_branch_route_metrics, validate_s4_branching_scenario)
    records = []
    conditions = [(speed, condition) for speed in protocol['target_speed_scales'] for condition in protocol['observation_conditions']]
    for group in range(protocol['groups']):
        speed, condition = conditions[group % len(conditions)]
        for member, bias in enumerate(('upper','lower')):
            noise_seed = protocol['variants'][0]['seed_start'] + group*2 + member
            parent = {'episode_index': noise_seed, 'episode_seed': noise_seed,
                      'layout_seed': protocol['layout_seed_start']+group,
                      'mirror_group_id': f"cwm-newscene-{protocol['layout_seed_start']+group}",
                      'mirror_pair_member': bias, 'target_motion_mode': 'adaptive_branching',
                      'target_speed_scale': speed, 'obstacle_count': 1, 'level': None,
                      'model_split': protocol['stage'], 'observation_condition': condition['name'],
                      'pursuit_overrides': copy.deepcopy(condition['overrides']),
                      'execution': {'enabled': False, 'action_delay_steps': 0, 'command_noise_std': 0.}}
            env = base.env_class(base.configuration(parent), obstacle_count=1, target_speed_scale=speed)
            try:
                original = s4_adaptive_branching_scenario(env, parent['layout_seed'], bias, protocol['scene_variation'])
                error = None
            except ValueError as exc:
                original, error = None, str(exc)
            for variant in protocol['variants']:
                record = {**copy.deepcopy(parent), 'episode_index': variant['seed_start']+group*2+member,
                          'controller_sampling_identity': noise_seed, 'variant': variant['name'],
                          'geometry_axis': copy.deepcopy(variant)}
                record['route_valid'] = False
                if original is None:
                    record['route_error'] = error
                else:
                    scenario = transform(original, variant)
                    record['scenario'] = scenario_metadata(scenario)
                    try:
                        validate_s4_branching_scenario(env, scenario)
                        record['route_validation'] = s4_branch_route_metrics(env, scenario)
                        record['route_valid'] = True
                    except ValueError as exc:
                        record['route_error'] = str(exc)
                records.append(record)
    return records


def response_statistics(windows, read, horizon, threshold):
    grouped = {}
    for window in windows:
        values = arrays(read(window['arrays_path']))
        common = values['valid'][1:,:horizon] & values['valid'][:1,:horizon]
        effect = np.linalg.norm(values['target'][1:,:horizon]-values['target'][:1,:horizon],axis=-1)
        flipped = ((values['branch_sign_label_only'][1:,:horizon] != values['branch_sign_label_only'][:1,:horizon])
                   & (values['branch_sign_label_only'][1:,:horizon] != 0)
                   & (values['branch_sign_label_only'][:1,:horizon] != 0) & common)
        g = grouped.setdefault(window['group'], {'effects': [], 'full_pairs': 0, 'flip_pairs': 0})
        g['effects'].extend(effect[common].tolist())
        g['full_pairs'] += int(common.all(-1).sum())
        g['flip_pairs'] += int(flipped.any(-1).sum())
    by_group = {g: {'valid_points': len(v['effects']), 'mean_response_m': float(np.mean(v['effects'])) if v['effects'] else None,
                    'over_threshold_fraction': float(np.mean(np.asarray(v['effects']) > threshold)) if v['effects'] else None,
                    'over_threshold_points': int(np.count_nonzero(np.asarray(v['effects']) > threshold)),
                    'full_pairs': v['full_pairs'], 'flip_pairs': v['flip_pairs']} for g,v in grouped.items()}
    supported = [v for v in by_group.values() if v['valid_points']]
    return {'horizon': horizon, 'groups_with_valid_support': len(supported), 'by_group': by_group,
            'signal_groups': sorted(g for g,v in by_group.items() if v['over_threshold_points']),
            'valid_points': sum(v['valid_points'] for v in supported),
            'group_equal_mean_response_m': float(np.mean([v['mean_response_m'] for v in supported])) if supported else None,
            'group_equal_over_threshold_fraction': float(np.mean([v['over_threshold_fraction'] for v in supported])) if supported else None}


def variant_summary(variant, scenes, episodes, windows, read, protocol):
    selected = [r for r in scenes if r['variant'] == variant]
    rows = [r for r in episodes if r['variant'] == variant]
    probes = [r for r in windows if r['variant'] == variant]
    response = {str(h): response_statistics(probes,read,h,protocol['effect_threshold_m']) for h in protocol['response_horizons']}
    invalid_branches = 0
    for window in probes:
        values = arrays(read(window['arrays_path']))
        observed = values['termination'] != 'after_terminal'
        invalid_branches += int(((~values['valid']) & observed).any(-1).sum())
    gate = protocol['scene_gate']
    legality = (len(rows) == len(selected) and all(r['route_valid'] for r in selected)
                and sum(r['target_invalid_episode'] for r in rows) <= gate['maximum_original_target_invalid_episodes']
                and invalid_branches <= gate['maximum_stress_target_invalid_branches'])
    signal = response['8']
    support = (signal['groups_with_valid_support'] == protocol['groups']
               and len(signal['signal_groups']) >= gate['minimum_signal_groups_at_8_steps']
               and signal['group_equal_over_threshold_fraction'] is not None
               and signal['group_equal_over_threshold_fraction'] >= gate['minimum_effect_over_threshold_fraction_at_8_steps'])
    return {'variant': variant, 'assigned_episodes': len(selected), 'route_invalid_scenes': sum(not r['route_valid'] for r in selected),
            'completed_episodes': len(rows), 'original_outcomes': {k:sum(r[k] for r in rows) for k in
                ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode')},
            'windows':len(probes), 'stress_target_invalid_branches':invalid_branches,'response':response,
            'finite_legality_gate_passed':bool(legality),'response_support_gate_passed':bool(support),
            'eligible_for_fresh_confirmation':bool(legality and support),'model_training_authorized_by_this_report':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule',type=Path,default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    parser.add_argument('--protocol',type=Path,default=ROOT/'scene_protocol.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if sha(args.capsule) != BASELINE_SHA:
        raise ValueError('Protected original capsule mismatch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    base = BackboneTap(args.capsule,args.output/'restored')
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {Path(__file__).resolve(),args.protocol.resolve()}
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        destination = args.output/'source'/source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
    started = time.perf_counter()
    scenes = scene_records(base,protocol)
    (args.output/'scenes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scenes))
    episodes,windows,skips = [],[],[]
    for scene in scenes:
        if not scene['route_valid']:
            continue
        index = scene['episode_index']
        observed_steps = []
        def observer(env,observation,actions,sequence):
            if env.step_count not in protocol['snapshot_steps']:
                return
            observed_steps.append(int(env.step_count))
            before = fingerprint(env)
            reference,_ = base.evaluator.belief_reference(observation,base.planner_config)
            proposals = s4_commands(sequence,observation,reference,protocol)
            # Same external noise for all alternatives, including anchor.
            key = int.from_bytes(hashlib.sha256(f"{protocol['noise_seed']}/{scene['controller_sampling_identity']}/{env.step_count}".encode()).digest()[:4],'little')
            results = [s4_rollout(base,env,proposed,key) for proposed in proposals]
            values = {k:np.stack([v[k] for v in results]) for k in results[0]}
            path = args.output/'windows'/f'{index}_step{env.step_count}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**values,proposed=proposals,public_reference=reference)
            windows.append({'episode_index':index,'variant':scene['variant'],'group':scene['mirror_group_id'],
                            'step':int(env.step_count),'noise_key':key,'arrays_path':path.relative_to(args.output).as_posix(),
                            'arrays_sha256':sha(path),'parent_unchanged':True})
            if before != fingerprint(env):
                raise AssertionError('Shadow probes changed original parent')
        # All controller stochastic inputs match the reference group/mirror.
        # Only outer records and artifact paths use variant-specific indices.
        controller_record = {**scene,'episode_index':scene['controller_sampling_identity']}
        observed = args.output/'observed'/f'{index}.npz'
        plain = args.output/'plain'/f'{index}.npz'
        row,_ = base.run(controller_record,observed,observer)
        original,_ = base.run(controller_record,plain)
        if not compare_arrays(arrays(observed.read_bytes()),arrays(plain.read_bytes())):
            raise AssertionError('New-scene shadow changed original trajectories')
        keys = ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason')
        if any(row[k] != original[k] for k in keys):
            raise AssertionError('New-scene shadow changed physical outcomes')
        if bool(target_bad(arrays(observed.read_bytes())['target_positions'],scene['scenario']).any()) != row['target_invalid_episode']:
            raise AssertionError('Target physical invalidity differs from original flag')
        episodes.append({'episode_index':index,'variant':scene['variant'],'group':scene['mirror_group_id'],
                         **{k:row[k] for k in keys},'trajectory_byte_equal':True,
                         'observed_sha256':sha(observed),'plain_sha256':sha(plain)})
        skips.extend({'episode_index':index,'variant':scene['variant'],'step':step,'reason':'original_terminated_before_snapshot'}
                     for step in protocol['snapshot_steps'] if step not in observed_steps)
        for name,value in (('episodes',episodes),('windows',windows),('skips',skips)):
            (args.output/(name+'.json')).write_text(json.dumps(value,indent=2))
        print(json.dumps({'episodes':len(episodes),'variant':scene['variant'],'windows':len(windows)}),flush=True)
    verify(base.root,base.capsule_manifest)
    if any(sha(ROOT.parent/n) != h for n,h in hashes.items()):
        raise AssertionError('Run-used new-scene sources changed')
    results = [variant_summary(v['name'],scenes,episodes,windows,lambda n:(args.output/n).read_bytes(),protocol) for v in protocol['variants']]
    summary = {'status':'finite_scene_diagnostic_finished_not_model_qualification','protocol':protocol,'source_hashes':hashes,
               'assigned_scenes':len(scenes),'episodes':len(episodes),'windows':len(windows),'skips':len(skips),
               'scene_sha256':sha(args.output/'scenes.jsonl'),'variants':results,
               'baseline_capsule_sha256':BASELINE_SHA,'enhanced_control_enabled':False,'new_model_trained':False,
               'holdout_used':False,'elapsed_seconds':time.perf_counter()-started,'scope':protocol['scope']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
