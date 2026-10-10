"""Public-only reconstruction of the frozen finite generator; no control changes."""
import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def clip_rows(values, maximum):
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows * np.minimum(1., float(maximum) / np.maximum(norms, 1e-12))


def public_interceptor(own, agent, known_positions, reference):
    positions = {agent: np.asarray(own, dtype=np.float64)}
    positions.update({peer: np.asarray(position) for peer, position in known_positions.items()})
    if any(not np.isfinite(v).all() or v.shape != (3,) for v in positions.values()):
        raise ValueError('Finite public peer positions required')
    return min(positions, key=lambda peer: float(np.linalg.norm(positions[peer] - reference[0])))


def tracking_commands(own, agent, interceptor, reference, velocity, config, directions, clip_speed=True):
    """Copy only public tracking algebra, independent of the original generator."""
    own, reference, velocity = map(np.asarray, (own, reference, velocity))
    if (own.shape != (3,) or reference.shape != (config.horizon_steps, 3) or velocity.shape != (3,) or
        any(not np.isfinite(v).all() for v in (own, reference, velocity))):
        raise ValueError('Finite original public tracking inputs required')
    result = []
    for scale in config.perimeter_scales:
        current, actions = own.astype(np.float64).copy(), []
        perimeter = config.role_perimeter_m * float(scale)
        for t in range(config.horizon_steps):
            if t >= config.control_horizon_steps and actions:
                action = actions[-1].copy()
            else:
                target = reference[t]
                target_velocity = (reference[t] - reference[t-1]) / config.dt_seconds if t > 0 else velocity
                target_point = target if agent == interceptor else target + directions[agent % len(directions)] * perimeter
                action = config.slot_gain * (target_point - current)
                action += config.target_velocity_gain * target_velocity
                if clip_speed:
                    action = clip_rows(action[None, :], config.max_speed_mps)[0]
            actions.append(action)
            current = current + action * config.dt_seconds
        result.append(np.stack(actions))
    return np.stack(result)


def duplicate_reason(actual, unclipped, is_interceptor):
    if actual.shape != (2, 8, 3) or unclipped.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(unclipped).all():
        raise ValueError('Frozen finite two-candidate diagnosis contract required')
    if not np.array_equal(actual[0], actual[1]):
        if is_interceptor:
            raise ValueError('Interceptor perimeter invariance unexpectedly broken')
        return 'distinct'
    if is_interceptor:
        if not np.array_equal(unclipped[0], unclipped[1]):
            raise ValueError('Unclipped interceptor invariance unexpectedly broken')
        return 'interceptor_perimeter_invariant'
    if not np.array_equal(unclipped[0], unclipped[1]):
        return 'speed_clipping_induced_duplicate'
    return 'other_public_geometry_duplicate'


def summarize_records(rows):
    result = {}
    for split in ('train', 'development_validation'):
        result[split] = {}
        for support in ('all_collected', 'complete_union'):
            selected = [r for r in rows if r['split'] == split and (support == 'all_collected' or r['complete_union'])]
            if not selected:
                raise ValueError('Empty fixed diagnostic support')
            groups = collections.defaultdict(list)
            for r in selected:
                groups[r['group']].append(r)
            reasons = collections.Counter(r['duplicate_reason'] for r in selected)
            result[split][support] = {
                'calls': len(selected), 'groups': len(groups),
                'unique_candidate_count_histogram': dict(sorted(collections.Counter(str(r['unique_candidates']) for r in selected).items())),
                'duplicate_reason_counts': dict(sorted(reasons.items())),
                'group_equal_duplicate_reason_fractions': {
                    key: float(np.mean([np.mean([r['duplicate_reason'] == key for r in group]) for group in groups.values()]))
                    for key in sorted(reasons)},
                'group_equal_single_candidate_fraction': float(np.mean([
                    np.mean([r['unique_candidates'] == 1 for r in group]) for group in groups.values()]))}
    return result


def run(artifact, output):
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    if any(protocol[k] for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used', 'prior_gates_overridden')):
        raise ValueError('Diagnostic cannot enable, train or override failures')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT.parent / 'cwm_v23'))
    from cost_data_release import audit_data, frozen_configs, sha, BASELINE_SHA, resolve_artifact, verify_manifest
    from local_shadow import restore_public_call
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA or BASELINE_SHA != protocol['baseline_capsule_sha256']:
        raise ValueError('Original capsule differs')
    configs = frozen_configs(capsule, output / 'restored')
    archive = resolve_artifact(artifact, output / 'assembled')
    integrity = verify_manifest(archive)
    if integrity['sha256'] != protocol['data_archive_sha256'] or list(configs[0].perimeter_scales) != protocol['expected_perimeter_scales']:
        raise ValueError('Pinned DATA archive or original perimeter scales differ')
    from encirclement3d.minimax_mpc import belief_reference
    from encirclement3d.pursuit_env import TETRAHEDRON_DIRECTIONS
    from encirclement3d.distributed_dn_mpc import _qdr_execution_aware
    records = []
    with zipfile.ZipFile(archive) as z:
        calls, audited = audit_data(z.read, configs, output / 'restored')
        if len(calls) != protocol['source_calls'] or audited['skipped_calls']:
            raise ValueError('Full fixed V23 diagnosis population required')
        print(json.dumps({'stage': 'full_original_data_public_cost_audit_passed', 'calls': len(calls)}), flush=True)
        for c in calls:
            row, values = c['record'], c['values']
            snapshot = json.loads(z.read('data/' + row['context_path']))
            call = restore_public_call(snapshot, *configs, values['backbone'])
            if _qdr_execution_aware(call['observation']) or call['scenarios'].trajectories.shape != (1, 8, 3):
                raise ValueError('Frozen non-QDR one-path tracking contract required')
            reference = np.average(call['scenarios'].trajectories, axis=0, weights=call['scenarios'].normalized_weights)
            own = call['observation']['defender_positions'][row['agent']]
            interceptor = public_interceptor(own, row['agent'], {peer: m.position for peer, m in call['known'].items()}, reference)
            _, velocity = belief_reference(call['observation'], configs[0], anchor_index=row['agent'])
            rebuilt = tracking_commands(own, row['agent'], interceptor, reference, velocity, configs[0], TETRAHEDRON_DIRECTIONS)
            actual = call['local_candidates']
            if not np.array_equal(rebuilt, actual) or not np.array_equal(actual, values['actual_candidates']):
                raise ValueError('Independent public tracking algebra differs from original candidates')
            unclipped = tracking_commands(own, row['agent'], interceptor, reference, velocity, configs[0], TETRAHEDRON_DIRECTIONS, False)
            reason = duplicate_reason(actual, unclipped, row['agent'] == interceptor)
            records.append({**{k: row[k] for k in ('episode_index', 'step', 'agent', 'group', 'split')},
                'raw_candidates': len(actual), 'unique_candidates': 1 if reason != 'distinct' else 2,
                'public_interceptor': int(interceptor), 'duplicate_reason': reason,
                'complete_union': bool(values['valid'].all() and values['anchor_valid'].all()),
                'context_sha256': row['context_sha256'], 'independent_original_candidate_bytes_equal': True,
                'unclipped_diagnostic_pair_equal': bool(np.array_equal(unclipped[0], unclipped[1]))})
    records_raw = (json.dumps(records, indent=2, allow_nan=False) + '\n').encode('utf8')
    with (output / 'records.json').open('xb') as file:
        file.write(records_raw)
    summary = {'status': 'posthoc_public_candidate_cardinality_finished_not_promoted', 'protocol': protocol,
        'calls': len(records), 'records_sha256': hashlib.sha256(records_raw).hexdigest(),
        'data_archive_sha256': integrity['sha256'], 'full_original_data_public_cost_audit_passed': True,
        'all_independent_original_candidate_bytes_equal': True, 'supports': summarize_records(records),
        'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
        'source_hashes': {p.name: sha(p) for p in (ROOT / 'candidate_cardinality.py', ROOT / 'diagnostic_protocol.json')},
        'scope': protocol['scope']}
    source = output / 'source'
    source.mkdir()
    for name in summary['source_hashes']:
        with (source / name).open('xb') as file:
            file.write((ROOT / name).read_bytes())
    with (output / 'summary.json').open('x', encoding='utf8', newline='\n') as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': summary['status'], 'supports': summary['supports']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact, args.output)


if __name__ == '__main__':
    main()
