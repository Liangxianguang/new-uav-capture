"""Package and recheck two representative entry fault-injection replays."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from original_entry_boundary import BASELINE_SHA, ROOT, REFERENCE, compare_rows

MODES = ('plain', 'off', 'refusal', 'load_failure', 'prediction_failure', 'synthetic_shadow')
FLAGS = ('all_outcomes_equal_to_off', 'all_trajectories_byte_equal_to_off',
         'historical_capsule_trajectories_equal', 'commands_and_plans_equal_to_plain',
         'source_and_artifact_integrity_after_replay')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def exact(left, right, fields):
    return all(left[k].shape == right[k].shape and left[k].dtype == right[k].dtype and
               left[k].tobytes() == right[k].tobytes() for k in fields)


def audit(read, capsule):
    if digest(capsule.read_bytes()) != BASELINE_SHA:
        raise ValueError('Historical capsule differs')
    summaries = [json.loads(read(stage + '/summary.json')) for stage in ('primary', 'repeated')]
    records = summaries[0]['modes']['plain']['rows']
    if not records:
        raise ValueError('Empty representative replay')
    with zipfile.ZipFile(capsule) as baseline:
        historical = {r['episode_index']: r for r in map(json.loads,
            baseline.read(REFERENCE + '/episodes.jsonl').decode().splitlines())}
        for stage, summary in zip(('primary', 'repeated'), summaries):
            if (summary['status'] != 'passed_original_entry_optional_boundary_fallback_audit' or
                    summary['enhanced_control_enabled'] is not False or
                    summary['baseline_capsule_sha256'] != BASELINE_SHA or
                    any(summary[key] is not True for key in FLAGS) or
                    set(summary['modes']) != set(MODES) or summary['episodes'] != len(records)):
                raise ValueError('Not a completed fault-injection replay')
            if summary['source_hashes'] != summaries[0]['source_hashes']:
                raise ValueError('Two entry replay sources differ')
            for name, sha in summary['source_hashes'].items():
                if digest((ROOT / name).read_bytes()) != sha or digest(read('source/' + name)) != sha:
                    raise ValueError('Saved or current entry source differs')
            for mode in MODES:
                measured = summary['modes'][mode]
                rows = measured['rows']
                if [r['episode_index'] for r in rows] != [r['episode_index'] for r in records]:
                    raise ValueError('Fixed replay support differs')
                if any(e['control_eligible'] is not False for e in measured['events']):
                    raise ValueError('Fault injection must not control')
                loads, predicts = measured['load_calls'], measured['prediction_calls']
                expected = {'plain': (0, 0), 'off': (0, 0), 'refusal': (0, 0),
                            'load_failure': (1, 0), 'prediction_failure': (1, 1)}
                if mode in expected and (loads, predicts) != expected[mode]:
                    raise ValueError('Fault path was not exercised')
                statuses = {'plain': 'off', 'off': 'off', 'refusal': 'qualification_failed',
                            'load_failure': 'load_failed', 'prediction_failure': 'prediction_failed',
                            'synthetic_shadow': 'shadow_ready'}
                if measured['status'] != statuses[mode] or (mode != 'plain' and not measured['events']):
                    raise ValueError('Missing or incorrect entry events')
                if mode == 'synthetic_shadow' and not (loads == 1 and predicts == len(measured['events'])):
                    raise ValueError('Synthetic shadow did not run on all plans')
                for row, reference in zip(rows, records):
                    if compare_rows(reference, row) or compare_rows(historical[row['episode_index']], row):
                        raise ValueError('Physical outcome differs')
                    name = f"level{row['level']}_{row['episode_index']}.npz"
                    physical = arrays(read(f'{stage}/{mode}/{name}'))
                    command_name = name[:-4] + '.commands.npz'
                    commands = arrays(read(f'{stage}/{mode}/{command_name}'))
                    archived = arrays(baseline.read(REFERENCE + '/trajectories/' + name))
                    plain = arrays(read(f'{stage}/plain/{command_name}'))
                    if not exact(physical, archived, ('defender_positions', 'target_positions')):
                        raise ValueError('Physical trajectory differs from historical capsule')
                    if not exact(commands, plain, ('commanded', 'planned')):
                        raise ValueError('DN-MPC plan or post-CBF command differs from plain')
                    for filename, fields in ((name, ('defender_positions', 'target_positions')),
                                             (command_name, ('commanded', 'planned'))):
                        if not exact(arrays(read(f'primary/{mode}/{filename}')),
                                     arrays(read(f'repeated/{mode}/{filename}')), fields):
                            raise ValueError('Independent entry replay arrays differ')
    return {'status': 'passed_two_process_post_plan_fault_injection_replay',
            'episodes_per_mode_per_run': len(records), 'independent_runs': 2, 'modes': list(MODES),
            'episode_indices': [r['episode_index'] for r in records],
            'levels': [r['level'] for r in records], 'baseline_capsule_sha256': BASELINE_SHA,
            'historical_defender_target_trajectory_bytes_equal': True,
            'planned_and_post_cbf_command_bytes_equal_to_plain': True,
            'two_run_arrays_exact': True, 'enhanced_control_enabled': False,
            'real_checkpoint_loaded': False, 'holdout_used': False,
            'source_hashes': summaries[0]['source_hashes'], 'scope': summaries[0]['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path)
    parser.add_argument('--repeated', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--capsule', type=Path, default=ROOT / 'experiments/cwm_v1/baseline/capsule.zip')
    args = parser.parse_args()
    if args.verify:
        with zipfile.ZipFile(args.verify) as z:
            manifest = json.loads(z.read('ARTIFACT_MANIFEST.json'))
            if set(z.namelist()) != set(manifest) | {'ARTIFACT_MANIFEST.json'}:
                raise ValueError('Archive member population differs')
            if any(digest(z.read(name)) != sha for name, sha in manifest.items()):
                raise ValueError('Archive member hash differs')
            result = audit(z.read, args.capsule)
            if json.loads(z.read('release_summary.json')) != result:
                raise ValueError('Archive summary differs from independently recomputed result')
        print(json.dumps({'status': result['status'], 'sha256': digest(args.verify.read_bytes())}))
        return
    if any(x is None for x in (args.primary, args.repeated, args.output, args.report)):
        parser.error('Two completed runs, output archive and report required')
    members = {}
    for stage, directory in (('primary', args.primary), ('repeated', args.repeated)):
        members[stage + '/summary.json'] = (directory / 'summary.json').read_bytes()
        for mode in MODES:
            for p in sorted((directory / mode).glob('*.npz')):
                members[f'{stage}/{mode}/{p.name}'] = p.read_bytes()
    summary = json.loads(members['primary/summary.json'])
    for name in summary['source_hashes']:
        members['source/' + name] = (ROOT / name).read_bytes()
    result = audit(members.__getitem__, args.capsule)
    members['release_summary.json'] = json.dumps(result, indent=2, allow_nan=False).encode()
    members['verification_source/release_entry_audit.py'] = Path(__file__).read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in members.items():
            z.writestr(name, raw)
        z.writestr('ARTIFACT_MANIFEST.json', json.dumps({k: digest(v) for k, v in members.items()}, indent=2))
    with zipfile.ZipFile(args.output) as z:
        if audit(z.read, args.capsule) != result:
            raise ValueError('Archived replay differs')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf8') as file:
        json.dump({**result, 'artifact_sha256': digest(args.output.read_bytes()),
                   'artifact_bytes': args.output.stat().st_size}, file, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'artifact_bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
