"""Archive two real original-entry regressions, then re-run from ZIP evidence.

This ONLY verifies disabled/refused entry. Active-model faults, latency, complete
Levels and efficacy are NOT exercised and must remain explicitly false/pending.
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'cwm_v28'))
from package_ranking_release import safe_name

STATUS = 'original_sequential_entry_off_refusal_regression_passed'
SCHEMA = 'cwm_v29.disabled_original_entry_regression_archive.v1'
SELF_SOURCE = 'verification_source/experiments/cwm_v29/release_fallback_regression.py'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_report(read, stage):
    report = json.loads(read(stage + '/summary.json'))
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    if report['status'] != STATUS or report['protocol'] != protocol:
        raise ValueError('Published original-entry regression protocol/status differs')
    if any(report[k] is not False for k in ('enhanced_control_enabled', 'holdout_used', 'new_model_trained',
        'active_research_selector_exercised', 'actual_v28_checkpoint_load_failure_exercised', 'actual_v28_forward_failure_exercised')):
        raise ValueError('Regression cannot qualify active control or faults')
    if set(report['modes']) != {'plain', 'off', 'refusal'}:
        raise ValueError('All original/off/refusal modes required')
    names = {'summary.json'}
    for name, value in report['source_hashes'].items():
        safe_name(name)
        if digest(read(stage + '/source/' + name)) != value or digest((ROOT / name).read_bytes()) != value:
            raise ValueError('Original-entry used source differs')
        names.add('source/' + name)
    for mode, evidence in report['modes'].items():
        if (evidence['original_parent_classes_used'] is not True or evidence['optional_context_or_model_read'] is not False or
            evidence['historical_full_trajectory_equal'] is not True or evidence['plain_plan_and_cbf_commands_equal'] is not True or
            [r['episode_index'] for r in evidence['rows']] != protocol['fallback_record_ids']):
            raise ValueError('Complete actual fixed historical regression required')
        for row in evidence['rows']:
            stem = f"{mode}/level{row['level']}_{row['episode_index']}"
            names.update({stem + '.npz', stem + '.commands.npz'})
    return report, names


def same_arrays(a, b):
    with np.load(io.BytesIO(a), allow_pickle=False) as first, np.load(io.BytesIO(b), allow_pickle=False) as second:
        return set(first.files) == set(second.files) and all(first[k].dtype == second[k].dtype and
            first[k].shape == second[k].shape and first[k].tobytes() == second[k].tobytes() for k in first.files)


def compare(read, first, second):
    a, names = validate_report(read, first)
    b, other = validate_report(read, second)
    if a != b or names != other:
        raise ValueError('Independent original-entry regression reports differ')
    for name in names:
        if name.endswith('.npz') and not same_arrays(read(first+'/'+name), read(second+'/'+name)):
            raise ValueError('Independent original full trajectory/plan/CBF commands differ')
    return a, names


def verify(artifact, output):
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(artifact) as archive:
        names = [safe_name(i.filename) for i in archive.infolist()]
        if len({n.casefold() for n in names}) != len(names) or 'ARTIFACT_MANIFEST.json' not in names:
            raise ValueError('Invalid archive inventory')
        manifest = json.loads(archive.read('ARTIFACT_MANIFEST.json'))
        if manifest['schema'] != SCHEMA or set(manifest['members']) != set(names)-{'ARTIFACT_MANIFEST.json'}:
            raise ValueError('Complete disabled-entry archive inventory differs')
        for name, value in manifest['members'].items():
            if digest(archive.read(name)) != value:
                raise ValueError('Disabled-entry archive digest differs')
        previous, evidence = compare(archive.read, 'primary', 'repeated')
        expected = {stage+'/'+name for stage in ('primary', 'repeated') for name in evidence} | {SELF_SOURCE}
        if set(manifest['members']) != expected:
            raise ValueError('Disabled-entry archive added/omitted required evidence')
        if archive.read(SELF_SOURCE) != Path(__file__).read_bytes():
            raise ValueError('Regression replay/release source differs')
        fresh = output / 'real_original_replay'
        with (output/'stdout.log').open('x', encoding='utf8') as stdout, (output/'stderr.log').open('x', encoding='utf8') as stderr:
            subprocess.run([sys.executable, str(HERE/'sequential_entry.py'), '--fallback-regression', '--output', str(fresh)],
                           cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        def read(name):
            stage, relative = name.split('/', 1)
            return (fresh/relative).read_bytes() if stage == 'fresh' else archive.read(name)
        compare(read, 'primary', 'fresh')
        result = {'status': 'two_disabled_entry_regressions_archived_and_real_original_entry_replayed',
            'artifact_sha256': digest(Path(artifact).read_bytes()), 'artifact_bytes': Path(artifact).stat().st_size,
            'members': len(expected), 'fixed_historical_record_ids': previous['protocol']['fallback_record_ids'],
            'all_original_trajectories_plans_cbf_commands_replayed': True, 'optional_model_or_context_read': False,
            'enhanced_control_enabled': False, 'holdout_used': False, 'active_research_selector_exercised': False,
            'actual_v28_checkpoint_faults_exercised': False, 'latency_qualified': False,
            'scope': 'Two historical records, exact off/refusal paths only. NOT V28 training, active selector, full Levels, safety/latency qualification or capture gains.'}
    with (output/'summary.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    return result


def package(primary, repeated, artifact, replay_output):
    if artifact.exists() or replay_output.exists():
        raise ValueError('Exclusive new regression archive/replay outputs required')
    roots = {'primary': primary, 'repeated': repeated}
    def read(name):
        stage, relative = name.split('/', 1)
        return (roots[stage]/relative).read_bytes()
    _, evidence = compare(read, 'primary', 'repeated')
    members = {stage+'/'+name: read(stage+'/'+name) for stage in roots for name in evidence}
    members[SELF_SOURCE] = Path(__file__).read_bytes()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)
        archive.writestr('ARTIFACT_MANIFEST.json', json.dumps({'schema': SCHEMA,
            'members': {n: digest(raw) for n, raw in members.items()}}, indent=2))
    return verify(artifact, replay_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    for name in ('primary', 'repeated', 'artifact'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--replay-output', type=Path, required=True)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify, args.replay_output)
    else:
        if any(getattr(args, n) is None for n in ('primary', 'repeated', 'artifact')):
            parser.error('Both complete actual original regressions and artifact output required')
        result = package(args.primary, args.repeated, args.artifact, args.replay_output)
    print(json.dumps(result), flush=True)
