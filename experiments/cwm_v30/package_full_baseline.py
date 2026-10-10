"""Complete original baseline + ALL off/refusal audit, then extracted real replay.

Requires matching Git history/source and the original Python environment. Never
qualifies causal control/faults/latency/holdout. Streaming archive/hash handling;
short NEW absolute replay directory required on Windows. No evidence subsets.
"""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v28'), str(HERE.parent/'cwm_v21')]
from audit_full_baseline import (validate_saved, verify_inventory, sha, read_json, read_lines,
    index_steps, episode_steps, require_equal, STEP_TIMING, EPISODE_TIMING, array_equal, summarize)
from package_ranking_release import safe_name, stream_digest
from artifact_transport import split_archive, resolve_artifact

SCHEMA = 'cwm_v30.complete_original_population_baseline_archive.v1'
MANIFEST = 'ARTIFACT_MANIFEST.json'
CAPSULE = 'experiments/cwm_v1/baseline/capsule.zip'
STATUS = 'independent_full_original_population_off_refusal_replay_passed'
FALSE_FLAGS = ('enhanced_control_enabled', 'holdout_used', 'new_model_trained',
    'active_model_faults_exercised', 'latency_qualified', 'artifact_packaged_and_replayed')


def write_json(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def audit_inventory(baseline, audit, capsule):
    """Semantic comparison of ALL saved evidence; not fresh native execution."""
    previous, records, original_rows, offsets, baseline_hashes = validate_saved(baseline, capsule)
    report = read_json(audit/'summary.json')
    if (report['status'] != STATUS or report['protocol'] != previous['protocol'] or
        report['baseline_summary_sha256'] != baseline_hashes['summary.json'] or
        report['original8Level_equal_macro'] != previous['original8Level_equal_macro'] or
        set(report['modes']) != {'off', 'refusal'} or
        report['excluded_nondeterministic_step_fields'] != sorted(STEP_TIMING) or
        report['excluded_nondeterministic_episode_fields'] != sorted(EPISODE_TIMING) or
        any(report[k] is not False for k in FALSE_FLAGS)):
        raise ValueError('Complete original audit/default-off/timing scope differs')
    if (read_json(audit/'audited_baseline_manifest.json') != baseline_hashes or
        sha(audit/'audited_baseline_manifest.json') != report['audited_baseline_manifest_sha256']):
        raise ValueError('Audited complete baseline manifest differs')
    hashes = {'summary.json': sha(audit/'summary.json'),
        'audited_baseline_manifest.json': report['audited_baseline_manifest_sha256']}
    for name, digest in report['source_hashes'].items():
        safe_name(name)
        if not name.startswith('experiments/') or sha(ROOT/name) != digest:
            raise ValueError('Full audit replay-used source differs')
        hashes['source/'+name] = digest
    for mode, evidence in report['modes'].items():
        if (evidence['episodes'] != 1280 or evidence['by_level'] != previous['by_level'] or
            evidence['original_parent_classes_used'] is not True or evidence['optional_context_or_model_read'] is not False or
            any(evidence[k] is not True for k in ('full_trajectory_plan_cbf_bytes_equal',
                'all_nontiming_native_episode_and_step_diagnostics_equal'))):
            raise ValueError('Complete actual original off/refusal support required')
        hashes[mode+'/episodes.jsonl'] = evidence['episodes_sha256']
        hashes[mode+'/steps.jsonl'] = evidence['steps_sha256']
        actual_rows = read_lines(audit/mode/'episodes.jsonl')
        if len(actual_rows) != 1280:
            raise ValueError('No truncated original audit episodes')
        actual_offsets = index_steps(audit/mode/'steps.jsonl', records)
        for original, actual, record in zip(original_rows, actual_rows, records):
            require_equal(original, actual, EPISODE_TIMING|{'trajectory_sha256', 'commands_sha256'}, 'Archived full original audit episode')
            stem = f"trajectories/level{record['level']}_{record['episode_index']}"
            for name, key in ((stem+'.npz', 'trajectory_sha256'), (stem+'.commands.npz', 'commands_sha256')):
                hashes[mode+'/'+name] = actual[key]
                array_equal(baseline/name, audit/mode/name)
            left = episode_steps(baseline/'steps.jsonl', offsets[record['episode_index']])
            right = episode_steps(audit/mode/'steps.jsonl', actual_offsets[record['episode_index']])
            if len(left) != len(right):
                raise ValueError('Complete archived step support differs')
            for a, b in zip(left, right):
                require_equal(a, b, STEP_TIMING, 'Archived full original audit step')
        if summarize(actual_rows) != evidence['by_level']:
            raise ValueError('Archived full original group-aware metrics differ')
    verify_inventory(audit, hashes)
    return previous, report, baseline_hashes, hashes


def inventory(baseline, audit):
    capsule = ROOT/CAPSULE
    previous, report, baseline_hashes, audit_hashes = audit_inventory(baseline, audit, capsule)
    members = {stage+'/'+name: folder/name for stage, folder, hashes in
        (('baseline', baseline, baseline_hashes), ('audit', audit, audit_hashes)) for name in hashes}
    sources = set(previous['source_hashes']) | set(report['source_hashes']) | {
        'experiments/cwm_v30/package_full_baseline.py', 'experiments/cwm_v30/audit_full_baseline.py',
        'experiments/cwm_v28/package_ranking_release.py', 'experiments/cwm_v21/artifact_transport.py'}
    members.update({'verification_source/'+name: ROOT/name for name in sources})
    members['dependencies/'+CAPSULE] = capsule
    for name, path in members.items():
        safe_name(name)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Required regular full original evidence missing')
    return members, previous, report


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [safe_name(i.filename) for i in infos]
        if (len(set(n.casefold() for n in names)) != len(names) or MANIFEST not in names or
            any(i.is_dir() or (i.external_attr >> 16)&0o170000 == 0o120000 for i in infos)):
            raise ValueError('Unsafe/aliased/nonregular full baseline ZIP inventory')
        manifest = json.loads(archive.read(MANIFEST))
        if manifest['schema'] != SCHEMA or set(manifest['members']) != set(names)-{MANIFEST}:
            raise ValueError('Full baseline manifest inventory differs')
        for info in infos:
            if info.filename == MANIFEST:
                continue
            value = manifest['members'][info.filename]
            with archive.open(info) as stream:
                digest = stream_digest(stream)  # Reads CRC as well as all hashes.
            if value['bytes'] != info.file_size or value['sha256'] != digest:
                raise ValueError('Full baseline archive member bytes/hash differ')
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'members': len(names)-1}


def compare_fresh(baseline, prior_audit, fresh):
    _, old, _, _ = audit_inventory(baseline, prior_audit, ROOT/CAPSULE)
    _, new, _, _ = audit_inventory(baseline, fresh, ROOT/CAPSULE)
    excluded = {'source_hashes', 'audited_baseline_manifest_sha256', 'baseline_summary_sha256', 'modes'}
    require_equal(old, new, excluded, 'Fresh all-population native audit scope')
    for mode in ('off', 'refusal'):
        require_equal(old['modes'][mode], new['modes'][mode], {'episodes_sha256', 'steps_sha256'}, 'Fresh native audit complete mode')
    return new


def replay_archive(artifact, output):
    resolved = resolve_artifact(artifact, output.parent/(output.name+'-transport'))
    integrity = verify_archive(resolved)
    output.mkdir(parents=True, exist_ok=False)
    extracted = output/'e'
    extracted.mkdir()
    with zipfile.ZipFile(resolved) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        for info in archive.infolist():
            target = extracted.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('xb') as destination:
                while chunk := source.read(1024*1024):
                    destination.write(chunk)
    original, previous = extracted/'baseline', extracted/'audit'
    members, report, _ = inventory(original, previous)
    expected = set(members)
    if set(manifest['members']) != expected:
        raise ValueError('Added/omitted scientific evidence or original dependency')
    for name, path in members.items():
        # Includes ALL verification source/current capsule. Never dynamically
        # import or execute arbitrary extracted archive code.
        if sha(path) != manifest['members'][name]['sha256']:
            raise ValueError('Original/matching-checkout full evidence differs')
    fresh = output/'f'
    with (output/'stdout.log').open('x', encoding='utf8') as stdout, (output/'stderr.log').open('x', encoding='utf8') as stderr:
        subprocess.run([sys.executable, str(HERE/'audit_full_baseline.py'), '--reference', str(original),
            '--output', str(fresh)], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    compare_fresh(original, previous, fresh)
    verify_inventory(extracted, {n: v['sha256'] for n, v in manifest['members'].items()})
    if verify_archive(resolved) != integrity:
        raise ValueError('Full archive changed during actual extracted replay')
    # Revalidate all current run-used source and ALL saved episodes/steps/plans
    # after native subprocess, before issuing a replay certificate.
    inventory(original, previous)
    result = {'status': 'complete_original_population_archive_independently_replayed', 'artifact': integrity,
        'baseline_summary_sha256': sha(original/'summary.json'), 'prior_full_audit_summary_sha256': sha(previous/'summary.json'),
        'fresh_full_audit_summary_sha256': sha(fresh/'summary.json'), 'original8Level_equal_macro': report['original8Level_equal_macro'],
        'episodes_each_mode': 1280, 'modes': ['off', 'refusal'],
        'all_full_original_trajectories_plans_cbf_commands_equal': True,
        'all_nontiming_native_episode_step_diagnostics_equal': True, 'artifact_packaged_and_replayed': True,
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False,
        'active_model_faults_exercised': False, 'latency_qualified': False,
        'scope': 'Archive + corresponding Git checkout/history + original Python runtime. ALL original1280 development episodes, off/refusal EACH1280, and fresh ALL-mode native archive replay. No active causal control/faults, new geometry/holdout benefit, latency, cross-platform or formal safety qualification.'}
    write_json(output/'summary.json', result)
    return result


def package(baseline, audit, artifact, replay_output, parts_output=None):
    if artifact.exists() or replay_output.exists() or (parts_output is not None and parts_output.exists()):
        raise ValueError('Exclusive complete archive/replay/parts outputs required')
    members, _, _ = inventory(baseline, audit)
    expected = {n: {'bytes': p.stat().st_size, 'sha256': sha(p)} for n, p in members.items()}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr(MANIFEST, json.dumps({'schema': SCHEMA, 'members': expected}, indent=2))
    result = replay_archive(artifact, replay_output)
    if parts_output is not None:
        split_archive(artifact, parts_output)
        joined = resolve_artifact(parts_output, replay_output/'transport_roundtrip')
        if sha(joined) != result['artifact']['sha256']:
            raise ValueError('Lossless full original evidence transport differs')
        write_json(replay_output/'transport_verified.json', {'parts_manifest_sha256': sha(parts_output),
            'full_archive_sha256': sha(joined), 'transport_only_not_another_native_replay': True})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    for name in ('baseline', 'audit', 'artifact', 'parts-output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--replay-output', type=Path, required=True)
    args = parser.parse_args()
    if args.verify:
        if any(getattr(args, n) is not None for n in ('baseline', 'audit', 'artifact', 'parts_output')):
            parser.error('Verify cannot also package')
        report = replay_archive(args.verify, args.replay_output)
    else:
        if any(getattr(args, n) is None for n in ('baseline', 'audit', 'artifact')):
            parser.error('ALL complete original baseline/audit/archive required')
        report = package(args.baseline, args.audit, args.artifact, args.replay_output, args.parts_output)
    print(json.dumps(report), flush=True)
