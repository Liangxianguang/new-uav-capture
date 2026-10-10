"""V2 exclusive full V28 archive with exact per-run checkpoint identities.

Requires the corresponding Git checkout (including preregistration history) and
the original Python environment. Not a bare-ZIP or cross-platform guarantee.
No training, gate changes, best-seed selection, or online promotion occur here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'cwm_v21'))
from artifact_transport import file_digest, resolve_artifact, split_archive
from repeatability_identity import verify_pair, IDENTITY_SCHEMA

SCHEMA = 'cwm_v28.complete_fixed_ranking_archive.v2'
MANIFEST = 'ARTIFACT_MANIFEST.json'
DEPENDENCIES = ('experiments/cwm_v1/baseline/capsule.zip',
                'experiments/cwm_v26/artifacts/real_sequential_shadow_20261010.zip')


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)


def safe_name(name):
    """Reject traversal, Windows aliases/devices and noncanonical ZIP names."""
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
        any(ord(c) < 32 for c in name)):
        raise ValueError('Unsafe evidence member')
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe evidence member')
    devices = {'con', 'prn', 'aux', 'nul', *('com' + str(i) for i in range(1, 10)),
               *('lpt' + str(i) for i in range(1, 10))}
    if any(p.endswith((' ', '.')) or p.split('.')[0].lower() in devices or
           any(c in p for c in '<>"|?*') for p in path.parts):
        raise ValueError('Unsafe Windows evidence member')
    return name


def source_members(stage, report, directory):
    return {stage + '/source/' + safe_name(name): directory / 'source' / name
            for name in report['source_hashes']}


def require_false(report, *names):
    if any(report[name] is not False for name in names):
        raise ValueError('Offline/default-off/untouched-holdout contract differs')


def inventory(data, audit, primary, repeated, reload):
    """Only complete declared evidence, never restored trees or user extras."""
    protocol = read_json(HERE / 'data_protocol.json')
    training = read_json(HERE / 'training_protocol.json')
    d, a, p, r, v = [read_json(path / 'summary.json') for path in (data, audit, primary, repeated, reload)]
    if (d['status'] != 'fresh_bounded_sequential_data_collected_pending_independent_audit' or
        d['protocol'] != protocol or d['data_gate_passed'] is not True or
        d['training_authorized_by_this_report'] is not False or len(d['episodes']) != 64):
        raise ValueError('Complete fixed fresh dataset required')
    require_false(d, 'enhanced_control_enabled', 'new_model_trained', 'holdout_used')
    if (a['status'] != 'independent_fresh_sequential_public_branch_cost_audit_passed' or
        a['protocol'] != protocol or a['data_summary_sha256'] != file_digest(data / 'summary.json') or
        a['data_gate_passed'] is not True or a['training_authorized_by_this_report'] is not True):
        raise ValueError('Complete independent data audit required')
    require_false(a, 'enhanced_control_enabled', 'new_model_trained', 'holdout_used')
    if (v['status'] != 'two_complete_fresh_ranking_runs_reloaded_original_costs_verified' or
        v['models_per_run'] != 9 or v['complete_runs'] != 2 or
        any(v[k] is not True for k in ('all_final_weights_optimizer_rng_histories_equal',
            'all_prediction_bytes_reloaded', 'train_only_normalization_verified',
            'actual_geometry_shared_cost_contributions_recomputed')) or
        v['data_summary_sha256'] != file_digest(data / 'summary.json') or
        v['data_audit_summary_sha256'] != file_digest(audit / 'summary.json') or
        v['primary_summary_sha256'] != file_digest(primary / 'summary.json') or
        v['repeated_summary_sha256'] != file_digest(repeated / 'summary.json')):
        raise ValueError('Complete two-run reload evidence required')
    require_false(v, 'enhanced_control_enabled', 'holdout_used', 'artifact_packaged_and_replayed')
    identity = verify_pair(primary, repeated, p, r)
    expected_sources = {name: file_digest(HERE/name) for name in ('release_ranking_training_v2.py', 'repeatability_identity.py')}
    if (v.get('repeatability_identity') != identity or identity['schema'] != IDENTITY_SCHEMA or
        v.get('reload_source_hashes') != expected_sources):
        raise ValueError('Exact cross-run contents/new reload source evidence differs')
    if p['qualification'] != v['qualification'] or p['primary_research_eligible'] != v['primary_research_eligible']:
        raise ValueError('Training/reload qualifications differ')
    members = {}
    manifest = read_json(audit / 'audited_data_manifest.json')
    if file_digest(audit / 'audited_data_manifest.json') != a['audited_data_manifest_sha256']:
        raise ValueError('Audited data evidence manifest differs')
    records = read_json(data / 'records.json')
    required = {'summary.json', 'preregistration.json', 'scenes.jsonl', 'records.json', 'episodes.json'}
    required.update(name for row in records for name in (row['context_path'], row.get('arrays_path')) if name is not None)
    required.update('source/' + safe_name(name) for name in d['source_hashes'])
    if set(manifest) != required:
        raise ValueError('Complete declared dataset evidence required')
    for name, digest in manifest.items():
        name = safe_name(name)
        if file_digest(data / name) != digest:
            raise ValueError('Audited data changed: ' + name)
        members['data/' + name] = data / name
    if (len(records) != a['checked_calls'] or len(records) != d['records'] or
        sum('arrays_path' in row for row in records) != a['supported_calls'] or a['supported_calls'] != d['calls']):
        raise ValueError('Complete sequential call support differs')
    for name, key in (('checked_calls.json', 'checked_calls_sha256'),
                      ('full_label_cost_checks.json', 'full_label_cost_checks_sha256')):
        if file_digest(audit / name) != a[key]:
            raise ValueError('Completed audit evidence differs')
    for name in ('summary.json', 'checked_calls.json', 'full_label_cost_checks.json', 'audited_data_manifest.json'):
        members['audit/' + name] = audit / name
    members.update(source_members('audit', a, audit))
    episodes = read_json(data / 'episodes.json')
    if episodes != d['episodes'] or [e['episode_index'] for e in episodes] != list(range(994010, 994074)):
        raise ValueError('Complete fixed64episode order required')
    for episode in episodes:
        identifier = str(episode['episode_index'])
        if episode['trajectory_commands_plans_equal'] is not True:
            raise ValueError('Original trajectory/plan/command equality required')
        for kind in ('observed', 'plain'):
            path = data / kind / (identifier + '.npz')
            if file_digest(path) != episode[kind + '_sha256']:
                raise ValueError('Saved original trajectory differs')
            for suffix in ('.npz', '.commands.npz'):
                members['data/' + kind + '/' + identifier + suffix] = data / kind / (identifier + suffix)
        for suffix in ('.npz', '.commands.npz', '.public252.npz'):
            members['audit/trajectories/' + identifier + suffix] = audit / 'trajectories' / (identifier + suffix)
    expected = {(m, s) for m in training['models'] for s in training['training']['seeds']}
    for stage, directory, report in (('primary', primary, p), ('repeated', repeated, r)):
        if (report['status'] != 'offline_fixed_fresh_ranking_training_finished_pending_two_run_release' or
            report['protocol'] != training or report['data_summary_sha256'] != file_digest(data / 'summary.json') or
            report['data_audit_summary_sha256'] != file_digest(audit / 'summary.json') or len(report['models']) != 9 or
            {(row['configuration'], row['seed']) for row in report['models']} != expected):
            raise ValueError('Both complete fixed nine-model runs required')
        require_false(report, 'enhanced_control_enabled', 'holdout_used', 'prior_gate_overridden',
                      'common_motion_in_response_optimizer', 'two_complete_runs_verified')
        names = ['summary.json', 'normalization.npz', 'population_weights.json',
                 'initial_gru_score_audit.json', 'initial_cv_score_audit.json']
        for seed in training['training']['seeds']:
            names.extend(f'cv_seed{seed}_{tail}' for tail in ('motion_stage.pt', 'motion_stage_history.json',
                                                           'deployed_score_audit.json', 'ranking_cache.npz'))
        for row in report['models']:
            identifier = f"{row['configuration']}_seed{row['seed']}"
            if file_digest(directory / (identifier + '.pt')) != row['checkpoint_sha256']:
                raise ValueError('Final checkpoint digest differs')
            names.extend(identifier + tail for tail in ('.pt', '_history.json', '_decisions.json',
                '_library_contributions.json', '_train_predictions.npz', '_development_validation_predictions.npz'))
        members.update({stage + '/' + n: directory / n for n in names})
        members.update(source_members(stage, report, directory))
    members['reload/summary.json'] = reload / 'summary.json'
    sources = {}
    for stage, directory, report in (('data', data, d), ('audit', audit, a), ('primary', primary, p), ('repeated', repeated, r)):
        for name, digest in report['source_hashes'].items():
            safe_name(name)
            if file_digest(ROOT / name) != digest or file_digest(directory / 'source' / name) != digest:
                raise ValueError('Saved/current used source differs: ' + name)
            sources[name] = ROOT / name
    for name in ('package_ranking_release_v2.py', 'release_ranking_training_v2.py', 'repeatability_identity.py', 'audit_ranking_data.py',
                 'data_protocol.json', 'training_protocol.json'):
        relative = 'experiments/cwm_v28/' + name
        sources[relative] = ROOT / relative
    sources['experiments/cwm_v21/artifact_transport.py'] = HERE.parent / 'cwm_v21/artifact_transport.py'
    members.update({'verification_source/' + n: path for n, path in sources.items()})
    for name, digest in zip(DEPENDENCIES, (protocol['baseline_capsule_sha256'], protocol['proposal_archive_sha256'])):
        if file_digest(ROOT / name) != digest:
            raise ValueError('Pinned original/public proposer evidence differs')
        members['dependencies/' + name] = ROOT / name
    for name, path in members.items():
        safe_name(name)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing/nonregular release evidence: ' + name)
    return members, v


def stream_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def verify_archive(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [safe_name(i.filename) for i in infos]
        if len({n.casefold() for n in names}) != len(names) or MANIFEST not in names:
            raise ValueError('Duplicate/case-aliased or missing archive manifest')
        if any(i.is_dir() or (i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos):
            raise ValueError('Nonregular ZIP member')
        manifest = json.loads(archive.read(MANIFEST))
        if manifest['schema'] != SCHEMA or set(manifest['members']) != set(names) - {MANIFEST}:
            raise ValueError('Exact archive member inventory differs')
        for info in infos:
            if info.filename == MANIFEST:
                continue
            row = manifest['members'][info.filename]
            with archive.open(info) as stream:
                digest = stream_digest(stream)  # Also checks ZIP CRC, not just central-directory metadata.
            if row['bytes'] != info.file_size or row['sha256'] != digest:
                raise ValueError('Archive member length/hash differs: ' + info.filename)
    return {'sha256': file_digest(archive_path), 'bytes': archive_path.stat().st_size,
            'members': len(names) - 1}


def extract_archive(archive_path, destination):
    integrity = verify_archive(archive_path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('xb') as out:
                for block in iter(lambda: source.read(1 << 20), b''):
                    out.write(block)
    return integrity


def npz_payloads(path):
    # ZIP timestamps may differ; exact uncompressed NPY bytes must not.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate NPZ field')
        result = {}
        for name in names:
            with archive.open(name) as stream:
                result[name] = stream_digest(stream)
        return result


def compare_replayed_data(original, fresh):
    old, new = read_json(original / 'summary.json'), read_json(fresh / 'summary.json')
    # Different __main__/import bookkeeping is not a different physical outcome.
    # Every source in BOTH sets is nevertheless checked against this Git checkout.
    for directory, report in ((original, old), (fresh, new)):
        for name, digest in report['source_hashes'].items():
            safe_name(name)
            if file_digest(ROOT / name) != digest or file_digest(directory / 'source' / name) != digest:
                raise ValueError('Replay-used source differs')
    if {k: v for k, v in old.items() if k != 'source_hashes'} != {k: v for k, v in new.items() if k != 'source_hashes'}:
        raise ValueError('Independent extracted data-audit semantics differ')
    for name in ('checked_calls.json', 'full_label_cost_checks.json', 'audited_data_manifest.json'):
        if read_json(original / name) != read_json(fresh / name):
            raise ValueError('Independent extracted complete audit evidence differs: ' + name)
    expected = {f'{episode}{suffix}' for episode in range(994010, 994074)
                for suffix in ('.npz', '.commands.npz', '.public252.npz')}
    for directory in (original, fresh):
        if {p.name for p in (directory / 'trajectories').iterdir() if p.is_file()} != expected:
            raise ValueError('Independent complete trajectory evidence missing/extra')
    for name in sorted(expected):
        if npz_payloads(original / 'trajectories' / name) != npz_payloads(fresh / 'trajectories' / name):
            raise ValueError('Independent public252/trajectory/plan/command replay differs')


def run_child(script, arguments, output):
    """Fresh processes avoid accumulated evaluator hooks/module-state snapshots."""
    command = [sys.executable, str(HERE / script), *[str(a) for a in arguments]]
    with (output / (script + '.stdout.log')).open('x', encoding='utf8') as stdout, \
         (output / (script + '.stderr.log')).open('x', encoding='utf8') as stderr:
        print(json.dumps({'status': 'real_extracted_replay_started', 'entry': script}), flush=True)
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    print(json.dumps({'status': 'real_extracted_replay_finished', 'entry': script}), flush=True)


def check_extracted_evidence(extracted, manifest):
    for name, row in manifest['members'].items():
        path = extracted / safe_name(name)
        if path.is_symlink() or path.stat().st_size != row['bytes'] or file_digest(path) != row['sha256']:
            raise ValueError('Extracted evidence changed during replay: ' + name)
        if name.startswith(('verification_source/', 'dependencies/')):
            relative = name.split('/', 1)[1]
            if file_digest(ROOT / relative) != file_digest(path):
                raise ValueError('Matching Git checkout/runtime evidence required: ' + relative)


def replay_archive(artifact, output):
    """Hash checks alone never create a completed release certificate."""
    output.mkdir(parents=True, exist_ok=False)
    archive_path = resolve_artifact(artifact, output / 'assembled')
    extracted = output / 'extracted'
    integrity = extract_archive(archive_path, extracted)
    manifest = read_json(extracted / MANIFEST)
    check_extracted_evidence(extracted, manifest)
    roots = {n: extracted / n for n in ('data', 'audit', 'primary', 'repeated', 'reload')}
    members, previous = inventory(**roots)
    # Exact curated inventory prevents an honestly rehashed archive omitting evidence.
    if set(manifest['members']) != set(members) | {'release_contract.json'}:
        raise ValueError('Extracted archive omitted/added required full evidence')
    contract = read_json(extracted / 'release_contract.json')
    if contract != release_contract(previous):
        raise ValueError('Archive release contract differs')
    fresh_data_audit = output / 'independent_data_audit'
    run_child('audit_ranking_data.py', ['--data', roots['data'], '--output', fresh_data_audit], output)
    compare_replayed_data(roots['audit'], fresh_data_audit)
    fresh_reload = output / 'independent_model_reload'
    # Preserve exact original audit identity in checkpoints, but only AFTER the
    # extracted data and original audit have independently been re-executed above.
    run_child('release_ranking_training_v2.py', ['--data', roots['data'], '--audit', roots['audit'],
        '--primary', roots['primary'], '--repeated', roots['repeated'], '--output', fresh_reload], output)
    if read_json(fresh_reload / 'summary.json') != previous:
        raise ValueError('Extracted two-run original-cost reload result differs')
    # Refuse input mutation during long real replays.
    if verify_archive(archive_path) != integrity:
        raise ValueError('Archive changed during independent replay')
    check_extracted_evidence(extracted, manifest)
    inventory(**roots)
    result = {'status': 'complete_fixed_ranking_archive_independently_replayed', 'artifact': integrity,
        'real_public_context_all_branch_replay_passed': True, 'all64_original_trajectories_plans_commands_equal': True,
        'all18_final_models_original_costs_reloaded': True, 'two_complete_runs_equal': True,
        'artifact_packaged_and_replayed': True, 'primary_research_eligible': previous['primary_research_eligible'],
        'qualification': previous['qualification'], 'enhanced_control_enabled': False, 'holdout_used': False,
        'scope': 'Archive + corresponding Git checkout/history + original Python environment. Offline qualification only; '
                 'no active selector, untouched holdout or complete-Level/new-scene closed-loop safety/capture/latency claim.'}
    write_json(output / 'summary.json', result)
    return result


def release_contract(previous):
    return {'schema': SCHEMA, 'prior_complete_reload': previous,
            'archive_requires_independent_extracted_replay': True,
            'enhanced_control_enabled': False, 'holdout_used': False,
            'standalone_zip_portability_claimed': False}


def package(data, audit, primary, repeated, reload, archive_path, replay_output, parts_output=None):
    if archive_path.exists() or replay_output.exists() or (parts_output is not None and parts_output.exists()):
        raise ValueError('Exclusive fresh artifact/replay/transport outputs required')
    members, previous = inventory(data, audit, primary, repeated, reload)
    expected = {n: {'bytes': p.stat().st_size, 'sha256': file_digest(p)} for n, p in members.items()}
    raw = json.dumps(release_contract(previous), indent=2).encode('utf8')
    expected['release_contract.json'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr('release_contract.json', raw)
        archive.writestr(MANIFEST, json.dumps({'schema': SCHEMA, 'members': expected}, indent=2))
    verify_archive(archive_path)
    result = replay_archive(archive_path, replay_output)
    if parts_output is not None:
        split_archive(archive_path, parts_output)
        joined = resolve_artifact(parts_output, replay_output / 'transport_roundtrip')
        if file_digest(joined) != result['artifact']['sha256']:
            raise ValueError('Lossless transport differs from fully replayed artifact')
        write_json(replay_output / 'transport_verified.json', {
            'parts_manifest_sha256': file_digest(parts_output), 'full_archive_sha256': file_digest(joined),
            'semantic_replay_certificate_sha256': file_digest(replay_output / 'summary.json'),
            'transport_only_not_second_semantic_replay': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    for name in ('data', 'audit', 'primary', 'repeated', 'reload', 'archive', 'parts-output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--replay-output', type=Path, required=True)
    args = parser.parse_args()
    if args.verify:
        if any(getattr(args, n) is not None for n in ('data', 'audit', 'primary', 'repeated', 'reload', 'archive', 'parts_output')):
            parser.error('Verification cannot also package/overwrite artifacts')
        report = replay_archive(args.verify, args.replay_output)
    else:
        if any(getattr(args, n) is None for n in ('data', 'audit', 'primary', 'repeated', 'reload', 'archive')):
            parser.error('Complete data/audit/BOTH runs/reload/archive required')
        report = package(args.data, args.audit, args.primary, args.repeated, args.reload,
                         args.archive, args.replay_output, args.parts_output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
