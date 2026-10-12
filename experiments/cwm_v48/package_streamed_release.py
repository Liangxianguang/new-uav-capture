"""Full V34/V36/V37 evidence archive, followed by extracted scientific replay.

Requires matching Git history/source and original Python environment. Never
trains, promotes a model, changes a failed gate or certifies closed-loop gains.
All outputs must be NEW. Failures and their subprocess logs remain evidence.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v48'), str(HERE.parent/'cwm_v35'),
               str(HERE.parent/'cwm_v28'), str(HERE.parent/'cwm_v21')]
from archive_io import (SCHEMA, MANIFEST, sha, safe_name, regular, read_json,
                        write_json, verify_archive, extract_archive, npz_payloads)
from audited_stream import AuditedCorpus
import evidence as run_evidence
from repeatability_identity import exact_tree
from artifact_transport import resolve_artifact, split_archive

DEPENDENCIES = {'experiments/cwm_v1/baseline/capsule.zip': 'baseline_capsule_sha256',
                'experiments/cwm_v26/artifacts/real_sequential_shadow_20261010.zip': 'proposal_archive_sha256'}
DATA_ENTRY = 'experiments/cwm_v34/audit_prefix_data.py'
MODEL_ENTRY = 'experiments/cwm_v48/reload_streamed.py'


def require_equal(left, right, label):
    if not exact_tree(left, right):
        raise ValueError('Exact typed evidence differs: ' + label)


def check_sources(directory, report, key='source_hashes'):
    for name, digest in report[key].items():
        if sha(regular(ROOT, name)) != digest or sha(regular(directory, 'source/'+name)) != digest:
            raise ValueError('Current/saved scientific source identity differs: ' + name)


def checked_members(stage, directory, hashes):
    result = {}
    for name, digest in hashes.items():
        path = regular(directory, name)
        if sha(path) != digest:
            raise ValueError('Declared archive evidence changed: ' + stage + '/' + name)
        result[stage+'/'+safe_name(name)] = path
    return result


def validate_reload(report, directories, reports):
    if (report['status'] != 'two_complete_streamed_runs_all18models_cold_reloaded_original_costs_verified' or
        type(report['models_per_run']) is not int or report['models_per_run'] != 9 or
        type(report['complete_runs']) is not int or report['complete_runs'] != 2 or
        any(report[k] is not True for k in ('all12_checkpoint_contents_and_history_equal',
            'all_predictions_original_costs_gates_reexecuted', 'all_optimizer_sampler_torch_rng_actually_restored',
            'all_declared_native_evidence_unchanged')) or
        any(report[k] is not False for k in ('enhanced_control_enabled', 'holdout_used',
            'new_training_performed_by_reload', 'artifact_packaged_and_replayed'))):
        raise ValueError('Complete offline two-run ALL18-model reload required')
    counts = [report[k] for k in ('identical_serialized_checkpoint_pairs', 'different_serialized_checkpoint_pairs')]
    if any(type(n) is not int or not 0 <= n <= 12 for n in counts) or sum(counts) != 12:
        raise ValueError('All12 checkpoint pair serialization identities required')
    for stage, directory in directories.items():
        key = stage+'_summary_sha256' if stage != 'audit' else 'data_audit_summary_sha256'
        if report[key] != sha(regular(directory, 'summary.json')):
            raise ValueError('Complete reload input identity differs: ' + stage)
    for source in reports:
        require_equal(source['qualification'], report['qualification'], 'reload qualification')
        if source['primary_research_eligible'] is not report['primary_research_eligible']:
            raise ValueError('Failed/eligible primary result must remain unchanged')
    if type(report['primary_research_eligible']) is not bool or len(report['runs']) != 2:
        raise ValueError('Complete unchanged two-run reload results required')
    for run in report['runs']:
        if (run['models'] != 9 or run['all_predictions_original_costs_gates_reexecuted'] is not True or
            run['primary_research_eligible'] is not report['primary_research_eligible']):
            raise ValueError('Complete per-run reload results required')
    expected = {p.relative_to(ROOT).as_posix(): sha(p) for p in (HERE.parent/'cwm_v48').glob('*.py')}
    require_equal(report['reload_source_hashes'], expected, 'reload used sources')


def trajectory_names(episodes):
    ids = [e['episode_index'] for e in episodes]
    if len(ids) != 512 or any(type(i) is not int for i in ids) or len(set(ids)) != 512:
        raise ValueError('ALL512 unique native trajectory identities required')
    if any(e['unhooked_original_trace_equal'] is not True for e in episodes):
        raise ValueError('ALL unhooked original trajectory/plan/CBF comparisons required')
    return {str(i)+suffix for i in ids for suffix in ('.npz', '.commands.npz')}


def inventory(data, audit, primary, repeated, reload):
    directories = {'data': data, 'audit': audit, 'primary': primary, 'repeated': repeated}
    corpus = AuditedCorpus(data, audit)
    reports = [run_evidence.summary(d, corpus) for d in (primary, repeated)]
    run_hashes = [run_evidence.inventory(d, r, corpus) for d, r in zip((primary, repeated), reports)]
    run_evidence.require_same_summary(*reports)
    previous = read_json(regular(reload, 'summary.json'))
    validate_reload(previous, directories, reports)
    check_sources(reload, previous, 'reload_source_hashes')
    members = checked_members('data', data, corpus.manifest)
    names = {'summary.json', 'checked_calls.json', 'full_label_cost_checks.json', 'audited_data_manifest.json'}
    names.update('source/'+name for name in corpus.audit_report['source_hashes'])
    trajectories = trajectory_names(corpus.data_report['episodes'])
    names.update('trajectories/'+name for name in trajectories)
    if {p.name for p in (audit/'trajectories').iterdir()} != trajectories:
        raise ValueError('Complete native audited trajectory inventory differs')
    members.update(checked_members('audit', audit, {n: sha(regular(audit, n)) for n in names}))
    for stage, directory, hashes in zip(('primary', 'repeated'), (primary, repeated), run_hashes):
        members.update(checked_members(stage, directory, hashes))
    names = {'summary.json', *('source/'+name for name in previous['reload_source_hashes'])}
    for stage, report in zip(('primary', 'repeated'), reports):
        names.update(f"{stage}/{m['configuration']}_seed{m['seed']}.json" for m in report['models'])
    members.update(checked_members('reload', reload, {n: sha(regular(reload, n)) for n in names}))
    sources = {}
    for directory, report in ((data, corpus.data_report), (audit, corpus.audit_report),
                               (primary, reports[0]), (repeated, reports[1])):
        check_sources(directory, report)
        sources.update(report['source_hashes'])
    sources.update(previous['reload_source_hashes'])
    for path in (*HERE.glob('*.py'), HERE.parent/'cwm_v21/artifact_transport.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    members.update(checked_members('verification_source', ROOT, sources))
    for name, key in DEPENDENCIES.items():
        path = regular(ROOT, name)
        if sha(path) != corpus.protocol[key]:
            raise ValueError('Pinned original/proposer archive identity differs')
        members['dependencies/'+name] = path
    return members, previous


def release_contract(previous):
    return {'schema': SCHEMA, 'prior_complete_reload': previous,
        'archive_requires_independent_extracted_replay': True, 'native_episodes': 512,
        'independent_layout_noise_families': 64, 'complete_runs': 2, 'models_per_run': 9,
        'enhanced_control_enabled': False, 'holdout_used': False,
        'standalone_zip_portability_claimed': False,
        'scope': 'Full offline evidence only; original8Levels/new-scene paired closed-loop capture, safety, latency, active fault fallback and untouched holdout remain required.'}


def check_extracted(extracted, manifest):
    for name, row in manifest['members'].items():
        path = regular(extracted, name)
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Extracted evidence changed during replay: ' + name)
        if name.startswith(('verification_source/', 'dependencies/')):
            if sha(regular(ROOT, name.split('/', 1)[1])) != row['sha256']:
                raise ValueError('Matching Git checkout/runtime dependency required: ' + name)


def compare_data_audit(original, fresh, episodes):
    old, new = [read_json(regular(d, 'summary.json')) for d in (original, fresh)]
    for directory, report in ((original, old), (fresh, new)):
        check_sources(directory, report)
    # __main__/import bookkeeping may change source membership; every used source
    # in BOTH sets is still checked. No scientific report field is omitted.
    require_equal({k: v for k, v in old.items() if k != 'source_hashes'},
                  {k: v for k, v in new.items() if k != 'source_hashes'}, 'native data replay summary')
    for name in ('checked_calls.json', 'full_label_cost_checks.json', 'audited_data_manifest.json'):
        require_equal(read_json(regular(original, name)), read_json(regular(fresh, name)), name)
    expected = trajectory_names(episodes)
    for directory in (original, fresh):
        if {p.name for p in (directory/'trajectories').iterdir()} != expected:
            raise ValueError('ALL512 replayed trajectories and command plans required')
    for name in sorted(expected):
        require_equal(npz_payloads(regular(original, 'trajectories/'+name)),
                      npz_payloads(regular(fresh, 'trajectories/'+name)), 'trajectory/plan/CBF NPY bytes')


def compare_model_reload(original, fresh, previous):
    require_equal(read_json(regular(fresh, 'summary.json')), previous, 'ALL18-model extracted reload summary')
    for directory in (original, fresh):
        check_sources(directory, previous, 'reload_source_hashes')
    for stage in ('primary', 'repeated'):
        expected = {p.name for p in (original/stage).iterdir()}
        actual = {p.name for p in (fresh/stage).iterdir()}
        if len(expected) != 9 or actual != expected:
            raise ValueError('ALL18 cold model restoration reports required')
        for name in expected:
            require_equal(read_json(regular(original, stage+'/'+name)),
                          read_json(regular(fresh, stage+'/'+name)), 'cold model/optimizer/RNG restoration')


def run_child(entry, arguments, output):
    command = [sys.executable, str(regular(ROOT, entry)), *map(str, arguments)]
    stem = Path(entry).stem
    with (output/(stem+'.stdout.log')).open('x', encoding='utf8') as stdout, \
         (output/(stem+'.stderr.log')).open('x', encoding='utf8') as stderr:
        print(json.dumps({'status': 'full_extracted_replay_started', 'entry': entry}), flush=True)
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    print(json.dumps({'status': 'full_extracted_replay_finished', 'entry': entry}), flush=True)


def replay_archive(artifact, output):
    if output.exists():
        raise ValueError('Exclusive NEW complete replay output required')
    output.mkdir(parents=True, exist_ok=False)
    path = resolve_artifact(artifact, output/'assembled')
    extracted = output/'extracted'
    integrity = extract_archive(path, extracted)
    manifest = read_json(extracted/MANIFEST)
    check_extracted(extracted, manifest)
    roots = {name: extracted/name for name in ('data', 'audit', 'primary', 'repeated', 'reload')}
    members, previous = inventory(**roots)
    if set(manifest['members']) != set(members) | {'release_contract.json'}:
        raise ValueError('Archive omitted/added required complete scientific evidence')
    require_equal(read_json(extracted/'release_contract.json'), release_contract(previous), 'release contract')
    fresh_audit = output/'independent_data_audit'
    run_child(DATA_ENTRY, ['--data', roots['data'], '--output', fresh_audit], output)
    compare_data_audit(roots['audit'], fresh_audit, read_json(roots['data']/'episodes.json'))
    fresh_reload = output/'independent_model_reload'
    # Checkpoint provenance retains the original audit SHA. Use it only AFTER
    # independently reproducing every original audit field/physical branch above.
    run_child(MODEL_ENTRY, ['--data', roots['data'], '--audit', roots['audit'], '--primary', roots['primary'],
        '--repeated', roots['repeated'], '--output', fresh_reload], output)
    compare_model_reload(roots['reload'], fresh_reload, previous)
    if verify_archive(path) != integrity:
        raise ValueError('Archive changed during full scientific replay')
    check_extracted(extracted, manifest)
    inventory(**roots)
    result = {'status': 'complete_streamed_prefix_archive_independently_replayed', 'artifact': integrity,
        'all512_original_trajectories_plans_CBF_commands_replayed': True,
        'all_public_contexts_augmented_physical_branches_original_costs_replayed': True,
        'all18_final_models_optimizer_RNG_predictions_original_costs_gates_reloaded': True,
        'two_complete_fixed_runs_equal': True, 'artifact_packaged_and_replayed': True,
        'prior_complete_reload_summary_sha256': sha(roots['reload']/'summary.json'),
        'extracted_independent_data_audit_summary_sha256': sha(fresh_audit/'summary.json'),
        'extracted_independent_model_reload_summary_sha256': sha(fresh_reload/'summary.json'),
        'primary_research_eligible': previous['primary_research_eligible'], 'qualification': previous['qualification'],
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_training_performed': False,
        'closed_loop_capture_safety_latency_qualified': False, 'active_model_fault_fallback_qualified': False,
        'scope': release_contract(previous)['scope'] + ' Requires matching Git history/source and original Python environment.'}
    write_json(output/'summary.json', result)
    return result


def package(data, audit, primary, repeated, reload, archive, replay_output, parts_output=None):
    outputs = [archive, replay_output, *([parts_output] if parts_output is not None else [])]
    if any(p.exists() for p in outputs) or len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError('Exclusive distinct NEW archive/replay/transport outputs required')
    resolved = [p.resolve() for p in outputs]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(resolved) for b in resolved[i+1:]):
        raise ValueError('Archive/replay/transport outputs cannot contain one another')
    if any(p.is_relative_to(d.resolve()) for p in resolved for d in (data, audit, primary, repeated, reload)):
        raise ValueError('Archive/replay/transport outputs cannot alter scientific input directories')
    if parts_output is not None:
        prefix = parts_output.name.removesuffix('.parts.json')
        if not parts_output.name.endswith('.parts.json') or any(parts_output.parent.glob(prefix+'.part[0-9]*')):
            raise ValueError('Exclusive canonical transport manifest/parts required')
    members, previous = inventory(data, audit, primary, repeated, reload)
    expected = {n: {'bytes': p.stat().st_size, 'sha256': sha(p)} for n, p in members.items()}
    raw = json.dumps(release_contract(previous), indent=2, allow_nan=False).encode('utf8')
    expected['release_contract.json'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as out:
        for name, path in sorted(members.items()):
            safe_name(name)
            out.write(path, name)
        out.writestr('release_contract.json', raw)
        out.writestr(MANIFEST, json.dumps({'schema': SCHEMA, 'members': expected}, indent=2, allow_nan=False))
    verify_archive(archive)
    result = replay_archive(archive, replay_output)
    if parts_output is not None:
        split_archive(archive, parts_output)
        joined = resolve_artifact(parts_output, replay_output/'transport_roundtrip')
        if sha(joined) != result['artifact']['sha256']:
            raise ValueError('Lossless transport differs from scientifically replayed archive')
        write_json(replay_output/'transport_verified.json', {'parts_manifest_sha256': sha(parts_output),
            'full_archive_sha256': sha(joined), 'scientific_replay_certificate_sha256': sha(replay_output/'summary.json'),
            'transport_only_not_second_scientific_replay': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    for name in ('data', 'audit', 'primary', 'repeated', 'reload', 'archive', 'parts-output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--replay-output', type=Path, required=True)
    args = parser.parse_args()
    fields = ('data', 'audit', 'primary', 'repeated', 'reload', 'archive', 'parts_output')
    if args.verify is not None:
        if any(getattr(args, k) is not None for k in fields):
            parser.error('Verification cannot also package artifacts')
        result = replay_archive(args.verify, args.replay_output)
    else:
        if any(getattr(args, k) is None for k in fields[:-1]):
            parser.error('Complete data/audit/BOTH runs/reload/archive required')
        result = package(*[getattr(args, k) for k in fields[:-1]], args.replay_output, args.parts_output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
