"""ALL1280 original + ALL64 fresh development scenes, ALL8 fixed methods.

Research simulation only after qualified prefix archive and mandatory L2 preload.
No holdout, optimizer, sampling override, deployment or independent replay claim.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), *[str(HERE.parent/n) for n in ('cwm_v41', 'cwm_v39', 'cwm_v30', 'cwm_v16')]]
from benchmark_loader import load_benchmark, recheck_prefix
from prefix_entry import PrefixSequentialEntry
from prefix_selector import ready
from development_statistics import load_plan, arithmetic_view, specification, method_order, evaluate_development
from baseline_full_levels import Baseline, sha, verify, write_json, json_diagnostics
from measured_original_cycle import MeasuredOriginalCycle
from closed_loop_populations import original_population, restricted_safe_capture_time
from closed_loop_gates import validate_rows
from collect_scenes import scene_records
from collect_prefix_data import GitHubAPI, credential_token
from archive_io import read_json, regular


PROTOCOLS = ('cwm_v16/scene_protocol.json', 'cwm_v28/data_protocol.json', 'cwm_v28/training_protocol.json',
    'cwm_v30/closed_loop_protocol.json', 'cwm_v34/data_protocol.json', 'cwm_v34/training_protocol.json',
    'cwm_v36/protocol.json', 'cwm_v39/protocol.json')


def used_paths():
    paths = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}
    return paths | set(HERE.glob('*.py')) | {ROOT/'experiments'/p for p in PROTOCOLS}


def published_sources():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if GitHubAPI(credential_token()).head() != commit:
        raise ValueError('Publish exact fixed evaluation HEAD before native outputs or model control')
    for path in used_paths():
        relative = path.relative_to(ROOT).as_posix()
        if subprocess.check_output(['git', 'show', commit+':'+relative], cwd=ROOT) != path.read_bytes():
            raise ValueError('Actual evaluation source differs from exact published bytes: '+relative)
    return {'commit': commit, 'remote': 'https://github.com/Liangxianguang/new-uav-capture.git',
        'branch': 'refs/heads/causal-world-model-v1-20261009', 'exact_pushed_sources_verified': True}


def capture_sources(output, prereg):
    hashes = {}
    for path in used_paths():
        name = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        if subprocess.check_output(['git', 'show', prereg['commit']+':'+name], cwd=ROOT) != raw:
            raise ValueError('Later imported source must match the published preregistration')
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw); hashes[name] = sha(path)
    return hashes


def fresh_records(base, plan, bundle):
    if not ready(bundle):
        raise ValueError('Qualified prefix and complete benchmark required BEFORE fresh scene generation')
    inherited = read_json(ROOT/'experiments/cwm_v30/closed_loop_protocol.json')
    template = read_json(ROOT/'experiments/cwm_v16/scene_protocol.json')
    # Inherit the untouched V30 evaluation distribution, not V34 TRAIN scenes.
    old = read_json(ROOT/'experiments/cwm_v28/data_protocol.json')
    for key in ('target_speed_scales', 'observation_conditions', 'scene_variation'):
        if template[key] != old[key]:
            raise ValueError('Inherited fresh development physics/observation specification differs')
    fixed = {**template, **copy.deepcopy(inherited['fresh_new_scene_development']),
        'stage': 'fresh_v42_prefix_closed_loop_development_not_holdout'}
    records = scene_records(base, fixed)
    expected = specification('fresh_geometry', plan); by_id = {r['episode_index']: r for r in records}
    if len(records) != 64 or len(by_id) != 64 or set(by_id) != {r['episode_index'] for r in expected}:
        raise ValueError('All64 fixed fresh development identities required')
    result = [by_id[r['episode_index']] for r in expected]
    if any(any(actual[k] != assigned[k] for k in assigned) for actual, assigned in zip(result, expected)):
        raise ValueError('Fresh scene axis/group/mirror/Level differs')
    return result


def validate_output(output, inputs):
    output = Path(output)
    if output.exists():
        raise ValueError('Exclusive NEW evaluation output required')
    resolved = output.resolve()
    if (resolved == ROOT.resolve() or resolved.is_relative_to(ROOT/'experiments') or
        any((p := Path(value).resolve()) == resolved or p.is_relative_to(resolved) or
            (p.is_dir() and resolved.is_relative_to(p)) for value in inputs if value is not None)):
        raise ValueError('Evaluation output must not contain, overlap or overwrite input evidence/sources')
    return output


def validate_support(row, events, method):
    failures = sum(e['status'] not in ('research_selection_ready', 'missing_received_peer_context') for e in events)
    supported = sum(e['status'] == 'research_selection_ready' for e in events)
    if method == 'original':
        if events:
            raise ValueError('Original parent classes must not inspect optional contexts')
    else:
        if not events:
            raise ValueError('Expanded control ran without actual sequential research calls')
        if not supported:
            failures += 1  # Never label an entirely unsupported expanded policy effective.
        if any(e.get('score') != method.removesuffix('_shared') for e in events if e['status'] == 'research_selection_ready'):
            raise ValueError('Actual selected scorer differs from fixed method')
    row.update(optional_failure_calls=failures, research_calls=len(events),
        choice_changed_calls=sum(e.get('choice_changed', False) for e in events), supported_research_calls=supported)


def save_evidence_manifest(output, rows, sources):
    names = {'preregistration.json', 'prefix_certificate.json', 'legacy_certificate.json', 'scenes.json',
             'benchmark_preload.json'} | {'source/'+n for n in sources}
    for cohort, population in rows.items():
        names |= {cohort+'/episodes.jsonl', cohort+'/steps.jsonl'}
        for row in population:
            names |= {row[k] for k in ('trajectory_path', 'commands_path', 'events_path')}
            events_path = regular(output, row['events_path'])
            if sha(events_path) != row['events_sha256']:
                raise ValueError('Saved native events changed')
            for event in read_json(events_path):
                for field in ('arrays', 'context'):
                    if field+'_path' in event:
                        relative = (Path(row['events_path']).parent/event[field+'_path']).as_posix()
                        if sha(regular(output, relative)) != event[field+'_sha256']:
                            raise ValueError('Saved public context/selection evidence differs')
                        names.add(relative)
    return {name: {'bytes': regular(output, name).stat().st_size, 'sha256': sha(regular(output, name))}
            for name in sorted(names)}


def run(prefix_artifact, prefix_certificate, legacy_artifact, legacy_certificate, output):
    output = validate_output(output, (prefix_artifact, prefix_certificate, legacy_artifact, legacy_certificate))
    plan = load_plan(); view = arithmetic_view(plan)
    prereg = published_sources()
    output.mkdir(parents=True, exist_ok=False); write_json(output/'preregistration.json', prereg)
    started = time.perf_counter()
    bundle = load_benchmark(prefix_artifact, prefix_certificate, legacy_artifact, legacy_certificate,
                            output/'cache', mode='research_prepare')
    readiness = time.perf_counter()-started
    if not ready(bundle):
        write_json(output/'refusal.json', {'status': 'fixed_prefix_primary_failed_no_active_evaluation',
            'bundle_status': bundle.status, 'fresh_scenes_generated': False, 'research_actions_exercised': False,
            'enhanced_control_enabled': False, 'holdout_used': False})
        return
    # No environment or new physical scene exists before independent qualification.
    for value, name, digest in ((prefix_certificate, 'prefix_certificate.json', bundle.prefix_bundle.certificate_sha256),
        (legacy_certificate, 'legacy_certificate.json', bundle.legacy_evidence['certificate_sha256'])):
        path = Path(value)
        if sha(path) != digest: raise ValueError('Loaded native certificate identity changed')
        (output/name).write_bytes(path.read_bytes())
    write_json(output/'benchmark_preload.json', {'status': bundle.status,
        'prefix_archive_sha256': bundle.prefix_bundle.archive_sha256,
        'prefix_certificate_sha256': bundle.prefix_bundle.certificate_sha256,
        'selected_median_seeds': bundle.selected_median_seeds, 'legacy_evidence': bundle.legacy_evidence,
        'deployment_control_eligible': False, 'native_model_faults_verified': False})
    capsule = ROOT/'experiments/cwm_v1/baseline/capsule.zip'
    if sha(capsule) != view['baseline_capsule_sha256']: raise ValueError('Original capsule changed')
    base = Baseline(capsule, output/'r')
    inherited = read_json(ROOT/'experiments/cwm_v30/closed_loop_protocol.json')
    populations = {'original': original_population(capsule, inherited), 'fresh_geometry': fresh_records(base, plan, bundle)}
    write_json(output/'scenes.json', populations)
    if any(not r.get('route_valid', True) for records in populations.values() for r in records):
        raise ValueError('Assigned scene route generation failed; preserve ALL scenes without replacement')
    for cohort, records in populations.items():
        expected = specification(cohort, plan)
        if [{k: r[k] for k in expected[0]} for r in records] != expected:
            raise ValueError('Complete assigned development population/order required')
    sources = capture_sources(output, prereg)
    entry, measurement = PrefixSequentialEntry(base), MeasuredOriginalCycle(base)
    measurement.install(); rows = {c: [] for c in populations}
    try:
        for cohort, records in populations.items():
            folder = output/cohort; folder.mkdir()
            with (folder/'episodes.jsonl').open('x', encoding='utf8') as episode_file, (folder/'steps.jsonl').open('x', encoding='utf8') as step_file:
                for record in records:
                    for method in method_order(cohort, record['episode_index'], plan):
                        if not ready(bundle): raise ValueError('Actual benchmark failed; no policy substitution')
                        destination = folder/str(record['episode_index'])/method; destination.mkdir(parents=True)
                        entry.configure('plain' if method == 'original' else 'research_eval',
                            bundle=None if method == 'original' else bundle, score=method.removesuffix('_shared'), output=destination)
                        entry.episode = record['episode_index']; measurement.reset(); commands, plans = [], []
                        def observe(env, observation, action, sequence):
                            commands.append(action.copy()); plans.append(sequence.copy())
                        controller_record = {**record, 'episode_index': record.get('controller_sampling_identity', record['episode_index'])}
                        path = destination/'trajectory.npz'; row, steps = base.run(controller_record, path, observe)
                        np.savez_compressed(destination/'commands.npz', commanded=np.stack(commands), planned=np.stack(plans))
                        timings = measurement.validate(steps)
                        if row['planner_local_solver_failures'] != timings['planner_local_solver_failures']:
                            raise ValueError('Actual episode/step local failure totals differ')
                        row.update(**{k: record[k] for k in ('episode_index', 'level', 'variant', 'mirror_group_id', 'mirror_pair_member')},
                            scoring_method=method, **timings)
                        row['restricted_safe_capture_time_seconds'] = restricted_safe_capture_time(row)
                        validate_support(row, entry.events, method)
                        write_json(destination/'events.json', entry.events)
                        for kind, target in (('trajectory', path), ('commands', destination/'commands.npz'), ('events', destination/'events.json')):
                            row[kind+'_path'], row[kind+'_sha256'] = target.relative_to(output).as_posix(), sha(target)
                        episode_file.write(json.dumps(json_diagnostics(row), allow_nan=False)+'\n'); episode_file.flush()
                        for i, step in enumerate(steps):
                            step_file.write(json.dumps(json_diagnostics({'cohort': cohort, 'scoring_method': method,
                                'episode_index': record['episode_index'], 'end_to_end_control_latency_ms': measurement.samples[i], **step}), allow_nan=False)+'\n')
                        step_file.flush(); rows[cohort].append(row)
                        write_json(output/'progress.json', {'status': 'partial_prefix_native_development_not_qualification',
                            'completed_method_episodes': {c: len(v) for c, v in rows.items()},
                            'expected_method_episodes': {c: len(v)*8 for c, v in populations.items()},
                            'deployment_control_enabled': False, 'holdout_used': False})
                        if row['optional_failure_calls'] or not ready(bundle):
                            raise ValueError('Actual optional control failed; retain failure, no scorer/seed/gate switching')
                    print(json.dumps({'cohort': cohort, 'completed_method_episodes': len(rows[cohort])}), flush=True)
            validate_rows(rows[cohort], specification(cohort, plan), view)
    finally:
        entry.close(); measurement.close()
    stats = evaluate_development(rows, plan)
    verify(base.root, base.capsule_manifest)
    if (sha(capsule) != view['baseline_capsule_sha256'] or
        any(sha(ROOT/name) != digest or sha(output/'source'/name) != digest for name, digest in sources.items()) or
        any(p.relative_to(ROOT).as_posix() not in sources for p in used_paths())):
        raise ValueError('Native-used original capsule/evaluation sources changed or lacked preregistration')
    recheck_prefix(bundle.prefix_bundle, prefix_artifact, prefix_certificate, output/'cache/prefix')
    # The legacy archive/certificate identities remain the predeclared failed ones.
    from benchmark_loader import prefix as loader_prefix, legacy_release, bound_manifest
    legacy_path = loader_prefix.resolve_artifact(Path(legacy_artifact), output/'cache/legacy/assembled')
    if (sha(legacy_path) != bundle.legacy_evidence['archive_sha256'] or
        sha(Path(legacy_certificate)) != bundle.legacy_evidence['certificate_sha256']):
        raise ValueError('Legacy evidence changed during active research execution')
    extracted = output/'cache/legacy/extracted'
    legacy_release.check_extracted_evidence(extracted, bound_manifest(legacy_path, extracted))
    legacy_release.inventory(**{n: extracted/n for n in ('data', 'audit', 'primary', 'repeated', 'reload')})
    manifest = save_evidence_manifest(output, rows, sources); write_json(output/'evidence_manifest.json', manifest)
    write_json(output/'summary.json', {'status': 'complete_prefix_native_development_pending_independent_replay',
        'plan': plan, 'preregistration': prereg, 'source_hashes': sources, 'development_statistics': stats,
        'prefix_archive_sha256': bundle.prefix_bundle.archive_sha256,
        'prefix_certificate_sha256': bundle.prefix_bundle.certificate_sha256, 'legacy_evidence': bundle.legacy_evidence,
        'model_readiness_seconds_outside_original_timed_plan': readiness,
        'cohort_method_episodes': {c: len(v) for c, v in rows.items()}, 'evidence_manifest_sha256': sha(output/'evidence_manifest.json'),
        'research_actions_exercised': True, 'native_policy_execution_verified': True,
        'native_policy_replay_verified': False, 'active_model_fault_fallback_verified': False,
        'artifact_packaged_and_replayed': False, 'holdout_authorized_by_this_report': False,
        'holdout_used': False, 'enhanced_control_enabled': False, 'deployment_control_enabled': False,
        'scope': 'ALL1280 original+ALL64 fresh development scenes, ALL8 fixed policies and actual full-cycle timing. '
                 'Execution only; no independent native replay, faults, holdout, deployment or formal safety qualification.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix-artifact', 'prefix-certificate', 'legacy-artifact', 'legacy-certificate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    existed = args.output.exists()
    try:
        run(args.prefix_artifact, args.prefix_certificate, args.legacy_artifact, args.legacy_certificate, args.output)
    except Exception as error:
        if not existed and args.output.is_dir() and not (args.output/'summary.json').exists():
            write_json(args.output/'implementation_failure.json', {'status': 'incomplete_prefix_development_no_qualification',
                'exception_type': type(error).__name__, 'exception': str(error)[:300],
                'enhanced_control_enabled': False, 'holdout_used': False, 'holdout_authorized_by_this_report': False})
        raise


if __name__ == '__main__':
    main()
