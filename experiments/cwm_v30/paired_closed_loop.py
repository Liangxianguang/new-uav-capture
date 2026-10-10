"""Full original/fresh DEVELOPMENT policy execution, qualified artifact ONLY.

No holdout CLI, optimizer, best-model switch or deployment promotion. A failed
qualified-bundle load or optional call stops the experiment, never silently
turning a fixed response control into original control and calling it efficacy.
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
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v29'), str(HERE.parent/'cwm_v16')]
from baseline_full_levels import Baseline, publication, sha, verify, write_json, json_diagnostics
from closed_loop_populations import original_population, validate_protocol, restricted_safe_capture_time
from closed_loop_gates import evaluate_development, method_order, specification, validate_rows
from measured_original_cycle import MeasuredOriginalCycle
from research_bundle import load_bundle, resolve_artifact
from sequential_entry import SequentialResearchEntry
from collect_scenes import scene_records


class CompleteSequentialEntry(SequentialResearchEntry):
    def flush(self):
        # V29's captured call omits ordinal, but its unchanged public serializer
        # requires it. Add ONLY the already-public call identity to the copied
        # evidence before serialization; original planner/inputs are untouched.
        # Keep previously released V29 bytes/proofs intact.
        for ordinal, call, _ in self.pending:
            call['ordinal'] = ordinal
        super().flush()


def published_sources():
    proof = publication()
    paths = set(HERE.glob('*.py')) | {HERE/'closed_loop_protocol.json'}
    paths |= {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}
    paths |= {HERE.parent/'cwm_v16/scene_protocol.json', HERE.parent/'cwm_v28/data_protocol.json',
        HERE.parent/'cwm_v28/training_protocol.json'}
    for path in paths:
        raw = subprocess.check_output(['git', 'show', proof['commit']+':'+path.relative_to(ROOT).as_posix()], cwd=ROOT)
        if raw != path.read_bytes():
            raise ValueError('Publish exact used evaluation/measurement/gate sources before execution')
    return proof


def fresh_records(base, protocol, bundle):
    if bundle.authorized is not True or bundle.deployment_control_eligible is not False:
        raise ValueError('Fixed artifact-qualified primary required BEFORE new scene generation')
    template = json.loads((HERE.parent/'cwm_v16/scene_protocol.json').read_bytes())
    data = json.loads((HERE.parent/'cwm_v28/data_protocol.json').read_bytes())
    for key in ('target_speed_scales', 'observation_conditions', 'scene_variation'):
        if template[key] != data[key]:
            raise ValueError('Prior public geometry physics/observation specification differs')
    fixed = {**template, **copy.deepcopy(protocol['fresh_new_scene_development']),
        'stage': 'fresh_v30_closed_loop_development_not_holdout'}
    records = scene_records(base, fixed)
    expected = specification('fresh_geometry', protocol)
    by_id = {r['episode_index']: r for r in records}
    if len(by_id) != 64 or set(by_id) != {r['episode_index'] for r in expected}:
        raise ValueError('Complete fresh scene identity population differs')
    result = [by_id[r['episode_index']] for r in expected]
    for actual, assigned in zip(result, expected):
        if any(actual[k] != assigned[k] for k in assigned):
            raise ValueError('Fresh scene axis/group/mirror differs')
    return result


def capture_sources(output):
    paths = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}
    paths |= {HERE/'closed_loop_protocol.json', HERE.parent/'cwm_v16/scene_protocol.json',
        HERE.parent/'cwm_v28/data_protocol.json', HERE.parent/'cwm_v28/training_protocol.json'}
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    for name in hashes:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    return hashes


def run(artifact, certificate, output):
    protocol = json.loads((HERE/'closed_loop_protocol.json').read_bytes())
    validate_protocol(protocol)
    prereg = published_sources()  # Before new outputs, simulation or model control.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preregistration.json', prereg)
    certificate_raw = certificate.read_bytes()
    (output/'release_certificate.json').write_bytes(certificate_raw)
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    base = Baseline(capsule, output/'r')
    ready_started = time.perf_counter()
    bundle = load_bundle(artifact, certificate, output/'cache')
    readiness_seconds = time.perf_counter()-ready_started
    if bundle.authorized is not True:
        write_json(output/'refusal.json', {'status': 'fixed_primary_offline_qualification_failed_no_active_evaluation',
            'bundle_status': bundle.status, 'artifact_sha256': bundle.archive_sha256,
            'certificate_sha256': bundle.certificate_sha256, 'fresh_scenes_generated': False,
            'research_actions_exercised': False, 'enhanced_control_enabled': False, 'holdout_used': False})
        return
    resolved = resolve_artifact(artifact, output/'cache')
    if sha(resolved) != bundle.archive_sha256 or sha(output/'release_certificate.json') != bundle.certificate_sha256:
        raise ValueError('Exact loaded artifact/certificate identity differs')
    populations = {'original': original_population(capsule, protocol), 'fresh_geometry': fresh_records(base, protocol, bundle)}
    write_json(output/'scenes.json', populations)  # Retain all assigned routes, including invalids.
    if any(not r.get('route_valid', True) for records in populations.values() for r in records):
        raise ValueError('Assigned fresh route generation failed; keep ALL records, no filtering/replacement or qualification')
    sources = capture_sources(output)
    entry, measurement = CompleteSequentialEntry(base), MeasuredOriginalCycle(base)
    measurement.install()
    rows_by_cohort = {c: [] for c in populations}
    try:
        for cohort, records in populations.items():
            folder = output/cohort
            folder.mkdir()
            with (folder/'episodes.jsonl').open('x', encoding='utf8') as episode_file, (folder/'steps.jsonl').open('x', encoding='utf8') as step_file:
                for record in records:
                    for method in method_order(cohort, record['episode_index'], protocol):
                        if bundle.authorized is not True:
                            raise ValueError('Fixed scoring bundle failed; no silent original-control substitution')
                        destination = folder/str(record['episode_index'])/method
                        destination.mkdir(parents=True)
                        score = method.removesuffix('_shared')
                        entry.configure('plain' if method == 'original' else 'research_eval',
                            bundle=None if method == 'original' else bundle, score=score, output=destination)
                        entry.episode = record['episode_index']
                        measurement.reset()
                        commands, plans = [], []
                        def observe(env, observation, action, sequence):
                            commands.append(action.copy()); plans.append(sequence.copy())
                        controller_record = {**record, 'episode_index': record.get('controller_sampling_identity', record['episode_index'])}
                        path = destination/'trajectory.npz'
                        row, steps = base.run(controller_record, path, observe)
                        np.savez_compressed(destination/'commands.npz', commanded=np.stack(commands), planned=np.stack(plans))
                        timings = measurement.validate(steps)
                        if row['planner_local_solver_failures'] != timings['planner_local_solver_failures']:
                            raise ValueError('Native episode/step failure totals differ')
                        row.update(episode_index=record['episode_index'], level=record['level'], variant=record['variant'],
                            mirror_group_id=record['mirror_group_id'], mirror_pair_member=record['mirror_pair_member'],
                            scoring_method=method, **timings)
                        row['restricted_safe_capture_time_seconds'] = restricted_safe_capture_time(row)
                        row['optional_failure_calls'] = sum(e['status'] not in ('research_selection_ready', 'missing_received_peer_context') for e in entry.events)
                        row['research_calls'] = len(entry.events)
                        row['choice_changed_calls'] = sum(e.get('choice_changed', False) for e in entry.events)
                        row['supported_research_calls'] = sum(e['status'] == 'research_selection_ready' for e in entry.events)
                        row['trajectory_path'] = path.relative_to(output).as_posix()
                        row['commands_path'] = (destination/'commands.npz').relative_to(output).as_posix()
                        row['trajectory_sha256'], row['commands_sha256'] = sha(path), sha(destination/'commands.npz')
                        write_json(destination/'events.json', entry.events)
                        row['events_path'] = (destination/'events.json').relative_to(output).as_posix()
                        row['events_sha256'] = sha(destination/'events.json')
                        episode_file.write(json.dumps(json_diagnostics(row), allow_nan=False)+'\n'); episode_file.flush()
                        for i, step in enumerate(steps):
                            step_file.write(json.dumps(json_diagnostics({'cohort': cohort, 'scoring_method': method,
                                'episode_index': record['episode_index'], 'end_to_end_control_latency_ms': measurement.samples[i], **step}), allow_nan=False)+'\n')
                        step_file.flush()
                        rows_by_cohort[cohort].append(row)
                        write_json(output/'progress.json', {'status': 'partial_actual_development_policy_execution_not_qualification',
                            'completed_method_episodes': {c: len(v) for c, v in rows_by_cohort.items()},
                            'expected_method_episodes': {'original': 8960, 'fresh_geometry': 448},
                            'research_actions_exercised': True, 'deployment_control_enabled': False, 'holdout_used': False})
                        if row['optional_failure_calls'] or bundle.authorized is not True:
                            raise ValueError('Actual optional inference/proposal/capture failed; evidence retained, no model/gate switching')
                    print(json.dumps({'cohort': cohort, 'completed_method_episodes': len(rows_by_cohort[cohort])}), flush=True)
            validate_rows(rows_by_cohort[cohort], specification(cohort, protocol), protocol)
    finally:
        entry.close(); measurement.close()
    stats = evaluate_development(rows_by_cohort, protocol)
    verify(base.root, base.capsule_manifest)
    if (sha(capsule) != protocol['baseline_capsule_sha256'] or sha(resolved) != bundle.archive_sha256 or
        sha(certificate) != bundle.certificate_sha256 or any(sha(ROOT/name) != digest for name, digest in sources.items())):
        raise ValueError('Original capsule or active evaluation sources changed')
    report = {'status': 'complete_actual_development_policy_execution_pending_independent_replay', 'protocol': protocol,
        'preregistration': prereg, 'source_hashes': sources, 'development_statistics': stats,
        'artifact_sha256': bundle.archive_sha256, 'certificate_sha256': bundle.certificate_sha256,
        'model_readiness_seconds_outside_original_timed_plan': readiness_seconds,
        'scenes_sha256': sha(output/'scenes.json'), 'cohort_evidence': {c: {'episodes': len(rows_by_cohort[c]),
            'episodes_sha256': sha(output/c/'episodes.jsonl'), 'steps_sha256': sha(output/c/'steps.jsonl')} for c in populations},
        'research_actions_exercised': True, 'deployment_control_enabled': False, 'enhanced_control_enabled': False,
        'holdout_used': False, 'holdout_authorized_by_this_report': False, 'native_policy_replay_verified': False,
        'active_model_faults_exercised': False, 'artifact_packaged_and_replayed': False,
        'scope': 'ALL1280 original + ALL64 fresh development scenes, ALL7 fixed policies. Complete actual simulation and measured native cycles, not independent active replay/faults/holdout/deployment or formal CBF guarantee.'}
    write_json(output/'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'certificate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    existed = args.output.exists()
    try:
        run(args.artifact, args.certificate, args.output)
    except Exception as error:
        if not existed and args.output.is_dir() and not (args.output/'summary.json').exists():
            write_json(args.output/'implementation_failure.json', {'status': 'incomplete_development_no_qualification',
                'exception_type': type(error).__name__, 'exception': str(error)[:300],
                'enhanced_control_enabled': False, 'holdout_used': False, 'holdout_authorized_by_this_report': False})
        raise
