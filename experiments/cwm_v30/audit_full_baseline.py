"""Fresh-process ALL1280 original-entry off/refusal replay, never active control.

Requires a complete published baseline; partial progress cannot authorize this
audit. Reexecutes each full episode, not just saved-summary arithmetic. Timing
measurements are retained but are not deterministic or latency qualification.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v29'), str(HERE.parent/'cwm_v28')]
from baseline_full_levels import Baseline, REMOTE, BRANCH, verify, json_diagnostics, summarize, write_json
from closed_loop_populations import original_population, restricted_safe_capture_time
from sequential_entry import SequentialResearchEntry
from package_ranking_release import safe_name

STATUS = 'complete_original8Level_plus_warmup_baseline_pending_independent_replay'
STEP_TIMING = frozenset(('predictor_latency_ms', 'qdr_latency_ms', 'rnic_latency_ms',
    'planner_latency_ms', 'safety_latency_ms', 'total_control_latency_ms'))
EPISODE_TIMING = frozenset(('mean_planner_latency_ms', 'mean_predictor_latency_ms',
    'qdr_latency_ms', 'mean_rnic_latency_ms', 'mean_safety_latency_ms', 'mean_total_control_latency_ms'))


def read_json(path):
    return json.loads(path.read_bytes())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            value.update(chunk)
    return value.hexdigest()


def read_lines(path):
    with path.open('r', encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_steps(path, records):
    """Scan ALL diagnostics, retaining byte offsets/counts, not all step dicts."""
    assigned = {r['episode_index']: r for r in records}
    expected_order = [r['episode_index'] for r in records]
    index, last = {}, None
    with path.open('rb') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            identifier = row['episode_index']
            if identifier not in assigned:
                raise ValueError('Unexpected original step identity')
            if identifier != last:
                if identifier in index or len(index) >= len(expected_order) or identifier != expected_order[len(index)]:
                    raise ValueError('Full original episode-step blocks reordered/repeated')
                index[identifier] = {'start': start, 'stop': start, 'count': 0}
                last = identifier
            item = index[identifier]
            item['count'] += 1
            item['stop'] = stream.tell()
            if row['step'] != item['count'] or item['count'] > 250:
                raise ValueError('Complete ordered original step support required')
            if any(row[k] != assigned[identifier][k] for k in ('level', 'variant')):
                raise ValueError('Original step Level/variant differs')
    if list(index) != expected_order or any(v['count'] < 1 for v in index.values()):
        raise ValueError('Complete ALL-episode original step support required')
    return index


def episode_steps(path, item):
    rows = []
    with path.open('rb') as stream:
        stream.seek(item['start'])
        while stream.tell() < item['stop']:
            rows.append(json.loads(stream.readline()))
        if stream.tell() != item['stop'] or len(rows) != item['count']:
            raise ValueError('Indexed native episode step bytes changed')
    return rows


def require_equal(reference, actual, excluded, label):
    left = {k: v for k, v in reference.items() if k not in excluded}
    normalized = json.loads(json.dumps(json_diagnostics(actual), allow_nan=False))
    right = {k: v for k, v in normalized.items() if k not in excluded}
    def canonical(value):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    # Python considers True==1, 1==1.0 and +0.0==-0.0. Native emitted JSON
    # types/signs must also agree, not merely approximate numeric equality.
    if canonical(left) != canonical(right):
        different = sorted(k for k in set(left)|set(right) if k not in left or k not in right or canonical(left[k]) != canonical(right[k]))
        raise ValueError(label + ' differs: ' + ', '.join(different[:12]))


def array_equal(reference, actual):
    with np.load(reference, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        if (set(left.files) != set(right.files) or any(left[k].dtype != right[k].dtype or
            left[k].shape != right[k].shape or left[k].tobytes() != right[k].tobytes() for k in left.files)):
            raise ValueError('Full numeric trajectory/plan/CBF command bytes differ')


def macro(by_level):
    return {key: float(np.mean([by_level[str(level)][key] for level in range(1, 9)]))
            for key in by_level['1'] if key.startswith('variant_equal_')}


def validate_saved(reference, capsule):
    """Full inventory before outputs; terminal report never suffices by itself."""
    reference = reference.resolve()
    protocol = read_json(HERE/'closed_loop_protocol.json')
    records = original_population(capsule, protocol)
    report = read_json(reference/'summary.json')
    if (report['status'] != STATUS or report['protocol'] != protocol or report['episodes'] != 1280 or
        report['archived_original_episodes_reproduced'] != 256 or
        report['warmup_level_excluded_from_primary_macro'] is not True or
        any(report[key] is not False for key in ('enhanced_control_enabled', 'holdout_used', 'new_model_trained')) or
        report['baseline_capsule_sha256'] != sha(capsule)):
        raise ValueError('Complete original full population/default-off report required')
    prereg = read_json(reference/'preregistration.json')
    if (prereg != report['preregistration'] or prereg.get('exact_pushed_sources_verified') is not True or
        prereg.get('remote') != REMOTE or prereg.get('branch') != BRANCH or
        not isinstance(prereg.get('commit'), str) or len(prereg['commit']) != 40 or
        any(c not in '0123456789abcdef' for c in prereg['commit'])):
        raise ValueError('Published preregistration identity differs')
    # Prior exact remote check is recorded by the original entry; require every
    # used-source byte to match that retained Git commit, not today's HEAD.
    commit = prereg['commit']
    sources = report['source_hashes']
    required_sources = {'experiments/cwm_v30/'+name for name in
        ('closed_loop_protocol.json', 'closed_loop_populations.py', 'baseline_full_levels.py')}
    if not required_sources <= set(sources):
        raise ValueError('Original baseline used-source closure incomplete')
    hashes = {'summary.json': sha(reference/'summary.json'), 'preregistration.json': sha(reference/'preregistration.json')}
    for name, key in (('scenes.json', 'scene_sha256'), ('episodes.jsonl', 'episodes_sha256'), ('steps.jsonl', 'steps_sha256')):
        hashes[name] = report[key]
    for name, digest in sources.items():
        safe_name(name)
        if not name.startswith('experiments/'):
            raise ValueError('Unexpected baseline source member')
        path = ROOT/name
        raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest or sha(path) != digest:
            raise ValueError('Published original used-source bytes differ')
        hashes['source/'+name] = digest
    if read_json(reference/'scenes.json') != records:
        raise ValueError('Saved full original scene population differs')
    rows = read_lines(reference/'episodes.jsonl')
    if len(rows) != 1280 or [r['episode_index'] for r in rows] != [r['episode_index'] for r in records]:
        raise ValueError('Complete unchanged original episode support required')
    for record, row in zip(records, rows):
        if any(row[k] != record[k] for k in ('level', 'variant', 'mirror_group_id', 'mirror_pair_member')):
            raise ValueError('Original episode group/Level/variant identity differs')
        if row['restricted_safe_capture_time_seconds'] != restricted_safe_capture_time(row):
            raise ValueError('Saved restricted safe capture time differs')
        name = f"trajectories/level{row['level']}_{row['episode_index']}.npz"
        hashes[name] = row['trajectory_sha256']
        hashes[name.removesuffix('.npz')+'.commands.npz'] = row['commands_sha256']
    by_level = summarize(rows)
    if report['by_level'] != by_level or report['original8Level_equal_macro'] != macro(by_level):
        raise ValueError('Full group-aware original aggregate differs')
    steps = index_steps(reference/'steps.jsonl', records)
    verify_inventory(reference, hashes)
    return report, records, rows, steps, hashes


def verify_inventory(root, hashes):
    for name, digest in hashes.items():
        safe_name(name)
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError('Full baseline evidence hash/path differs: '+name)


def run(reference, output):
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    report, records, rows, steps, hashes = validate_saved(reference, capsule)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'audited_baseline_manifest.json', hashes)
    base = Baseline(capsule, output/'r')
    entry = SequentialResearchEntry(base)
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')} | {HERE/'closed_loop_protocol.json'}
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    for name in source_hashes:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    modes = {}
    try:
        for mode in ('off', 'refusal'):
            results = []
            folder = output/mode
            folder.mkdir()
            with (folder/'episodes.jsonl').open('x', encoding='utf8') as episode_file, (folder/'steps.jsonl').open('x', encoding='utf8') as step_file:
                for record, expected in zip(records, rows):
                    entry.configure(mode)  # No artifact, certificate, loader or mock authorization.
                    if (base.evaluator.DistributedMinimaxDNMPC is not entry.parent_planner or
                        base.evaluator.PredictionRuntime is not entry.parent_runtime):
                        raise ValueError('Off/refusal did not use exact original parent classes')
                    entry.episode = record['episode_index']
                    path = folder/'trajectories'/f"level{record['level']}_{record['episode_index']}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    commands, plans = [], []
                    def observe(env, observation, action, sequence):
                        commands.append(action.copy()); plans.append(sequence.copy())
                    row, actual_steps = base.run(record, path, observe)
                    np.savez_compressed(path.with_suffix('.commands.npz'), commanded=np.stack(commands), planned=np.stack(plans))
                    for name in (path.name, path.with_suffix('.commands.npz').name):
                        array_equal(reference/'trajectories'/name, path.parent/name)
                    row.update(variant=record['variant'], mirror_group_id=record['mirror_group_id'], mirror_pair_member=record['mirror_pair_member'])
                    row['restricted_safe_capture_time_seconds'] = restricted_safe_capture_time(row)
                    require_equal(expected, row, EPISODE_TIMING|{'trajectory_sha256', 'commands_sha256'}, 'Native episode diagnostics/outcomes')
                    observed = [json_diagnostics({'level': record['level'], 'variant': record['variant'],
                        'episode_index': record['episode_index'], **s}) for s in actual_steps]
                    expected_steps = episode_steps(reference/'steps.jsonl', steps[record['episode_index']])
                    if len(observed) != len(expected_steps):
                        raise ValueError('Complete native step count differs')
                    for a, b in zip(expected_steps, observed):
                        require_equal(a, b, STEP_TIMING, 'Native step diagnostics')
                        step_file.write(json.dumps(b, allow_nan=False)+'\n')
                    if entry.history or entry.events or entry.pending:
                        raise ValueError('Off/refusal read optional model/context')
                    row['trajectory_sha256'], row['commands_sha256'] = sha(path), sha(path.with_suffix('.commands.npz'))
                    episode_file.write(json.dumps(json_diagnostics(row), allow_nan=False)+'\n')
                    episode_file.flush(); step_file.flush()
                    results.append(row)
                    if len(results)%16 == 0:
                        write_json(output/'progress.json', {'status': 'partial_full_population_off_refusal_replay_not_qualification',
                            'mode': mode, 'completed_episodes_this_mode': len(results), 'expected_episodes_each_mode': 1280,
                            'enhanced_control_enabled': False, 'holdout_used': False})
                        print(json.dumps({'mode': mode, 'completed_episodes': len(results)}), flush=True)
            if len(results) != 1280 or summarize(results) != report['by_level']:
                raise ValueError('Complete full-population replay aggregate differs')
            modes[mode] = {'episodes': len(results), 'original_parent_classes_used': True,
                'optional_context_or_model_read': False, 'full_trajectory_plan_cbf_bytes_equal': True,
                'all_nontiming_native_episode_and_step_diagnostics_equal': True,
                'episodes_sha256': sha(folder/'episodes.jsonl'), 'steps_sha256': sha(folder/'steps.jsonl'),
                'by_level': summarize(results)}
    finally:
        entry.close()
    verify_inventory(reference, hashes)
    verify(base.root, base.capsule_manifest)
    if sha(capsule) != report['baseline_capsule_sha256'] or any(sha(ROOT/name) != digest for name, digest in source_hashes.items()):
        raise ValueError('Original capsule or replay sources changed during audit')
    write_json(output/'summary.json', {'status': 'independent_full_original_population_off_refusal_replay_passed',
        'baseline_summary_sha256': hashes['summary.json'], 'protocol': report['protocol'], 'modes': modes,
        'source_hashes': source_hashes, 'audited_baseline_manifest_sha256': sha(output/'audited_baseline_manifest.json'),
        'original8Level_equal_macro': report['original8Level_equal_macro'],
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False,
        'active_model_faults_exercised': False, 'latency_qualified': False, 'artifact_packaged_and_replayed': False,
        'excluded_nondeterministic_step_fields': sorted(STEP_TIMING), 'excluded_nondeterministic_episode_fields': sorted(EPISODE_TIMING),
        'scope': 'ALL1280 original development episodes in EACH off/refusal mode. Actual original evaluator, numeric trajectories/plans/CBF and all non-timing native diagnostics replayed. Timing retained, not certified. Not active efficacy, actual-model faults, fresh scene/holdout confirmation or cross-platform/formal safety proof.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    run(args.reference.resolve(), args.output.resolve())
