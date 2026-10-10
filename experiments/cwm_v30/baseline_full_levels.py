"""ALL1280 frozen original scenes, including complete8Levels plus L0 diagnostic.

Original baseline ONLY; never constructs CWM, fresh scenes, holdout or optimizer.
Published before start, exclusive output, no skip/filter/timeout/gate relaxation.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v1'), str(HERE.parent/'cwm_v25')]
from closed_loop_populations import original_population, validate_protocol, restricted_safe_capture_time
from baseline import Baseline
from freeze_baseline import sha, verify, REFERENCE
from original_entry_boundary import compare_rows

REMOTE = 'https://github.com/Liangxianguang/new-uav-capture.git'
BRANCH = 'refs/heads/causal-world-model-v1-20261009'


def write_json(path, value):
    path.write_text(json.dumps(json_diagnostics(value), indent=2, allow_nan=False), encoding='utf8')


def json_diagnostics(value):
    """Losslessly label absent/infinite native diagnostics, never fake zero."""
    if isinstance(value, dict):
        return {k: json_diagnostics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_diagnostics(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_diagnostics(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {'native_nonfinite_float': 'nan' if math.isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity'}
    return value


def publication():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    commit = git('rev-parse', 'HEAD')
    remote = git('-c', 'http.sslBackend=schannel', 'ls-remote', REMOTE, BRANCH)
    if not remote or remote.split()[0] != commit:
        raise ValueError('Publish exact fixed population/source HEAD before full baseline run')
    for name in ('closed_loop_protocol.json', 'closed_loop_populations.py', 'baseline_full_levels.py'):
        relative = 'experiments/cwm_v30/' + name
        raw = subprocess.check_output(['git', 'show', commit+':'+relative], cwd=ROOT)
        if raw != (ROOT/relative).read_bytes():
            raise ValueError('Published baseline source/protocol bytes differ')
    return {'commit': commit, 'remote': REMOTE, 'branch': BRANCH, 'exact_pushed_sources_verified': True}


def summarize(rows):
    result = {}
    for level in range(9):
        chosen = [r for r in rows if r['level'] == level]
        if not chosen:
            continue
        variants = {}
        for variant in sorted({r['variant'] for r in chosen}):
            subset = [r for r in chosen if r['variant'] == variant]
            grouped = defaultdict(list)
            for row in subset:
                grouped[row['mirror_group_id']].append(row)
            metrics = ('safe_capture_success', 'collision', 'boundary_violation', 'timeout', 'target_invalid_episode')
            variants[variant] = {'episodes': len(subset), 'groups': len(grouped),
                **{k+'_rate': float(np.mean([r[k] for r in subset])) for k in metrics},
                **{'group_equal_'+k+'_rate': float(np.mean([
                    np.mean([r[k] for r in group]) for group in grouped.values()])) for k in metrics},
                'group_equal_restricted_safe_capture_time_seconds': float(np.mean([
                    np.mean([r['restricted_safe_capture_time_seconds'] for r in group]) for group in grouped.values()]))}
        result[str(level)] = {'episodes': len(chosen), 'variants': variants,
            **{'variant_equal_'+k+'_rate': float(np.mean([v['group_equal_'+k+'_rate'] for v in variants.values()])) for k in metrics},
            'variant_equal_restricted_safe_capture_time_seconds': float(np.mean([
                v['group_equal_restricted_safe_capture_time_seconds'] for v in variants.values()]))}
    return result


def run(output):
    protocol = json.loads((HERE/'closed_loop_protocol.json').read_bytes())
    validate_protocol(protocol)
    prereg = publication()  # BEFORE outputs or new observations.
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    records = original_population(capsule, protocol)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preregistration.json', prereg)
    write_json(output/'scenes.json', records)
    base = Baseline(capsule, output/'r')  # Short restoration prefix for Windows.
    historical = {r['episode_index']: r for r in map(json.loads,
        (base.root/REFERENCE/'episodes.jsonl').read_text().splitlines())}
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')} | {HERE/'closed_loop_protocol.json'}
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    for name in hashes:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    rows, checked = [], []
    with (output/'episodes.jsonl').open('x', encoding='utf8') as episode_file, (output/'steps.jsonl').open('x', encoding='utf8') as step_file:
        for index, record in enumerate(records):
            path = output/'trajectories'/f"level{record['level']}_{record['episode_index']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            commands, plans = [], []
            def observe(env, observation, action, sequence):
                commands.append(action.copy()); plans.append(sequence.copy())
            row, steps = base.run(record, path, observe)
            np.savez_compressed(path.with_suffix('.commands.npz'), commanded=np.stack(commands), planned=np.stack(plans))
            row.update(variant=record['variant'], mirror_group_id=record['mirror_group_id'], mirror_pair_member=record['mirror_pair_member'])
            row['restricted_safe_capture_time_seconds'] = restricted_safe_capture_time(row)
            row['trajectory_sha256'], row['commands_sha256'] = sha(path), sha(path.with_suffix('.commands.npz'))
            if record['episode_index'] in historical:
                if compare_rows(historical[record['episode_index']], row):
                    raise ValueError('Full-Level run changed previously archived baseline outcome')
                reference = base.root/REFERENCE/'trajectories'/path.name
                with np.load(reference, allow_pickle=False) as a, np.load(path, allow_pickle=False) as b:
                    if (set(a.files) != set(b.files) or any(a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or
                        a[k].tobytes() != b[k].tobytes() for k in a.files)):
                        raise ValueError('Full-Level run changed archived original full trajectory')
                checked.append(record['episode_index'])
            rows.append(row)
            episode_file.write(json.dumps(json_diagnostics(row), allow_nan=False)+'\n'); episode_file.flush()
            for step in steps:
                step_file.write(json.dumps(json_diagnostics({'level': record['level'], 'variant': record['variant'],
                    'episode_index': record['episode_index'], **step}), allow_nan=False)+'\n')
            step_file.flush()
            if (index+1) % 16 == 0 or index+1 == len(records):
                progress = {'status': 'partial_original_full_population_baseline_not_qualification',
                    'completed_episodes': len(rows), 'expected_episodes': len(records), 'by_level': summarize(rows),
                    'enhanced_control_enabled': False, 'holdout_used': False}
                write_json(output/'progress.json', progress)
                print(json.dumps({'completed_episodes': len(rows), 'level': record['level'], 'variant': record['variant']}), flush=True)
    if set(checked) != set(historical) or len(rows) != 1280:
        raise ValueError('Incomplete full population or archived256-episode historical support')
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT/name) != digest for name, digest in hashes.items()):
        raise ValueError('Full-Level baseline sources changed during run')
    by_level = summarize(rows)
    macro = {metric: float(np.mean([by_level[str(level)][metric] for level in range(1, 9)]))
             for metric in by_level['1'] if metric.startswith('variant_equal_')}
    write_json(output/'summary.json', {'status': 'complete_original8Level_plus_warmup_baseline_pending_independent_replay',
        'protocol': protocol, 'preregistration': prereg, 'source_hashes': hashes,
        'episodes': len(rows), 'by_level': by_level, 'original8Level_equal_macro': macro,
        'warmup_level_excluded_from_primary_macro': True, 'archived_original_episodes_reproduced': len(checked),
        'scene_sha256': sha(output/'scenes.json'), 'episodes_sha256': sha(output/'episodes.jsonl'),
        'steps_sha256': sha(output/'steps.jsonl'), 'baseline_capsule_sha256': sha(capsule),
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False,
        'scope': 'Complete existing development baseline ONLY. Not independent new-data generalization, causal enhancement, paired active efficacy, latency qualification or a formal CBF guarantee. Independent full replay/publication still pending.'})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    print(json.dumps({'completed_episodes': len(run(args.output))}), flush=True)
