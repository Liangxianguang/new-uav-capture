"""Publish audited original-controller V23 DATA ONLY, not enhanced performance."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_data_audit import audit_data
from train_cost_response import frozen_configs, sha, BASELINE_SHA
from frozen_training_release import verify_manifest
sys.path.insert(0, str(ROOT.parent / 'cwm_v21'))
from artifact_transport import split_archive, resolve_artifact


def data_only_summary(calls, report, public):
    """Derive support/capture evidence; never call capture a world-model result."""
    if (report['status'] != 'fresh_cost_contrast_data_finished_not_promoted' or
        public['status'] != 'independent_original_public_cost_data_replay_frames_equal' or
        not report['training_eligible'] or any(stage[key] for stage in (report, public)
        for key in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used'))):
        raise ValueError('Original-only qualified DATA status differs')
    indices = {split: [c for c in calls if c['record']['split'] == split]
               for split in ('train', 'development_validation')}
    epochs = report['episodes']
    if len(epochs) != 64 or len(public['episodes']) != 64 or len(calls) != 1536:
        raise ValueError('Complete fixed V23 DATA population differs')
    return {'status': 'passed_independent_fresh_original_cost_data_release',
        'scope': 'Original-controller data/public replay only; ongoing separate model training is NOT included. No enhanced-controller performance or research-model qualification.',
        'episodes': len(epochs), 'calls': len(calls), 'skipped_calls': report['skipped_calls'],
        'split_calls': {k: len(v) for k, v in indices.items()},
        'split_groups': {k: sorted({c['record']['group'] for c in v}) for k, v in indices.items()},
        'full_union_calls': {k: sum(bool(c['values']['valid'].all() and c['values']['anchor_valid'].all()) for c in v)
                             for k, v in indices.items()},
        'original_controller_outcomes': {k: sum(bool(e[k]) for e in epochs) for k in
            ('safe_capture_success', 'collision', 'boundary_violation', 'timeout', 'target_invalid_episode')},
        'original_observer_plain_public_trajectories_equal': True,
        'independent252_history_gru_and_delayed_peer_frames_verified': True,
        'original_geometry_masks_noise_costs_and_decisions_verified': True,
        'training_eligible_data_only': True, 'model_training_artifacts_included': False,
        'enhanced_control_enabled': False, 'holdout_used': False,
        'baseline_capsule_sha256': BASELINE_SHA}


def audit_data_archive(read, configs, restored):
    calls, report = audit_data(read, configs, restored)
    summary = data_only_summary(calls, report, json.loads(read('public/summary.json')))
    for stage in ('data', 'public'):
        summary[stage + '_summary_sha256'] = hashlib.sha256(read(stage + '/summary.json')).hexdigest()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'public', 'output', 'parts-output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--restored-output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original capsule changed')
    configs = frozen_configs(capsule, args.restored_output)
    if args.verify:
        artifact = resolve_artifact(args.verify, args.restored_output.parent / 'assembled_data')
        integrity = verify_manifest(artifact)
        with zipfile.ZipFile(artifact) as archive:
            result = audit_data_archive(archive.read, configs, args.restored_output)
            if json.loads(archive.read('data_release_summary.json')) != result:
                raise ValueError('Archived DATA release summary differs')
        print(json.dumps({'status': result['status'], **integrity}), flush=True)
        return
    if any(v is None for v in (args.data, args.public, args.output)):
        parser.error('Full original-controller DATA and PUBLIC replay required')
    roots = {'data': args.data, 'public': args.public}
    def read(name):
        stage, relative = name.split('/', 1)
        return (roots[stage] / relative).read_bytes()
    result = audit_data_archive(read, configs, args.restored_output)
    print(json.dumps({'status': 'prearchive_data_audit_passed', 'calls': result['calls'],
                      'full_union_calls': result['full_union_calls']}), flush=True)
    members = {}
    for stage, root in roots.items():
        for path in root.rglob('*'):
            relative = path.relative_to(root)
            if path.is_file() and not {'restored', '__pycache__'}.intersection(relative.parts):
                members[stage + '/' + relative.as_posix()] = path.read_bytes()
    members['data_release_summary.json'] = json.dumps(result, indent=2).encode('utf8')
    for path in ROOT.glob('*.py'):
        members['verification_source/' + path.name] = path.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr('ARTIFACT_MANIFEST.json', json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
    integrity = verify_manifest(args.output)
    with zipfile.ZipFile(args.output) as archive:
        recomputed = audit_data_archive(archive.read, configs, args.restored_output)
        if json.loads(archive.read('data_release_summary.json')) != recomputed or result != recomputed:
            raise ValueError('Archived DATA evidence differs from prearchive audit')
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original capsule changed during DATA release')
    transport = None
    if args.parts_output:
        transport = split_archive(args.output, args.parts_output)
        joined = resolve_artifact(args.parts_output, args.restored_output.parent / 'assembled_data')
        if sha(joined) != integrity['sha256']:
            raise ValueError('Lossless DATA transport differs from audited archive')
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    with (reports / 'data_release_manifest.json').open('x', encoding='utf8') as file:
        json.dump({**result, 'artifact': integrity, 'transport': transport}, file, indent=2)
    print(json.dumps({'status': result['status'], **integrity}), flush=True)


if __name__ == '__main__':
    main()
