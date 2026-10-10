"""Recompute public mediator, eighteen cores and original full-cost gates."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_mediated_response import (audit_data, FrozenMediator, attach_public_estimates,
    LocalTwoHead, state_digest, core_kind, normalization, control_metrics, prepare_decisions,
    evaluate, decision_records, qualification, frozen_configs, sha, check_archive, BASELINE_SHA)
from geometry_release import arrays, compare_arrays
from two_head_release import audit_sources
from training_release import tensor_tree_equal


def validate_report(report, protocol):
    if report['protocol'] != protocol or report['status'] != 'offline_mediated_response_finished_not_promoted':
        raise ValueError('Frozen training protocol/status mismatch')
    if any(report[k] for k in ('baseline_weights_included', 'enhanced_control_enabled', 'holdout_used',
                               'prior_gate_overridden', 'mediator_optimized', 'private_labels_model_inputs')):
        raise ValueError('Frozen original/public mediator offline contract mismatch')
    if not report['mediator_unchanged']:
        raise ValueError('Frozen mediator mutation claim')
    expected = {(name, seed) for name in protocol['models'] for seed in protocol['training']['seeds']}
    if len(report['models']) != len(expected) or {(r['configuration'], r['seed']) for r in report['models']} != expected:
        raise ValueError('Fixed nine-core population mismatch')


def audit(read, source_read, configs, restored, mediator_path):
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    calls, _ = audit_data(read, source_read, configs, restored)
    indices = {k: [i for i, c in enumerate(calls) if c['record']['split'] == k]
               for k in ('train', 'development_validation')}
    counts = {k: sum(calls[i]['record']['all_candidates_full_horizon_valid'] for i in v) for k, v in indices.items()}
    for k, name in (('train', 'train'), ('development_validation', 'development')):
        if counts[k] < protocol['additional_data_gate']['minimum_full_support_' + name + '_calls']:
            raise ValueError('Complete paired decision support failed')
    mediator = FrozenMediator(mediator_path, protocol)
    diagnostics = attach_public_estimates(calls, mediator)
    expected_estimates = {f'call{i}': c['estimated_commands'] for i, c in enumerate(calls)}
    mean, scale = normalization(calls, indices['train'])
    contexts = prepare_decisions(calls, indices['development_validation'], read, configs)
    controls = control_metrics(calls, indices['development_validation'])
    summaries, checkpoints = [], []
    for stage in ('primary', 'retrained'):
        report = json.loads(read(stage + '/summary.json'))
        validate_report(report, protocol)
        audit_sources(read, stage, report)
        if (json.loads(read(stage + '/source/cwm_v15/training_protocol.json')) != protocol
                or json.loads(read(stage + '/source/cwm_v15/data_protocol.json')) != json.loads((ROOT/'data_protocol.json').read_text())):
            raise ValueError('Saved preregistered protocol mismatch')
        # Match every archived dependency to the verifier's current source.
        for name, digest in report['source_hashes'].items():
            if sha(ROOT.parent / name) != digest:
                raise ValueError('Run-used source differs from independent verifier')
        if any(report[key] != hashlib.sha256(read(stage_name + '/summary.json')).hexdigest()
               for key, stage_name in (('data_summary_sha256', 'data'), ('public_summary_sha256', 'public'))):
            raise ValueError('Fresh original/public provenance mismatch')
        if (report['mediator_checkpoint_sha256'] != mediator.checkpoint_sha256 or report['mediator_state_digest'] != mediator.digest
                or report['controls'] != controls or report['full_support_calls'] != counts
                or report['split_calls'] != {k: len(v) for k, v in indices.items()}
                or report['split_groups'] != {k: sorted({calls[i]['record']['group'] for i in v}) for k, v in indices.items()}):
            raise ValueError('Actual public mediator/split/support/control mismatch')
        if (not compare_arrays(expected_estimates, arrays(read(stage + '/public_estimates.npz')))
                or diagnostics != json.loads(read(stage + '/mediator_diagnostics.json'))):
            raise ValueError('Independent public command estimates/diagnostics mismatch')
        if not compare_arrays({'mean': mean, 'scale': scale}, arrays(read(stage + '/normalization.npz'))):
            raise ValueError('Train-only normalizer mismatch')
        decision_set, checkpoint_set = {}, {}
        for row in report['models']:
            name, seed = row['configuration'], row['seed']
            identifier = f'{name}_seed{seed}'
            raw = read(f'{stage}/{identifier}.pt')
            if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
                raise ValueError('Core checkpoint digest mismatch')
            checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            if (checkpoint['configuration'] != name or checkpoint['seed'] != seed or checkpoint['protocol'] != protocol
                    or any(checkpoint[k] for k in ('online_promoted', 'baseline_weights_included', 'mediator_optimized'))
                    or checkpoint['source_hashes'] != report['source_hashes']
                    or checkpoint['data_summary_sha256'] != report['data_summary_sha256']
                    or checkpoint['public_summary_sha256'] != report['public_summary_sha256']
                    or checkpoint['mediator_checkpoint_sha256'] != mediator.checkpoint_sha256):
                raise ValueError('Frozen core checkpoint contract mismatch')
            if not compare_arrays({'mean': mean, 'scale': scale}, {'mean': checkpoint['normalizer_mean'].numpy(), 'scale': checkpoint['normalizer_scale'].numpy()}):
                raise ValueError('Checkpoint normalizer mismatch')
            config = protocol['training']
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = LocalTwoHead(core_kind(name), config['motion_scale_m'], config['response_scale_m'])
            initial = state_digest(model)
            if initial != row['initial_state_digest'] or initial != checkpoint['initial_state_digest']:
                raise ValueError('Matched core initialization mismatch')
            model.load_state_dict(checkpoint['model_state'], strict=True)
            measured, outputs = evaluate(model, name, calls, indices['development_validation'], mean, scale)
            if measured != row['development']:
                raise ValueError('Independent target/response metrics mismatch')
            expected = {f'call{j}_{key}': v for j, o in enumerate(outputs) for key, v in o.items()}
            if not compare_arrays(expected, arrays(read(f'{stage}/{identifier}_predictions.npz'))):
                raise ValueError('Reloaded prediction bytes mismatch')
            decisions = decision_records(calls, indices['development_validation'], outputs, contexts)
            if decisions != json.loads(read(f'{stage}/{identifier}_decisions.json')):
                raise ValueError('Independent full original cost/ranking mismatch')
            decision_set[identifier] = decisions
            if (row['parameters'] != sum(p.numel() for p in model.parameters())
                    or row['trainable_parameters'] != sum(p.numel() for p in model.parameters() if p.requires_grad)):
                raise ValueError('Core parameter budget mismatch')
            history = json.loads(read(f'{stage}/{identifier}_history.json'))
            if ([r['epoch'] for r in history] != list(range(1, config['epochs'] + 1))
                    or not np.isfinite([r['mean_batch_loss_motion_response_energy'] for r in history]).all()):
                raise ValueError('Fixed training budget/history mismatch')
            checkpoint_set[identifier] = checkpoint
        selected, gates = qualification(report['models'], controls, decision_set, protocol)
        if report['selected_median_seeds'] != selected or report['qualification'] != gates:
            raise ValueError('Independent median selection/development gate mismatch')
        for seed in protocol['training']['seeds']:
            pair = [checkpoint_set[f'{name}_seed{seed}'] for name in ('raw_plain', 'mediated_plain')]
            if (pair[0]['initial_state_digest'] != pair[1]['initial_state_digest']
                    or not tensor_tree_equal(pair[0]['sampler_rng_state'], pair[1]['sampler_rng_state'])
                    or not torch.equal(pair[0]['torch_rng_state'], pair[1]['torch_rng_state'])):
                raise ValueError('Paired initialization/sampling/RNG mismatch')
        summaries.append(report)
        checkpoints.append(checkpoint_set)
    for key in summaries[0].keys() - {'models'}:
        if summaries[0][key] != summaries[1][key]:
            raise ValueError('Independent retraining summaries differ')
    for identifier, checkpoint in checkpoints[0].items():
        if (not tensor_tree_equal(checkpoint, checkpoints[1][identifier])
                or read(f'primary/{identifier}_history.json') != read(f'retrained/{identifier}_history.json')):
            raise ValueError('Independent core weights/optimizer/RNG/history differ')
    if not mediator.unchanged():
        raise ValueError('Frozen mediator mutated during independent audit')
    return {'status': 'passed_independent_frozen_mediator_and_eighteen_core_recomputation',
            'data_episodes': 64, 'data_groups': 32, 'train_calls': len(indices['train']),
            'development_calls': len(indices['development_validation']), 'full_support_calls': counts,
            'models_per_run': 9, 'independent_training_runs': 2, 'weights_optimizer_rng_history_equal': True,
            'public252_histories_verified': True, 'public_mediator_estimates_byte_equal': True,
            'reload_prediction_bytes_equal': True, 'all_learned_full_cost_rankings_recomputed': True,
            'mediator_checkpoint_sha256': mediator.checkpoint_sha256, 'mediator_unchanged': True,
            'qualification': summaries[0]['qualification'], 'enhanced_control_enabled': False,
            'holdout_used': False, 'scope': protocol['later_required']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ('data', 'public', 'primary', 'retrained'):
        parser.add_argument('--' + stage, type=Path)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'cwm_v7/artifacts/paired_training_20261010.zip')
    parser.add_argument('--capsule', type=Path, default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    parser.add_argument('--mediator', type=Path, default=ROOT.parent/'cwm_v14/artifacts/public_mechanism_training_20261010.zip')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    data_protocol = json.loads((ROOT/'data_protocol.json').read_text())
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    if (sha(args.capsule) != BASELINE_SHA or sha(args.source) != data_protocol['source_archive_sha256']
            or sha(args.mediator) != protocol['mediator_archive_sha256']):
        raise ValueError('Protected baseline/candidate/mediator source changed')
    if any(protocol[k] for k in ('enhanced_control_enabled', 'original_gru_in_optimizer', 'mediator_in_optimizer',
                                'private_labels_model_inputs', 'holdout_used', 'prior_gate_override_allowed')):
        raise ValueError('Offline public frozen protocol overridden')
    for path in (args.source, args.mediator):
        check_archive(path, 'ARTIFACT_MANIFEST.json', False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    restored = ROOT.parent.parent/'results/cwm_v15/audit_restored'
    configs = frozen_configs(args.capsule, restored)
    with zipfile.ZipFile(args.source) as source:
        if args.verify:
            integrity = check_archive(args.verify, 'ARTIFACT_MANIFEST.json', False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read, source.read, configs, restored, args.mediator)
            print(json.dumps({**result, 'artifact_sha256': integrity['sha256']}), flush=True)
            return
        members = {}
        for stage in ('data', 'public', 'primary', 'retrained'):
            run = getattr(args, stage)
            if run is None:
                parser.error('Completed data/public replay and two independent training runs required')
            for file in run.rglob('*'):
                relative = file.relative_to(run)
                if file.is_file() and 'restored' not in relative.parts:
                    members[f'{stage}/{relative.as_posix()}'] = file.read_bytes()
        for file in ROOT.glob('*.py'):
            members['verification_source/' + file.name] = file.read_bytes()
        result = audit(members.__getitem__, source.read, configs, restored, args.mediator)
        path = ROOT/'artifacts/mediated_response_training_20261010.zip'
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
            archive.writestr('ARTIFACT_MANIFEST.json', json.dumps({n: hashlib.sha256(r).hexdigest() for n, r in members.items()}, indent=2))
        integrity = check_archive(path, 'ARTIFACT_MANIFEST.json', False)
        with zipfile.ZipFile(path) as archive:
            audit(archive.read, source.read, configs, restored, args.mediator)
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    for stage, run in (('data', args.data), ('public', args.public), ('training', args.primary)):
        with (reports/f'{stage}_summary.json').open('xb') as file:
            file.write((run/'summary.json').read_bytes())
    with (reports/'release_manifest.json').open('x') as file:
        json.dump({**result, 'artifact_sha256': integrity['sha256'], 'artifact_bytes': path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, 'artifact_sha256': integrity['sha256'], 'artifact_bytes': path.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
